"""Turn (state, fields) into model inputs, one forward pass per window.

Layout of one sequence:

    [CLS]? Fields:\\n [order_id] text: Order number\\n ... Document:\\n
    <document tokens>
    \\nAnswers:\\n[order_id] = ... [damage] = [SEP]?

Every field ends with a query marker placed *after* the document, and the
model reads each field's answer off the hidden state at that marker. With a
decoder backbone (causal attention) this is what lets the field's query see
the whole document; with an encoder it costs nothing either way.

Documents longer than the budget are split into overlapping windows, and
fields that do not fit beside the document in one pass are split into
groups, each group getting its own passes over the whole input (the same
thing Clairic's server does).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Field

IGNORE = -100
BIO_O, BIO_B, BIO_I = 0, 1, 2


@dataclass
class Feature:
    input_ids: list[int]
    doc_start: int                      # sequence index of the first document token
    doc_offsets: list[tuple[int, int]]  # absolute char offsets of this window's doc tokens
    query_pos: list[int]                # sequence index of each field's query marker
    fields: list[Field]
    window: int = 0
    group: int = 0
    # Training labels, one entry per field. Pointer labels are 0 for "absent"
    # and 1 + sequence position otherwise; list fields use BIO labels instead.
    start_labels: list[int] | None = None
    end_labels: list[int] | None = None
    bio_labels: list[list[int]] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def doc_end(self) -> int:
        return self.doc_start + len(self.doc_offsets)


def _field_text(f: Field) -> str:
    kind = f"list of {f.items}" if f.is_list else f.type
    extra = []
    if f.unit:
        extra.append(f"unit {f.unit}")
    if f.type == "number" and f.integer:
        extra.append("integer")
    head = f"[{f.name}] {kind}" + (f" ({', '.join(extra)})" if extra else "")
    return head + (f": {f.instructions}" if f.instructions else "") + "\n"


class Encoder:
    def __init__(self, tokenizer, max_len: int = 512, stride: int = 128, max_schema_frac: float = 0.5):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("a fast tokenizer is required (character offsets)")
        self.tok = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.max_schema_frac = max_schema_frac
        self.prefix = [tokenizer.cls_token_id] if tokenizer.cls_token_id is not None else []
        self.suffix = (
            [tokenizer.sep_token_id]
            if tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None
            else []
        )
        self.head_ids = self._ids("Fields:\n")
        self.doc_open = self._ids("Document:\n")
        self.doc_close = self._ids("\nAnswers:\n")

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    # -- field grouping -----------------------------------------------------

    def _field_ids(self, f: Field) -> tuple[list[int], list[int]]:
        # The query restates the whole field (name, type, instructions) right at
        # the marker the answer is read from. With only "[name] =" there, the
        # model has to match the name back to the schema block, and a small
        # model learns positional shortcuts instead ("3rd query = the date").
        desc = _field_text(f)
        return self._ids(desc), self._ids(desc.rstrip("\n") + " =")

    def group_fields(self, fields: list[Field]) -> list[list[tuple[Field, list[int], list[int]]]]:
        """Pack fields into groups whose schema fits in max_schema_frac of the window."""
        fixed = len(self.prefix) + len(self.suffix) + len(self.head_ids) + len(self.doc_open) + len(self.doc_close)
        budget = int(self.max_len * self.max_schema_frac) - fixed
        groups, current, used = [], [], 0
        for f in fields:
            desc, query = self._field_ids(f)
            cost = len(desc) + len(query)
            if cost > budget:
                # Overlong instructions: keep the head of the description.
                desc = desc[: max(1, budget - len(query))]
                cost = len(desc) + len(query)
            if current and used + cost > budget:
                groups.append(current)
                current, used = [], 0
            current.append((f, desc, query))
            used += cost
        if current:
            groups.append(current)
        return groups

    # -- windows ------------------------------------------------------------

    def tokenize_doc(self, state: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = self.tok(state, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = [], []
        for tid, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
            ids.append(tid)
            offsets.append((s, e))
        return ids, offsets

    def encode(self, state: str, fields: list[Field], doc=None) -> list[Feature]:
        doc_ids, doc_offsets = doc if doc is not None else self.tokenize_doc(state)
        features = []
        for g, group in enumerate(self.group_fields(fields)):
            schema = [i for _, desc, _ in group for i in desc]
            queries = [q for _, _, q in group]
            fixed = (
                len(self.prefix) + len(self.head_ids) + len(schema) + len(self.doc_open)
                + len(self.doc_close) + sum(len(q) for q in queries) + len(self.suffix)
            )
            avail = max(16, self.max_len - fixed)
            starts = [0]
            if len(doc_ids) > avail:
                step = max(1, avail - min(self.stride, avail // 2))
                s = 0
                while s + avail < len(doc_ids):
                    s += step
                    starts.append(min(s, len(doc_ids) - avail))
                starts = sorted(set(starts))
            for w, s in enumerate(starts):
                window_ids = doc_ids[s : s + avail]
                ids = self.prefix + self.head_ids + schema + self.doc_open
                doc_start = len(ids)
                ids = ids + window_ids + self.doc_close
                query_pos = []
                for q in queries:
                    ids = ids + q
                    query_pos.append(len(ids) - 1)
                ids = ids + self.suffix
                features.append(
                    Feature(
                        input_ids=ids,
                        doc_start=doc_start,
                        doc_offsets=doc_offsets[s : s + avail],
                        query_pos=query_pos,
                        fields=[f for f, _, _ in group],
                        window=w,
                        group=g,
                    )
                )
        return features

    # -- labels -------------------------------------------------------------

    @staticmethod
    def span_to_tokens(feature: Feature, s: int, e: int) -> tuple[int, int] | None:
        """Sequence positions of the tokens covering chars [s, e), if inside the window."""
        offs = feature.doc_offsets
        if not offs or s < offs[0][0] or e > offs[-1][1]:
            return None
        first = last = None
        for i, (ts, te) in enumerate(offs):
            if te <= ts:  # zero-width special tokens
                continue
            if ts < e and te > s:
                if first is None:
                    first = i
                last = i
        if first is None:
            return None
        return feature.doc_start + first, feature.doc_start + last

    def add_labels(self, feature: Feature, answers: dict[str, list[tuple[int, int]]]) -> Feature:
        seq_len = len(feature.input_ids)
        starts, ends, bios = [], [], []
        for f in feature.fields:
            spans = answers.get(f.name, [])
            if f.is_list:
                tags = [IGNORE] * seq_len
                for p in range(feature.doc_start, feature.doc_end):
                    tags[p] = BIO_O
                for s, e in spans:
                    tok = self.span_to_tokens(feature, s, e)
                    if tok is None:
                        continue
                    a, b = tok
                    tags[a] = BIO_B
                    for p in range(a + 1, b + 1):
                        tags[p] = BIO_I
                starts.append(IGNORE)
                ends.append(IGNORE)
                bios.append(tags)
            else:
                label = None
                for s, e in spans:  # first alternative inside this window
                    label = self.span_to_tokens(feature, s, e)
                    if label is not None:
                        break
                if label is None:
                    starts.append(0)
                    ends.append(0)
                else:
                    starts.append(label[0] + 1)
                    ends.append(label[1] + 1)
                bios.append([IGNORE] * seq_len)
        feature.start_labels, feature.end_labels, feature.bio_labels = starts, ends, bios
        return feature

    def window_has_answer(self, feature: Feature) -> bool:
        if any(x > 0 for x in feature.start_labels or []):
            return True
        return any(t == BIO_B for tags in feature.bio_labels or [] for t in tags)
