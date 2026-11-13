"""Convert token-tagged NER datasets into `list` fields.

Each entity type becomes one list field ("people", "organizations", ...) whose
items are every mention in the passage. Types with no mention become empty
lists, which teaches the BIO head to return nothing rather than guess.

    python -m sysone_extract.data.ner --hf eriktks/conll2003 --split train --out conll_train.jsonl

Input rows need `tokens` and `ner_tags` (IOB2 label ids, with the label names
in the dataset features), which covers CoNLL-2003, WikiNER, OntoNotes ports
and most token-classification datasets on the Hub.
"""

from __future__ import annotations

import argparse
import random

from . import write_jsonl

FIELD_FOR = {
    "PER": ("people", "Names of people mentioned"),
    "ORG": ("organizations", "Names of companies, institutions and other organizations"),
    "LOC": ("locations", "Names of places: countries, cities, regions, landmarks"),
    "MISC": ("other_names", "Other named things: nationalities, events, products, works"),
    "GPE": ("places", "Countries, cities and states mentioned"),
    "DATE": ("dates", "Dates mentioned"),
    "MONEY": ("amounts", "Monetary amounts mentioned"),
    "PRODUCT": ("products", "Products mentioned"),
    "EVENT": ("events", "Named events mentioned"),
}


def _detokenize(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text, offsets = "", []
    no_space_before = {".", ",", ";", ":", "!", "?", ")", "]", "}", "'s", "n't", "%"}
    for i, tok in enumerate(tokens):
        if i and tok not in no_space_before and not text.endswith(("(", "[", "{", "$")):
            text += " "
        offsets.append((len(text), len(text) + len(tok)))
        text += tok
    return text, offsets


def build_records(rows, label_names: list[str], docs_per_record: int = 4, seed: int = 0, prefix: str = "ner"):
    """Join consecutive sentences into short passages and emit list fields."""
    rng = random.Random(seed)
    types = sorted({n.split("-", 1)[1] for n in label_names if "-" in n})
    buf, n = [], 0

    def flush():
        nonlocal n
        tokens, tags = [], []
        for row in buf:
            tokens += row["tokens"]
            tags += [label_names[t] for t in row["ner_tags"]]
        text, offs = _detokenize(tokens)
        spans = {t: [] for t in types}
        cur = None
        for (s, e), tag in zip(offs, tags):
            if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur[0] != tag[2:])):
                if cur:
                    spans[cur[0]].append([cur[1], cur[2]])
                cur = [tag[2:], s, e]
            elif tag.startswith("I-"):
                cur[2] = e
            else:
                if cur:
                    spans[cur[0]].append([cur[1], cur[2]])
                cur = None
        if cur:
            spans[cur[0]].append([cur[1], cur[2]])
        fields, answers = {}, {}
        for t in types:
            name, instructions = FIELD_FOR.get(t, (t.lower() + "_mentions", f"Mentions of type {t}"))
            fields[name] = {"type": "list", "items": "text", "instructions": instructions}
            answers[name] = spans[t]
        n += 1
        return {"id": f"{prefix}-{n}", "source": prefix, "state": text, "fields": fields, "answers": answers}

    for row in rows:
        if not row["tokens"]:
            continue
        buf.append(row)
        if len(buf) >= rng.randint(1, docs_per_record):
            yield flush()
            buf = []
    if buf:
        yield flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="ner")
    args = ap.parse_args(argv)
    from datasets import load_dataset

    ds = load_dataset(args.hf, split=args.split)
    names = ds.features["ner_tags"].feature.names
    n = write_jsonl(args.out, build_records(ds, names, prefix=args.prefix))
    print(f"wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
