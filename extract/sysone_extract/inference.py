"""Run the model on a request and decode typed answers.

Decoding mirrors the request types:

* scalar fields (text, number, date): the best spans across every window are
  ranked by their joint start/end probability, and the first one that parses
  as the field's type becomes the value. `guess` is the top raw span even when
  the field comes back null, and `span` gives its character offsets.
* list fields: BIO tags give any number of spans; each item must parse as the
  list's item type.

Every scalar answer carries `p_null` (probability the value is absent from the
input) and `confidence` (probability the returned span is right, given that a
value is present).
"""

from __future__ import annotations

import math
import threading
import time

import torch

from .encoding import BIO_B, BIO_I, BIO_O, Encoder, Feature
from .model import ExtractorModel, collate, model_inputs
from .normalize import parse_value
from .schema import Field, parse_fields

NULL_THRESHOLD = 0.5
MAX_ANSWER_TOKENS = 64
TOP_K = 20


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _joined(state: str, i: int) -> bool:
    """True when the characters at i-1 and i belong to the same word.

    Letters and digits join, and so do digits around a separator ("1249.90",
    "1,234", "1'250"), so a number is never cut at its decimal point.
    """
    if i <= 0 or i >= len(state):
        return False
    a, b = state[i - 1], state[i]
    if a.isalnum() and b.isalnum():
        return True
    if b in ".,'" and a.isdigit() and i + 1 < len(state) and state[i + 1].isdigit():
        return True
    if a in ".,'" and b.isdigit() and i >= 2 and state[i - 2].isdigit():
        return True
    return False


def _trim(state: str, s: int, e: int) -> tuple[int, int]:
    """Strip whitespace, then widen the span to whole words.

    Subword tokenizers can put a span boundary inside a word ("$12" out of
    "$1249.90"); a value is never part of a word, so snap outwards.
    """
    while s < e and state[s].isspace():
        s += 1
    while e > s and state[e - 1].isspace():
        e -= 1
    if e <= s:
        return s, e
    while _joined(state, s):
        s -= 1
    while _joined(state, e):
        e += 1
    return s, e


class Extractor:
    """Wraps a model and tokenizer; thread-safe, one forward pass at a time."""

    def __init__(self, model: ExtractorModel, tokenizer, max_len: int = 512, stride: int = 128,
                 device=None, batch_size: int = 8, null_threshold: float = NULL_THRESHOLD,
                 calibration: dict | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = Encoder(tokenizer, max_len=max_len, stride=stride)
        self.device = device or next(model.parameters()).device
        self.batch_size = batch_size
        self.null_threshold = null_threshold
        self.calibration = calibration
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model_dir: str, device: str = "auto", dtype=None, **kwargs) -> "Extractor":
        dev = pick_device(device)
        model, tokenizer, config = ExtractorModel.load(model_dir, device=dev, dtype=dtype)
        kwargs.setdefault("max_len", config.get("max_len", 512))
        kwargs.setdefault("stride", config.get("stride", 128))
        kwargs.setdefault("calibration", config.get("calibration"))
        return cls(model, tokenizer, device=dev, **kwargs)

    def _calibrated(self, kind: str, p: float) -> float:
        from .calibrate import apply

        return apply(p, (self.calibration or {}).get(kind))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    # -- forward --------------------------------------------------------------

    @torch.no_grad()
    def _run(self, features: list[Feature]) -> list[dict]:
        """Per-feature, per-field probabilities (CPU tensors)."""
        results = []
        pad = self.tokenizer.pad_token_id or 0
        was_training = self.model.training
        self.model.eval()
        try:
            for i in range(0, len(features), self.batch_size):
                chunk = features[i : i + self.batch_size]
                batch = collate(chunk, pad, device=self.device)
                out = self.model(**model_inputs(batch))
                start = out["start"].log_softmax(-1).cpu()
                end = out["end"].log_softmax(-1).cpu()
                bio = out["bio"].softmax(-1).cpu()
                n = 0
                for f in chunk:
                    k = len(f.query_pos)
                    results.append({"start": start[n : n + k], "end": end[n : n + k], "bio": bio[n : n + k]})
                    n += k
        finally:
            if was_training:
                self.model.train()
        return results

    # -- decoding -------------------------------------------------------------

    @staticmethod
    def _scalar_candidates(feature: Feature, start: torch.Tensor, end: torch.Tensor):
        """(log joint prob, char start, char end, log cond prob) for the best spans, plus p_null."""
        p_null = 0.5 * (math.exp(start[0].item()) + math.exp(end[0].item()))
        lo, hi = feature.doc_start + 1, feature.doc_end + 1  # +1: index 0 is "absent"
        if hi <= lo:
            return [], p_null
        s_scores, e_scores = start[lo:hi], end[lo:hi]
        k = min(TOP_K, hi - lo)
        s_top = torch.topk(s_scores, k)
        e_top = torch.topk(e_scores, k)
        log_not_null = math.log(max(1e-9, 1 - math.exp(start[0].item()))) + math.log(
            max(1e-9, 1 - math.exp(end[0].item()))
        )
        cands = []
        for sv, si in zip(s_top.values.tolist(), s_top.indices.tolist()):
            for ev, ei in zip(e_top.values.tolist(), e_top.indices.tolist()):
                if ei < si or ei - si + 1 > MAX_ANSWER_TOKENS:
                    continue
                cs = feature.doc_offsets[si][0]
                ce = feature.doc_offsets[ei][1]
                joint = sv + ev
                cands.append((joint, cs, ce, joint - log_not_null))
        return cands, p_null

    @staticmethod
    def _list_spans(feature: Feature, bio: torch.Tensor):
        """Spans from BIO tags: (char start, char end, mean tag prob); and p(any item)."""
        probs = bio[feature.doc_start : feature.doc_end]  # [T, 3]
        if probs.shape[0] == 0:
            return [], 0.0
        tags = probs.argmax(-1).tolist()
        spans, cur = [], None
        for t, tag in enumerate(tags):
            if tag == BIO_B or (tag == BIO_I and cur is None):
                if cur:
                    spans.append(cur)
                cur = [t, t, [probs[t, tag].item()]]
            elif tag == BIO_I:
                cur[1] = t
                cur[2].append(probs[t, tag].item())
            else:
                if cur:
                    spans.append(cur)
                cur = None
        if cur:
            spans.append(cur)
        out = [
            (feature.doc_offsets[a][0], feature.doc_offsets[b][1], sum(ps) / len(ps))
            for a, b, ps in spans
        ]
        p_any = probs[:, BIO_B].max().item()
        return out, p_any

    def _decode_scalar(self, state: str, f: Field, parts) -> dict:
        p_null = self._calibrated("p_null", min(p for _, p in parts) if parts else 1.0)
        best: dict[tuple[int, int], tuple[float, float]] = {}
        for cands, _ in parts:
            for joint, cs, ce, cond in cands:
                key = _trim(state, cs, ce)
                if key[1] <= key[0]:
                    continue
                if key not in best or joint > best[key][0]:
                    best[key] = (joint, cond)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
        answer = {"type": f.type, "p_null": round(p_null, 6)}
        if f.type == "number" and f.unit:
            answer["unit"] = f.unit
        if not ranked:
            answer.update(guess=None, span=None, confidence=0.0, value=None)
            return answer
        def conf(cond):
            return round(self._calibrated("confidence", min(1.0, math.exp(cond))), 6)

        (gs, ge), (_, gcond) = ranked[0]
        answer["guess"] = state[gs:ge]
        answer["span"] = [gs, ge]
        answer["confidence"] = conf(gcond)
        if p_null >= self.null_threshold:
            answer["value"] = None
            return answer
        for (cs, ce), (_, cond) in ranked:
            value = parse_value(state[cs:ce], f)
            if value is not None:
                answer.update(guess=state[cs:ce], span=[cs, ce], confidence=conf(cond), value=value)
                return answer
        # Nothing in the candidate list is a valid value of this type.
        answer["value"] = None
        return answer

    def _decode_list(self, state: str, f: Field, parts) -> dict:
        p_any = max((p for _, p in parts), default=0.0)
        spans: list[tuple[int, int, float]] = []
        for items, _ in parts:
            for cs, ce, conf in items:
                cs, ce = _trim(state, cs, ce)
                if ce <= cs:
                    continue
                # Overlapping windows report the same item twice; keep one.
                dup = next((i for i, (s, e, _) in enumerate(spans) if s < ce and cs < e), None)
                if dup is None:
                    spans.append((cs, ce, conf))
                elif conf > spans[dup][2]:
                    spans[dup] = (cs, ce, conf)
        spans.sort()
        values, kept = [], []
        for cs, ce, conf in spans:
            v = parse_value(state[cs:ce], f)
            if v is not None:
                values.append(v)
                kept.append((cs, ce, conf))
        return {
            "type": "list",
            "items": f.items,
            "p_null": round(1 - p_any, 6),
            "value": values,
            "spans": [[s, e] for s, e, _ in kept],
            "confidence": [round(c, 6) for _, _, c in kept],
        }

    # -- public ---------------------------------------------------------------

    def extract(self, state: str, fields) -> dict:
        """`fields` is either the request's field map or a list of Field."""
        if isinstance(fields, dict):
            fields = parse_fields(fields)
        t0 = time.perf_counter()
        with self._lock:
            features = self.encoder.encode(state, fields)
            outputs = self._run(features)
        per_field: dict[str, list] = {f.name: [] for f in fields}
        for feat, out in zip(features, outputs):
            for i, f in enumerate(feat.fields):
                if f.is_list:
                    per_field[f.name].append(self._list_spans(feat, out["bio"][i]))
                else:
                    per_field[f.name].append(self._scalar_candidates(feat, out["start"][i], out["end"][i]))
        answers = {}
        for f in fields:
            parts = per_field[f.name]
            answers[f.name] = self._decode_list(state, f, parts) if f.is_list else self._decode_scalar(state, f, parts)
        return {
            "answers": answers,
            "usage": {
                "passes": len({(x.group, x.window) for x in features}),
                "input_tokens": sum(len(x.input_ids) for x in features),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        }
