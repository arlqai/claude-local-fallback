"""Score a model on held-out records.

Reports, per field type: exact match and token F1 of the returned value
against the gold span(s), null accuracy (did it return null exactly when the
gold is absent), list precision/recall/F1, calibration of `confidence`
(expected calibration error, ECE), and latency.

    python -m sysone_extract.evaluate --model runs/v0 --data dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict

from .data import read_jsonl
from .normalize import normalize_answer, parse_value
from .schema import validate_record


def _f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g:
        return float(p == g)
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def _value_match(pred, gold_text: str, field) -> bool:
    if field.value_type == "text":
        return normalize_answer(str(pred)) == normalize_answer(gold_text)
    gold = parse_value(gold_text, field)
    if gold is None:  # gold span not parseable: fall back to text match
        return False
    if field.value_type == "number":
        return abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    return pred == gold


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float | None:
    if not pairs:
        return None
    buckets = defaultdict(list)
    for conf, ok in pairs:
        buckets[min(bins - 1, int(conf * bins))].append((conf, ok))
    total = len(pairs)
    return sum(
        len(b) / total * abs(statistics.fmean(c for c, _ in b) - statistics.fmean(float(o) for _, o in b))
        for b in buckets.values()
    )


def evaluate(extractor, records, limit: int = 0) -> dict:
    stats = defaultdict(lambda: defaultdict(float))
    calib: list[tuple[float, bool]] = []
    null_calib: list[tuple[float, bool]] = []
    latencies = []
    n_records = 0
    for rec in records:
        state, fields, answers = validate_record(rec)
        out = extractor.extract(state, fields)
        latencies.append(out["usage"]["latency_ms"])
        n_records += 1
        for f in fields:
            pred = out["answers"][f.name]
            gold_spans = answers[f.name]
            gold_texts = [state[s:e] for s, e in gold_spans]
            st = stats[f.type]
            st["n"] += 1
            if f.is_list:
                p = {normalize_answer(state[s:e]) for s, e in pred["spans"]}
                g = {normalize_answer(t) for t in gold_texts}
                st["tp"] += len(p & g)
                st["fp"] += len(p - g)
                st["fn"] += len(g - p)
                st["null_ok"] += float((not p) == (not g))
                continue
            gold_absent = not gold_spans
            pred_absent = pred["value"] is None
            st["null_ok"] += float(gold_absent == pred_absent)
            null_calib.append((pred["p_null"], gold_absent))
            if gold_absent:
                st["absent"] += 1
                continue
            st["present"] += 1
            if pred_absent:
                continue
            em = any(_value_match(pred["value"], t, f) for t in gold_texts)
            f1 = max(_f1(pred["guess"] or "", t) for t in gold_texts)
            st["em"] += float(em)
            st["f1"] += f1
            calib.append((pred["confidence"], em))

    report = {"records": n_records, "by_type": {}}
    all_present = all_em = all_f1 = all_null = all_n = 0
    for ftype, st in sorted(stats.items()):
        row = {"fields": int(st["n"]), "null_accuracy": round(st["null_ok"] / st["n"], 4)}
        if ftype == "list":
            tp, fp, fn = st["tp"], st["fp"], st["fn"]
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec_ = tp / (tp + fn) if tp + fn else 0.0
            row.update(
                precision=round(prec, 4), recall=round(rec_, 4),
                f1=round(2 * prec * rec_ / (prec + rec_), 4) if prec + rec_ else 0.0,
            )
        else:
            present = st["present"]
            row.update(
                present=int(present), absent=int(st["absent"]),
                exact_match=round(st["em"] / present, 4) if present else None,
                f1=round(st["f1"] / present, 4) if present else None,
            )
            all_present += present
            all_em += st["em"]
            all_f1 += st["f1"]
        all_null += st["null_ok"]
        all_n += st["n"]
        report["by_type"][ftype] = row
    report["overall"] = {
        "exact_match": round(all_em / all_present, 4) if all_present else None,
        "f1": round(all_f1 / all_present, 4) if all_present else None,
        "null_accuracy": round(all_null / all_n, 4) if all_n else None,
        "confidence_ece": round(ece(calib), 4) if calib else None,
        "p_null_ece": round(ece(null_calib), 4) if null_calib else None,
    }
    if latencies:
        lat = sorted(latencies)
        report["latency_ms"] = {"p50": lat[len(lat) // 2], "p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))]}
    # One number to select checkpoints on: answer quality and null handling.
    o = report["overall"]
    report["score"] = round(((o["f1"] or 0) + (o["null_accuracy"] or 0)) / 2, 4)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    from .inference import Extractor

    extractor = Extractor.load(args.model, device=args.device)
    records = list(read_jsonl(args.data))
    if args.limit:
        records = records[: args.limit]
    report = evaluate(extractor, records)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
