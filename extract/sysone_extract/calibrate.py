"""Calibrate `confidence` and `p_null` on held-out records (Platt scaling).

Raw span probabilities are spread over subword tokens, so they understate how
often the top span is right. A two-parameter logistic map per score, fitted
on held-out data, fixes that without touching the model; the parameters are
stored in extractor.json and applied at inference.

    python -m sysone_extract.calibrate --model runs/v0 --data calib.jsonl

Use records the checkpoint was not selected on when you can: train.py
calibrates on its dev set by default, which is slightly optimistic.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

from .data import read_jsonl
from .normalize import normalize_answer, parse_value
from .schema import validate_record


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def apply(p: float, params) -> float:
    if not params:
        return p
    a, b = params
    return 1 / (1 + math.exp(-(a * _logit(p) + b)))


def fit_platt(pairs: list[tuple[float, bool]]) -> list[float] | None:
    """Fit sigmoid(a * logit(p) + b) to (score, outcome) pairs."""
    if len(pairs) < 20 or len({ok for _, ok in pairs}) < 2:
        return None  # too few, or only one outcome: nothing to learn
    x = torch.tensor([_logit(p) for p, _ in pairs], dtype=torch.float64)
    y = torch.tensor([float(ok) for _, ok in pairs], dtype=torch.float64)
    params = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([params], lr=0.5, max_iter=200)

    def closure():
        opt.zero_grad()
        z = params[0] * x + params[1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + 1e-4 * (params ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return [round(float(v), 6) for v in params.detach()]


def collect(extractor, records) -> tuple[list, list]:
    """Uncalibrated (confidence, correct) and (p_null, absent) pairs."""
    saved, extractor.calibration = extractor.calibration, None
    conf, null = [], []
    try:
        for rec in records:
            state, fields, answers = validate_record(rec)
            out = extractor.extract(state, fields)["answers"]
            for f in fields:
                if f.is_list:
                    continue
                pred, gold = out[f.name], answers[f.name]
                null.append((pred["p_null"], not gold))
                if gold and pred["guess"] is not None:
                    texts = [state[s:e] for s, e in gold]
                    if f.value_type == "text":
                        ok = any(normalize_answer(pred["guess"]) == normalize_answer(t) for t in texts)
                    else:
                        ok = pred["value"] is not None and any(pred["value"] == parse_value(t, f) for t in texts)
                    conf.append((pred["confidence"], ok))
    finally:
        extractor.calibration = saved
    return conf, null


def calibrate(extractor, records, model_dir: str | None = None) -> dict:
    conf, null = collect(extractor, records)
    calib = {"confidence": fit_platt(conf), "p_null": fit_platt(null), "fitted_on": len(records)}
    extractor.calibration = calib
    if model_dir:
        path = os.path.join(model_dir, "extractor.json")
        with open(path) as fh:
            config = json.load(fh)
        config["calibration"] = calib
        with open(path, "w") as fh:
            json.dump(config, fh, indent=2)
    return calib


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args(argv)
    from .evaluate import evaluate
    from .inference import Extractor

    extractor = Extractor.load(args.model, device=args.device)
    records = list(read_jsonl(args.data))[: args.limit or None]
    print("calibration:", json.dumps(calibrate(extractor, records, args.model)))
    print("after:", json.dumps(evaluate(extractor, records)["overall"]))


if __name__ == "__main__":
    main()
