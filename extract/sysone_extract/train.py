"""Fine-tune a backbone with extraction heads.

    python -m sysone_extract.train \\
        --train squad_train.jsonl conll_train.jsonl synth.jsonl \\
        --dev squad_dev.jsonl \\
        --backbone answerdotai/ModernBERT-base --out runs/v0

Works on CUDA, Apple Silicon (MPS) and CPU. Every epoch it evaluates on the
dev set and keeps the best checkpoint in --out.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from .data import read_jsonl
from .encoding import Encoder
from .evaluate import evaluate
from .inference import Extractor, pick_device
from .model import ExtractorModel, build, collate, model_inputs
from .schema import SchemaError, validate_record


def augment_fields(fields, rng: random.Random, drop: float):
    """Shuffle field order and drop some fields.

    Without this, a schema that always comes in the same order lets the model
    learn "the third query is the date" instead of reading the field's name
    and instructions, and it falls apart on any other schema.
    """
    fields = list(fields)
    rng.shuffle(fields)
    if drop > 0 and len(fields) > 1:
        kept = [f for f in fields if rng.random() >= drop]
        fields = kept or [rng.choice(fields)]
    return fields


def prepare(records, encoder: Encoder, stats: dict):
    """Validate records and tokenize each document once."""
    out = []
    for rec in records:
        try:
            state, fields, answers = validate_record(rec)
        except SchemaError:
            stats["bad_records"] = stats.get("bad_records", 0) + 1
            continue
        out.append((state, fields, answers, encoder.tokenize_doc(state)))
    return out


def make_features(prepared, encoder: Encoder, rng: random.Random, null_windows: int = 1,
                  field_drop: float = 0.0):
    """One epoch of training features, with a fresh field augmentation.

    Long documents keep every window that holds an answer plus `null_windows`
    windows without one, so the model also learns to say "not here".
    """
    feats = []
    for state, fields, answers, doc in prepared:
        fields = augment_fields(fields, rng, field_drop)
        windows = [encoder.add_labels(f, answers) for f in encoder.encode(state, fields, doc=doc)]
        with_ans = [f for f in windows if encoder.window_has_answer(f)]
        without = [f for f in windows if not encoder.window_has_answer(f)]
        rng.shuffle(without)
        feats.extend(with_ans + without[: max(null_windows, 0 if with_ans else 1)])
    return feats


def batches(features, batch_size: int, max_tokens: int, rng: random.Random):
    """Length-bucketed batches: sort within chunks, then shuffle the batches."""
    order = list(range(len(features)))
    rng.shuffle(order)
    out = []
    chunk = batch_size * 50
    for i in range(0, len(order), chunk):
        block = sorted(order[i : i + chunk], key=lambda j: len(features[j].input_ids))
        cur, longest = [], 0
        for j in block:
            n = len(features[j].input_ids)
            if cur and (len(cur) >= batch_size or max(longest, n) * (len(cur) + 1) > max_tokens):
                out.append(cur)
                cur, longest = [], 0
            cur.append(j)
            longest = max(longest, n)
        if cur:
            out.append(cur)
    rng.shuffle(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev")
    ap.add_argument("--dev-frac", type=float, default=0.02, help="held out from --train when --dev is absent")
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    ap.add_argument("--init", help="continue from a saved extractor directory instead of --backbone")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--field-drop", type=float, default=0.3,
                    help="probability of dropping each field of a record, per epoch (fields are always shuffled)")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batch-tokens", type=int, default=8192)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5, help="backbone learning rate")
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast (CUDA, recent MPS)")
    ap.add_argument("--dev-limit", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=0, help="steps; 0 = once per epoch")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-calibrate", action="store_true", help="skip fitting confidence calibration on --dev")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    records = [r for path in args.train for r in read_jsonl(path)]
    rng.shuffle(records)
    if args.dev:
        dev = list(read_jsonl(args.dev))
    else:
        n_dev = max(1, int(len(records) * args.dev_frac))
        dev, records = records[:n_dev], records[n_dev:]
    dev = dev[: args.dev_limit] if args.dev_limit else dev
    print(f"train records {len(records)}, dev records {len(dev)}, device {device}")

    if args.init:
        model, tokenizer, _ = ExtractorModel.load(args.init, device="cpu")
        backbone_name = json.load(open(os.path.join(args.init, "extractor.json"))).get("backbone", args.init)
    else:
        model, tokenizer = build(args.backbone)
        backbone_name = args.backbone
    model.to(device).train()

    encoder = Encoder(tokenizer, max_len=args.max_len, stride=args.stride)
    stats = {}
    prepared = prepare(records, encoder, stats)

    def epoch_features():
        return make_features(prepared, encoder, rng, field_drop=args.field_drop)

    features = epoch_features()
    n_fields = sum(len(f.fields) for f in features)
    print(f"{len(features)} training windows, {n_fields} field queries per epoch {stats or ''}")

    head_params = list(model.head_parameters())
    head_ids = {id(p) for p in head_params}
    decay, no_decay = [], []
    for name, p in model.backbone.named_parameters():
        (no_decay if p.ndim < 2 or "norm" in name.lower() or "bias" in name else decay).append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": no_decay, "lr": args.lr, "weight_decay": 0.0},
            {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0},
        ]
    )
    assert not any(id(p) in head_ids for p in decay + no_decay)

    steps_per_epoch = math.ceil(len(batches(features, args.batch_size, args.max_batch_tokens, random.Random(0))) / args.grad_accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup = int(total_steps * args.warmup)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    pad = tokenizer.pad_token_id or 0
    autocast = torch.autocast(device.type, dtype=torch.bfloat16) if args.bf16 else torch.autocast(device.type, enabled=False)
    extractor = Extractor(model, tokenizer, max_len=args.max_len, stride=args.stride, device=device)

    best = -1.0
    step, micro, running, t0 = 0, 0, 0.0, time.time()
    extra = {"backbone": backbone_name, "max_len": args.max_len, "stride": args.stride}

    def run_eval(tag):
        nonlocal best
        if not dev:
            model.save(args.out, tokenizer, extra)
            return
        report = evaluate(extractor, dev)
        print(f"[eval {tag}] score {report['score']} overall {report['overall']}")
        if report["score"] > best:
            best = report["score"]
            model.save(args.out, tokenizer, {**extra, "dev_report": report})
            print(f"  saved best checkpoint to {args.out}")
        model.train()

    epoch = 0
    while step < total_steps:
        epoch += 1
        if epoch > 1:
            features = epoch_features()
        for idx in batches(features, args.batch_size, args.max_batch_tokens, rng):
            batch = collate([features[i] for i in idx], pad, device=device)
            with autocast:
                out = model(**model_inputs(batch))
            loss = ExtractorModel.loss(out, batch) / args.grad_accum
            loss.backward()
            running += loss.item()
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                print(f"step {step}/{total_steps} epoch {epoch} loss {running / args.log_every:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step", flush=True)
                running = 0.0
            if args.eval_every and step % args.eval_every == 0:
                run_eval(f"step {step}")
            if step >= total_steps:
                break
        if not args.eval_every:
            run_eval(f"epoch {epoch}")
    if args.eval_every:
        run_eval("final")
    print(f"done; best dev score {best}")

    if dev and not args.no_calibrate:
        from .calibrate import calibrate

        best_model = Extractor.load(args.out, device=str(device))
        calib = calibrate(best_model, dev, args.out)
        report = evaluate(best_model, dev)
        print(f"calibrated on {len(dev)} dev records: {calib}")
        print(f"  after calibration: {report['overall']}")


if __name__ == "__main__":
    main()
