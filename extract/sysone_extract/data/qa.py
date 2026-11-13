"""Convert extractive QA datasets into multi-field extraction records.

Any SQuAD-format dataset works: every question about the same passage becomes
one field of a single record, typed from its gold answer (dates and numbers
are recognised, the rest is text), and unanswerable questions become absent
fields. SQuAD 2.0 is the natural first source because about a third of its
questions have no answer, which is what teaches calibrated nulls.

    # from a SQuAD JSON file
    python -m sysone_extract.data.qa --squad-json train-v2.0.json --out squad_train.jsonl
    # or straight from the Hugging Face Hub (needs `pip install datasets`)
    python -m sysone_extract.data.qa --hf rajpurkar/squad_v2 --split train --out squad_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict

from ..normalize import infer_type
from . import slug, unique_name, write_jsonl


def _rows_from_squad_json(path: str):
    with open(path) as fh:
        data = json.load(fh)["data"]
    for article in data:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                answers = [] if qa.get("is_impossible") else qa.get("answers", [])
                yield {
                    "context": para["context"],
                    "question": qa["question"],
                    "answers": {"text": [a["text"] for a in answers], "answer_start": [a["answer_start"] for a in answers]},
                }


def _rows_from_hf(name: str, split: str):
    from datasets import load_dataset  # optional dependency

    for row in load_dataset(name, split=split):
        yield {"context": row["context"], "question": row["question"], "answers": row["answers"]}


def build_records(rows, max_fields: int = 8, seed: int = 0, prefix: str = "qa"):
    """Group rows by passage, then emit records of up to `max_fields` fields."""
    rng = random.Random(seed)
    by_context: "OrderedDict[str, list]" = OrderedDict()
    for row in rows:
        by_context.setdefault(row["context"], []).append(row)
    n = 0
    for context, rows_ in by_context.items():
        rng.shuffle(rows_)
        for i in range(0, len(rows_), max_fields):
            chunk = rows_[i : i + max_fields]
            fields, answers, taken = {}, {}, set()
            for row in chunk:
                texts = row["answers"]["text"]
                starts = row["answers"]["answer_start"]
                spans = []
                for t, s in zip(texts, starts):
                    if context[s : s + len(t)] == t and (s, s + len(t)) not in spans:
                        spans.append((s, s + len(t)))
                if texts and not spans:
                    continue  # misaligned gold answer: skip rather than train on noise
                name = unique_name(slug(row["question"]), taken)
                ftype = infer_type(context[spans[0][0] : spans[0][1]]) if spans else "text"
                fields[name] = {"type": ftype, "instructions": row["question"].strip()}
                answers[name] = [list(s) for s in spans]
            if fields:
                n += 1
                yield {"id": f"{prefix}-{n}", "source": prefix, "state": context, "fields": fields, "answers": answers}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--squad-json", help="SQuAD-format JSON file")
    src.add_argument("--hf", help="Hugging Face dataset with context/question/answers columns")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-fields", type=int, default=8)
    ap.add_argument("--prefix", default="squad")
    args = ap.parse_args(argv)
    rows = _rows_from_squad_json(args.squad_json) if args.squad_json else _rows_from_hf(args.hf, args.split)
    n = write_jsonl(args.out, build_records(rows, args.max_fields, prefix=args.prefix))
    print(f"wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
