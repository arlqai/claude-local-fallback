"""Label your own documents with a teacher LLM, then keep only verifiable answers.

QA datasets teach "find the answer to a question". Real extraction requests
look different: invoices, emails, logs, tickets, with schemas of short field
names. This script asks a teacher model to invent a realistic schema for each
document and answer it, then throws away every answer that is not a verbatim
span of the document (and every number or date that does not parse). What
survives is grounded by construction: the student never learns from a value
the teacher made up.

The teacher is any OpenAI-compatible endpoint. With claude-local-fallback
installed, LiteLLM on :4001 already serves both teachers:

    # llm-d on the cluster: large context, concurrent requests
    python -m sysone_extract.data.synth --docs ./my_docs --out synth.jsonl \\
        --base-url http://127.0.0.1:4001/v1 --model local-llmd --concurrency 16

    # or the local Qwen through mlx-lm (it serves one request at a time)
    python -m sysone_extract.data.synth --docs ./my_docs --out synth.jsonl \\
        --model local-qwen --concurrency 1

`--docs` is a directory of .txt/.md/.json/.eml/.log files, or a JSONL file
with a "text" key per line.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys

import httpx

from ..normalize import parse_date, parse_number
from . import slug, unique_name

SYSTEM = """You design data-extraction schemas and answer them from a document.

Given a document, return JSON only, no prose:
{"fields": [{"name": "...", "type": "text|number|date|list", "instructions": "...", "answer": ...}]}

Rules:
- 4 to 8 fields a developer would really ask for when processing this kind of document.
- name: short snake_case, like an API field (invoice_number, due_date, attendees).
- instructions: one short sentence a developer would write. Vary the phrasing.
- Use every type when the document allows it. "list" answers are arrays of strings.
- answer: copy the text EXACTLY as it appears in the document, character for character.
  Never reformat, compute, translate or summarise. For a number or date, copy the
  span that contains it ("$1,240.50", "March 3, 2026").
- Include 1 or 2 fields that are plausible for this kind of document but whose value
  is NOT in it. Their answer is null (for a list: []).
"""

DOC_EXTS = {".txt", ".md", ".json", ".eml", ".log", ".csv", ".html", ".xml", ".yaml", ".yml"}


def load_docs(path: str, min_chars: int, max_chars: int):
    if os.path.isfile(path) and path.endswith(".jsonl"):
        with open(path) as fh:
            for i, line in enumerate(fh):
                if line.strip():
                    obj = json.loads(line)
                    yield obj.get("id", f"doc-{i}"), obj["text"]
        return
    for root, _, files in os.walk(path):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in DOC_EXTS:
                continue
            full = os.path.join(root, name)
            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            text = text.strip()
            if len(text) < min_chars:
                continue
            # Long files are cut into chunks on paragraph boundaries.
            if len(text) <= max_chars:
                yield os.path.relpath(full, path), text
                continue
            chunk, part = "", 0
            for para in re.split(r"\n\s*\n", text):
                if chunk and len(chunk) + len(para) > max_chars:
                    part += 1
                    yield f"{os.path.relpath(full, path)}#{part}", chunk.strip()
                    chunk = ""
                chunk += para + "\n\n"
            if len(chunk.strip()) >= min_chars:
                yield f"{os.path.relpath(full, path)}#{part + 1}", chunk.strip()


def _json_from(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(text[start : end + 1])
    except ValueError:
        return None


def _locate(doc: str, quote: str, used: set) -> tuple[int, int] | None:
    """Character span of an exact quote; prefer an occurrence not used yet."""
    quote = quote.strip()
    if not quote:
        return None
    first = None
    for m in re.finditer(re.escape(quote), doc):
        span = (m.start(), m.end())
        if span not in used:
            return span
        first = first or span
    return first


def verify(doc: str, proposal: dict, stats: dict) -> tuple[dict, dict] | None:
    """Keep the fields whose answers are grounded and well-typed."""
    fields, answers, taken = {}, {}, set()
    for item in proposal.get("fields", []) if isinstance(proposal, dict) else []:
        if not isinstance(item, dict):
            continue
        ftype = item.get("type", "text")
        if ftype not in ("text", "number", "date", "list"):
            stats["bad_type"] += 1
            continue
        name = unique_name(slug(str(item.get("name") or item.get("instructions") or "field"), 40), taken)
        answer = item.get("answer")
        spans, used = [], set()
        if ftype == "list":
            values = answer if isinstance(answer, list) else ([] if answer in (None, "") else None)
            if values is None:
                stats["bad_answer"] += 1
                continue
            ok = True
            for v in values:
                span = _locate(doc, str(v), used) if isinstance(v, (str, int, float)) else None
                if span is None:
                    ok = False
                    break
                used.add(span)
                spans.append(span)
            if not ok:
                stats["not_grounded"] += 1
                continue
        elif answer not in (None, "", []):
            span = _locate(doc, str(answer), used)
            if span is None:
                stats["not_grounded"] += 1
                continue
            text = doc[span[0] : span[1]]
            if (ftype == "number" and parse_number(text) is None) or (ftype == "date" and parse_date(text) is None):
                stats["bad_value"] += 1
                continue
            spans.append(span)
        spec = {"type": ftype, "instructions": str(item.get("instructions", "")).strip()}
        if ftype == "list":
            spec["items"] = "text"
        fields[name] = spec
        answers[name] = [list(s) for s in sorted(spans)]
        stats["kept"] += 1
    if not fields:
        return None
    return fields, answers


async def label_one(client, args, doc_id, doc, sem, stats):
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Document:\n<<<\n{doc}\n>>>"},
        ],
        "temperature": args.temperature,
        "max_tokens": 1500,
    }
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(f"{args.base_url}/chat/completions", json=payload)
                if r.status_code in (429, 500, 502, 503, 504, 529):
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"] or ""
                break
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                if attempt == 2:
                    stats["errors"] += 1
                    print(f"  {doc_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    return None
                await asyncio.sleep(2 ** attempt)
        else:
            stats["errors"] += 1
            return None
    proposal = _json_from(content)
    if proposal is None:
        stats["bad_json"] += 1
        return None
    out = verify(doc, proposal, stats)
    if out is None:
        return None
    fields, answers = out
    return {"id": f"synth-{doc_id}", "source": f"synth:{args.model}", "state": doc, "fields": fields, "answers": answers}


async def run(args):
    docs = list(load_docs(args.docs, args.min_chars, args.max_chars))
    random.Random(0).shuffle(docs)
    if args.limit:
        docs = docs[: args.limit]
    print(f"{len(docs)} documents -> teacher {args.model} at {args.base_url}")
    headers = {"authorization": f"Bearer {os.environ.get(args.api_key_env, 'local')}"}
    stats = {k: 0 for k in ("kept", "not_grounded", "bad_value", "bad_type", "bad_answer", "bad_json", "errors")}
    sem = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(connect=10, read=args.timeout, write=30, pool=None)
    written = 0
    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        tasks = [asyncio.create_task(label_one(client, args, i, d, sem, stats)) for i, d in docs]
        with open(args.out, "a" if args.append else "w") as fh:
            for done, task in enumerate(asyncio.as_completed(tasks), 1):
                rec = await task
                if rec:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    written += 1
                if done % 25 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)} docs, {written} records, fields {stats}")
    print(f"wrote {written} records to {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--base-url", default="http://127.0.0.1:4001/v1")
    ap.add_argument("--model", default="local-llmd")
    ap.add_argument("--api-key-env", default="TEACHER_API_KEY")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
