"""Dataset builders. Each writes training records (see schema.py) as JSONL."""

import json
import re


def slug(text: str, max_len: int = 48) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    name = "_".join(words)[:max_len].strip("_") or "field"
    if name[0].isdigit():
        name = "f_" + name
    return name


def unique_name(name: str, taken: set) -> str:
    out, i = name, 2
    while out in taken:
        out = f"{name}_{i}"
        i += 1
    taken.add(out)
    return out


def write_jsonl(path: str, records) -> int:
    n = 0
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
