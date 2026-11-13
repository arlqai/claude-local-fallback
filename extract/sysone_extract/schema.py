"""Request and training-record schema.

A request is a document (`state`) plus a map of named, typed fields. A training
record adds the gold answers as character spans into `state`:

    {
      "id": "...",
      "state": "Hi, order A-88412 arrived on Sept 28, 2026 ...",
      "fields": {
        "order_id":     {"type": "text",   "instructions": "Order number"},
        "delivered_on": {"type": "date",   "instructions": "Delivery date"},
        "damage":       {"type": "list",   "instructions": "Each kind of damage"},
        "tracking":     {"type": "text"}
      },
      "answers": {
        "order_id":     [[10, 17]],
        "delivered_on": [[29, 42]],
        "damage":       [[50, 64], [71, 84]],
        "tracking":     []
      }
    }

For scalar types (text, number, date) a list of several spans means "any of
these is acceptable" (SQuAD-style alternatives); training uses the first. For
`list`, every span is one item. An empty list means the value is absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SCALAR_TYPES = ("text", "number", "date")
TYPES = SCALAR_TYPES + ("list",)
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class SchemaError(ValueError):
    """A request or record that cannot be processed."""


@dataclass
class Field:
    name: str
    type: str = "text"
    instructions: str = ""
    # number options
    decimals: int | None = None
    integer: bool = False
    unit: str | None = None
    # list options: the type of each item
    items: str = "text"
    # date options
    dayfirst: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def is_list(self) -> bool:
        return self.type == "list"

    @property
    def value_type(self) -> str:
        """The type each extracted value must parse as."""
        return self.items if self.is_list else self.type


def parse_field(name: str, spec: dict | None) -> Field:
    if not NAME_RE.match(name):
        raise SchemaError(f"field name {name!r} must match {NAME_RE.pattern}")
    spec = dict(spec or {})
    ftype = spec.pop("type", "text")
    if ftype not in TYPES:
        raise SchemaError(f"field {name!r}: unsupported type {ftype!r} (supported: {', '.join(TYPES)})")
    items = spec.pop("items", "text")
    if items not in SCALAR_TYPES:
        raise SchemaError(f"field {name!r}: list items must be one of {SCALAR_TYPES}")
    decimals = spec.pop("decimals", None)
    if decimals is not None and (not isinstance(decimals, int) or not 0 <= decimals <= 12):
        raise SchemaError(f"field {name!r}: decimals must be an integer in 0..12")
    instructions = spec.pop("instructions", "") or ""
    if not isinstance(instructions, str):
        raise SchemaError(f"field {name!r}: instructions must be a string")
    return Field(
        name=name,
        type=ftype,
        instructions=instructions.strip(),
        decimals=decimals,
        integer=bool(spec.pop("integer", False)),
        unit=spec.pop("unit", None),
        items=items,
        dayfirst=bool(spec.pop("dayfirst", False)),
        extra=spec,
    )


def parse_fields(fields: dict) -> list[Field]:
    if not isinstance(fields, dict) or not fields:
        raise SchemaError("`fields` must be a non-empty object")
    return [parse_field(name, spec) for name, spec in fields.items()]


def validate_record(record: dict) -> tuple[str, list[Field], dict[str, list[tuple[int, int]]]]:
    """Check a training record and return (state, fields, answers)."""
    state = record.get("state")
    if not isinstance(state, str) or not state:
        raise SchemaError("record has no `state` text")
    fields = parse_fields(record.get("fields", {}))
    raw = record.get("answers", {}) or {}
    answers: dict[str, list[tuple[int, int]]] = {}
    for f in fields:
        spans = raw.get(f.name, [])
        if spans is None:
            spans = []
        clean = []
        for span in spans:
            s, e = int(span[0]), int(span[1])
            if not 0 <= s < e <= len(state):
                raise SchemaError(f"field {f.name!r}: span {span} outside the state")
            clean.append((s, e))
        answers[f.name] = clean
    return state, fields, answers
