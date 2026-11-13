"""Typed parsing of extracted spans.

The model only ever points at text in the input. These functions decide
whether a span is a valid value of the requested type and convert it, which
is what makes every answer either a well-typed value or null: the decoder
walks the model's candidate spans in order and keeps the first one that
parses.
"""

from __future__ import annotations

import datetime as dt
import re
import string

from dateutil import parser as dateparser

from .schema import Field

_MULTIPLIERS = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mn": 1e6, "million": 1e6, "millions": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "billions": 1e9,
    "md": 1e9, "milliard": 1e9, "milliards": 1e9,
}
# A run of digits with separators. Separators may be , . space, NBSP, narrow
# NBSP or apostrophe (Swiss thousands).
_NUMBER_RE = re.compile(r"\d[\d.,  ' ]*\d|\d")
# What may surround the number in a span that still counts as "a number".
_ALLOWED_WORD_RE = re.compile(
    r"^(?:[$€£¥₹%]|usd|eur|gbp|chf|jpy|cad|aud|dollars?|euros?|pounds?|"
    r"percent|pct|per\s?cent|k|m|mn|b|bn|md|thousand|millions?|billions?|"
    r"milliards?|approx\.?|about|around|~|≈|\+|-|−|\(|\)|units?|items?|"
    r"days?|hours?|minutes?|years?|kg|g|km|m|cm|mm|lbs?|tons?)$",
    re.I,
)


def _to_float(digits: str) -> float | None:
    """Interpret one digit run, deciding which separator is the decimal mark."""
    s = digits.replace(" ", " ").replace(" ", " ").replace("'", " ")
    s = s.replace(" ", "")
    if "," in s and "." in s:
        # Whichever comes last is the decimal mark: 1,234.56 / 1.234,56
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and parts[0] != "0"):
            s = s.replace(",", "")  # thousands: 1,234 or 1,234,567
        else:
            s = s.replace(",", ".")  # decimal comma: 3,5
    elif s.count(".") > 1:
        s = s.replace(".", "")  # 1.234.567
    try:
        return float(s)
    except ValueError:
        return None


def parse_number(text: str, f: Field | None = None) -> float | int | None:
    """Parse a span that contains exactly one number, else None.

    "$649.00" -> 649.0, "1 234,5 €" -> 1234.5, "(12)" -> -12, "2.5 million"
    -> 2500000.0. A span with two numbers ("$649 plus $52") is rejected: it is
    ambiguous, and the decoder moves on to the next candidate.
    """
    text = text.strip()
    matches = list(_NUMBER_RE.finditer(text))
    if len(matches) != 1:
        return None
    m = matches[0]
    value = _to_float(m.group(0).strip(" ,."))
    if value is None:
        return None
    before, after = text[: m.start()], text[m.end():]
    for word in re.findall(r"[^\s\d]+", before + " " + after):
        word = word.strip(string.punctuation.replace("%", "").replace("$", "")) or word
        if word and not _ALLOWED_WORD_RE.match(word):
            return None
    tail = after.strip().lower().rstrip(".")
    for word, factor in _MULTIPLIERS.items():
        if re.match(rf"^{word}\b", tail):
            value *= factor
            break
    stripped = before.strip()
    if stripped.endswith(("-", "−")) or (stripped.endswith("(") and after.strip().startswith(")")):
        value = -value
    if f is not None:
        if f.integer:
            return int(round(value))
        if f.decimals is not None:
            return round(value, f.decimals)
    return int(value) if value.is_integer() and "." not in m.group(0) and "," not in m.group(0) else value


_SENTINELS = (dt.datetime(1904, 1, 1), dt.datetime(1908, 2, 2))
_MONTH_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|janv|févr|fevr|mars|avr|mai|juin|juil|"
    r"août|aout|sept|oct|nov|déc|dec)[a-zé]*\.?\b",
    re.I,
)


def parse_date(text: str, f: Field | None = None) -> str | None:
    """Parse a span holding a full calendar date into ISO YYYY-MM-DD, else None.

    Partial dates ("March 2026", "the 28th") are rejected in v0: the model
    should point at a span that pins the date down.
    """
    text = text.strip()
    if not re.search(r"\d", text):
        return None
    if re.fullmatch(r"[\d\s.,]+", text) and not re.search(r"\d[./-]\d", text):
        return None  # a bare number is not a date
    dayfirst = bool(f and f.dayfirst)
    try:
        a = dateparser.parse(text, default=_SENTINELS[0], dayfirst=dayfirst)
        b = dateparser.parse(text, default=_SENTINELS[1], dayfirst=dayfirst)
    except (ValueError, OverflowError, TypeError):
        return None
    if a is None or b is None:
        return None
    # A component that differs between the two parses came from the defaults.
    if a.year != b.year or a.month != b.month or a.day != b.day:
        return None
    return a.date().isoformat()


def parse_text(text: str, f: Field | None = None) -> str | None:
    text = text.strip()
    return text or None


PARSERS = {"text": parse_text, "number": parse_number, "date": parse_date}


def parse_value(text: str, f: Field):
    """Parse `text` as a value of the field's (item) type, or None."""
    return PARSERS[f.value_type](text, f)


def infer_type(text: str) -> str:
    """Best-guess type for a gold answer, used when converting QA datasets."""
    if parse_date(text) is not None:
        return "date"
    if parse_number(text) is not None and re.fullmatch(
        r"[-−+(]?\s*[$€£¥]?\s*[\d.,  ' ]+\s*(%|percent|[$€£¥])?\s*\)?", text.strip()
    ):
        return "number"
    return "text"


# SQuAD's answer normalisation, used for EM/F1.
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())
