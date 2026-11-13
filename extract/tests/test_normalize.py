import pytest

from sysone_extract.normalize import infer_type, parse_date, parse_number
from sysone_extract.schema import Field


@pytest.mark.parametrize("text,value", [
    ("$649.00", 649.0),
    ("701.10 USD", 701.1),
    ("1,234,567", 1234567),
    ("1.234,56 €", 1234.56),
    ("1 234,5", 1234.5),
    ("3,5", 3.5),
    ("12%", 12),
    ("(1,200.00)", -1200.0),
    ("-42", -42),
    ("2.5 million", 2_500_000.0),
    ("about 30 days", 30),
    ("1'250.75 CHF", 1250.75),
])
def test_parse_number(text, value):
    assert parse_number(text) == pytest.approx(value)


@pytest.mark.parametrize("text", ["$649.00 plus $52.10", "order A-88412 arrived", "", "cracked screen"])
def test_parse_number_rejects(text):
    assert parse_number(text) is None


def test_number_options():
    assert parse_number("$701.1", Field("x", "number", decimals=2)) == 701.1
    assert parse_number("11.6 days", Field("x", "number", integer=True)) == 12


@pytest.mark.parametrize("text,iso", [
    ("Sept 28, 2026", "2026-09-28"),
    ("28 September 2026", "2026-09-28"),
    ("2026-10-09", "2026-10-09"),
    ("October 9, 2026", "2026-10-09"),
])
def test_parse_date(text, iso):
    assert parse_date(text) == iso


def test_parse_date_dayfirst():
    assert parse_date("03/04/2026", Field("d", "date", dayfirst=True)) == "2026-04-03"
    assert parse_date("03/04/2026") == "2026-03-04"


@pytest.mark.parametrize("text", ["March 2026", "the 28th", "2026", "649.00", "tomorrow"])
def test_parse_date_rejects_partial(text):
    assert parse_date(text) is None


def test_infer_type():
    assert infer_type("Sept 28, 2026") == "date"
    assert infer_type("$1,200") == "number"
    assert infer_type("the Eiffel Tower") == "text"
    assert infer_type("about 30 days") == "text"
