"""Train a tiny random backbone on templated documents, then serve it.

This checks that labels, heads, decoding, saving, calibration and the HTTP
API are wired together correctly. It says nothing about real-world accuracy.
"""

import importlib
import json

import pytest

from sysone_extract import train
from sysone_extract.data import write_jsonl
from sysone_extract.inference import Extractor

from .fixtures import build_backbone, make_records, perturb_schemas

EMAIL = ("Hello, my order C-55120 was delivered on 3 March 2026 with scratched lid and "
         "broken hinge. I paid $1249.90. Please fix this. Regards, Priya")
FIELDS = {
    "tracking_number": {"type": "text", "instructions": "Shipment tracking number"},
    "amount_paid": {"type": "number", "decimals": 2, "instructions": "Amount paid"},
    "order_id": {"type": "text", "instructions": "Order number"},
    "damage": {"type": "list", "instructions": "Each kind of damage described"},
    "delivered_on": {"type": "date", "instructions": "Delivery date"},
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    backbone = build_backbone(str(root / "backbone"), "llama")
    write_jsonl(root / "train.jsonl", make_records(600, seed=1))
    write_jsonl(root / "dev.jsonl", perturb_schemas(make_records(60, seed=2), seed=3))
    out = root / "model"
    train.main([
        "--train", str(root / "train.jsonl"), "--dev", str(root / "dev.jsonl"),
        "--backbone", backbone, "--out", str(out), "--epochs", "12",
        "--batch-size", "16", "--grad-accum", "1", "--lr", "5e-4", "--head-lr", "1e-3",
        "--log-every", "10000", "--device", "cpu",
    ])
    return out


def test_learns_unseen_schemas(trained):
    config = json.loads((trained / "extractor.json").read_text())
    assert config["dev_report"]["overall"]["exact_match"] > 0.9
    assert config["dev_report"]["overall"]["null_accuracy"] > 0.9
    assert "calibration" in config


def test_typed_answers(trained):
    ex = Extractor.load(str(trained), device="cpu")
    out = ex.extract(EMAIL, FIELDS)
    a = out["answers"]
    assert a["order_id"]["value"] == "C-55120"
    assert EMAIL[slice(*a["order_id"]["span"])] == "C-55120"
    assert a["delivered_on"]["value"] == "2026-03-03"
    assert a["amount_paid"]["value"] == pytest.approx(1249.9)
    assert a["damage"]["value"] == ["scratched lid", "broken hinge"]
    assert a["tracking_number"]["value"] is None
    assert a["tracking_number"]["p_null"] > 0.5
    assert out["usage"]["passes"] == 1


def test_http_api(trained, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("EXTRACT_MODEL", str(trained))
    monkeypatch.setenv("EXTRACT_DEVICE", "cpu")
    serve = importlib.reload(importlib.import_module("sysone_extract.serve"))
    with TestClient(serve.app) as client:
        assert client.get("/health").json()["loaded"] is False  # lazy load
        r = client.post("/v1/extract", json={"state": EMAIL, "fields": FIELDS})
        assert r.status_code == 200, r.text
        assert r.json()["answers"]["order_id"]["value"] == "C-55120"
        assert client.get("/health").json()["loaded"] is True
        bad = client.post("/v1/extract", json={"state": EMAIL, "fields": {"x": {"type": "colour"}}})
        assert bad.status_code == 400
        assert client.post("/v1/unload").json() == {"loaded": False}
