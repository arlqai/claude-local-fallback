import pytest
from transformers import AutoTokenizer

from sysone_extract.encoding import BIO_B, BIO_I, IGNORE, Encoder
from sysone_extract.schema import parse_fields, validate_record

from .fixtures import build_backbone, make_records


@pytest.fixture(scope="module", params=["bert", "llama"])
def tokenizer(request, tmp_path_factory):
    path = build_backbone(str(tmp_path_factory.mktemp(request.param)), request.param)
    return AutoTokenizer.from_pretrained(path)


def _token_text(feature, state, a, b):
    s = feature.doc_offsets[a - feature.doc_start][0]
    e = feature.doc_offsets[b - feature.doc_start][1]
    return state[s:e].strip()


def test_labels_point_at_gold_spans(tokenizer):
    enc = Encoder(tokenizer, max_len=512)
    for rec in make_records(30, seed=1):
        state, fields, answers = validate_record(rec)
        (feat,) = enc.encode(state, fields)
        enc.add_labels(feat, answers)
        assert len(feat.input_ids) <= 512
        for i, f in enumerate(feat.fields):
            gold = [state[s:e] for s, e in answers[f.name]]
            if f.is_list:
                tags = feat.bio_labels[i]
                found, cur = [], None
                for p, t in enumerate(tags):
                    if t == BIO_B:
                        cur = [p, p]
                        found.append(cur)
                    elif t == BIO_I:
                        cur[1] = p
                assert [_token_text(feat, state, a, b) for a, b in found] == gold
                assert all(t == IGNORE for t in tags[: feat.doc_start])
            elif not gold:
                assert feat.start_labels[i] == 0 and feat.end_labels[i] == 0
            else:
                a, b = feat.start_labels[i] - 1, feat.end_labels[i] - 1
                assert _token_text(feat, state, a, b) == gold[0]


def test_query_markers_follow_document(tokenizer):
    enc = Encoder(tokenizer, max_len=512)
    rec = make_records(1)[0]
    state, fields, _ = validate_record(rec)
    (feat,) = enc.encode(state, fields)
    assert len(feat.query_pos) == len(fields)
    assert min(feat.query_pos) >= feat.doc_end


def test_long_documents_are_windowed(tokenizer):
    enc = Encoder(tokenizer, max_len=128, stride=32)
    rec = make_records(1, seed=3)[0]
    filler = "Nothing relevant here. " * 60
    state = filler + rec["state"] + " " + filler
    shift = len(filler)
    answers = {k: [[s + shift, e + shift] for s, e in v] for k, v in rec["answers"].items()}
    fields = parse_fields(rec["fields"])
    feats = enc.encode(state, fields)
    assert len({f.window for f in feats}) > 1
    for f in feats:
        assert len(f.input_ids) <= 128
    # Every gold span is fully inside at least one window.
    labelled = [enc.add_labels(f, {k: [tuple(s) for s in v] for k, v in answers.items()}) for f in feats]
    for name, spans in answers.items():
        if not spans or name == "damage":
            continue
        assert any(
            f.start_labels[i] > 0
            for f in labelled for i, fld in enumerate(f.fields) if fld.name == name
        ), name


def test_many_fields_are_grouped(tokenizer):
    enc = Encoder(tokenizer, max_len=128)
    fields = parse_fields({f"field_{i}": {"type": "text", "instructions": "Some longer description of the value"}
                           for i in range(40)})
    feats = enc.encode("A short document.", fields)
    assert len({f.group for f in feats}) > 1
    assert sorted(f.name for feat in feats for f in feat.fields) == sorted(f.name for f in fields)
    assert all(len(f.input_ids) <= 128 for f in feats)
