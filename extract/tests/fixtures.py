"""Offline fixtures: templated documents with known answers, and tiny backbones.

Nothing here downloads anything. The backbones are randomly initialised, so
the end-to-end test checks that the pipeline *can learn* (labels, heads,
decoding and saving are wired correctly), not how good a real model is.
"""

from __future__ import annotations

import os
import random

FIRST = ["Dana", "Lucas", "Inès", "Marco", "Aiko", "Priya", "Tom", "Léa", "Omar", "Sven"]
DAMAGE = ["cracked screen", "dented corner", "missing charger", "scratched lid", "broken hinge", "torn box"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
VENDORS = ["Acme Corp", "Globex", "Initech", "Umbrella Ltd", "Stark Industries", "Hooli"]


def _date(rng):
    y, m, d = rng.choice([2025, 2026]), rng.randint(1, 12), rng.randint(1, 28)
    style = rng.randint(0, 2)
    if style == 0:
        text = f"{MONTHS[m - 1]} {d}, {y}"
    elif style == 1:
        text = f"{d} {MONTHS[m - 1]} {y}"
    else:
        text = f"{y}-{m:02d}-{d:02d}"
    return text, f"{y}-{m:02d}-{d:02d}"


def _put(parts, text):
    """Append text to the document under construction and return its span."""
    start = sum(len(p) for p in parts)
    parts.append(text)
    return [start, start + len(text)]


def support_email(rng):
    parts, ans = [], {}
    order = f"{rng.choice('ABCDEFGH')}-{rng.randint(10000, 99999)}"
    date_text, _ = _date(rng)
    amount = f"${rng.randint(20, 2000)}.{rng.randint(0, 99):02d}"
    damage = rng.sample(DAMAGE, rng.randint(1, 3))
    name = rng.choice(FIRST)
    _put(parts, rng.choice(["Hi, order ", "Hello, my order ", "Dear support, order "]))
    ans["order_id"] = [_put(parts, order)]
    _put(parts, rng.choice([" arrived on ", " was delivered on ", " came on "]))
    ans["delivered_on"] = [_put(parts, date_text)]
    _put(parts, " with ")
    ans["damage"] = []
    for i, d in enumerate(damage):
        if i:
            _put(parts, " and " if i == len(damage) - 1 else ", ")
        ans["damage"].append(_put(parts, d))
    _put(parts, ". I paid ")
    ans["amount_paid"] = [_put(parts, amount)]
    _put(parts, rng.choice([" and would like a refund. Thanks, ", ". Please fix this. Regards, "]))
    ans["customer_name"] = [_put(parts, name)]
    ans["tracking_number"] = []
    fields = {
        "order_id": {"type": "text", "instructions": "Order number"},
        "delivered_on": {"type": "date", "instructions": "Delivery date"},
        "damage": {"type": "list", "instructions": "Each kind of damage described"},
        "amount_paid": {"type": "number", "decimals": 2, "instructions": "Amount paid"},
        "customer_name": {"type": "text", "instructions": "Customer's first name"},
        "tracking_number": {"type": "text", "instructions": "Shipment tracking number"},
    }
    return "".join(parts), fields, ans


def invoice(rng):
    parts, ans = [], {}
    _put(parts, "INVOICE ")
    ans["invoice_number"] = [_put(parts, f"INV-{rng.randint(1000, 9999)}")]
    _put(parts, "\nFrom: ")
    ans["vendor"] = [_put(parts, rng.choice(VENDORS))]
    _put(parts, "\nIssued: ")
    ans["issue_date"] = [_put(parts, _date(rng)[0])]
    has_due = rng.random() < 0.6
    if has_due:
        _put(parts, "\nDue: ")
        ans["due_date"] = [_put(parts, _date(rng)[0])]
    else:
        ans["due_date"] = []
    _put(parts, "\nTotal due: ")
    ans["total"] = [_put(parts, f"{rng.randint(100, 90000):,}.{rng.randint(0, 99):02d} EUR")]
    _put(parts, "\nThank you for your business.")
    fields = {
        "invoice_number": {"type": "text", "instructions": "Invoice number"},
        "vendor": {"type": "text", "instructions": "Who issued the invoice"},
        "issue_date": {"type": "date", "instructions": "Date the invoice was issued"},
        "due_date": {"type": "date", "instructions": "Payment due date"},
        "total": {"type": "number", "instructions": "Total amount due"},
    }
    return "".join(parts), fields, ans


def make_records(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        state, fields, answers = (support_email if rng.random() < 0.5 else invoice)(rng)
        out.append({"id": f"fx-{seed}-{i}", "state": state, "fields": fields, "answers": answers})
    return out


def perturb_schemas(records: list[dict], seed: int = 0) -> list[dict]:
    """Same documents, but each schema is reordered and has fields removed."""
    rng = random.Random(seed)
    out = []
    for r in records:
        names = list(r["fields"])
        rng.shuffle(names)
        names = names[: rng.randint(1, len(names))]
        out.append({**r, "fields": {n: r["fields"][n] for n in names},
                    "answers": {n: r["answers"][n] for n in names}})
    return out


def build_backbone(out_dir: str, kind: str = "bert", records=None) -> str:
    """Train a small tokenizer on fixture text and save a random tiny backbone."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    os.makedirs(out_dir, exist_ok=True)
    records = records or make_records(400, seed=99)
    corpus = [r["state"] for r in records] + [
        "Fields: Document: Answers: text number date list of " + " ".join(r["fields"]) for r in records
    ] + [v.get("instructions", "") for r in records for v in r["fields"].values()]

    if kind == "bert":
        from transformers import BertConfig, BertModel

        tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tk.decoder = decoders.WordPiece()
        tk.train_from_iterator(corpus, trainers.WordPieceTrainer(
            vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
        tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                                      cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
        config = BertConfig(vocab_size=tok.vocab_size, hidden_size=128, num_hidden_layers=2,
                            num_attention_heads=4, intermediate_size=256, max_position_embeddings=512)
        model = BertModel(config)
    else:  # a decoder-only backbone with byte-level BPE, like Qwen
        from transformers import LlamaConfig, LlamaModel

        tk = Tokenizer(models.BPE())
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        tk.post_processor = processors.ByteLevel(trim_offsets=False)
        tk.train_from_iterator(corpus, trainers.BpeTrainer(
            vocab_size=2000, special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
        config = LlamaConfig(vocab_size=tok.vocab_size, hidden_size=128, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=4, intermediate_size=256,
                             max_position_embeddings=1024)
        model = LlamaModel(config)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
