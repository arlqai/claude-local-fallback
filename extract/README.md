# sysone-extract

Part of claude-local-fallback: once a model is deployed with
`./install.sh --extract-model DIR`, it is served on `:8090` and reachable
through the router as `POST :4000/v1/extract`.

A small, local **System One extraction model**, in the spirit of Cloudflare's
[Clairic](https://blog.cloudflare.com/clairic-extraction-model/): give it a
document and a schema of named, typed fields, and it returns one typed value
per field from a **single forward pass**. It generates no tokens, so every
answer is a valid value of the declared type (or `null`), grounded in a
character span of the input, and comes with calibrated `p_null` and
`confidence` scores.

```bash
curl -s localhost:4000/v1/extract -d '{
  "state": "Hi, order A-88412 arrived on Sept 28, 2026 with a cracked screen and a dented corner. I paid $649.00 ...",
  "fields": {
    "order_id":        {"type": "text",   "instructions": "Order number"},
    "delivered_on":    {"type": "date",   "instructions": "Delivery date"},
    "damage":          {"type": "list",   "instructions": "Each kind of damage described"},
    "amount_paid":     {"type": "number", "decimals": 2, "instructions": "Amount paid"},
    "tracking_number": {"type": "text"}
  }}'
```
```json
{"answers": {
  "order_id":        {"type": "text",   "value": "A-88412",    "span": [10, 17],  "p_null": 0.0,     "confidence": 0.9996},
  "delivered_on":    {"type": "date",   "value": "2026-09-28", "span": [29, 42],  "p_null": 0.00007, "confidence": 0.487},
  "damage":          {"type": "list",   "value": ["cracked screen", "dented corner"], "spans": [[50, 64], [71, 84]]},
  "amount_paid":     {"type": "number", "value": 649.0,        "span": [93, 100], "p_null": 0.0,     "confidence": 0.9989},
  "tracking_number": {"type": "text",   "value": null, "guess": "A-88412",       "p_null": 0.591,   "confidence": 0.656}},
 "usage": {"passes": 1, "input_tokens": 189, "latency_ms": 18.2}}
```

(Real output from the tiny test model trained on toy invoices and emails.
`tracking_number` has no instructions and was never seen in training, which
is why the model is only 59% sure it is absent.)

## Status

**v0, verified on toy data only.** The whole pipeline (labels, model heads,
typed decoding, windowing, calibration, training, HTTP API, router
integration) is tested end to end on randomly initialised tiny backbones and
templated documents: 39 tests, all passing on CPU in under a minute. That
proves the pipeline is wired correctly and learnable. It says nothing yet
about accuracy on real documents: that needs a real backbone and real
training data, which is the next step (see *Train a real model*).

What v0 does: `text`, `number`, `date` and `list` fields, null detection,
calibrated confidence, long documents (sliding windows), unlimited fields
(split into groups of passes).

What it does not do yet, compared with Clairic: computed values (a total
that is not written in the document), `estimate` and `forecast` fields,
`time`/`datetime`/`duration`, and images. See *Roadmap*.

## How it works

```
 [CLS]? Fields:
   [order_id] text: Order number              <- schema first: a decoder reads the
   [amount_paid] number: Amount paid             document already knowing what to look for
 Document:
   Hi, order A-88412 arrived on Sept 28 ...   <- one window of the document
 Answers:
   [order_id] text: Order number =       <q1> <- one query marker per field, after the
   [amount_paid] number: Amount paid =   <q2>    document, so it can attend to all of it
```

Any Hugging Face backbone (encoder or decoder) reads the sequence once. For
every field, the hidden state at its query marker `q` scores each document
token `h_t`:

| Head | Computes | Used for |
|---|---|---|
| Pointer | `start_t = <W_s q, h_t>`, `end_t = <W_e q, h_t>`, plus an "absent" logit from `q` prepended to both | `text`, `number`, `date`; softmax over [absent] + tokens gives `p_null` |
| BIO tagger | three projections of `q` score each token Outside / Begin / Inside | `list`: any number of spans |

Typed decoding walks the top candidate spans across all windows and returns
the first one that parses as the field's type (`"$1,249.90"` → `1249.9`,
`"Sept 28, 2026"` → `"2026-09-28"`). Spans are snapped to whole words, since
subword tokenizers can cut `$1249.90` into `$12`. A span holding two numbers
(`"$649 plus $52"`) is rejected as ambiguous, and the decoder moves on to
the next candidate.

Two things learned the hard way during development, both covered by tests:

- **Shuffle and drop fields during training** (`--field-drop`, on by
  default). With fixed schemas, a model learns "the third query is the
  date": 96% exact match on the training schema fell to 33% once the
  schema was reordered.
- **Restate the full field description at the query marker.** With only
  `[name] =` there, the model must match the name back to the schema block,
  which small models learn slowly.

Raw span probabilities understate how often the top span is right, so
`train.py` fits Platt scaling on the dev set and stores it with the model.
On toy data, this took the expected calibration error of `confidence` from
0.42 down to 0.09 on data the calibration never saw. Use
`python -m sysone_extract.calibrate` to refit it on separate data.

## Layout

```
sysone_extract/
  schema.py      field types, request and training-record validation
  normalize.py   typed parsing of spans (numbers, dates), SQuAD normalisation
  encoding.py    sequence layout, windows, field groups, labels
  model.py       backbone + pointer / null / BIO heads, save/load, batching
  inference.py   Extractor: windowed inference, typed decoding, calibration
  train.py       fine-tuning (CUDA / MPS / CPU), per-epoch eval, best checkpoint
  evaluate.py    EM/F1 per type, null accuracy, list P/R/F1, ECE, latency
  calibrate.py   Platt scaling of confidence and p_null
  serve.py       FastAPI server: POST /v1/extract, lazy load, idle unload
  data/qa.py     SQuAD-format QA datasets -> multi-field records
  data/ner.py    token-tagged NER datasets -> list fields
  data/synth.py  your own documents, labelled by a teacher LLM, verified
```

Deployment lives in the parent repo: `../install.sh --extract-model DIR`,
`../launchd/ai.arlq.sysone-extract.plist`, and the `/v1/extract` route in
`../router.py`.

Training records are JSONL, with answers as character spans (an empty list
means the value is absent):

```json
{"state": "...", "fields": {"order_id": {"type": "text", "instructions": "Order number"}},
 "answers": {"order_id": [[10, 17]]}}
```

## Train a real model

Training happens in a development environment, not in the runtime venv
that `install.sh` builds:

```bash
cd extract
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[serve,data,dev]'
python -m pytest            # 39 tests, about a minute on CPU, no downloads

# 1. General-purpose data
python -m sysone_extract.data.qa  --hf rajpurkar/squad_v2 --split train      --out data/squad_train.jsonl
python -m sysone_extract.data.qa  --hf rajpurkar/squad_v2 --split validation --out data/squad_dev.jsonl
python -m sysone_extract.data.ner --hf eriktks/conll2003  --split train      --out data/conll_train.jsonl

# 2. Your own documents, labelled by llm-d through claude-local-fallback's LiteLLM
#    (every answer that is not a verbatim span of the document is discarded)
python -m sysone_extract.data.synth --docs ~/some/documents --out data/synth.jsonl \
    --base-url http://127.0.0.1:4001/v1 --model local-llmd --concurrency 16

# 3. Train (decoder backbones learned the layout much faster in testing)
python -m sysone_extract.train \
    --train data/squad_train.jsonl data/conll_train.jsonl data/synth.jsonl \
    --dev data/squad_dev.jsonl --backbone Qwen/Qwen3-0.6B --out runs/v0 \
    --epochs 2 --batch-size 8 --grad-accum 4 --bf16

# 4. Score it, then deploy it with the rest of the stack
python -m sysone_extract.evaluate --model runs/v0 --data data/squad_dev.jsonl
cd .. && ./install.sh --extract-model extract/runs/v0
```

Notes:

- **Backbone.** `Qwen/Qwen3-0.6B` is the suggested start. Any backbone with
  a fast tokenizer works (`answerdotai/ModernBERT-base` is the obvious
  encoder alternative). Qwen3's tokenizer was not tested here because the
  build environment could not reach the Hub; its tokenizer family (byte-level
  BPE with offsets) is covered by the tests. Check it first with:
  `python -c "from transformers import AutoTokenizer as T; print(T.from_pretrained('Qwen/Qwen3-0.6B').is_fast)"`
- **Hardware.** Fully fine-tuning a 0.6B model with AdamW needs roughly
  10 GB plus activations, so it fits on a 32 GB Apple Silicon Mac, but expect
  hours rather than minutes there. Time one run with `--epochs 0.02` before
  starting a long one. A CUDA GPU (for example on the cluster that serves
  llm-d) is much faster. Inference is light: a few to tens of milliseconds
  per request.
- **The synthetic data is what makes it general.** SQuAD teaches "find the
  answer to a question" and CoNLL teaches lists, but real requests look like
  invoices, emails, logs and tickets with short field names. Label a few
  thousand of your own documents with `synth.py`, and keep some for an
  evaluation set the model never trains on.

## How it runs in the stack

The router forwards `POST :4000/v1/extract` to the extractor on :8090.
Requests never go to Anthropic, the caller's `Authorization` header is not
forwarded, and if the extractor is down the router answers a clean 503.

The extractor runs as its own launchd agent (`ai.arlq.sysone-extract`), in
its own venv under `~/.local/share/claude-local-router/extract/`. It follows
the same operational choices as the rest of the stack: the model loads
lazily, one forward pass runs at a time, and request size is checked with the
real tokenizer. Unlike mlx-lm, idle unloading happens inside the process
(`EXTRACT_IDLE_SECONDS`, 15 minutes by default), so it needs no watchdog.

## Roadmap

1. **A real v1:** Qwen3-0.6B trained on SQuAD 2.0, CoNLL and a few thousand
   verified synthetic records, then compared against an LLM on held-out
   documents for accuracy, latency and run-to-run stability (the
   comparison in Cloudflare's post).
2. **Computed values:** a small arithmetic head over extracted operands (sum,
   difference, date difference), so a total that is not written in the
   document can still be returned.
3. **`estimate` and `forecast`:** a quantile-regression head on `q` giving a
   distribution with p10/median/p90 inside declared bounds.
4. **MLX inference** for faster serving on Apple Silicon, and a
   `/v1/systemone`-compatible endpoint alongside `/v1/extract`.
