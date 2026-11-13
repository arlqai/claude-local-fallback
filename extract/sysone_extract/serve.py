"""HTTP server: POST /v1/extract with a Clairic-style request.

    EXTRACT_MODEL=runs/v0 uvicorn sysone_extract.serve:app --host 127.0.0.1 --port 8090

Request:   {"state": "...", "fields": {"order_id": {"type": "text", "instructions": "Order number"}}}
Response:  {"model": "...", "answers": {"order_id": {"type": "text", "value": "A-88412",
            "guess": "A-88412", "span": [10, 17], "p_null": 3e-06, "confidence": 0.99}},
            "usage": {"passes": 1, "input_tokens": 211, "latency_ms": 38.2}}

Operational behaviour carried over from claude-local-fallback:

* The model loads lazily on the first request, so the service idles at a few
  hundred MB, and unloads itself after EXTRACT_IDLE_SECONDS without traffic.
  Unlike mlx_lm.server we own this process, so no watchdog restart is needed.
* One forward pass at a time (requests queue), so concurrent calls cannot
  stack activations and run the GPU out of memory.
* The state is measured with the real tokenizer, not an estimate, and a
  request too large to finish in reasonable time is refused with a 413.
"""

from __future__ import annotations

import asyncio
import gc
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field as PField

from .inference import Extractor
from .schema import SchemaError, parse_fields

MODEL_DIR = os.environ.get("EXTRACT_MODEL", "")
MODEL_NAME = os.environ.get("EXTRACT_MODEL_NAME", os.path.basename(MODEL_DIR.rstrip("/")) or "extractor")
DEVICE = os.environ.get("EXTRACT_DEVICE", "auto")
IDLE_SECONDS = float(os.environ.get("EXTRACT_IDLE_SECONDS", "900"))
MAX_STATE_TOKENS = int(os.environ.get("EXTRACT_MAX_STATE_TOKENS", "65536"))
MAX_FIELDS = int(os.environ.get("EXTRACT_MAX_FIELDS", "256"))

_extractor: Extractor | None = None
_load_lock = threading.Lock()
_last_use = 0.0


class ExtractRequest(BaseModel):
    model: str | None = None
    state: str = PField(min_length=1)
    fields: dict
    images: list | None = None


def _get() -> Extractor:
    global _extractor
    with _load_lock:
        if _extractor is None:
            if not MODEL_DIR:
                raise HTTPException(503, "EXTRACT_MODEL is not set")
            t0 = time.time()
            _extractor = Extractor.load(MODEL_DIR, device=DEVICE)
            print(f"loaded {MODEL_DIR} on {_extractor.device} in {time.time() - t0:.1f}s", flush=True)
        return _extractor


def _unload() -> None:
    global _extractor
    with _load_lock:
        if _extractor is None:
            return
        device = _extractor.device
        _extractor = None
        gc.collect()
        try:
            import torch

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass
        print("unloaded model after idle period", flush=True)


async def _idle_watch():
    while True:
        await asyncio.sleep(min(60.0, max(5.0, IDLE_SECONDS / 10)))
        if _extractor is not None and IDLE_SECONDS > 0 and time.time() - _last_use > IDLE_SECONDS:
            await run_in_threadpool(_unload)


@asynccontextmanager
async def _lifespan(_app):
    task = asyncio.create_task(_idle_watch())
    yield
    task.cancel()


app = FastAPI(title="sysone-extract", lifespan=_lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "loaded": _extractor is not None,
            "idle_seconds": round(time.time() - _last_use, 1) if _last_use else None}


def _do_extract(req: ExtractRequest) -> dict:
    global _last_use
    _last_use = time.time()
    fields = parse_fields(req.fields)
    if len(fields) > MAX_FIELDS:
        raise SchemaError(f"at most {MAX_FIELDS} fields per request")
    ex = _get()
    n = ex.count_tokens(req.state)
    if n > MAX_STATE_TOKENS:
        raise HTTPException(413, f"state is {n} tokens; the limit is {MAX_STATE_TOKENS}")
    out = ex.extract(req.state, fields)
    _last_use = time.time()
    return {"model": MODEL_NAME, **out}


@app.post("/v1/extract")
async def extract(req: ExtractRequest):
    if req.images:
        raise HTTPException(400, "images are not supported by this model yet")
    try:
        return await run_in_threadpool(_do_extract, req)
    except SchemaError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/unload")
async def unload():
    await run_in_threadpool(_unload)
    return {"loaded": False}
