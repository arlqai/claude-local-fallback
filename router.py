"""
Anthropic-API router for Claude Code, with automatic + visible local fallback.

  Claude Code ──> this router (:4000)
                    ├─ model matching LOCAL_PREFIX ──> LiteLLM (:4001) ──> mlx-lm (:8080)
                    └─ anything else               ──> verbatim passthrough ──> api.anthropic.com
                                                        └─ on overload/outage ──> retried locally

Requests bound for Anthropic are forwarded byte-for-byte with headers untouched,
so the caller's own credential and beta headers travel unchanged. Requests bound
for the local model have the Authorization header stripped, so the upstream
credential is never handed to LiteLLM or written to its logs.

Fallback fires only for transport failures and overload/rate-limit statuses.
Auth, permission, and malformed-request errors are returned to the caller as-is,
because silently answering them with a weaker model would hide a real bug.

When a fallback happens it is made visible three ways: a banner prepended to the
assistant's first text block, a rate-limited desktop notification, and a FALLBACK
line in this process's log.
"""

import asyncio
import json
import os
import re
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

UPSTREAM = os.environ.get("ROUTER_UPSTREAM", "https://api.anthropic.com").rstrip("/")
LOCAL_BACKEND = os.environ.get("ROUTER_LOCAL_BACKEND", "http://127.0.0.1:4001").rstrip("/")
LOCAL_PREFIX = os.environ.get("ROUTER_LOCAL_PREFIX", "local")


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "")


LOCAL_STAMP = os.environ.get(
    "ROUTER_LOCAL_STAMP",
    os.path.expanduser("~/.local/share/claude-local-router/.last-local-use"),
)
# Touched whenever Anthropic misbehaves. The idle watchdog reads this and keeps
# the local model resident while trouble is recent, so a fallback during an
# outage does not pay the ~90s cold-load cost at the worst possible moment.
TROUBLE_STAMP = os.environ.get(
    "ROUTER_TROUBLE_STAMP",
    os.path.expanduser("~/.local/share/claude-local-router/.last-upstream-trouble"),
)

FALLBACK_ENABLED = _flag("ROUTER_FALLBACK")
FALLBACK_MODEL = os.environ.get("ROUTER_FALLBACK_MODEL", "local-qwen")
# Hard output cap for the local model. LiteLLM's `max_tokens` and mlx-lm's
# --max-tokens are only defaults: the request's own value wins, and Claude Code
# asks for far more. mlx-lm serves one request at a time, so a single runaway
# generation would otherwise hold the slot for half an hour while every queued
# request times out.
LOCAL_MAX_TOKENS = int(os.environ.get("ROUTER_LOCAL_MAX_TOKENS", "8192"))
# Local models the cap does not apply to: they are not served by the single
# mlx-lm slot (llm-d runs on the cluster with concurrent requests). Capping them
# only hurts: a long answer stops at the cap with stop_reason max_tokens, and
# Claude Code fails the turn as "exceeded the <requested> output token maximum".
UNCAPPED_MODELS = set(os.environ.get("ROUTER_UNCAPPED_MODELS", "local-llmd").split(","))

# mlx-lm reads a prompt at a few hundred tokens/s, slowing as it grows (39k
# tokens in 2 min, 63k in 5). Claude Code waits 4 min for the first byte, then
# 10 more, then fails, while mlx-lm keeps the one slot busy for ~40 min on a
# 258k-token history. ~48k is what mlx-lm reads inside that first 4 min; larger
# prompts go back to Claude Code at once as "prompt is too long", the error it
# answers by compacting, instead of tying up the slot. Estimate: 4 bytes/token.
MLX_MAX_PROMPT_TOKENS = int(os.environ.get("ROUTER_MLX_MAX_PROMPT_TOKENS", "49152"))

# Claude Code marks a 1M-context model by suffixing its name with `[1m]`, and
# sends that name as is from some paths (a skill's `model: local-llmd[1m]`).
# LiteLLM knows no `local-llmd[1m]`, so its `*` catch-all sent it to mlx-lm.
# The suffix is client-side only: strip it from local model names.
MODEL_SUFFIX_RE = re.compile(r"\[[^\]]*\]$")

# Transient statuses worth retrying upstream before giving up. Anthropic often
# returns 529 with `retry-after: 0`, meaning an immediate retry will likely
# succeed -- which is what Claude Code does on its own. Falling back on the
# first blip denies it that retry and silently downgrades the model.
RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}
# Be at least as patient as Claude Code's own retry, or the router "fails" on
# blips the unproxied client would have ridden out. Exponential backoff from
# RETRY_BASE_DELAY gives roughly 0.5+1+2+4+8 = ~15s of grace before falling back.
UPSTREAM_RETRIES = int(os.environ.get("ROUTER_UPSTREAM_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.environ.get("ROUTER_RETRY_BASE_DELAY", "0.5"))
RETRY_MAX_DELAY = float(os.environ.get("ROUTER_RETRY_MAX_DELAY", "8.0"))

# Statuses that mean "Anthropic is unavailable" once retries are exhausted.
# 429 is deliberately absent: a rate limit is not an outage. Passing it through
# lets Claude Code apply its own longer backoff, and means genuine quota
# exhaustion surfaces honestly instead of turning into a silent local answer.
FALLBACK_STATUSES = {500, 502, 503, 504, 529}
FALLBACK_ON_RATE_LIMIT = _flag("ROUTER_FALLBACK_ON_429", "0")
if FALLBACK_ON_RATE_LIMIT:
    FALLBACK_STATUSES = FALLBACK_STATUSES | {408, 429}

# Visibility controls.
FALLBACK_BANNER = _flag("ROUTER_FALLBACK_BANNER")
FALLBACK_NOTIFY = _flag("ROUTER_FALLBACK_NOTIFY")
NOTIFY_COOLDOWN = float(os.environ.get("ROUTER_NOTIFY_COOLDOWN", "60"))
BANNER = os.environ.get(
    "ROUTER_FALLBACK_BANNER_TEXT",
    "⚠️ **Local fallback active** — Anthropic was unavailable, so this reply came "
    f"from the local model ({FALLBACK_MODEL}), not Claude.",
)

# A Metal OOM kills mlx-lm's generation thread but not the process, and every
# completion then fails with this marker. LiteLLM relays it as a 404, which
# Claude Code reads as "model not found" and never retries. The router turns it
# into a retryable 529 and restarts mlx-lm at once instead of waiting for the
# 5-minute watchdog.
DEAD_GENERATION_MARKER = b"generation thread died"
MLX_LABEL = os.environ.get("ROUTER_MLX_LABEL", "ai.arlq.mlx-lm")
MLX_RESTART_COOLDOWN = float(os.environ.get("ROUTER_MLX_RESTART_COOLDOWN", "120"))
DEAD_GENERATION_RETRY_AFTER = "30"

# Generous read timeout: a cold mlx-lm load can take a while, and long agentic
# turns on either backend can legitimately exceed a default timeout.
TIMEOUT = httpx.Timeout(connect=10.0, read=1800.0, write=60.0, pool=10.0)

DROP_REQUEST_HEADERS = {
    "host", "content-length", "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade",
    "accept-encoding",
}
DROP_RESPONSE_HEADERS = {
    "content-length", "content-encoding", "connection", "keep-alive",
    "transfer-encoding", "trailer", "upgrade",
}

app = FastAPI()
client = httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False)

_last_notify = 0.0
_last_mlx_restart = float("-inf")


def _log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp} {msg}", flush=True)


async def _notify(reason: str) -> None:
    """Desktop notification, rate-limited so a multi-call turn cannot spam."""
    global _last_notify
    if not FALLBACK_NOTIFY:
        return
    now = time.monotonic()
    if now - _last_notify < NOTIFY_COOLDOWN:
        return
    _last_notify = now
    script = (
        f'display notification "Anthropic unavailable ({reason}). Answering with '
        f'{FALLBACK_MODEL}." with title "Claude Code: local fallback"'
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "/usr/bin/osascript", "-e", script,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        # Reap the child so it does not linger as a zombie.
        asyncio.create_task(proc.wait())
    except OSError:
        pass


async def _restart_mlx() -> None:
    """Kickstart mlx-lm, at most once per cooldown.

    Every request queued behind the dead thread fails the same way, so without
    the cooldown one death would restart the server once per request, killing
    the cold load the previous restart started.
    """
    global _last_mlx_restart
    now = time.monotonic()
    if now - _last_mlx_restart < MLX_RESTART_COOLDOWN:
        return
    _last_mlx_restart = now
    _log(f"MLX DEAD: generation thread died -> restarting {MLX_LABEL}")
    try:
        proc = await asyncio.create_subprocess_exec(
            "/bin/launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{MLX_LABEL}",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        asyncio.create_task(proc.wait())
    except OSError:
        pass


def _overloaded(message: str, *, target: str, fell_back: bool) -> Response:
    """An Anthropic-shaped 529, which Claude Code retries with backoff."""
    payload = {
        "type": "error",
        "error": {"type": "overloaded_error", "message": message},
    }
    headers = {"x-router-target": target, "retry-after": DEAD_GENERATION_RETRY_AFTER}
    if fell_back:
        headers["x-router-fallback"] = "true"
    return Response(
        content=json.dumps(payload).encode(),
        status_code=529,
        headers=headers,
        media_type="application/json",
    )


# A local backend rejects an over-long prompt with LiteLLM's
# ContextWindowExceededError (vLLM: "maximum context length is M tokens ...
# contains at least N input tokens"). Claude Code compacts the conversation and
# retries only on Anthropic's wording, "prompt is too long: N tokens > M
# maximum"; anything else fails the turn. The router rewrites one into the other.
CONTEXT_MAX_RE = re.compile(rb"maximum context length is (\d+) tokens")
CONTEXT_INPUT_RE = re.compile(rb"(?:contains at least|requested) (\d+)")


def _prompt_too_long_response(tokens: int, maximum: int, *, target: str, fell_back: bool):
    """Anthropic's prompt-too-long 400, which Claude Code answers by compacting."""
    body = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": f"prompt is too long: {tokens} tokens > {maximum} maximum",
        },
    }
    headers = {"x-router-target": target}
    if fell_back:
        headers["x-router-fallback"] = "true"
    return Response(
        content=json.dumps(body).encode(),
        status_code=400,
        headers=headers,
        media_type="application/json",
    )


def _prompt_too_long(payload: bytes, *, target: str, fell_back: bool):
    """Anthropic's prompt-too-long 400 for a backend context overflow, else None."""
    limit = CONTEXT_MAX_RE.search(payload)
    if not limit:
        return None
    maximum = int(limit.group(1))
    found = CONTEXT_INPUT_RE.search(payload)
    tokens = max(int(found.group(1)) if found else 0, maximum + 1)
    return _prompt_too_long_response(tokens, maximum, target=target, fell_back=fell_back)


def _too_long_for_mlx(path: str, body: bytes, model: str, *, fell_back: bool):
    """Refuse up front a prompt mlx-lm cannot read before Claude Code gives up."""
    if model in UNCAPPED_MODELS or not path.rstrip("/").endswith("v1/messages"):
        return None
    estimate = len(body) // 4
    if estimate <= MLX_MAX_PROMPT_TOKENS:
        return None
    _log(f"MLX TOO LONG: model={model} ~{estimate} tokens > {MLX_MAX_PROMPT_TOKENS}")
    return _prompt_too_long_response(
        estimate, MLX_MAX_PROMPT_TOKENS, target="local", fell_back=fell_back
    )


def _model_of(body: bytes) -> str:
    try:
        model = json.loads(body).get("model", "")
    except (ValueError, AttributeError):
        return ""
    return model if isinstance(model, str) else ""


def _is_streaming(body: bytes) -> bool:
    try:
        return bool(json.loads(body).get("stream"))
    except (ValueError, AttributeError):
        return False


def _rewrite_model(body: bytes, model: str) -> bytes:
    try:
        payload = json.loads(body)
    except ValueError:
        return body
    payload["model"] = model
    return json.dumps(payload).encode()


def _cap_max_tokens(body: bytes) -> bytes:
    try:
        payload = json.loads(body)
        requested = int(payload["max_tokens"])
    except (ValueError, KeyError, TypeError):
        return body
    if requested <= LOCAL_MAX_TOKENS or payload.get("model") in UNCAPPED_MODELS:
        return body
    payload["max_tokens"] = LOCAL_MAX_TOKENS
    return json.dumps(payload).encode()


# Efforts that survive the trip to a local backend. llm-d's vLLM engine accepts
# only low/medium/xhigh and 400s anything else; mlx-lm ignores the key outright.
# xhigh is absent on purpose: LiteLLM's Anthropic shim rewrites it to `high`
# before forwarding, which is exactly the value llm-d rejects.
LOCAL_EFFORTS = {"low", "medium"}


def _anthropic_effort(payload: dict):
    """The `reasoning_effort` LiteLLM derives from an Anthropic `thinking` block.

    Mirrors litellm's anthropic adapter: adaptive thinking takes
    `output_config.effort` (default medium); enabled thinking buckets
    `budget_tokens` with litellm's DEFAULT_REASONING_EFFORT_*_THINKING_BUDGET
    thresholds (1024 low, 2048 medium, 4096 high).
    """
    thinking = payload.get("thinking")
    if not isinstance(thinking, dict):
        return None
    kind = thinking.get("type")
    if kind == "adaptive":
        config = payload.get("output_config")
        return (config.get("effort") if isinstance(config, dict) else None) or "medium"
    if kind == "enabled":
        budget = thinking.get("budget_tokens") or 0
        if budget >= 4096:
            return "high"
        if budget >= 2048:
            return "medium"
        return "low" if budget >= 1024 else "minimal"
    return None


def _drop_unsupported_effort(body: bytes) -> bytes:
    """Drop the reasoning effort unless a local backend accepts its value.

    Claude Code asks for `high` (or above) on max-effort turns. On /v1/messages
    it says so with Anthropic's `thinking` (+ `output_config.effort`), which
    LiteLLM turns into `reasoning_effort` for the backend; OpenAI-shaped bodies
    carry `reasoning_effort` directly. llm-d 400s anything outside its list,
    and Claude Code does not retry a 400: the turn dies with "Unexpected
    reasoning effort". Dropping instead lands on the backend's default effort,
    which is xhigh on llm-d: one notch above the rejected `high`, and nothing
    is lost.
    """
    try:
        payload = json.loads(body)
    except ValueError:
        return body
    if not isinstance(payload, dict):
        return body
    changed = False
    effort = payload.get("reasoning_effort")
    if effort is not None and effort not in LOCAL_EFFORTS:
        del payload["reasoning_effort"]
        changed = True
    effort = _anthropic_effort(payload)
    if effort is not None and effort not in LOCAL_EFFORTS:
        del payload["thinking"]
        config = payload.get("output_config")
        if isinstance(config, dict):
            config.pop("effort", None)
            if not config:
                del payload["output_config"]
        changed = True
    return json.dumps(payload).encode() if changed else body


def _forward_headers(request: Request, *, local: bool) -> dict:
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in DROP_REQUEST_HEADERS
    }
    if local:
        # Never expose the subscription credential to the local backend.
        headers.pop("authorization", None)
        headers.pop("x-api-key", None)
        headers["authorization"] = "Bearer local"
    return headers


def _banner_into_json(payload: bytes) -> bytes:
    """Prepend the banner to the first text block of a non-streaming reply."""
    try:
        data = json.loads(payload)
        for block in data.get("content", []):
            if block.get("type") == "text":
                block["text"] = f"{BANNER}\n\n{block.get('text', '')}"
                return json.dumps(data).encode()
        # No text block (e.g. tool_use only): add one so the notice still shows.
        if isinstance(data.get("content"), list):
            data["content"].insert(0, {"type": "text", "text": BANNER})
            return json.dumps(data).encode()
    except (ValueError, AttributeError, TypeError):
        pass
    return payload


def _delta_event(index: int, text: str) -> bytes:
    """One Anthropic text_delta SSE event for an already-open block."""
    obj = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }
    return f"event: content_block_delta\ndata: {json.dumps(obj)}\n\n".encode()


async def _banner_injecting_stream(upstream):
    """Pass the SSE stream through, prepending the banner to the model's first
    text block.

    The banner rides as an extra text_delta on the block the model already
    opened, reusing its index. Emitting a separate block would collide with the
    model's own index 0 and require renumbering every later event. Injection
    happens after the terminating blank line of the content_block_start event so
    SSE framing stays intact, and any parse failure degrades to passthrough.
    """
    injected = False
    pending_index = None
    buf = b""
    try:
        async for chunk in upstream.aiter_bytes():
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line + b"\n"

                if pending_index is not None and line.strip() == b"":
                    yield _delta_event(pending_index, BANNER + "\n\n")
                    pending_index, injected = None, True
                    continue
                if injected or pending_index is not None:
                    continue
                if not line.startswith(b"data:"):
                    continue
                try:
                    obj = json.loads(line[len(b"data:"):].strip())
                except ValueError:
                    continue
                if obj.get("type") != "content_block_start":
                    continue
                if (obj.get("content_block") or {}).get("type") != "text":
                    continue
                pending_index = obj.get("index", 0)
        if buf:
            yield buf
    finally:
        await upstream.aclose()


def _touch_trouble_stamp() -> None:
    """Record that Anthropic just failed, so the watchdog stays warm."""
    try:
        with open(TROUBLE_STAMP, "w") as fh:
            fh.write(str(time.time()))
    except OSError:
        pass


def _touch_local_stamp() -> None:
    """Record the moment of the last local-model request.

    The idle watchdog reads this file's mtime. Deriving idleness from a log
    file's mtime would be unreliable: mlx-lm does not log every request, and
    other writers touch those logs.
    """
    try:
        with open(LOCAL_STAMP, "w") as fh:
            fh.write(str(time.time()))
    except OSError:
        pass


LITELLM_ONLY_FIELDS = {"provider_specific_fields"}


def _strip_litellm_fields(body: bytes) -> bytes:
    """Remove LiteLLM-only keys from message content blocks bound for Anthropic.

    LiteLLM's non-streaming Anthropic responses carry
    `"provider_specific_fields": null` on tool_use blocks. Claude Code keeps
    those blocks in its history, so once a session that used a local model
    talks to Anthropic again, every request 400s with
    `messages.N.content.M.tool_use.provider_specific_fields: Extra inputs are
    not permitted`. Stripping here also repairs sessions already carrying them.
    """
    if b"provider_specific_fields" not in body:
        return body
    try:
        payload = json.loads(body)
    except ValueError:
        return body
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        return body
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for key in LITELLM_ONLY_FIELDS & block.keys():
                        del block[key]
    return json.dumps(payload).encode()


async def _send(request: Request, path: str, body: bytes, *, local: bool):
    if local:
        _touch_local_stamp()
        if body:
            body = _cap_max_tokens(body)
            body = _drop_unsupported_effort(body)
    elif body:
        body = _strip_litellm_fields(body)
    base = LOCAL_BACKEND if local else UPSTREAM
    url = f"{base}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    upstream_request = client.build_request(
        request.method,
        url,
        headers=_forward_headers(request, local=local),
        content=body or None,
    )
    return await client.send(upstream_request, stream=True)


def _response_headers(upstream, *, target: str, fell_back: bool) -> dict:
    headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in DROP_RESPONSE_HEADERS
    }
    headers["x-router-target"] = target
    if fell_back:
        headers["x-router-fallback"] = "true"
    return headers


def _stream_response(upstream, *, target: str, fell_back: bool, banner: bool):
    async def passthrough():
        try:
            # Decoded bytes, not aiter_raw(): content-encoding is dropped from
            # the response headers, so compressed SSE would reach the client
            # undecodable and Claude Code would fall back to non-streaming.
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    gen = _banner_injecting_stream(upstream) if banner else passthrough()
    return StreamingResponse(
        gen,
        status_code=upstream.status_code,
        headers=_response_headers(upstream, target=target, fell_back=fell_back),
        media_type=upstream.headers.get("content-type"),
    )


async def _finish(upstream, *, target: str, fell_back: bool):
    """Build the client response, applying the banner when falling back."""
    media_type = upstream.headers.get("content-type") or ""
    banner = fell_back and FALLBACK_BANNER

    # Errors arrive before any stream starts, so the body is small and safe to read.
    if target == "local" and upstream.status_code == 404:
        payload = await upstream.aread()
        await upstream.aclose()
        if DEAD_GENERATION_MARKER in payload:
            await _restart_mlx()
            return _overloaded(
                "Local model crashed (generation thread died); restarting it.",
                target=target, fell_back=fell_back,
            )
        return Response(
            content=payload,
            status_code=404,
            headers=_response_headers(upstream, target=target, fell_back=fell_back),
            media_type=media_type or None,
        )

    if target == "local" and upstream.status_code == 400:
        payload = await upstream.aread()
        await upstream.aclose()
        rewritten = _prompt_too_long(payload, target=target, fell_back=fell_back)
        if rewritten is not None:
            return rewritten
        return Response(
            content=payload,
            status_code=400,
            headers=_response_headers(upstream, target=target, fell_back=fell_back),
            media_type=media_type or None,
        )

    if "text/event-stream" in media_type:
        return _stream_response(
            upstream, target=target, fell_back=fell_back, banner=banner
        )

    payload = await upstream.aread()
    await upstream.aclose()
    if banner and upstream.status_code == 200:
        payload = _banner_into_json(payload)
    return Response(
        content=payload,
        status_code=upstream.status_code,
        headers=_response_headers(upstream, target=target, fell_back=fell_back),
        media_type=media_type or None,
    )


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy(path: str, request: Request):
    body = await request.body()
    model = _model_of(body) if body else ""
    local = bool(model) and model.startswith(LOCAL_PREFIX)
    if local and MODEL_SUFFIX_RE.search(model):
        model = MODEL_SUFFIX_RE.sub("", model)
        body = _rewrite_model(body, model)
    if local:
        _log(f"LOCAL model={model} ~{len(body) // 4} tokens path=/{path}")
        refused = _too_long_for_mlx(path, body, model, fell_back=False)
        if refused is not None:
            return refused

    # Only inference calls are worth retrying elsewhere.
    eligible = (
        FALLBACK_ENABLED
        and not local
        and request.method == "POST"
        and path.rstrip("/").endswith("v1/messages")
    )

    async def fall_back(reason: str):
        _log(f"FALLBACK: {reason} -> {FALLBACK_MODEL} (model={model or '?'})")
        await _notify(reason)
        refused = _too_long_for_mlx(path, body, FALLBACK_MODEL, fell_back=True)
        if refused is not None:
            return refused
        upstream = await _send(
            request, path, _rewrite_model(body, FALLBACK_MODEL), local=True
        )
        return await _finish(upstream, target="local", fell_back=True)

    attempts = 1 + (UPSTREAM_RETRIES if eligible else 0)
    upstream = None
    for attempt in range(1, attempts + 1):
        try:
            upstream = await _send(request, path, body, local=local)
        except httpx.RequestError as exc:
            # Anthropic unreachable (offline, DNS failure, refused connection).
            if not eligible:
                raise
            _touch_trouble_stamp()
            if attempt == attempts:
                return await fall_back(type(exc).__name__)
            _log(f"RETRY {attempt}/{attempts - 1} after {type(exc).__name__}")
            await asyncio.sleep(
                min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            )
            continue

        if upstream.status_code not in RETRY_STATUSES or attempt == attempts:
            if upstream.status_code in RETRY_STATUSES:
                _touch_trouble_stamp()
            break

        _touch_trouble_stamp()

        # Exponential backoff, but never shorter than a server-sent retry-after.
        backoff = RETRY_BASE_DELAY * (2 ** (attempt - 1))
        try:
            backoff = max(backoff, float(upstream.headers.get("retry-after", "")))
        except ValueError:
            pass
        delay = min(max(backoff, 0.1), RETRY_MAX_DELAY)
        _log(
            f"RETRY {attempt}/{attempts - 1} after HTTP {upstream.status_code} "
            f"in {delay:.2f}s (retry-after="
            f"{upstream.headers.get('retry-after', '-')})"
        )
        await upstream.aclose()
        await asyncio.sleep(delay)

    if not local and upstream.status_code >= 400:
        _log(
            f"UPSTREAM {upstream.status_code} path=/{path} model={model or '?'} "
            f"retry-after={upstream.headers.get('retry-after', '-')} "
            f"req-id={upstream.headers.get('request-id', '-')} "
            f"ua={request.headers.get('user-agent', '-')[:60]!r} "
            f"stream={_is_streaming(body)}"
        )

    if eligible and upstream.status_code in FALLBACK_STATUSES:
        status = upstream.status_code
        await upstream.aclose()
        return await fall_back(f"HTTP {status}")

    return await _finish(
        upstream, target="local" if local else "anthropic", fell_back=False
    )
