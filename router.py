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
        async for chunk in upstream.aiter_raw():
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


async def _send(request: Request, path: str, body: bytes, *, local: bool):
    if local:
        _touch_local_stamp()
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
            async for chunk in upstream.aiter_raw():
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
