# claude-local-fallback

Run a local MLX model as a fallback for Claude Code on Apple Silicon, and switch
to it on demand — without touching your Claude subscription or billing.

```
Claude Code ──> router (:4000)
                  ├─ model local-*  ──> LiteLLM (:4001) ──> mlx-lm (:8080)
                  └─ anything else  ──> verbatim passthrough ──> api.anthropic.com
                                          └─ on outage, after retries ──> local
```

Model: `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` (~17GB on disk, ~14GB
resident while loaded).

## Why this exists

`brew services start mlx-lm` cannot be configured. The formula's `service` block
is just `run opt_bin/"mlx_lm.server"` with no arguments, and Homebrew regenerates
`homebrew.mxcl.mlx-lm.plist` from the formula on **every** start — so any
hand-edit (a `--model` flag, a port, sampling defaults) is silently discarded.

The fix is to stop using `brew services` and own the launch agent under a
different label. Homebrew only ever writes `homebrew.mxcl.*`, so it cannot touch
`ai.arlq.*`.

A second problem: Claude Code speaks the Anthropic Messages API, mlx-lm speaks
OpenAI. LiteLLM sits between them to translate, including SSE streaming and
tool calls.

## Install

Requires macOS, Apple Silicon, `brew install mlx-lm`, and Claude Code.

```sh
./install.sh
```

Then add to `~/.claude/settings.json` (not done for you — that file is yours):

```json
{
  "env": { "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000" },
  "model": "claude-opus-5[1m]"
}
```

## Usage

| Command | Goes to |
| --- | --- |
| `claude` | Your Claude subscription, unchanged |
| `claude --model local-qwen` | Local Qwen3-Coder |

Every response carries `x-router-target: anthropic\|local`, and fallback
responses also carry `x-router-fallback: true`.

## Fallback behaviour

Fallback is a last resort, not a hair trigger:

- **Retries first.** 5 attempts with exponential backoff (0.5/1/2/4/8s, ~15s of
  grace), honouring `retry-after`. Anthropic returns 529 with `retry-after: 0`
  fairly often; Claude Code rides those out on its own, so the router must be at
  least as patient or it "fails" where unproxied Claude Code succeeds.
- **Falls back on** `500, 502, 503, 504, 529` and transport failures
  (offline, DNS, refused connection).
- **Does not fall back on** `401`, `403`, `400`, or `429`. A rate limit is not an
  outage: passing it through lets Claude Code apply its own longer backoff, and
  keeps genuine quota exhaustion visible instead of turning it into a silent
  local answer. Set `ROUTER_FALLBACK_ON_429=1` to change that.

When it does fall back, it is made visible three ways: a banner prepended to the
assistant's first text block, a rate-limited desktop notification, and a
`FALLBACK` line in `~/Library/Logs/claude-router.log`.

## Memory

The agent passes no `--model`, so mlx-lm loads lazily: **0.09GB idle**, ~14GB
only once used. `mlx_lm.server` has no idle-unload option, so
`mlx-idle-unload.sh` (every 5 min) restarts the service after 15 min idle to
release it.

It stays warm while Anthropic is misbehaving. A cold load costs ~90s, and paying
that mid-outage is the worst possible moment — so the router stamps
`.last-upstream-trouble` on any upstream failure and the watchdog skips
unloading for 30 min afterwards. It also refuses to restart while a connection
to :8080 is still established, so long generations are not killed.

## Tunables

Set in the relevant plist's `EnvironmentVariables`, then reload (see below).

| Variable | Default | Effect |
| --- | --- | --- |
| `ROUTER_FALLBACK` | `1` | Master switch for fallback |
| `ROUTER_UPSTREAM_RETRIES` | `5` | Upstream attempts before falling back |
| `ROUTER_RETRY_BASE_DELAY` | `0.5` | Backoff base, seconds |
| `ROUTER_RETRY_MAX_DELAY` | `8.0` | Backoff ceiling, seconds |
| `ROUTER_FALLBACK_ON_429` | `0` | Treat rate limits as an outage |
| `ROUTER_FALLBACK_BANNER` | `1` | In-conversation banner |
| `ROUTER_FALLBACK_NOTIFY` | `1` | Desktop notification |
| `ROUTER_NOTIFY_COOLDOWN` | `60` | Notification rate limit, seconds |
| `MLX_IDLE_SECONDS` | `900` | Idle time before releasing memory |
| `MLX_KEEP_WARM_SECONDS` | `1800` | Stay warm after upstream trouble |

## Gotchas found the hard way

- **`launchctl kickstart -k` does not reload the plist.** It restarts the process
  from the *cached* job definition, so edits appear to do nothing. Always:
  ```sh
  launchctl bootout   gui/$(id -u)/ai.arlq.mlx-lm
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.arlq.mlx-lm.plist
  ```
- **`settings.json` `env` overrides shell environment variables.** Once
  `ANTHROPIC_BASE_URL` is in there, `ANTHROPIC_BASE_URL=... claude` is ignored.
  Use `claude --settings '{"env":{...}}'` to override for one run.
- **The 1M context window hinges on the literal `[1m]` in the model name**
  (`/\[1m\]/i.test(model)`). `--model opus` resolves to `claude-opus-5` — 200k,
  a 5× smaller window, and constant auto-compaction in long conversations. Use
  `claude-opus-5[1m]`. Pinning `model` in settings.json fixes the default but an
  explicit `--model` still overrides it.
- **LiteLLM needs the `hosted_vllm/` prefix, not `openai/`.** With `openai/`,
  LiteLLM 1.95 routes to `/v1/responses` (the OpenAI Responses API), which
  mlx-lm does not implement — every request 404s.
- **FastAPI must be held below 0.141.** LiteLLM 1.95 declares
  `fastapi>=0.136.3,<1.0` but imports `get_flat_dependant`, which FastAPI
  removed in 0.141 — the declared range is missing an upper bound, so a plain
  `pip install litellm[proxy]` resolves to a version that dies at startup with
  `ImportError`. `fastapi==0.140.0` is the newest that satisfies both, and
  `install.sh` asserts the import rather than trusting metadata.
- **Anthropic's `--fallback-model` flag only works with `--print`**, which is why
  fallback lives in the router instead.

## Caveats

- **All Claude Code traffic flows through `:4000`.** If the router dies, `claude`
  breaks. `KeepAlive` guards it; removing the `env` block reverts instantly.
- **Fallback is silent apart from the banner.** A 4-bit 30B model is noticeably
  weaker than Opus and has a much smaller context window. If you would rather
  never be downgraded without asking, set `ROUTER_FALLBACK=0` and switch
  deliberately with `--model local-qwen`.
- **Cold fallback takes ~90s** if the watchdog has released the model.
- **The router forwards your Claude Code credential to Anthropic verbatim.**
  Requests are byte-for-byte identical to unproxied ones and no credential is
  ever sent to LiteLLM (the router strips `Authorization` on the local path).
  Whether proxying subscription traffic this way fits Anthropic's terms is worth
  deciding for yourself; it is not something this README can settle.

## Layout

```
router.py               Routing, retries, fallback, banner injection
litellm.config.yaml     Anthropic <-> OpenAI translation for mlx-lm
mlx-idle-unload.sh      Idle watchdog
launchd/*.plist         Agent templates (__HOME__ rendered by install.sh)
install.sh              Deploy + load
requirements.txt        Pinned Python deps
```

Runtime lives in `~/.local/share/claude-local-router/`; logs in
`~/Library/Logs/{mlx-lm,litellm-local,claude-router,mlx-lm-watchdog}.log`.
