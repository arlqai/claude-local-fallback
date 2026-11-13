#!/bin/bash
# Deploy the local-fallback stack: copy code into the runtime directory, build
# the venv, render the launchd plists for this user, and load the agents.
#
# Idempotent: safe to re-run after editing router.py or the plists.
#
#   ./install.sh                        the fallback stack (and the extractor,
#                                       if a model was deployed before)
#   ./install.sh --extract-model DIR    also deploy a trained extractor model
#                                       (see extract/README.md) and serve it on
#                                       :8090, reachable as :4000/v1/extract

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${CLAUDE_FALLBACK_RUNTIME:-$HOME/.local/share/claude-local-router}"
AGENTS="$HOME/Library/LaunchAgents"
UID_NUM="$(id -u)"
LABELS=(ai.arlq.mlx-lm ai.arlq.litellm-local ai.arlq.claude-router ai.arlq.mlx-idle-unload)
EXTRACT_LABEL=ai.arlq.sysone-extract
EXTRACT_DIR="$RUNTIME/extract"

say() { printf '\033[1m==>\033[0m %s\n' "$1"; }
die() { printf '\033[31merror:\033[0m %s\n' "$1" >&2; exit 1; }

EXTRACT_MODEL_SRC=""
RELOAD_ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --reload-agents) RELOAD_ONLY=1 ;;
        --extract-model)
            [ $# -ge 2 ] || die "--extract-model needs a directory"
            EXTRACT_MODEL_SRC="$(cd "$2" && pwd)" || die "no such directory: $2"
            shift ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done

# The extractor is optional: its agent only runs once a model is deployed, so
# the stack installs and works without PyTorch. Decided here, before
# reload_agents, because the detached reload re-runs this script.
if [ -n "$EXTRACT_MODEL_SRC" ] || [ -f "$EXTRACT_DIR/model/extractor.json" ]; then
    LABELS+=("$EXTRACT_LABEL")
fi

# bootout then bootstrap: launchctl caches the job definition, so `kickstart`
# alone silently keeps running the previous plist's arguments.
reload_agents() {
    for L in "${LABELS[@]}"; do
        launchctl bootout "gui/$UID_NUM/$L" 2>/dev/null || true
    done
    # bootout returns before the old process exits: uvicorn drains open client
    # connections first, up to launchd's ~20s exit timeout. Bootstrapping a
    # label that is still draining fails, so retry until the old job is gone.
    for L in "${LABELS[@]}"; do
        loaded=""
        for _ in $(seq 40); do
            if launchctl bootstrap "gui/$UID_NUM" "$AGENTS/$L.plist" 2>/dev/null; then
                loaded=1
                break
            fi
            sleep 1
        done
        if [ -n "$loaded" ]; then
            printf '    loaded %s\n' "$L"
        else
            printf '    FAILED %s\n' "$L"
        fi
    done
    echo "reload-done"
}
if [ -n "$RELOAD_ONLY" ]; then
    reload_agents
    exit 0
fi

# --- prerequisites -----------------------------------------------------------
say "Checking prerequisites"
[ "$(uname -s)" = "Darwin" ] || die "macOS only (launchd + mlx)"
command -v python3 >/dev/null || die "python3 not found"
[ -x /opt/homebrew/opt/mlx-lm/bin/mlx_lm.server ] \
    || die "mlx-lm not found; install with: brew install mlx-lm"

# brew services would contend for port 8080, and it regenerates its own plist
# from the formula on every start (discarding any arguments), which is the whole
# reason this project exists.
if brew services list 2>/dev/null | grep -qE '^mlx-lm[[:space:]]+(started|scheduled)'; then
    say "Stopping brew's mlx-lm service (it would contend for port 8080)"
    brew services stop mlx-lm
fi

# --- runtime directory -------------------------------------------------------
say "Installing code into $RUNTIME"
mkdir -p "$RUNTIME"
cp "$REPO/router.py" "$REPO/litellm.config.yaml" "$REPO/mlx-idle-unload.sh" "$REPO/start-litellm.sh" "$RUNTIME/"
chmod +x "$RUNTIME/mlx-idle-unload.sh" "$RUNTIME/start-litellm.sh"

# --- virtualenv --------------------------------------------------------------
if [ ! -x "$RUNTIME/venv/bin/python" ]; then
    say "Creating virtualenv"
    python3 -m venv "$RUNTIME/venv"
fi
say "Installing Python dependencies (large tree; this takes a few minutes)"
"$RUNTIME/venv/bin/pip" install --quiet --upgrade pip
"$RUNTIME/venv/bin/pip" install --quiet -r "$REPO/requirements.txt"

# LiteLLM's declared fastapi range has no working upper bound (see
# requirements.txt), so assert the symbol its proxy needs rather than trusting
# metadata. Both checks are cheap and catch a broken resolve immediately.
"$RUNTIME/venv/bin/python" -c 'from fastapi.dependencies.utils import get_flat_dependant' \
    || die "FastAPI too new: litellm 1.95 needs get_flat_dependant (use fastapi<0.141)"
"$RUNTIME/venv/bin/python" -c 'import litellm' >/dev/null 2>&1 \
    || die "litellm failed to import"

# --- extractor (optional) ----------------------------------------------------
# Its own venv: PyTorch is large, and LiteLLM's FastAPI pin above must not be
# disturbed by anything the extractor pulls in.
if [[ " ${LABELS[*]} " == *" $EXTRACT_LABEL "* ]]; then
    say "Installing the extractor into $EXTRACT_DIR"
    mkdir -p "$EXTRACT_DIR"
    if [ ! -x "$EXTRACT_DIR/venv/bin/python" ]; then
        python3 -m venv "$EXTRACT_DIR/venv"
    fi
    "$EXTRACT_DIR/venv/bin/pip" install --quiet --upgrade pip
    "$EXTRACT_DIR/venv/bin/pip" install --quiet "$REPO/extract[serve]"
    "$EXTRACT_DIR/venv/bin/python" -c 'import sysone_extract.serve, torch' \
        || die "extractor failed to import"
    if [ -n "$EXTRACT_MODEL_SRC" ]; then
        [ -f "$EXTRACT_MODEL_SRC/extractor.json" ] \
            || die "$EXTRACT_MODEL_SRC is not a trained extractor (no extractor.json)"
        say "Deploying extractor model from $EXTRACT_MODEL_SRC"
        rm -rf "$EXTRACT_DIR/model.new"
        cp -R "$EXTRACT_MODEL_SRC" "$EXTRACT_DIR/model.new"
        rm -rf "$EXTRACT_DIR/model"
        mv "$EXTRACT_DIR/model.new" "$EXTRACT_DIR/model"
    fi
fi

# --- launchd agents ----------------------------------------------------------
say "Rendering and installing launchd agents"
mkdir -p "$AGENTS"
for L in "${LABELS[@]}"; do
    sed "s|__HOME__|$HOME|g" "$REPO/launchd/$L.plist" > "$AGENTS/$L.plist"
done

# Booting out the router kills any Claude Code session routed through it, and
# that session's tool runner then kills this script, leaving litellm and the
# router unloaded. So the reload runs detached: fork, setsid, and the parent
# exits, which reparents it to launchd, out of reach of both process-group and
# process-tree kills. This script only waits for it.
log="$RUNTIME/reload-agents.log"
: > "$log"
python3 -c 'import os, sys
if os.fork(): os._exit(0)
os.setsid()
os.execv(sys.argv[1], sys.argv[1:])' "$REPO/install.sh" --reload-agents >"$log" 2>&1 </dev/null
for _ in $(seq 180); do
    grep -q '^reload-done$' "$log" && break
    sleep 1
done
grep -v '^reload-done$' "$log" || true
grep -q '^reload-done$' "$log" || die "agent reload did not finish in 180s; see $log"

# --- health ------------------------------------------------------------------
say "Waiting for services"
if curl -fsS --retry 40 --retry-delay 2 --retry-connrefused --retry-all-errors \
        --max-time 180 http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "    mlx-lm  :8080 ok"
else
    echo "    mlx-lm  :8080 NOT READY"
fi
if curl -fsS --retry 40 --retry-delay 2 --retry-connrefused --retry-all-errors \
        --max-time 180 http://127.0.0.1:4001/health/liveliness >/dev/null 2>&1; then
    echo "    litellm :4001 ok"
else
    echo "    litellm :4001 NOT READY"
fi
# The router proxies to Anthropic, so an unauthenticated probe answering 401 is
# the expected healthy result. Only a connection failure ("000") is a problem.
code="$(curl -s -o /dev/null -w '%{http_code}' --retry 20 --retry-delay 1 \
        --retry-connrefused --max-time 60 http://127.0.0.1:4000/v1/models 2>/dev/null || true)"
if [ -n "$code" ] && [ "$code" != "000" ]; then
    echo "    router  :4000 ok (HTTP $code from Anthropic passthrough)"
else
    echo "    router  :4000 NOT READY"
fi
if [[ " ${LABELS[*]} " == *" $EXTRACT_LABEL "* ]]; then
    # /health does not load the model, so this stays cheap.
    if curl -fsS --retry 20 --retry-delay 1 --retry-connrefused --max-time 60 \
            http://127.0.0.1:8090/health >/dev/null 2>&1; then
        echo "    extract :8090 ok (model loads on first request; POST :4000/v1/extract)"
    else
        echo "    extract :8090 NOT READY (see ~/Library/Logs/sysone-extract.log)"
    fi
fi

say "Done"
printf '%s\n' \
    "" \
    "To route Claude Code through the router, add to ~/.claude/settings.json:" \
    "" \
    '    "env":   { "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000" }' \
    '    "model": "claude-opus-5[1m]"' \
    "" \
    "This is not done automatically - that file is yours to own." \
    "" \
    "Two gotchas worth knowing:" \
    "  * settings.json env overrides shell environment variables, so once it is" \
    "    set, exporting ANTHROPIC_BASE_URL has no effect. Pass --settings to" \
    "    Claude Code to override it for a single run." \
    "  * The 1M context window is enabled by the literal [1m] in the model name." \
    "    Selecting opus instead of claude-opus-5[1m] silently drops you to 200k." \
    "" \
    "Afterwards: plain claude uses your subscription, and" \
    "claude --model local-qwen uses the local model. See README.md."
