#!/bin/bash
# Restart mlx-lm once it has been idle long enough, so the ~14GB model is not
# held resident forever. mlx_lm.server has no idle-unload option, and the plist
# passes no --model, so a restart returns the process to ~90MB and the model
# reloads lazily on the next request.
#
# Idleness comes from the router's stamp file rather than a log mtime: mlx-lm
# does not log every request, and other writers touch those logs.

set -uo pipefail

LABEL="ai.arlq.mlx-lm"
STAMP="${HOME}/.local/share/claude-local-router/.last-local-use"
TROUBLE="${HOME}/.local/share/claude-local-router/.last-upstream-trouble"
LOG="${HOME}/Library/Logs/mlx-lm-watchdog.log"
IDLE_SECONDS="${MLX_IDLE_SECONDS:-900}"   # 15 minutes
# While Anthropic is misbehaving, keep the model resident: a cold load costs
# ~90s, and paying that mid-outage is exactly when it hurts most.
KEEP_WARM_SECONDS="${MLX_KEEP_WARM_SECONDS:-1800}"  # 30 minutes
RSS_THRESHOLD_KB="${MLX_RSS_THRESHOLD_KB:-1048576}"  # 1GB => a model is loaded
PORT="${MLX_PORT:-8080}"

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "$LOG"; }

PID=$(launchctl print "gui/$(id -u)/${LABEL}" 2>/dev/null | awk '/^\tpid = /{print $3}')
[ -n "${PID:-}" ] || exit 0

# Nothing loaded yet: no memory to reclaim.
RSS=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
[ -n "${RSS:-}" ] || exit 0
[ "$RSS" -ge "$RSS_THRESHOLD_KB" ] || exit 0

# Never restart before the model has been used at least once.
[ -f "$STAMP" ] || exit 0
NOW=$(date +%s)
LAST=$(stat -f %m "$STAMP" 2>/dev/null || echo 0)
IDLE=$(( NOW - LAST ))
[ "$IDLE" -ge "$IDLE_SECONDS" ] || exit 0

# Anthropic recently failed: stay loaded so the next fallback is instant.
if [ -f "$TROUBLE" ]; then
    TROUBLE_AGE=$(( NOW - $(stat -f %m "$TROUBLE" 2>/dev/null || echo 0) ))
    if [ "$TROUBLE_AGE" -lt "$KEEP_WARM_SECONDS" ]; then
        log "idle ${IDLE}s but upstream trouble ${TROUBLE_AGE}s ago - staying warm"
        exit 0
    fi
fi

# A request may be in flight even though the stamp is old (long generation).
# Restarting then would kill it, so skip while any client is still connected.
if lsof -nP -iTCP:"${PORT}" -sTCP:ESTABLISHED >/dev/null 2>&1; then
    log "idle ${IDLE}s but connections still established on :${PORT} - skipping"
    exit 0
fi

GB=$(awk -v k="$RSS" 'BEGIN{printf "%.2f", k/1048576}')
log "idle ${IDLE}s, RSS ${GB}GB - restarting ${LABEL} to release memory"
launchctl kickstart -k "gui/$(id -u)/${LABEL}" >/dev/null 2>&1 \
    && log "restart issued" \
    || log "restart FAILED"
