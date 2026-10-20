#!/usr/bin/env bash
# Starts LiteLLM with the llm-d API key pulled from Scaleway Secret Manager
# (project platform, secret /platform/llm-d/secrets, field API_KEY). The key only
# lives in this process's environment. If the fetch fails (offline, scw not
# logged in), LiteLLM still starts: local-qwen keeps working, local-llmd 401s.
# Parsed with the venv's python: launchd's PATH has no jq.
set -uo pipefail
DIR="$(dirname "$0")"

LLM_D_API_KEY=$(scw secret version access-by-path \
  project-id=c5fa53ce-c6b7-43fd-b77a-9a3ce1c940aa \
  secret-path=/platform/llm-d secret-name=secrets revision=latest_enabled -o json \
  | "$DIR/venv/bin/python" -c 'import sys, json, base64; print(json.loads(base64.b64decode(json.load(sys.stdin)["data"]))["API_KEY"])') \
  || LLM_D_API_KEY=""
[ -n "$LLM_D_API_KEY" ] || echo "start-litellm: could not fetch llm-d API key; local-llmd disabled" >&2
export LLM_D_API_KEY

exec "$DIR/venv/bin/litellm" "$@"
