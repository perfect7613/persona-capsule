#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ "${CONFIRM_CREDENTIALS_ROTATED:-false}" != "true" ]]; then
  echo "Deployment refused. Rotate exposed credentials, then set CONFIRM_CREDENTIALS_ROTATED=true." >&2
  exit 1
fi

: "${MODAL_TOKEN_ID:?Missing MODAL_TOKEN_ID}"
: "${MODAL_TOKEN_SECRET:?Missing MODAL_TOKEN_SECRET}"
: "${SPACE_HF_SERVICE_TOKEN:?Missing SPACE_HF_SERVICE_TOKEN}"

uv run modal secret create persona-capsule-huggingface \
  "HF_TOKEN=${SPACE_HF_SERVICE_TOKEN}" \
  "HF_DEEP_REPO_PREFIX=${HF_DEEP_REPO_PREFIX:-persona-capsule}"

uv run modal deploy modal_minicpm.py
uv run modal deploy modal_flux.py
uv run modal deploy modal_nemotron.py
uv run modal deploy modal_deep.py
