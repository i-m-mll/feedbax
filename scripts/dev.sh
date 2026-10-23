#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to run the Studio frontend" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv run uvicorn feedbax.web.app:app --reload --port 8000 &
else
  uvicorn feedbax.web.app:app --reload --port 8000 &
fi
UVICORN_PID=$!
trap 'kill "$UVICORN_PID" 2>/dev/null || true' EXIT

pushd "$ROOT_DIR/web" >/dev/null
npm install
npm run dev
popd >/dev/null
