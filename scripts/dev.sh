#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v pnpm >/dev/null 2>&1; then
  echo "pnpm is required. Install with: npm install -g pnpm" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv run uvicorn feedbax.web.app:app --reload --port 8000 &
else
  uvicorn feedbax.web.app:app --reload --port 8000 &
fi
UVICORN_PID=$!

pushd "$ROOT_DIR/web" >/dev/null
pnpm install
pnpm dev
popd >/dev/null

kill $UVICORN_PID
