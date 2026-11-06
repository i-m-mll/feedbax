#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pushd "$ROOT_DIR/web" >/dev/null
bun install --frozen-lockfile
bun run build
popd >/dev/null
