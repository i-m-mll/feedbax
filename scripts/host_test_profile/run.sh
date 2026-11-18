#!/usr/bin/env bash
# usage: scripts/host_test_profile/run.sh <record-dir> <pytest args...>
# Runs pytest with every Python process recording Host, worker and job timings into <record-dir>.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
record="$1"; shift
mkdir -p "$record"
export FEEDBAX_HOST_PROFILE_DIR="$(cd "$record" && pwd)"
export PYTHONPATH="$here/hook${PYTHONPATH:+:$PYTHONPATH}"
cd "$here/../.."
exec .venv/bin/python -m pytest -p host_profile_plugin "$@"
