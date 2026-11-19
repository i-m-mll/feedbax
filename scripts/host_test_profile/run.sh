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
status=0
.venv/bin/python -m pytest -p host_profile_plugin "$@" || status=$?
.venv/bin/python scripts/host_test_profile/reap.py "$FEEDBAX_HOST_PROFILE_DIR" || status=1
exit "$status"
