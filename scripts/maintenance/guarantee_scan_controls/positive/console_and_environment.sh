#!/usr/bin/env bash
# Positive controls for the console-script and environment-variable channels.
set -euo pipefail
export FEEDBAX_JAX_COMPILATION_CACHE_DIR=/tmp/control
feedbax-analysis report --spec control.json
feedbax-figure resolve --with-lineage
