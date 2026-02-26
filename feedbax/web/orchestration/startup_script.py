"""Worker startup script injected as GCP instance metadata."""

STARTUP_SCRIPT = """#!/bin/bash
set -euo pipefail
# Install feedbax and uvicorn
pip install feedbax uvicorn httpx fastapi 2>&1 | tail -5

# Install and start Tailscale (if TS_AUTH_KEY is set)
if [[ -n "${TS_AUTH_KEY:-}" ]]; then
  curl -fsSL https://tailscale.com/install.sh | sh
  tailscale up --authkey="${TS_AUTH_KEY}" --hostname="feedbax-worker-$(hostname)" --accept-routes 2>&1
fi

# Start the worker in the background
nohup python -m feedbax.web.worker \\
  --host 0.0.0.0 \\
  --port ${WORKER_PORT:-8765} \\
  ${AUTH_TOKEN:+--auth-token "$AUTH_TOKEN"} \\
  >> /var/log/feedbax-worker.log 2>&1 &
echo "Worker started on port ${WORKER_PORT:-8765}"
"""
