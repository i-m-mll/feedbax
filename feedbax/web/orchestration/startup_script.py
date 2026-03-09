"""Worker startup script injected as GCP instance metadata."""


def make_startup_script(
    feedbax_install_cmd: str = "pip install feedbax",
) -> str:
    """Generate the GCP instance startup script.

    Args:
        feedbax_install_cmd: Shell command used to install feedbax on the
            instance.  Defaults to ``pip install feedbax`` for backwards
            compatibility; callers should pass a git-install command pointing
            at the desired branch/ref.

    Returns:
        A bash script string suitable for use as GCP instance metadata.
    """
    return f"""#!/bin/bash
set -euo pipefail
{feedbax_install_cmd}
pip install uvicorn httpx fastapi 2>&1 | tail -5

# Install and start Tailscale (if TS_AUTH_KEY is set)
if [[ -n "${{TS_AUTH_KEY:-}}" ]]; then
  curl -fsSL https://tailscale.com/install.sh | sh
  tailscale up --authkey="${{TS_AUTH_KEY}}" --hostname="feedbax-worker-$(hostname)" 2>&1
fi

# Start the worker in the background
nohup python -m feedbax.web.worker \\
  --host 0.0.0.0 --port ${{WORKER_PORT:-8765}} \\
  ${{AUTH_TOKEN:+--auth-token "$AUTH_TOKEN"}} \\
  >> /var/log/feedbax-worker.log 2>&1 &
echo "Worker started on port ${{WORKER_PORT:-8765}}"
"""


# Backwards-compat constant: any direct import of STARTUP_SCRIPT continues to
# work unchanged (uses the PyPI install default).
STARTUP_SCRIPT = make_startup_script()
