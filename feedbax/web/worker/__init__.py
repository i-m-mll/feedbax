"""Feedbax Studio headless training worker.

Run as:
    FEEDBAX_WORKER_AUTH_TOKEN=<secret> python -m feedbax.web.worker --host <HOST>
"""

from __future__ import annotations

import argparse
import ipaddress
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Feedbax headless training worker")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port to listen on")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host/interface to bind to (use 0.0.0.0 for remote access)",
    )
    args = parser.parse_args()

    auth_token = os.environ.get("FEEDBAX_WORKER_AUTH_TOKEN")
    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host.lower() == "localhost"
    if not loopback and not auth_token:
        parser.error(
            "non-loopback workers require FEEDBAX_WORKER_AUTH_TOKEN in the process environment"
        )

    import uvicorn

    from feedbax.web.worker.app import create_app

    app = create_app(auth_token=auth_token)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
