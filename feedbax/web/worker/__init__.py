"""Feedbax Studio headless training worker.

Run as:
    python -m feedbax.web.worker --port <PORT> [--host HOST] [--auth-token TOKEN]
"""
from __future__ import annotations

import argparse

import uvicorn

from feedbax.web.worker.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Feedbax headless training worker")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port to listen on")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host/interface to bind to (use 0.0.0.0 for remote access)",
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        default=None,
        dest="auth_token",
        help="Shared secret; when set all requests must include Authorization: Bearer <token>",
    )
    args = parser.parse_args()

    app = create_app(auth_token=args.auth_token)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
