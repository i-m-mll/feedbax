"""Feedbax Studio headless training worker.

Run as:
    python -m feedbax.web.worker --port <PORT>
"""
from __future__ import annotations

import argparse

import uvicorn

from feedbax.web.worker.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Feedbax headless training worker")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port to listen on")
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
