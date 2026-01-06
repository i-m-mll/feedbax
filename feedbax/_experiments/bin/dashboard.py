#!/usr/bin/env python
"""CLI entry point for launching the figure review dashboard."""

import argparse
import logging

from feedbax._experiments.dashboard.app import create_app

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the dashboard CLI."""
    parser = argparse.ArgumentParser(
        description="Launch the interactive figure review dashboard"
    )
    parser.add_argument(
        "--db-name",
        default="main",
        help="Name of the database to use (default: main)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the dashboard on (default: 8050)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to run the dashboard on (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Launching dashboard with database: {args.db_name}")
    logger.info(f"Dashboard will be available at http://{args.host}:{args.port}")

    # Create and run the app
    app = create_app(db_name=args.db_name)
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
