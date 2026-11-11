"""Run the host daemon under its own interpreter and session."""

import argparse
import asyncio
import signal
from contextlib import suppress
from pathlib import Path

from feedbax.host.daemon import Host


async def run(root: Path) -> None:
    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, current.cancel)
    loop.add_signal_handler(signal.SIGINT, current.cancel)
    with suppress(asyncio.CancelledError):
        await Host(root).run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    options = parser.parse_args()
    asyncio.run(run(options.root.resolve()))


if __name__ == "__main__":
    main()
