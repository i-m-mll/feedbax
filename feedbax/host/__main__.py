"""Run the detached Host or attach to a provisioned venue using stdio."""

import argparse
import asyncio
import json
import signal
import sys
from contextlib import suppress
from pathlib import Path

from feedbax.host.client import HostClient
from feedbax.host.io import encode
from feedbax.host.protocol import MAX_RPC_BYTES, HostError, JsonObject
from feedbax.host.session import SessionFence, initialize_session


async def run(root: Path, *, required_expiry_ms: int | None = None) -> None:
    from feedbax.host.daemon import Host

    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    assert current is not None
    loop.add_signal_handler(signal.SIGTERM, current.cancel)
    loop.add_signal_handler(signal.SIGINT, current.cancel)
    with suppress(asyncio.CancelledError):
        await Host(root, required_expiry_ms=required_expiry_ms).run()


def attach(root: Path, *, start_host: bool) -> None:
    """Relay ordinary Host RPCs. Closing stdin only disconnects the relay."""
    while frame := sys.stdin.buffer.readline(MAX_RPC_BYTES + 1):
        reply: JsonObject
        try:
            if len(frame) > MAX_RPC_BYTES or not frame.endswith(b"\n"):
                raise HostError("invalid_request", "Expected a complete Host request within the size limit.")
            if SessionFence.load(root) is None:
                raise HostError("session_uninitialized", "This venue has no initialized session.")
            reply = HostClient(root)._exchange(frame, timeout=65, start_host=start_host)
        except HostError as error:
            reply = {"error": {"code": error.code, "message": str(error)}}
        except (OSError, EOFError, ValueError, TypeError):
            reply = {"error": {"code": "disconnected", "message": "The requested venue Host is unavailable."}}
        sys.stdout.buffer.write(encode(reply))
        sys.stdout.buffer.flush()
        if len(frame) > MAX_RPC_BYTES or not frame.endswith(b"\n"):
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("operation", choices=("serve", "attach", "init-session"), nargs="?", default="serve")
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument("--require-session-until-ms", type=int, metavar="TIMESTAMP")
    options = parser.parse_args()
    if options.no_start and options.operation != "attach":
        parser.error("--no-start requires attach")
    if options.require_session_until_ms is not None and options.operation != "serve":
        parser.error("--require-session-until-ms requires serve")
    root = options.root.resolve()
    match options.operation:
        case "serve":
            try:
                asyncio.run(run(root, required_expiry_ms=options.require_session_until_ms))
            except HostError as error:
                parser.exit(1, f"{error.code}: {error}\n")
        case "attach":
            attach(root, start_host=not options.no_start)
        case "init-session":
            data = sys.stdin.buffer.read(MAX_RPC_BYTES + 1)
            if len(data) > MAX_RPC_BYTES:
                parser.error("Session initialization request exceeds the size limit.")
            try:
                initialize_session(root, **json.loads(data))
            except HostError as error:
                parser.exit(1, f"{error.code}: {error}\n")
            except (TypeError, ValueError, OSError):
                parser.exit(1, "Could not initialize the venue session.\n")


if __name__ == "__main__":
    main()
