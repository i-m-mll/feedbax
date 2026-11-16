"""Connect the production SSH/rsync Store transport to a local venue."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import sys

from feedbax.persistence.store_sync import RemoteStore, SSHStoreConnection


@dataclass(frozen=True)
class LoopbackStore:
    endpoint: RemoteStore
    trace: Path
    disconnect: Path

    def rsync_calls(self) -> tuple[str, ...]:
        return tuple(line for line in self.trace.read_text().splitlines() if "--server" in line)


def loopback_store(root: Path | str, directory: Path, *, download_gate: Path | None = None) -> LoopbackStore:
    directory.mkdir()
    trace = directory / "trace"
    disconnect = directory / "disconnect"
    ssh = directory / "ssh.py"
    ssh.write_text("""import os
from pathlib import Path
import select
import shlex
import subprocess
import sys
import time

trace, disconnect, download_gate, _, *args = sys.argv[1:]
with open(trace, "a") as log:
    log.write(shlex.join(shlex.split(" ".join(args))) + "\\n")
argv = shlex.split(" ".join(args))
if download_gate != "-" and "--server" in argv and "--sender" in argv:
    deadline = time.monotonic() + 14
    while Path(download_gate).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("The fixture did not release the download gate")
        time.sleep(.05)
fault = Path(disconnect)
if "--server" in argv and "--sender" not in argv and fault.exists():
    limit = int(fault.read_text())
    fault.unlink()
    with subprocess.Popen(argv, stdin=subprocess.PIPE) as receiver:
        while limit:
            if not select.select([sys.stdin], [], [], 5)[0]:
                raise TimeoutError("rsync did not send the expected fixture bytes")
            data = os.read(sys.stdin.fileno(), min(limit, 65536))
            if not data:
                break
            receiver.stdin.write(data)
            receiver.stdin.flush()
            limit -= len(data)
        receiver.stdin.close()
        receiver.wait()
    sys.exit(74)
os.execvp(argv[0], argv)
""")
    connection = SSHStoreConnection(
        "local-venue", ssh_argv=(sys.executable, str(ssh), str(trace), str(disconnect), str(download_gate or "-")),
    )
    return LoopbackStore(
        endpoint=RemoteStore(PurePosixPath(root), connection, python=sys.executable),
        trace=trace,
        disconnect=disconnect,
    )
