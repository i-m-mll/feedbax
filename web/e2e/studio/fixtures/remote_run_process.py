"""Interrupt exactly one worker belonging to the isolated machine fixture."""

from contextlib import closing
import os
from pathlib import Path
import signal
import sqlite3
import sys
from time import monotonic, sleep
from uuid import UUID

from feedbax.host.process import verify


def interrupt(session_id: UUID, attempt: str) -> None:
    home = Path(os.environ["FEEDBAX_STUDIO_TEST_HOME"]).resolve(strict=True)
    venue = home / "state" / "studio-e2e-remote" / "venues" / session_id.hex
    assert venue.resolve(strict=True) == venue and (venue / "resource.json").is_file()
    database = venue / "host" / "host.sqlite"
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as db:
        deadline = monotonic() + 4
        while True:
            pid, worker_token, state, checkpoint = db.execute(
                "SELECT worker_pid, worker_token, state, checkpoint FROM jobs WHERE id=?", (attempt,),
            ).fetchone()
            assert state == "running" and pid > 1
            if checkpoint is not None:
                assert verify(pid, worker_token)[0] == "verified"
                os.killpg(pid, signal.SIGKILL)
                return
            if monotonic() >= deadline:
                raise TimeoutError("The isolated worker did not publish a checkpoint.")
            sleep(.05)


if __name__ == "__main__":
    interrupt(UUID(sys.argv[1]), sys.argv[2])
