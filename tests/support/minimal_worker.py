"""A warm entrypoint that stands in for a worker at the Host's protocol, without importing JAX.

Each delivery of `launch(control)` publishes `checkpoint:1` with `CHECKPOINT`, waits until `release(control,
job_id)` or the Host's stop file, and reports `succeeded`, or `stopped` at `CHECKPOINT`. With `fail=True` it
reports the typed failure `update_failed` at once. `train_with(monkeypatch, control)` makes every
`JobService` launch its Trains with it, so Studio admits, warms, stops and resumes them as it does real ones.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

from tests.support.hang import HANG_SECONDS

ENTRYPOINT = "tests.support.minimal_worker:main"
CHECKPOINT = {"completed_updates": 1}
_REPOSITORY = Path(__file__).resolve().parents[2]


def launch(control: Path, *, fail: bool = False):
    from feedbax.host import Launch

    return Launch(
        (sys.executable, str(control), *(("fail",) if fail else ())),
        cwd=str(_REPOSITORY),
        warm_entrypoint=ENTRYPOINT,
    )


def release(control: Path, job_id: str) -> None:
    (control / job_id).touch()


def train_with(monkeypatch, control: Path) -> None:
    from feedbax.jobs import JobService

    init = JobService.__init__

    def minimal(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._training_worker = replace(launch(control), env=self._training_worker.env)

    monkeypatch.setattr(JobService, "__init__", minimal)


def main(args):
    control, *mode = args
    job_id = os.environ["FEEDBAX_HOST_JOB_ID"]
    stop = Path(os.environ["FEEDBAX_HOST_STOP_FILE"])
    released = Path(control) / job_id
    with Path(os.environ["FEEDBAX_HOST_EVENT_LOG"]).open("a", buffering=1) as events:
        def emit(kind, data):
            print(json.dumps({"kind": kind, "data": data}), file=events)

        if mode == ["fail"]:
            emit("result", {"outcome": "failed", "error": {"code": "update_failed", "message": "Update failed."}})
            return 1
        emit("publication", {"key": "checkpoint:1", "checkpoint": CHECKPOINT})
        deadline = time.monotonic() + HANG_SECONDS
        while not released.exists() and not stop.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Neither {released} nor the stop file appeared.")
            time.sleep(0.01)
        emit("result", {"outcome": "stopped", "checkpoint": CHECKPOINT} if stop.exists() else {"outcome": "succeeded"})
    return 0
