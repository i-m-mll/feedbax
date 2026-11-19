"""A warm entrypoint that stands in for a worker at the Host's protocol, without importing JAX.

Each delivery of `launch(control)` publishes `checkpoint:1` with `CHECKPOINT`, waits until `release(control,
job_id)` or the Host's stop file, and reports `succeeded`, or `stopped` at `CHECKPOINT`. With `fail=True` it
reports the typed failure `update_failed` at once. With `leave_thread=True` it first starts a thread that outlives
the delivery, and with `load=path` it first imports the Python file at `path`. `train_with(monkeypatch, control)` makes every
`JobService` launch its Trains with it, so Studio admits, warms, stops and resumes them as it does real ones.
"""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

from tests.support.hang import HANG_SECONDS

ENTRYPOINT = "tests.support.minimal_worker:main"
CHECKPOINT = {"completed_updates": 1}
_REPOSITORY = Path(__file__).resolve().parents[2]


def launch(
    control: Path, *, fail: bool = False, leave_thread: bool = False, load: Path | None = None,
    project_root: Path | None = None,
):
    from feedbax.host import Launch

    modes = (*(("fail",) if fail else ()), *(("thread",) if leave_thread else ()), *(("load", str(load)) if load else ()))
    return Launch(
        (sys.executable, str(control), *modes),
        cwd=str(_REPOSITORY),
        warm_entrypoint=ENTRYPOINT,
        project_root=str(project_root) if project_root is not None else None,
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
    if "thread" in mode:
        threading.Thread(target=threading.Event().wait, daemon=True).start()
    if "load" in mode:
        path = Path(mode[mode.index("load") + 1])
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = sys.modules[path.stem] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    job_id = os.environ["FEEDBAX_HOST_JOB_ID"]
    stop = Path(os.environ["FEEDBAX_HOST_STOP_FILE"])
    released = Path(control) / job_id
    with Path(os.environ["FEEDBAX_HOST_EVENT_LOG"]).open("a", buffering=1) as events:
        def emit(kind, data):
            print(json.dumps({"kind": kind, "data": data}), file=events)

        if "fail" in mode:
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
