"""Hold managed update boundaries for checkpoint selection tests."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from feedbax.contracts.method_checkpoint import CheckpointLatestPointer, method_checkpoint_extra_path
from feedbax.persistence.method_checkpoint import read_method_checkpoint_manifest
from tests.support.store_damage import writable_store_object


@contextmanager
def damaged_method_state(checkpoint):
    root = Path(checkpoint["checkpoint_root"])
    pointer = CheckpointLatestPointer.model_validate(checkpoint["pointer"])
    manifest = read_method_checkpoint_manifest(root, pointer)
    directory = (root / pointer.manifest_relative_path).parent
    path = directory / method_checkpoint_extra_path(manifest.extra)
    original = path.read_bytes()
    with writable_store_object(path):
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        try:
            yield
        finally:
            path.write_bytes(original)


def install_finite_gate(jobs, tmp_path, *, stop_at, resume_at):
    (tmp_path / "finite_gate.py").write_text("""
import json
import os
from pathlib import Path
from time import monotonic, sleep
from feedbax.training.method_runtime import TrainingLoop

run_block = TrainingLoop.run_block

def wait(path):
    deadline = monotonic() + 22
    while not path.exists():
        if monotonic() > deadline:
            raise TimeoutError(f"Jobs did not release {path.name}")
        sleep(.002)

def wrapped(self, state, *, completed, count):
    root = Path(os.environ["FINITE_PROOF_ROOT"])
    source = json.loads(Path(os.environ["FEEDBAX_HOST_INPUT"]).read_bytes())
    stop = Path(os.environ["FEEDBAX_HOST_STOP_FILE"])
    resumed = source.get("resume") is not None
    stop_at = int(os.environ["FINITE_STOP_AT"])
    resume_at = int(os.environ["FINITE_RESUME_AT"])
    if resumed and completed == resume_at:
        (root / "resumed-interior").touch()
        wait(root / "finish-train")

    if not resumed and completed < stop_at:
        count = min(count, stop_at - completed)
    result = run_block(self, state, completed=completed, count=count)
    if not resumed and result.completed == stop_at:
        (root / "before-stop").touch()
        wait(stop)
    return result

TrainingLoop.run_block = wrapped
""")
    launch = jobs._training_worker
    jobs._training_worker = replace(
        launch,
        env=launch.env
        | {
            "JAX_PLATFORMS": "cpu",
            "FINITE_PROOF_ROOT": str(tmp_path),
            "FINITE_STOP_AT": str(stop_at),
            "FINITE_RESUME_AT": str(resume_at),
            "PYTHONPATH": f"{tmp_path}:{Path.cwd()}",
        },
        # Preserve the canonical entrypoint authorized to publish typed Train results.
        preload=(*launch.preload, "finite_gate"),
    )
