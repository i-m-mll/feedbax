"""Hold managed update boundaries for checkpoint selection tests."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from feedbax.contracts.method_checkpoint import CheckpointLatestPointer
from feedbax.persistence.method_checkpoint import method_checkpoint_files
from tests.support.hang import HANG_SECONDS
from tests.support.store_damage import writable_store_object


@contextmanager
def damaged_checkpoint_model(checkpoint):
    root = Path(checkpoint["checkpoint_root"])
    pointer = CheckpointLatestPointer.model_validate(checkpoint["pointer"])
    files = method_checkpoint_files(root, pointer, include_extra=False)
    path = root / next(relative for relative in files if relative != Path(pointer.manifest_relative_path))
    original = path.read_bytes()
    with writable_store_object(path):
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        try:
            yield
        finally:
            path.write_bytes(original)


def install_finite_gate(jobs, tmp_path, *, stop_at, resume_at):
    (tmp_path / "finite_gate.py").write_text(f"HANG_SECONDS = {HANG_SECONDS}\n" + """
import json
import os
from pathlib import Path
from time import monotonic, sleep
from feedbax.training.method_runtime import TrainingLoop
from feedbax.training_host_worker import _Events

run_block = TrainingLoop.run_block

def wait(path):
    deadline = monotonic() + HANG_SECONDS
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
    if not resumed and completed < stop_at:
        count = min(count, stop_at - completed)
    result = run_block(self, state, completed=completed, count=count)
    factor_file = root / "resume-model-factor"
    if resumed and factor_file.is_file():
        import equinox as eqx
        import jax

        factor = float(factor_file.read_text())
        model = jax.tree.map(
            lambda value: value * factor if eqx.is_inexact_array(value) else value,
            result.state.model,
        )
        result = result._replace(state=result.state._replace(model=model))
    if not resumed and result.completed == stop_at:
        (root / "before-stop").touch()
        wait(stop)
    return result

def published(self, kind, data, *, sync=False):
    append(self, kind, data, sync=sync)
    source = json.loads(Path(os.environ["FEEDBAX_HOST_INPUT"]).read_bytes())
    if (
        kind == "publication" and source.get("resume") is not None
        and data["checkpoint"]["completed_updates"] == int(os.environ["FINITE_RESUME_AT"])
    ):
        root = Path(os.environ["FINITE_PROOF_ROOT"])
        (root / "resumed-interior").touch()
        wait(root / "finish-train")

TrainingLoop.run_block = wrapped
append = _Events.append
_Events.append = published
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
