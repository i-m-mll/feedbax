"""Hold managed update boundaries for checkpoint selection tests."""

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path


@contextmanager
def _damaged_prng(checkpoint):
    directory = (
        Path(checkpoint["checkpoint_root"]) / checkpoint["pointer"]["manifest_relative_path"]
    ).parent
    manifest = json.loads((directory / "manifest.json").read_bytes())
    (prng,) = (slot for slot in manifest["slots"] if slot["slot"] == "prng")
    path = directory / prng["relative_path"]
    original = path.read_bytes()
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
from feedbax.training import executor

entry = "execute_managed_training_run"
execute = getattr(executor, entry)

def wait(path):
    deadline = monotonic() + 22
    while not path.exists():
        if monotonic() > deadline:
            raise TimeoutError(f"Jobs did not release {path.name}")
        sleep(.002)

def wrapped(*args, **kwargs):
    root = Path(os.environ["FINITE_PROOF_ROOT"])
    source = json.loads(Path(os.environ["FEEDBAX_HOST_INPUT"]).read_bytes())
    stop = Path(os.environ["FEEDBAX_HOST_STOP_FILE"])
    commit = kwargs["committed_update_callback"]
    published = kwargs["checkpoint_published"]

    def on_commit(coordinate, completed, slots):
        commit(coordinate, completed, slots)
        if source.get("resume") is None and completed == int(os.environ["FINITE_STOP_AT"]):
            (root / "before-stop").touch()
            wait(stop)

    def on_publish(write):
        published(write)
        if source.get("resume") and write.latest_pointer.completed_training_batches == int(os.environ["FINITE_RESUME_AT"]):
            (root / "resumed-interior").touch()
            wait(root / "finish-train")

    kwargs.update(committed_update_callback=on_commit, checkpoint_published=on_publish)
    return execute(*args, **kwargs)

setattr(executor, entry, wrapped)
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
