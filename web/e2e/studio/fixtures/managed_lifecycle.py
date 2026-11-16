"""Observe the isolated Host and exact Store result for the mounted M2 gate."""

import json
import os
from pathlib import Path
import signal
import sys
import time

from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient, HostError


def read_job(host, project, attempt):
    (row,) = host.jobs_snapshot(
        project=project, job_id=attempt, include_launch=True, start_host=False,
    )["jobs"]
    assert row["id"] == attempt and row["project"] == project
    return row


def stopped_checkpoint(paths, host, project, attempt):
    from feedbax.contracts.base import ParentRef
    from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
    from feedbax.training.checkpoint_custody import authenticate_checkpoint_custody_ref

    row = read_job(host, project, attempt)
    assert row["state"] == "stopped", row
    checkpoint = PublishedTrainingCheckpoint.model_validate(row["checkpoint"])
    assert checkpoint.checkpoint_root.is_relative_to(paths.web / "projects" / project)
    pointer = checkpoint.pointer
    transaction = authenticate_checkpoint_custody_ref(
        ParentRef(
            kind="TrainingCheckpointTransactionManifest",
            id=pointer.transaction_id,
            role="training_checkpoint_custody",
            uri=pointer.manifest_relative_path,
            metadata={"manifest_sha256": pointer.manifest_sha256},
        ),
        allowed_root=checkpoint.checkpoint_root,
    ).manifest
    assert transaction.run_id == attempt == pointer.run_id
    assert transaction.completed_training_batches == checkpoint.completed_updates
    assert transaction.content_integrity_digest.transaction_root_sha256 == pointer.transaction_root_sha256
    return row["checkpoint"]


def restart(paths, host, project, attempt):
    checkpoint = stopped_checkpoint(paths, host, project, attempt)
    assert all(
        job["outcome"] is not None
        for job in host.jobs_snapshot(start_host=False)["jobs"]
    ), "The test Host must be idle before restart"
    old_pid = host.ping()["pid"]
    # The launcher owns this isolated home. Signal only the Host returned by its socket.
    os.kill(old_pid, signal.SIGKILL)
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        try:
            new_pid = host.ping()["pid"]
        except HostError as error:
            if error.code != "disconnected":
                raise
        else:
            if new_pid != old_pid:
                assert read_job(host, project, attempt)["checkpoint"] == checkpoint
                return {"old_host": old_pid, "new_host": new_pid, "checkpoint": checkpoint}
        time.sleep(0.05)
    raise RuntimeError("The isolated Host did not restart")


def resume(host, project, attempt, successor):
    row = read_job(host, project, successor)
    assert row["continues"] == attempt
    return {"host": host.ping()["pid"], "checkpoint": row["launch"]["payload"]["resume"]}


def declare_variants(paths, project, revision):
    from feedbax.contracts.project_io import dump_text_project, load_text_project
    from feedbax.contracts.project_sweep import Sweep

    root = paths.web / "projects" / project
    saved = load_text_project(root)
    assert saved.revision == revision and saved.project.sweep is None
    updated = saved.project.model_copy(update={
        "sweep": Sweep.model_validate({"rows": {"base": {}, "compare": {"from": "base"}}}),
    })
    authored = dump_text_project(updated, saved.configuration)
    for name in ("variants/base.json", "variants/compare.json", saved.declaration):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(authored[name])
    return {"revision": load_text_project(root).revision}


def completed_receipt(host, project, attempt, successor):
    from feedbax.job_completion import verify_bound_training_completion

    completed = read_job(host, project, successor)
    assert completed["continues"] == attempt
    verify_bound_training_completion(completed["outcome"]["completion"])
    return completed["outcome"]["receipt"]


def result(paths, host, project, attempt, successor, revision, evaluation, trial):
    from feedbax.jobs import JobService
    from feedbax.plugins.application import new_application_registry_bundle
    from feedbax.web.services.project_folder import ProjectFolderService

    completed_receipt(host, project, attempt, successor)
    folders = ProjectFolderService(paths.web / "projects")
    plan = folders.read_plan(project, revision=revision)
    jobs = JobService(host, new_application_registry_bundle())
    result = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=evaluation)
    with result.open_states(verify=True) as dataset:
        position = dataset.descriptor.attrs["scene_position"]
        samples = dataset.select(position, labels={"trial": trial})[0, 0].tolist()
    return {"receipt": result.receipt.model_dump(mode="json"), "samples": samples}


def main():
    action, home, project, attempt, *args = sys.argv[1:]
    root = Path(home).resolve(strict=True)
    assert root == Path(os.environ["FEEDBAX_STUDIO_TEST_HOME"]).resolve(strict=True)
    paths = AppPaths.resolve(instance_root=root)
    paths.configure_studio(desktop=False)
    host = HostClient(paths.host)
    match action:
        case "declare-variants":
            record = declare_variants(paths, project, attempt)
        case "checkpoint":
            record = stopped_checkpoint(paths, host, project, attempt)
        case "restart":
            record = restart(paths, host, project, attempt)
        case "resume":
            record = resume(host, project, attempt, *args)
        case "completed":
            record = completed_receipt(host, project, attempt, *args)
        case "result":
            record = result(paths, host, project, attempt, *args)
        case _:
            raise ValueError(f"Unknown lifecycle observation: {action}")
    print(json.dumps(record, allow_nan=False))


if __name__ == "__main__":
    main()
