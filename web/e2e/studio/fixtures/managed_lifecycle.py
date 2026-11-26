"""Observe the isolated Host and exact Store result for the mounted M2 gate."""

import json
import os
from pathlib import Path
import signal
import sys
import time

from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient, HostError
from tests.support.browser_seed_actions import NonSeed


def read_job(host, project, attempt):
    (row,) = host.jobs_snapshot(
        project=project, job_id=attempt, include_launch=True, start_host=False,
    )["jobs"]
    assert row["id"] == attempt and row["project"] == project
    return row


def stopped_checkpoint(paths, host, project, attempt):
    from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
    from feedbax.persistence.method_checkpoint import read_method_checkpoint_manifest
    from feedbax.persistence.canonical_train import BoundStudioTrainingJobInput
    from feedbax.web.services.project_folder import ProjectFolderService

    row = read_job(host, project, attempt)
    assert row["state"] == "stopped", row
    checkpoint = PublishedTrainingCheckpoint.model_validate(row["checkpoint"])
    _, location = ProjectFolderService(paths.web / "projects").open_native_id(project)
    assert checkpoint.checkpoint_root.is_relative_to(location.path)
    pointer = checkpoint.pointer
    transaction = read_method_checkpoint_manifest(checkpoint.checkpoint_root, pointer)
    launch = BoundStudioTrainingJobInput.model_validate(row["launch"]["payload"])
    assert transaction.run_id == attempt == pointer.run_id
    assert transaction.clock.completed_updates == checkpoint.completed_updates
    assert transaction.identity.request_id == row["request"] == launch.request_id
    assert transaction.identity.method_key == launch.spec.method
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
                assert stopped_checkpoint(paths, host, project, attempt) == checkpoint
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
    from feedbax.web.services.project_folder import ProjectFolderService

    _, location = ProjectFolderService(paths.web / "projects").open_native_id(project)
    root = location.path
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


def completed_train(host, project, attempt, successor):
    from feedbax.host.job_completion import verify_bound_training_completion

    completed = read_job(host, project, successor)
    assert completed["continues"] == attempt
    verify_bound_training_completion(completed["outcome"]["completion"])
    return completed


def completed_receipt(host, project, attempt, successor):
    return completed_train(host, project, attempt, successor)["outcome"]["receipt"]


def result(paths, host, project, attempt, successor, revision, evaluation, trial):
    from feedbax.contracts.artifact_custody import ContentObjectRef
    from feedbax.jobs import JobService
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.persistence.effective_success import SuccessView
    from feedbax.application.bundle import new_application_registry_bundle
    from feedbax.web.services.project_folder import ProjectFolderService

    train = completed_train(host, project, attempt, successor)
    folders = ProjectFolderService(paths.web / "projects")
    plan = folders.read_plan(project, revision=revision)
    jobs = JobService(host, new_application_registry_bundle())
    result = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=evaluation)
    rollout = read_job(host, project, evaluation)
    train_ref = ContentObjectRef.model_validate(train["outcome"]["store_success"])
    rollout_ref = ContentObjectRef.model_validate(rollout["outcome"]["store_success"])
    view = SuccessView(ImmutableArtifactBlobProvider(plan.store_path))
    receipt = view.read_receipt(rollout_ref)
    assert receipt.attempt == evaluation
    (bound,) = receipt.inputs["subject"]
    assert bound.evaluation_subject is not None
    assert bound.evaluation_subject.terminal_receipt == train_ref
    with result.open_states(verify=True) as dataset:
        position = dataset.descriptor.attrs["trial_record"]["positions"][0]
        samples = dataset.select(position, labels={"trial": trial})[0, 0].tolist()
    return {"receipt": result.receipt.model_dump(mode="json"), "samples": samples}


ACTIONS = {
    "declare-variants": NonSeed(lambda paths, host, *args: declare_variants(paths, *args),
                                "edits the already saved project at a checked revision"),
    "checkpoint": NonSeed(stopped_checkpoint, "reads a stopped Host attempt and its checkpoint"),
    "restart": NonSeed(restart, "restarts the isolated Host"),
    "resume": NonSeed(lambda paths, host, *args: resume(host, *args), "resumes an existing Host attempt"),
    "completed": NonSeed(lambda paths, host, *args: completed_receipt(host, *args), "reads a completed Train receipt"),
    "result": NonSeed(result, "reads a retained Host result"),
}


def main():
    action, home, project, attempt, *args = sys.argv[1:]
    root = Path(home).resolve(strict=True)
    paths = AppPaths.resolve(instance_root=root)
    paths.configure_studio(desktop=False)
    host = HostClient(paths.host)
    record = ACTIONS[action].command(paths, host, project, attempt, *args)
    print(json.dumps(record, allow_nan=False))


if __name__ == "__main__":
    main()
