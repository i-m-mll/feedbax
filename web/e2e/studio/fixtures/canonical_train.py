"""Ordinary folder inputs and read-only custody evidence for the mounted Train test."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient
from feedbax.host.plan_history import plan_run_key
from feedbax.job_completion import verify_bound_training_completion
from feedbax.jobs import JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.method_checkpoint import read_method_checkpoint_manifest
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.support.managed_train_fixture import managed_train_project


def successor_file():
    return {"kind": "step", "id": "successor", "step": {
        "kind": "train", "base": {"step": "train"},
        "inputs": {"checkpoint": {"step": "train", "checkpoint": "last"}},
        "deltas": [{
            "layer_id": "initialize",
            "patches": [
                {
                    "path": "checkpoint_initialization",
                    "op": "add",
                    "value": "initialize_from",
                },
                {"path": "training_config.n_batches", "value": 1200},
            ],
        }],
    }}


def ordinary_files(batches: str, *rates: str):
    example = managed_train_project(batches=int(batches))
    step = example.steps["train"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    if rates:
        step["sweep"] = {
            "axes": {"rate": {"values": {
                f"rate-{index}": {"label": rate, "patches": {
                    "settings.optimizer.learning_rate.value": {"value": float(rate)},
                }}
                for index, rate in enumerate(rates)
            }}},
        }
    trial = step["base"]["inline"]["trial_documents"]
    names = (trial[slot]["document"] for slot in ("model", "task", "objective"))
    files = [
        {"kind": "document", "id": name, "document": example.documents[name].model_dump(
            mode="json", by_alias=True, exclude_unset=True,
        )}
        for name in names
    ]
    files.append({"kind": "step", "id": "train", "step": step})
    return files


def seed_variants(home: str, project_id: str):
    *documents, train = ordinary_files("2")
    project = Project.model_validate({
        "documents": {file["id"]: file["document"] for file in documents},
        "steps": {"train": train["step"]},
        "sweep": {"rows": {
            "base": {"label": "Baseline"},
            "compare": {"from": "base", "label": "Comparison", "value": "comparison"},
            "control": {"from": "base", "label": "Control"},
        }},
    })
    root = AppPaths.resolve(instance_root=Path(home)).web / "projects" / project_id
    for name, data in dump_text_project(
        project, ProjectConfiguration(project=project_id, store="results"),
    ).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return ProjectFolderService(root.parent).open(project_id).snapshot.project_id


def last_train_interval(events, attempt: str):
    return next(
        event["data"]["metric_interval"]
        for event in reversed(events)
        if event["job_id"] == attempt
        and event["kind"] == "progress"
        and "metric_interval" in event["data"]
    )


def train_success(job, store):
    ref = ContentObjectRef.model_validate(job["outcome"]["store_success"])
    receipt = SuccessView(store).read_owned_success(ref).receipt
    assert (receipt.attempt, receipt.request.id) == (job["id"], job["request"])
    return ref, receipt


def observe(
    home: Path, project: str, revision: str, attempt: str, stopped_attempt: str, successor_attempt: str,
):
    paths = AppPaths.resolve(instance_root=home)
    folders = ProjectFolderService(paths.web / "projects")
    host = HostClient(paths.host)
    service = ProjectPlanService(folders, JobService(host, new_application_registry_bundle()))
    train = RunKey("default", "train", "default")
    successor = RunKey("default", "successor", "default")
    view = service.read(project, revision=revision, source_runs=(train, successor))
    plan = view.settled.plan
    rows = {run.key: run for run in plan.runs}
    assert not plan.diagnostics
    snapshot = host.jobs_snapshot(project=project, include_launch=True, start_host=False)
    jobs = {job["id"]: job for job in snapshot["jobs"]}
    assert set(jobs) == {attempt, stopped_attempt, successor_attempt}
    job = jobs[attempt]
    assert (job["id"], job["request"], job["delivery"]) == (attempt, rows[train].request_id, 1)
    launch = BoundStudioTrainingJobInput.model_validate(job["launch"]["payload"])
    assert launch.canonical_request is not None and launch.resume is None
    store = ImmutableArtifactBlobProvider(view.settled.store_path)
    assert launch.manifest_root == view.settled.store_path
    assert store.get_object(launch.canonical_request.request) == plan.requests[launch.request_id]
    assert all(
        store.get_object(ref) == plan.documents[ref.sha256]
        for ref in launch.canonical_request.documents
    )
    completion = job["outcome"]["completion"]
    verify_bound_training_completion(completion)
    assert completion["request_id"] == launch.request_id
    assert completion["producer_job_id"] == attempt
    (joined,) = view.attempts_by_run[train]
    assert joined.job_id == attempt
    assert joined.successful_receipt.model_dump(mode="json") == completion["receipt"]
    child = rows[successor]
    assert child.inputs["checkpoint"].producer == train
    decoded = decode_request(plan.requests[child.request_id], plan.documents.__getitem__)
    (checkpoint,) = decoded.inputs["checkpoint"]
    assert checkpoint.reference.request_id == launch.request_id
    child_job = jobs[successor_attempt]
    assert child_job["request"] == child.request_id
    stopped = jobs[stopped_attempt]
    assert (stopped["state"], child_job["continues"]) == ("stopped", stopped_attempt)
    assert child_job["launch"]["payload"]["resume"] == stopped["checkpoint"]
    (edge,) = child_job["edges"]
    assert edge["satisfied"] and edge["authenticated_completion"]
    assert edge["binding"] == {"dependency_job_id": attempt, "completion": completion}
    assert stopped["edges"] == child_job["edges"]
    child_completion = child_job["outcome"]["completion"]
    child_binding = verify_bound_training_completion(child_completion)
    assert child_completion["producer_job_id"] == successor_attempt
    assert child_completion["request_id"] == child.request_id
    joined_stopped, joined_child = view.attempts_by_run[successor]
    assert joined_child.job_id == successor_attempt
    assert joined_stopped.job_id == stopped_attempt
    assert joined_child.checkpoint_source == joined_stopped.checkpoint_source
    assert joined_child.successful_receipt.model_dump(mode="json") == child_completion["receipt"]
    assert view.history[successor].result.receipt == joined_child.successful_receipt

    predecessor = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
    stopped_checkpoint = PublishedTrainingCheckpoint.model_validate(stopped["checkpoint"])
    assert child_binding.spec.checkpoint_initialization == "initialize_from"
    assert child_binding.checkpoint_input is not None
    assert child_binding.checkpoint_input.checkpoint == predecessor
    assert child_binding.resume == stopped_checkpoint
    source_ref, _ = train_success(job, store)
    _, child_receipt = train_success(child_job, store)
    (source,) = child_receipt.inputs["checkpoint"]
    assert source.selection is not None and source.selection.receipt == source_ref
    assert child_receipt.recovery is not None
    assert (child_receipt.recovery.run_id, child_receipt.recovery.manifest_sha256) == (
        stopped_attempt, stopped_checkpoint.pointer.manifest_sha256,
    )

    ref = view.jobs.plan_ref
    assert ref is not None
    admitted_plan = json.loads(store.get_object(ContentObjectRef.model_validate(ref["plan"])))
    assert admitted_plan == {
        "revision": revision,
        "semantic_root": view.settled.semantic_root,
        "requests": {
            plan_run_key(key.variant, key.step, key.row): row.request_id for key, row in rows.items()
        },
    }
    terminal_loss = last_train_interval(
        host.events(projects=(project,), limit=1024)["events"], successor_attempt,
    )
    return {
        "project": project, "revision": revision, "semantic_root": view.settled.semantic_root,
        "request": launch.request_id, "attempt": attempt, "delivery": job["delivery"],
        "completion_updates": completion["checkpoint"]["completed_updates"],
        "receipt": completion["receipt"], "joined_status": joined.status,
        "worker_request_object": launch.canonical_request.request.sha256,
        "successor": {
            "request": child.request_id, "attempt": successor_attempt, "delivery": child_job["delivery"],
            "completion_updates": child_completion["checkpoint"]["completed_updates"],
            "receipt": child_completion["receipt"], "joined_status": joined_child.status,
            "initialization": child_binding.spec.checkpoint_initialization,
            "resumed_from": stopped["checkpoint"]["completed_updates"],
            "terminal_loss": terminal_loss,
        },
        "admitted_plan": admitted_plan,
        "checkpoint_input": checkpoint.reference.model_dump(mode="json"),
    }


def history_files():
    *documents, train = ordinary_files("2")
    sibling = deepcopy(train)
    sibling["id"] = "sibling"
    sibling["step"]["base"]["inline"]["seed"] = 91
    return [*documents, train, sibling]


def observe_swept(home: Path, project: str, successors: list[str]):
    host = HostClient(AppPaths.resolve(instance_root=home).host)
    jobs = {job["id"]: job for job in host.jobs_snapshot(project=project, start_host=False)["jobs"]}
    events = host.events(projects=(project,), limit=1024)["events"]
    result = {}
    for attempt in successors:
        job = jobs[attempt]
        (edge,) = job["edges"]
        assert edge["satisfied"] and edge["authenticated_completion"]
        source = edge["binding"]["completion"]
        producer = source["producer_job_id"]
        assert source == jobs[producer]["outcome"]["completion"]
        completion = job["outcome"]["completion"]
        assert (completion["producer_job_id"], completion["request_id"]) == (attempt, job["request"])
        verify_bound_training_completion(source)
        binding = verify_bound_training_completion(completion)
        policy = binding.spec.continuation
        assert policy is not None and binding.checkpoint_input is not None
        predecessor = PublishedTrainingCheckpoint.model_validate(source["checkpoint"])
        assert binding.checkpoint_input.checkpoint == predecessor
        assert policy.source_completed_updates == predecessor.completed_updates
        store = ImmutableArtifactBlobProvider(binding.manifest_root)
        source_ref, _ = train_success(jobs[producer], store)
        _, receipt = train_success(job, store)
        (selected,) = receipt.inputs["checkpoint"]
        assert selected.selection is not None and selected.selection.receipt == source_ref
        checkpoint = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
        manifest = read_method_checkpoint_manifest(checkpoint.checkpoint_root, checkpoint.pointer)
        assert manifest.clock.segment_start == predecessor.completed_updates
        interval = last_train_interval(events, attempt)
        result[attempt] = {
            "producer": producer,
            "receipt": completion["receipt"],
            "completed_updates": manifest.clock.completed_updates,
            "source_checkpoint": {
                "completed_updates": predecessor.completed_updates,
                "transaction_id": predecessor.pointer.transaction_id,
                "manifest_sha256": predecessor.pointer.manifest_sha256,
            },
            "schedule_origin": manifest.clock.schedule_origin,
            "initialization": policy.model_dump(mode="json"),
            "last_loss": interval["metrics"]["train_loss"]["last"],
        }
    return result


def observe_history(home: Path, project: str, revision: str):
    paths = AppPaths.resolve(instance_root=home)
    folders = ProjectFolderService(paths.web / "projects")
    host = HostClient(paths.host)
    service = ProjectPlanService(folders, JobService(host, new_application_registry_bundle()))
    view = service.read(project, revision=revision)
    history = view.history[RunKey("default", "train", "default")]
    snapshot = host.jobs_snapshot(project=project, start_host=False)
    jobs = {job["id"]: job for job in snapshot["jobs"]}
    events = host.events(projects=(project,), limit=1024)["events"]
    output = {}
    loss = {}
    for role, result in (("current", history.result), ("previous", history.previous)):
        if result is None:
            output[role] = None
            continue
        job = jobs[result.job_id]
        completion = job["outcome"]["completion"]
        verify_bound_training_completion(completion)
        assert completion["request_id"] == result.request_id
        assert completion["producer_job_id"] == result.job_id
        assert completion["receipt"] == result.receipt.model_dump(mode="json")
        output[role] = result.model_dump(mode="json")
        assert job["continues"] is None and result.progress is not None
        assert completion["checkpoint"]["completed_updates"] == result.progress.completed_updates
        loss[role] = [
            {"update": interval["end_inclusive"], "value": interval["metrics"]["train_loss"]["mean"]}
            for event in events
            if event["job_id"] == result.job_id and event["kind"] == "progress"
            and (interval := event["data"].get("metric_interval")) is not None
        ]
        assert loss[role][-1]["update"] == result.progress.completed_updates
    output["loss"] = loss
    ref = view.jobs.plan_ref
    assert ref is not None
    store = ImmutableArtifactBlobProvider(view.settled.store_path)
    output["admitted_plan"] = json.loads(store.get_object(ContentObjectRef.model_validate(ref["plan"])))
    return output


def run_saved_cli(home: str, folder_name: str, label: str) -> None:
    paths = AppPaths.resolve(instance_root=Path(home))
    paths.configure_studio(desktop=False)
    project = paths.web / "projects" / folder_name
    os.execv(sys.executable, [sys.executable, "-m", "feedbax.cli", "run", str(project), "train", "--label", label])


if __name__ == "__main__":
    action, *arguments = sys.argv[1:]
    if action == "managed-files" and not arguments:
        print(json.dumps([*ordinary_files("2"), successor_file()]))
    elif action == "ordinary-files":
        print(json.dumps(ordinary_files(*arguments)))
    elif action == "run-saved-cli":
        run_saved_cli(*arguments)
    elif action == "seed-variants":
        print(json.dumps(seed_variants(*arguments)))
    elif action == "history-files" and not arguments:
        print(json.dumps(history_files()))
    elif action == "observe":
        home, project, revision, attempt, stopped_attempt, successor_attempt = arguments
        print(json.dumps(observe(Path(home), project, revision, attempt, stopped_attempt, successor_attempt)))
    elif action == "observe-history":
        home, project, revision = arguments
        print(json.dumps(observe_history(Path(home), project, revision)))
    elif action == "observe-swept":
        home, project, *successors = arguments
        print(json.dumps(observe_swept(Path(home), project, successors)))
    else:
        raise ValueError(f"Unknown canonical Train fixture action {action!r}")
