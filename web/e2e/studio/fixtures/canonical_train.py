"""Ordinary folder inputs and read-only custody evidence for the mounted Train test."""

from copy import deepcopy
import json
from pathlib import Path
import sys

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.manifest import canonical_manifest_path, load_manifest
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import TrainingRequestSpec
from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient
from feedbax.host.plan_history import plan_run_key
from feedbax.job_completion import verify_bound_training_completion
from feedbax.jobs import JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
from feedbax.studio.training_requests import lower_studio_training_runs
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from web.e2e.studio.fixtures.reaching_project import reaching_project


def authored_files():
    project = reaching_project(batches=2)
    (run,) = lower_studio_training_runs(
        project.workspace, project.graph, stage_id="stage:train",
        registry_bundle=new_application_registry_bundle(),
    )
    spec = run.spec
    payload = TrainingRequestSpec(
        graph=spec.graph.inline,
        task=spec.task,
        task_binding_spec=spec.task_binding_spec,
        objective=spec.objective,
        training_config=spec.training_config,
        method_ref=spec.method_ref,
        method_payload=spec.method_payload,
    ).model_dump(mode="json", exclude_none=True)
    files = []
    for name, schema, field in (
        ("model", "feedbax.spec.graph", "graph"),
        ("task", "feedbax.spec.task", "task"),
    ):
        files.append({
            "kind": "document", "id": name,
            "document": {"schema": schema, "payload": payload[field]},
        })
        payload[field] = {"document": name}
    files.extend([
        {"kind": "step", "id": "train", "step": {"kind": "train", "base": {"inline": payload}}},
        {"kind": "step", "id": "successor", "step": {
            "kind": "train", "base": {"step": "train"},
            "inputs": {"checkpoint": {"step": "train", "checkpoint": "last"}},
            "deltas": [{
                "layer_id": "initialize",
                "patches": [
                    {
                        "path": "method_payload.payload.checkpoint_initialization",
                        "op": "add",
                        "value": "initialize_from",
                    },
                    {"path": "method_payload.payload.target_total", "value": 1200},
                    {"path": "training_config.n_batches", "value": 1200},
                ],
            }],
        }},
    ])
    return files


def last_train_interval(events, attempt: str):
    return next(
        event["data"]["metric_interval"]
        for event in reversed(events)
        if event["job_id"] == attempt
        and event["kind"] == "progress"
        and "metric_interval" in event["data"]
    )


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
    verify_bound_training_completion(child_completion)
    assert child_completion["producer_job_id"] == successor_attempt
    assert child_completion["request_id"] == child.request_id
    joined_stopped, joined_child = view.attempts_by_run[successor]
    assert joined_child.job_id == successor_attempt
    assert joined_stopped.job_id == stopped_attempt
    assert joined_child.checkpoint_source == joined_stopped.checkpoint_source
    assert joined_child.successful_receipt.model_dump(mode="json") == child_completion["receipt"]
    assert view.history[successor].result.receipt == joined_child.successful_receipt

    child_manifest = load_manifest(canonical_manifest_path(
        "TrainingRunManifest", stopped["outcome"]["receipt"]["manifest_id"], root=view.settled.store_path,
    ))
    initialization = child_manifest.provenance.metadata["checkpoint_initialization"]
    pointer = completion["checkpoint"]["pointer"]
    assert initialization["mode"] == "initialize_from"
    assert initialization["source"]["id"] == pointer["transaction_id"]
    assert initialization["source"]["metadata"]["manifest_sha256"] == pointer["manifest_sha256"]

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
            "initialization": initialization,
            "resumed_from": stopped["checkpoint"]["completed_updates"],
            "terminal_loss": terminal_loss,
        },
        "admitted_plan": admitted_plan,
        "checkpoint_input": checkpoint.reference.model_dump(mode="json"),
    }


def history_files():
    model, task, train, _ = authored_files()
    sibling = deepcopy(train)
    sibling["id"] = "sibling"
    sibling["step"]["base"]["inline"]["method_payload"]["payload"]["seed"] = 91
    return [model, task, train, sibling]


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
        verify_bound_training_completion(completion)
        manifest = load_manifest(canonical_manifest_path(
            "TrainingRunManifest", completion["receipt"]["manifest_id"], root=Path(completion["manifest_root"]),
        ))
        initialization = manifest.provenance.metadata["checkpoint_initialization"]
        transform = initialization["transformation"]
        pointer = source["checkpoint"]["pointer"]
        producer_source = transform["producer_source"]
        assert producer_source["id"] == pointer["transaction_id"]
        assert producer_source["metadata"]["manifest_sha256"] == pointer["manifest_sha256"]
        interval = last_train_interval(events, attempt)
        result[attempt] = {
            "producer": producer,
            "receipt": completion["receipt"],
            "completed_updates": completion["checkpoint"]["completed_updates"],
            "initialization": {"mode": initialization["mode"], **transform},
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
    output = {}
    for role, result in (("current", history.result), ("previous", history.previous)):
        assert result is not None
        job = jobs[result.job_id]
        completion = job["outcome"]["completion"]
        verify_bound_training_completion(completion)
        assert completion["request_id"] == result.request_id
        assert completion["receipt"] == result.receipt.model_dump(mode="json")
        output[role] = result.model_dump(mode="json")
    ref = view.jobs.plan_ref
    assert ref is not None
    store = ImmutableArtifactBlobProvider(view.settled.store_path)
    output["admitted_plan"] = json.loads(store.get_object(ContentObjectRef.model_validate(ref["plan"])))
    return output


if __name__ == "__main__":
    action, *arguments = sys.argv[1:]
    if action == "files" and not arguments:
        print(json.dumps(authored_files()))
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
