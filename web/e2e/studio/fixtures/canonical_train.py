"""Ordinary folder inputs and read-only custody evidence for the mounted Train test."""

from copy import deepcopy
import json
from pathlib import Path
import sys

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import TrainingRequestSpec
from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient
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
        }},
    ])
    return files


def observe(home: Path, project: str, revision: str, attempt: str):
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
    (job,) = snapshot["jobs"]
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
    assert successor not in view.attempts_by_run
    child = rows[successor]
    assert child.inputs["checkpoint"].producer == train
    decoded = decode_request(plan.requests[child.request_id], plan.documents.__getitem__)
    (checkpoint,) = decoded.inputs["checkpoint"]
    assert checkpoint.reference.request_id == launch.request_id
    return {
        "project": project, "revision": revision, "semantic_root": view.settled.semantic_root,
        "request": launch.request_id, "attempt": attempt, "delivery": job["delivery"],
        "completion_updates": completion["checkpoint"]["completed_updates"],
        "receipt": completion["receipt"], "joined_status": joined.status,
        "worker_request_object": launch.canonical_request.request.sha256,
        "successor": child.request_id, "checkpoint_input": checkpoint.reference.model_dump(mode="json"),
    }


def history_files():
    model, task, train, _ = authored_files()
    sibling = deepcopy(train)
    sibling["id"] = "sibling"
    sibling["step"]["base"]["inline"]["method_payload"]["payload"]["seed"] = 91
    return [model, task, train, sibling]


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
        home, project, revision, attempt = arguments
        print(json.dumps(observe(Path(home), project, revision, attempt)))
    elif action == "observe-history":
        home, project, revision = arguments
        print(json.dumps(observe_history(Path(home), project, revision)))
    else:
        raise ValueError(f"Unknown canonical Train fixture action {action!r}")
