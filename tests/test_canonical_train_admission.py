"""The saved Plan admits an exact Train request to the real local daemon and worker."""

import json
from hashlib import sha256

import pytest

from feedbax.contracts.manifest import canonical_manifest_path, load_manifest
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import TrainingRequestSpec
from feedbax.contracts.studio_project import StudioProjectCreateFile, StudioProjectPatch
from feedbax.contracts.training import TrainingRunSpec
from feedbax.host import Job, Launch
from feedbax.host.completion import CompletionFileError
from feedbax.job_completion import verify_bound_training_completion
from feedbax.jobs import JobConflict, JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_train import (
    BoundStudioTrainingJobInput,
    verify_training_request_binding,
)
from feedbax.studio.training_requests import lower_studio_training_runs
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_local_job_host import admit, job_row, restart, run_job, wait_until
from tests.test_local_job_host import host as host_fixture
from web.e2e.studio.fixtures.reaching_project import reaching_project


host = host_fixture


def test_saved_train_admission_uses_exact_plan_request_through_terminal_custody(
    host, tmp_path, monkeypatch
):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    registries = new_application_registry_bundle()
    jobs = JobService(host, registries)
    folders = ProjectFolderService(tmp_path / "projects")
    snapshot = folders.new().snapshot
    project = snapshot.project_id
    reaching = reaching_project(batches=2)
    (legacy,) = lower_studio_training_runs(
        reaching.workspace, reaching.graph, stage_id="stage:train", registry_bundle=registries
    )
    payload = TrainingRequestSpec(
        graph=legacy.spec.graph.inline,
        task=legacy.spec.task,
        task_binding_spec=legacy.spec.task_binding_spec,
        objective=legacy.spec.objective,
        method_ref=legacy.spec.method_ref,
        method_payload=legacy.spec.method_payload,
        training_config=legacy.spec.training_config,
    ).model_dump(mode="json", exclude_none=True)
    for name in ("train", "stable"):
        snapshot = folders.create_file(
            project,
            StudioProjectCreateFile.model_validate(
                {
                    "revision": snapshot.revision,
                    "file": {
                        "kind": "step",
                        "id": name,
                        "step": {"kind": "train", "base": {"inline": payload}},
                    },
                }
            ),
        ).snapshot

    train = RunKey("default", "train", "default")
    stable = RunKey("default", "stable", "default")
    reviewed = folders.read_plan(project, revision=snapshot.revision, source_runs=(train, stable))
    assert not reviewed.plan.diagnostics
    ids = {row.key: row.request_id for row in reviewed.plan.runs}
    request_id = ids[train]
    assert request_id == ids[stable] and request_id.startswith("train:")
    assert request_id != legacy.request_id
    assert not reviewed.store_path.exists()

    host.policy(compute=1)
    release = tmp_path / "release"
    gate = run_job(
        "gate",
        f"from pathlib import Path\nimport time\nwhile not Path({str(release)!r}).exists(): time.sleep(.01)",
        project=project,
    )
    admit(host, "gate", gate)
    wait_until(lambda: job_row(host, gate.id, "running"))
    submitted = jobs.admit_plan(folders, reviewed, run_keys=(train,), token="saved-train")
    (item,) = submitted.items
    assert item.disposition == "admitted"
    attempt = item.job_id
    (row,) = host.jobs(job_id=attempt, include_launch=True)
    assert row["request"] == request_id and row["state"] == "queued"
    launch = BoundStudioTrainingJobInput.model_validate(row["launch"]["payload"])
    assert launch.request_id == request_id and launch.manifest_root == reviewed.store_path
    assert launch.canonical_request is not None
    store = ImmutableArtifactBlobProvider(reviewed.store_path)
    encoded = store.get_object(launch.canonical_request.request)
    assert encoded == reviewed.plan.requests[request_id]
    refs = {ref.sha256: ref for ref in launch.canonical_request.documents}
    assert all(
        store.get_object(ref) == reviewed.plan.documents[digest] for digest, ref in refs.items()
    )
    decoded = decode_request(encoded, lambda digest: store.get_object(refs[digest]))
    assert decoded.id == request_id

    edited = folders.patch(
        project,
        StudioProjectPatch.model_validate(
            {
                "revision": snapshot.revision,
                "address": {"kind": "step", "id": "train"},
                "patches": [
                    {
                        "op": "add",
                        "path": "/base/inline/method_payload/payload/seed",
                        "value": 7,
                    }
                ],
            }
        ),
    )
    current = folders.read_plan(project, revision=edited.revision)
    current_ids = {row.key: row.request_id for row in current.plan.runs}
    assert current_ids[train] != request_id and current_ids[stable] == request_id
    assert jobs.admit_plan(folders, reviewed, run_keys=(train,), token="saved-train") == submitted
    with pytest.raises(JobConflict, match="another action"):
        jobs.admit_plan(folders, reviewed, run_keys=(stable,), token="saved-train")
    stale = jobs.admit_plan(folders, reviewed, run_keys=(train, stable), token="old-plan")
    rejected, attached = stale.items
    assert rejected.disposition == "rejected" and rejected.job_id is None
    assert (
        rejected.reason
        == "The selected Train request changed at payload.method_payload.payload.seed."
    )
    assert attached.disposition == "attached" and attached.job_id == attempt

    foreign_store = tmp_path / "foreign-store"
    foreign_launch = Launch(
        **(
            row["launch"]
            | {
                "payload": row["launch"]["payload"] | {"manifest_root": str(foreign_store)},
            }
        )
    )
    foreign = Job("foreign", project, request_id, foreign_launch)
    foreign_response = admit(host, "foreign-store", foreign)
    (foreign_item,) = foreign_response["items"]
    assert foreign_item["disposition"] == "rejected" and foreign_item["job_id"] is None
    assert "different store" in foreign_item["reason"]

    release.touch()
    finished = wait_until(lambda: job_row(host, attempt, "succeeded"), timeout=26)
    assert finished["delivery"] == 1 and finished["request"] == request_id
    completion = finished["outcome"]["completion"]
    assert completion["request_id"] == request_id and completion["producer_job_id"] == attempt
    assert completion["checkpoint"]["completed_updates"] == 2
    verify_bound_training_completion(completion)
    receipt = completion["receipt"]
    path = canonical_manifest_path(
        "TrainingRunManifest", receipt["manifest_id"], root=reviewed.store_path
    )
    assert sha256(path.read_bytes()).hexdigest() == receipt["manifest_sha256"]
    manifest = load_manifest(path)
    assert manifest.job_id == attempt
    assert TrainingRunSpec.model_validate(manifest.training_spec.inline) == launch.spec
    binding = reviewed.store_path / "checkpoints" / attempt / "binding.json"
    assert BoundStudioTrainingJobInput.model_validate_json(binding.read_bytes()) == launch
    assert sha256(binding.read_bytes()).hexdigest() == completion["binding_sha256"]

    execution = launch.spec.worker_execution.model_copy(update={"metadata": {"tampered": True}})
    for altered, error in (
        (launch.model_copy(update={"request_id": current_ids[train]}), "canonical Train request"),
        (
            launch.model_copy(
                update={
                    "spec": launch.spec.model_copy(update={"on_nan": "halt_restore_checkpoint"})
                }
            ),
            "bound execution payload",
        ),
        (
            launch.model_copy(
                update={"spec": launch.spec.model_copy(update={"worker_execution": execution})}
            ),
            "bound execution payload",
        ),
        (launch.model_copy(update={"manifest_root": foreign_store}), "object"),
    ):
        with pytest.raises(ValueError, match=error):
            verify_training_request_binding(altered)

    document, *_ = launch.canonical_request.documents
    object_path = store.object_path(document)
    moved_object = object_path.with_suffix(".held")
    object_path.rename(moved_object)
    try:
        with pytest.raises(CompletionFileError, match="Cannot authenticate canonical Train object"):
            verify_bound_training_completion(completion)
    finally:
        moved_object.rename(object_path)

    restart(host)
    reopened = JobService(host, registries)
    assert (
        reopened.admit_plan(folders, reviewed, run_keys=(train,), token="saved-train") == submitted
    )
    (admitted,) = [job for job in host.jobs(project=project) if job["id"] != gate.id]
    assert admitted["id"] == attempt and admitted["delivery"] == 1
    (terminal,) = [
        event
        for event in host.events(projects=(project,))["events"]
        if event["job_id"] == attempt and event["kind"] == "terminal"
    ]
    assert terminal["request"] == request_id and terminal["data"]["receipt"] == receipt
    (raw_submission,) = host.submissions(token="saved-train")
    assert raw_submission["record"]["command"]["requests"][item.run_key] == request_id
    assert json.loads(item.run_key) == ["default", "train", "default"]
