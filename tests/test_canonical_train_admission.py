"""The saved Plan admits an exact Train request to the real local daemon and worker."""

import json
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
from textwrap import dedent

import pytest

from feedbax.contracts.checkpoint_projection import ContinuationEncoding
from feedbax.contracts.jobs import StudioSubmissionActor
from feedbax.contracts.manifest import canonical_manifest_path, load_manifest
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import NamedMember, ProductRef
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef, read_store_closure_ref
from feedbax.contracts.studio_project import StudioProjectCreateFile, StudioProjectPatch
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.host import HostError, Job, Launch
from feedbax.host.completion import CompletionFileError
from feedbax.job_completion import verify_bound_training_completion, verify_training_model_product
from feedbax.jobs import JobConflict, JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView, run_ref_name
from feedbax.persistence.method_checkpoint import read_method_checkpoint_manifest
from feedbax.persistence.receipt_closure import ReceiptClosure
from feedbax.persistence.store_sync import sync_store
from feedbax.plugins.composition import load_application
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput, verify_training_request_binding
from feedbax.validity.evidence import evidence_objects
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.managed_train_fixture import managed_train_payload
from tests.support.host import admit, held_until_released, job_row, restart, run_job, token, wait_until


@pytest.mark.round(
    "The Host dies inside the terminal Store commit of a real Train, and the next Host "
    "republishes it over the committed receipt, or fails it when the committed state is "
    "damaged."
)
@pytest.mark.parametrize("damage_committed_state", [False, True], ids=["recover", "corrupt"])
def test_saved_train_admission_uses_exact_plan_request_through_terminal_custody(
    private_host, tmp_path, monkeypatch, damage_committed_state,
):
    host = private_host
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    registries = load_application().bundle
    jobs = JobService(host, registries)
    folders = ProjectFolderService(
        tmp_path / "projects", components=registries.components,
        functions=registries.expression_functions, python_tasks=registries.python_tasks,
        training_methods=registries.training_methods,
    )
    snapshot = folders.new().snapshot
    project = snapshot.project_id
    payload = managed_train_payload()
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
    assert not reviewed.store_path.exists()

    host.policy(compute=1)
    release = tmp_path / "release"
    gate = run_job(
        "gate",
        held_until_released(release),
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
    assert launch.spec == decoded.spec

    edited = folders.patch(
        project,
        StudioProjectPatch.model_validate(
            {
                "revision": snapshot.revision,
                "address": {"kind": "step", "id": "train"},
                "patches": [
                    {
                        "op": "add",
                        "path": "/base/inline/seed",
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
    assert "payload.seed" in rejected.reason
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

    # Interrupt the Store-to-Host commit, including a missing run ref after index commit.
    probe = tmp_path / "terminal-probe"
    probe.mkdir()
    marker = probe / "committed"
    (probe / "sitecustomize.py").write_text(dedent(f"""\
        import os
        from pathlib import Path
        from feedbax.host.store import Store
        from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
        from feedbax.persistence.effective_success import SuccessView

        finished = Store.finished
        publish_ref = SuccessView._publish_run_ref

        def interrupt(self, row, outcome):
            marker = Path({str(marker)!r})
            if {damage_committed_state!r} and row['id'] == {attempt!r} and 'store_success' in outcome and not marker.exists():
                store = ImmutableArtifactBlobProvider(Path({str(reviewed.store_path)!r}))
                receipt = SuccessView(store).read_attempt(row['id']).receipt
                store.object_path(receipt.continuation.state.extra.object).unlink()
                marker.touch()
                os._exit(81)
            return finished(self, row, outcome)

        def interrupt_ref(self, request_id, *args, **kwargs):
            marker = Path({str(marker)!r})
            if not {damage_committed_state!r} and request_id == {request_id!r} and not marker.exists():
                marker.touch()
                os._exit(81)
            return publish_ref(self, request_id, *args, **kwargs)

        Store.finished = interrupt
        SuccessView._publish_run_ref = interrupt_ref
        """))
    repo = Path(__file__).resolve().parents[1]
    with monkeypatch.context() as env:
        env.setenv("PYTHONPATH", os.pathsep.join((str(probe), str(repo), os.environ.get("PYTHONPATH", ""))))
        restart(host)
        release.touch()
        def interrupted():
            if marker.exists():
                return True
            try:
                row = job_row(host, attempt)
            except HostError as error:
                if error.code != "disconnected":
                    raise
                return marker.exists()
            if row and row["state"] in {"failed", "blocked", "stopped"}:
                pytest.fail(str(row["outcome"]))
            return False

        wait_until(interrupted)
        if not damage_committed_state:
            committed = SuccessView(store).read_attempt(attempt)
            assert committed is not None
            assert store.read_ref(run_ref_name(request_id)) is None

        def recovered():
            try:
                return job_row(host, attempt, "failed" if damage_committed_state else "succeeded")
            except HostError as error:
                if error.code != "disconnected":
                    raise
                return None

        finished = wait_until(recovered)
    if damage_committed_state:
        assert finished["delivery"] == 1
        assert finished["outcome"]["error"]["code"] == "publication_failed"
        return
    assert finished["delivery"] == 1 and finished["request"] == request_id
    root = read_store_closure_ref(store.read_ref(run_ref_name(request_id))).roots[0]
    assert root.receipt == committed.ref
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
    assert manifest.provenance.metadata["request_id"] == launch.request_id
    assert manifest.provenance.metadata["binding_sha256"] == completion["binding_sha256"]
    binding = reviewed.store_path / "checkpoints" / attempt / "binding.json"
    assert BoundStudioTrainingJobInput.model_validate_json(binding.read_bytes()) == launch
    assert sha256(binding.read_bytes()).hexdigest() == completion["binding_sha256"]

    (extra,) = [
        reviewed.store_path / file["path"] for file in completion["files"] if "/blobs/" in file["path"]
    ]
    latest = binding.with_name("latest.json")
    metric_log = reviewed.store_path / completion["metrics"]["path"]
    held_metric_log = metric_log.with_suffix(".held")
    with ExitStack() as held:
        metric_log.rename(held_metric_log)
        held.callback(held_metric_log.rename, metric_log)
        verify_bound_training_completion(completion)
        for source in (extra, latest):
            temporary = source.with_suffix(".held")
            source.rename(temporary)
            held.callback(temporary.rename, source)
        product, model = verify_training_model_product(completion)
        with pytest.raises(CompletionFileError):
            verify_bound_training_completion(completion)

    assert product.name == "model" and product.content.tree
    assert (product.content.sha256, product.content.size) == (
        model.tree.descriptor.sha256, model.tree.descriptor.size_bytes,
    )
    assert product.descriptor == model.encoding.model_dump(mode="json")
    tree = model.provider.get_tree(model.tree)
    success = SuccessView(store).read_attempt(attempt)
    assert success == committed
    assert success.ref.model_dump(mode="json") == finished["outcome"]["store_success"]
    assert success.receipt.products["checkpoint"] == (product,)
    (metrics,) = success.receipt.products["metrics"]
    history = [json.loads(line) for line in (reviewed.store_path / completion["metrics"]["path"]).read_text().splitlines()]
    assert [row["update"] for row in history] == [1, 2]
    assert metrics.content.sha256 == completion["metrics"]["sha256"]
    request = success.receipt.request
    assert store.get_object(request.object) == encoded
    assert {ref.sha256: ref for ref in request.documents} == refs
    selected = StoreClosureRef(roots=(ReceiptRoot(
        receipt=success.ref,
        product=ProductRef(request_id=request_id, role="checkpoint", select=NamedMember(member="model")),
    ),))
    packet_store = ImmutableArtifactBlobProvider(tmp_path / "selected-model-packet")
    destination = ImmutableArtifactBlobProvider(tmp_path / "selected-model-destination")
    sync_store(store, packet_store, selected, destination_ref="packets/model")
    sync_store(packet_store, destination, "packets/model", destination_ref="packets/model")
    packet_ref = destination.read_ref("packets/model")
    assert packet_ref is not None
    packet = read_store_closure_ref(packet_ref)
    expected = (
        success.ref, request.object, *request.documents, *evidence_objects(store, success.receipt.trace),
        model.tree.descriptor, *(member.object for member in tree.files),
    )
    transferred = ReceiptClosure(destination).walk(packet)
    assert {ref.sha256 for ref in transferred} == {ref.sha256 for ref in expected}
    excluded = {file["sha256"] for file in completion["files"] if "/blobs/" in file["path"]}
    assert excluded.isdisjoint(path.name for path in destination.root.glob("artifacts/sha256/*/*"))
    for ref in transferred:
        destination.verify_object(ref)

    altered = launch.model_copy(update={"spec": launch.spec.model_copy(update={"seed": 512})})
    with pytest.raises(ValueError, match="admitted execution settings"):
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


def _saved_step(folders, snapshot, name, step):
    return folders.create_file(
        snapshot.project_id,
        StudioProjectCreateFile.model_validate(
            {
                "revision": snapshot.revision,
                "file": {"kind": "step", "id": name, "step": step},
            }
        ),
    ).snapshot


def _checkpoint(outcome):
    checkpoint = PublishedTrainingCheckpoint.model_validate(outcome["checkpoint"])
    return read_method_checkpoint_manifest(checkpoint.checkpoint_root, checkpoint.pointer)


_CHAIN_UPDATES = 64


def _saved_chain(folders):
    snapshot = folders.new().snapshot
    payload = managed_train_payload(batches=_CHAIN_UPDATES)
    payload["training_config"]["snapshot_interval"] = 16
    snapshot = _saved_step(
        folders,
        snapshot,
        "prefix",
        {
            "kind": "train",
            "base": {"inline": payload},
        },
    )
    snapshot = _saved_step(
        folders,
        snapshot,
        "child",
        {
            "kind": "train",
            "base": {"step": "prefix"},
            "inputs": {"checkpoint": {"step": "prefix", "checkpoint": "last"}},
            "deltas": [
                {
                    "layer_id": "initialize",
                    "patches": [
                        {
                            "path": "checkpoint_initialization",
                            "op": "add",
                            "value": "initialize_from",
                        },
                        {"path": "seed", "op": "add", "value": 7},
                        {
                            "path": "settings.optimizer.learning_rate.value",
                            "value": 0.0005,
                        },
                    ],
                }
            ],
        },
    )
    prefix = RunKey("default", "prefix", "default")
    child = RunKey("default", "child", "default")
    reviewed = folders.read_plan(snapshot.project_id, revision=snapshot.revision)
    assert not reviewed.plan.diagnostics, reviewed.plan.diagnostics
    return reviewed, prefix, child


def test_saved_train_successor_waits_for_and_reuses_exact_completion(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    registries = load_application().bundle
    jobs = JobService(host, registries)
    folders = ProjectFolderService(
        tmp_path / "projects", components=registries.components,
        functions=registries.expression_functions, python_tasks=registries.python_tasks,
        training_methods=registries.training_methods,
    )
    reviewed, prefix, child = _saved_chain(folders)
    project = reviewed.project_id
    plan_rows = {row.key: row for row in reviewed.plan.runs}
    source_input = plan_rows[child].inputs["checkpoint"]
    assert source_input.producer == prefix

    host.policy(compute=1)
    actor = StudioSubmissionActor(client="cli", label="original chain")
    submitted = jobs.admit_plan(folders, reviewed, run_keys=(child, prefix), token=token("chain"), actor=actor)
    child_item, parent_item = submitted.items
    assert parent_item.disposition == child_item.disposition == "admitted"
    prefix_id, child_id = parent_item.job_id, child_item.job_id
    prefix_running = wait_until(lambda: job_row(host, prefix_id, "running"))
    assert prefix_running["request"] == plan_rows[prefix].request_id
    waiting = job_row(host, child_id, "queued")
    assert waiting["delivery"] == 0 and not waiting["edges"][0]["satisfied"]
    assert waiting["edges"][0]["job_id"] == prefix_id

    jobs.stop([prefix_id], "checkpoint")
    stopped_prefix = wait_until(lambda: job_row(host, prefix_id, "stopped"))
    assert 0 < stopped_prefix["outcome"]["checkpoint"]["completed_updates"] < _CHAIN_UPDATES
    waiting = job_row(host, child_id, "queued")
    assert waiting["delivery"] == 0 and waiting["edges"][0]["binding"] is None

    restart(host)
    assert jobs.admit_plan(folders, reviewed, run_keys=(child, prefix), token=token("chain")) == submitted
    prefix_resume = jobs.resume(prefix_id, token=token("resume-prefix"))
    assert prefix_resume.actor == StudioSubmissionActor(client="studio")
    attempts = {job.job_id: job for job in jobs.jobs(project=project).jobs}
    assert attempts[prefix_id].actor == actor
    succeeded = wait_until(lambda: job_row(host, prefix_resume.job_id, "succeeded"))
    producer = succeeded["outcome"]["completion"]
    verify_bound_training_completion(producer)
    assert producer["checkpoint"]["completed_updates"] == _CHAIN_UPDATES
    child_running = wait_until(lambda: job_row(host, child_id, "running"))
    assert child_running["request"] == plan_rows[child].request_id
    assert child_running["edges"][0]["binding"] == {
        "dependency_job_id": prefix_id,
        "completion": producer,
    }

    jobs.stop([child_id], "checkpoint")
    stopped_child = wait_until(lambda: job_row(host, child_id, "stopped"))
    child_outcome = stopped_child["outcome"]
    child_checkpoint = _checkpoint(child_outcome)
    assert 0 < child_checkpoint.clock.completed_updates < _CHAIN_UPDATES
    assert child_checkpoint.clock.seed == 7 and child_checkpoint.clock.segment_start == 0
    manifest = load_manifest(canonical_manifest_path(
        "TrainingRunManifest", child_outcome["receipt"]["manifest_id"], root=reviewed.store_path,
    ))
    initialization = manifest.provenance.metadata["checkpoint_initialization"]
    assert initialization["id"] == producer["checkpoint"]["pointer"]["transaction_id"]
    assert initialization["metadata"]["manifest_sha256"] == producer["checkpoint"]["pointer"]["manifest_sha256"]

    restart(host)
    resumed = jobs.resume(child_id, token=token("resume-child"))
    assert jobs.resume(child_id, token=token("resume-child")).job_id == resumed.job_id
    child_done = wait_until(lambda: job_row(host, resumed.job_id, "succeeded"))
    verify_bound_training_completion(child_done["outcome"]["completion"])
    store = SuccessView(ImmutableArtifactBlobProvider(reviewed.store_path))
    parent_success = store.read_attempt(prefix_resume.job_id)
    child_success = store.read_attempt(resumed.job_id)
    assert parent_success is not None and child_success is not None
    assert store.read_attempt(prefix_id) is None and store.read_attempt(child_id) is None
    (bound_input,) = child_success.receipt.inputs["checkpoint"]
    assert bound_input.selection.receipt == parent_success.ref
    assert bound_input.content == parent_success.receipt.products["checkpoint"][0].content
    assert child_done["edges"][0]["binding"] == child_running["edges"][0]["binding"]
    (bound,) = [
        event
        for event in host.events(projects=(project,))["events"]
        if event["job_id"] == child_id and event["kind"] == "dependency_bound"
    ]
    assert bound["data"] == child_done["edges"][0]["binding"]
    resumed_checkpoint = _checkpoint(child_done["outcome"])
    assert child_success.receipt.recovery.transaction_id == child_checkpoint.transaction_id
    uninterrupted = jobs.admit_plan(folders, reviewed, run_keys=(child,), token=token("control"))
    (control_item,) = uninterrupted.items
    control = wait_until(lambda: job_row(host, control_item.job_id, "succeeded"))
    assert _checkpoint(control["outcome"]).state_sha256 == resumed_checkpoint.state_sha256


@pytest.mark.parametrize("state_policy", ["preserve", "reset"])
def test_saved_stateful_train_fork(host, tmp_path, monkeypatch, state_policy):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    registries = load_application().bundle
    jobs = JobService(host, registries)
    folders = ProjectFolderService(
        tmp_path / "projects", components=registries.components,
        functions=registries.expression_functions, python_tasks=registries.python_tasks,
        training_methods=registries.training_methods,
    )
    snapshot = folders.new().snapshot
    source_payload = managed_train_payload(batches=4)
    source_payload["seed"] = 12
    snapshot = _saved_step(folders, snapshot, "prefix", {
        "kind": "train", "base": {"inline": source_payload},
    })
    preserve = state_policy == "preserve"
    child = deepcopy(source_payload)
    child["seed"] = 19
    child["training_config"]["n_batches"] = 3
    child["checkpoint_initialization"] = {
        "mode": "continue_from", "source_completed_updates": 4, "state": state_policy,
        "schedule": "restart" if preserve else "continue",
        "rng": "carry" if preserve else "reset",
    }
    child["settings"]["optimizer"]["learning_rate"]["value"] = 0.0005
    if not preserve:
        child["settings"]["optimizer"]["type"] = "adamw"
        child["trial_documents"]["model"]["interior"]["components"]["cell"]["fields"]["trainable"] = False
    snapshot = _saved_step(folders, snapshot, "child", {
        "kind": "train", "base": {"inline": child},
        "inputs": {"checkpoint": {"step": "prefix", "checkpoint": "last"}},
    })
    incompatible = deepcopy(child)
    incompatible["checkpoint_initialization"]["state"] = "preserve"
    incompatible["settings"]["optimizer"]["type"] = "sgd"
    snapshot = _saved_step(folders, snapshot, "incompatible", {
        "kind": "train", "base": {"inline": incompatible},
        "inputs": {"checkpoint": {"step": "prefix", "checkpoint": "last"}},
    })
    reviewed = folders.read_plan(snapshot.project_id, revision=snapshot.revision)
    keys = tuple(RunKey("default", name, "default") for name in ("prefix", "child", "incompatible"))
    submitted = jobs.admit_plan(folders, reviewed, run_keys=keys, token=token("stateful"))
    _, child_attempt, incompatible_attempt = (item.job_id for item in submitted.items)

    def settled(job_id):
        def read():
            row = job_row(host, job_id)
            return row if row and row["state"] in {"succeeded", "failed", "blocked", "stopped"} else None

        return wait_until(read)

    finished = settled(child_attempt)
    assert finished["state"] == "succeeded", finished["outcome"]
    checkpoint = _checkpoint(finished["outcome"])
    assert checkpoint.clock.segment_start == 4
    assert checkpoint.clock.completed_updates == 7
    assert checkpoint.clock.schedule_origin == (4 if preserve else 0)
    assert checkpoint.clock.seed == (12 if preserve else 19)
    store = SuccessView(ImmutableArtifactBlobProvider(reviewed.store_path))
    (bound,) = store.read_attempt(child_attempt).receipt.inputs["checkpoint"]
    projection = ContinuationEncoding.model_validate(bound.descriptor)
    assert (projection.state is not None) == preserve
    assert projection.seed == (12 if preserve else None)

    incompatible_result = settled(incompatible_attempt)
    assert incompatible_result["state"] == "failed", incompatible_result["outcome"]
    assert "checkpoint" in incompatible_result["outcome"]["error"]["message"].lower()
