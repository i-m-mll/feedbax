"""The saved Plan admits an exact Train request to the real local daemon and worker."""

import json
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
import pytest

from feedbax.contracts.base import ParentRef
from feedbax.contracts.checkpoints import CheckpointLatestPointer
from feedbax.contracts.manifest import canonical_manifest_path, load_manifest
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import EncodedRequest, decode_request
from feedbax.contracts.request_refs import NamedMember, ProductRef
from feedbax.contracts.request_specs import TrainingRequestSpec
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef, read_store_closure_ref
from feedbax.contracts.studio_project import StudioProjectCreateFile, StudioProjectPatch
from feedbax.contracts.training import OptimizerSpec, TrainingRunSpec
from feedbax.host import Job, Launch
from feedbax.host.completion import CompletionFileError
from feedbax.job_completion import verify_bound_training_completion, verify_training_model_product
from feedbax.jobs import JobConflict, JobService
from feedbax.orchestration.schedule_eval import (
    LEARNING_RATE_SCHEDULE_ID,
    learning_rate_from_build_optimizer,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView, store_request
from feedbax.persistence.receipt_closure import ReceiptClosure
from feedbax.persistence.store_sync import sync_store
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_train import (
    BoundStudioTrainingJobInput,
    verify_training_request_binding,
)
from feedbax.studio.training_requests import lower_studio_training_runs
from feedbax.training.checkpoint_custody import (
    CheckpointCustodyError,
    bind_published_checkpoint_content,
    materialize_checkpoint_content,
    require_checkpoint_semantic_state_equal,
    resolve_checkpoint_custody_ref,
)
from feedbax.training.invocation import prepare_training_invocation
from feedbax.training.standard_train_checkpoint import STANDARD_TRAIN_STATE_MEDIA_TYPE
from feedbax.training.studio_executor import studio_training_key
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_local_job_host import admit, job_row, restart, run_job, wait_until
from tests.test_local_job_host import host as host_fixture
from web.e2e.studio.fixtures.reaching_project import reaching_project


host = host_fixture


def _train_payload(registries, *, batches):
    reaching = reaching_project(batches=batches)
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
    return legacy, payload


def test_saved_train_admission_uses_exact_plan_request_through_terminal_custody(
    host, tmp_path, monkeypatch
):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    registries = new_application_registry_bundle()
    jobs = JobService(host, registries)
    folders = ProjectFolderService(tmp_path / "projects")
    snapshot = folders.new().snapshot
    project = snapshot.project_id
    legacy, payload = _train_payload(registries, batches=2)
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

    siblings = {
        Path(file["path"]).name.split("-", 1)[0]: reviewed.store_path / file["path"]
        for file in completion["files"]
        if "/blobs/" in file["path"]
    }
    assert set(siblings) == {"optimizer", "prng", "batch_counter"}
    latest = binding.with_name("latest.json")
    with ExitStack() as held:
        for source in (*siblings.values(), latest):
            temporary = source.with_suffix(".held")
            source.rename(temporary)
            held.callback(temporary.rename, source)
        product, model = verify_training_model_product(completion)
        with pytest.raises(CompletionFileError, match="checkpoint slot"):
            verify_bound_training_completion(completion)

    assert product.name == "model" and product.content.tree
    assert (product.content.sha256, product.content.size) == (
        model.tree.descriptor.sha256, model.tree.descriptor.size_bytes,
    )
    assert product.descriptor == model.encoding.model_dump(mode="json")
    tree = model.provider.get_tree(model.tree)
    assert {member.relative_path for member in tree.files} == {"graph.json", "model.safetensors"}
    source = ImmutableArtifactBlobProvider(tmp_path / "selected-model-source")
    destination = ImmutableArtifactBlobProvider(tmp_path / "selected-model-destination")
    sync_store(
        model.provider, source, StoreClosureRef(roots=(model.tree,)),
        destination_ref="proof/model-source",
    )
    request = store_request(source, EncodedRequest(
        request_id, encoded, {digest: store.get_object(ref) for digest, ref in refs.items()},
    ))
    success = SuccessView(source).publish(request, attempt=attempt, products={"checkpoint": (product,)})
    selected = StoreClosureRef(roots=(ReceiptRoot(
        receipt=success.ref,
        product=ProductRef(request_id=request_id, role="checkpoint", select=NamedMember(member="model")),
    ),))
    sync_store(source, destination, selected, destination_ref="proof/selected-model")
    packet_ref = destination.read_ref("proof/selected-model")
    assert packet_ref is not None
    packet = read_store_closure_ref(packet_ref)
    expected = (
        success.ref, request.object, *request.documents, model.tree.descriptor,
        *(member.object for member in tree.files),
    )
    transferred = ReceiptClosure(destination).walk(packet)
    assert {ref.sha256 for ref in transferred} == {ref.sha256 for ref in expected}
    excluded = {file["sha256"] for file in completion["files"] if "/blobs/" in file["path"]}
    assert excluded.isdisjoint(path.name for path in destination.root.glob("artifacts/sha256/*/*"))
    for ref in transferred:
        destination.verify_object(ref)

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
    checkpoint = outcome["checkpoint"]
    pointer = checkpoint["pointer"]
    resolved = resolve_checkpoint_custody_ref(
        ParentRef(
            kind="TrainingCheckpointTransactionManifest",
            id=pointer["transaction_id"],
            role="training_checkpoint_custody",
            uri=pointer["manifest_relative_path"],
            metadata={"manifest_sha256": pointer["manifest_sha256"]},
        ),
        allowed_root=Path(checkpoint["checkpoint_root"]),
    )
    return resolved, resolved.slots


_CHAIN_UPDATES = 64


def _saved_chain(folders, registries):
    snapshot = folders.new().snapshot
    _, payload = _train_payload(registries, batches=_CHAIN_UPDATES)
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
                            "path": "method_payload.payload.checkpoint_initialization",
                            "op": "add",
                            "value": "initialize_from",
                        },
                        {"path": "method_payload.payload.seed", "op": "add", "value": 7},
                        {
                            "path": "method_payload.payload.optimizer.params.learning_rate",
                            "value": 0.0005,
                        },
                        {"path": "training_config.learning_rate", "value": 0.0005},
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
    registries = new_application_registry_bundle()
    jobs = JobService(host, registries)
    folders = ProjectFolderService(tmp_path / "projects")
    reviewed, prefix, child = _saved_chain(folders, registries)
    project = reviewed.project_id
    plan_rows = {row.key: row for row in reviewed.plan.runs}
    source_input = plan_rows[child].inputs["checkpoint"]
    assert source_input.producer == prefix

    host.policy(compute=1)
    submitted = jobs.admit_plan(folders, reviewed, run_keys=(child, prefix), token="chain")
    child_item, parent_item = submitted.items
    assert parent_item.disposition == child_item.disposition == "admitted"
    prefix_id, child_id = parent_item.job_id, child_item.job_id
    prefix_running = wait_until(lambda: job_row(host, prefix_id, "running"))
    assert prefix_running["request"] == plan_rows[prefix].request_id
    waiting = job_row(host, child_id, "queued")
    assert waiting["delivery"] == 0 and not waiting["edges"][0]["satisfied"]
    assert waiting["edges"][0]["job_id"] == prefix_id

    jobs.stop([prefix_id], "checkpoint")
    stopped_prefix = wait_until(lambda: job_row(host, prefix_id, "stopped"), timeout=22)
    assert 0 < stopped_prefix["outcome"]["checkpoint"]["completed_updates"] < _CHAIN_UPDATES
    waiting = job_row(host, child_id, "queued")
    assert waiting["delivery"] == 0 and waiting["edges"][0]["binding"] is None

    restart(host)
    assert jobs.admit_plan(folders, reviewed, run_keys=(child, prefix), token="chain") == submitted
    prefix_resume = jobs.resume(prefix_id, token="resume-prefix")
    succeeded = wait_until(lambda: job_row(host, prefix_resume.job_id, "succeeded"), timeout=22)
    producer = succeeded["outcome"]["completion"]
    verify_bound_training_completion(producer)
    assert producer["checkpoint"]["completed_updates"] == _CHAIN_UPDATES
    child_running = wait_until(lambda: job_row(host, child_id, "running"), timeout=22)
    assert child_running["request"] == plan_rows[child].request_id
    assert child_running["edges"][0]["binding"] == {
        "dependency_job_id": prefix_id,
        "completion": producer,
    }

    jobs.stop([child_id], "checkpoint")
    stopped_child = wait_until(lambda: job_row(host, child_id, "stopped"), timeout=22)
    child_outcome = stopped_child["outcome"]
    child_checkpoint, slots = _checkpoint(child_outcome)
    steps = child_outcome["checkpoint"]["completed_updates"]
    assert 0 < steps < _CHAIN_UPDATES
    assert int(slots["batch_counter"]) == steps
    assert int(slots["optimizer"][1][0].count) == steps
    key = studio_training_key(7, "update")
    for _ in range(steps):
        key, _ = jr.split(key)
    np.testing.assert_array_equal(slots["prng"], key)
    (child_launch,) = host.jobs(job_id=child_id, include_launch=True)
    native = BoundStudioTrainingJobInput.model_validate(child_launch["launch"]["payload"]).spec
    preparation = prepare_training_invocation(
        native, registries, run_id="carried-model-control", resume=False,
    )
    parent_checkpoint, parent_slots = _checkpoint(succeeded["outcome"])

    def readout(model):
        return model.nodes["readout"].layer.weight

    assert not np.array_equal(
        readout(preparation.initial_slots["model"]),
        readout(parent_slots["model"]),
    )
    expected = dict(preparation.initial_slots) | {"model": parent_slots["model"]}
    kernel = preparation.kernel_context["studio_graph_update"]
    for _ in range(steps):
        expected = kernel.reported_update(expected)
    for slot in ("model", "optimizer", "prng", "batch_counter"):
        actual_leaves = jax.tree.leaves(slots[slot])
        expected_leaves = jax.tree.leaves(expected[slot])
        for actual_value, expected_value in zip(actual_leaves, expected_leaves, strict=True):
            if isinstance(expected_value, jax.Array):
                np.testing.assert_array_equal(actual_value, expected_value)

    manifest = load_manifest(
        canonical_manifest_path(
            "TrainingRunManifest",
            child_outcome["receipt"]["manifest_id"],
            root=reviewed.store_path,
        )
    )
    initialization = manifest.provenance.metadata["checkpoint_initialization"]
    assert initialization["mode"] == "initialize_from"
    assert initialization["source"]["id"] == producer["checkpoint"]["pointer"]["transaction_id"]
    assert (
        initialization["source"]["metadata"]["manifest_sha256"]
        == (producer["checkpoint"]["pointer"]["manifest_sha256"])
    )

    restart(host)
    resumed = jobs.resume(child_id, token="resume-child")
    assert jobs.resume(child_id, token="resume-child").job_id == resumed.job_id
    child_done = wait_until(lambda: job_row(host, resumed.job_id, "succeeded"), timeout=22)
    verify_bound_training_completion(child_done["outcome"]["completion"])
    assert child_done["edges"][0]["binding"] == child_running["edges"][0]["binding"]
    (bound,) = [
        event
        for event in host.events(projects=(project,))["events"]
        if event["job_id"] == child_id and event["kind"] == "dependency_bound"
    ]
    assert bound["data"] == child_done["edges"][0]["binding"]
    resumed_checkpoint, _ = _checkpoint(child_done["outcome"])
    assert resumed_checkpoint.manifest.parent_lineage[0].transaction_id == (
        child_checkpoint.manifest.transaction_id
    )

    uninterrupted = jobs.admit_plan(folders, reviewed, run_keys=(child,), token="control")
    (control_item,) = uninterrupted.items
    control = wait_until(lambda: job_row(host, control_item.job_id, "succeeded"), timeout=22)
    control_checkpoint, _ = _checkpoint(control["outcome"])
    for slot in ("model", "optimizer", "prng", "batch_counter"):
        require_checkpoint_semantic_state_equal(
            control_checkpoint.manifest,
            resumed_checkpoint.manifest,
            slot,
        )

    release = tmp_path / "release-tamper"
    gate = run_job(
        "tamper-gate",
        f"from pathlib import Path\nimport time\nwhile not Path({str(release)!r}).exists(): time.sleep(.01)",
        project=project,
    )
    admit(host, "tamper-gate", gate)
    wait_until(lambda: job_row(host, gate.id, "running"))
    tampered = jobs.admit_plan(folders, reviewed, run_keys=(child,), token="tamper")
    (tampered_item,) = tampered.items

    def pinned():
        row = job_row(host, tampered_item.job_id, "queued")
        return row and row["edges"][0]["binding"]

    assert wait_until(pinned) == child_running["edges"][0]["binding"]
    pointer = CheckpointLatestPointer.model_validate(producer["checkpoint"]["pointer"])
    root = Path(producer["checkpoint"]["checkpoint_root"])
    transaction = (root / pointer.manifest_relative_path).parent
    unneeded = {
        slot.slot: transaction / slot.relative_path
        for slot in parent_checkpoint.manifest.slots if slot.slot != "model"
    }
    assert set(unneeded) == {"optimizer", "prng", "batch_counter"}
    assert all(
        slot.media_type == STANDARD_TRAIN_STATE_MEDIA_TYPE
        for slot in parent_checkpoint.manifest.slots if slot.slot in unneeded
    )
    held = {original: original.with_suffix(".held") for original in unneeded.values()}
    for original, moved in held.items():
        original.rename(moved)
    try:
        verify_bound_training_completion(producer, scope="model_only")
        with pytest.raises(CompletionFileError):
            verify_bound_training_completion(producer)
        content = ImmutableArtifactBlobProvider(reviewed.store_path)
        _, projection = bind_published_checkpoint_content(
            root, pointer, content, scope="model_only",
        )
        assert projection.root_kind == "checkpoint-model-custody"
        tree_paths = {file.relative_path for file in content.get_tree(projection.tree).files}
        assert pointer.manifest_relative_path in tree_paths
        assert any("/semantic-model/" in path for path in tree_paths)
        assert all(original.relative_to(root).as_posix() not in tree_paths for original in held)
        with materialize_checkpoint_content(
            projection, content, owner="successor-control", scope="model_only",
        ) as staged:
            assert (staged / pointer.manifest_relative_path).is_file()
            assert all(not (staged / original.relative_to(root)).exists() for original in held)
        with pytest.raises(CheckpointCustodyError, match="full custody"):
            with materialize_checkpoint_content(projection, content, owner="resume-control"):
                pytest.fail("Full Resume received a model-only projection")
        release.touch()
        blocked = wait_until(lambda: job_row(host, tampered_item.job_id, "blocked"))
        assert blocked["outcome"]["error"]["code"] == "dependency_corrupt"
        assert blocked["delivery"] == 0
    finally:
        for original, moved in held.items():
            moved.rename(original)
    assert jobs.admit_plan(folders, reviewed, run_keys=(child,), token="tamper") == tampered


@pytest.mark.parametrize("optimizer_policy", ["preserve", "reset"])
def test_saved_stateful_train_fork(host, tmp_path, monkeypatch, optimizer_policy):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    registries = new_application_registry_bundle()
    jobs = JobService(host, registries)
    folders = ProjectFolderService(tmp_path / "projects")
    snapshot = folders.new().snapshot
    boundary, additional = 4, 16
    rate = 0.001
    _, source_payload = _train_payload(registries, batches=boundary)
    source_payload["graph"]["nodes"]["readout"]["params"]["trainable"] = False
    source_method = source_payload["method_payload"]["payload"]
    del source_method["optimizer"]["params"]["learning_rate"]
    source_method["optimizer"]["lr_schedule"] = {
        "kind": "constant", "origin": {"kind": "run_start"}, "learning_rate_0": rate,
    }
    snapshot = _saved_step(
        folders, snapshot, "prefix", {"kind": "train", "base": {"inline": source_payload}},
    )
    child_payload = deepcopy(source_payload)
    child_method = child_payload["method_payload"]["payload"]
    child_method["target_total"] = boundary + additional
    progress = {"source_completed_batches": boundary, "additional_batches": additional}
    policy = {
        "mode": "continue_from", "progress": progress, "optimizer": optimizer_policy,
        "schedule": "continue", "rng": "carry", "cursor": "carry",
    }
    preserve = optimizer_policy == "preserve"
    if preserve:
        policy["schedule"] = "restart"
        child_method["optimizer"]["lr_schedule"] = {
            "kind": "delayed_cosine", "origin": {"kind": "segment_start"},
            "learning_rate_0": rate, "total_steps": additional, "cosine_annealing_alpha": 0.2,
        }
        window = {}
        for i, name in enumerate(("boundary", "first_update", "second_update")):
            window[name] = learning_rate_from_build_optimizer(
                OptimizerSpec.model_validate(child_method["optimizer"]),
                sample_step=boundary + i, schedule_origin_step=boundary,
                current_step=boundary, optimizer_count_at_current_step=boundary,
            )
        progress["schedule_discontinuity_exemptions"] = [{
            "schedule_id": LEARNING_RATE_SCHEDULE_ID, "boundary_batch": boundary,
            "expected_source_state": dict.fromkeys(window, window["boundary"]),
            "expected_target_state": window, "intended_first_update_behavior": "decrease",
            "reason": "Restart a segment-local cosine at the selected checkpoint.",
        }]
    else:
        child_payload["graph"]["nodes"]["readout"]["params"]["trainable"] = True
        child_payload["graph"]["nodes"]["cell"]["params"]["trainable"] = False
        child_method["optimizer"]["type"] = "adamw"
        child_method["seed"] = 19
        policy["rng"] = "reset"
    child_method["checkpoint_initialization"] = policy
    child_payload["training_config"].update(n_batches=additional, batch_size=3, snapshot_interval=8)
    snapshot = _saved_step(folders, snapshot, "child", {
        "kind": "train", "base": {"inline": child_payload},
        "inputs": {"checkpoint": {"step": "prefix", "checkpoint": "last"}},
    })
    invalid_payload = deepcopy(child_payload)
    invalid_payload["method_payload"]["payload"]["optimizer"]["type"] = "sgd" if preserve else "adam"
    if not preserve:
        invalid_payload["method_payload"]["payload"]["checkpoint_initialization"]["optimizer"] = "preserve"
    snapshot = _saved_step(folders, snapshot, "incompatible", {
        "kind": "train", "base": {"inline": invalid_payload},
        "inputs": {"checkpoint": {"step": "prefix", "checkpoint": "last"}},
    })
    prefix, child, incompatible = (
        RunKey("default", name, "default") for name in ("prefix", "child", "incompatible")
    )
    reviewed = folders.read_plan(snapshot.project_id, revision=snapshot.revision)
    assert not reviewed.plan.diagnostics, reviewed.plan.diagnostics

    def await_state(job_id, state):
        def read():
            row = job_row(host, job_id)
            if row and row["state"] in {state, "succeeded", "stopped", "failed", "blocked"}:
                return row
            return None

        row = wait_until(read, timeout=18)
        assert row["state"] == state, row.get("outcome")
        return row

    host.policy(compute=1)
    submitted = jobs.admit_plan(folders, reviewed, run_keys=(prefix, child), token="stateful")
    prefix_id, child_id = (item.job_id for item in submitted.items)
    source_done = await_state(prefix_id, "succeeded")
    producer = source_done["outcome"]["completion"]
    source_custody, source_slots = _checkpoint(source_done["outcome"])
    verify_bound_training_completion(producer)
    running = await_state(child_id, "running")
    assert running["edges"][0]["binding"]["completion"] == producer
    jobs.stop([child_id], "checkpoint")
    stopped = await_state(child_id, "stopped")["outcome"]
    stopped_custody, stopped_slots = _checkpoint(stopped)
    updates = stopped["checkpoint"]["completed_updates"]
    assert boundary < updates < boundary + additional
    assert stopped_custody.manifest.segment_lineage.start_batch == boundary
    assert stopped_custody.manifest.completed_coordinate.schedule_origin_step == (
        boundary if preserve else 0
    )
    assert int(stopped_slots["batch_counter"]) == updates

    (launched,) = host.jobs(job_id=child_id, include_launch=True)
    spec = BoundStudioTrainingJobInput.model_validate(launched["launch"]["payload"]).spec
    prepared = prepare_training_invocation(spec, registries, run_id="stateful-control", resume=False)
    expected = dict(prepared.initial_slots)
    if preserve:
        expected.update(source_slots)
    else:
        def parameters(graph):
            return graph.nodes["cell"].cell, graph.nodes["readout"].layer
        expected["model"] = eqx.tree_at(parameters, expected["model"], parameters(source_slots["model"]))
        expected["batch_counter"] = source_slots["batch_counter"]
    kernel = prepared.kernel_context["studio_graph_update"]
    for _ in range(boundary, updates):
        expected = kernel.reported_update(expected)
    for slot in ("model", "optimizer", "prng", "batch_counter"):
        for actual, wanted in zip(jax.tree.leaves(stopped_slots[slot]), jax.tree.leaves(expected[slot]), strict=True):
            np.testing.assert_array_equal(actual, wanted)
    count = int(stopped_slots["optimizer"][-1].count)
    assert count == (updates if preserve else updates - boundary)
    if preserve:
        position = updates - boundary - 1
        expected_rate = rate * (0.2 + 0.8 * (1 + np.cos(np.pi * position / additional)) / 2)
        np.testing.assert_allclose(stopped_slots["optimizer"][-1].hyperparams["learning_rate"], expected_rate)
    else:
        for actual, wanted in zip(
            jax.tree.leaves(stopped_slots["model"].nodes["cell"]),
            jax.tree.leaves(source_slots["model"].nodes["cell"]), strict=True,
        ):
            np.testing.assert_array_equal(actual, wanted)
        assert not np.array_equal(
            stopped_slots["model"].nodes["readout"].layer.weight,
            source_slots["model"].nodes["readout"].layer.weight,
        )

    manifest = load_manifest(canonical_manifest_path(
        "TrainingRunManifest", stopped["receipt"]["manifest_id"], root=reviewed.store_path,
    ))
    initialization = manifest.provenance.metadata["checkpoint_initialization"]
    transform = initialization["transformation"]
    assert initialization["mode"] == "continue_from"
    assert transform["producer_source"]["id"] == source_custody.manifest.transaction_id
    assert (
        transform["producer_source"]["metadata"]["manifest_sha256"]
        == producer["checkpoint"]["pointer"]["manifest_sha256"]
    )
    assert transform["producer_content_binding"]["root_kind"] == "checkpoint-custody"
    bootstrap = resolve_checkpoint_custody_ref(
        ParentRef.model_validate(initialization["source"]).model_copy(update={
            "uri": f"transactions/{transform['fork_transaction_id']}/manifest.json",
        }),
        allowed_root=Path(stopped["checkpoint"]["checkpoint_root"]),
    )
    assert bootstrap.manifest.completed_training_batches == boundary
    assert bootstrap.manifest.metadata["optimizer_step"] == (boundary if preserve else 0)
    assert int(bootstrap.slots["optimizer"][-1].count) == (boundary if preserve else 0)
    assert (transform["optimizer_policy"], transform["rng_policy"], transform["cursor_policy"]) == (
        optimizer_policy, policy["rng"], "carry",
    )
    assert {item["slot"] for item in transform["accounting"] if item["disposition"] == "deliberately_reset"} == (
        set() if preserve else {"optimizer", "prng"}
    )

    restart(host)
    resumed = jobs.resume(child_id, token="resume-stateful")
    resumed_done = await_state(resumed.job_id, "succeeded")["outcome"]
    verify_bound_training_completion(resumed_done["completion"])
    resumed_custody, _ = _checkpoint(resumed_done)
    assert resumed_custody.manifest.parent_lineage[0].transaction_id == stopped_custody.manifest.transaction_id

    (control,) = jobs.admit_plan(folders, reviewed, run_keys=(child,), token="control").items
    control_done = await_state(control.job_id, "succeeded")["outcome"]
    verify_bound_training_completion(control_done["completion"])
    control_custody, _ = _checkpoint(control_done)
    for slot in ("model", "optimizer", "prng", "batch_counter"):
        require_checkpoint_semantic_state_equal(control_custody.manifest, resumed_custody.manifest, slot)

    (bad,) = jobs.admit_plan(folders, reviewed, run_keys=(incompatible,), token="incompatible").items
    refusal = await_state(bad.job_id, "failed")["outcome"]
    assert "/slots/optimizer" in json.dumps(refusal)
    assert not refusal.get("checkpoint")
