"""Saved no-plugin Train-to-Evaluate through the local Host and dataset custody."""

from hashlib import sha256
import json

import equinox as eqx
import numpy as np
import pytest

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.contracts.dataset import DatasetProductRef
from feedbax.contracts.jobs import StudioSavedEvaluationReplayRequest, StudioSavedTrainKey
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_specs import TrainingRequestSpec
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint, StudioGraphMethodPayload
from feedbax.jobs import JobConflict, JobNotFound, JobService
from feedbax.persistence.array_container import ArrayContainerError, ArrayContainerReader
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_evaluate import ROLLOUT_TYPE, STATES_SCHEMA, state_key
from feedbax.training import evaluate_worker
from feedbax.training.checkpoint_custody import open_published_model_slot
from feedbax.training.studio_executor import studio_training_key
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from feedbax.web.worker.execution import prototypes_from_task_bindings
from tests.test_local_job_host import admit, host as host_fixture, job_row, restart, run_job, wait_until
from tests.test_studio_training_host_worker import _authored_run

host = host_fixture


def authored_evaluation_project():
    spec = _authored_run(batches=2, sampled_hold=True).spec
    train_payload = TrainingRequestSpec(
        graph=spec.graph.inline, task=spec.task, task_binding_spec=spec.task_binding_spec,
        objective=spec.objective, method_ref=spec.method_ref, method_payload=spec.method_payload,
        training_config=spec.training_config,
    ).model_dump(mode="json", exclude_none=True)
    step = {
        "kind": "evaluate",
        "base": {"inline": {
            "evaluation_type": ROLLOUT_TYPE,
            "params": {
                "seed": 731,
                "trials": ["alpha", "beta", "gamma"],
                "trial_policy": "paired",
                "retention": "observations",
                "variation": [],
            },
        }},
        "inputs": {"subject": {"step": "train", "checkpoint": "last"}},
    }
    sweep = {"axes": {"mass": {"values": {
        name: {"value": mass, "patches": {"params.variation": {"value": [
            {"path": "nodes.plant.params.mass", "op": "replace", "value": mass},
        ]}}}
        for name, mass in (("nominal", 1), ("heavy", 2))
    }}}}
    authored = Project.model_validate({"steps": {
        "train": {"kind": "train", "base": {"inline": train_payload}},
        "baseline": step,
        "conditions": step | {"sweep": sweep},
        "unsupported-condition": step | {"base": {"inline": {
            "evaluation_type": ROLLOUT_TYPE,
            "params": step["base"]["inline"]["params"] | {
                "variation": [{"path": "nodes.plant.params.damping", "op": "replace", "value": 1}],
            },
        }}},
        "unsupported": step | {
            "inputs": {"subject": {"step": "train", "checkpoint": {"best_by": "loss"}}},
        },
    }})
    return authored, spec


def test_saved_train_to_evaluate_with_paired_conditions(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    registry = new_application_registry_bundle()
    jobs = JobService(host, registry, enable_evaluation_candidate=True)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "reaching"
    authored, spec = authored_evaluation_project()
    for name, data in dump_text_project(
        authored, ProjectConfiguration(project=project, store="results"),
    ).items():
        path = folders.root / project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    plan = folders.read_plan(project, revision=None)
    assert not plan.plan.diagnostics
    by_key = {run.key: run for run in plan.plan.runs}
    train = RunKey("default", "train", "default")
    baseline = RunKey("default", "baseline", "default")
    conditions = {
        run.coordinates["mass"].value_id: run.key
        for run in plan.plan.runs if run.key.step == "conditions"
    }
    assert conditions.keys() == {"nominal", "heavy"}
    keys = (baseline, conditions["nominal"], conditions["heavy"])
    assert len({by_key[key].request_id for key in keys}) == 3
    assert all(by_key[key].request_id.startswith("evaluate:") for key in keys)

    def launch(service, rows, token):
        return service.admit_plan(folders, plan, run_keys=rows, token=token)

    gated = JobService(host, registry)
    (disabled,) = launch(gated, (baseline,), "evaluate-held").items
    assert disabled.disposition == "rejected"
    assert disabled.reason.startswith("cannot_run /evaluation_type")
    unsupported = RunKey("default", "unsupported", "default")
    (selector,) = launch(jobs, (unsupported,), "unsupported-selector").items
    assert selector.disposition == "rejected"
    assert selector.reason.startswith("cannot_run /inputs/subject/checkpoint")
    (missing,) = launch(jobs, (baseline,), "missing-subject").items
    assert missing.disposition == "rejected"
    assert missing.reason.startswith("cannot_run /inputs/subject")

    host.policy(compute=1)
    release = tmp_path / "release"
    gate = run_job(
        "gate",
        f"from pathlib import Path\nimport time\nwhile not Path({str(release)!r}).exists(): time.sleep(.01)",
        project=project,
    )
    admit(host, "gate", gate)
    wait_until(lambda: job_row(host, gate.id, "running"))
    (source,) = launch(jobs, (train,), "saved-train").items
    submitted = launch(jobs, keys, "evaluate")
    assert [item.disposition for item in submitted.items] == ["admitted"] * 3
    assert launch(jobs, keys, "evaluate") == submitted
    (attached,) = launch(jobs, (baseline,), "evaluate-live").items
    assert attached.disposition == "attached" and attached.job_id == submitted.items[0].job_id
    with monkeypatch.context() as environment:
        environment.setenv("JAX_ENABLE_X64", "1")
        different_execution = JobService(host, registry, enable_evaluation_candidate=True)
        (conflict,) = launch(different_execution, (baseline,), "evaluate-live-conflict").items
        assert conflict.disposition == "rejected"
        assert "different execution binding" in conflict.reason
    for item in submitted.items:
        (row,) = host.jobs(job_id=item.job_id, include_launch=True)
        assert row["state"] == "queued" and not row["edges"][0]["satisfied"]
        assert row["launch"]["warm_entrypoint"] == f"{evaluate_worker.__name__}:main"
        assert row["edges"][0]["authenticated_completion"]
    unavailable = RunKey("default", "unsupported-condition", "default")
    (queued,) = launch(jobs, (unavailable,), "evaluate-queued-stop").items
    (stopped,) = jobs.stop((queued.job_id,), "checkpoint")
    assert stopped.job.status == "stopped" and not stopped.job.can_resume
    with pytest.raises(JobConflict, match="resubmit the saved Evaluate row"):
        jobs.resume(queued.job_id, "evaluate-resume")
    release.touch()

    trained = wait_until(lambda: job_row(host, source.job_id, "succeeded"), timeout=24)
    checkpoint = PublishedTrainingCheckpoint.model_validate(trained["outcome"]["completion"]["checkpoint"])
    slot = open_published_model_slot(checkpoint.checkpoint_root, checkpoint.pointer)
    assert slot is not None
    seed = StudioGraphMethodPayload.model_validate(spec.method_payload.payload).seed

    def template(initialization_seed):
        return compile_graph(
            GraphDocument(graph=slot.graph),
            registry.components,
            input_prototypes=prototypes_from_task_bindings(spec.task_binding_spec),
            initialization_key=studio_training_key(initialization_seed, "model"),
            artifact_provider=slot.provider,
        ).graph

    fresh, alternate = template(seed), template((seed + 1) % 2**32)
    restored = slot.restore(alternate)
    assert eqx.tree_equal(restored.nodes["readout"].layer, fresh.nodes["readout"].layer)
    assert not eqx.tree_equal(restored.nodes["readout"].layer, alternate.nodes["readout"].layer)
    assert not eqx.tree_equal(restored.nodes["cell"].cell, fresh.nodes["cell"].cell)
    root = plan.store_path
    store = ImmutableArtifactBlobProvider(root)
    outcomes = {}
    for key, item in zip(keys, submitted.items, strict=True):
        row = wait_until(lambda: (
            job_row(host, item.job_id, "succeeded") or job_row(host, item.job_id, "failed")
        ), timeout=22)
        assert row["state"] == "succeeded", row.get("outcome")
        assert row["request"] == by_key[key].request_id
        assert row["edges"][0]["binding"]["completion"]["producer_job_id"] == source.job_id
        outcomes[key] = jobs.read_evaluation_result(project=project, store=root, job_id=item.job_id)
        with pytest.raises(JobNotFound):
            gated.read_evaluation_result(project=project, store=root, job_id=item.job_id)

    joined = ProjectPlanService(folders, jobs).read(project, revision=plan.revision)
    for key, item in zip(keys, submitted.items, strict=True):
        (attempt,) = joined.attempts_by_run[key]
        assert attempt.job_id == item.job_id and attempt.successful_receipt == outcomes[key].receipt
        assert attempt.checkpoint_source.job_id == source.job_id
        assert attempt.output_size_bytes == outcomes[key].size_bytes
    position = state_key("plant", "effector/position")
    target = "/v1/task/output/reach_target"
    selected = []
    read_tensor = ArrayContainerReader.read_tensor

    def observed(reader, name):
        selected.append(name)
        return read_tensor(reader, name)

    with monkeypatch.context() as context:
        context.setattr(ArrayContainerReader, "read_tensor", observed)
        with outcomes[baseline].open_states() as dataset:
            assert not selected
            assert dataset.descriptor.schema_id == STATES_SCHEMA
            assert dataset.descriptor.dimensions == {
                "checkpoint": 1, "condition": 1, "trial": 3,
                "state_time": 101, "transition": 100, "xy": 2,
            }
            first = dataset.select(position, labels={"trial": "alpha"})[0, 0, :3]
        assert selected == [position]
        selected.clear()
        replay = outcomes[baseline].replay_trial("alpha", anchor_id="effector")
        assert set(selected) == {"state_time", position}
    assert replay.identity.stable_id == "alpha" and replay.time.length == 101
    assert replay.time.values[0] == 0.0
    assert replay.tracks[0].selector.compact == position
    np.testing.assert_array_equal(replay.tracks[0].samples[:3], first)

    arrays = {}
    for key, result in outcomes.items():
        manifest = result._manifest()
        (artifact,) = manifest.artifacts
        ref = DatasetProductRef(
            content={"sha256": artifact.sha256, "size": artifact.size_bytes},
            descriptor_sha256=artifact.metadata["descriptor_sha256"],
        )
        path = store.object_path(ref.object_ref)
        assert result.size_bytes == path.stat().st_size > 0
        assert sha256(path.read_bytes()).hexdigest() == ref.content.sha256
        assert manifest.input_training_runs[0].id.endswith(source.job_id)
        assert manifest.provenance.parents[0].metadata["completed_updates"] == 2
        with result.open_states() as dataset:
            pos = dataset.data(position)[0, 0]
            tgt = dataset.data(target)[0, 0]
            loss = dataset.data("objective/total")[0, 0]
            np.testing.assert_allclose(loss, np.sum((pos[:, -1] - tgt[:, -1]) ** 2, axis=-1), rtol=2e-5)
            assert len({value.tobytes() for value in tgt}) > 1
            arrays[key] = pos, tgt, dataset.data("condition")[0]
    base_pos, base_tgt, base_condition = arrays[baseline]
    nominal_pos, nominal_tgt, nominal_condition = arrays[conditions["nominal"]]
    heavy_pos, heavy_tgt, heavy_condition = arrays[conditions["heavy"]]
    np.testing.assert_array_equal(nominal_tgt, base_tgt)
    np.testing.assert_array_equal(heavy_tgt, base_tgt)
    np.testing.assert_allclose(nominal_pos, base_pos, rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(heavy_pos[:, 1:], base_pos[:, 1:] * 0.5, rtol=3e-5, atol=1e-8)
    np.testing.assert_array_equal(heavy_pos[:, 0], base_pos[:, 0])
    assert len({base_condition, nominal_condition, heavy_condition}) == 3

    live = {decision.sha256 for decision in store.collect_garbage().decisions if decision.decision == "live"}
    for result in outcomes.values():
        assert result.receipt.manifest_sha256 in live
        assert result._manifest().artifacts[0].sha256 in live

    (unsupported_condition,) = launch(jobs, (unavailable,), "unsupported-condition").items
    failed = wait_until(lambda: job_row(host, unsupported_condition.job_id, "failed"))
    assert failed["outcome"]["error"]["code"] == "cannot_run"
    assert failed["outcome"]["error"]["path"] == "/params/variation/0"

    restart(host)
    assert launch(jobs, keys, "evaluate") == submitted
    original = outcomes[baseline]
    artifact = original._manifest().artifacts[0]
    ref = DatasetProductRef(
        content={"sha256": artifact.sha256, "size": artifact.size_bytes},
        descriptor_sha256=artifact.metadata["descriptor_sha256"],
    )
    data_path = store.object_path(ref.object_ref)
    raw = data_path.read_bytes()
    header_end = 8 + int.from_bytes(raw[:8], "little")
    begin, _ = json.loads(raw[8:header_end])[position]["data_offsets"]
    offset = header_end + begin
    try:
        data_path.write_bytes(raw[:offset] + bytes([raw[offset] ^ 1]) + raw[offset + 1:])
        with original.open_states() as dataset:
            np.testing.assert_array_equal(dataset.data(target)[0, 0], base_tgt)
            with pytest.raises(ArrayContainerError):
                dataset.data(position)
    finally:
        data_path.write_bytes(raw)

    (rerun,) = launch(jobs, (baseline,), "evaluate-rerun").items
    assert rerun.disposition == "admitted" and rerun.job_id != submitted.items[0].job_id
    original_trial = ProjectPlanService(folders, jobs).evaluation_replay(
        project, StudioSavedEvaluationReplayRequest(
            revision=plan.revision,
            key=StudioSavedTrainKey.model_validate(baseline, from_attributes=True),
            job_id=submitted.items[0].job_id,
            trial="alpha",
        ),
    )
    assert original_trial == replay
    manifest_ref, raw = store.resolve_manifest("EvaluationRunManifest", outcomes[baseline].receipt.manifest_id)
    manifest_path = store.object_path(manifest_ref.object)
    try:
        manifest_path.write_bytes(raw + b"\n")
        with pytest.raises(JobConflict, match="cannot be authenticated"):
            jobs.read_evaluation_result(project=project, store=root, job_id=submitted.items[0].job_id)
    finally:
        manifest_path.write_bytes(raw)

    authored_path = folders.root / project / "steps" / "baseline.json"
    changed = json.loads(authored_path.read_bytes())
    changed["base"]["inline"]["params"]["seed"] += 1
    authored_path.write_text(json.dumps(changed))
    (stale,) = launch(jobs, (baseline,), "stale-evaluate").items
    assert stale.disposition == "rejected"
    assert "payload.params.seed" in stale.reason
