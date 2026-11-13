"""Saved no-plugin Train-to-Evaluate through the local Host and dataset custody."""

from hashlib import sha256
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.analysis.builtin_operations import (
    AlignedVelocityParameters,
    VelocitySummaryParameters,
    aligned_velocity,
    summarize_velocity,
)
from feedbax.analysis.figure_presets import velocity_summary_figure
from feedbax.compiler import GraphDocument, compile_graph
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.dataset import DatasetProductRef
from feedbax.contracts.jobs import StudioSavedEvaluationReplayRequest, StudioSavedTrainKey
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint, StudioGraphMethodPayload
from feedbax.jobs import JobConflict, JobNotFound, JobService
from feedbax.persistence.array_container import ArrayContainerError, ArrayContainerReader
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.plot.dataset_figure import render
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.runtime._graph_key_schedule import derive_key
from feedbax.studio.canonical_evaluate import RAW_STATES_SCHEMA, ROLLOUT_TYPE, STATES_SCHEMA, state_key
from feedbax.training import evaluate_worker
from feedbax.training.checkpoint_custody import open_published_model_slot
from feedbax.training.studio_executor import studio_training_key
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from feedbax.web.worker.execution import prototypes_from_task_bindings
from tests.test_local_job_host import admit, host as host_fixture, job_row, restart, run_job, wait_until
from tests.test_studio_training_host_worker import _authored_spec

host = host_fixture


def authored_evaluation_project():
    from tests.test_studio_training_host_worker import _authored_request

    spec = _authored_spec(batches=2, sampled_hold=True)
    train_payload = _authored_request(batches=2, sampled_hold=True).model_dump(mode="json", exclude_none=True)
    step = {
        "kind": "evaluate",
        "base": {"inline": {
            "evaluation_type": ROLLOUT_TYPE,
            "params": {
                "seed": 731,
                "trials": ["alpha", "beta", "gamma"],
                "trial_policy": "paired",
                "variation": [],
            },
        }},
        "inputs": {"subject": {"step": "train", "checkpoint": "last"}},
    }
    sweep = {"axes": {"mass": {"values": {
        name: {"value": mass, "patches": {"params.variation": {"value": [
            {"path": "graph.nodes.plant.params.mass", "op": "replace", "value": mass},
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


def _save_project(folders, project, authored):
    for name, data in dump_text_project(
        authored, ProjectConfiguration(project=project, store="results"),
    ).items():
        path = folders.root / project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_saved_train_to_evaluate_with_paired_conditions(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    registry = new_application_registry_bundle()
    jobs = JobService(host, registry, enable_evaluation_candidate=True)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "reaching"
    authored, spec = authored_evaluation_project()
    definition = authored.model_dump(mode="json", by_alias=True, exclude_unset=True)
    baseline_step = definition["steps"]["baseline"]
    legacy = json.loads(json.dumps(baseline_step))
    legacy["base"]["inline"]["params"]["retention"] = "observations"
    definition["steps"]["legacy"] = legacy
    stoppable = json.loads(json.dumps(baseline_step))
    stoppable["base"]["inline"]["params"]["seed"] += 1
    definition["steps"]["stoppable"] = stoppable
    _save_project(folders, project, Project.model_validate(definition))
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
    stoppable_key = RunKey("default", "stoppable", "default")
    (queued,) = launch(jobs, (stoppable_key,), "evaluate-queued-stop").items
    assert queued.disposition == "admitted"
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
    (invalid_condition,) = (
        problem for problem in joined.evaluation_problems if problem.step == "unsupported-condition"
    )
    assert invalid_condition.path == "/params/variation/0"
    for key, item in zip(keys, submitted.items, strict=True):
        (attempt,) = joined.attempts_by_run[key]
        assert attempt.job_id == item.job_id and attempt.successful_receipt == outcomes[key].receipt
        assert attempt.checkpoint_source.job_id == source.job_id
        assert attempt.output_size_bytes == outcomes[key].size_bytes
    position = "/v1/model/state/plant/effector~1position"
    carry = "/v1/model/carry/cell:hidden/hidden"
    plant_fields = {
        state_key("plant", f"{state}/{symbol}"): (state, leaf, units)
        for state in ("skeleton", "effector")
        for symbol, leaf, units in (
            ("position", "pos", "m"), ("velocity", "vel", "m/s"), ("force", "force", "N"),
        )
    }
    cell_fields = {state_key("cell", symbol): symbol for symbol in ("hidden", "output")}
    target = "/v1/task/output/reach_target"
    selected = []
    read_tensor = ArrayContainerReader.read_tensor

    def observed(reader, name):
        selected.append(name)
        return read_tensor(reader, name)

    replay_request = StudioSavedEvaluationReplayRequest(
        revision=plan.revision,
        key=StudioSavedTrainKey.model_validate(baseline, from_attributes=True),
        job_id=submitted.items[0].job_id,
        trial="alpha",
    )
    service = ProjectPlanService(folders, jobs)
    with monkeypatch.context() as context:
        context.setattr(ArrayContainerReader, "read_tensor", observed)
        with outcomes[baseline].open_states() as dataset:
            assert not selected
            descriptor = dataset.descriptor
            assert descriptor.schema_id == RAW_STATES_SCHEMA
            assert descriptor.dimensions == {
                "checkpoint": 1, "condition": 1, "trial": 3,
                "state_time": 101, "transition": 100, "xy": 2, "cell:unit": 100,
            }
            assert {key for key in descriptor.variables if key.startswith("/v1/model/")} == {
                *plant_fields, *cell_fields, carry,
            }
            state_axes = ("checkpoint", "condition", "trial", "state_time")
            for key, (_, _, units) in plant_fields.items():
                field = descriptor.variables[key]
                assert (field.dims, field.units, field.frame) == ((*state_axes, "xy"), units, "world")
            for key in (*cell_fields, carry):
                field = descriptor.variables[key]
                assert (field.dims, field.units) == ((*state_axes, "cell:unit"), "1")
            assert descriptor.coords["state_time"].units == "s"
            assert descriptor.attrs["carry_address_migration"] == "legacy-gru-hidden-self-loop.v1"
            assert descriptor.attrs["source_graph_sha256"] == sha256(
                canonical_json_v2_bytes(slot.graph.model_dump(mode="json"))
            ).hexdigest()
            assert dataset.data("trial").tolist() == ["alpha", "beta", "gamma"]
            first = dataset.select(position, labels={"trial": "alpha"})[0, 0, :3]
        assert selected == [position]
        selected.clear()
        with outcomes[baseline].open_states() as dataset:
            hidden = dataset.select(carry, labels={"trial": "alpha"})[0, 0, :3]
        assert hidden.shape == (3, 100) and selected == [carry]
        selected.clear()
        replay = service.evaluation_replay(project, replay_request)
        assert selected == ["state_time", position]
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
    trial_key = derive_key(jax.random.PRNGKey(731), "evaluate-trial", "alpha")
    graph_key = derive_key(trial_key, "studio-trial", "graph")
    state = restored.init_state(key=graph_key)
    input_name = next(name for name, port in restored.input_bindings.items() if port == ("cell", "input"))
    cycle = restored.initial_cycle_port_values(
        state, inputs={input_name: jnp.asarray(base_tgt[0, 0])}, key=graph_key,
    )
    with outcomes[baseline].open_states() as dataset:
        np.testing.assert_allclose(dataset.data("state_time"), np.arange(101) * 0.01, atol=1e-7)
        np.testing.assert_array_equal(dataset.data("transition_start"), dataset.data("state_time")[:-1])
        np.testing.assert_array_equal(dataset.data("transition_end"), dataset.data("state_time")[1:])
        state_fields = {key: dataset.data(key)[0, 0, 0, :4] for key in (*plant_fields, *cell_fields, carry)}
        velocity_key = state_key("plant", "effector/velocity")
        projected = aligned_velocity(dataset, AlignedVelocityParameters(
            position=position, velocity=velocity_key, target=target,
        ))
        summary = summarize_velocity(projected, VelocitySummaryParameters(ddof=1))
        reach = base_tgt[:, -1] - base_pos[:, 0]
        direction = reach / np.linalg.norm(reach, axis=-1, keepdims=True)
        expected = np.einsum("sti,si->st", dataset.data(velocity_key)[0, 0], direction)
        np.testing.assert_allclose(summary.data("mean")[0, 0], expected.mean(axis=0), atol=1e-6)
        np.testing.assert_allclose(
            summary.data("standard_deviation")[0, 0], expected.std(axis=0, ddof=1), atol=1e-6,
        )
        np.testing.assert_array_equal(summary.data("state_time"), dataset.data("state_time"))
        figure = render(velocity_summary_figure(), {"summary": summary})
        center, upper, lower = figure.data
        assert center.name == dataset.data("condition")[0]
        for trace, name in ((center, "mean"), (upper, "upper"), (lower, "lower")):
            np.testing.assert_array_equal(trace.x, dataset.data("state_time"))
            np.testing.assert_array_equal(trace.y, summary.data(name)[0, 0])
    for sample in range(4):
        plant_state = state.get(restored.nodes["plant"].state_index)
        cell_state = state.get(restored.nodes["cell"].state_index)
        for key, (part, leaf, _) in plant_fields.items():
            value = plant_state.effector if part == "effector" else plant_state.plant.skeleton
            np.testing.assert_allclose(state_fields[key][sample], getattr(value, leaf), atol=1e-7)
        for key, symbol in cell_fields.items():
            np.testing.assert_allclose(state_fields[key][sample], getattr(cell_state, symbol), atol=1e-7)
        np.testing.assert_allclose(state_fields[carry][sample], cycle["cell", "hidden"], atol=1e-7)
        if sample < 3:
            _, state, cycle = restored.step(
                {input_name: jnp.asarray(base_tgt[0, sample])},
                state, cycle, key=graph_key, t=sample,
            )
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

    unavailable = RunKey("default", "unsupported-condition", "default")
    (unsupported_condition,) = launch(jobs, (unavailable,), "unsupported-condition").items
    assert unsupported_condition.disposition == "rejected"
    assert unsupported_condition.job_id is None
    assert "cannot_run /params/variation/0" in unsupported_condition.reason

    (legacy_attempt,) = launch(jobs, (RunKey("default", "legacy", "default"),), "legacy-observations").items
    legacy_job = wait_until(lambda: (
        job_row(host, legacy_attempt.job_id, "succeeded") or job_row(host, legacy_attempt.job_id, "failed")
    ))
    assert legacy_job["state"] == "succeeded", legacy_job.get("outcome")
    legacy_result = jobs.read_evaluation_result(project=project, store=root, job_id=legacy_attempt.job_id)
    assert legacy_result.size_bytes == 13328
    with legacy_result.open_states() as dataset:
        assert dataset.descriptor.schema_id == STATES_SCHEMA
        assert carry not in dataset.descriptor.variables
        np.testing.assert_array_equal(dataset.data(position)[0, 0], base_pos)

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
    parts = json.loads(raw[8:header_end])

    def corrupt_part(name):
        begin, _ = parts[name]["data_offsets"]
        offset = header_end + begin
        data_path.write_bytes(raw[:offset] + bytes([raw[offset] ^ 1]) + raw[offset + 1:])

    try:
        corrupt_part(carry)
        assert service.evaluation_replay(project, replay_request) == replay
        with original.open_states() as dataset:
            with pytest.raises(ArrayContainerError):
                dataset.select(carry, labels={"trial": "alpha"})
        corrupt_part(position)
        with pytest.raises(JobConflict, match="could not be read and authenticated"):
            service.evaluation_replay(project, replay_request)
        with original.open_states() as dataset:
            np.testing.assert_array_equal(dataset.data(target)[0, 0], base_tgt)
    finally:
        data_path.write_bytes(raw)

    (rerun,) = launch(jobs, (baseline,), "evaluate-rerun").items
    assert rerun.disposition == "admitted" and rerun.job_id != submitted.items[0].job_id
    original_trial = service.evaluation_replay(project, replay_request)
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


def test_saved_raw_state_refuses_undeclared_registered_owner(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    jobs = JobService(host, new_application_registry_bundle(), enable_evaluation_candidate=True)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "unsupported-owner"
    authored, _ = authored_evaluation_project()
    definition = authored.model_dump(mode="json", by_alias=True, exclude_unset=True)
    graph = definition["steps"]["train"]["base"]["inline"]["graph"]
    graph["nodes"]["passthrough"] = {
        "type": "feedbax.Gain", "params": {"gain": 1.0},
        "input_ports": ["input"], "output_ports": ["output"],
    }
    graph["wires"].append({
        "source_node": "cell", "source_port": "hidden",
        "target_node": "passthrough", "target_port": "input",
    })
    graph["output_ports"].append("passthrough")
    graph["output_bindings"]["passthrough"] = ["passthrough", "output"]
    _save_project(folders, project, Project.model_validate(definition))
    reviewed = folders.read_plan(project, revision=None)
    assert not reviewed.plan.diagnostics
    (producer,) = jobs.admit_plan(
        folders, reviewed, run_keys=(RunKey("default", "train", "default"),), token="train",
    ).items
    trained = wait_until(lambda: (
        job_row(host, producer.job_id, "succeeded") or job_row(host, producer.job_id, "failed")
    ), timeout=24)
    assert trained["state"] == "succeeded", trained.get("outcome")
    (consumer,) = jobs.admit_plan(
        folders, reviewed, run_keys=(RunKey("default", "baseline", "default"),), token="evaluate",
    ).items
    assert consumer.disposition == "rejected"
    assert consumer.job_id is None
    assert "cannot_run /inputs/subject/graph/nodes/passthrough" in consumer.reason
