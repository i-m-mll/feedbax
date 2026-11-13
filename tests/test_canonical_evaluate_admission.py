"""Saved no-plugin Train-to-Evaluate through the local Host and dataset custody."""

from hashlib import sha256
import json

import equinox as eqx
import jax
import numpy as np
import pytest

from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.dataset import DatasetProductRef
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint, StudioGraphMethodPayload
from feedbax.jobs import JobConflict, JobNotFound, JobService
from feedbax.persistence.array_container import ArrayContainerError, ArrayContainerReader
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.runtime.raw_state import StateLayoutError, bind_raw_state
from feedbax.studio.canonical_evaluate import RAW_STATES_SCHEMA, ROLLOUT_TYPE, STATES_SCHEMA, state_key
from feedbax.tasks.document import TaskParameters
from feedbax.training import evaluate_worker
from feedbax.training.checkpoint_custody import open_published_model_slot
from feedbax.training.studio_executor import studio_training_key
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
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


def managed_evaluation_definition():
    from tests.test_document_train_pair import authored_documents, document_project

    raw = document_project(authored_documents())
    legacy, _ = authored_evaluation_project()
    raw["steps"].update({
        name: step.model_dump(mode="json", by_alias=True, exclude_unset=True)
        for name, step in legacy.steps.items()
        if name != "train"
    })
    for name, step in raw["steps"].items():
        if name != "train":
            params = step["base"]["inline"]["params"]
            params.pop("trials")
            params["trial_set"] = {"kind": "every_condition"}
    return raw


def test_managed_plan_resolves_task_owned_trial_ids():
    from feedbax.contracts.project_plan import compile_project
    from feedbax.contracts.task_document import TaskDocument

    raw = managed_evaluation_definition()
    task = TaskDocument.model_validate(raw["documents"]["task"]["payload"])
    assert len(task.condition_ids) == 28
    assert len({tuple(row.variables["start"]) for row in task.validation_trials}) == 4
    assert all(sum(other.variables["start"] == row.variables["start"] for other in task.validation_trials) == 7
               for row in task.validation_trials)
    assert task.model_copy(update={"validation_trials": task.validation_trials[::-1]}).condition_ids == task.condition_ids[::-1]
    params = raw["steps"]["baseline"]["base"]["inline"]["params"]
    selected = [task.condition_ids[i] for i in (10, 0, 5)]
    params["trial_set"] = {"kind": "conditions", "ids": selected}
    plan = compile_project(Project.model_validate(raw))
    baseline = next(run for run in plan.runs if run.key.step == "baseline")
    spec = decode_request(plan.requests[baseline.request_id], plan.documents.__getitem__).spec
    assert spec.params["trials"] == selected
    assert "trial_set" not in spec.params
    params["trial_set"]["ids"] = ["does-not-exist"]
    plan = compile_project(Project.model_validate(raw))
    assert not any(run.key.step == "baseline" for run in plan.runs)
    assert any(problem.step == "baseline" and problem.path == "payload.params.trial_set"
               and "does-not-exist" in problem.detail for problem in plan.diagnostics)
    params.pop("trial_set")
    params["trials"] = selected
    plan = compile_project(Project.model_validate(raw))
    assert any(problem.step == "baseline" and problem.path == "payload.params.trial_set"
               and "Declare a trial-set rule" in problem.detail for problem in plan.diagnostics)


def test_saved_train_to_evaluate_with_paired_conditions(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    registry = new_application_registry_bundle()
    jobs = JobService(host, registry, enable_evaluation_candidate=True)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "reaching"
    definition = managed_evaluation_definition()
    baseline_step = definition["steps"]["baseline"]
    legacy = json.loads(json.dumps(baseline_step))
    legacy["base"]["inline"]["params"]["retention"] = "observations"
    definition["steps"]["legacy"] = legacy
    stoppable = json.loads(json.dumps(baseline_step))
    stoppable["base"]["inline"]["params"]["seed"] += 1
    definition["steps"]["stoppable"] = stoppable
    _save_project(folders, project, Project.model_validate(definition))
    plan = folders.read_plan(project, revision=None)
    assert [(item.step, item.path) for item in plan.plan.diagnostics] == [
        ("unsupported-condition", "/params/variation/0"),
    ]
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
    spec = decode_request(plan.plan.requests[by_key[train].request_id], plan.plan.documents.__getitem__).spec
    documents = spec.trial_documents
    seed = StudioGraphMethodPayload.model_validate(spec.method_payload.payload).seed

    def template(initialization_seed):
        result = compile_trial_execution(
            documents.model_name, documents.model, documents.task_source,
            registry.components, registry.expression_functions, objective=documents.objective,
            input_overrides=documents.input_overrides,
            initialization_key=studio_training_key(initialization_seed, "model"),
            artifact_provider=slot.provider,
        )
        assert isinstance(result, PreparedTrialExecution), result
        return result

    fresh, alternate = template(seed), template((seed + 1) % 2**32)
    restored = slot.restore(alternate.executable.graph)
    def cell(graph):
        return graph.nodes["wrapper"].nodes["controller"].nodes["cell"].cell
    assert not eqx.tree_equal(cell(restored), cell(alternate.executable.graph))
    assert not eqx.tree_equal(cell(restored), cell(fresh.executable.graph))
    layout = bind_raw_state(restored, fresh.executable.resolved.graph)
    trial_ids = documents.task_source.condition_ids
    reference = fresh.execution.evaluate(
        restored, fresh.objective, TaskParameters(), key=jax.random.PRNGKey(731),
        trial_ids=trial_ids, capture_state=layout.capture,
    )
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
    position = "/v1/model/state/plant/effector~1position"
    carry = next(name for name in reference.history if name.startswith("/v1/model/carry/"))
    plant_fields = {
        state_key("plant", f"{state}/{symbol}"): (state, leaf, units)
        for state in ("skeleton", "effector")
        for symbol, leaf, units in (
            ("position", "pos", "m"), ("velocity", "vel", "m/s"), ("force", "force", "N"),
        )
    }
    target = "/v1/task/output/target_pos"
    selected = []
    read_slice = ArrayContainerReader.read_unverified_slice

    def observed(reader, name, selection):
        selected.append(name)
        return read_slice(reader, name, selection)

    with monkeypatch.context() as context:
        context.setattr(ArrayContainerReader, "read_unverified_slice", observed)
        with outcomes[baseline].open_states() as dataset:
            assert not selected
            descriptor = dataset.descriptor
            assert descriptor.schema_id == RAW_STATES_SCHEMA
            assert {name: descriptor.dimensions[name] for name in (
                "checkpoint", "condition", "trial", "state_time", "transition", "xy",
            )} == {"checkpoint": 1, "condition": 1, "trial": 28, "state_time": 81, "transition": 80, "xy": 2}
            assert {key for key in descriptor.variables if key.startswith("/v1/model/")} == set(reference.history)
            state_axes = ("checkpoint", "condition", "trial", "state_time")
            for key, (_, _, units) in plant_fields.items():
                field = descriptor.variables[key]
                assert (field.dims, field.units, field.frame) == ((*state_axes, "xy"), units, "world")
            assert descriptor.coords["state_time"].units == "s"
            assert descriptor.attrs["scene_position"] == position
            target_field = descriptor.variables[target]
            assert (target_field.dims, target_field.units, target_field.frame) == (
                ("checkpoint", "condition", "trial", "transition", "xy"), "m", "world",
            )
            assert dataset.data("xy").tolist() == ["x", "y"]
            assert descriptor.attrs["source_graph_sha256"] == sha256(
                canonical_json_v2_bytes(slot.graph.model_dump(mode="json"))
            ).hexdigest()
            assert dataset.data("trial").tolist() == list(trial_ids)
            selected.clear()
            first = dataset.select(position, labels={"trial": trial_ids[0]})[0, 0, :3]
        assert selected == [position]
        selected.clear()
        with outcomes[baseline].open_states() as dataset:
            hidden = dataset.select(carry, labels={"trial": trial_ids[0]})[0, 0, :3]
        assert selected == [carry]
        selected.clear()
        replay = outcomes[baseline].replay_trial(trial_ids[0], anchor_id="effector")
        assert selected == ["state_time", position]
        assert replay.identity.stable_id == trial_ids[0]
        np.testing.assert_array_equal(np.asarray(replay.tracks[0].samples[:3], dtype=first.dtype), first)
    np.testing.assert_allclose(first, reference.history[position][0, :3], atol=1e-7)
    np.testing.assert_allclose(hidden, reference.history[carry][0, :3], atol=1e-7)
    subset = fresh.execution.evaluate(
        restored, fresh.objective, TaskParameters(), key=jax.random.PRNGKey(731),
        trial_ids=(trial_ids[1], trial_ids[0]), capture_state=layout.capture,
    )
    np.testing.assert_allclose(subset.history[position], np.asarray(reference.history[position])[[1, 0]], atol=1e-7)
    for index, condition in enumerate((1, 0)):
        authored = documents.task_source.validation_trials[condition]
        np.testing.assert_allclose(subset.trials.inputs["target_pos"][index, -1], authored.variables["target"])
        np.testing.assert_allclose(subset.history[position][index, 0], authored.variables["start"])

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
            assert np.isfinite(loss) and loss > 0
            assert dataset.data("objective/coverage/effort").item() == 28
            assert len({value.tobytes() for value in tgt}) > 1
            arrays[key] = pos, tgt, dataset.data("condition")[0]
    base_pos, base_tgt, base_condition = arrays[baseline]
    with outcomes[baseline].open_states() as dataset:
        from feedbax.analysis.builtin_operations import AlignedVelocityParameters, aligned_velocity

        analysis = aligned_velocity(dataset, AlignedVelocityParameters(
            position=position, velocity=state_key("plant", "effector/velocity"), target=target,
        ))
        assert analysis.data("trial").tolist() == list(trial_ids)
        assert np.isfinite(analysis.data("parallel_velocity")).all()
        np.testing.assert_allclose(dataset.data("state_time"), np.arange(81) * documents.model.step_seconds, atol=1e-7)
        np.testing.assert_array_equal(dataset.data("transition_start"), dataset.data("state_time")[:-1])
        np.testing.assert_array_equal(dataset.data("transition_end"), dataset.data("state_time")[1:])
        for name, values in reference.history.items():
            np.testing.assert_allclose(dataset.data(name)[0, 0], values, atol=1e-7)
        np.testing.assert_allclose(dataset.data("objective/total").item(), reference.objective.loss, rtol=2e-5)
    nominal_pos, nominal_tgt, nominal_condition = arrays[conditions["nominal"]]
    heavy_pos, heavy_tgt, heavy_condition = arrays[conditions["heavy"]]
    np.testing.assert_array_equal(nominal_tgt, base_tgt)
    np.testing.assert_array_equal(heavy_tgt, base_tgt)
    np.testing.assert_allclose(nominal_pos, base_pos, rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(
        heavy_pos[:, 1:] - heavy_pos[:, :1], (base_pos[:, 1:] - base_pos[:, :1]) * 0.5,
        rtol=3e-5, atol=2 * np.finfo(base_pos.dtype).eps,
    )
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

    (legacy_attempt,) = launch(jobs, (RunKey("default", "legacy", "default"),), "legacy-observations").items
    legacy_job = wait_until(lambda: (
        job_row(host, legacy_attempt.job_id, "succeeded") or job_row(host, legacy_attempt.job_id, "failed")
    ))
    assert legacy_job["state"] == "succeeded", legacy_job.get("outcome")
    legacy_result = jobs.read_evaluation_result(project=project, store=root, job_id=legacy_attempt.job_id)
    assert legacy_result.size_bytes > 0
    with legacy_result.open_states() as dataset:
        assert dataset.descriptor.schema_id == STATES_SCHEMA
        assert "source_graph_sha256" not in dataset.descriptor.attrs
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
        with original.open_states(verify=True) as dataset:
            np.testing.assert_array_equal(dataset.select(position, labels={"trial": trial_ids[0]})[0, 0], base_pos[0])
            with pytest.raises(ArrayContainerError):
                dataset.select(carry, labels={"trial": trial_ids[0]})
        corrupt_part(position)
        with original.open_states(verify=True) as dataset:
            with pytest.raises(ArrayContainerError):
                dataset.select(position, labels={"trial": trial_ids[0]})
            np.testing.assert_array_equal(dataset.data(target)[0, 0], base_tgt)
    finally:
        data_path.write_bytes(raw)

    (rerun,) = launch(jobs, (baseline,), "evaluate-rerun").items
    assert rerun.disposition == "admitted" and rerun.job_id != submitted.items[0].job_id
    with original.open_states() as dataset:
        np.testing.assert_array_equal(dataset.select(position, labels={"trial": trial_ids[0]})[0, 0], base_pos[0])
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
    registry = jobs.registries
    definition = managed_evaluation_definition()
    _save_project(folders, project, Project.model_validate(definition))
    reviewed = folders.read_plan(project, revision=None)
    assert not reviewed.plan.diagnostics
    run = next(run for run in reviewed.plan.runs if run.key.step == "train")
    spec = decode_request(reviewed.plan.requests[run.request_id], reviewed.plan.documents.__getitem__).spec
    documents = spec.trial_documents
    from feedbax.compiler import GraphDocument, compile_graph
    from feedbax.contracts.graph import ComponentSpec, WireSpec

    prepared = compile_trial_execution(
        documents.model_name, documents.model, documents.task_source,
        registry.components, registry.expression_functions, input_overrides=documents.input_overrides,
    )
    assert isinstance(prepared, PreparedTrialExecution), prepared
    graph = prepared.executable.resolved.graph
    graph.nodes["passthrough"] = ComponentSpec(
        type="feedbax.Gain", params={"gain": 1.0}, input_ports=["input"], output_ports=["output"],
    )
    graph.wires.append(WireSpec(
        source_node="wrapper", source_port="force", target_node="passthrough", target_port="input",
    ))
    compiled = compile_graph(
        GraphDocument(graph=graph), registry.components,
        input_prototypes={graph.input_bindings["reach_target"]: jax.ShapeDtypeStruct((2,), np.float32)},
    )
    with pytest.raises(StateLayoutError, match="no declared raw-state layout") as failure:
        bind_raw_state(compiled.graph, compiled.resolved.graph)
    assert failure.value.path == "/inputs/subject/graph/nodes/passthrough"
