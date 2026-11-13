"""Exercise saved Model attachment differences through document Train and Evaluate."""

from copy import deepcopy
from hashlib import sha256

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
from feedbax.contracts.authored_canonical import canonical_sha256
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.expression_values import LiteralValue, NameValue, ValueBase, ValueType
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_edit import EvaluateAttachmentRequest
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.objective_document import ObjectiveTerm
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.rollout import RolloutParameters
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.studio_project import StudioProjectPatchBatch
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.contracts.task_document import Cell, Signal
from feedbax.jobs import JobService
from feedbax.mechanics.muscles import RigidTendonHillMuscleThelen
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_evaluate import CannotRun
from feedbax.training.checkpoint_custody import open_published_model_slot
from feedbax.training.evaluation_variant import resolve_evaluation_variant, validate_evaluation_source
from feedbax.web.services.evaluate_attachment import (
    EvaluateAttachmentRefused,
    prepare_evaluate_attachment,
    read_evaluate_attachments,
)
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import read_settled_project_plan
from tests.support.managed_import_probe import check_managed_worker, probe_managed_workers
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_train_pair import authored, document_project  # noqa: F401
from tests.test_local_job_host import host, job_row, wait_until  # noqa: F401
from tests.test_model_document_plan import (
    _add_attachment,
    _attachment_address,
    _attachment_owner,
    _end,
    _model_controller,
    _port,
    _scale_attachment,
    _wire,
)


def test_saved_addressed_condition_restores_document_train_model(host, tmp_path, authored):  # noqa: F811
    registry = new_application_registry_bundle()
    jobs = JobService(host, registry, enable_evaluation_candidate=True)
    imports = tmp_path / "worker-imports"
    probe_managed_workers(jobs, imports)
    folders = ProjectFolderService(
        tmp_path / "projects",
        components=registry.components,
        functions=registry.expression_functions,
    )
    project = "addressed-condition"
    raw = document_project(authored, evaluation=True)
    raw["steps"]["condition"] = deepcopy(raw["steps"]["baseline"])
    _model_controller(raw)["wires"]["read"]["attachments"] = [
        _scale_attachment("retain", 0.8),
        _scale_attachment("remove", 0.5),
    ]
    _save_project(folders, project, Project.model_validate(raw))
    original = folders.read_plan(project, revision=None)
    previous = {run.key: run.request_id for run in original.plan.runs}
    train = RunKey("default", "train", "default")
    baseline = RunKey("default", "baseline", "default")
    condition = RunKey("default", "condition", "default")
    assert previous[baseline] == previous[condition]

    owner = _attachment_owner("read")
    selection = {
        "run": {"step": "condition", "row": "default", "variant": "default"},
        "scope": {"kind": "step"},
    }
    invalid = EvaluateAttachmentRequest.model_validate(
        selection
        | {
            "revision": original.revision,
            "difference": _add_attachment(owner, "invalid", "missing"),
        }
    )
    with pytest.raises(
        EvaluateAttachmentRefused, match="anchor 'missing' no longer exists"
    ) as refused:
        prepare_evaluate_attachment(folders, project, invalid, registry.components)
    assert refused.value.problem.address == invalid.difference.address

    differences = (
        _add_attachment(owner, "add", "retain", 0.4),
        {
            "op": "change",
            "address": _attachment_address(owner, "retain"),
            "attachment": _scale_attachment("retain", 0.6),
        },
        {"op": "remove", "address": _attachment_address(owner, "remove")},
    )
    plan = original
    for difference in differences:
        request = EvaluateAttachmentRequest.model_validate(
            selection
            | {
                "revision": plan.revision,
                "difference": difference,
            }
        )
        edit = prepare_evaluate_attachment(folders, project, request, registry.components)
        assert edit.changed == request.difference.address
        folders.patch(project, StudioProjectPatchBatch(revision=plan.revision, files=[edit.file]))
        plan = folders.read_plan(project, revision=None)

    saved = read_evaluate_attachments(
        folders, project, request.model_copy(update={"revision": plan.revision})
    )
    assert saved.records == saved.effective
    assert [item.op for item in saved.records] == ["add", "change", "remove"]
    assert folders.read(project).project.documents == Project.model_validate(raw).documents
    current = {run.key: run.request_id for run in plan.plan.runs}
    assert current.keys() == previous.keys()
    assert {key for key in current if current[key] != previous[key]} == {condition}

    jobs_by_run = {}
    for key in (train, baseline, condition):
        (admitted,) = jobs.admit_plan(folders, plan, run_keys=(key,), token=key.step).items
        assert admitted.disposition == "admitted", admitted
        jobs_by_run[key] = admitted.job_id

    completions = {}
    for run, job_id in jobs_by_run.items():
        terminal = wait_until(
            lambda: (job_row(host, job_id, "succeeded") or job_row(host, job_id, "failed")),
            timeout=26,
        )
        assert terminal["state"] == "succeeded", terminal.get("outcome")
        completions[run] = terminal
        check_managed_worker(
            imports,
            job_id,
            required=(
                "feedbax.training.managed_studio" if run == train
                else "feedbax.training.evaluate_worker"
            ),
        )
    checkpoint = completions[train]["outcome"]["completion"]
    for run in (baseline, condition):
        (edge,) = completions[run]["edges"]
        assert edge["binding"]["completion"] == checkpoint
    published = PublishedTrainingCheckpoint.model_validate(checkpoint["checkpoint"])
    slot = open_published_model_slot(published.checkpoint_root, published.pointer)
    assert slot is not None
    source_hash = sha256(canonical_json_v2_bytes(slot.graph.model_dump(mode="json"))).hexdigest()
    train_request = decode_request(
        plan.plan.requests[current[train]],
        plan.plan.documents.__getitem__,
    ).spec
    condition_request = decode_request(
        plan.plan.requests[current[condition]],
        plan.plan.documents.__getitem__,
    ).spec
    params = RolloutParameters.model_validate(condition_request.params)
    variation_hash = canonical_sha256(params.model_dump(mode="json")["variation"])
    documents = train_request.trial_documents
    assert documents is not None
    variant = resolve_evaluation_variant(slot.graph, documents.task, params.variation)
    task_hash = sha256(canonical_json_v2_bytes(variant.task.model_dump(mode="json"))).hexdigest()
    paired = compile_trial_execution(
        documents.model_name,
        documents.model,
        variant.task,
        registry.components,
        registry.expression_functions,
        input_overrides=documents.input_overrides,
        graph_override=variant.graph,
    )
    assert isinstance(paired, PreparedTrialExecution), paired
    effective_hash = canonical_sha256(
        {
            "resolved_graph": paired.executable.resolved.resolved_sha256,
            "checkpoint_values": slot.arrays.sha256,
            "source_overrides": {},
        }
    )

    def open_states(run):
        return jobs.read_evaluation_result(
            project=project,
            store=plan.store_path,
            job_id=jobs_by_run[run],
        ).open_states()

    with open_states(baseline) as control, open_states(condition) as treated:
        for states in (control, treated):
            assert states.data("checkpoint").item() == 2
            assert states.data("objective/coverage/effort").item() == 28
            assert states.data("state_time").shape == (81,)
            assert states.descriptor.attrs["source_graph_sha256"] == source_hash
        assert "effective_graph_sha256" not in control.descriptor.attrs
        attrs = treated.descriptor.attrs
        assert attrs["effective_graph_sha256"] == effective_hash
        assert attrs["effective_task_sha256"] == task_hash
        assert attrs["variation_sha256"] == treated.data("condition").item() == variation_hash
        assert control.data("condition").item() != treated.data("condition").item()
        control_loss = control.data("objective/total").item()
        assert np.isfinite(control_loss) and control_loss > 0
        assert treated.data("objective/total").item() == pytest.approx(
            control_loss * ((0.6 * 0.4) / (0.8 * 0.5)) ** 2,
            rel=1e-5,
        )
        carry = "/v1/model/carry/wrapper/controller/carry/carry"
        np.testing.assert_array_equal(control.data(carry), treated.data(carry))


def _owner_field(owner, field):
    return ModelAddress.model_validate({
        "document": "reach-model",
        "composites": (),
        "element": {
            "kind": "field", "parent": {"kind": "component", "id": owner},
            "path": ["fields", field],
        },
    })


def _set_model_field(address, value):
    return {"op": "set_field", "address": address.model_dump(mode="json"), "value": value}


def _thelen_project(documents):
    _, _, task, objective = documents
    raw = document_project(documents, evaluation=True)
    model = raw["documents"]["reach-model"]["payload"]
    config = {
        "optimal_muscle_length": 0.16,
        "max_isometric_force": 710.0,
        "tendon_slack_length": 0.12,
        "vmax_factor": 7.5,
        "initial_activation": 0.25,
    }
    model["interior"]["components"]["muscle"] = {
        "kind": "leaf", "name": "Muscle", "type": "feedbax.RigidTendonHillMuscleThelen",
        "fields": config | {"vmax_factor": 10.0},
    }
    samples = {"excitation": 0.5, "musculotendon_length": 0.36, "musculotendon_velocity": -0.31}
    signals = list(task.signals)
    for port, sample in samples.items():
        model["ports"]["inputs"].append(_port(port, port, shape=[]))
        model["interior"]["wires"][port] = _wire({"owner": "inputs", "port": port}, _end("muscle", port))
        signals.append(Signal(
            id=port, name=port, type=ValueType(base=ValueBase.REAL),
            cells=(Cell(first_epoch="hold", last_epoch="move", value=LiteralValue(value=sample)),),
        ))
    model["ports"]["outputs"].append(_port("muscle-force", "muscle_force", shape=[]))
    model["interior"]["wires"]["muscle-force"] = _wire(
        _end("muscle", "force"), {"owner": "outputs", "port": "muscle-force"},
    )
    address = ModelAddress.model_validate({
        "document": "reach-model", "composites": (),
        "element": {"kind": "port", "end": {"owner": "outputs", "port": "muscle-force"}},
    })
    term = ObjectiveTerm(id="muscle-force", measure=NameValue(reference=address.to_text()))
    raw["documents"]["objective"]["payload"] = objective.model_copy(
        update={"terms": (*objective.terms, term)},
    ).model_dump(mode="json")
    raw["documents"]["task"]["payload"] = task.model_copy(update={"signals": tuple(signals)}).model_dump(mode="json")
    raw["steps"]["train"]["model_attachments"] = {
        "source": [_set_model_field(_owner_field("muscle", "vmax_factor"), config["vmax_factor"])],
    }
    params = raw["steps"]["baseline"]["base"]["inline"]["params"]
    params["trial_set"] = {"kind": "conditions", "ids": [task.condition_ids[0]]}
    raw["steps"]["condition"] = deepcopy(raw["steps"]["baseline"])
    raw["steps"]["condition"]["model_attachments"] = {
        "source": [_set_model_field(_owner_field("muscle", "optimal_muscle_length"), 0.24)],
    }
    return raw, config | {"dt": model["step_seconds"]}, samples


def test_saved_source_field_rebuilds_derived_owner(host, tmp_path, authored):  # noqa: F811
    registry = new_application_registry_bundle()
    folders = ProjectFolderService(
        tmp_path / "projects", components=registry.components, functions=registry.expression_functions,
    )
    jobs = JobService(host, registry, enable_evaluation_candidate=True)
    raw, config, samples = _thelen_project(authored)
    _save_project(folders, "thelen", Project.model_validate(raw))
    snapshot = folders.read_plan("thelen", revision=None)
    assert not snapshot.plan.diagnostics, snapshot.plan.diagnostics
    keys = {step: RunKey("default", step, "default") for step in ("train", "baseline", "condition")}
    admissions = {
        step: jobs.admit_plan(folders, snapshot, run_keys=(key,), token=step).items[0]
        for step, key in keys.items()
    }
    completions = {}
    for step, admission in admissions.items():
        assert admission.disposition == "admitted", admission
        done = wait_until(
            lambda: job_row(host, admission.job_id, "succeeded") or job_row(host, admission.job_id, "failed"),
            timeout=25,
        )
        assert done["state"] == "succeeded", done["outcome"]
        completions[step] = done
    checkpoint = completions["train"]["outcome"]["completion"]
    published = PublishedTrainingCheckpoint.model_validate(checkpoint["checkpoint"])
    slot = open_published_model_slot(published.checkpoint_root, published.pointer)
    assert slot is not None
    assert slot.graph.nodes["muscle"].params["vmax_factor"] == config["vmax_factor"]
    assert slot.graph.nodes["muscle"].params["optimal_muscle_length"] == config["optimal_muscle_length"]
    condition = next(run for run in snapshot.plan.runs if run.key == keys["condition"])
    request = decode_request(snapshot.plan.requests[condition.request_id], snapshot.plan.documents.__getitem__)
    params = RolloutParameters.model_validate(request.spec.params)
    (difference,) = params.variation
    assert difference.path == "graph.nodes.muscle.params.optimal_muscle_length"
    assert difference.value == 0.24
    train_run = next(run for run in snapshot.plan.runs if run.key == keys["train"])
    documents = decode_request(
        snapshot.plan.requests[train_run.request_id], snapshot.plan.documents.__getitem__,
    ).spec.trial_documents
    variant = resolve_evaluation_variant(slot.graph, documents.task_source, params.variation)
    expected_overrides = {semantic_key("model", "value", ("muscle",), "optimal_muscle_length"): 0.24}
    assert variant.source_overrides == expected_overrides
    paired = compile_trial_execution(
        documents.model_name, documents.model, documents.task_source,
        registry.components, registry.expression_functions,
        input_overrides=documents.input_overrides, graph_override=variant.graph,
    )
    assert isinstance(paired, PreparedTrialExecution), paired
    effective_hash = canonical_sha256({
        "resolved_graph": paired.executable.resolved.resolved_sha256,
        "checkpoint_values": slot.arrays.sha256,
        "source_overrides": expected_overrides,
    })

    def fresh_rollout(muscle):
        initial = eqx.nn.State(muscle)
        inputs = {name: jnp.asarray(value, dtype=jnp.float32) for name, value in samples.items()}
        key = jax.random.key(0)

        def step(state, _):
            outputs, state = muscle(inputs, state, key=key)
            return state, (outputs["force"], outputs["activation"])

        _, (force, activation) = jax.lax.scan(step, initial, None, length=80)
        return np.mean(force), np.r_[initial.get(muscle.state_index), activation]

    baseline = RigidTendonHillMuscleThelen(**config)
    varied = RigidTendonHillMuscleThelen(**(config | {"optimal_muscle_length": 0.24}))
    stale = eqx.tree_at(lambda muscle: muscle.optimal_muscle_length, baseline, 0.24)
    assert varied.vmax != stale.vmax
    baseline_force, baseline_activation = fresh_rollout(baseline)
    varied_force, varied_activation = fresh_rollout(varied)
    assert not np.isclose(varied_force, baseline_force, rtol=1e-3)
    assert not np.isclose(varied_force, fresh_rollout(stale)[0], rtol=1e-3)
    for step, expected_force, expected_activation in (
        ("baseline", baseline_force, baseline_activation),
        ("condition", varied_force, varied_activation),
    ):
        (edge,) = completions[step]["edges"]
        assert edge["binding"]["completion"] == checkpoint
        result = jobs.read_evaluation_result(
            project="thelen", store=snapshot.store_path, job_id=admissions[step].job_id,
        )
        with result.open_states() as states:
            if step == "condition":
                assert states.descriptor.attrs["effective_graph_sha256"] == effective_hash
            assert states.data("objective/coverage/muscle-force").item() == 1
            assert states.data("objective/terms/muscle-force").item() == pytest.approx(expected_force, rel=1e-6)
            np.testing.assert_allclose(states.data("/v1/model/state/muscle/activation")[0, 0, 0],
                                       expected_activation, rtol=1e-6, atol=1e-7)


def test_saved_evaluate_refuses_undeclared_subject_owner(host, tmp_path, authored):  # noqa: F811
    raw = document_project(authored, evaluation=True)
    valid = Project.model_validate(raw)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "undeclared-owner"
    _save_project(folders, project, valid)
    initial = folders.read_plan(project, revision=None).plan
    assert not initial.diagnostics

    controller = _model_controller(raw)
    controller["components"]["passthrough"] = {
        "kind": "leaf", "name": "Passthrough", "type": "feedbax.Gain", "fields": {"gain": 1.0},
    }
    original = controller["wires"]["read"]
    original["target"], output = _end("passthrough", "input"), original["target"]
    controller["wires"]["read-output"] = _wire(_end("passthrough", "output"), output)
    _save_project(folders, project, Project.model_validate(raw))
    snapshot = folders.read_plan(project, revision=None)
    plan = snapshot.plan
    assert {run.key.step for run in plan.runs} == {"train"}
    (problem,) = plan.diagnostics
    assert problem.step == "baseline"
    assert problem.model_address == ModelAddress.model_validate({
        "document": "reach-model", "composites": ["wrapper", "controller"],
        "element": {"kind": "component", "id": "passthrough"},
    })
    assert problem.path == "/inputs/subject/graph/subgraphs/wrapper/subgraphs/controller/nodes/passthrough"
    assert "feedbax.Gain" in problem.detail and "no declared Evaluate layout" in problem.detail
    jobs = JobService(host, new_application_registry_bundle(), enable_evaluation_candidate=True)
    key = RunKey("default", "baseline", "default")
    view = read_settled_project_plan(snapshot, jobs)
    row = next(row for row in view.status_rows() if row.key == key)
    assert row.status == "cannot_run"
    assert problem.path in row.reason
    (refused,) = jobs.admit_plan(folders, snapshot, run_keys=(key,), token="unsupported").items
    assert refused.disposition == "rejected"
    assert not host.jobs()

    _save_project(folders, project, valid)
    restored = folders.read_plan(project, revision=None).plan
    assert not restored.diagnostics
    assert [(run.key, run.request_id) for run in restored.runs] == [
        (run.key, run.request_id) for run in initial.runs
    ]


def test_native_subject_refuses_before_backend_realization():
    native = GraphSpec.model_validate({"nodes": {"physics": {"type": "feedbax.AcausalSystem"}}})
    with pytest.raises(CannotRun, match="no qualified Evaluate realization") as refused:
        validate_evaluation_source(native)
    assert refused.value.path == "/inputs/subject/graph/nodes/physics"


def test_authored_variation_refuses_added_native_owner(authored):  # noqa: F811
    _, _, task, _ = authored
    variation = {"native": OverridePatch(
        path="graph.nodes.physics", op="add", value={"type": "feedbax.AcausalSystem"},
    )}
    with pytest.raises(CannotRun, match="no qualified declaration") as refused:
        resolve_evaluation_variant(GraphSpec(), task, variation)
    assert refused.value.path == "/params/variation/native"
