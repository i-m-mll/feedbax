"""The document pair uses the native update and fixed validation scorer."""

import json
from copy import deepcopy
from dataclasses import replace
from functools import wraps

import equinox as eqx
import jax.random as jr
import numpy as np
import pytest

from feedbax.builtins.training import StandardTrainingSettings, supervised_gradient
from feedbax.compiler import trial_execution
from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution, resolve_trial_execution
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.objective_document import ObjectiveDocument, ObjectiveTerm
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import PlanRequests, ProjectPlanMemo, RunKey, compile_project
from feedbax.contracts.project_plan_io import dump_plan_requests, load_plan_requests
from feedbax.contracts.request_identity import decode_request, encode_request
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.contracts.task_document import TASK_DOCUMENT_SCHEMA_ID
from feedbax.contracts.jobs import JobStatus
from feedbax.expression_builtins import register_builtins
from feedbax.expression_registry import FunctionRegistry
from feedbax.plugins.declarations import DistributionSource, callable_declaration
from feedbax.runtime._graph_key_schedule import model_initialization_key
from feedbax.tasks.document import TaskParameters
from feedbax.tasks.registry import application_python_tasks
from feedbax.training.evaluate_worker import score_paired_validation
from feedbax.training.method_runtime import TrainingLoop, initialize_method, replica_value
from feedbax.training.methods import TrainingMethodRegistry
from feedbax.web.services import project_plan as window_service
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.managed_train_fixture import managed_evaluation_step
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_task_parity import _model_payload, _pair_model, example, expression


def authored_documents():
    functions = FunctionRegistry()
    register_builtins(functions)
    functions.seal()
    model = ModelDocument.model_validate(_model_payload())
    interface, _ = _pair_model()
    task, _ = example("delayed", functions, interface=interface, reference_validation=True)
    task = task.model_copy(update={"id": "delayed-reaches"})
    (command,) = interface.outputs
    objective = ObjectiveDocument(
        id="movement-effort",
        terms=(
            ObjectiveTerm(
                id="effort", measure=expression("sqnorm(command)", (command,)), epochs=("move",)
            ),
        ),
    )
    return functions, model, task, objective


@pytest.fixture
def authored():
    return authored_documents()


@pytest.fixture
def prepared_trials(monkeypatch):
    calls = []

    @wraps(resolve_trial_execution)
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return resolve_trial_execution(*args, **kwargs)

    monkeypatch.setattr(trial_execution, "resolve_trial_execution", prepare)
    return calls


def test_bounded_train_and_evaluate_share_trial_objective(authored):
    functions, model, task, objective = authored
    registry = ComponentRegistry()
    seed = 10
    validation_key = jr.key(3)
    paired = compile_trial_execution(
        model,
        task,
        registry,
        functions,
        objective=objective,
        input_overrides={"reach_target": "target_pos"},
        initialization_key=model_initialization_key(seed),
    )
    assert isinstance(paired, PreparedTrialExecution), paired
    assert paired.objective is not None
    graph = paired.executable.graph
    assert (
        ModelAddress.from_text(paired.execution.observations[0].selector).element.end.port
        == "command"
    )
    methods = TrainingMethodRegistry()
    methods.register_declaration(
        supervised_gradient, callable_declaration(supervised_gradient),
        key="feedbax.supervised_gradient", source=DistributionSource("feedbax", "feedbax", "0.2.0"),
    )
    method = methods.resolve("feedbax.supervised_gradient")

    def observe(state, ctx):
        next_state, metrics = supervised_gradient(state, ctx)
        trials, trial_keys = ctx.sample_trials()
        (command,) = ctx.rollout(state.model, trials, trial_keys).values()
        return next_state, {**metrics, "command": command, "move_start": trials.epochs.starts[:, -1]}

    settings = StandardTrainingSettings.model_validate({"optimizer": {"type": "adam", "learning_rate": {"type": "constant_schedule", "value": 0.003}}})
    initial = initialize_method(method, settings, models=(graph,), seed=seed, replica_ids=("0",))
    result = TrainingLoop(
        replace(method, update=observe), settings, execution=paired.execution, objective=paired.objective,
        batch_size=4, seed=seed, replica_ids=("0",),
    ).run_block(initial, completed=0, count=1)
    assert not result.rejected and result.completed == 1
    command = result.metrics["command"][0, 0]
    move_start = result.metrics["move_start"][0, 0]
    assert len(np.unique(move_start)) > 1
    movement = np.arange(paired.execution.task.n_steps) >= move_start[:, None]
    oracle = np.mean(
        np.sum(np.where(movement, np.sum(command**2, axis=-1), 0), axis=-1)
        / np.sum(movement, axis=-1)
    )
    assert result.metrics["train_loss"][0, 0] == pytest.approx(float(oracle), rel=1e-5)
    assert result.metrics["grad_norm"][0, 0] > 0
    trained = replica_value(result.state.model, 0)
    assert not eqx.tree_equal(graph, trained)

    parameters = TaskParameters()
    validation = score_paired_validation(trained, paired.execution, paired.objective, parameters, key=validation_key)
    assert int(validation.objective.terms["effort"].covered) == 28
    assert validation.trials.epochs.starts.shape[0] == 28


def test_plan_round_trips_the_authored_task_and_objective(authored):
    _, model, task, objective = authored
    raw = document_project(authored)
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    assert plan.runs[0].admission_problem is None
    request_id, data = next(iter(plan.requests.items()))
    decoded = decode_request(data, plan.documents.__getitem__)
    assert decoded.id == request_id
    assert isinstance(decoded.spec, DocumentTrainingRequestSpec)
    assert decoded.spec.trial_documents.task == task
    assert decoded.spec.trial_documents.objective == objective
    assert encode_request(decoded.spec, decoded.inputs).data == data


def test_plan_refuses_each_row_with_an_unwired_linear_input(authored, prepared_trials):
    raw = document_project(authored)
    model = raw["documents"]["reach-model"]["payload"]
    controller = model["interior"]["components"]["wrapper"]["interior"]["components"]["controller"]
    del controller["interior"]["wires"]["decode"]
    name = (
        "documents.reach-model.payload.interior.components.wrapper.interior"
        ".components.controller.interior.components.readout.name"
    )
    raw["sweep"] = {"rows": {
        "base": {},
        "renamed": {"patches": {name: {"value": "Renamed readout"}}},
    }}
    _two_optimizer_rates(raw)
    plan = compile_project(Project.model_validate(raw), memo=ProjectPlanMemo())
    assert not plan.runs
    assert len(prepared_trials) == 2
    original = controller["interior"]["components"]["readout"]["name"]
    assert {
        (issue.variant, issue.step, issue.row, issue.detail)
        for issue in plan.diagnostics
        if issue.path.endswith(".components.readout")
    } == {
        (variant, "train", row, f"{label} input 'input' needs a connection.")
        for variant, label in (("base", original), ("renamed", "Renamed readout"))
        for row in ("rate=low", "rate=high")
    }


def test_plan_reports_every_component_with_invalid_parameters(authored):
    raw = document_project(authored)
    interior = raw["documents"]["reach-model"]["payload"]["interior"]
    for name in ("left", "right"):
        interior["components"][name] = {
            "kind": "leaf", "name": name.title(), "type": "feedbax.equinox.Linear", "fields": {},
        }
    plan = compile_project(Project.model_validate(raw), memo=ProjectPlanMemo())
    assert sorted(
        issue.model_address.element.id for issue in plan.diagnostics
        if issue.model_address is not None and issue.model_address.element.kind == "component"
    ) == ["left", "right"]


def test_plan_reports_every_unwired_required_input(authored):
    raw = document_project(authored)
    model = raw["documents"]["reach-model"]["payload"]
    controller = model["interior"]["components"]["wrapper"]["interior"]["components"]["controller"]
    del controller["interior"]["wires"]["decode"]
    del controller["interior"]["wires"]["target"]
    plan = compile_project(Project.model_validate(raw), memo=ProjectPlanMemo())
    assert sorted(issue.path.rsplit(".", 1)[-1] for issue in plan.diagnostics) == ["cell", "readout"]


def test_trial_refusals_keep_the_model_address_for_each_step(authored, prepared_trials):
    raw = document_project(authored)
    model = raw["documents"]["reach-model"]
    model["payload"]["interior"]["components"]["plant"]["type"] = "unknown.Plant"
    raw["documents"]["other-model"] = deepcopy(model)
    _two_optimizer_rates(raw)
    other = deepcopy(raw["steps"]["train"])
    other["base"]["inline"]["trial_documents"].update(
        model={"document": "other-model"},
    )
    raw["steps"]["other"] = other

    plan = compile_project(Project.model_validate(raw), memo=ProjectPlanMemo())
    assert not plan.runs
    assert len(prepared_trials) == 1
    assert len(plan.diagnostics) == 4
    for issue in plan.diagnostics:
        document = {"train": "reach-model", "other": "other-model"}[issue.step]
        assert issue.variant == "default"
        assert issue.row in {"rate=low", "rate=high"}
        assert issue.document == document
        assert issue.model_address == ModelAddress.model_validate({
            "composites": [], "element": {"kind": "component", "id": "plant"},
        })


def test_variant_model_field_edit_does_not_reach_inheriting_variants(tmp_path, authored):
    from fastapi.testclient import TestClient

    from feedbax.web.app import create_app
    from feedbax.web.services.project_folder import ProjectFolderService
    from tests.test_canonical_evaluate_admission import _save_project

    raw = document_project(authored)
    raw["sweep"] = {"rows": {"base": {}, "compare": {"from": "base"}, "follow": {"from": "compare"}}}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "variant", Project.model_validate(raw))
    owner = {"composites": [], "element": {"kind": "component", "id": "plant"}}
    revision = folders.read("variant").revision
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        result = client.post("/api/projects/variant/model/fields", json={
            "revision": revision, "document": "reach-model", "owner": owner, "chosen_steps": ["train"],
            "run": {"variant": "compare", "step": "train", "row": "default"},
            "address": owner | {"element": {"kind": "field", "parent": owner["element"], "path": ["fields", "mass"]}},
            "edit": {"kind": "text", "text": "2"},
        })
    assert result.status_code == 422, result.text
    assert "supplies inherited patches" in result.json()["detail"]
    assert folders.read("variant").revision == revision


def test_document_learning_rate_sweep_projects_only_after_expansion(authored, prepared_trials):
    raw = document_project(authored)
    raw["steps"]["train"]["sweep"] = {"axes": {"rate": {
        "path": "settings.optimizer.learning_rate.value",
        "range": {"start": 1e-4, "stop": 1e-2, "count": 3, "scale": "log"},
    }}}
    memo = ProjectPlanMemo()
    plan = compile_project(Project.model_validate(raw), memo=memo)
    assert not plan.diagnostics, plan.diagnostics
    assert len(plan.requests) == len(plan.runs) == 3
    assert len(prepared_trials) == 1
    reviewed = PlanRequests.from_plan(plan)
    restored, _ = load_plan_requests(dump_plan_requests(reviewed, None))
    assert restored.requests == reviewed.requests
    assert restored.documents == reviewed.documents

    rates = []
    for run in plan.runs:
        data = plan.requests[run.request_id]
        payload = json.loads(data)["spec"]
        assert "learning_rate" not in payload["training_config"]
        decoded = decode_request(data, plan.documents.__getitem__)
        assert isinstance(decoded.spec, DocumentTrainingRequestSpec)
        optimizer = decoded.spec.settings["optimizer"]
        rate = optimizer["learning_rate"]["value"]
        assert isinstance(rate, float)
        assert encode_request(decoded.spec, decoded.inputs).data == data
        rates.append(rate)
    np.testing.assert_allclose(rates, [0.0001, 0.001, 0.01])

    raw["steps"]["train"]["base"]["inline"]["training_config"]["learning_rate"] = 0.004
    before = len(prepared_trials)
    refused = compile_project(Project.model_validate(raw), memo=memo)
    assert not refused.runs
    assert len(prepared_trials) == before + 1
    assert len(refused.diagnostics) == 3
    assert all(problem.path == "payload.training_config.learning_rate" for problem in refused.diagnostics)


def _two_optimizer_rates(raw):
    path = "settings.optimizer.learning_rate.value"
    raw["steps"]["train"]["sweep"] = {"axes": {"rate": {"values": {
        name: {"patches": {path: {"value": value}}}
        for name, value in (("low", 0.001), ("high", 0.01))
    }}}}


def test_shared_trial_resolution_tracks_document_aliases_and_row_edits(authored):
    raw = document_project(authored)
    _two_optimizer_rates(raw)
    raw["documents"]["task-alias"] = {
        "schema": "test.alias", "payload": {"nested": {"document": "task"}},
    }
    step = raw["steps"]["train"]
    step["base"]["inline"]["trial_documents"]["task"] = {"document": "task-alias", "pointer": "/nested"}
    low, high = (RunKey("default", "train", f"rate={name}") for name in ("low", "high"))
    memo = ProjectPlanMemo()

    def compile_both():
        project = Project.model_validate(raw)
        fast = compile_project(project, memo=memo)
        tracked = compile_project(project, source_runs=(low, high))
        assert fast == replace(tracked, selected_definitions={})
        return fast, tracked

    first, _ = compile_both()
    assert not first.diagnostics, first.diagnostics
    assert compile_both()[0] == first
    assert all({"task-alias", "task"} <= row.document_uses for row in first.runs)

    raw["documents"]["task"]["payload"]["id"] = "edited-task"
    edited, tracked = compile_both()
    assert {run.request_id for run in first.runs}.isdisjoint(edited.requests)
    assert tracked.sources[high]["/payload/trial_documents/task/id"].name == "task"

    values = step["sweep"]["axes"]["rate"]["values"]
    values["high"]["patches"]["documents.task.payload.id"] = {"value": "row-task"}
    patched, tracked = compile_both()
    assert not patched.diagnostics
    assert patched.runs[0].request_id == edited.runs[0].request_id
    assert patched.runs[1].request_id != edited.runs[1].request_id
    assert tracked.sources[high]["/payload/trial_documents/task/id"].kind == "sweep_value"

    values["high"]["patches"]["training_config.learning_rate"] = {"value": 0.1, "op": "add"}
    refused, _ = compile_both()
    assert [run.key for run in refused.runs] == [low]
    assert [(problem.row, problem.path) for problem in refused.diagnostics] == [
        (high.row, "payload.training_config.learning_rate"),
    ]


def test_saved_window_reuses_encoded_document_conditions(authored, tmp_path, monkeypatch):
    raw = document_project(authored)
    _two_optimizer_rates(raw)
    folders = ProjectFolderService(tmp_path)
    evaluation_tasks = application_python_tasks()

    def projected(view):
        return view.window_map(evaluation_tasks=evaluation_tasks).model_dump(mode="json")

    def read():
        _save_project(folders, "cohort", Project.model_validate(raw))
        return window_service.ProjectPlanRead.without_history(folders.read_plan("cohort", revision=None))

    def verify(view):
        with monkeypatch.context() as patch:
            def unexpected_decode(*args, **kwargs):
                pytest.fail("Window decoded a Plan Train request")

            patch.setattr(window_service, "decode_request", unexpected_decode)
            return projected(view)

    original = read()
    before = verify(original)
    conditions = before["rows"][0]["evaluation_conditions"]
    assert conditions and all(row["evaluation_conditions"] == conditions for row in before["rows"])
    assert all(row["status"] == JobStatus.NOT_RUN for row in before["rows"])
    assert projected(window_service.ProjectPlanRead.without_history(
        folders.read_plan("cohort", revision=original.settled.revision),
    )) == before
    variables = raw["documents"]["task"]["payload"]["validation_trials"][0]["variables"]
    target = variables["target"]
    assert isinstance(target, list)
    target[:] = [1.0, -0.0]
    after = verify(read())
    assert after["rows"][0]["evaluation_conditions"] != conditions
    assert projected(original) == before


def test_trial_preparation_distinguishes_effective_scientific_inputs(authored, prepared_trials):
    raw = document_project(authored)
    mass = ModelAddress.model_validate({
        "composites": [],
        "element": {"kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"]},
    })
    task = "documents.task.payload.id"
    objective = "documents.objective.payload.terms.0.weight.value"
    binding = "steps.train.base.inline.trial_documents.input_overrides.reach_target"
    raw["sweep"] = {"rows": {
        "base": {},
        "model": {"patches": {f"documents.reach-model.{mass.to_text()}": {"value": 2.0}}},
        "task": {"patches": {task: {"value": "alternate-task"}}},
        "objective": {"patches": {objective: {"value": 2}}},
        "bad-binding": {"patches": {binding: {"value": "missing-signal"}}},
    }}
    _two_optimizer_rates(raw)
    plan = compile_project(Project.model_validate(raw), memo=ProjectPlanMemo())
    original_mass = authored[1].interior.components["plant"].fields["mass"]
    assert len(prepared_trials) == 5
    assert {
        (
            model.interior.components["plant"].fields["mass"], task_source.id,
            kwargs["objective"].terms[0].weight.value, kwargs["input_overrides"]["reach_target"],
        )
        for (model, task_source, *_), kwargs in prepared_trials
    } == {
        (original_mass, "delayed-reaches", 1, "target_pos"),
        (2.0, "delayed-reaches", 1, "target_pos"),
        (original_mass, "alternate-task", 1, "target_pos"),
        (original_mass, "delayed-reaches", 2, "target_pos"),
        (original_mass, "delayed-reaches", 1, "missing-signal"),
    }
    assert len(plan.requests) == len(plan.runs) == 8
    assert len(plan.diagnostics) == 2
    assert {(issue.variant, issue.row) for issue in plan.diagnostics} == {
        ("bad-binding", "rate=low"), ("bad-binding", "rate=high"),
    }


def test_managed_train_stores_only_scientific_model_names(authored):
    from feedbax.contracts.canonical_json import canonical_json_v2_bytes
    from feedbax.contracts.request_identity import request_id

    plan = compile_project(Project.model_validate(document_project(authored)))
    identity, data = next(iter(plan.requests.items()))
    stored = json.loads(data)
    assert stored["schema"] == "feedbax.request.train.documents.v3"
    model = stored["spec"]["trial_documents"]["model"]
    assert model["ports"]["inputs"][0]["name"] == "reach_target"
    wrapper = model["interior"]["components"]["wrapper"]
    assert "name" not in wrapper
    assert all("name" not in port for port in wrapper["ports"]["inputs"])
    assert all("name" not in port for port in wrapper["ports"]["outputs"])
    decoded = decode_request(data, plan.documents.__getitem__)
    assert decoded.id == identity
    restored = decoded.spec.trial_documents.model
    assert restored.scientific_projection() == authored[1].scientific_projection()
    assert encode_request(decoded.spec, decoded.inputs).data == data
    assert identity != request_id("train", stored["spec"], decoded.inputs)

    previous = stored | {"schema": "feedbax.request.train.documents.v2"}
    with pytest.raises(ValueError, match="Unsupported stored request envelope"):
        decode_request(canonical_json_v2_bytes(previous), plan.documents.__getitem__)
    wrapper["name"] = "wrapper"
    with pytest.raises(ValueError, match="did not round-trip"):
        decode_request(canonical_json_v2_bytes(stored), plan.documents.__getitem__)


def document_project(authored, *, evaluation=False):
    from feedbax.contracts.objective_document import OBJECTIVE_DOCUMENT_SCHEMA_ID

    _, model, task, objective = authored
    raw = {
        "documents": {
            "reach-model": {"schema": MODEL_DOCUMENT_SCHEMA_ID, "payload": model.model_dump(mode="json")},
            "task": {"schema": TASK_DOCUMENT_SCHEMA_ID, "payload": task.model_dump(mode="json")},
            "objective": {
                "schema": OBJECTIVE_DOCUMENT_SCHEMA_ID,
                "payload": objective.model_dump(mode="json"),
            },
        },
        "steps": {
            "train": {"kind": "train", "base": {"inline": {
                "trial_documents": {
                    "model": {"document": "reach-model"},
                    "task": {"document": "task"},
                    "objective": {"document": "objective"},
                    "input_overrides": {"reach_target": "target_pos"},
                },
                "training_config": {"n_batches": 2, "batch_size": 4},
                "method": "feedbax.supervised_gradient",
                "settings": {"optimizer": {"type": "adam", "learning_rate": {"type": "constant_schedule", "value": 0.001}}},
            }}},
        },
    }
    if evaluation:
        raw["steps"]["baseline"] = managed_evaluation_step("train")
    return raw


def test_managed_host_train_and_evaluate_share_document_runtime(host, tmp_path, authored, monkeypatch):
    from feedbax.contracts.project_plan import RunKey
    from feedbax.jobs import JobService
    from feedbax.plugins.composition import load_application
    from feedbax.web.services.project_folder import ProjectFolderService
    from tests.test_canonical_evaluate_admission import _save_project
    from tests.support.host import job_row, token, wait_until

    registry = load_application().bundle
    jobs = JobService(host, registry)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "document-pair"
    raw = document_project(authored, evaluation=True)
    payload = raw["steps"]["train"]["base"]["inline"]["settings"]
    rate = 0.003
    payload["optimizer"]["learning_rate"]["value"] = rate
    _save_project(folders, project, Project.model_validate(raw))
    plan = folders.read_plan(project, revision=None)
    train = RunKey("default", "train", "default")
    evaluate = RunKey("default", "baseline", "default")
    (train_job,) = jobs.admit_plan(folders, plan, run_keys=(train,), token=token("paired-train")).items
    assert train_job.disposition == "admitted", train_job
    (accepted,) = host.jobs(job_id=train_job.job_id, include_launch=True)
    prepared = accepted["launch"]["payload"]["spec"]
    assert prepared["settings"]["optimizer"]["learning_rate"]["value"] == rate
    # A GPU venue passes its backend and x64 settings to ordinary Jobs admission.
    with monkeypatch.context() as venue:
        venue.setenv("JAX_PLATFORMS", "cuda")
        venue.setenv("JAX_ENABLE_X64", "1")
        venue_jobs = JobService(host, registry)
    (evaluation_job,) = venue_jobs.admit_plan(
        folders, plan, run_keys=(evaluate,), token=token("paired-evaluate")
    ).items
    assert evaluation_job.disposition == "admitted", evaluation_job
    (accepted_evaluation,) = host.jobs(job_id=evaluation_job.job_id, include_launch=True)
    assert accepted_evaluation["launch"]["env"]["JAX_PLATFORMS"] == "cpu"
    assert accepted_evaluation["launch"]["env"]["JAX_ENABLE_X64"] == "0"

    def finished(job):
        return wait_until(
            lambda: (job_row(host, job.job_id, "succeeded") or job_row(host, job.job_id, "failed")),
        )

    trained = finished(train_job)
    assert trained["state"] == "succeeded", trained.get("outcome")
    evaluated = finished(evaluation_job)
    assert evaluated["state"] == "succeeded", evaluated.get("outcome")
    result = jobs.read_evaluation_result(
        project=plan.project_id, store=plan.store_path, job_id=evaluation_job.job_id
    )
    with result.open_states() as dataset:
        assert np.isfinite(dataset.data("objective/total")).all()
        assert int(dataset.data("objective/coverage/effort").item()) == 28


def _training_document(raw, *steps):
    """Move the first step's method, settings and run controls into a `training` document the steps read."""
    from feedbax.contracts.training_document import TRAINING_DOCUMENT_KEYS, TRAINING_DOCUMENT_SCHEMA_ID

    inline = raw["steps"][steps[0]]["base"]["inline"]
    raw["documents"]["training"] = {
        "schema": TRAINING_DOCUMENT_SCHEMA_ID, "payload": {key: inline[key] for key in TRAINING_DOCUMENT_KEYS},
    }
    for step in steps:
        raw["steps"][step]["base"]["inline"] |= {
            key: {"document": "training", "pointer": f"/{key}"} for key in TRAINING_DOCUMENT_KEYS
        }
    return raw


def _requests(raw, step):
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    return {run.key.row: run.request_id for run in plan.runs if run.key.step == step}


def test_training_document_sweep_compiles_as_inline_settings(authored):
    inline = document_project(authored)
    _two_optimizer_rates(inline)
    shared = _training_document(deepcopy(inline), "train")
    assert len(_requests(inline, "train")) == 2
    assert _requests(shared, "train") == _requests(inline, "train")

    shared["steps"]["other"] = deepcopy(shared["steps"]["train"]) | {"sweep": None}
    stored = deepcopy(shared["documents"]["training"])
    unswept = deepcopy(shared)
    unswept["steps"]["train"]["sweep"] = None
    assert _requests(shared, "other") == _requests(unswept, "other")
    assert len(set(_requests(shared, "train").values())) == 2
    assert shared["documents"]["training"] == stored

    keys = [RunKey("default", "train", f"rate={name}") for name in ("low", "high")]
    tracked = compile_project(Project.model_validate(shared), source_runs=tuple(keys))
    rate = "/payload/settings/optimizer/learning_rate/value"
    for key in keys:
        definition = tracked.selected_definitions[key]
        assert definition.document_ids["payload.settings"] == "training"
        assert "training" in definition.document_uses
        assert tracked.sources[key][rate].kind == "sweep_value"


def test_variant_write_through_a_step_reference_names_the_document_form(authored):
    raw = _training_document(document_project(authored), "train")
    raw["sweep"] = {"rows": {"base": {}, "fast": {"patches": {
        "steps.train.base.inline.settings.optimizer.learning_rate.value": {"value": 0.01},
    }}}}
    plan = compile_project(Project.model_validate(raw))
    (problem,) = plan.diagnostics
    assert (problem.variant, problem.step) == ("fast", "train")
    assert "documents.training.payload.settings.optimizer.learning_rate.value" in problem.detail

    raw["sweep"]["rows"]["fast"]["patches"] = {
        "documents.training.payload.settings.optimizer.learning_rate.value": {"value": 0.01},
    }
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    assert len({run.request_id for run in plan.runs}) == 2


@pytest.mark.parametrize(("key", "value", "path", "place"), [
    ("settings", {"optimizer": {"type": "adam", "learning_rate": {"type": "constant_schedule", "value": "fast"}}},
     "/settings/optimizer", "method"),
    ("method", "missing.method", "/method", "method"),
])
def test_training_document_problem_is_placed_in_the_document(authored, key, value, path, place):
    from feedbax.contracts.studio_place import definition_place, path_parts

    raw = _training_document(document_project(authored), "train")
    raw["documents"]["training"]["payload"][key] = value
    plan = compile_project(Project.model_validate(raw))
    assert plan.diagnostics
    for problem in plan.diagnostics:
        assert problem.document == "training"
        assert problem.document_path.startswith(path)
        located = definition_place(path_parts(problem.path), lambda _slot: problem.document)
        assert (located.part, located.document) == (place, "training")
