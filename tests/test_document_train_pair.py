"""The document pair uses the native update and fixed validation scorer."""

import json
from copy import deepcopy
from dataclasses import replace
from functools import wraps

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from feedbax.compiler import trial_execution
from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.objective_document import ObjectiveDocument, ObjectiveTerm
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import PlanRequests, ProjectPlanMemo, compile_project
from feedbax.contracts.project_plan_io import dump_plan_requests, load_plan_requests
from feedbax.contracts.request_identity import decode_request, encode_request
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.contracts.rollout import ROLLOUT_TYPE
from feedbax.contracts.task_document import TASK_DOCUMENT_SCHEMA_ID
from feedbax.contracts.jobs import JobStatus
from feedbax.expression_builtins import register_builtins
from feedbax.expression_registry import FunctionRegistry
from feedbax.tasks.document import TaskParameters
from feedbax.tasks.reaching import builtin_python_tasks
from feedbax.training.evaluate_worker import score_paired_validation
from feedbax.training.managed_studio import prepare_trial_update
from feedbax.web.services import project_plan as window_service
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_task_parity import _model_payload, _pair_model, example, expression
from tests.test_local_job_host import host as host_fixture  # noqa: F401


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

    @wraps(compile_trial_execution)
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return compile_trial_execution(*args, **kwargs)

    monkeypatch.setattr(trial_execution, "compile_trial_execution", prepare)
    return calls


def test_bounded_train_and_evaluate_share_trial_objective(authored):
    functions, model, task, objective = authored
    registry = ComponentRegistry()
    keys = jr.split(jr.PRNGKey(10), 3)
    paired = compile_trial_execution(
        "reach-model",
        model,
        task,
        registry,
        functions,
        objective=objective,
        input_overrides={"reach_target": "target_pos"},
        initialization_key=keys[0],
    )
    assert isinstance(paired, PreparedTrialExecution), paired
    assert paired.objective is not None
    graph = paired.executable.graph
    assert (
        ModelAddress.from_text(paired.execution.observations[0].selector).element.end.port
        == "command"
    )
    opt = optax.adam(0.003)
    parameters = TaskParameters()
    kernel = prepare_trial_update(
        graph,
        paired.execution,
        paired.objective,
        parameters=parameters,
        run_key=keys[1],
        batch_size=4,
        trainable_filter=eqx.is_inexact_array,
        optimizer=opt,
    )
    slots = {
        "model": graph,
        "optimizer": opt.init(eqx.filter(graph, eqx.is_inexact_array)),
        "prng": keys[2],
        "batch_counter": jnp.array(0, dtype=jnp.int32),
    }
    _, update_key = jr.split(keys[2])
    expected = paired.execution.training_batch(
        graph,
        paired.objective,
        parameters,
        run_key=keys[1],
        update_key=update_key,
        count=4,
    )
    assert int(expected.terms["effort"].covered) == 4
    from feedbax.runtime._graph_key_schedule import sample_training_trials

    def score_one(key):
        trial = paired.execution.task.ahead(parameters, keys[1], key)
        return trial, key

    trials, trial_keys = sample_training_trials(score_one, update_key, 4)
    assert len(np.unique(np.asarray(trials.epochs.starts[:, 1]))) > 1
    scored = paired.execution.score(
        graph, paired.objective, trials, run_key=keys[1], trial_keys=trial_keys
    )
    (command,) = scored.retained.values()
    movement = jnp.arange(paired.execution.task.n_steps) >= trials.epochs.starts[:, -1, None]
    oracle = jnp.mean(
        jnp.sum(jnp.where(movement, jnp.sum(command**2, axis=-1), 0), axis=-1)
        / jnp.sum(movement, axis=-1)
    )
    assert float(expected.loss) == pytest.approx(float(oracle), rel=1e-5)
    update = kernel(slots)
    assert float(update["train_loss"]) == pytest.approx(float(expected.loss), rel=1e-5)
    assert float(update["grad_norm"]) > 0
    assert int(update["batch_counter"]) == 1
    assert not eqx.tree_equal(graph, update["model"])

    uninterrupted = kernel(update)
    restored = jax.tree.map(
        lambda value: jnp.asarray(np.asarray(value)) if eqx.is_array(value) else value, update
    )
    resumed = kernel(restored)
    for name in ("model", "optimizer", "prng", "batch_counter"):
        assert eqx.tree_equal(uninterrupted[name], resumed[name])

    validation = score_paired_validation(
        resumed["model"],
        paired.execution,
        paired.objective,
        parameters,
        key=keys[1],
    )
    repeated = paired.execution.validation(
        resumed["model"],
        paired.objective,
        parameters,
        key=keys[1],
    )
    assert eqx.tree_equal(validation.objective, repeated.objective)
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


def test_trial_refusals_keep_the_model_address_for_each_step(authored, prepared_trials):
    raw = document_project(authored)
    model = raw["documents"]["reach-model"]
    model["payload"]["interior"]["components"]["plant"]["type"] = "unknown.Plant"
    raw["documents"]["other-model"] = deepcopy(model)
    _two_optimizer_rates(raw)
    other = deepcopy(raw["steps"]["train"])
    other["base"]["inline"]["trial_documents"].update(
        model_name="other-model", model={"document": "other-model"},
    )
    raw["steps"]["other"] = other

    plan = compile_project(Project.model_validate(raw), memo=ProjectPlanMemo())
    assert not plan.runs
    assert len(prepared_trials) == 2
    assert len(plan.diagnostics) == 4
    for issue in plan.diagnostics:
        document = {"train": "reach-model", "other": "other-model"}[issue.step]
        assert issue.variant == "default"
        assert issue.row in {"rate=low", "rate=high"}
        assert issue.document == document
        assert issue.model_address == ModelAddress.model_validate({
            "document": document, "composites": [], "element": {"kind": "component", "id": "plant"},
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
    owner = {"document": "reach-model", "composites": [], "element": {"kind": "component", "id": "plant"}}
    revision = folders.read("variant").revision
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        result = client.post("/api/projects/variant/model/fields", json={
            "revision": revision, "owner": owner, "lit_steps": ["train"],
            "run": {"variant": "compare", "step": "train", "row": "default"},
            "address": owner | {"element": {"kind": "field", "parent": owner["element"], "path": ["fields", "mass"]}},
            "edit": {"kind": "text", "text": "2"},
        })
    assert result.status_code == 422, result.text
    assert "supplies inherited patches" in result.json()["detail"]
    assert folders.read("variant").revision == revision


def test_document_learning_rate_sweep_projects_only_after_expansion(authored, prepared_trials):
    from feedbax.plugins.application import new_application_registry_bundle
    from feedbax.studio.canonical_train import lower_canonical_train

    raw = document_project(authored)
    raw["steps"]["train"]["sweep"] = {"axes": {"rate": {
        "path": "method_payload.payload.optimizer.params.learning_rate",
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

    registry = new_application_registry_bundle()
    rates = []
    for run in plan.runs:
        data = plan.requests[run.request_id]
        payload = json.loads(data)["spec"]
        assert "learning_rate" not in payload["training_config"]
        decoded = decode_request(data, plan.documents.__getitem__)
        assert isinstance(decoded.spec, DocumentTrainingRequestSpec)
        optimizer = decoded.spec.method_payload.payload["optimizer"]
        rate = optimizer["params"]["learning_rate"]
        assert isinstance(rate, float)
        assert encode_request(decoded.spec, decoded.inputs).data == data
        spec = lower_canonical_train(decoded.spec, registry.training_programs, registries=registry)
        assert spec.training_config.learning_rate == rate
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
    path = "method_payload.payload.optimizer.params.learning_rate"
    raw["steps"]["train"]["sweep"] = {"axes": {"rate": {"values": {
        name: {"patches": {path: {"value": value}}}
        for name, value in (("low", 0.001), ("high", 0.01))
    }}}}


def test_saved_window_reuses_encoded_document_conditions(authored, tmp_path, monkeypatch):
    raw = document_project(authored)
    _two_optimizer_rates(raw)
    folders = ProjectFolderService(tmp_path)
    evaluation_tasks = builtin_python_tasks()

    def projected(view):
        return view.window_map(evaluation_tasks=evaluation_tasks).model_dump(mode="json")

    def read():
        _save_project(folders, "cohort", Project.model_validate(raw))
        return window_service.ProjectPlanRead.without_history(folders.read_plan("cohort", revision=None))

    def verify(view):
        settled = view.settled
        plan = settled.plan
        baseline = replace(view, settled=replace(settled, plan=replace(plan, _train_evaluation_sources={})))
        original = projected(baseline)
        with monkeypatch.context() as patch:
            def unexpected_decode(*args, **kwargs):
                pytest.fail("Window decoded a Plan Train request")

            patch.setattr(window_service, "decode_request", unexpected_decode)
            assert projected(view) == original
        return original

    original = read()
    before = verify(original)
    conditions = before["rows"][0]["evaluation_conditions"]
    assert conditions and all(row["evaluation_conditions"] == conditions for row in before["rows"])
    assert all(row["status"] == JobStatus.NOT_RUN for row in before["rows"])
    assert projected(window_service.ProjectPlanRead.without_history(
        folders.read_plan("cohort", revision=original.settled.revision),
    )) == before
    settled = original.settled
    plan = settled.plan
    digest = next(iter(plan.documents))
    altered = replace(plan, documents={**plan.documents, digest: b"{}"})
    with pytest.raises(ValueError, match="Invalid request document bytes"):
        projected(replace(original, settled=replace(settled, plan=altered)))

    variables = raw["documents"]["task"]["payload"]["validation_trials"][0]["variables"]
    target = variables["target"]
    assert isinstance(target, list)
    target[:] = [1.0, -0.0]
    after = verify(read())
    assert after["rows"][0]["evaluation_conditions"] != conditions
    assert projected(original) == before


@pytest.mark.parametrize("wide, reject", [(1e20, True), (1e21, False)])
def test_saved_window_keeps_verification_for_wide_float_requests(authored, tmp_path, wide, reject):
    raw = document_project(authored)
    raw["steps"]["train"]["base"]["inline"]["method_payload"]["metadata"] = {"value": wide}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "wide", Project.model_validate(raw))
    settled = folders.read_plan("wide", revision=None)
    view = window_service.ProjectPlanRead.without_history(settled)
    (run,) = settled.plan.runs
    assert settled.plan.train_evaluation_source(run.request_id) is None
    evaluation_tasks = builtin_python_tasks()
    if reject:
        with pytest.raises(ValueError, match="unsafe_integer"):
            view.window_map(evaluation_tasks=evaluation_tasks)
    else:
        (row,) = view.window_map(evaluation_tasks=evaluation_tasks).rows
        assert row.evaluation_conditions


def test_trial_preparation_distinguishes_effective_scientific_inputs(authored, prepared_trials):
    raw = document_project(authored)
    mass = ModelAddress.model_validate({
        "document": "reach-model", "composites": [],
        "element": {"kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"]},
    })
    task = "documents.task.payload.id"
    objective = "documents.objective.payload.terms.0.weight.value"
    binding = "steps.train.base.inline.trial_documents.input_overrides.reach_target"
    raw["sweep"] = {"rows": {
        "base": {},
        "model": {"patches": {f"steps.train.{mass.to_text()}": {"value": 2.0}}},
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
        for (_, model, task_source, *_), kwargs in prepared_trials
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
    assert stored["schema"] == "feedbax.request.train.documents.v2"
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

    previous = stored | {"schema": "feedbax.request.train.documents.v1"}
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
                    "model_name": "reach-model",
                    "model": {"document": "reach-model"},
                    "task": {"document": "task"},
                    "objective": {"document": "objective"},
                    "input_overrides": {"reach_target": "target_pos"},
                },
                "training_config": {"n_batches": 2, "n_reach_steps": 80, "batch_size": 4},
                "method_ref": {"package": "feedbax", "name": "studio_graph", "version": "v1"},
                "method_payload": {
                    "schema_id": "feedbax.spec.training_method.studio_graph",
                    "schema_version": "feedbax.spec.training_method.studio_graph.v1",
                    "payload": {
                        "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}},
                        "target_total": 2,
                        "key_schedule": "independent_v1",
                    },
                },
            }}},
        },
    }
    if evaluation:
        raw["steps"]["baseline"] = {
            "kind": "evaluate",
            "base": {"inline": {
                "evaluation_type": ROLLOUT_TYPE,
                "params": {
                    "seed": 731, "trial_policy": "paired", "variation": [],
                    "trial_set": {"kind": "every_condition"},
                },
            }},
            "inputs": {"subject": {"step": "train", "checkpoint": "last"}},
        }
    return raw


def test_managed_host_train_and_evaluate_share_document_runtime(host_fixture, tmp_path, authored):  # noqa: F811
    host = host_fixture
    from feedbax.contracts.project_plan import RunKey
    from feedbax.jobs import JobService
    from feedbax.plugins.application import new_application_registry_bundle
    from feedbax.web.services.project_folder import ProjectFolderService
    from tests.test_canonical_evaluate_admission import _save_project
    from tests.test_local_job_host import job_row, wait_until

    registry = new_application_registry_bundle()
    jobs = JobService(host, registry)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "document-pair"
    raw = document_project(authored, evaluation=True)
    payload = raw["steps"]["train"]["base"]["inline"]["method_payload"]["payload"]
    rate = 0.003
    payload["optimizer"]["params"]["learning_rate"] = rate
    _save_project(folders, project, Project.model_validate(raw))
    plan = folders.read_plan(project, revision=None)
    train = RunKey("default", "train", "default")
    evaluate = RunKey("default", "baseline", "default")
    (train_job,) = jobs.admit_plan(folders, plan, run_keys=(train,), token="paired-train").items
    assert train_job.disposition == "admitted", train_job
    (accepted,) = host.jobs(job_id=train_job.job_id, include_launch=True)
    prepared = accepted["launch"]["payload"]["spec"]
    assert prepared["training_config"]["learning_rate"] == rate
    assert prepared["method_payload"]["payload"]["optimizer"]["params"]["learning_rate"] == rate
    (evaluation_job,) = jobs.admit_plan(
        folders, plan, run_keys=(evaluate,), token="paired-evaluate"
    ).items
    assert evaluation_job.disposition == "admitted", evaluation_job

    def finished(job):
        return wait_until(
            lambda: (job_row(host, job.job_id, "succeeded") or job_row(host, job.job_id, "failed")),
            timeout=26,
        )

    trained = finished(train_job)
    assert trained["state"] == "succeeded", trained.get("outcome")
    evaluated = finished(evaluation_job)
    assert evaluated["state"] == "succeeded", evaluated.get("outcome")
    result = jobs.read_evaluation_result(
        project=project, store=plan.store_path, job_id=evaluation_job.job_id
    )
    with result.open_states() as dataset:
        assert np.isfinite(dataset.data("objective/total")).all()
        assert int(dataset.data("objective/coverage/effort").item()) == 28
