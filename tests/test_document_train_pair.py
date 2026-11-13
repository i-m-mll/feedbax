"""The document pair uses the native update and fixed validation scorer."""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.objective_document import ObjectiveDocument, ObjectiveTerm
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import PlanRequests, compile_project
from feedbax.contracts.project_plan_io import dump_plan_requests, load_plan_requests
from feedbax.contracts.request_identity import decode_request, encode_request
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec, TrainingRequestSpec
from feedbax.contracts.task_document import TASK_DOCUMENT_SCHEMA_ID
from feedbax.expression_builtins import register_builtins
from feedbax.expression_registry import FunctionRegistry
from feedbax.tasks.document import TaskParameters
from feedbax.training.evaluate_worker import score_paired_validation
from feedbax.training.managed_studio import prepare_trial_update
from tests.test_document_task_parity import _model_payload, _pair_model, example, expression, legacy
from tests.test_model_document_plan import _model_project
from tests.test_local_job_host import host as host_fixture  # noqa: F401


def authored_documents():
    functions = FunctionRegistry()
    register_builtins(functions)
    functions.seal()
    model = ModelDocument.model_validate(_model_payload())
    interface, _ = _pair_model()
    task, _ = example("delayed", functions, interface=interface, old=legacy("delayed"))
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
    raw = _model_project()
    raw["steps"].pop("baseline")
    raw["documents"]["reach-model"]["payload"] = model.model_dump(mode="json")
    legacy_plan = compile_project(Project.model_validate(raw))
    legacy_request = next(iter(legacy_plan.requests.values()))
    decoded = decode_request(legacy_request, legacy_plan.documents.__getitem__)
    assert decoded.spec.trial_documents is None
    assert encode_request(decoded.spec, decoded.inputs).data == legacy_request

    assert legacy_plan.runs[0].admission_problem.path == "/trial_documents"
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


def test_document_learning_rate_sweep_projects_only_after_expansion(authored):
    from feedbax.plugins.application import new_application_registry_bundle
    from feedbax.studio.canonical_train import lower_canonical_train

    raw = document_project(authored)
    raw["steps"]["train"]["sweep"] = {"axes": {"rate": {
        "path": "method_payload.payload.optimizer.params.learning_rate",
        "range": {"start": 1e-4, "stop": 1e-2, "count": 3, "scale": "log"},
    }}}
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    assert len(plan.requests) == len(plan.runs) == 3
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
    refused = compile_project(Project.model_validate(raw))
    assert not refused.runs
    assert any(
        problem.path == "payload.training_config.learning_rate" for problem in refused.diagnostics
    )


def test_previous_document_train_bytes_and_mirror_are_preserved(authored):
    from feedbax.plugins.application import new_application_registry_bundle
    from feedbax.studio.canonical_train import lower_canonical_train

    raw = document_project(authored)
    plan = compile_project(Project.model_validate(raw))
    data = next(iter(plan.requests.values()))
    managed = decode_request(data, plan.documents.__getitem__).spec
    assert isinstance(managed, DocumentTrainingRequestSpec)
    previous_payload = managed.model_dump(mode="json") | {
        "graph": None, "task": None, "objective": None,
    }
    previous_payload["method_payload"]["payload"]["optimizer"]["params"]["learning_rate"] = 0.004
    previous_payload["training_config"]["learning_rate"] = 0.004
    previous = TrainingRequestSpec.model_validate(previous_payload)
    encoded = encode_request(previous, {})
    assert encoded.id == "train:da9a620135a874b2e33809b91c3d704c527dd1e91db0bcd0575cf19bdeb2a9e6"
    decoded = decode_request(encoded.data, encoded.documents.__getitem__)
    assert isinstance(decoded.spec, TrainingRequestSpec)
    assert encode_request(decoded.spec, decoded.inputs) == encoded

    registry = new_application_registry_bundle()
    restored = lower_canonical_train(decoded.spec, registry.training_programs, registries=registry)
    assert restored.training_config.learning_rate == 0.004
    decoded.spec.training_config.learning_rate = 0.005
    with pytest.raises(ValueError, match="/training_config/learning_rate"):
        lower_canonical_train(decoded.spec, registry.training_programs, registries=registry)


def document_project(authored, *, evaluation=False):
    _, model, task, objective = authored
    raw = _model_project()
    if not evaluation:
        raw["steps"].pop("baseline")
    raw["documents"]["reach-model"]["payload"] = model.model_dump(mode="json")
    from feedbax.contracts.objective_document import OBJECTIVE_DOCUMENT_SCHEMA_ID

    raw["documents"].update(
        {
            "task": {"schema": TASK_DOCUMENT_SCHEMA_ID, "payload": task.model_dump(mode="json")},
            "objective": {
                "schema": OBJECTIVE_DOCUMENT_SCHEMA_ID,
                "payload": objective.model_dump(mode="json"),
            },
        }
    )
    inline = raw["steps"]["train"]["base"]["inline"]
    inline.pop("graph")
    inline.pop("task_binding_spec")
    inline.pop("task")
    inline.pop("objective")
    inline["trial_documents"] = {
        "model_name": "reach-model",
        "model": {"document": "reach-model"},
        "task": {"document": "task"},
        "objective": {"document": "objective"},
        "input_overrides": {"reach_target": "target_pos"},
    }
    inline["training_config"] = {"n_batches": 2, "n_reach_steps": 80, "batch_size": 4}
    return raw


def test_entry_points_refuse_the_other_request_family(authored, tmp_path):
    from feedbax.studio.canonical_train import lower_canonical_train
    from feedbax.training.executor import TrainingRunExecutorError, execute_training_run_spec
    from feedbax.plugins.application import new_application_registry_bundle
    from tests.test_studio_training_host_worker import _authored_request

    registries = new_application_registry_bundle()
    with pytest.raises(ValueError, match="/trial_documents"):
        lower_canonical_train(
            _authored_request(), registries.training_programs, registries=registries
        )
    plan = compile_project(Project.model_validate(document_project(authored)))
    request = decode_request(next(iter(plan.requests.values())), plan.documents.__getitem__).spec
    spec = lower_canonical_train(request, registries.training_programs, registries=registries)
    with pytest.raises(TrainingRunExecutorError, match="/trial_documents"):
        execute_training_run_spec(
            spec, registry=registries.training_programs, manifest_root=tmp_path
        )


def test_managed_host_train_and_evaluate_share_document_runtime(host_fixture, tmp_path, authored):  # noqa: F811
    host = host_fixture
    from feedbax.contracts.project_plan import RunKey
    from feedbax.jobs import JobService
    from feedbax.plugins.application import new_application_registry_bundle
    from feedbax.web.services.project_folder import ProjectFolderService
    from tests.test_canonical_evaluate_admission import _save_project
    from tests.test_local_job_host import job_row, wait_until

    registry = new_application_registry_bundle()
    jobs = JobService(host, registry, enable_evaluation_candidate=True)
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
