"""Exercise the installed examples through ordinary project and training entry points."""

from math import isfinite

import equinox as eqx
import jax.random as jr
import numpy as np

from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.objective_document import OBJECTIVE_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project_plan import compile_project
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import TrainingRequestSpec
from feedbax.contracts.task_document import TASK_DOCUMENT_SCHEMA_ID
from feedbax.examples import load_shipped_projects
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_train import lower_canonical_train
from feedbax.tasks.document import TaskParameters
from feedbax.training.managed_studio import prepare_managed_training


_VALIDATION_COUNTS = {"simple-reaches": 28, "delayed-reaches": 28, "stabilization": 4}
_REACH_TERMS = {"effector_position", "effector_final_velocity", "nn_hidden", "nn_output"}


def test_shipped_trial_documents_have_unique_names():
    projects = load_shipped_projects()
    assert "point-mass-reaching" in projects
    seen: dict[tuple[str, str], str] = {}
    for folder, project in projects.items():
        for name, document in project.documents.items():
            entry = document.schema_id, name
            assert entry not in seen, (entry, seen[entry], folder)
            seen[entry] = folder

    assert seen[MODEL_DOCUMENT_SCHEMA_ID, "point-mass"] == "point-mass-reaching"
    for name in _VALIDATION_COUNTS:
        assert seen[TASK_DOCUMENT_SCHEMA_ID, f"{name}-task"] == "point-mass-reaching"
        assert seen[OBJECTIVE_DOCUMENT_SCHEMA_ID, f"{name}-objective"] == "point-mass-reaching"


def test_shipped_trial_examples_plan_train_and_score_validation():
    project = load_shipped_projects()["point-mass-reaching"]
    registries = new_application_registry_bundle()
    plan = compile_project(
        project, components=registries.components, functions=registries.expression_functions,
    )
    assert not plan.diagnostics, plan.diagnostics
    assert {run.key.step for run in plan.runs} == _VALIDATION_COUNTS.keys()

    for run in plan.runs:
        name = run.key.step
        assert run.admission_problem is None, run.admission_problem
        request = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
        assert isinstance(request, TrainingRequestSpec)
        assert request.trial_documents is not None
        documents = request.trial_documents
        assert documents.model_name == "point-mass"
        assert documents.task.id == documents.objective.id == name
        spec = lower_canonical_train(request, registries.training_programs, registries=registries)
        prepared = prepare_managed_training(spec, registries)
        update = prepared.kernel_context["studio_graph_update"]
        slots = prepared.initial_slots
        initial_model = slots["model"]
        for _ in range(spec.training_config.n_batches):
            slots = update(slots)
            assert isfinite(float(slots["train_loss"]))
            grad_norm = float(slots["grad_norm"])
            assert grad_norm > 0 and isfinite(grad_norm)
        assert int(slots["batch_counter"]) == 2
        assert not eqx.tree_equal(initial_model, slots["model"])

        paired = compile_trial_execution(
            documents.model_name, documents.model, documents.task,
            registries.components, registries.expression_functions,
            objective=documents.objective, input_overrides=documents.input_overrides,
        )
        assert isinstance(paired, PreparedTrialExecution), paired
        assert paired.objective is not None
        result = paired.execution.validation(
            slots["model"], paired.objective, TaskParameters(), key=jr.key(27),
        )
        expected_terms = _REACH_TERMS | (
            {"effector_fixation"} if name == "delayed-reaches" else set()
        )
        assert result.objective.terms.keys() == expected_terms
        assert all(
            int(term.covered) == _VALIDATION_COUNTS[name] for term in result.objective.terms.values()
        )
        assert isfinite(float(result.objective.loss))

        retained = {}
        for address, value in result.retained.items():
            element = ModelAddress.from_text(address).element
            assert element.kind == "port" and element.end.owner == "outputs"
            retained[element.end.port] = value
        assert retained.keys() == {"hand", "velocity", "command", "activity"}
        np.testing.assert_array_equal(retained["hand"][:, 0], result.trials.variables["start"])
