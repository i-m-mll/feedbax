"""Exercise the installed examples through ordinary project and training entry points."""

from math import isfinite
from pathlib import Path

import equinox as eqx
import jax.random as jr
import numpy as np

from feedbax.execution.trial_resolution import PreparedTrialExecution, compile_trial_execution
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.objective_document import OBJECTIVE_DOCUMENT_SCHEMA_ID
from feedbax.planner.project_plan import compile_project
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.contracts.task_document import TASK_DOCUMENT_SCHEMA_ID
from feedbax.contracts.train_settings import normalize_train_method
from feedbax.examples import load_shipped_projects
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.model_tree import MODEL_CHECKPOINT_STORE
from feedbax.application.bundle import new_application_registry_bundle
from feedbax.runtime._graph_key_schedule import model_initialization_key
from feedbax.tasks.document import TaskParameters
from feedbax.training.method_runtime import TrainingLoop, initialize_method, replica_value
from feedbax.host.method_checkpoint import validate_method_extra
from feedbax.host.model_checkpoint import SemanticModelSlot, SemanticModelWriter


_VALIDATION_COUNTS = {"simple-reaches": 28, "delayed-reaches": 28, "stabilization": 4}
_REACH_TERMS = {"effector_position", "effector_final_velocity", "nn_hidden", "nn_output"}
_POINT_MASS_REFERENCE = Path(__file__).parent / "fixtures" / "point_mass_one_update.npz"


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


def test_shipped_trial_examples_plan_train_and_score_validation(tmp_path):
    project = load_shipped_projects()["point-mass-reaching"]
    registries = new_application_registry_bundle()
    plan = compile_project(
        project, components=registries.components, functions=registries.expression_functions,
        python_tasks=registries.python_tasks, training_methods=registries.training_methods,
    )
    assert not plan.diagnostics, plan.diagnostics
    assert {run.key.step for run in plan.runs} == _VALIDATION_COUNTS.keys()

    for run in plan.runs:
        name = run.key.step
        assert run.admission_problem is None, run.admission_problem
        request = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
        assert isinstance(request, DocumentTrainingRequestSpec)
        documents = request.trial_documents
        assert documents.task.id == documents.objective.id == name
        normalized, method, settings = normalize_train_method(request, registries.training_methods)
        seed = normalized.seed or 0
        paired = compile_trial_execution(
            documents.model, documents.task_source,
            registries.components, registries.expression_functions, python_tasks=registries.python_tasks,
            objective=documents.objective, input_bindings=documents.input_bindings,
            initialization_key=model_initialization_key(seed),
        )
        assert isinstance(paired, PreparedTrialExecution), paired
        assert paired.objective is not None
        initial_model = paired.executable.graph
        ids = ("0",)
        state = initialize_method(method, settings, models=(initial_model,), seed=seed, replica_ids=ids)
        validate_method_extra(state.extra, replicas=len(ids))
        loop = TrainingLoop(
            method, settings, execution=paired.execution, objective=paired.objective,
            batch_size=request.training_config.batch_size, seed=seed, replica_ids=ids,
        )
        completed = 0
        if name == "simple-reaches":
            first = loop.run_block(state, completed=0, count=1)
            first_validation = paired.execution.validation(
                replica_value(first.state.model, 0), paired.objective, TaskParameters(), key=jr.key(27),
            )
            paths = paired.signal_scope.paths
            values = {paths[address]: array for address, array in first_validation.retained.items()}
            # Same seed and task on the pre-reseed graph with separate position and velocity outputs.
            with np.load(_POINT_MASS_REFERENCE, allow_pickle=False) as reference:
                np.testing.assert_allclose(values["effector", "pos"], reference["hand"], rtol=2e-6, atol=2e-7)
                np.testing.assert_allclose(values["effector", "vel"], reference["velocity"], rtol=2e-6, atol=2e-7)
            state, completed = first.state, first.completed
        trained = loop.run_block(state, completed=completed, count=request.target_total - completed)
        assert trained.completed == request.target_total == 2
        assert np.isfinite(trained.metrics["train_loss"]).all()
        grad_norm = trained.metrics["grad_norm"]
        assert np.isfinite(grad_norm).all() and (grad_norm > 0).all()
        model = replica_value(trained.state.model, 0)
        assert not eqx.tree_equal(initial_model, model)

        writer = SemanticModelWriter.for_graph(paired.executable.resolved.graph, None, registries.components)
        assert writer is not None
        transaction = tmp_path / name
        retained = writer.publish(transaction, (model,), templates=(initial_model,), replica_ids=ids)
        provider = ImmutableArtifactBlobProvider(transaction / MODEL_CHECKPOINT_STORE)
        slot = SemanticModelSlot.open_store(provider, retained.tree, retained.encoding, components=registries.components)
        restored = slot.restore(initial_model)
        assert eqx.tree_equal(restored, model)

        result = paired.execution.validation(
            restored, paired.objective, TaskParameters(), key=jr.key(27),
        )
        expected_terms = _REACH_TERMS | (
            {"effector_fixation"} if name == "delayed-reaches" else set()
        )
        assert result.objective.terms.keys() == expected_terms
        assert all(
            int(term.covered) == _VALIDATION_COUNTS[name] for term in result.objective.terms.values()
        )
        assert isfinite(float(result.objective.loss))

        paths = paired.signal_scope.paths
        retained = {paths[address]: value for address, value in result.retained.items()}
        assert retained.keys() == {("effector", "pos"), ("effector", "vel"), ("command",), ("activity",)}
        np.testing.assert_array_equal(retained["effector", "pos"][:, 0], result.trials.variables["start"])
