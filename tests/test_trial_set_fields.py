"""A Set in trials must switch the authored model without changing off trials."""

from dataclasses import replace

import jax.random as jr
import numpy as np

from feedbax.builtins.training import StandardTrainingSettings, supervised_gradient
from feedbax.application.composition import application_python_tasks
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.project import Project
from feedbax.contracts.request_identity import decode_request
from feedbax.execution.trial_resolution import PreparedTrialExecution, compile_trial_execution
from feedbax.planner.project_plan import compile_project
from feedbax.runtime._graph_key_schedule import model_initialization_key
from feedbax.tasks.document import TaskParameters
from feedbax.training.method_runtime import TrainingLoop, initialize_method
from feedbax.training.methods import TrainingMethodRegistry
from feedbax.plugins.declarations import DistributionSource, callable_declaration
from tests.test_document_train_pair import authored_documents, document_project


def test_weighted_set_switches_real_training_trials_at_the_window_boundaries():
    functions, model, task, objective = authored = authored_documents()
    raw = document_project(authored)
    entries = {
        name: {"differences": {"/interior/wires/effector-wire/attachments": {
            "value": [{
                "kind": "intervention", "id": name, "operation": "add", "fields": {"value": vector},
                "window": {"epoch": "move", "offsets": [15, 30, 45], "duration": 5},
            }],
        }}}
        for name, vector in {"x": [1.0, 0.0], "y": [0.0, 2.0], "z": [-3.0, -3.0]}.items()
    }
    raw["documents"]["bank"] = {"schema": "feedbax.set_document.v1", "payload": {"entries": entries}}
    raw["steps"]["train"]["set_fields"] = {"bank": {
        "axis": {"set": "bank"},
        "sampling": {"kind": "weighted", "weights": {"x": 2, "y": 3, "z": 4}, "none": 1},
    }}
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    (run,) = plan.runs
    assert run.admission_problem is None, run.problems
    document = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec.trial_documents
    registry = ComponentRegistry()
    seed = 4

    def prepare(model, task):
        result = compile_trial_execution(
            model, task, registry, functions, python_tasks=application_python_tasks(),
            objective=objective, input_overrides=document.input_overrides,
            initialization_key=model_initialization_key(seed),
        )
        assert isinstance(result, PreparedTrialExecution), result
        return result

    compiled = prepare(document.model, document.task)
    nominal = prepare(model, task)
    parameters = TaskParameters()
    sampled, _ = compiled.execution.sample_trials(parameters, run_key=jr.key(0), update_key=jr.key(1), count=2000)
    assert np.allclose(np.bincount(sampled.variables["bank"], minlength=4) / 2000, [0.1, 0.2, 0.3, 0.4], atol=0.035)
    observed_perturbations = np.asarray(sampled.variables["bank"][:64])
    assert set(observed_perturbations) == {0, 1, 2, 3}
    methods = TrainingMethodRegistry()
    methods.register_declaration(
        supervised_gradient, callable_declaration(supervised_gradient),
        key="feedbax.supervised_gradient", source=DistributionSource("feedbax", "feedbax", "0.2.0"),
    )
    method = methods.resolve("feedbax.supervised_gradient")
    settings = StandardTrainingSettings.model_validate({
        "optimizer": {"type": "adam", "learning_rate": {"type": "constant_schedule", "value": 0.003}},
    })

    def inspect_update(state, ctx):
        updated, metrics = supervised_gradient(state, ctx)
        trials, keys = ctx.sample_trials()
        (union_output,) = ctx.rollout(state.model, trials, keys).values()
        (base_output,) = nominal.execution.rollout_batch(nominal.executable.graph, trials, keys)[0].values()
        return updated, {
            **metrics, "delta": union_output - base_output, "bank": trials.variables["bank"],
            "move": trials.epochs.starts[:, -1],
        }

    graph = compiled.executable.graph
    initial = initialize_method(method, settings, models=(graph,), seed=seed, replica_ids=("0",))
    result = TrainingLoop(
        replace(method, update=inspect_update), settings, execution=compiled.execution,
        objective=compiled.objective, batch_size=64, seed=seed, replica_ids=("0",),
    ).run_block(initial, completed=0, count=1)
    assert result.completed == 1 and not result.rejected
    recorded = np.asarray(result.metrics["bank"][0, 0])
    movements = np.asarray(result.metrics["move"][0, 0])
    deltas = np.asarray(result.metrics["delta"][0, 0])
    assert set(recorded) == {0, 1, 2, 3}
    assert len(set(movements)) > 1
    starts = set()
    deviations = ((0, 0), (1, 0), (0, 2), (-3, -3))
    for entry, movement, delta in zip(recorded, movements, deltas, strict=True):
        changed = np.flatnonzero(np.linalg.norm(delta, axis=1) > 1e-6)
        expected = np.zeros_like(delta)
        if entry:
            onset = int(changed[0])
            starts.add(onset - movement)
            assert onset - movement in (15, 30, 45)
            expected[onset:onset + 5] = deviations[entry]
        np.testing.assert_allclose(
            delta, expected, rtol=1e-5, atol=1e-6,
        )
    assert starts == {15, 30, 45}
