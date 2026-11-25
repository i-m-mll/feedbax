"""Exercise Delay Initial through the paired Model and Task runtime."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.application.bundle import new_application_registry_bundle
from feedbax.builtins.training import StandardTrainingSettings
from feedbax.contracts.expression_values import Level, ScopedValue, ValueBase, ValueType
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import ModelDocument
from feedbax.execution.trial_resolution import compile_trial_execution
from feedbax.objectives.document import ObjectiveDocument, ObjectiveTerm
from feedbax.tasks.document import Cell, Epoch, Signal, TaskDocument, TaskParameters, Variable
from feedbax.contracts.task_document import Length
from feedbax.training.method_runtime import TrainingLoop, initialize_method, replica_value
from tests.test_document_task_parity import expression
from tests.test_model_document_plan import _end, _wire


_VECTOR = ValueType(base=ValueBase.REAL, shape=(2,))
_CONTEXT = ScopedValue(id="trial_context", name="trial_context", type=_VECTOR, level=Level.TRIAL)
_INITIAL = np.array([2.0, -1.0])


def _model(initial, *, nested):
    ports = {
        "inputs": [{"id": name, "name": name, "type": {"shape": [2], "unit": "1"}} for name in ("drive", "context")],
        "outputs": [{"id": "hidden", "name": "hidden", "type": {"shape": [2], "unit": "1"}}],
    }
    interior = {
        "components": {
            "cell": {"kind": "leaf", "name": "Cell", "type": "feedbax.GRU", "fields": {"hidden_size": 2}},
            "encoder": {"kind": "leaf", "name": "Encoder", "type": "feedbax.Linear", "fields": {"output_size": 2}},
        },
        "wires": {
            "drive": _wire({"owner": "inputs", "port": "drive"}, _end("cell", "input")),
            "context": _wire({"owner": "inputs", "port": "context"}, _end("encoder", "input")),
            "feedback": _wire(_end("cell", "hidden"), _end("cell", "hidden"), (
                {"kind": "delay", "id": "lag", "delay": {"value": 1, "unit": "st"}, "initial_source": initial},
            )),
            "output": _wire(_end("cell", "hidden"), {"owner": "outputs", "port": "hidden"}),
        },
    }
    if nested:
        interior = {
            "components": {"core": {"kind": "composite", "name": "Core", "ports": ports, "interior": interior}},
            "wires": {
                **{name: _wire({"owner": "inputs", "port": name}, _end("core", name)) for name in ("drive", "context")},
                "hidden": _wire(_end("core", "hidden"), {"owner": "outputs", "port": "hidden"}),
            },
        }
    return ModelDocument.model_validate({"step_seconds": 0.01, "ports": ports, "interior": interior})

def _task():
    return TaskDocument(
        id="encoding",
        variables=(Variable(id="trial_context", name="trial_context", type=_VECTOR, value=expression("[2, -1]")),),
        epochs=tuple(Epoch(id=name, name=name, end=Length(length=expression("1 st"))) for name in ("first", "later")),
        signals=(
            Signal(id="drive", name="drive", type=_VECTOR, cells=(
                Cell(id="drive", first_epoch="first", last_epoch="later", value=expression("[0, 0]")),
            )),
            Signal(id="context_input", name="context", type=_VECTOR, cells=(
                Cell(id="context-first", first_epoch="first", last_epoch="first", value=expression("trial_context", (_CONTEXT,))),
                Cell(id="context-later", first_epoch="later", last_epoch="later", value=expression("[9, 4]")),
            )),
        ),
    )


@pytest.mark.parametrize(("initial", "nested"), [
    pytest.param({"kind": "wire", "source": _end("encoder", "output")}, False, id="encoder"),
    pytest.param({"kind": "wire", "source": _end("encoder", "output")}, True, id="nested-encoder"),
    pytest.param({"kind": "wire", "source": {"owner": "inputs", "port": "context"}}, False, id="model-input"),
    pytest.param({"kind": "task", "variable": "trial_context"}, False, id="task-variable"),
])
def test_delay_initial_receives_trial_context_and_trains_encoder(initial, nested):
    bundle = new_application_registry_bundle()
    bundle.seal()
    output = ModelAddress.model_validate({"composites": [], "element": {
        "kind": "port", "end": {"owner": "outputs", "port": "hidden"},
    }})
    hidden = ScopedValue(id=output.to_text(), name="hidden", type=_VECTOR, level=Level.STEP)
    objective = ObjectiveDocument(id="hidden-loss", terms=(
        ObjectiveTerm(id="magnitude", measure=expression("sqnorm(hidden)", (hidden,)), reduction="last"),
    ))
    key = jax.random.key(24)
    paired = compile_trial_execution(
        _model(initial, nested=nested), _task(), bundle.components, bundle.expression_functions, python_tasks=bundle.python_tasks,
        objective=objective, initialization_key=key,
    )
    assert not isinstance(paired, tuple), paired
    graph, execution = paired.executable.graph, paired.execution
    trial = execution.task.ahead(TaskParameters(), key, key)
    model_inputs = {port: trial.inputs[signal] for port, signal in execution.input_bindings}
    np.testing.assert_array_equal(model_inputs["context"], [[2, -1], [9, 4]])
    reset = execution.reset.prepare(
        graph, trial.inits, key=key, inputs=model_inputs, input_series=True, reset_values=trial.variables,
    )
    core = graph.nodes["core"] if nested else graph
    encoder = core.nodes["encoder"].layer
    uses_encoder = initial["kind"] == "wire" and initial["source"]["owner"] == "component"
    expected = encoder(jnp.asarray(_INITIAL)) if uses_encoder else _INITIAL
    carry = reset.cycle_init["__nested__", "core"] if nested else reset.cycle_init
    np.testing.assert_allclose(carry["cell", "hidden"], expected)
    retained, _ = execution.rollout(graph, trial, key)
    expected_hidden = core.nodes["cell"].cell(model_inputs["drive"][0], jnp.asarray(expected))
    np.testing.assert_allclose(retained[hidden.id][0], expected_hidden, rtol=1e-6)
    if uses_encoder:
        method = bundle.training_methods.resolve("feedbax.supervised_gradient")
        settings = StandardTrainingSettings()
        starting = initialize_method(method, settings, models=(graph,), seed=24, replica_ids=("0",))
        result = TrainingLoop(
            method, settings, execution=execution, objective=paired.objective, batch_size=2, seed=24, replica_ids=("0",),
        ).run_block(starting, completed=0, count=1)
        assert result.completed == 1 and not result.rejected
        trained = replica_value(result.state.model, 0)
        updated = (trained.nodes["core"] if nested else trained).nodes["encoder"].layer.weight
        assert not np.array_equal(updated, encoder.weight)
