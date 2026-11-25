"""Independent discrete matrix checks on the authored rollout transition."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.application.composition import load_application
from feedbax.compiler.model_edit import apply_model_edit
from feedbax.compiler.model_ports import ModelRegistryPorts
from feedbax.contracts.expression_values import Level, ScopedValue, ValueType
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.model_edit import ModelEditCommand, ModelEditResult, ModelEditState
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.task_document import TaskDocument
from feedbax.execution.trial_execution import sample_trial
from feedbax.execution.trial_resolution import PreparedTrialExecution, compile_trial_execution
from feedbax.linear import Linearization, QuadraticCost, linearize, quadratic_cost
from feedbax.objectives.document import ObjectiveExecutor, compile_objective
from feedbax.tasks.document import TaskParameters


A = np.array([[1, .1, .2, 0], [0, .91, 0, .3], [0, 0, .82, 0], [.1, 0, 0, .73]])
B = np.array([[1, 0], [.1, .2], [0, 1], [.3, 0]])


def _port(owner, port, direction):
    return ModelAddress.model_validate({
        "composites": [], "element": {"kind": "port", "end": (
            {"owner": owner, "port": port}
            if owner in {"inputs", "outputs"} else
            {"owner": "component", "component": owner, "port": port, "direction": direction}
        )},
    })


def _pair(delayed, *, grouped=False):
    def wire(a, b, attachments=()):
        def end(value):
            owner, name = value.split(".")
            return {"owner": owner, "port": name} if owner in {"inputs", "outputs"} else {
                "owner": "component", "component": owner, "port": name,
            }
        return {"source": end(a), "target": end(b), "attachments": attachments}

    def component(name, **fields):
        return {"kind": "leaf", "name": name, "type": "feedbax." + name, "fields": fields}

    parts = {"plant": component(
        "LinearStateSpace", A=A.tolist(), B=B.tolist(), initial_state=[.15, .25, -.4, .1],
    )}
    wires = {"drive": wire("inputs.command", "plant.force"), "observe": wire("plant.state", "outputs.position")}
    if delayed:
        parts |= {"split": component("Split", sizes=[2, 2]), "sum": component("Sum")}
        wires |= {
            "drive": wire("inputs.command", "sum.a"),
            "sense": wire("plant.state", "split.input", [
                {"kind": "delay", "id": "tap", "delay": {"value": 3, "unit": "st"}},
            ]),
            "feedback": wire("split.out_0", "sum.b"),
            "actuate": wire("sum.output", "plant.force"),
        }
    model = ModelDocument.model_validate({
        "step_seconds": .1, "ports": {
            "inputs": [{"id": "command", "name": "command", "type": {"shape": [2]}}],
            "outputs": [{"id": "position", "name": "state view", "type": {"shape": [4]}}],
        },
        "interior": {"components": parts, "wires": wires},
    })
    task = TaskDocument.model_validate({
        "id": "linear", "variables": [],
        "epochs": [{"id": "main", "name": "Main", "end": {
            "kind": "length", "length": {"kind": "literal", "value": 6, "unit": "st"},
        }}],
        "signals": [{
            "id": "command", "name": "command", "type": {"base": "real", "shape": [2]},
            "cells": [{"id": "cmd", "first_epoch": "main", "last_epoch": "main", "value": {
                "kind": "call", "function": "vector", "args": [
                    {"kind": "literal", "value": .2}, {"kind": "literal", "value": -.3},
                ],
            }}],
        }],
    })
    bundle = load_application().bundle
    if grouped:
        ports = ModelRegistryPorts(bundle.components)
        operation = ModelEditCommand.model_validate({"operation": {
            "op": "group", "name": "Plant and split",
            "selection": [
                {"composites": [], "element": {"kind": "component", "id": name}}
                for name in ("plant", "split")
            ],
        }})
        edit = apply_model_edit(
            ModelEditState(document=model), operation, ports=ports.port, component_problem=ports.placement_problem,
        )
        assert isinstance(edit, ModelEditResult), edit
        model = edit.state.document
    return compile_trial_execution(
        model, task, bundle.components, bundle.expression_functions,
        python_tasks=bundle.python_tasks, initialization_key=jax.random.key(1),
    )


@pytest.mark.parametrize(("delayed", "grouped"), [(False, False), (True, False), (True, True)])
def test_linearize_authored_plant_and_delay(delayed, grouped):
    with jax.enable_x64(True):
        pairing = _pair(delayed, grouped=grouped)
        assert isinstance(pairing, PreparedTrialExecution), pairing
        result = linearize(pairing, inputs=[_port("inputs", "command", "input")], outputs=[_port("outputs", "position", "output")])
    assert isinstance(result, Linearization), result
    if delayed:
        expected_a = np.zeros((16, 16))
        expected_a[:4, :4] = A
        expected_a[:4, 4:6] = B
        expected_a[4:12, 8:16] = np.eye(8)
        expected_a[12:16, :4] = np.eye(4)
        expected_b = np.concatenate((B, np.zeros((12, 2))))
        assert result.layout[1].shape == (3, 4) and "tap" in result.layout[1].key
    else:
        expected_a, expected_b = A, B
    np.testing.assert_array_equal(result.A, expected_a)
    np.testing.assert_array_equal(result.B, expected_b)
    np.testing.assert_array_equal(result.C, expected_a[:4])
    np.testing.assert_array_equal(result.D, B)
    assert result.A.dtype == np.dtype("float64")


def test_opening_replaces_the_nominal_driver_at_the_authored_target():
    with jax.enable_x64(True):
        pairing = _pair(False)
        assert isinstance(pairing, PreparedTrialExecution), pairing
        wire = ModelAddress.model_validate({"composites": [], "element": {"kind": "wire", "id": "drive"}})
        result = linearize(
            pairing, inputs=[_port("plant", "force", "input")], outputs=[_port("outputs", "position", "output")],
            openings=[wire], about=1,
        )
    assert isinstance(result, Linearization), result
    np.testing.assert_array_equal(result.B, B)
    np.testing.assert_allclose(result.outputs[0], A @ A @ np.array([.15, .25, -.4, .1]), rtol=0, atol=1e-15)


def test_quadratic_blocks_reproduce_the_compiled_objective_and_refuse_temporal_coupling():
    with jax.enable_x64(True):
        pairing = _pair(False)
        assert isinstance(pairing, PreparedTrialExecution), pairing
        source = pairing.execution.task
        key = jax.random.key(42)
        trial = sample_trial(source, TaskParameters(), key, key)
        terms = (
            ("tracking", "state", 2.0, "sum"), ("terminal", "state", 3.0, "last"),
            ("effort", "command", .5, "sum"), ("mix", "mixed", .4, "sum"),
        )
        document = ObjectiveDocument.model_validate({"id": "cost", "terms": [
            {"id": name, "measure": {"kind": "call", "function": "sqnorm",
             "args": [{"kind": "name", "reference": signal}]},
             "weight": {"kind": "literal", "value": weight}, "reduction": reduction}
            for name, signal, weight, reduction in terms
        ]})
        names = tuple(
            ScopedValue(id=name, name=name, type=ValueType(base="real", shape=(width,)), level=Level.STEP)
            for name, width in (("state", 4), ("command", 2), ("mixed", 4))
        )
        objective = compile_objective(document, names, load_application().bundle.expression_functions, task=source)
        assert isinstance(objective, ObjectiveExecutor), objective
        drive = jnp.eye(4, 2)

        def observe(x, u):
            return {"state": x[1:], "command": u, "mixed": x[:-1] + u @ drive.T}

        cost = quadratic_cost(objective, trial, observe, state_width=4, command_width=2)
        assert isinstance(cost, QuadraticCost), cost
        x = jax.random.normal(key, (source.n_steps + 1, 4))
        u = jax.random.normal(jax.random.fold_in(key, 1), (source.n_steps, 2))
        expected = sum(term.value for term in objective.evaluate_trial(observe(x, u), trial, key, key).values())
        np.testing.assert_allclose(cost.evaluate(x, u), expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(cost.Q[0], .4 * np.eye(4), rtol=0, atol=1e-15)
        np.testing.assert_allclose(cost.Q[1], 2.4 * np.eye(4), rtol=0, atol=1e-15)
        np.testing.assert_allclose(cost.R[0], .9 * np.eye(2), rtol=0, atol=1e-15)
        np.testing.assert_allclose(cost.N[0], .4 * drive, rtol=0, atol=1e-15)
        np.testing.assert_allclose(cost.Qf, 5 * np.eye(4), rtol=0, atol=1e-15)

        def coupled(x, u):
            return observe(x, u) | {"mixed": x[:-1] + x[1:]}

        refused = quadratic_cost(objective, trial, coupled, state_width=4, command_width=2)
        assert isinstance(refused, tuple) and "stage-separable" in refused[0].reason
