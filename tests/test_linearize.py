"""Independent discrete matrix checks on the authored rollout transition."""

import jax
import numpy as np
import pytest

from feedbax.application.composition import load_application
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.task_document import TaskDocument
from feedbax.execution.trial_resolution import PreparedTrialExecution, compile_trial_execution
from feedbax.linear import Linearization, linearize


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


def _pair(delayed):
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
    return compile_trial_execution(
        model, task, bundle.components, bundle.expression_functions,
        python_tasks=bundle.python_tasks, initialization_key=jax.random.key(1),
    )


@pytest.mark.parametrize("delayed", [False, True])
def test_linearize_authored_plant_and_delay(delayed):
    with jax.enable_x64(True):
        pairing = _pair(delayed)
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
