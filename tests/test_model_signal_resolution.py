"""The model signal table preserves declared meaning through scoped wiring."""

import json
from importlib.resources import files

import jax
import numpy as np

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.model_document import lower_model_document
from feedbax.compiler.model_signals import boundary_signal_port, resolve_model_signals
from feedbax.compiler.resolution import resolve_graph_parameters
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import LENGTH, TIME, ValueBase, ValueType
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import ModelDocument


def _example():
    return json.loads(files("feedbax.examples").joinpath(
        "point-mass-reaching/documents/point-mass.json",
    ).read_text())["payload"]


def _resolve(payload):
    components = ComponentRegistry()
    model = ModelDocument.model_validate(payload)
    lowered = lower_model_document(model, components)
    prototypes = {lowered.graph.input_bindings["target_pos"]: jax.ShapeDtypeStruct((2,), np.float32)}
    resolved = resolve_graph_parameters(lowered.graph, prototypes, components, allow_incomplete=True)
    return resolve_model_signals(model, resolved, components), lowered.interface(
        model, components, input_prototypes=prototypes,
    )


def _component_port(component, direction, port, *, scope=()):
    return ModelAddress.model_validate({"composites": scope, "element": {"kind": "port", "end": {
        "owner": "component", "component": component, "direction": direction, "port": port,
    }}})


def test_shipped_point_mass_outputs_keep_state_meaning_without_stored_units():
    model = _example()
    assert all(not port["type"].get("unit") for port in model["ports"]["outputs"] if port["id"] in {
        "hand", "velocity",
    })
    signals, interface = _resolve(model)
    assert not signals.problems
    hand = signals.at(boundary_signal_port((), "outputs", "hand"))
    assert hand.type == ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH, space="point")
    assert hand.origin == ModelAddress.at_state((), "plant", "effector/position")
    assert hand.unit == "m"
    effector = _component_port("plant", "output", "effector")
    recorded_position = next(leaf for leaf in interface.readable_leaves if leaf.port == effector and leaf.path == ("pos",))
    assert recorded_position.origin == hand.origin
    assert signals.at(recorded_position.address) == recorded_position
    velocity = signals.at(boundary_signal_port((), "outputs", "velocity"))
    assert velocity.type == ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH.combine(TIME, -1), space="free")
    assert velocity.origin == ModelAddress.at_state((), "plant", "effector/velocity")
    assert velocity.unit == "m/s"
    assert {port.name: port.type for port in interface.outputs}["hand"] == hand.type
    # An open readout can supply a force input without acquiring the input's type itself.
    assert signals.at(_component_port("plant", "input", "force")).dimensions == UNITS["N"].dimensions
    assert signals.at(boundary_signal_port((), "outputs", "command")).dimensions is None


def test_bad_wire_is_local_and_leaves_other_signals_readable():
    model = _example()
    model["interior"]["components"]["point_receiver"] = {
        "kind": "composite", "name": "Point receiver",
        "ports": {
            "inputs": [{"id": "in", "name": "In", "type": {"shape": [2], "space": "point"}}],
            "outputs": [{"id": "out", "name": "Out", "type": {}}],
        },
        "interior": {"wires": {"pass": {
            "source": {"owner": "inputs", "port": "in"},
            "target": {"owner": "outputs", "port": "out"},
        }}},
    }
    model["interior"]["wires"]["bad"] = {
        "source": {"owner": "component", "component": "velocity", "port": "output"},
        "target": {"owner": "component", "component": "point_receiver", "port": "in"},
    }
    signals, interface = _resolve(model)
    (problem,) = signals.problems
    assert problem.address == ModelAddress.model_validate({
        "composites": [], "element": {"kind": "wire", "id": "bad"},
    })
    assert "point" in problem.reason and "free" in problem.reason
    hand = interface.signals.at(boundary_signal_port((), "outputs", "hand"))
    assert hand.origin == ModelAddress.at_state((), "plant", "effector/position")
    downstream = signals.at(_component_port("point_receiver", "output", "out"))
    assert downstream.origin is None and downstream.dimensions is None
    assert downstream.type.space is None
    assert downstream.type.shape == (2,)
