from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from feedbax.compiler import GraphCompilationError, GraphDocument, compile_graph
from feedbax.component_registry import ComponentRegistry
from feedbax.runtime.graph import init_state_from_component
from feedbax.runtime.mapping import WhereDict
from feedbax.tasks import TaskTrialSpec, prepare_trial
from feedbax.web.worker.execution import compile_training_run, rollout_graph


def _compile(graph):
    # Exercise the same serialized GraphDocument boundary as the Studio worker.
    document = GraphDocument(graph=graph)
    serialized = GraphDocument.model_validate_json(document.model_dump_json())
    return compile_graph(serialized, ComponentRegistry())


def _tap(namespace, compact, *, operation="scale", **intervention):
    return {
        "id": "change",
        "type": "intervention",
        "position": {"afterNode": "source"},
        "transform": {
            "type": "intervention",
            "intervention": {
                "operation": operation,
                "target_selector": {"namespace": namespace, "compact": compact},
                **intervention,
            },
        },
    }


def _gain():
    return {
        "type": "Gain",
        "params": {"gain": 1.0},
        "input_ports": ["input"],
        "output_ports": ["output"],
    }


def _wire(source_node, target_node, source_port="output", **fields):
    return {
        "source_node": source_node,
        "source_port": source_port,
        "target_node": target_node,
        "target_port": "input",
        **fields,
    }


def _fanout():
    return {
        "nodes": {
            "source": {"type": "Constant", "params": {"value": 2.0}, "output_ports": ["output"]},
            "left": _gain(),
            "right": _gain(),
        },
        "wires": [_wire("source", "left"), _wire("source", "right")],
        "output_ports": ["direct", "left", "right"],
        "output_bindings": {
            "direct": ["source", "output"],
            "left": ["left", "output"],
            "right": ["right", "output"],
        },
    }


@pytest.mark.parametrize(
    ("namespace", "compact", "expected"),
    [
        ("graph_edge", "edge:source.output->left.input", {"direct": 2, "left": 1, "right": 2}),
        ("graph_port", "port:source.output", {"direct": 1, "left": 1, "right": 1}),
    ],
)
def test_serialized_tap_changes_only_its_selected_fanout(namespace, compact, expected):
    spec = _fanout()
    spec["taps"] = [_tap(namespace, compact, operation="clamp", bounds={"min": 0, "max": 1})]
    compiled = _compile(spec)
    graph = compiled.graph
    outputs, _ = graph({}, init_state_from_component(graph), key=jr.key(0))

    assert outputs == expected
    assert compiled.resolved.graph.taps is None
    assert compiled.resolved.graph.nodes["__studio_tap__change__transform"].type.endswith(
        "Saturation"
    )
    generated = [
        entry
        for entry in compiled.record.source_map.entries
        if entry.origin == "compiler-generated"
    ]
    assert len(generated) == (3 if namespace == "graph_edge" else 4)
    assert {entry.authored_path for entry in generated} == {"/graph/taps/0"}


@pytest.mark.parametrize(
    ("phase", "first", "first_write"),
    [("read", [6, 8], [4, 5]), ("write", [3, 4], [8, 10])],
)
def test_serialized_recurrent_tap_preserves_owned_state_reset_and_distinguishes_phases(
    phase, first, first_write
):
    spec = {
        "nodes": {
            "plant": {
                "type": "LinearStateSpace",
                "params": {
                    "A": [[1.0, 0.0], [0.0, 1.0]],
                    "B": [[1.0, 0.0], [0.0, 1.0]],
                    "dt": 0.02,
                    "initial_state": [1.0, 2.0],
                    "pos_slice": [0, 1],
                    "vel_slice": [1, 2],
                },
                "input_ports": ["force", "epsilon"],
                "output_ports": ["effector", "state"],
            },
            "sink": _gain(),
        },
        "wires": [
            _wire(
                "plant",
                "sink",
                "state",
                temporality="recurrent",
                recurrent_initializer={"kind": "state_output"},
            )
        ],
        "input_ports": ["force"],
        "input_bindings": {"force": ["plant", "force"]},
        "output_ports": ["read", "plant"],
        "output_bindings": {"read": ["sink", "output"], "plant": ["plant", "state"]},
        "taps": [
            _tap(
                "recurrent_carry",
                "edge:plant.state->sink.input",
                parameters={"phase": phase},
                value={"mode": "constant", "value": 2.0},
            )
        ],
    }
    graph = _compile(spec).graph
    prepared = prepare_trial(
        graph,
        TaskTrialSpec(
            inits=WhereDict({"plant.vector": jnp.array([3.0, 4.0])}),
            targets=WhereDict(),
            inputs={"force": jnp.ones((2, 2))},
        ),
    )
    reset = graph.initial_cycle_port_values(prepared.init_state)
    assert len(reset) == 1
    np.testing.assert_allclose(next(iter(reset.values())), [3, 4])
    outputs, _, carry = graph.step(
        {"force": prepared.inputs["force"][0]},
        jax.tree.map(lambda leaf: leaf, prepared.init_state),
        reset,
        key=jr.key(1),
    )
    np.testing.assert_allclose(outputs["read"], first)
    np.testing.assert_allclose(next(iter(carry.values())), first_write)
    trajectory, _, _ = graph.rollout_with_trace(
        prepared.inputs,
        prepared.init_state,
        key=jr.key(2),
        n_steps=prepared.n_steps,
        trace=(),
    )
    np.testing.assert_allclose(trajectory["read"], [first, [8, 10]])
    np.testing.assert_allclose(trajectory["plant"], [[4, 5], [5, 6]])


@pytest.mark.parametrize(
    ("tap", "message"),
    [
        (
            _tap(
                "state_path", "path:states.plant.effector", value={"mode": "constant", "value": 2.0}
            ),
            "unsupported writable tap target",
        ),
        (
            _tap("graph_port", "port:source.output", operation="noise"),
            "unsupported executable intervention operation",
        ),
        (
            _tap("graph_port", "port:source.output", value={"mode": "constant", "value": [2.0]}),
            "expected a finite scalar number",
        ),
        (
            _tap("graph_port", "port:source.output", operation="clamp", bounds={"min": 0}),
            "both scalar bounds",
        ),
    ],
)
def test_unsupported_serialized_tap_fails_at_authored_tap(tap, message):
    with pytest.raises(GraphCompilationError, match=message) as error:
        _compile({**_fanout(), "taps": [tap]})
    assert error.value.record.diagnostics[0].source_anchor.authored_path == "/graph/taps/0"


def test_studio_worker_rollout_executes_authored_tap():
    spec = _fanout()
    spec["taps"] = [
        _tap(
            "graph_edge", "edge:source.output->left.input", value={"mode": "constant", "value": 3.0}
        )
    ]
    compiled = compile_training_run(
        graph_spec=spec,
        training_spec={
            "optimizer": {"type": "adam"},
            "loss": {
                "type": "TargetStateLoss",
                "label": "left_zero",
                "selector": "graph_output:left",
                "target_value": 0.0,
                "weight": 1.0,
                "norm": "squared_l2",
            },
            "n_batches": 1,
            "batch_size": 1,
        },
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec={
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [],
            "bindings": [],
        },
        cfg=SimpleNamespace(n_reach_steps=2),
        component_registry=ComponentRegistry(),
    )
    rollout = rollout_graph(compiled.graph, compiled, key=jr.key(0))
    np.testing.assert_allclose(rollout["outputs"]["left"], [6.0, 6.0])
    np.testing.assert_allclose(rollout["outputs"]["right"], [2.0, 2.0])


@pytest.mark.parametrize(
    ("parameters", "initializer", "message"),
    [
        ({}, {"kind": "state_output"}, "require parameters.phase"),
        (
            {"phase": "write"},
            {"kind": "constant", "value": 0.0},
            "requires a state_output initializer",
        ),
        (
            {"phase": "read"},
            {"kind": "state_output", "seed_channel_history": True},
            "seeds channel history",
        ),
        (
            {"phase": "write"},
            {"kind": "state_output", "state_slot": "direct"},
            "cannot preserve a graph state_slot alias",
        ),
    ],
)
def test_unsupported_recurrent_edit_fails_closed(parameters, initializer, message):
    spec = _fanout()
    spec["wires"][0].update(temporality="recurrent", recurrent_initializer=initializer)
    spec["taps"] = [
        _tap(
            "recurrent_carry",
            "edge:source.output->left.input",
            parameters=parameters,
            value={"mode": "constant", "value": 2.0},
        )
    ]
    with pytest.raises(GraphCompilationError, match=message) as error:
        _compile(spec)
    assert error.value.record.diagnostics[0].source_anchor.authored_path == "/graph/taps/0"


def test_tap_id_collision_fails_without_replacing_authored_node():
    spec = _fanout()
    spec["nodes"]["__studio_tap__change__transform"] = _gain()
    spec["taps"] = [
        _tap(
            "graph_edge", "edge:source.output->left.input", value={"mode": "constant", "value": 2.0}
        )
    ]
    with pytest.raises(GraphCompilationError, match="generated node .* already exists"):
        _compile(spec)


def test_nested_legacy_barnacle_intervention_requires_reauthoring():
    child = _fanout()
    child["barnacles"] = {
        "source": [{"id": "legacy", "kind": "intervention", "timing": "output", "label": "Legacy"}]
    }
    with pytest.raises(GraphCompilationError, match="Legacy barnacle intervention") as error:
        _compile({"subgraphs": {"child": child}})
    assert (
        error.value.record.diagnostics[0].source_anchor.authored_path
        == "/graph/subgraphs/child/barnacles/source/0"
    )
