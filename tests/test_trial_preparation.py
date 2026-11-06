import equinox as eqx
import jax.numpy as jnp
import pytest
from equinox.nn import StateIndex

from feedbax.runtime.mapping import WhereDict
from feedbax.contracts.graph import StudioTaskBindingSpec
from feedbax.runtime.components import Constant, Gain
from feedbax.runtime.graph import Component, Graph, Wire, init_state_from_component
from feedbax.runtime.task_bindings import expose_task_inputs
from feedbax.tasks import (
    TaskTrialSpec,
    TrialTimeline,
    infer_n_steps,
    prepare_inputs,
    prepare_trial,
    safe_state_set,
    set_state_by_path,
    set_state_matching_dtypes,
    where_key_to_path,
)


class _NodeState(eqx.Module):
    hidden: jnp.ndarray
    output: jnp.ndarray


class _Params(eqx.Module):
    static_gain: jnp.ndarray | None
    dynamic_gain: jnp.ndarray


class _InterventionNode(Component):
    input_ports = ("input", "params_override")
    output_ports = ("output",)

    state_index: StateIndex
    params_index: StateIndex
    hidden: jnp.ndarray
    readout: jnp.ndarray

    def __init__(self):
        self.state_index = StateIndex(
            _NodeState(
                hidden=jnp.zeros((2,), dtype=jnp.float32),
                output=jnp.zeros((2,), dtype=jnp.float32),
            )
        )
        self.params_index = StateIndex(
            _Params(
                static_gain=jnp.asarray(1.0, dtype=jnp.float32),
                dynamic_gain=jnp.asarray(0.0, dtype=jnp.float32),
            )
        )
        self.hidden = jnp.asarray([1.0], dtype=jnp.float32)
        self.readout = jnp.asarray([2.0], dtype=jnp.float32)

    def __call__(self, inputs, state, *, key):
        return {"output": inputs["input"]}, state

def _graph() -> Graph:
    return Graph(
        nodes={"net": _InterventionNode()},
        wires=(),
        input_ports=("input", "gain"),
        output_ports=("output",),
        input_bindings={
            "input": ("net", "input"),
            "gain": ("net", "params_override"),
        },
        output_bindings={"output": ("net", "output")},
    )


def test_where_key_to_path_strips_unique_suffix() -> None:
    assert where_key_to_path("net.hidden#initial") == "net.hidden"
    assert where_key_to_path(lambda state: state.net.hidden) == "net.hidden"


def test_prepare_inputs_wraps_single_required_port() -> None:
    graph = Graph(nodes={"gain": Gain(1.0)}, input_ports=("input",),
                  input_bindings={"input": ("gain", "input")})
    inputs = jnp.zeros((3, 2), dtype=jnp.float32)

    prepared = prepare_inputs(graph, inputs)

    assert set(prepared) == {"input"}
    assert jnp.array_equal(prepared["input"], inputs)


def test_set_state_by_path_updates_graph_node_state_attribute() -> None:
    graph = _graph()
    state = init_state_from_component(graph)

    updated = set_state_by_path(graph, state, "net.hidden", jnp.asarray([5.0, 6.0]))

    assert jnp.array_equal(graph.state_view(updated).net.hidden, jnp.asarray([5.0, 6.0]))


def test_set_state_matching_dtypes_casts_replacement_to_existing_dtype() -> None:
    graph = _graph()
    state = init_state_from_component(graph)
    node = graph.get_node("net")
    replacement = _Params(
        static_gain=jnp.asarray(2, dtype=jnp.int32),
        dynamic_gain=jnp.asarray(3, dtype=jnp.int32),
    )

    updated = safe_state_set(state, node.params_index, replacement)

    assert updated.get(node.params_index).static_gain.dtype == jnp.float32
    assert updated.get(node.params_index).dynamic_gain.dtype == jnp.float32
    assert safe_state_set is set_state_matching_dtypes


def test_prepare_trial_applies_owned_resets_and_passes_typed_input() -> None:
    graph = _graph()
    trial_spec = TaskTrialSpec(
        inits=WhereDict({
            "net.hidden": jnp.asarray([7.0, 8.0], dtype=jnp.float32),
            "net.params_index.static_gain": jnp.asarray(4.0),
        }),
        targets=WhereDict(),
        inputs={
            "input": jnp.zeros((5, 2), dtype=jnp.float32),
            "gain": _Params(
                static_gain=None,
                dynamic_gain=jnp.asarray([1.0, 2.0, 3.0]),
            ),
        },
        timeline=TrialTimeline(n_steps=3),
    )

    prepared = prepare_trial(graph, trial_spec)

    assert prepared.n_steps == 3
    assert set(prepared.inputs) == {"input", "gain"}
    assert jnp.array_equal(
        graph.state_view(prepared.init_state).net.hidden, jnp.asarray([7.0, 8.0])
    )
    node = graph.get_node("net")
    assert prepared.init_state.get(node.params_index).static_gain == jnp.asarray(4.0)
    assert prepared.inputs["gain"].dynamic_gain.shape == (3,)


@pytest.mark.parametrize(
    ("wires", "input_ports", "message"),
    [
        (
            (Wire("source", "output", "gain", "input"),),
            (),
            r"gain\.input.*task:command.*wire source\.output",
        ),
        ((), ("task:command->gain.input",), r"command-binding.*already declared"),
    ],
)
def test_task_exposure_preserves_existing_input_authority(wires, input_ports, message) -> None:
    graph = Graph(
        nodes={"source": Constant(2.0), "gain": Gain(3.0)},
        wires=wires,
        input_ports=input_ports,
    )
    binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "exposed_data": [
                {
                    "id": "command",
                    "label": "Command",
                    "kind": "derived",
                    "path": "command",
                    "bindable": True,
                }
            ],
            "bindings": [
                {
                    "id": "command-binding",
                    "source_data_id": "command",
                    "target_node_id": "gain",
                    "target_port": "input",
                    "role": "model_input",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match=message):
        expose_task_inputs(graph, binding_spec)


def test_infer_n_steps_prefers_timeline_over_input_length() -> None:
    timeline = TrialTimeline(n_steps=2)

    assert infer_n_steps({"input": jnp.zeros((5, 1))}, timeline) == 2


def test_graph_get_node_attrs_returns_public_named_node_params() -> None:
    graph = _graph()

    hidden, readout = graph.get_node_attrs("net", "hidden", "readout")

    assert jnp.array_equal(hidden, jnp.asarray([1.0], dtype=jnp.float32))
    assert jnp.array_equal(readout, jnp.asarray([2.0], dtype=jnp.float32))
