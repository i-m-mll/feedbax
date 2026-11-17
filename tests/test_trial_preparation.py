import equinox as eqx
import jax.numpy as jnp
import pytest
from equinox.nn import StateIndex

from feedbax.contracts.graph import StudioTaskBindingSpec
from feedbax.runtime.components import Constant, Gain
from feedbax.runtime.graph import Component, Graph, Wire
from feedbax.runtime.task_bindings import expose_task_inputs


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


def test_graph_get_node_attrs_returns_public_named_node_params() -> None:
    graph = _graph()

    hidden, readout = graph.get_node_attrs("net", "hidden", "readout")

    assert jnp.array_equal(hidden, jnp.asarray([1.0], dtype=jnp.float32))
    assert jnp.array_equal(readout, jnp.asarray([2.0], dtype=jnp.float32))
