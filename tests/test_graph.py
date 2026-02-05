import equinox as eqx
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp

from feedbax.graph import Component, Graph, Wire, init_state_from_component
from feedbax.iterate import iterate_component


class Increment(Component):
    input_ports = ("x",)
    output_ports = ("x",)

    def __call__(self, inputs, state, *, key):
        x = inputs["x"]
        return {"x": x + 1}, state


class Counter(Component):
    input_ports = ("input",)
    output_ports = ("output",)

    state_index: StateIndex

    def __init__(self):
        self.state_index = StateIndex(jnp.array(0))

    def __call__(self, inputs, state, *, key):
        count = state.get(self.state_index)
        count = count + 1
        state = state.set(self.state_index, count)
        return {"output": count}, state


def test_graph_cycle_iteration():
    node = Increment()
    graph = Graph(
        nodes={"inc": node},
        wires=(Wire("inc", "x", "inc", "x"),),
        input_ports=(),
        output_ports=("x",),
        input_bindings={},
        output_bindings={"x": ("inc", "x")},
    )

    outputs, _ = graph(
        {},
        init_state_from_component(graph),
        key=jax.random.PRNGKey(0),
        n_steps=3,
        cycle_init={("inc", "x"): jnp.array(0)},
    )

    assert (outputs["x"] == jnp.array([1, 2, 3])).all()


def test_iterate_component_state_history():
    component = Counter()
    inputs = jnp.zeros((2,))
    outputs, final_state, history = iterate_component(
        component,
        inputs,
        init_state_from_component(component),
        n_steps=2,
        key=jax.random.PRNGKey(0),
    )

    assert history is not None
    assert (history == jnp.array([0, 1, 2])).all()
