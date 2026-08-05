"""Tests for the shared rollout scan harness in ``feedbax.runtime._rollout``.

``Graph._call_with_iteration`` and ``iterate_component`` both drive this
harness; these tests lock in the two properties that let one implementation
serve both call sites — the step-carry variant is inert when unused, and the
step-input selection is an identity for time-major inputs while keeping the
gather's truncate/clamp semantics otherwise.
"""

import equinox as eqx
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import pytest

from feedbax.runtime._rollout import scan_rollout, select_step_inputs
from feedbax.runtime.graph import Component, init_state_from_component


class _Accumulate(Component):
    input_ports = ("x",)
    output_ports = ("y",)

    state_index: StateIndex

    def __init__(self):
        self.state_index = StateIndex(jnp.zeros(2))

    def __call__(self, inputs, state, *, key):
        h = jnp.tanh(state.get(self.state_index) + inputs["x"])
        state = state.set(self.state_index, h)
        return {"y": h}, state


def _legacy_gather(inputs, n_steps):
    return jax.vmap(lambda i: jt.map(lambda x: x[i], inputs))(jnp.arange(n_steps))


@pytest.mark.parametrize("n_steps", [3, 5, 7])
def test_select_step_inputs_matches_gather(n_steps):
    """Selection matches the gather it replaces, for every leading-dim regime."""
    inputs = {
        "u": jr.normal(jr.PRNGKey(0), (5, 3)),
        "v": jnp.arange(5, dtype=jnp.float32),
    }
    selected = select_step_inputs(inputs, n_steps)
    gathered = _legacy_gather(inputs, n_steps)

    for key in inputs:
        assert selected[key].shape == gathered[key].shape
        assert selected[key].dtype == gathered[key].dtype
        assert jnp.array_equal(selected[key], gathered[key])


def test_select_step_inputs_is_identity_for_time_major_inputs():
    """Time-major inputs of the right length are passed through untouched."""
    inputs = {"u": jnp.zeros((4, 3)), "v": jnp.ones((4,))}
    selected = select_step_inputs(inputs, 4)

    assert selected is inputs

    jaxpr = jax.make_jaxpr(lambda x: select_step_inputs(x, 4))(inputs)
    assert "gather" not in jaxpr.pretty_print(use_color=False)


def _run(init_step_carry, step_fn, component, init_state, n_steps, **kwargs):
    inputs = {"x": jr.normal(jr.PRNGKey(1), (n_steps, 2))}
    return scan_rollout(
        step_fn,
        init_state,
        inputs,
        jr.split(jr.PRNGKey(2), n_steps),
        n_steps,
        state_view_fn=component.state_view,
        init_step_carry=init_step_carry,
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"save_history": False},
        {"checkpoint": True},
        {"streaming_loss_fn": lambda view, t: jnp.sum(view**2)},
    ],
)
def test_scan_rollout_step_carry_is_inert_when_unused(kwargs):
    """Threading an unused step carry does not change any rollout result.

    This is what lets the cyclic (``Graph``) and acyclic (``iterate_component``)
    call sites share one harness: the cycle-values carry is a parameter, not a
    forked copy of the scan.
    """
    component = _Accumulate()
    init_state = init_state_from_component(component)
    n_steps = 4

    def step_fn(step_input, state, step_carry, step_key, t):
        outputs, new_state = component(step_input, state, key=step_key)
        return outputs, new_state, step_carry

    def step_fn_with_carry(step_input, state, step_carry, step_key, t):
        outputs, new_state = component(step_input, state, key=step_key)
        return outputs, new_state, {"seen": step_carry["seen"] + 1}

    without = _run(None, step_fn, component, init_state, n_steps, **kwargs)
    with_carry = _run(
        {"seen": jnp.int32(0)}, step_fn_with_carry, component, init_state, n_steps, **kwargs
    )

    without_leaves = jt.leaves(eqx.filter(without, eqx.is_array))
    with_carry_leaves = jt.leaves(eqx.filter(with_carry, eqx.is_array))
    assert len(without_leaves) == len(with_carry_leaves)
    for a, b in zip(without_leaves, with_carry_leaves):
        assert jnp.array_equal(a, b)


def test_scan_rollout_history_prepends_initial_state_view():
    component = _Accumulate()
    init_state = init_state_from_component(component)

    def step_fn(step_input, state, step_carry, step_key, t):
        outputs, new_state = component(step_input, state, key=step_key)
        return outputs, new_state, step_carry

    outputs, final_state, history = _run(None, step_fn, component, init_state, 4)

    assert history.shape == (5, 2)
    assert jnp.array_equal(history[0], jnp.zeros(2))
    assert jnp.array_equal(history[-1], final_state.get(component.state_index))
    assert isinstance(final_state, State)
    assert outputs["y"].shape == (4, 2)


def test_scan_rollout_without_history_returns_none():
    component = _Accumulate()
    init_state = init_state_from_component(component)

    def step_fn(step_input, state, step_carry, step_key, t):
        outputs, new_state = component(step_input, state, key=step_key)
        return outputs, new_state, step_carry

    _, _, aux = _run(None, step_fn, component, init_state, 4, save_history=False)
    assert aux is None
