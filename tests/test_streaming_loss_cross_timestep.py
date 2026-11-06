"""Streaming-loss tests for cross-timestep finite-difference terms."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import pytest
from equinox.nn import State, StateIndex

from feedbax.runtime.streaming import init_streaming_state_window, update_streaming_state_window
from feedbax.runtime._rollout import scan_rollout
from feedbax.objectives.loss import (
    CompositeLoss,
    ModelLoss,
    NthDifferenceLoss,
    StateDerivativeLoss,
    TargetSpec,
    TargetStateLoss,
    nan_safe_mse,
)
from feedbax.objectives.streaming import make_streaming_loss_fn
from feedbax.runtime.mapping import WhereDict
from feedbax.tasks import TaskTrialSpec


class _FakeState(eqx.Module):
    x: jnp.ndarray
    hidden: jnp.ndarray


def _full_state(values: jnp.ndarray) -> _FakeState:
    return _FakeState(
        x=values[None],
        hidden=values[None],
    )


def _step_state(values: jnp.ndarray, t: int) -> _FakeState:
    return _FakeState(
        x=values[t],
        hidden=values[t],
    )


def _stream_loss(loss, values: jnp.ndarray):
    streaming_fn = make_streaming_loss_fn(
        loss,
        trial_specs=None,
        model=None,
        n_steps=values.shape[0],
    )
    order = getattr(streaming_fn, "streaming_order", 0)
    window = init_streaming_state_window(_step_state(values, 0), order)
    total = jnp.asarray(0.0, dtype=values.dtype)

    for t in range(values.shape[0]):
        state_t = _step_state(values, t)
        loss_input = state_t
        if order > 0:
            window = update_streaming_state_window(window, state_t)
            loss_input = window
        total = total + streaming_fn(loss_input, jnp.asarray(t))

    return total


def test_streaming_order_one_matches_full_trajectory_loss():
    t = jnp.arange(7, dtype=jnp.float32)[:, None]
    values = jnp.concatenate([0.2 * t, -0.1 * t], axis=-1)
    loss = StateDerivativeLoss(label="state_derivative", where=lambda s: s.hidden)

    full = loss(_full_state(values), trial_specs=None, model=None).total
    streamed = _stream_loss(loss, values)

    assert jnp.allclose(streamed, full)


@pytest.mark.parametrize("feature_shape", [(6,), (2, 3), ()])
def test_streaming_order_two_matches_retained_value_and_gradient(feature_shape):
    values = jnp.broadcast_to(
        jnp.array([0.0, 0.0, 1.0, 4.0, 9.0]).reshape((5,) + (1,) * len(feature_shape)),
        (5,) + feature_shape,
    )
    loss = NthDifferenceLoss(label="second_difference", order=2, where=lambda s: s.x)

    def retained(values):
        return loss(_full_state(values), trial_specs=None, model=None).total

    full, full_grad = jax.value_and_grad(retained)(values)
    streamed, streamed_grad = jax.value_and_grad(lambda x: _stream_loss(loss, x))(values)

    # Each feature has second differences [1, 2, 2] and contributes 3.
    expected_grad = jnp.array([1.0, 0.0, -1.0, -2.0, 2.0]) * (2 / 3)
    assert full == pytest.approx(3 * values.size / 5)
    assert jnp.allclose(full_grad.reshape(5, -1), expected_grad[:, None])
    unbatched = NthDifferenceLoss(order=2, time_axis=0)(values, None, None).value
    assert unbatched == pytest.approx(float(full))
    assert streamed == pytest.approx(float(full))
    assert jnp.allclose(streamed_grad, full_grad)


def test_streaming_uses_max_order_for_composite_cross_timestep_terms():
    t = jnp.arange(10, dtype=jnp.float32)[:, None]
    values = jnp.concatenate([jnp.sin(0.2 * t), t**2], axis=-1)
    first = NthDifferenceLoss(label="first", order=1, where=lambda s: s.x)
    second = NthDifferenceLoss(label="second", order=2, where=lambda s: s.x)
    loss = CompositeLoss(
        terms={"first": first, "second": second},
        weights={"first": 0.5, "second": 2.0},
    )

    streaming_fn = make_streaming_loss_fn(
        loss,
        trial_specs=None,
        model=None,
        n_steps=values.shape[0],
    )
    full = loss(_full_state(values), trial_specs=None, model=None).total
    streamed = _stream_loss(loss, values)

    assert streaming_fn.streaming_order == 2
    assert jnp.allclose(streamed, full, atol=1e-6)


class _TrialSpecs(eqx.Module):
    targets: dict = eqx.field(default_factory=dict)


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("loss_dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("state_dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_scan_streaming_loss_dtype_and_gradient_parity(
    order,
    loss_dtype,
    state_dtype,
    checkpoint,
):
    """Real scans preserve loss precision independently of the state precision."""
    with jax.experimental.enable_x64():
        n_steps = 5
        index = StateIndex(jnp.zeros(2, dtype=state_dtype))
        init_state = State(index)
        keys = jr.split(jr.PRNGKey(12), n_steps)
        inputs = jnp.linspace(-0.4, 0.8, n_steps * 2, dtype=state_dtype).reshape(-1, 2)
        specs = _TrialSpecs()

        def norm(x):
            return jnp.sum(x.astype(loss_dtype) ** 2, axis=-1)

        if order == 0:
            loss = TargetStateLoss(
                label="target",
                where=lambda s: s.x,
                norm=norm,
                spec=TargetSpec(value=jnp.asarray(0.2, dtype=state_dtype)),
            )
        else:
            loss = NthDifferenceLoss(order=order, where=lambda s: s.x, norm=norm)

        def run(scale, streaming):
            def step_fn(step_input, state, step_carry, key, t):
                x = jnp.tanh(
                    state.get(index)
                    + scale * step_input
                    + 0.01 * jr.normal(key, (2,), dtype=state_dtype)
                    + 0.02 * (t.astype(state_dtype) + step_carry)
                )
                return x, state.set(index, x), step_carry + state_dtype(1)

            streaming_fn = make_streaming_loss_fn(loss, specs, None, n_steps) if streaming else None
            outputs, state, aux = scan_rollout(
                step_fn,
                init_state,
                inputs,
                keys,
                n_steps,
                state_view_fn=lambda s, _: _FakeState(s.get(index), s.get(index)),
                init_step_carry=state_dtype(0),
                checkpoint=checkpoint,
                streaming_loss_fn=streaming_fn,
            )
            if streaming:
                value = aux
            else:
                history = jt.map(lambda x: x[None, 1:], aux)
                value = loss(history, specs, None).total
            return value, (outputs, state.get(index))

        def streamed(scale):
            return run(scale, True)

        def full(scale):
            return run(scale, False)

        scale = state_dtype(0.7)
        for transform in (lambda fn: fn, jax.jit):
            actual = transform(jax.value_and_grad(streamed, has_aux=True))(scale)
            expected = transform(jax.value_and_grad(full, has_aux=True))(scale)
            assert actual[0][0].dtype == loss_dtype
            for a, b in zip(jt.leaves(actual), jt.leaves(expected)):
                assert jnp.allclose(a, b, rtol=2e-6, atol=1e-7)

        scales = jnp.array([0.4, 0.7], dtype=state_dtype)
        actual = jax.jit(jax.vmap(jax.value_and_grad(streamed, has_aux=True)))(scales)
        expected = jax.jit(jax.vmap(jax.value_and_grad(full, has_aux=True)))(scales)
        assert actual[0][0].dtype == loss_dtype
        for a, b in zip(jt.leaves(actual), jt.leaves(expected)):
            assert jnp.allclose(a, b, rtol=2e-6, atol=1e-7)


def test_mixed_streaming_objective_matches_retained_rollout():
    """A weighted target, model and two window orders share the scan timeline."""
    n_steps = 4
    index = StateIndex(jnp.zeros((1,)))
    inputs = jnp.array([[1.0], [2.0], [4.0], [7.0]])
    keys = jr.split(jr.PRNGKey(13), n_steps)

    def where(state):
        return state.x

    loss = CompositeLoss(
        terms={
            "tracking": TargetStateLoss(
                label="tracking",
                where=where,
                spec=TargetSpec(
                    value=jnp.array(-10.0),
                    time_idxs=jnp.array([-1]),
                    discount=lambda spec: spec.inputs["discount"],
                ),
            ),
            "model": ModelLoss(label="model", loss_fn=jnp.square),
            "velocity": NthDifferenceLoss(label="velocity", order=1, where=where),
            "acceleration": NthDifferenceLoss(label="acceleration", order=2, where=where),
        },
        weights={"tracking": 1.0, "model": 2.0, "velocity": 3.0, "acceleration": 0.5},
    )

    targets = jnp.array([[1.0], [100.0], [4.0]])
    weight_inputs = {
        "mask": jnp.array([True, False, True]),
        "discount": jnp.array([2.0, 3.0, 4.0]),
    }

    def specs(value, weight_inputs):
        return TaskTrialSpec(
            inits=WhereDict(),
            targets=WhereDict({
                where: {"tracking": TargetSpec(
                    value=value, time_mask=lambda spec: spec.inputs["mask"]
                )},
            }),
            inputs=weight_inputs,
        )

    trial = specs(targets, weight_inputs)
    batch = specs(targets[None], jt.map(lambda weight: weight[None], weight_inputs))

    def run(scale, *, streaming):
        def step_fn(step_input, state, carry, key, t):
            x = scale * step_input
            return x, state.set(index, x), carry

        step_loss = make_streaming_loss_fn(loss, trial, scale, n_steps) if streaming else None
        outputs, state, aux = scan_rollout(
            step_fn,
            State(index),
            inputs,
            keys,
            n_steps,
            state_view_fn=lambda s, _: _FakeState(s.get(index), s.get(index)),
            streaming_loss_fn=step_loss,
        )
        value = aux if streaming else loss(jt.map(lambda x: x[None, 1:], aux), batch, scale).total
        return value, (outputs, state.get(index))

    retained = jax.jit(jax.value_and_grad(lambda m: run(m, streaming=False), has_aux=True))
    streamed = jax.jit(jax.value_and_grad(lambda m: run(m, streaming=True), has_aux=True))
    expected = retained(jnp.array(2.0))
    actual = streamed(jnp.array(2.0))

    # At scale 2, tracking contributes 418, model 8, velocity 56 and
    # acceleration 2. Tracking contributes 584 to the derivative.
    assert expected[0][0] == pytest.approx(484.0)
    assert expected[1] == pytest.approx(650.0)
    for got, want in zip(jt.leaves(actual), jt.leaves(expected), strict=True):
        assert jnp.allclose(got, want)


def test_missing_targets_produce_finite_gradients():
    target = jnp.array([3.0, jnp.nan, jnp.inf])
    value, gradient = jax.jit(jax.value_and_grad(nan_safe_mse))(
        jnp.array([1.0, 2.0, jnp.nan]), target
    )

    assert value == 4.0
    assert jnp.array_equal(gradient, jnp.array([-4.0, 0.0, 0.0]))


def test_retained_and_streaming_reject_short_weights():
    loss = TargetStateLoss(
        label="target",
        where=lambda state: state.x,
        spec=TargetSpec(value=0.0, discount=jnp.array([1.0, 2.0])),
    )
    specs = _TrialSpecs()
    values = jnp.ones((4, 1))

    with pytest.raises(ValueError, match=r"Time weight must be scalar or shape \(3,\)"):
        loss(_full_state(values), specs, None)
    with pytest.raises(ValueError, match=r"Time weight must be scalar or shape \(3,\)"):
        make_streaming_loss_fn(loss, specs, None, n_steps=4)


def test_static_trial_targets_follow_trials_when_batch_size_matches_horizon():
    def where(state):
        return state.x

    target = jnp.array([[1.0], [2.0], [3.0]])
    n_steps = 4
    loss = TargetStateLoss("tracking", where)

    def specs(value):
        return _TrialSpecs(targets={loss.key: TargetSpec(value=value)})

    def retained(scale):
        values = jnp.broadcast_to((scale * target)[:, None, :], (3, n_steps, 1))
        return loss(_FakeState(values, values), specs(target), None).total

    def streamed(scale):
        def trial(value):
            contribution = make_streaming_loss_fn(loss, specs(value), None, n_steps)
            state = _FakeState(scale * value, scale * value)
            return jax.vmap(lambda t: contribution(state, t))(jnp.arange(n_steps)).sum()

        return jax.vmap(trial)(target).mean()

    for objective in (retained, streamed):
        value, gradient = jax.jit(jax.value_and_grad(objective))(1.0)
        assert value == 0.0
        assert gradient == 0.0
