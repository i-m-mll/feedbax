from __future__ import annotations

from inspect import Parameter, signature

import jax
import jax.numpy as jnp
import optax
import pytest

from feedbax.training.optimizers import OPTIMIZERS, SCHEDULES, _SIGNATURE_DEFAULTS, build_schedule, build_step_optimizer


pytestmark = pytest.mark.feedbax_contract


@pytest.mark.parametrize("key", sorted(OPTIMIZERS.keys() | SCHEDULES.keys()))
def test_owned_defaults_agree_with_optax_signatures(key):
    entry = (OPTIMIZERS | SCHEDULES)[key]
    expected = {
        name: parameter.default for name, parameter in signature(entry.factory).parameters.items()
        if name in entry.kinds and parameter.default is not Parameter.empty
    }
    assert _SIGNATURE_DEFAULTS[key] == expected
    spec = entry.model()
    for name, default in expected.items():
        if entry.kinds[name] == "schedule" and default is not None:
            assert spec.model_dump()[name] == {"type": "constant_schedule", "value": default}
        else:
            assert getattr(spec, name) == default


@pytest.mark.parametrize("key", sorted(OPTIMIZERS))
def test_every_registered_optimizer_steps_with_its_defaults(key: str) -> None:
    params = {"w": jnp.ones((3, 2)), "b": jnp.ones(2)}
    optimizer = build_step_optimizer(OPTIMIZERS[key].model(), gradient_clip=1.0)
    update = optax.with_extra_args_support(optimizer).update
    updates, _ = jax.jit(lambda state: update(params, state, params, schedule_step=3))(optimizer.init(params))
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(updates))


@pytest.mark.parametrize("key", sorted(SCHEDULES))
def test_every_registered_schedule_evaluates_with_its_defaults(key: str) -> None:
    assert jnp.isfinite(build_schedule(SCHEDULES[key].model())(5))


@pytest.mark.parametrize("key", sorted(SCHEDULES))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["float32", "float64"])
def test_scheduled_optimizer_keeps_the_parameter_dtype(key, dtype):
    with jax.enable_x64():
        schedule = SCHEDULES[key].model()
        optimizer = build_step_optimizer(OPTIMIZERS["adam"].model(learning_rate=schedule))
        params = jnp.ones(3, dtype=dtype)
        initial = optimizer.init(params)

        @jax.jit
        def step(state, clock):
            # TrainingLoop chooses between accepting a candidate and keeping the current state.
            _, next_state = optimizer.update(params, state, params, schedule_step=clock)
            return jax.lax.cond(jnp.all(jnp.isfinite(params)), lambda: next_state, lambda: state)

        updated = step(initial, jnp.uint32(2))
        initial_rate = initial[0].hyperparams["learning_rate"]
        updated_rate = updated[0].hyperparams["learning_rate"]
        assert initial_rate.dtype == updated_rate.dtype == params.dtype
        assert jnp.allclose(updated_rate, build_schedule(schedule)(2))
