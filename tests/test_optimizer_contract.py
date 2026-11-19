from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from feedbax.training.optimizers import OPTIMIZERS, SCHEDULES, build_schedule, build_step_optimizer


pytestmark = pytest.mark.feedbax_contract


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
