"""A real joint-limit crossing through each analytical step path."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.mechanics import DiffraxBackend, Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.arm import TwoLinkArm, TwoLinkArmState
from feedbax.runtime.graph import init_state_from_component
from feedbax.training.rl.env import RLEnvConfig, rl_env_reset_sampled, rl_env_step


@pytest.mark.usefixtures("enable_jax_x64")
@pytest.mark.parametrize("path", ["legacy", "diffrax", "rl"])
def test_two_link_step_projects_only_the_crossing_angle(path):
    key = jax.random.PRNGKey(0)
    dt = 0.02
    arm = TwoLinkArm()
    dtype = jnp.float32 if path == "rl" else jnp.float64
    start = TwoLinkArmState(
        angle=jnp.array([0.01, 0.5], dtype=dtype),
        d_angle=jnp.array([-2.0, 0.3], dtype=dtype),
        torque=jnp.array([0.4, -0.3], dtype=dtype),
        effector_force=jnp.zeros(2, dtype=dtype),
    )
    action = jnp.zeros(2, dtype=dtype)

    def step(clip_states):
        plant = DirectForceInput(arm, clip_states=clip_states)
        if path == "rl":
            config = RLEnvConfig(n_steps=3, dt=dt, frame_skip=1, n_joints=2, n_muscles=2)
            state = rl_env_reset_sampled(plant, config, key)
            state = state._replace(
                plant_state=eqx.tree_at(lambda s: s.skeleton, state.plant_state, start)
            )
            state, obs, _, _ = eqx.filter_jit(rl_env_step)(plant, config, state, action)
            return state.plant_state.skeleton, obs[6:8]

        backend = DiffraxBackend(control_dt=dt) if path == "diffrax" else None
        mechanics = Mechanics(plant, dt, backend=backend)
        state = init_state_from_component(mechanics)
        initial = state.get(mechanics.state_index)
        initial = eqx.tree_at(lambda s: s.plant.skeleton, initial, start)
        state = state.set(mechanics.state_index, initial)
        outputs, state = eqx.filter_jit(mechanics)({"force": action}, state, key=key)
        stepped = state.get(mechanics.state_index).plant.skeleton
        np.testing.assert_allclose(outputs["effector"].vel, arm.effector(stepped).vel)
        return stepped, outputs["effector"].pos

    projected, effector_pos = step(True)
    unprojected, _ = step(False)

    assert unprojected.angle[0] < 0.0
    np.testing.assert_array_equal(projected.angle, [0.0, unprojected.angle[1]])
    for field in ("d_angle", "torque", "effector_force"):
        np.testing.assert_array_equal(getattr(projected, field), getattr(unprojected, field))
    for actual, before in zip(jax.tree.leaves(projected), jax.tree.leaves(start), strict=True):
        assert actual.dtype == before.dtype
    np.testing.assert_allclose(effector_pos, arm.effector(projected).pos)
