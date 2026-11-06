import equinox as eqx
from equinox.nn import State
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.mechanics.musculoskeletal import (
    CompliantTendonMusculoskeletalArm,
    RigidTendonMusculoskeletalArm,
)
from feedbax.mechanics.skeleton.arm import TwoLinkArmState
from feedbax.mechanics.skeleton.arm_dae import TwoLinkArmDAE


@pytest.mark.parametrize(
    "arm_type",
    [TwoLinkArmDAE, RigidTendonMusculoskeletalArm, CompliantTendonMusculoskeletalArm],
)
def test_arm_dynamics_uses_live_inertia_across_traces(arm_type):
    arm = arm_type()
    moving = TwoLinkArmState(
        angle=jnp.array([0.41, 0.72]),
        d_angle=jnp.array([0.9, -1.3]),
    )
    if isinstance(arm, TwoLinkArmDAE):
        control = jnp.array([0.4, -0.3])

        def mass(model):
            return model.params.m

        def acceleration(model, control):
            return model.vector_field(0.0, moving, control).d_angle
    else:
        control = jnp.full(arm.n_muscles, 0.25)
        if isinstance(arm, RigidTendonMusculoskeletalArm):
            state = State(arm).get(arm.state_index)
            field = RigidTendonMusculoskeletalArm._vector_field
        else:
            state = arm.init_system_state(key=jax.random.key(0))
            field = CompliantTendonMusculoskeletalArm.vector_field
        state = eqx.tree_at(lambda s: s.arm, state, moving)

        def mass(model):
            return model.arm_m

        def acceleration(model, control):
            return field(model, 0.0, state, control).arm.d_angle

    with jax.checking_leaks():
        compiled = jax.jit(lambda u: acceleration(arm, u))(control)
        eager = acceleration(arm, control)
        heavier = eqx.tree_at(mass, arm, mass(arm).at[1].multiply(1.7))
        updated = jax.jit(lambda u: acceleration(heavier, u))(control)

    np.testing.assert_allclose(compiled, eager, rtol=2e-5, atol=5e-4)
    np.testing.assert_allclose(updated, acceleration(heavier, control), rtol=2e-5, atol=5e-4)
    assert np.linalg.norm(updated - compiled) > 0.1
