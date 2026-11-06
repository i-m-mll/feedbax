import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.mechanics.musculoskeletal import (
    CompliantTendonMusculoskeletalArm,
    RigidTendonMusculoskeletalArm,
)
from feedbax.mechanics.skeleton.arm import TwoLinkArm, TwoLinkArmState
from feedbax.mechanics.skeleton.arm_dae import TwoLinkArmDAE


@pytest.fixture
def moving_arm():
    return TwoLinkArmState(angle=jnp.array([0.41, 0.72]), d_angle=jnp.array([0.9, -1.3]))


def _position_derivative(cartesian, state):
    step = 0.002
    ahead = eqx.tree_at(lambda arm: arm.angle, state, state.angle + step * state.d_angle)
    behind = eqx.tree_at(lambda arm: arm.angle, state, state.angle - step * state.d_angle)
    return (cartesian(ahead).pos - cartesian(behind).pos) / (2 * step)


@pytest.mark.parametrize(
    ("arm_type", "cartesian_method"),
    [
        (TwoLinkArm, "forward_kinematics"),
        (TwoLinkArmDAE, "forward_kinematics"),
        (RigidTendonMusculoskeletalArm, "_effector"),
        (CompliantTendonMusculoskeletalArm, "_effector"),
    ],
)
def test_cartesian_velocity_tracks_position(arm_type, cartesian_method, moving_arm):
    cartesian = getattr(arm_type(), cartesian_method)

    np.testing.assert_allclose(
        cartesian(moving_arm).vel,
        _position_derivative(cartesian, moving_arm),
        atol=3e-5,
        rtol=3e-4,
    )


@pytest.mark.parametrize("arm_type", [TwoLinkArm, TwoLinkArmDAE])
def test_effector_force_preserves_virtual_work(arm_type, moving_arm):
    arm = arm_type()
    force = jnp.array([2.1, -3.7])
    torque = arm.effector_force_to_torques(moving_arm.angle, force)
    effector_motion = _position_derivative(arm.effector, moving_arm)

    np.testing.assert_allclose(
        torque @ moving_arm.d_angle, force @ effector_motion, atol=1e-4, rtol=3e-4
    )
