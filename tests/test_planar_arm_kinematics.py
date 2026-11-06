import equinox as eqx
import jax
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


def test_inverse_kinematics_reuses_arm_across_jit_traces(moving_arm):
    arm = TwoLinkArm()
    target = arm.effector(moving_arm)

    with jax.checking_leaks():
        first = jax.jit(lambda x: arm.inverse_kinematics(x))(target)
        eager = arm.inverse_kinematics(target)
        second = jax.jit(lambda x: arm.inverse_kinematics(x))(target)

    for result in (first, eager, second):
        np.testing.assert_allclose(result.angle, moving_arm.angle, atol=1e-6)
        np.testing.assert_allclose(result.d_angle, moving_arm.d_angle, atol=1e-6)


def test_vector_field_jit_uses_replaced_arm_mass(moving_arm):
    masses = jnp.array([1.4, 1.7])
    arm = eqx.tree_at(lambda arm: arm.m, TwoLinkArm(), masses)
    torque = jnp.array([0.4, -0.3])

    with jax.checking_leaks():
        actual = jax.jit(lambda u: arm.vector_field(0.0, moving_arm, u))(torque)

    expected = TwoLinkArm(m=masses).vector_field(0.0, moving_arm, torque)
    np.testing.assert_allclose(actual.d_angle, expected.d_angle, rtol=1e-6)
