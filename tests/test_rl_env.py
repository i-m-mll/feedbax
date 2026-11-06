"""Tests for feedbax.training.rl.env."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.mechanics.analytical_plant import AnalyticalMusculoskeletalPlant
from feedbax.mechanics.body import default_2link_bounds, sample_preset
from feedbax.mechanics.model_builder import ChainConfig
from feedbax.mechanics.muscle_config import default_6muscle_2link_topology
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.arm import TwoLinkArm
from feedbax.training.rl.env import (
    RLEnvConfig,
    auto_reset,
    rl_env_get_obs,
    rl_env_reset,
    rl_env_reset_sampled,
    rl_env_step,
)
from feedbax.training.rl.tasks import (
    TASK_HOLD,
    TASK_REACH,
    TASK_SWING,
    TASK_TRACK,
    TaskParams,
    reconstruct_trajectory,
    sample_task_params_jax,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def plant():
    return DirectForceInput(TwoLinkArm(), clip_states=False)


@pytest.fixture
def config():
    return RLEnvConfig(
        n_steps=100,
        dt=0.01,
        frame_skip=1,
        n_joints=2,
        n_muscles=2,
        action_scale=10.0,
        action_offset=-5.0,
    )


SEGMENT_LENGTHS = jnp.array([0.3, 0.25])


@pytest.fixture
def task(key):
    return sample_task_params_jax(
        key, 0, 100, 0.01,
        segment_lengths=SEGMENT_LENGTHS,
        use_fk=False,
        max_target_distance=10.0,
        use_curriculum=False,
        single_task=False,
    )


@pytest.fixture
def state(plant, config, task, key):
    return rl_env_reset(plant, config, task, key)


class TestRLEnvReset:
    def test_shapes(self, state, config):
        assert state.muscle_activations.shape == (config.n_muscles,)
        assert state.t_index == 0

    def test_finite(self, state):
        assert jnp.all(jnp.isfinite(state.muscle_activations))


class TestRLEnvStep:
    def test_step(self, plant, config, state):
        action = jnp.ones(config.n_muscles) * 0.5
        new_state, obs, reward, done = rl_env_step(plant, config, state, action)
        assert new_state.t_index == 1
        assert jnp.isfinite(reward)
        assert obs.ndim == 1

    def test_done_at_end(self, plant, config, state):
        """The episode reports done once the configured horizon is reached.

        Scanned under one compiled rollout: the same ``config.n_steps`` control
        steps, but the step function is traced once instead of dispatched from
        the host on every iteration.
        """
        action = jnp.zeros(config.n_muscles)

        @eqx.filter_jit
        def rollout(initial_state):
            def step(step_state, _):
                next_state, _obs, _reward, done = rl_env_step(
                    plant, config, step_state, action
                )
                return next_state, done

            return jax.lax.scan(step, initial_state, None, length=config.n_steps)

        _final_state, done_per_step = rollout(state)
        assert float(done_per_step[-1]) == 1.0

    def test_frame_skip_matches_five_physics_steps(self, plant, config, task, key):
        fine = replace(config, dt=0.002)
        skipped = replace(fine, frame_skip=5)
        initial = rl_env_reset(plant, fine, task, key)
        action = jnp.full(config.n_muscles, 0.8)
        step = eqx.filter_jit(rl_env_step)
        expected = initial
        for _ in range(5):
            expected, _, _, _ = step(plant, fine, expected, action)

        actual, _, _, _ = step(plant, skipped, initial, action)

        assert actual.t_index == 1
        np.testing.assert_allclose(actual.muscle_activations, expected.muscle_activations)
        np.testing.assert_allclose(
            actual.plant_state.skeleton.angle, expected.plant_state.skeleton.angle, atol=1e-7,
        )
        np.testing.assert_allclose(
            actual.plant_state.skeleton.d_angle, expected.plant_state.skeleton.d_angle, atol=1e-7,
        )


@pytest.mark.parametrize(
    "task_type", [TASK_REACH, TASK_HOLD, TASK_TRACK, TASK_SWING],
    ids=["reach", "hold", "track", "swing"],
)
def test_sampled_task_schedule_uses_control_clock(plant, config, key, task_type):
    def sample(dt, frame_skip):
        sampling_config = replace(
            config,
            n_steps=101,
            dt=dt,
            frame_skip=frame_skip,
            default_task_type=task_type,
        )
        return rl_env_reset_sampled(plant, sampling_config, key).task

    fine = sample(0.002, 5)
    coarse = sample(0.01, 1)
    fine_pos, fine_vel = reconstruct_trajectory(fine)
    coarse_pos, coarse_vel = reconstruct_trajectory(coarse)

    np.testing.assert_allclose(fine_pos, coarse_pos, atol=1e-6)
    np.testing.assert_allclose(fine_vel, coarse_vel, atol=1e-6)
    assert float(fine.tf) == pytest.approx(1.0)
    if task_type == TASK_HOLD:
        assert fine.perturb_time_idx == coarse.perturb_time_idx
        np.testing.assert_allclose(fine.perturb_force, coarse.perturb_force)
    else:
        assert jnp.max(jnp.linalg.norm(fine_vel, axis=-1)) > 0.1
    if task_type == TASK_SWING:
        angles = jnp.arctan2(fine_pos[:, 1], fine_pos[:, 0])
        # At 0.5 Hz the swing peaks after 0.5 s and crosses zero again at 1 s.
        np.testing.assert_allclose(angles[jnp.array([0, 50, 100])], [0.0, 0.6, 0.0], atol=1e-6)


@pytest.mark.parametrize("arm_kind", ["direct", "analytical"])
def test_hold_force_tracks_pose_and_releases(arm_kind, key):
    if arm_kind == "direct":
        plant = DirectForceInput(TwoLinkArm())
    else:
        plant = AnalyticalMusculoskeletalPlant.from_body_preset(
            sample_preset(default_2link_bounds(), key),
            ChainConfig(n_joints=2, muscle_topology=default_6muscle_2link_topology()),
        )
    config = RLEnvConfig(
        n_steps=3, dt=0.01, frame_skip=5, n_joints=2, n_muscles=plant.input_size,
    )
    force = jnp.array([4.0, -2.0])
    target = jnp.array([0.3, 0.3])
    task = TaskParams(
        task_type=TASK_HOLD,
        start_pos=target,
        end_pos=target,
        perturb_time_idx=1,
        perturb_force=force,
        control_points=jnp.broadcast_to(target, (6, 2)),
        t0=jnp.array(0.0),
        tf=jnp.array(config.n_steps * config.frame_skip * config.dt),
        dt=jnp.array(config.frame_skip * config.dt),
        n_steps=config.n_steps,
    )
    action = jnp.zeros(plant.input_size)
    arm = plant.skeleton
    state = rl_env_reset(plant, config, task, key)
    reference = eqx.tree_at(
        lambda s: (s.skeleton.angle, s.skeleton.d_angle),
        state.plant_state,
        (jnp.array([0.6, 0.9]), jnp.array([2.0, -1.0])),
    )
    actuator_torque = jnp.array([0.4, -0.3])
    preexisting_force = jnp.array([-1.0, 3.0])
    initial_skeleton = eqx.tree_at(lambda s: s.torque, reference.skeleton, actuator_torque)
    # Assign the same load twice, then let the HOLD perturbation replace it.
    for _ in range(2):
        initial_skeleton = arm.update_state_given_effector_force(
            preexisting_force, initial_skeleton,
        )
    state = state._replace(
        plant_state=eqx.tree_at(lambda s: s.skeleton, reference, initial_skeleton),
    )

    @jax.jit
    def reference_substep(plant_state, endpoint_force):
        skeleton = plant_state.skeleton

        def virtual_work(angle):
            posed = eqx.tree_at(lambda s: s.angle, skeleton, angle)
            return endpoint_force @ arm.effector(posed).pos

        # Use the explicit joint torque input to add the acceleration from
        # virtual work and the actuator without setting a held load.
        joint_torque = actuator_torque + jax.grad(virtual_work)(skeleton.angle)
        baseline = arm.vector_field(0.0, skeleton, jnp.zeros(2))
        driven = arm.vector_field(0.0, skeleton, joint_torque)
        derivative = plant.vector_field(0.0, plant_state, action)
        acceleration = derivative.skeleton.d_angle + (driven.d_angle - baseline.d_angle)
        return eqx.tree_at(
            lambda s: (s.skeleton.angle, s.skeleton.d_angle),
            plant_state,
            (
                skeleton.angle + config.dt * derivative.skeleton.angle,
                skeleton.d_angle + config.dt * acceleration,
            ),
        )

    for endpoint_force in (preexisting_force, force, jnp.zeros_like(force)):
        for _ in range(config.frame_skip):
            reference = reference_substep(reference, endpoint_force)
        state, _, _, _ = rl_env_step(plant, config, state, action)
        actual = state.plant_state.skeleton
        np.testing.assert_allclose(actual.angle, reference.skeleton.angle, atol=1e-6)
        np.testing.assert_allclose(actual.d_angle, reference.skeleton.d_angle, atol=1e-6)


class TestRLEnvObs:
    def test_shape(self, plant, config, state):
        obs = rl_env_get_obs(plant, config, state)
        expected_dim = config.n_joints * 2 + config.n_muscles + 2 + 2 + 2 + 1
        assert obs.shape == (expected_dim,)
        assert jnp.all(jnp.isfinite(obs))


class TestAutoReset:
    def test_reset_on_done(self, plant, config, state, key):
        done = jnp.array(1.0)
        new_state = auto_reset(plant, config, state, done, key)
        assert new_state.t_index == 0

    def test_no_reset_on_not_done(self, plant, config, state, key):
        # Step once to get t_index=1
        action = jnp.zeros(config.n_muscles)
        state, _, _, _ = rl_env_step(plant, config, state, action)
        done = jnp.array(0.0)
        new_state = auto_reset(plant, config, state, done, key)
        assert new_state.t_index == 1
