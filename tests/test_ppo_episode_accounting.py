"""Analytical acceptance for PPO's compiled episode accounting."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.runtime.state import CartesianState
from feedbax.training.rl.env import RLEnvConfig, rl_env_reset, rl_env_step_with_outcome
from feedbax.training.rl.obs_norm import init_obs_norm
from feedbax.training.rl.policy import ActorCritic, init_lattice_noise
from feedbax.training.rl.ppo import (
    PPOConfig,
    TrainingEnhancements,
    _EpisodeCarry,
    _PPOTrainCarry,
    _run_ppo_updates,
    _stack_pytrees,
    train_ppo_batched,
)
from feedbax.training.rl.tasks import TASK_HOLD, TASK_REACH, init_curriculum, reach_task_params


class _PointState(CartesianState):
    @property
    def qpos(self):
        return self.pos

    @property
    def qvel(self):
        return self.vel


class _CoastingPointMass(PointMass):
    start_x: jax.Array = eqx.field(kw_only=True)

    def init(self, *, key):
        return _PointState(
            pos=jnp.stack([self.start_x, jnp.zeros_like(self.start_x)]),
            vel=jnp.array([1.0, 0.0]),
        )

    def vector_field(self, t, state, input):
        return _PointState(pos=state.vel, vel=jnp.zeros_like(state.vel))


class _CoastingPlant(DirectForceInput):
    @property
    def segment_lengths(self):
        # Degenerate forward kinematics makes sampled targets the origin.
        return jnp.zeros(2)


def _plant(start_x):
    skeleton = _CoastingPointMass(mass=jnp.array(1.0), start_x=jnp.asarray(start_x))
    return _CoastingPlant(skeleton, clip_states=False)


def _config():
    return RLEnvConfig(
        n_steps=3,
        dt=1.0,
        frame_skip=1,
        n_joints=2,
        n_muscles=2,
        action_scale=0.0,
        effort_weight=0.0,
        velocity_weight=0.0,
        hold_bonus=2.0,
        hold_threshold=0.1,
        default_task_type=TASK_REACH,
    )


def test_reach_uses_endpoint_even_when_the_effector_matches_the_reference():
    plant = _plant(-1.0)
    config = _config()
    key = jax.random.PRNGKey(0)
    task = reach_task_params(jnp.array([0.0, 0.0]), jnp.array([2.0, 0.0]), 3, 1.0)
    state = rl_env_reset(plant, config, task, key)

    state, _, reward, outcome = rl_env_step_with_outcome(plant, config, state, jnp.zeros(2))
    assert reward == 2.0
    assert not outcome.completed
    assert outcome.success_eligible
    assert not outcome.success


def test_training_reports_completed_returns_and_uses_current_stage_evidence():
    # Bodies reach x=0 on step 1, 2, never within the horizon, and exactly
    # on the horizon respectively. Rewards are -|x| plus 2 at the origin.
    plants = _stack_pytrees(*[_plant(x) for x in (-1.0, -2.0, -10.0, -3.0)])
    _, metrics = train_ppo_batched(
        plants,
        _config(),
        PPOConfig(
            n_steps_per_update=2, total_timesteps=34, n_epochs=1, n_minibatches=1,
            hidden_dim=4, hidden_layers=1,
        ),
        jax.random.PRNGKey(1),
        n_envs=1,
        enhancements=TrainingEnhancements(curriculum=True),
    )

    expected_counts = [[2, 1, 0, 0], [2, 1, 1, 1], [2, 1, 1, 1]]
    np.testing.assert_array_equal(metrics["per_body_completed_episodes"][:3], expected_counts)
    np.testing.assert_array_equal(metrics["per_body_eligible_episodes"][:3], expected_counts)
    np.testing.assert_allclose(metrics["per_body_mean_return"][:3], [
        [2, 1, 0, 0], [2, 1, -24, -1], [2, 1, -24, -1],
    ])
    np.testing.assert_allclose(metrics["per_body_success_rate"][:3], [
        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 0, 1],
    ])

    stages = np.asarray(metrics["curriculum_stages"])
    np.testing.assert_array_equal(stages[:5, :2], [[0, 0]] * 4 + [[1, 1]])
    np.testing.assert_array_equal(np.asarray(metrics["per_body_success_rate"])[:, :2], 1)
    np.testing.assert_array_equal(stages[:, 2], 0)
    # The fourth body finishes every three steps across two-step windows.
    # Empty windows preserve its streak. The completion on update 9 still
    # counts as success in metrics but comes from the previous stage.
    np.testing.assert_array_equal(stages[:, 3], [0] * 7 + [1] * 9 + [2])
    assert metrics["per_body_success_rate"][8][3] == 1


def test_reward_annealing_changes_collected_distance_penalty():
    plants = _stack_pytrees(_plant(-2.0))
    env_config = eqx.tree_at(lambda config: config.velocity_weight, _config(), 1.0)
    ppo_config = PPOConfig(
        n_steps_per_update=2, total_timesteps=8, n_epochs=1, n_minibatches=1,
        hidden_dim=4, hidden_layers=1,
    )
    key = jax.random.PRNGKey(3)
    _, annealed = train_ppo_batched(
        plants, env_config, ppo_config, key, n_envs=1,
        enhancements=TrainingEnhancements(
            reward_annealing=True, annealing_start_fraction=0.5,
        ),
    )
    _, default = train_ppo_batched(plants, env_config, ppo_config, key, n_envs=1)

    # Each update finishes a reach with distances 1 and 0, a +2 proximity
    # bonus, and a -0.3 velocity penalty at the target.
    np.testing.assert_allclose(annealed["per_body_mean_return"], [[0.7], [0.7], [1.2], [1.7]])
    np.testing.assert_allclose(default["per_body_mean_return"], [[0.7]] * 4)
    np.testing.assert_array_equal(annealed["distance_weight_schedule"], [1, 1, 0.5, 0])
    assert "distance_weight_schedule" not in default


def test_mixed_lanes_use_distinct_return_and_success_denominators():
    config = _config()
    key = jax.random.PRNGKey(2)
    reach = reach_task_params(
        jnp.zeros(2), jnp.zeros(2), config.n_steps, config.dt,
    )._replace(task_type=jnp.array(TASK_REACH))
    hold = reach._replace(task_type=jnp.array(TASK_HOLD))
    states = _stack_pytrees(*[
        rl_env_reset(_plant(x), config, task, key)
        for x, task in [(-1.0, reach), (-2.0, reach), (-10.0, reach), (-3.0, hold)]
    ])
    policy = ActorCritic(obs_dim=13, action_dim=2, hidden_dim=4, hidden_layers=1, key=key)
    dynamic_policy, static_policy = eqx.partition(_stack_pytrees(policy), eqx.is_array)
    optimizer = optax.adam(3e-4)
    carry = _PPOTrainCarry(
        key=key,
        dynamic_policy=dynamic_policy,
        opt_state=_stack_pytrees(optimizer.init(eqx.filter(policy, eqx.is_array))),
        states=_stack_pytrees(states),
        episodes=_EpisodeCarry(jnp.zeros((1, 4)), jnp.zeros((1, 4), dtype=jnp.int32)),
        obs_norm=_stack_pytrees(init_obs_norm(13)),
        lattice=_stack_pytrees(init_lattice_noise(4, key)),
        curriculum=_stack_pytrees(init_curriculum()),
    )
    # A reset at x=-10 cannot create a second completion within the window.
    _, history = _run_ppo_updates(
        _stack_pytrees(_plant(-10.0)), config, carry, static_policy, optimizer,
        total_updates=1, n_steps=3, n_envs=4, n_minibatches=1, n_train_iters=1,
        gamma=0.99, gae_lambda=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
        lattice_resample_interval=2, curriculum_arm_reach=0.5,
        reward_ramp_updates=jnp.array(0.0), use_obs_norm=jnp.array(True),
        use_lattice=jnp.array(True), use_curriculum=jnp.array(True),
    )
    np.testing.assert_array_equal(history.per_body_completed_episodes, [[4]])
    np.testing.assert_array_equal(history.per_body_eligible_episodes, [[3]])
    np.testing.assert_allclose(history.per_body_success_rate, [[2 / 3]])
    np.testing.assert_allclose(history.per_body_mean_return, [[-5.5]])
