"""Analytical acceptance for PPO's compiled rollout and update paths."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy.special import digamma, expit
from scipy.stats import beta

from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.runtime.state import CartesianState
from feedbax.training.rl.env import RLEnvConfig, rl_env_reset, rl_env_step_with_outcome
from feedbax.training.rl.obs_norm import ObsNormState, init_obs_norm
from feedbax.training.rl.policy import (
    ActorCritic,
    BetaParams,
    init_lattice_noise,
    load_policy,
    save_policy,
)
from feedbax.training.rl.ppo import (
    PPOConfig,
    TrainingEnhancements,
    _EpisodeCarry,
    _PPOTrainCarry,
    _collect_rollout_extended,
    _ppo_loss,
    _run_ppo_updates,
    _stack_pytrees,
    _update_one,
    collect_rollouts_batched,
    train_ppo_batched,
)
from feedbax.training.rl.tasks import (
    TASK_HOLD,
    TASK_REACH,
    TASK_SWING,
    init_curriculum,
    reach_task_params,
)


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


def _coasting_observations(x, phase, active_muscles):
    raw = np.zeros((*x.shape, 13))
    raw[..., 0] = raw[..., 6] = x
    raw[..., 2] = 1.0
    raw[..., 4:6] = np.where(phase[..., None] == 0, 0, active_muscles)
    raw[..., -1] = phase
    return raw


def test_swing_rollout_uses_control_clock():
    skeleton = _CoastingPointMass(mass=jnp.array(1.0), start_x=jnp.array(0.0))
    plant = _stack_pytrees(DirectForceInput(skeleton, clip_states=False))
    policy_key, rollout_key = jax.random.split(jax.random.key(42))
    policy = _stack_pytrees(ActorCritic(13, 2, 4, 1, key=policy_key))
    fine = replace(_config(), n_steps=6, dt=0.002, frame_skip=5)
    coarse = replace(fine, dt=0.01, frame_skip=1)
    task_types = jnp.array([TASK_SWING])

    fine_result = collect_rollouts_batched(plant, fine, policy, rollout_key, task_types)
    coarse_result = collect_rollouts_batched(plant, coarse, policy, rollout_key, task_types)

    expected_seconds = [[[0.0, 0.01, 0.02, 0.03, 0.04, 0.05]]]
    np.testing.assert_allclose(fine_result["timestamps"], expected_seconds)
    np.testing.assert_allclose(fine_result["timestamps"], coarse_result["timestamps"])
    np.testing.assert_allclose(
        fine_result["effector_pos"][..., 0], fine_result["timestamps"], atol=1e-7,
    )
    np.testing.assert_allclose(fine_result["task_target"], coarse_result["task_target"], atol=1e-6)


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


@pytest.mark.parametrize("n_minibatches", [0, -1, 4, 8])
def test_training_rejects_invalid_minibatch_partitions(n_minibatches):
    with pytest.raises(ValueError, match=r"n_minibatches .*rollout size per body \(6\)"):
        train_ppo_batched(
            _stack_pytrees(_plant(-1.0)),
            _config(),
            PPOConfig(
                n_steps_per_update=2, n_minibatches=n_minibatches, total_timesteps=0,
            ),
            jax.random.PRNGKey(0),
            n_envs=3,
        )


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


@pytest.mark.parametrize("obs_norm", [True, False], ids=["normalized", "disabled"])
def test_trained_policy_executes_on_raw_observations_after_save_and_load(tmp_path, obs_norm):
    starts = np.array([-4.0, -18.0], dtype=np.float32)
    plants = _stack_pytrees(*[_plant(x) for x in starts])
    config = eqx.tree_at(lambda cfg: (cfg.tau_act, cfg.tau_deact), _config(), (1.0, 1.0))
    key = jax.random.PRNGKey(7)
    policy, _ = train_ppo_batched(
        plants,
        config,
        PPOConfig(
            n_steps_per_update=2, total_timesteps=4, n_epochs=1, n_minibatches=1,
            hidden_dim=4, hidden_layers=1,
        ),
        key,
        n_envs=1,
        enhancements=TrainingEnhancements(obs_norm=obs_norm),
    )

    raw_obs = _coasting_observations(starts, np.zeros_like(starts), np.zeros(2)).astype(np.float32)
    if obs_norm:
        # Each body coasts for three steps, then resets for the fourth observation.
        training_x = starts[:, None] + np.array([0, 1, 2, 0])
        np.testing.assert_array_equal(policy.obs_norm.count, [4, 4])
        np.testing.assert_allclose(policy.obs_norm.mean[:, 0], training_x.mean(axis=1))
        np.testing.assert_allclose(policy.obs_norm.var[:, 0], training_x.var(axis=1))
        mean = np.asarray(policy.obs_norm.mean)
        variance = np.asarray(policy.obs_norm.var)
        network_obs = np.clip((raw_obs - mean) / np.sqrt(variance + 1e-8), -10, 10)
    else:
        network_obs = raw_obs

    # Reuse only the network weights in the oracle and feed it the expected inputs.
    bare = ActorCritic(obs_dim=13, action_dim=2, hidden_dim=4, hidden_layers=1, key=key)
    bare = eqx.tree_at(lambda p: (p.actor, p.critic), bare, (policy.actor, policy.critic))
    forward = eqx.filter_vmap(ActorCritic.forward)
    sample = eqx.filter_vmap(ActorCritic.sample_action)
    expected_forward = forward(bare, jnp.asarray(network_obs))
    rollout_key = jax.random.key(17)
    rollout_keys = jax.random.split(rollout_key, len(starts))
    step_keys = jax.vmap(lambda k: jax.random.split(k, 3)[0])(rollout_keys)
    action_keys = jax.vmap(lambda k: jax.random.split(k)[1])(step_keys)
    expected_sample = sample(bare, jnp.asarray(network_obs), action_keys)

    path = tmp_path / "policy.eqx"
    save_policy(path, policy)
    for candidate in (policy, load_policy(path)):
        actual_forward = forward(candidate, jnp.asarray(raw_obs))
        for actual, expected in zip(actual_forward, expected_forward, strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-6)
        for actual, expected in zip(
            sample(candidate, jnp.asarray(raw_obs), action_keys), expected_sample, strict=True,
        ):
            np.testing.assert_allclose(actual, expected, atol=1e-6)
        rollouts = collect_rollouts_batched(
            plants, config, candidate, rollout_key, jnp.array([TASK_REACH]),
        )
        # Unit time constants make the first activation equal the sampled action.
        np.testing.assert_allclose(
            rollouts["muscle_activations"][:, 0, 1], expected_sample[0], atol=1e-6,
        )

    if obs_norm:
        body = jax.tree.map(lambda leaf: leaf[1] if eqx.is_array(leaf) else leaf, policy)
        configured = ActorCritic(
            13, 2, 4, 1, key=key, obs_norm=body.obs_norm, obs_norm_eps=0.5, obs_norm_clip=0.6,
        )
        configured = eqx.tree_at(
            lambda p: (p.actor, p.critic), configured, (body.actor, body.critic),
        )
        save_policy(path, configured)
        probe = raw_obs[1].copy()
        probe[0] += 0.5
        expected_obs = np.clip((probe - mean[1]) / np.sqrt(variance[1] + 0.5), -0.6, 0.6)
        single_bare = eqx.tree_at(lambda p: p.obs_norm, body, None)
        actual = load_policy(path).forward(jnp.asarray(probe))
        expected = single_bare.forward(jnp.asarray(expected_obs))
        for actual_part, expected_part in zip(actual, expected, strict=True):
            np.testing.assert_allclose(actual_part, expected_part, atol=1e-6)


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


def test_mixed_lanes_use_distinct_episode_denominators_and_pool_raw_observations(monkeypatch):
    monkeypatch.setattr(BetaParams, "sample_logits", lambda self, key: jnp.zeros(2))
    config = eqx.tree_at(lambda cfg: (cfg.tau_act, cfg.tau_deact), _config(), (1.0, 1.0))
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
    final_carry, history = _run_ppo_updates(
        _stack_pytrees(_plant(-10.0)), config, carry, static_policy, optimizer,
        total_updates=2, n_steps=3, n_envs=4, n_minibatches=1, n_epochs=1,
        gamma=0.99, gae_lambda=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
        lattice_resample_interval=2, curriculum_arm_reach=0.5,
        reward_ramp_updates=jnp.array(0.0), use_obs_norm=jnp.array(True),
        use_lattice=True, use_curriculum=jnp.array(True),
    )
    np.testing.assert_array_equal(history.per_body_completed_episodes[0], [4])
    np.testing.assert_array_equal(history.per_body_eligible_episodes[0], [3])
    np.testing.assert_allclose(history.per_body_success_rate[0], [2 / 3])
    np.testing.assert_allclose(history.per_body_mean_return[0], [-5.5])

    # Raw x before each action, including the immediate resets to x=-10.
    x = np.array([
        [-1, -2, -10, -3],
        [-10, -1, -9, -2],
        [-9, -10, -8, -1],
        [-8, -9, -10, -10],
        [-10, -8, -9, -9],
        [-9, -10, -8, -8],
    ])
    phase = np.array([
        [0, 0, 0, 0],
        [0, 0.5, 0.5, 0.5],
        [0.5, 0, 1, 1],
        [1, 0.5, 0, 0],
        [0, 1, 0.5, 0.5],
        [0.5, 0, 1, 1],
    ])
    raw = _coasting_observations(x, phase, np.array([0.5, 0.5])).reshape(x.size, -1)
    np.testing.assert_array_equal(final_carry.obs_norm.count, [len(raw)])
    np.testing.assert_allclose(final_carry.obs_norm.mean[0], raw.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(final_carry.obs_norm.var[0], raw.var(axis=0), atol=1e-6)


@pytest.mark.parametrize("use_lattice", [False, True], ids=["ordinary", "lattice"])
def test_ppo_replays_tail_draws_with_their_sampling_distribution(use_lattice, monkeypatch):
    # Force both clamp tails without extreme concentrations in the density oracle.
    tail_logits = jnp.array([-12.0, 22.0])
    tails = expit(np.asarray(tail_logits, dtype=np.float64))
    monkeypatch.setattr(BetaParams, "sample_logits", lambda self, key: tail_logits)
    config = eqx.tree_at(lambda cfg: (cfg.tau_act, cfg.tau_deact), _config(), (1.0, 1.0))
    plant = _plant(-3.0)
    key = jax.random.PRNGKey(42)
    policy = ActorCritic(obs_dim=13, action_dim=2, hidden_dim=4, hidden_layers=1, key=key)
    task = reach_task_params(
        jnp.zeros(2), jnp.zeros(2), config.n_steps, config.dt,
    )._replace(task_type=jnp.array(TASK_REACH))
    lane_starts = (-3.0, -5.0, -12.0)
    n_envs = len(lane_starts)
    states = _stack_pytrees(*[
        rl_env_reset(_plant(x), config, task, key) for x in lane_starts
    ])
    episodes = _EpisodeCarry(jnp.zeros(n_envs), jnp.zeros(n_envs, dtype=jnp.int32))
    lattice = init_lattice_noise(4, key)
    lattice = eqx.tree_at(
        lambda state: (state.noise, state.steps_since_resample),
        lattice,
        (jnp.array([0.7, -0.5, 0.3, 0.9]), jnp.array(1, dtype=jnp.int32)),
    )
    collect = eqx.filter_jit(_collect_rollout_extended)

    prior = ObsNormState(jnp.array(2.0), jnp.full(13, 0.25), jnp.full(13, 0.25))

    def collect_prefix(n_steps, use_obs_norm):
        return collect(
            plant, config, policy, states, episodes, key, n_steps, n_envs,
            prior, lattice, init_curriculum(), 1, 0.5,
            jnp.array(use_obs_norm), use_lattice, jnp.array(False), jnp.array(1.0),
        )

    # The point mass moves one unit per step, with zero targets and no resets.
    # With unit time constants, each action sets the next muscle activations.
    clipped_tails = np.clip(tails, 1e-4, 1.0 - 1e-4)
    x = np.arange(3)[:, None] + np.array(lane_starts)
    phase = np.broadcast_to(np.arange(3)[:, None] / (config.n_steps - 1), x.shape)
    raw = _coasting_observations(x, phase, clipped_tails)
    scoring_obs = np.clip(
        (raw - np.asarray(prior.mean)) / np.sqrt(np.asarray(prior.var) + 1e-8), -10, 10,
    )
    _, rollout, last_values, *_ = collect_prefix(2, True)
    np.testing.assert_allclose(rollout.obs, scoring_obs[:2], atol=1e-6)
    _, expected_bootstrap = policy.dist_and_value(jnp.asarray(scoring_obs[2]))
    np.testing.assert_allclose(last_values, expected_bootstrap, atol=1e-6)

    after_first, disabled, _, after_first_lattice, _, _, unchanged = collect_prefix(1, False)
    np.testing.assert_array_equal(disabled.obs, raw[:1])
    for actual, expected in zip(jax.tree.leaves(unchanged), jax.tree.leaves(prior), strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        rollout.action_logits, np.broadcast_to(tail_logits, rollout.action_logits.shape),
    )
    np.testing.assert_allclose(
        after_first.muscle_activations,
        np.broadcast_to(clipped_tails, (n_envs, policy.action_dim)),
    )
    replay_noise = None
    expected_noise = None
    if use_lattice:
        assert not np.allclose(lattice.noise, after_first_lattice.noise)
        expected_noise = jnp.stack([lattice.noise, after_first_lattice.noise])
        np.testing.assert_allclose(rollout.latent_noise, expected_noise)
        replay_noise = jnp.repeat(rollout.latent_noise, n_envs, axis=0)
    else:
        assert rollout.latent_noise is None

    obs = rollout.obs.reshape(-1, policy.obs_dim)
    action_logits = rollout.action_logits.reshape(-1, policy.action_dim)
    actions = expit(np.asarray(action_logits, dtype=np.float64))
    old_logp = rollout.log_probs.reshape(-1)
    clip_eps = 4.0

    @eqx.filter_jit
    def ratios(actor_critic):
        def ratio(o, a, logp, noise):
            loss, _ = _ppo_loss(
                actor_critic, o[None], a[None], logp[None], jnp.ones(1), jnp.zeros(1),
                clip_eps, 0.0, 0.0, latent_noise=None if noise is None else noise[None],
            )
            return -loss

        return jax.vmap(ratio)(obs, action_logits, old_logp, replay_noise)

    np.testing.assert_allclose(ratios(policy), 1.0, atol=2e-6)
    changed_policy = eqx.tree_at(
        lambda model: model.actor.layers[-1].bias,
        policy,
        policy.actor.layers[-1].bias + jnp.array([0.2, -0.1, 0.1, -0.2]),
    )

    def beta_parameters(actor_critic):
        hidden, output = actor_critic.actor.layers
        features = np.tanh(np.asarray(obs) @ np.asarray(hidden.weight).T + np.asarray(hidden.bias))
        if expected_noise is not None:
            features += np.repeat(expected_noise, n_envs, axis=0)
        logits = features @ np.asarray(output.weight).T + np.asarray(output.bias)
        alpha, beta_param = np.split(1.0 + np.logaddexp(0, logits), 2, axis=-1)
        return alpha, beta_param, logits, features

    old_alpha, old_beta, _, _ = beta_parameters(policy)
    alpha, beta_param, logits, features = beta_parameters(changed_policy)
    logp = beta.logpdf(actions, alpha, beta_param).sum(axis=-1)
    reference_old_logp = beta.logpdf(actions, old_alpha, old_beta).sum(axis=-1)
    np.testing.assert_allclose(old_logp, reference_old_logp, atol=1e-5)
    expected_ratio = np.exp(logp - reference_old_logp)
    assert np.all(expected_ratio < 1 + clip_eps)
    np.testing.assert_allclose(ratios(changed_policy), expected_ratio, rtol=1e-5)

    # With gamma zero, these rewards give each sample a distinct advantage.
    advantages = jnp.array([[-3.0, 1.0, 4.0], [2.0, -1.0, 0.0]])
    rollout = rollout._replace(rewards=rollout.values + advantages)
    weights = (np.ravel(advantages) - np.mean(advantages)) / np.std(advantages)
    # The optimizer records the sum of singleton gradients with parameters fixed.
    optimizer = optax.GradientTransformation(
        lambda _: jnp.zeros_like(changed_policy.actor.layers[-1].weight),
        lambda grads, accumulated, _: (
            jax.tree.map(jnp.zeros_like, grads),
            accumulated + grads.actor.layers[-1].weight,
        ),
    )
    _, accumulated = eqx.filter_jit(_update_one)(
        changed_policy, optimizer.init(changed_policy), rollout, last_values, key, optimizer,
        gamma=0.0, gae_lambda=0.0, clip_eps=clip_eps, vf_coef=0.0, ent_coef=0.0,
        n_minibatches=actions.shape[0], n_epochs=1,
    )
    log_density_gradient = np.concatenate([
        np.log(actions) + digamma(alpha + beta_param) - digamma(alpha),
        np.log1p(-actions) + digamma(alpha + beta_param) - digamma(beta_param),
    ], axis=-1)
    loss_logit_gradient = -(weights * expected_ratio)[:, None] * log_density_gradient * expit(logits)
    expected_gradient = loss_logit_gradient.T @ features
    # Individual float32 tail scores amplify roundoff before the gradients are summed.
    np.testing.assert_allclose(accumulated, expected_gradient, atol=3e-4, rtol=2e-5)
