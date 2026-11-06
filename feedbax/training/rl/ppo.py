"""Vectorized PPO + GAE training for RL policies.

Rollout collection uses jax.vmap over N parallel environments and
jax.lax.scan over timesteps. GAE is computed with a reversed scan.
Generalized to work with any feedbax AbstractPlant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import optax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from feedbax.mechanics.plant import AbstractPlant
from feedbax.training.rl.env import (
    RLEnvConfig,
    RLEnvState,
    auto_reset,
    rl_env_get_obs,
    rl_env_reset,
    rl_env_reset_sampled,
    rl_env_step,
    rl_env_step_with_outcome,
)
from feedbax.training.rl.obs_norm import (
    ObsNormState,
    init_obs_norm,
    normalize_obs,
    update_obs_norm,
)
from feedbax.training.rl.policy import (
    ActorCritic,
    LatticeNoiseState,
    _beta_sample_from_logits,
    _clip_action,
    init_lattice_noise,
    maybe_resample_noise,
)
from feedbax.training.rl.tasks import (
    CurriculumState,
    init_curriculum,
    reconstruct_trajectory,
    sample_task_params_jax,
    update_curriculum,
)


def _stack_pytrees(*trees):
    """Stack compatible pytrees along a new leading axis.

    Handles non-array leaves (e.g. activation functions in eqx.nn.MLP)
    by keeping the first tree's value. Array leaves are stacked with
    ``jnp.stack``.
    """
    all_flat = [jt.flatten(t) for t in trees]
    template_treedef = all_flat[0][1]
    stacked_leaves = []
    for leaves in zip(*[f[0] for f in all_flat]):
        if eqx.is_array(leaves[0]):
            stacked_leaves.append(jnp.stack(leaves))
        else:
            stacked_leaves.append(leaves[0])
    return jt.unflatten(template_treedef, stacked_leaves)


class PPOConfig(eqx.Module):
    """PPO training hyperparameters.

    Attributes:
        lr: Learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_eps: PPO clipping epsilon.
        n_epochs: Number of optimization epochs per update.
        n_minibatches: Number of minibatches per epoch.
        n_steps_per_update: Rollout length per update.
        total_timesteps: Total environment steps to train.
        ent_coef: Entropy bonus coefficient.
        vf_coef: Value function loss coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        hidden_dim: Hidden layer width for actor and critic MLPs.
        hidden_layers: Number of hidden layers.
    """

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    n_epochs: int = 10
    n_minibatches: int = 4
    n_steps_per_update: int = 256
    total_timesteps: int = 2_000_000
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    hidden_layers: int = 2


class Rollout(NamedTuple):
    """Collected rollout data with shape ``(n_steps, n_envs, ...)``.

    Attributes:
        obs: Observations.
        action_logits: Raw Beta draws in logit coordinates, before execution clipping.
        log_probs: Log densities of the draws in action coordinates.
        values: Value estimates.
        rewards: Rewards received.
        dones: Episode termination flags.
        latent_noise: Fixed collection noise, shared by environments at each
            timestep, or None when LATTICE is disabled.
    """

    obs: Float[Array, "T N obs_dim"]
    action_logits: Float[Array, "T N action_dim"]
    log_probs: Float[Array, "T N"]
    values: Float[Array, "T N"]
    rewards: Float[Array, "T N"]
    dones: Bool[Array, "T N"]
    latent_noise: Float[Array, "T latent_dim"] | None = None


class _EpisodeCarry(NamedTuple):
    returns: Float[Array, " N"]
    stages: Int[Array, " N"]


class _EpisodeTotals(NamedTuple):
    return_sum: Float[Array, ""]
    completed: Int[Array, ""]
    eligible: Int[Array, ""]
    successes: Int[Array, ""]
    stage_eligible: Int[Array, ""]
    stage_successes: Int[Array, ""]


class _PPOTrainCarry(NamedTuple):
    key: PRNGKeyArray
    dynamic_policy: ActorCritic
    opt_state: optax.OptState
    states: RLEnvState
    episodes: _EpisodeCarry
    obs_norm: ObsNormState
    lattice: LatticeNoiseState
    curriculum: CurriculumState


class _PPOTrainHistory(NamedTuple):
    per_body_mean_return: Float[Array, " B"]
    per_body_success_rate: Float[Array, " B"]
    per_body_completed_episodes: Int[Array, " B"]
    per_body_eligible_episodes: Int[Array, " B"]
    curriculum_stage: Int[Array, " B"]
    distance_weight: Float[Array, ""]


def compute_gae_scan(
    rewards: Float[Array, "T N"],
    values: Float[Array, "T N"],
    dones: Float[Array, "T N"],
    last_values: Float[Array, " N"],
    gamma: float,
    gae_lambda: float,
) -> tuple[Float[Array, "T N"], Float[Array, "T N"]]:
    """GAE via reversed ``jax.lax.scan`` — fully JIT-compatible.

    Args:
        rewards: Per-step rewards, shape ``(T, N)``.
        values: Per-step value estimates, shape ``(T, N)``.
        dones: Per-step done flags, shape ``(T, N)``.
        last_values: Bootstrapped values for final step, shape ``(N,)``.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        Tuple of (advantages, returns), each shape ``(T, N)``.
    """
    T = rewards.shape[0]
    values_ext = jnp.concatenate([values, last_values[None]], axis=0)

    def scan_fn(gae, t_rev):
        t = T - 1 - t_rev
        next_value = jnp.where(dones[t], 0.0, values_ext[t + 1])
        next_gae = jnp.where(dones[t], 0.0, gae)
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * next_gae
        return gae, gae

    _, advantages_rev = jax.lax.scan(scan_fn, jnp.zeros_like(last_values), jnp.arange(T))
    advantages = jnp.flip(advantages_rev, axis=0)
    returns = advantages + values
    return advantages, returns


def _ppo_loss(
    policy: ActorCritic,
    obs: Float[Array, "B obs_dim"],
    action_logits: Float[Array, "B action_dim"],
    old_logp: Float[Array, " B"],
    advantages: Float[Array, " B"],
    returns: Float[Array, " B"],
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    latent_noise: Float[Array, "B latent_dim"] | None = None,
) -> tuple[Float[Array, ""], tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]]:
    """PPO clipped loss function.

    Args:
        policy: Actor-critic policy.
        obs: Minibatch observations.
        action_logits: Minibatch Beta draws in logit coordinates.
        old_logp: Old log probabilities.
        advantages: Advantages.
        returns: Target returns.
        clip_eps: Clipping epsilon.
        vf_coef: Value loss coefficient.
        ent_coef: Entropy bonus coefficient.
        latent_noise: The fixed noise sampled during collection for each row.
            Its dynamics are policy-independent, so only the conditional Beta
            density enters the likelihood ratio.

    Returns:
        Tuple of (total_loss, (policy_loss, value_loss, entropy)).
    """
    dist, value = policy.dist_and_value(obs, latent_noise)
    logp = dist.joint_log_prob_from_logits(action_logits)
    ratio = jnp.exp(logp - old_logp)
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = jnp.mean((returns - value) ** 2)
    entropy = jnp.mean(jnp.sum(dist.entropy(), axis=-1))
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return loss, (policy_loss, value_loss, entropy)


@eqx.filter_jit
def collect_rollouts_batched(
    batched_plant: AbstractPlant,
    env_config: RLEnvConfig,
    batched_policy: ActorCritic,
    key: PRNGKeyArray,
    task_types: Array,
) -> dict:
    """Collect R rollouts per body for B bodies simultaneously.

    Uses scan-based rollout collection vmapped over bodies and rollouts.

    Args:
        batched_plant: MJXPlant with leading ``(B,)`` on array leaves.
        env_config: Shared environment configuration.
        batched_policy: ActorCritic with leading ``(B,)`` on array leaves.
        key: PRNG key.
        task_types: Task type per rollout, shape ``(R,)``.

    Returns:
        Dict with arrays of shape ``(B, R, T, ...)``.
        Keys: ``timestamps``, ``task_target``, ``joint_angles``,
        ``joint_velocities``, ``muscle_activations``, ``effector_pos``.
    """
    n_bodies = jt.leaves(batched_plant)[0].shape[0]
    n_rollouts = task_types.shape[0]
    n_steps = env_config.n_steps

    all_keys = jax.random.split(key, n_bodies * n_rollouts)
    all_keys = all_keys.reshape(n_bodies, n_rollouts, 2)

    def single_rollout(plant, policy, key, task_type):
        key, task_key, reset_key = jax.random.split(key, 3)
        seg_lens = getattr(plant, "segment_lengths", None)
        use_fk = seg_lens is not None
        if seg_lens is None:
            seg_lens = jnp.zeros((env_config.n_joints,), dtype=jnp.float32)
        task = sample_task_params_jax(
            task_key,
            task_type,
            env_config.n_steps,
            env_config.dt,
            segment_lengths=seg_lens,
            use_fk=use_fk,
            max_target_distance=0.0,
            use_curriculum=False,
            single_task=True,
        )
        state = rl_env_reset(plant, env_config, task, reset_key)

        sk0 = state.plant_state.skeleton
        eff0 = plant.skeleton.effector(sk0)

        def step_fn(carry, _):
            state, key = carry
            key, act_key = jax.random.split(key)
            obs = rl_env_get_obs(plant, env_config, state)
            action, _, _ = policy.sample_action(obs, act_key)
            next_state, _, _, _ = rl_env_step(plant, env_config, state, action)
            sk = next_state.plant_state.skeleton
            eff = plant.skeleton.effector(sk)
            return (next_state, key), (
                sk.qpos,
                sk.qvel,
                next_state.muscle_activations,
                eff.pos,
            )

        (_, _), (qpos_rest, qvel_rest, acts_rest, eff_rest) = jax.lax.scan(
            step_fn,
            (state, key),
            None,
            length=n_steps - 1,
        )

        qpos = jnp.concatenate([sk0.qpos[None], qpos_rest])
        qvel = jnp.concatenate([sk0.qvel[None], qvel_rest])
        acts = jnp.concatenate([state.muscle_activations[None], acts_rest])
        eff_pos = jnp.concatenate([eff0.pos[None], eff_rest])

        target_pos, _ = reconstruct_trajectory(task)
        timestamps = jnp.arange(n_steps) * env_config.dt
        return {
            "timestamps": timestamps,
            "task_target": target_pos,
            "joint_angles": qpos,
            "joint_velocities": qvel,
            "muscle_activations": acts,
            "effector_pos": eff_pos,
        }

    def body_rollouts(plant, policy, keys):
        return eqx.filter_vmap(
            lambda k, tt: single_rollout(plant, policy, k, tt),
        )(keys, task_types)

    return eqx.filter_vmap(body_rollouts)(
        batched_plant,
        batched_policy,
        all_keys,
    )


# ---------------------------------------------------------------------------
# Extended training: obs norm, LATTICE noise, curriculum, reward annealing
# Bug: 2055433 -- Composes all training enhancements into a single loop.
# ---------------------------------------------------------------------------


@dataclass
class TrainingEnhancements:
    """Configuration for optional training enhancements.

    Flags are used to gate enhancements dynamically inside JIT-compiled
    functions, enabling a single compiled program across combinations.

    Attributes:
        obs_norm: Enable running observation normalization (Welford).
        lattice_noise: Enable LATTICE temporally-correlated latent exploration.
        lattice_resample_interval: Steps between LATTICE noise resamples.
        curriculum: Enable progressive target distance curriculum.
        curriculum_arm_reach: Total arm reach for curriculum distance scaling.
        reward_annealing: Linearly anneal the additive distance penalty to zero.
        annealing_start_fraction: Keep full weight through this fraction of
            training, measured by the one-based update number divided by the
            total updates. Must be in [0, 1] when enabled. At 1 the weight stays
            constant. Otherwise the final update uses zero, even in a one-update run.
    """

    obs_norm: bool = False
    lattice_noise: bool = False
    lattice_resample_interval: int = 8
    curriculum: bool = False
    curriculum_arm_reach: float = 0.5
    reward_annealing: bool = False
    annealing_start_fraction: float = 0.7


def _collect_rollout_extended(
    plant: AbstractPlant,
    cfg: RLEnvConfig,
    policy: ActorCritic,
    states: RLEnvState,
    episodes: _EpisodeCarry,
    key: PRNGKeyArray,
    n_steps: int,
    n_envs: int,
    obs_norm_state: ObsNormState,
    lattice_state: LatticeNoiseState,
    curriculum_state: CurriculumState,
    lattice_resample_interval: int,
    curriculum_arm_reach: float,
    use_obs_norm: Bool[Array, ""],
    use_lattice: bool,
    use_curriculum: Bool[Array, ""],
    distance_weight: Float[Array, ""],
) -> tuple[
    RLEnvState,
    Rollout,
    Float[Array, " N"],
    LatticeNoiseState,
    _EpisodeCarry,
    _EpisodeTotals,
]:
    """Collect a batched rollout with optional enhancements.

    Carries unfinished returns and the stage of each episode into the next
    window. Only episodes completed in this window contribute to its totals.

    Args:
        plant: The plant model.
        cfg: Environment configuration.
        policy: Actor-critic policy.
        states: Batched environment states.
        episodes: Unfinished episode returns and starting curriculum stages.
        key: PRNG key.
        n_steps: Number of steps to collect.
        n_envs: Number of parallel environments.
        obs_norm_state: Running obs statistics for normalization.
        lattice_state: LATTICE noise state.
        curriculum_state: Curriculum state for distance scaling.
        lattice_resample_interval: Steps between noise resamples.
        curriculum_arm_reach: Total arm reach for curriculum distance scaling.
        use_obs_norm: Whether to normalize observations.
        use_lattice: Whether to apply LATTICE noise.
        use_curriculum: Whether to clamp reach target distance.
        distance_weight: Shared distance penalty weight for this rollout.

    Returns:
        Tuple of (final_states, rollout, last_values,
        updated_lattice_state, unfinished_episodes, episode_totals).
    """
    v_get_obs = jax.vmap(rl_env_get_obs, in_axes=(None, None, 0))
    v_step = jax.vmap(
        lambda state, action: rl_env_step_with_outcome(
            plant, cfg, state, action, distance_weight=distance_weight,
        ),
    )
    use_obs_norm = jnp.asarray(use_obs_norm)
    use_curriculum = jnp.asarray(use_curriculum)

    max_target_distance = curriculum_state.max_target_fraction * curriculum_arm_reach
    v_auto_reset = jax.vmap(
        lambda state, completed, key: auto_reset(
            plant, cfg, state, completed, key,
            max_target_distance=max_target_distance, use_curriculum=use_curriculum,
        ),
    )

    def scan_step(carry, _):
        states, key, lat_state, episodes = carry

        if use_lattice:
            key, act_key, reset_key, noise_key = jax.random.split(key, 4)
        else:
            key, act_key, reset_key = jax.random.split(key, 3)

        obs = v_get_obs(plant, cfg, states)
        obs_normed = normalize_obs(obs_norm_state, obs)
        obs = jnp.where(use_obs_norm, obs_normed, obs)

        act_keys = jax.random.split(act_key, n_envs)
        latent_noise = lat_state.noise if use_lattice else None
        action_logits, log_probs, values = jax.vmap(
            lambda observation, act_key: policy._draw_action_logits(
                observation, act_key, latent_noise,
            ),
        )(obs, act_keys)

        env_actions = _clip_action(_beta_sample_from_logits(action_logits))
        states, _, rewards, outcome = v_step(states, env_actions)
        completed = outcome.completed
        returns = episodes.returns + rewards
        eligible = completed & outcome.success_eligible
        current_stage = episodes.stages == curriculum_state.stage
        episodes = _EpisodeCarry(
            returns=jnp.where(completed, 0.0, returns),
            stages=jnp.where(completed, curriculum_state.stage, episodes.stages),
        )

        reset_keys = jax.random.split(reset_key, n_envs)
        states = v_auto_reset(states, completed, reset_keys)

        if use_lattice:
            lat_state = maybe_resample_noise(lat_state, noise_key, lattice_resample_interval)

        rollout_step = Rollout(obs, action_logits, log_probs, values, rewards, completed, latent_noise)
        episode_totals = _EpisodeTotals(
            return_sum=jnp.sum(jnp.where(completed, returns, 0.0)),
            completed=jnp.sum(completed),
            eligible=jnp.sum(eligible),
            successes=jnp.sum(outcome.success),
            stage_eligible=jnp.sum(eligible & current_stage),
            stage_successes=jnp.sum(outcome.success & current_stage),
        )
        return (states, key, lat_state, episodes), (rollout_step, episode_totals)

    (states, key, lattice_state, episodes), (rollout, episode_totals) = jax.lax.scan(
        scan_step, (states, key, lattice_state, episodes), None, length=n_steps,
    )
    episode_totals = jt.map(jnp.sum, episode_totals)

    # Bootstrap last values
    final_obs = v_get_obs(plant, cfg, states)
    final_obs_normed = normalize_obs(obs_norm_state, final_obs)
    final_obs = jnp.where(use_obs_norm, final_obs_normed, final_obs)
    _, _, last_values = jax.vmap(policy.sample_action)(
        final_obs,
        jax.random.split(key, n_envs),
    )

    return states, rollout, last_values, lattice_state, episodes, episode_totals


@eqx.filter_jit
def _batched_collect_rollouts_extended(
    batched_plant: AbstractPlant,
    env_config: RLEnvConfig,
    policy: ActorCritic,
    states: RLEnvState,
    episodes: _EpisodeCarry,
    key: PRNGKeyArray,
    obs_norm_state: ObsNormState,
    lattice_state: LatticeNoiseState,
    curriculum_state: CurriculumState,
    n_steps: int,
    n_envs: int,
    lattice_resample_interval: int,
    curriculum_arm_reach: float,
    use_obs_norm: Bool[Array, ""],
    use_lattice: bool,
    use_curriculum: Bool[Array, ""],
    distance_weight: Float[Array, ""],
) -> tuple[
    RLEnvState,
    Rollout,
    Float[Array, " B N"],
    LatticeNoiseState,
    _EpisodeCarry,
    _EpisodeTotals,
]:
    """Vectorized rollout collection across bodies."""
    n_bodies = jt.leaves(batched_plant)[0].shape[0]
    keys = jax.random.split(key, n_bodies)
    # Bug: a39e7d0 — capture broadcast args via closure to avoid PjitFunction leaf
    # bug with explicit in_axes in JAX 0.9 + equinox 0.13.4.
    return eqx.filter_vmap(
        lambda pl, pol, st, ep, k, on_st, lat_st, cur_st: _collect_rollout_extended(
            pl,
            env_config,
            pol,
            st,
            ep,
            k,
            n_steps,
            n_envs,
            on_st,
            lat_st,
            cur_st,
            lattice_resample_interval,
            curriculum_arm_reach,
            use_obs_norm,
            use_lattice,
            use_curriculum,
            distance_weight,
        ),
    )(batched_plant, policy, states, episodes, keys, obs_norm_state, lattice_state, curriculum_state)


def _update_one(
    policy: ActorCritic,
    opt_state: optax.OptState,
    rollout: Rollout,
    last_values: Float[Array, " N"],
    key: PRNGKeyArray,
    optimizer: optax.GradientTransformation,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    n_minibatches: int,
    n_train_iters: int,
) -> tuple[ActorCritic, optax.OptState]:
    """Single-body PPO update step."""
    advantages, returns = compute_gae_scan(
        rollout.rewards,
        rollout.values,
        rollout.dones,
        last_values,
        gamma,
        gae_lambda,
    )
    flat_obs = rollout.obs.reshape(-1, rollout.obs.shape[-1])
    flat_action_logits = rollout.action_logits.reshape(-1, rollout.action_logits.shape[-1])
    flat_logp = rollout.log_probs.reshape(-1)
    flat_adv = advantages.reshape(-1)
    flat_ret = returns.reshape(-1)
    flat_adv = (flat_adv - jnp.mean(flat_adv)) / (jnp.std(flat_adv) + 1e-8)

    n_envs = rollout.obs.shape[1]
    batch_size = flat_obs.shape[0]
    minibatch_size = batch_size // n_minibatches

    dynamic_policy, static_policy = eqx.partition(policy, eqx.is_array)

    def train_step(i, carry):
        dynamic_policy, opt_state, key = carry
        policy = eqx.combine(dynamic_policy, static_policy)
        perm_key = jax.random.fold_in(key, i // n_minibatches)
        perm = jax.random.permutation(perm_key, batch_size)
        start = (i % n_minibatches) * minibatch_size
        mb_idx = jax.lax.dynamic_slice(perm, (start,), (minibatch_size,))

        # Flat samples are time-major, with one noise vector shared per timestep.
        latent_noise = (
            None if rollout.latent_noise is None else rollout.latent_noise[mb_idx // n_envs]
        )
        (_, _), grads = eqx.filter_value_and_grad(
            _ppo_loss,
            has_aux=True,
        )(
            policy,
            flat_obs[mb_idx],
            flat_action_logits[mb_idx],
            flat_logp[mb_idx],
            flat_adv[mb_idx],
            flat_ret[mb_idx],
            clip_eps,
            vf_coef,
            ent_coef,
            latent_noise,
        )

        updates, opt_state = optimizer.update(grads, opt_state, policy)
        policy = eqx.apply_updates(policy, updates)
        dynamic_policy, _ = eqx.partition(policy, eqx.is_array)
        return dynamic_policy, opt_state, key

    dynamic_policy, opt_state, _ = jax.lax.fori_loop(
        0,
        n_train_iters,
        train_step,
        (dynamic_policy, opt_state, key),
    )
    policy = eqx.combine(dynamic_policy, static_policy)
    return policy, opt_state


@eqx.filter_jit
def _batched_update(
    policy: ActorCritic,
    opt_state: optax.OptState,
    rollout: Rollout,
    last_values: Float[Array, " B N"],
    key: PRNGKeyArray,
    optimizer: optax.GradientTransformation,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    n_minibatches: int,
    n_train_iters: int,
) -> tuple[ActorCritic, optax.OptState]:
    """Vectorized PPO update over bodies."""
    n_bodies = rollout.obs.shape[0]
    keys = jax.random.split(key, n_bodies)
    # Bug: a39e7d0 — capture broadcast args via closure to avoid PjitFunction leaf
    # bug with explicit in_axes in JAX 0.9 + equinox 0.13.4.
    return eqx.filter_vmap(
        lambda pol, opt, roll, lv, k: _update_one(
            pol,
            opt,
            roll,
            lv,
            k,
            optimizer,
            gamma,
            gae_lambda,
            clip_eps,
            vf_coef,
            ent_coef,
            n_minibatches,
            n_train_iters,
        ),
    )(policy, opt_state, rollout, last_values, keys)


@eqx.filter_jit
def _update_obs_norm_batched(
    obs_norm_st: ObsNormState,
    obs_batch: Float[Array, "B S obs_dim"],
) -> ObsNormState:
    return eqx.filter_vmap(update_obs_norm)(obs_norm_st, obs_batch)


@eqx.filter_jit
def _update_curriculum_batched(
    curric_st: CurriculumState,
    success_rates: Float[Array, " B"],
) -> CurriculumState:
    return eqx.filter_vmap(update_curriculum)(curric_st, success_rates)


@eqx.filter_jit
def _run_ppo_updates(
    batched_plant: AbstractPlant,
    env_config: RLEnvConfig,
    carry: _PPOTrainCarry,
    static_policy: ActorCritic,
    optimizer: optax.GradientTransformation,
    total_updates: int,
    n_steps: int,
    n_envs: int,
    n_minibatches: int,
    n_train_iters: int,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    lattice_resample_interval: int,
    curriculum_arm_reach: float,
    reward_ramp_updates: Float[Array, ""],
    use_obs_norm: Bool[Array, ""],
    use_lattice: bool,
    use_curriculum: Bool[Array, ""],
) -> tuple[_PPOTrainCarry, _PPOTrainHistory]:
    """Run the PPO outer update loop as one compiled JAX scan.

    The host formats final metrics after this returns; rollout collection,
    updates, and enhancement-state transitions all stay inside the compiled
    loop.
    """
    anneal_reward = reward_ramp_updates > 0
    ramp_length = jnp.where(anneal_reward, reward_ramp_updates, 1.0)

    def scan_step(carry: _PPOTrainCarry, update_index: Int[Array, ""]):
        remaining_updates = total_updates - 1 - update_index
        distance_weight = jnp.where(
            anneal_reward, jnp.minimum(remaining_updates / ramp_length, 1.0), 1.0,
        )
        key, collect_key, update_key = jax.random.split(carry.key, 3)
        policy = eqx.combine(carry.dynamic_policy, static_policy)
        states, rollout, last_values, lattice, episodes, totals = (
            _batched_collect_rollouts_extended(
                batched_plant,
                env_config,
                policy,
                carry.states,
                carry.episodes,
                collect_key,
                carry.obs_norm,
                carry.lattice,
                carry.curriculum,
                n_steps,
                n_envs,
                lattice_resample_interval,
                curriculum_arm_reach,
                use_obs_norm,
                use_lattice,
                use_curriculum,
                distance_weight,
            )
        )

        policy, opt_state = _batched_update(
            policy,
            carry.opt_state,
            rollout,
            last_values,
            update_key,
            optimizer,
            gamma,
            gae_lambda,
            clip_eps,
            vf_coef,
            ent_coef,
            n_minibatches,
            n_train_iters,
        )

        n_bodies = rollout.obs.shape[0]
        obs_dim = rollout.obs.shape[-1]
        all_obs = rollout.obs.reshape(n_bodies, -1, obs_dim)
        updated_obs_norm = _update_obs_norm_batched(carry.obs_norm, all_obs)
        obs_norm = jt.map(
            lambda old, new: jnp.where(use_obs_norm, new, old),
            carry.obs_norm,
            updated_obs_norm,
        )

        updated_curriculum = _update_curriculum_batched(
            carry.curriculum,
            totals.stage_successes / jnp.maximum(totals.stage_eligible, 1),
        )
        has_curriculum_evidence = use_curriculum & (totals.stage_eligible > 0)
        curriculum = jt.map(
            lambda old, new: jnp.where(has_curriculum_evidence, new, old),
            carry.curriculum,
            updated_curriculum,
        )

        next_carry = _PPOTrainCarry(
            key=key,
            dynamic_policy=eqx.filter(policy, eqx.is_array),
            opt_state=opt_state,
            states=states,
            episodes=episodes,
            obs_norm=obs_norm,
            lattice=lattice,
            curriculum=curriculum,
        )
        history = _PPOTrainHistory(
            per_body_mean_return=totals.return_sum / jnp.maximum(totals.completed, 1),
            per_body_success_rate=totals.successes / jnp.maximum(totals.eligible, 1),
            per_body_completed_episodes=totals.completed,
            per_body_eligible_episodes=totals.eligible,
            curriculum_stage=curriculum.stage,
            distance_weight=distance_weight,
        )
        return next_carry, history

    return jax.lax.scan(scan_step, carry, jnp.arange(total_updates))


def train_ppo_batched(
    batched_plant: AbstractPlant,
    env_config: RLEnvConfig,
    ppo_config: PPOConfig,
    key: PRNGKeyArray,
    n_envs: int = 512,
    enhancements: TrainingEnhancements | None = None,
) -> tuple[ActorCritic, dict[str, object]]:
    """Train independent PPO policies for B bodies.

    The batch axis represents independent bodies/policies. Optional
    observation normalization, LATTICE exploration noise, curriculum learning,
    and distance penalty annealing run inside the same
    JIT-compiled rollout/update path.

    Args:
        batched_plant: MJXPlant with leading ``(B,)`` dim on array leaves.
        env_config: RLEnvConfig (shared across all bodies).
        ppo_config: PPO hyperparameters.
        key: PRNG key.
        n_envs: Number of parallel environments per body.
        enhancements: Optional enhancement config. If ``None``, defaults to
            ``TrainingEnhancements()`` with all enhancements disabled.

    Returns:
        Tuple of (batched ActorCritic with ``(B,)`` leading dim,
        metrics dict with per-body episode statistics for each update and
        enhancement-specific data). Mean return includes all completed episodes.
        Success rate includes only completed episodes with a success criterion.
        Each metric is zero when its corresponding episode count is zero.
    """
    if enhancements is None:
        enhancements = TrainingEnhancements()

    n_bodies = jt.leaves(batched_plant)[0].shape[0]
    obs_dim = env_config.n_joints * 2 + env_config.n_muscles + 2 + 2 + 2 + 1
    action_dim = env_config.n_muscles

    # Extract config as concrete Python values for JIT tracing
    n_steps = int(ppo_config.n_steps_per_update)
    total_timesteps = int(ppo_config.total_timesteps)
    n_epochs = int(ppo_config.n_epochs)
    n_minibatches = int(ppo_config.n_minibatches)
    gamma = float(ppo_config.gamma)
    gae_lambda = float(ppo_config.gae_lambda)
    clip_eps = float(ppo_config.clip_eps)
    vf_coef = float(ppo_config.vf_coef)
    ent_coef = float(ppo_config.ent_coef)
    timesteps_per_update = n_steps * n_envs
    n_train_iters = n_epochs * n_minibatches

    total_updates = (total_timesteps + timesteps_per_update - 1) // timesteps_per_update
    reward_ramp_updates = 0.0
    if enhancements.reward_annealing:
        start = enhancements.annealing_start_fraction
        if not 0.0 <= start <= 1.0:
            raise ValueError("annealing_start_fraction must be in [0, 1]")
        reward_ramp_updates = total_updates * (1.0 - start)

    # B independent policies
    key, init_key, env_key = jax.random.split(key, 3)
    init_keys = jax.random.split(init_key, n_bodies)
    policies = [
        ActorCritic(
            obs_dim,
            action_dim,
            int(ppo_config.hidden_dim),
            int(ppo_config.hidden_layers),
            key=k,
        )
        for k in init_keys
    ]
    batched_policy = _stack_pytrees(*policies)

    # Optimizer + B opt_states
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(ppo_config.max_grad_norm)),
        optax.adam(float(ppo_config.lr)),
    )
    opt_states = [optimizer.init(eqx.filter(p, eqx.is_array)) for p in policies]
    batched_opt_state = _stack_pytrees(*opt_states)

    # --- Initialize enhancement states ---
    # Bug: 2055433 -- per-body enhancement state initialization.
    batched_obs_norm = _stack_pytrees(*[init_obs_norm(obs_dim) for _ in range(n_bodies)])

    latent_dim = policies[0].actor.layers[-1].in_features
    if enhancements.lattice_noise:
        key, lattice_key = jax.random.split(key)
        lattice_keys = jax.random.split(lattice_key, n_bodies)
        batched_lattice = eqx.filter_vmap(
            lambda k: init_lattice_noise(latent_dim, k),
        )(lattice_keys)
    else:
        zero_noise = jnp.zeros((latent_dim,))
        zero_state = LatticeNoiseState(
            noise=zero_noise,
            steps_since_resample=jnp.array(0, dtype=jnp.int32),
        )
        batched_lattice = _stack_pytrees(*[zero_state for _ in range(n_bodies)])

    batched_curriculum = _stack_pytrees(*[init_curriculum() for _ in range(n_bodies)])

    use_obs_norm = jnp.asarray(enhancements.obs_norm, dtype=bool)
    use_lattice = enhancements.lattice_noise
    use_curriculum = jnp.asarray(enhancements.curriculum, dtype=bool)

    def init_body(plant, key, max_target_fraction):
        return jax.vmap(
            lambda k: rl_env_reset_sampled(
                plant, env_config, k,
                max_target_distance=max_target_fraction * enhancements.curriculum_arm_reach,
                use_curriculum=use_curriculum,
            ),
        )(jax.random.split(key, n_envs))

    batched_states = eqx.filter_vmap(init_body)(
        batched_plant, jax.random.split(env_key, n_bodies), batched_curriculum.max_target_fraction,
    )
    episodes = _EpisodeCarry(
        returns=jnp.zeros((n_bodies, n_envs)),
        stages=jnp.broadcast_to(batched_curriculum.stage[:, None], (n_bodies, n_envs)),
    )

    dynamic_policy, static_policy = eqx.partition(batched_policy, eqx.is_array)
    carry = _PPOTrainCarry(
        key=key,
        dynamic_policy=dynamic_policy,
        opt_state=batched_opt_state,
        states=batched_states,
        episodes=episodes,
        obs_norm=batched_obs_norm,
        lattice=batched_lattice,
        curriculum=batched_curriculum,
    )
    carry, history = _run_ppo_updates(
        batched_plant,
        env_config,
        carry,
        static_policy,
        optimizer,
        total_updates,
        n_steps,
        n_envs,
        n_minibatches,
        n_train_iters,
        gamma,
        gae_lambda,
        clip_eps,
        vf_coef,
        ent_coef,
        enhancements.lattice_resample_interval,
        enhancements.curriculum_arm_reach,
        jnp.asarray(reward_ramp_updates),
        use_obs_norm,
        use_lattice,
        use_curriculum,
    )
    batched_policy = eqx.combine(carry.dynamic_policy, static_policy)
    batched_obs_norm = carry.obs_norm

    metrics: dict[str, object] = {
        "updates": total_updates,
        "timesteps": total_updates * timesteps_per_update,
        "per_body_mean_return": list(history.per_body_mean_return),
        "per_body_success_rate": list(history.per_body_success_rate),
        "per_body_completed_episodes": list(history.per_body_completed_episodes),
        "per_body_eligible_episodes": list(history.per_body_eligible_episodes),
    }
    if enhancements.curriculum:
        metrics["curriculum_stages"] = list(history.curriculum_stage)
    if enhancements.reward_annealing:
        metrics["distance_weight_schedule"] = list(history.distance_weight)
    if enhancements.obs_norm:
        metrics["obs_norm_mean"] = batched_obs_norm.mean

    return batched_policy, metrics
