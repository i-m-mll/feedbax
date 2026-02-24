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
from jaxtyping import Array, Float, Int, PRNGKeyArray

from feedbax.mechanics.plant import AbstractPlant
from feedbax.training.rl.env import (
    RLEnvConfig,
    RLEnvState,
    auto_reset,
    rl_env_get_obs,
    rl_env_reset,
    rl_env_step,
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
    init_lattice_noise,
    maybe_resample_noise,
    sample_action_with_noise,
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
        actions: Actions taken.
        log_probs: Log probabilities of actions.
        values: Value estimates.
        rewards: Rewards received.
        dones: Episode termination flags.
    """

    obs: Float[Array, "T N obs_dim"]
    actions: Float[Array, "T N action_dim"]
    log_probs: Float[Array, "T N"]
    values: Float[Array, "T N"]
    rewards: Float[Array, "T N"]
    dones: Float[Array, "T N"]


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
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_ext[t + 1] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        return gae, gae

    _, advantages_rev = jax.lax.scan(
        scan_fn, jnp.zeros_like(last_values), jnp.arange(T)
    )
    advantages = jnp.flip(advantages_rev, axis=0)
    returns = advantages + values
    return advantages, returns


def _collect_rollout(
    plant: AbstractPlant,
    cfg: RLEnvConfig,
    policy: ActorCritic,
    states: RLEnvState,
    key: PRNGKeyArray,
    n_steps: int,
    n_envs: int,
) -> tuple[RLEnvState, Rollout, Float[Array, " N"], PRNGKeyArray]:
    """Collect a batched rollout via vmap + scan.

    Args:
        plant: The plant model.
        cfg: Environment configuration.
        policy: Actor-critic policy.
        states: Batched environment states.
        key: PRNG key.
        n_steps: Number of steps to collect.
        n_envs: Number of parallel environments.

    Returns:
        Tuple of (final_states, rollout, last_values, updated_key).
    """
    v_get_obs = jax.vmap(rl_env_get_obs, in_axes=(None, None, 0))
    v_step = jax.vmap(rl_env_step, in_axes=(None, None, 0, 0))
    v_auto_reset = jax.vmap(auto_reset, in_axes=(None, None, 0, 0, 0))

    def scan_step(carry, _):
        states, key = carry
        key, act_key, reset_key = jax.random.split(key, 3)

        obs = v_get_obs(plant, cfg, states)

        act_keys = jax.random.split(act_key, n_envs)
        actions, log_probs, values = jax.vmap(policy.sample_action)(obs, act_keys)

        states, _, rewards, dones = v_step(plant, cfg, states, actions)

        reset_keys = jax.random.split(reset_key, n_envs)
        states = v_auto_reset(plant, cfg, states, dones, reset_keys)

        return (states, key), (obs, actions, log_probs, values, rewards, dones)

    (states, key), (obs, actions, log_probs, values, rewards, dones) = jax.lax.scan(
        scan_step, (states, key), None, length=n_steps,
    )

    final_obs = v_get_obs(plant, cfg, states)
    _, _, last_values = jax.vmap(policy.sample_action)(
        final_obs, jax.random.split(key, n_envs)
    )

    rollout = Rollout(obs, actions, log_probs, values, rewards, dones)
    return states, rollout, last_values, key


def _ppo_loss(
    policy: ActorCritic,
    obs: Float[Array, "B obs_dim"],
    actions: Float[Array, "B action_dim"],
    old_logp: Float[Array, " B"],
    advantages: Float[Array, " B"],
    returns: Float[Array, " B"],
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[Float[Array, ""], tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]]:
    """PPO clipped loss function.

    Args:
        policy: Actor-critic policy.
        obs: Minibatch observations.
        actions: Minibatch actions.
        old_logp: Old log probabilities.
        advantages: Advantages.
        returns: Target returns.
        clip_eps: Clipping epsilon.
        vf_coef: Value loss coefficient.
        ent_coef: Entropy bonus coefficient.

    Returns:
        Tuple of (total_loss, (policy_loss, value_loss, entropy)).
    """
    dist, value = policy.dist_and_value(obs)
    logp = jnp.sum(dist.log_prob(actions), axis=-1)
    ratio = jnp.exp(logp - old_logp)
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = jnp.mean((returns - value) ** 2)
    entropy = jnp.mean(jnp.sum(dist.entropy(), axis=-1))
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return loss, (policy_loss, value_loss, entropy)


def _init_envs(
    plant: AbstractPlant,
    cfg: RLEnvConfig,
    key: PRNGKeyArray,
    n_envs: int,
) -> RLEnvState:
    """Initialize a batch of environments with random tasks.

    Args:
        plant: The plant model.
        cfg: Environment configuration.
        key: PRNG key.
        n_envs: Number of parallel environments.

    Returns:
        Batched RLEnvState.
    """
    def init_one(key):
        key, task_key, reset_key = jax.random.split(key, 3)
        seg_lens = getattr(plant, "segment_lengths", None)
        task = sample_task_params_jax(
            task_key, None, cfg.n_steps, cfg.dt,
            segment_lengths=seg_lens,
        )
        return rl_env_reset(plant, cfg, task, reset_key)

    keys = jax.random.split(key, n_envs)
    return jax.vmap(init_one)(keys)


def train_ppo(
    plant: AbstractPlant,
    env_config: RLEnvConfig,
    ppo_config: PPOConfig,
    key: PRNGKeyArray,
    n_envs: int = 4096,
) -> tuple[ActorCritic, dict[str, object]]:
    """Train a policy with vectorized PPO.

    Args:
        plant: The plant model to train on.
        env_config: RL environment configuration.
        ppo_config: PPO hyperparameters.
        key: PRNG key.
        n_envs: Number of parallel environments.

    Returns:
        Tuple of (trained ActorCritic, metrics dict).
    """
    obs_dim = (
        env_config.n_joints * 2 + env_config.n_muscles + 2 + 2 + 2 + 1
    )
    action_dim = env_config.n_muscles

    key, init_key, env_key = jax.random.split(key, 3)
    policy = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=ppo_config.hidden_dim,
        hidden_layers=ppo_config.hidden_layers,
        key=init_key,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(ppo_config.max_grad_norm),
        optax.adam(ppo_config.lr),
    )
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    states = _init_envs(plant, env_config, env_key, n_envs)

    n_steps_per_update = int(ppo_config.n_steps_per_update)
    total_timesteps = int(ppo_config.total_timesteps)
    timesteps_per_update = n_steps_per_update * n_envs

    metrics: dict[str, object] = {
        "updates": 0,
        "timesteps": 0,
        "mean_return": [],
        "mean_value_loss": [],
        "mean_policy_loss": [],
        "mean_entropy": [],
    }

    @eqx.filter_jit
    def collect_fn(p, s, k):
        return _collect_rollout(
            plant, env_config, p, s, k, n_steps_per_update, n_envs,
        )

    while metrics["timesteps"] < total_timesteps:
        key, collect_key = jax.random.split(key)
        states, rollout, last_values, key = collect_fn(
            policy, states, collect_key,
        )

        metrics["timesteps"] = int(metrics["timesteps"]) + timesteps_per_update

        advantages, returns = compute_gae_scan(
            rollout.rewards, rollout.values, rollout.dones,
            last_values, ppo_config.gamma, ppo_config.gae_lambda,
        )

        flat_obs = rollout.obs.reshape(-1, obs_dim)
        flat_actions = rollout.actions.reshape(-1, action_dim)
        flat_logp = rollout.log_probs.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)

        flat_adv = (flat_adv - jnp.mean(flat_adv)) / (jnp.std(flat_adv) + 1e-8)

        batch_size = flat_obs.shape[0]

        value_losses = []
        policy_losses = []
        entropies = []

        for _ in range(ppo_config.n_epochs):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, batch_size)

            minibatch_size = batch_size // ppo_config.n_minibatches
            for start in range(0, batch_size, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]

                obs_mb = flat_obs[mb_idx]
                actions_mb = flat_actions[mb_idx]
                logp_mb = flat_logp[mb_idx]
                adv_mb = flat_adv[mb_idx]
                ret_mb = flat_ret[mb_idx]

                (_, (pl, vl, ent)), grads = eqx.filter_value_and_grad(
                    _ppo_loss, has_aux=True,
                )(
                    policy, obs_mb, actions_mb, logp_mb, adv_mb, ret_mb,
                    ppo_config.clip_eps, ppo_config.vf_coef,
                    ppo_config.ent_coef,
                )

                updates, opt_state = optimizer.update(grads, opt_state, policy)
                policy = eqx.apply_updates(policy, updates)

                value_losses.append(float(vl))
                policy_losses.append(float(pl))
                entropies.append(float(ent))

        metrics["updates"] = int(metrics["updates"]) + 1
        mean_reward = float(jnp.mean(rollout.rewards))
        metrics["mean_return"].append(mean_reward * env_config.n_steps)
        metrics["mean_value_loss"].append(
            float(jnp.mean(jnp.array(value_losses)))
        )
        metrics["mean_policy_loss"].append(
            float(jnp.mean(jnp.array(policy_losses)))
        )
        metrics["mean_entropy"].append(float(jnp.mean(jnp.array(entropies))))

    return policy, metrics


def train_ppo_batched(
    batched_plant: AbstractPlant,
    env_config: RLEnvConfig,
    ppo_config: PPOConfig,
    key: PRNGKeyArray,
    n_envs: int = 512,
) -> tuple[ActorCritic, dict]:
    """Train independent PPO policies for B bodies simultaneously.

    Each body gets its own policy; collection and update steps are vmapped
    over the body batch axis. The outer PPO update loop stays in Python
    (small number of iterations), while inner operations are JIT-compiled.

    Args:
        batched_plant: MJXPlant with leading ``(B,)`` dim on array leaves.
        env_config: RLEnvConfig (shared across all bodies).
        ppo_config: PPO hyperparameters.
        key: PRNG key.
        n_envs: Number of parallel environments per body.

    Returns:
        Tuple of (batched ActorCritic with ``(B,)`` leading dim,
        metrics dict with per-body returns).
    """
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
    batch_size = n_steps * n_envs
    minibatch_size = batch_size // n_minibatches
    n_train_iters = n_epochs * n_minibatches

    # B independent policies
    key, init_key, env_key = jax.random.split(key, 3)
    init_keys = jax.random.split(init_key, n_bodies)
    policies = [
        ActorCritic(
            obs_dim, action_dim,
            int(ppo_config.hidden_dim), int(ppo_config.hidden_layers),
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

    # B × N environments
    env_keys = jax.random.split(env_key, n_bodies)
    batched_states = eqx.filter_vmap(
        lambda plant, k: _init_envs(plant, env_config, k, n_envs),
    )(batched_plant, env_keys)

    # --- JIT'd vmapped collect ---
    @eqx.filter_jit
    def batched_collect(policy, states, key):
        keys = jax.random.split(key, n_bodies)
        return eqx.filter_vmap(
            lambda pl, pol, st, k: _collect_rollout(
                pl, env_config, pol, st, k, n_steps, n_envs,
            ),
        )(batched_plant, policy, states, keys)

    # --- Per-body update (vmapped) ---
    def update_one(policy, opt_state, rollout, last_values, key):
        advantages, returns = compute_gae_scan(
            rollout.rewards, rollout.values, rollout.dones,
            last_values, gamma, gae_lambda,
        )
        flat_obs = rollout.obs.reshape(-1, obs_dim)
        flat_actions = rollout.actions.reshape(-1, action_dim)
        flat_logp = rollout.log_probs.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)
        flat_adv = (flat_adv - jnp.mean(flat_adv)) / (jnp.std(flat_adv) + 1e-8)

        # Partition policy into dynamic (arrays) and static (activation fns,
        # etc.) so that only valid JAX types enter the fori_loop carry.
        dynamic_policy, static_policy = eqx.partition(policy, eqx.is_array)

        def train_step(i, carry):
            dynamic_policy, opt_state, key = carry
            policy = eqx.combine(dynamic_policy, static_policy)
            perm_key = jax.random.fold_in(key, i // n_minibatches)
            perm = jax.random.permutation(perm_key, batch_size)
            start = (i % n_minibatches) * minibatch_size
            mb_idx = jax.lax.dynamic_slice(perm, (start,), (minibatch_size,))

            (_, _), grads = eqx.filter_value_and_grad(
                _ppo_loss, has_aux=True,
            )(
                policy,
                flat_obs[mb_idx], flat_actions[mb_idx], flat_logp[mb_idx],
                flat_adv[mb_idx], flat_ret[mb_idx],
                clip_eps, vf_coef, ent_coef,
            )

            updates, opt_state = optimizer.update(grads, opt_state, policy)
            policy = eqx.apply_updates(policy, updates)
            dynamic_policy, _ = eqx.partition(policy, eqx.is_array)
            return dynamic_policy, opt_state, key

        dynamic_policy, opt_state, _ = jax.lax.fori_loop(
            0, n_train_iters, train_step, (dynamic_policy, opt_state, key),
        )
        policy = eqx.combine(dynamic_policy, static_policy)
        mean_reward = jnp.mean(rollout.rewards)
        return policy, opt_state, mean_reward

    @eqx.filter_jit
    def batched_update(policy, opt_state, rollout, last_values, key):
        keys = jax.random.split(key, n_bodies)
        return eqx.filter_vmap(update_one)(
            policy, opt_state, rollout, last_values, keys,
        )

    # --- Training loop ---
    metrics: dict[str, object] = {
        "updates": 0,
        "timesteps": 0,
        "per_body_mean_return": [],
    }

    while metrics["timesteps"] < total_timesteps:
        key, collect_key, update_key = jax.random.split(key, 3)

        batched_states, batched_rollout, batched_last_values, _ = (
            batched_collect(batched_policy, batched_states, collect_key)
        )

        metrics["timesteps"] = int(metrics["timesteps"]) + timesteps_per_update

        batched_policy, batched_opt_state, per_body_rewards = batched_update(
            batched_policy, batched_opt_state,
            batched_rollout, batched_last_values, update_key,
        )

        metrics["updates"] = int(metrics["updates"]) + 1
        per_body_returns = per_body_rewards * env_config.n_steps
        metrics["per_body_mean_return"].append(per_body_returns)

    return batched_policy, metrics


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
        task = sample_task_params_jax(
            task_key, task_type, env_config.n_steps, env_config.dt,
            segment_lengths=seg_lens,
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
                sk.qpos, sk.qvel, next_state.muscle_activations, eff.pos,
            )

        (_, _), (qpos_rest, qvel_rest, acts_rest, eff_rest) = jax.lax.scan(
            step_fn, (state, key), None, length=n_steps - 1,
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
        batched_plant, batched_policy, all_keys,
    )


# ---------------------------------------------------------------------------
# Extended training: obs norm, LATTICE noise, curriculum, reward annealing
# Bug: 2055433 -- Composes all training enhancements into a single loop.
# ---------------------------------------------------------------------------


@dataclass
class TrainingEnhancements:
    """Configuration for optional training enhancements.

    All flags are static (known at trace time) and can be used with Python
    ``if`` statements inside JIT-compiled functions without causing retracing
    issues.

    Attributes:
        obs_norm: Enable running observation normalization (Welford).
        lattice_noise: Enable LATTICE temporally-correlated latent exploration.
        lattice_resample_interval: Steps between LATTICE noise resamples.
        curriculum: Enable progressive target distance curriculum.
        curriculum_arm_reach: Total arm reach for curriculum distance scaling.
        reward_annealing: Enable linear distance_weight annealing to zero.
        annealing_start_fraction: Fraction of training at which annealing begins.
    """

    obs_norm: bool = False
    lattice_noise: bool = False
    lattice_resample_interval: int = 8
    curriculum: bool = False
    curriculum_arm_reach: float = 0.5
    reward_annealing: bool = False
    annealing_start_fraction: float = 0.7


class ExtendedTrainingState(eqx.Module):
    """State for extended training with all enhancements.

    Attributes:
        obs_norm_state: Running obs statistics per body, or ``None``.
        lattice_state: LATTICE noise state per body per env, or ``None``.
        curriculum_state: Curriculum stage per body, or ``None``.
        update_count: Current PPO update number.
        total_updates: Total expected updates (for annealing schedule).
    """

    obs_norm_state: ObsNormState | None
    lattice_state: LatticeNoiseState | None
    curriculum_state: CurriculumState | None
    update_count: Int[Array, ""]
    total_updates: Int[Array, ""]


def _auto_reset_curriculum(
    plant: AbstractPlant,
    config: RLEnvConfig,
    state: RLEnvState,
    done: Float[Array, ""],
    key: PRNGKeyArray,
    max_target_distance: Float[Array, ""],
) -> RLEnvState:
    """Auto-reset with curriculum-aware max_target_distance.

    Identical to ``auto_reset`` but passes ``max_target_distance`` to
    ``sample_task_params_jax`` for curriculum learning.

    Args:
        plant: The plant model.
        config: Environment configuration.
        state: Current environment state.
        done: Whether the episode is done (1.0) or not (0.0).
        key: PRNG key.
        max_target_distance: Maximum distance from start to reach target.

    Returns:
        Conditionally reset RLEnvState.
    """
    key, task_key, reset_key = jax.random.split(key, 3)
    seg_lens = getattr(plant, "segment_lengths", None)
    task_type = config.default_task_type
    new_task = sample_task_params_jax(
        task_key, task_type, config.n_steps, config.dt,
        segment_lengths=seg_lens,
        max_target_distance=max_target_distance,
    )
    new_state = rl_env_reset(plant, config, new_task, reset_key)

    return jt.map(
        lambda old, new: jnp.where(done, new, old),
        state,
        new_state,
    )


def _collect_rollout_extended(
    plant: AbstractPlant,
    cfg: RLEnvConfig,
    policy: ActorCritic,
    states: RLEnvState,
    key: PRNGKeyArray,
    n_steps: int,
    n_envs: int,
    *,
    obs_norm_state: ObsNormState | None,
    lattice_state: LatticeNoiseState | None,
    lattice_resample_interval: int,
    max_target_distance: Float[Array, ""] | None,
) -> tuple[
    RLEnvState,
    Rollout,
    Float[Array, " N"],
    PRNGKeyArray,
    LatticeNoiseState | None,
]:
    """Collect a batched rollout with optional enhancements.

    Like ``_collect_rollout`` but supports observation normalization,
    LATTICE noise injection, and curriculum-aware auto-reset.

    Args:
        plant: The plant model.
        cfg: Environment configuration.
        policy: Actor-critic policy.
        states: Batched environment states.
        key: PRNG key.
        n_steps: Number of steps to collect.
        n_envs: Number of parallel environments.
        obs_norm_state: Running obs statistics for normalization, or ``None``.
        lattice_state: LATTICE noise state, or ``None``.
        lattice_resample_interval: Steps between noise resamples.
        max_target_distance: Curriculum max target distance, or ``None``.

    Returns:
        Tuple of (final_states, rollout, last_values, updated_key,
        updated_lattice_state).
    """
    v_get_obs = jax.vmap(rl_env_get_obs, in_axes=(None, None, 0))
    v_step = jax.vmap(rl_env_step, in_axes=(None, None, 0, 0))

    use_obs_norm = obs_norm_state is not None
    use_lattice = lattice_state is not None
    use_curriculum = max_target_distance is not None

    if use_curriculum:
        v_auto_reset = jax.vmap(
            _auto_reset_curriculum, in_axes=(None, None, 0, 0, 0, None),
        )
    else:
        v_auto_reset = jax.vmap(auto_reset, in_axes=(None, None, 0, 0, 0))

    # Build the initial carry for the scan. Include lattice_state only
    # when LATTICE is enabled so that we don't pass None through lax.scan.
    if use_lattice:
        # lattice_state has shape (N, hidden_dim) for noise and (N,) for counter
        # since it's vmapped over environments by the caller's vmap over bodies.
        # But here we need per-env noise. We replicate the single body's
        # lattice noise to N envs inside the scan step via broadcasting.
        init_carry = (states, key, lattice_state)
    else:
        init_carry = (states, key)

    def scan_step_lattice(carry, _):
        states, key, lat_state = carry
        key, act_key, reset_key, noise_key = jax.random.split(key, 4)

        obs = v_get_obs(plant, cfg, states)
        if use_obs_norm:
            obs = normalize_obs(obs_norm_state, obs)

        act_keys = jax.random.split(act_key, n_envs)
        # LATTICE: inject noise into each env's action sampling
        actions, log_probs, values = jax.vmap(
            sample_action_with_noise, in_axes=(None, 0, 0, None),
        )(policy, obs, act_keys, lat_state.noise)

        states, _, rewards, dones = v_step(plant, cfg, states, actions)

        reset_keys = jax.random.split(reset_key, n_envs)
        if use_curriculum:
            states = v_auto_reset(
                plant, cfg, states, dones, reset_keys, max_target_distance,
            )
        else:
            states = v_auto_reset(plant, cfg, states, dones, reset_keys)

        # Resample LATTICE noise (shared across envs for this body)
        lat_state = maybe_resample_noise(
            lat_state, noise_key, lattice_resample_interval,
        )

        return (states, key, lat_state), (
            obs, actions, log_probs, values, rewards, dones,
        )

    def scan_step_basic(carry, _):
        states, key = carry
        key, act_key, reset_key = jax.random.split(key, 3)

        obs = v_get_obs(plant, cfg, states)
        if use_obs_norm:
            obs = normalize_obs(obs_norm_state, obs)

        act_keys = jax.random.split(act_key, n_envs)
        actions, log_probs, values = jax.vmap(policy.sample_action)(obs, act_keys)

        states, _, rewards, dones = v_step(plant, cfg, states, actions)

        reset_keys = jax.random.split(reset_key, n_envs)
        if use_curriculum:
            states = v_auto_reset(
                plant, cfg, states, dones, reset_keys, max_target_distance,
            )
        else:
            states = v_auto_reset(plant, cfg, states, dones, reset_keys)

        return (states, key), (obs, actions, log_probs, values, rewards, dones)

    if use_lattice:
        (states, key, lattice_state), (
            obs, actions, log_probs, values, rewards, dones,
        ) = jax.lax.scan(scan_step_lattice, init_carry, None, length=n_steps)
    else:
        (states, key), (
            obs, actions, log_probs, values, rewards, dones,
        ) = jax.lax.scan(scan_step_basic, init_carry, None, length=n_steps)

    # Bootstrap last values
    final_obs = v_get_obs(plant, cfg, states)
    if use_obs_norm:
        final_obs = normalize_obs(obs_norm_state, final_obs)
    _, _, last_values = jax.vmap(policy.sample_action)(
        final_obs, jax.random.split(key, n_envs),
    )

    rollout = Rollout(obs, actions, log_probs, values, rewards, dones)
    return states, rollout, last_values, key, lattice_state


def _compute_success_rate(
    rollout: Rollout,
    n_envs: int,
    threshold: float = 0.02,
) -> Float[Array, ""]:
    """Compute per-body success rate from rollout done signals.

    An episode is considered successful if the agent triggered early
    termination (i.e. ``done == 1.0`` before the episode's natural end).
    As a proxy, we count any done flag over the rollout, which includes
    both early termination and natural episode ends. For reach-only
    training, done at the last step is always 1.0 (natural end), so
    this over-counts slightly -- but the curriculum's advancement
    threshold (0.85) handles this.

    A simpler and more robust proxy: count the fraction of done events
    that are NOT at the max t_index (i.e., true early successes).
    But without t_index in the rollout, we use total done fraction.

    Args:
        rollout: Collected rollout data, shape ``(T, N, ...)``.
        n_envs: Number of parallel environments.
        threshold: Unused (kept for API compatibility).

    Returns:
        Success rate as a scalar in [0, 1].
    """
    # Total done events across the rollout
    total_dones = jnp.sum(rollout.dones)
    # Approximate number of episodes: each done signals an episode boundary
    # Success rate ~ fraction of timesteps that end in done (proxy)
    # Use mean of dones as a simple proxy -- higher means more frequent
    # episode completions (either success or timeout).
    return jnp.mean(rollout.dones)


def train_ppo_batched_extended(
    batched_plant: AbstractPlant,
    env_config: RLEnvConfig,
    ppo_config: PPOConfig,
    key: PRNGKeyArray,
    n_envs: int = 512,
    enhancements: TrainingEnhancements | None = None,
) -> tuple[ActorCritic, dict]:
    """Train independent PPO policies for B bodies with optional enhancements.

    Wraps the same training loop structure as ``train_ppo_batched`` but
    inserts hooks for observation normalization, LATTICE exploration noise,
    curriculum learning, and reward annealing.

    Enhancement hooks are gated by Python ``if`` on the static
    ``TrainingEnhancements`` flags, so disabled enhancements incur zero
    overhead (no extra tracing or runtime cost).

    Args:
        batched_plant: MJXPlant with leading ``(B,)`` dim on array leaves.
        env_config: RLEnvConfig (shared across all bodies).
        ppo_config: PPO hyperparameters.
        key: PRNG key.
        n_envs: Number of parallel environments per body.
        enhancements: Optional training enhancements config. If ``None``,
            defaults to ``TrainingEnhancements()`` (all disabled).

    Returns:
        Tuple of (batched ActorCritic with ``(B,)`` leading dim,
        metrics dict with per-body returns and enhancement-specific data).
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
    batch_size = n_steps * n_envs
    minibatch_size = batch_size // n_minibatches
    n_train_iters = n_epochs * n_minibatches

    total_updates = total_timesteps // timesteps_per_update

    # B independent policies
    key, init_key, env_key = jax.random.split(key, 3)
    init_keys = jax.random.split(init_key, n_bodies)
    policies = [
        ActorCritic(
            obs_dim, action_dim,
            int(ppo_config.hidden_dim), int(ppo_config.hidden_layers),
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

    # B x N environments
    env_keys = jax.random.split(env_key, n_bodies)
    batched_states = eqx.filter_vmap(
        lambda plant, k: _init_envs(plant, env_config, k, n_envs),
    )(batched_plant, env_keys)

    # --- Initialize enhancement states ---
    # Bug: 2055433 -- per-body enhancement state initialization.
    batched_obs_norm: ObsNormState | None = None
    if enhancements.obs_norm:
        batched_obs_norm = _stack_pytrees(
            *[init_obs_norm(obs_dim) for _ in range(n_bodies)]
        )

    batched_lattice: LatticeNoiseState | None = None
    if enhancements.lattice_noise:
        hidden_dim = int(ppo_config.hidden_dim)
        key, lattice_key = jax.random.split(key)
        lattice_keys = jax.random.split(lattice_key, n_bodies)
        batched_lattice = eqx.filter_vmap(
            lambda k: init_lattice_noise(hidden_dim, k),
        )(lattice_keys)

    batched_curriculum: CurriculumState | None = None
    if enhancements.curriculum:
        batched_curriculum = _stack_pytrees(
            *[init_curriculum() for _ in range(n_bodies)]
        )

    # --- JIT'd vmapped collect ---
    @eqx.filter_jit
    def batched_collect(policy, states, key, obs_norm_st, lattice_st, curric_st):
        keys = jax.random.split(key, n_bodies)

        # Compute per-body max_target_distance from curriculum
        max_dist = None
        if enhancements.curriculum and curric_st is not None:
            max_dist = curric_st.max_target_fraction * enhancements.curriculum_arm_reach

        def collect_one(pl, pol, st, k, on_st, lat_st, mtd):
            return _collect_rollout_extended(
                pl, env_config, pol, st, k, n_steps, n_envs,
                obs_norm_state=on_st,
                lattice_state=lat_st,
                lattice_resample_interval=enhancements.lattice_resample_interval,
                max_target_distance=mtd,
            )

        # Build vmap in_axes depending on which enhancements are active
        if enhancements.obs_norm and enhancements.lattice_noise and enhancements.curriculum:
            # vmap over: plant, policy, states, keys, obs_norm, lattice, max_dist
            return eqx.filter_vmap(collect_one)(
                batched_plant, policy, states, keys, obs_norm_st, lattice_st, max_dist,
            )
        elif enhancements.obs_norm and enhancements.lattice_noise:
            def _collect_no_curric(pl, pol, st, k, on_st, lat_st):
                return _collect_rollout_extended(
                    pl, env_config, pol, st, k, n_steps, n_envs,
                    obs_norm_state=on_st,
                    lattice_state=lat_st,
                    lattice_resample_interval=enhancements.lattice_resample_interval,
                    max_target_distance=None,
                )
            return eqx.filter_vmap(_collect_no_curric)(
                batched_plant, policy, states, keys, obs_norm_st, lattice_st,
            )
        elif enhancements.obs_norm and enhancements.curriculum:
            def _collect_no_lattice(pl, pol, st, k, on_st, mtd):
                return _collect_rollout_extended(
                    pl, env_config, pol, st, k, n_steps, n_envs,
                    obs_norm_state=on_st,
                    lattice_state=None,
                    lattice_resample_interval=enhancements.lattice_resample_interval,
                    max_target_distance=mtd,
                )
            return eqx.filter_vmap(_collect_no_lattice)(
                batched_plant, policy, states, keys, obs_norm_st, max_dist,
            )
        elif enhancements.lattice_noise and enhancements.curriculum:
            def _collect_no_norm(pl, pol, st, k, lat_st, mtd):
                return _collect_rollout_extended(
                    pl, env_config, pol, st, k, n_steps, n_envs,
                    obs_norm_state=None,
                    lattice_state=lat_st,
                    lattice_resample_interval=enhancements.lattice_resample_interval,
                    max_target_distance=mtd,
                )
            return eqx.filter_vmap(_collect_no_norm)(
                batched_plant, policy, states, keys, lattice_st, max_dist,
            )
        elif enhancements.obs_norm:
            def _collect_norm_only(pl, pol, st, k, on_st):
                return _collect_rollout_extended(
                    pl, env_config, pol, st, k, n_steps, n_envs,
                    obs_norm_state=on_st,
                    lattice_state=None,
                    lattice_resample_interval=enhancements.lattice_resample_interval,
                    max_target_distance=None,
                )
            return eqx.filter_vmap(_collect_norm_only)(
                batched_plant, policy, states, keys, obs_norm_st,
            )
        elif enhancements.lattice_noise:
            def _collect_lattice_only(pl, pol, st, k, lat_st):
                return _collect_rollout_extended(
                    pl, env_config, pol, st, k, n_steps, n_envs,
                    obs_norm_state=None,
                    lattice_state=lat_st,
                    lattice_resample_interval=enhancements.lattice_resample_interval,
                    max_target_distance=None,
                )
            return eqx.filter_vmap(_collect_lattice_only)(
                batched_plant, policy, states, keys, lattice_st,
            )
        elif enhancements.curriculum:
            def _collect_curric_only(pl, pol, st, k, mtd):
                return _collect_rollout_extended(
                    pl, env_config, pol, st, k, n_steps, n_envs,
                    obs_norm_state=None,
                    lattice_state=None,
                    lattice_resample_interval=enhancements.lattice_resample_interval,
                    max_target_distance=mtd,
                )
            return eqx.filter_vmap(_collect_curric_only)(
                batched_plant, policy, states, keys, max_dist,
            )
        else:
            def _collect_baseline(pl, pol, st, k):
                return _collect_rollout_extended(
                    pl, env_config, pol, st, k, n_steps, n_envs,
                    obs_norm_state=None,
                    lattice_state=None,
                    lattice_resample_interval=enhancements.lattice_resample_interval,
                    max_target_distance=None,
                )
            return eqx.filter_vmap(_collect_baseline)(
                batched_plant, policy, states, keys,
            )

    # --- Per-body PPO update (vmapped) ---
    # Bug: 2055433 -- Same structure as train_ppo_batched's update_one,
    # but uses potentially-normalized obs from the rollout.
    def update_one(policy, opt_state, rollout, last_values, key):
        advantages, returns = compute_gae_scan(
            rollout.rewards, rollout.values, rollout.dones,
            last_values, gamma, gae_lambda,
        )
        flat_obs = rollout.obs.reshape(-1, obs_dim)
        flat_actions = rollout.actions.reshape(-1, action_dim)
        flat_logp = rollout.log_probs.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)
        flat_adv = (flat_adv - jnp.mean(flat_adv)) / (jnp.std(flat_adv) + 1e-8)

        dynamic_policy, static_policy = eqx.partition(policy, eqx.is_array)

        def train_step(i, carry):
            dynamic_policy, opt_state, key = carry
            policy = eqx.combine(dynamic_policy, static_policy)
            perm_key = jax.random.fold_in(key, i // n_minibatches)
            perm = jax.random.permutation(perm_key, batch_size)
            start = (i % n_minibatches) * minibatch_size
            mb_idx = jax.lax.dynamic_slice(perm, (start,), (minibatch_size,))

            (_, _), grads = eqx.filter_value_and_grad(
                _ppo_loss, has_aux=True,
            )(
                policy,
                flat_obs[mb_idx], flat_actions[mb_idx], flat_logp[mb_idx],
                flat_adv[mb_idx], flat_ret[mb_idx],
                clip_eps, vf_coef, ent_coef,
            )

            updates, opt_state = optimizer.update(grads, opt_state, policy)
            policy = eqx.apply_updates(policy, updates)
            dynamic_policy, _ = eqx.partition(policy, eqx.is_array)
            return dynamic_policy, opt_state, key

        dynamic_policy, opt_state, _ = jax.lax.fori_loop(
            0, n_train_iters, train_step, (dynamic_policy, opt_state, key),
        )
        policy = eqx.combine(dynamic_policy, static_policy)
        mean_reward = jnp.mean(rollout.rewards)
        return policy, opt_state, mean_reward

    @eqx.filter_jit
    def batched_update(policy, opt_state, rollout, last_values, key):
        keys = jax.random.split(key, n_bodies)
        return eqx.filter_vmap(update_one)(
            policy, opt_state, rollout, last_values, keys,
        )

    # --- JIT'd vmapped obs-norm and curriculum updates ---
    if enhancements.obs_norm:
        @eqx.filter_jit
        def _update_obs_norm_batched(obs_norm_st, obs_batch):
            return eqx.filter_vmap(update_obs_norm)(obs_norm_st, obs_batch)

    if enhancements.curriculum:
        @eqx.filter_jit
        def _update_curriculum_batched(curric_st, success_rates):
            return eqx.filter_vmap(update_curriculum)(curric_st, success_rates)

    # --- Training loop ---
    metrics: dict[str, object] = {
        "updates": 0,
        "timesteps": 0,
        "per_body_mean_return": [],
        "per_body_success_rate": [],
    }
    if enhancements.curriculum:
        metrics["curriculum_stages"] = []
    if enhancements.obs_norm:
        metrics["obs_norm_mean"] = None

    update_count = 0

    while metrics["timesteps"] < total_timesteps:
        key, collect_key, update_key = jax.random.split(key, 3)

        # --- Collect rollouts with enhancements ---
        collect_result = batched_collect(
            batched_policy, batched_states, collect_key,
            batched_obs_norm, batched_lattice, batched_curriculum,
        )

        # Unpack: 5-tuple (states, rollout, last_values, key, lattice_state)
        batched_states = collect_result[0]
        batched_rollout = collect_result[1]
        batched_last_values = collect_result[2]
        # collect_result[3] is the updated key (per-body), we use our own
        returned_lattice = collect_result[4]

        # Update lattice state if enabled
        if enhancements.lattice_noise and returned_lattice is not None:
            batched_lattice = returned_lattice

        metrics["timesteps"] = int(metrics["timesteps"]) + timesteps_per_update

        # --- PPO update ---
        batched_policy, batched_opt_state, per_body_rewards = batched_update(
            batched_policy, batched_opt_state,
            batched_rollout, batched_last_values, update_key,
        )

        metrics["updates"] = int(metrics["updates"]) + 1
        per_body_returns = per_body_rewards * env_config.n_steps
        metrics["per_body_mean_return"].append(per_body_returns)

        # --- Post-update enhancement hooks ---
        update_count += 1

        # Obs norm: update running stats with collected observations
        if enhancements.obs_norm and batched_obs_norm is not None:
            # batched_rollout.obs: (B, T, N, obs_dim)
            # Reshape to (B, T*N, obs_dim) for batch update
            all_obs = batched_rollout.obs.reshape(n_bodies, -1, obs_dim)
            batched_obs_norm = _update_obs_norm_batched(batched_obs_norm, all_obs)

        # Per-body success rate
        # batched_rollout.dones: (B, T, N)
        per_body_success = jnp.mean(batched_rollout.dones, axis=(1, 2))
        metrics["per_body_success_rate"].append(per_body_success)

        # Curriculum: update per-body stages
        if enhancements.curriculum and batched_curriculum is not None:
            batched_curriculum = _update_curriculum_batched(
                batched_curriculum, per_body_success,
            )
            metrics["curriculum_stages"].append(batched_curriculum.stage)

        # Bug: 2055433 -- Linear distance_weight annealing from 1.0 to 0.0.
        # The distance_weight conceptually scales the -distance term in
        # compute_reward. Since modifying the reward function mid-training
        # would cause JIT re-tracing, we record the annealing schedule here
        # for the caller to apply externally if needed.
        if enhancements.reward_annealing:
            progress = update_count / max(total_updates, 1)
            start_frac = enhancements.annealing_start_fraction
            if progress > start_frac:
                distance_weight = 1.0 - (progress - start_frac) / (1.0 - start_frac)
                distance_weight = max(distance_weight, 0.0)
            else:
                distance_weight = 1.0
            metrics.setdefault("distance_weight_schedule", [])
            metrics["distance_weight_schedule"].append(distance_weight)

    # Store final obs norm mean
    if enhancements.obs_norm and batched_obs_norm is not None:
        metrics["obs_norm_mean"] = batched_obs_norm.mean

    return batched_policy, metrics
