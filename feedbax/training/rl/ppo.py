"""Vectorized PPO + GAE training for RL policies.

Rollout collection uses jax.vmap over N parallel environments and
jax.lax.scan over timesteps. GAE is computed with a reversed scan.
Generalized to work with any feedbax AbstractPlant.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from feedbax.mechanics.plant import AbstractPlant
from feedbax.training.rl.env import (
    RLEnvConfig,
    RLEnvState,
    auto_reset,
    rl_env_get_obs,
    rl_env_reset,
    rl_env_step,
)
from feedbax.training.rl.policy import ActorCritic
from feedbax.training.rl.tasks import (
    reconstruct_trajectory,
    sample_task_params_jax,
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
        task = sample_task_params_jax(task_key, None, cfg.n_steps, cfg.dt)
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
        task = sample_task_params_jax(
            task_key, task_type, env_config.n_steps, env_config.dt
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
