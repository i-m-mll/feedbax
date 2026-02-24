"""Functional RL environment for GPU-parallel training.

All state lives in RLEnvState. Functions are pure and JIT/vmap-compatible.
Works with any feedbax AbstractPlant via Diffrax Euler integration.
"""

from __future__ import annotations

from typing import NamedTuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Float, PRNGKeyArray

from feedbax.mechanics.plant import AbstractPlant, PlantState
from feedbax.training.rl.rewards import compute_reward
from feedbax.training.rl.tasks import (
    TASK_HOLD,
    TASK_REACH,
    TaskParams,
    sample_task_params_jax,
    target_at_t,
)


class RLEnvConfig(eqx.Module):
    """Static configuration for the RL environment.

    Attributes:
        n_steps: Number of control steps per episode.  Each control step runs
            ``frame_skip`` physics sub-steps, so the episode spans
            ``n_steps * frame_skip * dt`` seconds of simulated time.
        dt: Physics timestep in seconds.
        frame_skip: Number of physics sub-steps per control step.  The action
            (muscle command) is held constant across sub-steps while activation
            dynamics and physics are integrated at the finer ``dt`` resolution.
            With ``dt=0.002`` and ``frame_skip=5`` the control rate is 100 Hz.
        n_joints: Number of joints.
        n_muscles: Number of muscle actuators.
        tau_act: Muscle activation time constant in seconds.
        tau_deact: Muscle deactivation time constant in seconds.
        effort_weight: Reward effort penalty weight.
        velocity_weight: Reward velocity penalty weight.
        hold_bonus: Reward hold bonus.
        hold_threshold: Reward hold threshold in meters.
        action_scale: Scale factor for actions (maps [0,1] to torque range).
        action_offset: Offset for actions after scaling.
    """

    n_steps: int = eqx.field(static=True)
    dt: float
    n_joints: int = eqx.field(static=True)
    n_muscles: int = eqx.field(static=True)
    frame_skip: int = eqx.field(static=True, default=5)
    tau_act: float = 0.01
    tau_deact: float = 0.04
    effort_weight: float = 0.005
    velocity_weight: float = 0.1
    hold_bonus: float = 1.0
    hold_threshold: float = 0.02
    action_scale: float = 1.0
    action_offset: float = 0.0


class RLEnvState(NamedTuple):
    """Carries simulation state and episode bookkeeping.

    Attributes:
        plant_state: Current plant state (skeleton + optional muscles).
        muscle_activations: Current muscle activation levels, shape ``(n_muscles,)``.
        t_index: Current timestep index.
        task: Current task specification.
    """

    plant_state: PlantState
    muscle_activations: Float[Array, " n_muscles"]
    t_index: Float[Array, ""]
    task: TaskParams


def rl_env_reset(
    plant: AbstractPlant,
    config: RLEnvConfig,
    task: TaskParams,
    key: PRNGKeyArray,
) -> RLEnvState:
    """Reset the RL environment with a new task.

    Args:
        plant: The plant model.
        config: Environment configuration.
        task: Task specification for this episode.
        key: PRNG key for state initialization.

    Returns:
        Initial RLEnvState.
    """
    plant_state = plant.init(key=key)

    return RLEnvState(
        plant_state=plant_state,
        muscle_activations=jnp.zeros(config.n_muscles),
        t_index=jnp.array(0, dtype=jnp.int32),
        task=task,
    )


def rl_env_get_obs(
    plant: AbstractPlant,
    config: RLEnvConfig,
    state: RLEnvState,
) -> Float[Array, " obs_dim"]:
    """Extract observation vector from environment state.

    Observation includes: joint positions, joint velocities, muscle activations,
    effector position, target position, target velocity, and phase.

    Args:
        plant: The plant model.
        config: Environment configuration.
        state: Current environment state.

    Returns:
        Flat observation array.
    """
    skeleton_state = state.plant_state.skeleton
    effector = plant.skeleton.effector(skeleton_state)
    t = state.t_index
    target_pos, target_vel = target_at_t(state.task, t)
    phase = jnp.array([t / jnp.maximum(config.n_steps - 1, 1)])

    # Extract joint positions and velocities via named attributes.
    # MJXSkeletonState uses .qpos/.qvel; TwoLinkArmState uses .angle/.d_angle.
    # Bug: 67e2e5e — replaces brittle jt.leaves ordering.
    if hasattr(skeleton_state, "qpos"):
        qpos = skeleton_state.qpos
        qvel = skeleton_state.qvel
    elif hasattr(skeleton_state, "angle"):
        qpos = skeleton_state.angle
        qvel = skeleton_state.d_angle
    else:
        raise TypeError(
            f"Unknown skeleton state type {type(skeleton_state).__name__}: "
            "expected .qpos/.qvel or .angle/.d_angle attributes"
        )

    return jnp.concatenate([
        jnp.atleast_1d(qpos),
        jnp.atleast_1d(qvel),
        state.muscle_activations,
        effector.pos,
        target_pos,
        target_vel,
        phase,
    ])


def _physics_substep(
    plant: AbstractPlant,
    config: RLEnvConfig,
    action: Float[Array, " n_muscles"],
    carry: tuple[PlantState, Float[Array, " n_muscles"]],
) -> tuple[PlantState, Float[Array, " n_muscles"]]:
    """Run one physics sub-step: activation dynamics + Euler integration.

    The muscle command (``action``) is held constant; only activations and
    the plant state evolve.

    Args:
        plant: The plant model.
        config: Environment configuration.
        action: Muscle excitation command (constant across sub-steps).
        carry: Tuple of (plant_state, muscle_activations).

    Returns:
        Updated (plant_state, muscle_activations) after one ``dt`` step.
    """
    plant_state, activations = carry

    # First-order muscle activation dynamics (per sub-step at physics dt)
    tau = jnp.where(
        action > activations,
        config.tau_act,
        config.tau_deact,
    )
    da = (action - activations) / jnp.maximum(tau, 1e-6)
    activations = jnp.clip(activations + config.dt * da, 0.0, 1.0)

    # Scale activations to control signal
    ctrl = activations * config.action_scale + config.action_offset

    # Diffrax Euler integration
    term = dfx.ODETerm(plant.vector_field)
    solver = dfx.Euler()
    plant_state = plant.kinematics_update(ctrl, plant_state)
    plant_state, _, _, _, _ = solver.step(
        term, 0, config.dt, plant_state, ctrl, None, made_jump=False,
    )

    return plant_state, activations


def rl_env_step(
    plant: AbstractPlant,
    config: RLEnvConfig,
    state: RLEnvState,
    action: Float[Array, " n_muscles"],
) -> tuple[RLEnvState, Float[Array, " obs_dim"], Float[Array, ""], Float[Array, ""]]:
    """Step the RL environment forward by one control step.

    Runs ``config.frame_skip`` physics sub-steps per control step via
    ``jax.lax.fori_loop``.  The action (muscle command) is held constant
    across sub-steps while activation dynamics and physics are integrated
    at the finer ``dt`` resolution.  Only the final sub-step state is
    returned.  Bug: 67e2e5e

    Args:
        plant: The plant model.
        config: Environment configuration.
        state: Current environment state.
        action: Muscle excitation commands in [0, 1], shape ``(n_muscles,)``.

    Returns:
        Tuple of (new_state, observation, reward, done).
    """
    action = jnp.clip(action, 0.0, 1.0)
    t = state.t_index

    # Apply HOLD perturbation as an impulse force before the physics sub-steps.
    # Bug: 67e2e5e — The force is set on the effector body at the perturbation
    # timestep, persists across frame_skip sub-steps, then is cleared after
    # integration so it acts as a single control-step impulse.
    is_perturb_step = (t == state.task.perturb_time_idx)
    is_hold = (jnp.asarray(state.task.task_type) == TASK_HOLD)
    apply_perturb = is_perturb_step & is_hold

    perturbed_skeleton = plant.skeleton.update_state_given_effector_force(
        state.task.perturb_force, state.plant_state.skeleton,
    )
    perturbed_plant_state = eqx.tree_at(
        lambda s: s.skeleton,
        state.plant_state,
        jt.map(
            lambda orig, pert: jnp.where(apply_perturb, pert, orig),
            state.plant_state.skeleton,
            perturbed_skeleton,
        ),
    )

    # Run frame_skip physics sub-steps with constant action.
    init_carry = (perturbed_plant_state, state.muscle_activations)

    def substep_body(_, carry):
        return _physics_substep(plant, config, action, carry)

    new_plant_state, new_activations = jax.lax.fori_loop(
        0, config.frame_skip, substep_body, init_carry,
    )

    # Clear external forces after the physics sub-steps so the perturbation
    # does not leak into subsequent control steps.
    cleared_skeleton = plant.skeleton.update_state_given_effector_force(
        jnp.zeros(2), new_plant_state.skeleton,
    )
    new_plant_state = eqx.tree_at(
        lambda s: s.skeleton,
        new_plant_state,
        jt.map(
            lambda orig, cleared: jnp.where(apply_perturb, cleared, orig),
            new_plant_state.skeleton,
            cleared_skeleton,
        ),
    )

    # Compute effector state — use analytical velocity from the skeleton
    # (e.g. MJX's data.site_xvelp) instead of finite-difference approximation.
    # Bug: 67e2e5e
    effector = plant.skeleton.effector(new_plant_state.skeleton)
    effector_vel = effector.vel

    # Reward
    target_pos, target_vel = target_at_t(state.task, t)
    reward = compute_reward(
        task_type=jnp.asarray(state.task.task_type, dtype=jnp.float32),
        effector_pos=effector.pos,
        target_pos=target_pos,
        effector_vel=effector_vel,
        target_vel=target_vel,
        muscle_excitations=action,
        effort_weight=config.effort_weight,
        velocity_weight=config.velocity_weight,
        hold_bonus=config.hold_bonus,
        hold_threshold=config.hold_threshold,
        step=t,
        n_steps=config.n_steps,
    )

    # Bug: 67e2e5e — Early termination on success for REACH tasks.
    # Agent is done when it reaches within hold_threshold of the target.
    distance = jnp.linalg.norm(effector.pos - target_pos)
    reached_target = distance < config.hold_threshold
    is_reach = (jnp.asarray(state.task.task_type) == TASK_REACH)
    early_done = is_reach & reached_target

    new_t = t + 1
    done = jnp.maximum(
        (new_t >= config.n_steps).astype(jnp.float32),
        early_done.astype(jnp.float32),
    )

    new_state = RLEnvState(
        plant_state=new_plant_state,
        muscle_activations=new_activations,
        t_index=new_t,
        task=state.task,
    )

    obs = rl_env_get_obs(plant, config, new_state)
    return new_state, obs, reward, done


def auto_reset(
    plant: AbstractPlant,
    config: RLEnvConfig,
    state: RLEnvState,
    done: Float[Array, ""],
    key: PRNGKeyArray,
) -> RLEnvState:
    """Reset environments that are done, keeping others unchanged.

    Args:
        plant: The plant model.
        config: Environment configuration.
        state: Current environment state.
        done: Whether the episode is done (1.0) or not (0.0).
        key: PRNG key for new task and state generation.

    Returns:
        Conditionally reset RLEnvState.
    """
    key, task_key, reset_key = jax.random.split(key, 3)
    new_task = sample_task_params_jax(
        task_key, None, config.n_steps, config.dt
    )
    new_state = rl_env_reset(plant, config, new_task, reset_key)

    return jt.map(
        lambda old, new: jnp.where(done, new, old),
        state,
        new_state,
    )
