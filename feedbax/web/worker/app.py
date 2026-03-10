"""Feedbax Studio headless training worker FastAPI app."""
from __future__ import annotations

import asyncio
import collections
import json
import queue
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


class WorkerStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


# Maximum number of past events to buffer per job for from_seq replay.
_EVENT_BUFFER_MAX = 1000


@dataclass
class _Job:
    job_id: str
    total_batches: int
    event_queue: queue.Queue
    stop_event: threading.Event
    # Parsed training configuration dict passed from the API layer.
    training_config: Optional[Dict[str, Any]] = None
    # Buffer of (seq, event_dict) for replay support.
    event_buffer: Deque[Tuple[int, dict]] = field(
        default_factory=lambda: collections.deque(maxlen=_EVENT_BUFFER_MAX)
    )
    thread: Optional[threading.Thread] = None
    status: WorkerStatus = WorkerStatus.IDLE
    # Spec dicts forwarded from the API layer.
    training_spec: Optional[Dict[str, Any]] = None
    task_spec: Optional[Dict[str, Any]] = None
    # Graph spec dict forwarded from the API layer for network param extraction.
    graph_spec: Optional[Dict[str, Any]] = None
    # Path to the serialized checkpoint file after training completes.
    checkpoint_path: Optional[str] = None
    batch: int = 0
    last_loss: float = 0.0
    snapshot_interval: int = 100
    # Monotonically increasing sequence counter; protected by _seq_lock.
    _seq: int = 0
    _seq_lock: threading.Lock = field(default_factory=threading.Lock)

    def next_seq(self) -> int:
        """Return the next sequence number and advance the counter."""
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
            return seq


def _make_trajectory_event(job: _Job, batch: int, loss: float) -> dict:
    """Generate a synthetic 2D reaching trajectory snapshot."""
    n_steps = 50
    t = np.linspace(0.0, 0.5, n_steps).tolist()
    target_x = random.uniform(0.1, 0.3)
    target_y = random.uniform(0.1, 0.3)
    noise_scale = loss * 0.1
    rng = np.random.default_rng()
    noise_x = rng.normal(0.0, noise_scale, n_steps)
    noise_y = rng.normal(0.0, noise_scale, n_steps)
    progress = np.linspace(0.0, 1.0, n_steps)
    effector = [
        [float(target_x * s + nx), float(target_y * s + ny)]
        for s, nx, ny in zip(progress, noise_x, noise_y)
    ]
    return {
        "type": "training_trajectory",
        "job_id": job.job_id,
        "batch": batch,
        "trajectory": {
            "effector": effector,
            "target": [target_x, target_y],
            "t": t,
            "n_steps": n_steps,
        },
    }


# ---------------------------------------------------------------------------
# Training configuration extraction
# ---------------------------------------------------------------------------


@dataclass
class _TrainingCfg:
    """Normalized training configuration for _run_training_real."""

    n_batches: int = 2000
    batch_size: int = 128
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    hidden_dim: int = 128
    network_type: str = "gru"
    n_reach_steps: int = 80
    effort_weight: float = 2.5
    snapshot_interval: int = 100


def _extract_training_cfg(
    training_config: Optional[Dict[str, Any]],
    task_spec: Optional[Dict[str, Any]] = None,
) -> _TrainingCfg:
    """Parse a raw config dict into a normalized _TrainingCfg.

    Falls back to defaults for any missing or invalid field.

    Args:
        training_config: Optional dict from the ``/start`` request body.
        task_spec: Optional task spec dict; overrides task params such as
            ``n_reach_steps`` and ``effort_weight`` when present.

    Returns:
        A _TrainingCfg with all fields populated.
    """
    cfg = _TrainingCfg()
    if training_config is None and task_spec is None:
        return cfg

    if training_config is not None:
        def _get(key: str, default, cast=None):
            val = training_config.get(key, default)
            if val is None:
                return default
            try:
                return cast(val) if cast is not None else val
            except (TypeError, ValueError):
                return default

        cfg.n_batches = _get("n_batches", cfg.n_batches, int)
        cfg.batch_size = _get("batch_size", cfg.batch_size, int)
        cfg.learning_rate = _get("learning_rate", cfg.learning_rate, float)
        cfg.grad_clip = _get("grad_clip", cfg.grad_clip, float)
        cfg.hidden_dim = _get("hidden_dim", cfg.hidden_dim, int)
        cfg.network_type = _get("network_type", cfg.network_type, str)
        cfg.n_reach_steps = _get("n_reach_steps", cfg.n_reach_steps, int)
        cfg.effort_weight = _get("effort_weight", cfg.effort_weight, float)
        cfg.snapshot_interval = _get("snapshot_interval", cfg.snapshot_interval, int)

    if task_spec is not None:
        task_params = task_spec.get("params", {})
        for key, cast in [("n_reach_steps", int), ("effort_weight", float)]:
            if key in task_params:
                try:
                    setattr(cfg, key, cast(task_params[key]))
                except (TypeError, ValueError):
                    pass

    return cfg


# ---------------------------------------------------------------------------
# Spec-driven optimizer and loss-weight helpers
# ---------------------------------------------------------------------------


def _build_optimizer_from_spec(
    training_spec: Optional[Dict[str, Any]],
    cfg: "_TrainingCfg",
):
    """Build an optax optimizer from a training spec dict.

    Args:
        training_spec: Optional spec dict with an ``optimizer`` sub-dict.
        cfg: Parsed training config (provides fallback learning rate and
            grad-clip).

    Returns:
        An ``optax.GradientTransformation``.
    """
    import optax  # imported here so the module loads without JAX

    clip = optax.clip_by_global_norm(cfg.grad_clip)
    if training_spec is None:
        return optax.chain(clip, optax.adamw(cfg.learning_rate, weight_decay=1e-6))

    opt_spec = training_spec.get("optimizer", {})
    opt_type = str(opt_spec.get("type", "adamw")).lower()
    params = opt_spec.get("params", {})

    def _p(key, default):
        v = params.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    lr = _p("learning_rate", cfg.learning_rate)

    if opt_type == "adam":
        inner = optax.adam(lr, b1=_p("b1", 0.9), b2=_p("b2", 0.999))
    elif opt_type == "sgd":
        inner = optax.sgd(lr, momentum=_p("momentum", 0.0))
    elif opt_type == "rmsprop":
        inner = optax.rmsprop(lr, decay=_p("decay", 0.9))
    else:  # adamw default
        inner = optax.adamw(
            lr,
            b1=_p("b1", 0.9),
            b2=_p("b2", 0.999),
            weight_decay=_p("weight_decay", 1e-6),
        )

    return optax.chain(clip, inner)


def _extract_effort_weight_from_spec(
    training_spec: Optional[Dict[str, Any]], default: float
) -> float:
    """Extract effort loss weight from a training spec.

    Args:
        training_spec: Optional spec dict; reads
            ``loss.children.effort.weight`` when present.
        default: Value to return when the key is absent or invalid.

    Returns:
        The effort weight as a float.
    """
    if training_spec is None:
        return default
    try:
        return float(training_spec["loss"]["children"]["effort"]["weight"])
    except (KeyError, TypeError, ValueError):
        return default


def _extract_graph_params(graph_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract model-construction parameters from a graph spec dict.

    Reads from the Network node's internal subgraph when available (the
    subgraph is the authoritative source per the graph-as-model principle).
    Falls back to outer Network node params for backwards compatibility with
    graphs that pre-date the real composite subgraph.

    Also extracts ``input_size`` from the outer Network node params (the
    subgraph cell's input_size is an internal wiring detail; the outer
    Network param is the canonical interface dimension), and ``dt`` from the
    first mechanics/plant node found in the top-level graph.

    Returns a dict with keys:
        hidden_type: equinox cell class (default eqx.nn.GRUCell)
        hidden_size: hidden state dimension (default 128)
        out_size: output dimension (default 6)
        out_nonlinearity: activation callable (default jax.nn.sigmoid)
        input_size: network input dimension (default 17)
        dt: control timestep in seconds (default 0.01)
    """
    import equinox as eqx
    import jax

    CELL_MAP = {
        "GRU": eqx.nn.GRUCell,
        "LSTM": eqx.nn.LSTMCell,
        # Legacy outer-param values (kept for backwards compat)
        "GRUCell": eqx.nn.GRUCell,
        "LSTMCell": eqx.nn.LSTMCell,
        "SimpleRNNCell": eqx.nn.GRUCell,
    }
    NONLINEARITY_MAP = {
        "sigmoid": jax.nn.sigmoid,
        "relu": jax.nn.relu,
        "tanh": jax.nn.tanh,
        "softmax": jax.nn.softmax,
        "identity": lambda x: x,
    }
    # Node types that carry a ``dt`` param representing the mechanics timestep.
    # Bug: cb13bdc — mechanics dt should come from the graph spec, not be hardcoded.
    _MECHANICS_NODE_TYPES = frozenset({
        "TwoLinkArm",
        "PointMass",
        "Mechanics",
        "Arm6MuscleRigidTendon",
        "PointMass8MuscleRelu",
        "AcausalSystem",
    })

    defaults = {
        "hidden_type": eqx.nn.GRUCell,
        "hidden_size": 128,
        "out_size": 6,
        "out_nonlinearity": jax.nn.sigmoid,
        # obs: joint_angles(2) + joint_vels(2) + activations(6)
        #      + effector_pos(2) + target_pos(2) + target_vel(2) + phase(1) = 17
        "input_size": 17,
        "dt": 0.01,
    }

    if graph_spec is None:
        return defaults

    nodes = graph_spec.get("nodes", {})
    network_node_id = next(
        (nid for nid, n in nodes.items() if n.get("type") == "Network"),
        None,
    )

    # ------------------------------------------------------------------
    # Extract dt from the first mechanics node in the top-level graph.
    # Bug: cb13bdc — read dt from graph spec instead of hardcoding.
    # ------------------------------------------------------------------
    result = dict(defaults)
    mechanics_node = next(
        (n for n in nodes.values() if n.get("type") in _MECHANICS_NODE_TYPES),
        None,
    )
    if mechanics_node is not None:
        mech_params = mechanics_node.get("params", {})
        try:
            result["dt"] = float(mech_params.get("dt", defaults["dt"]))
        except (TypeError, ValueError):
            pass

    if network_node_id is None:
        return result

    # ------------------------------------------------------------------
    # Extract input_size from outer Network node params.
    # The outer param is the canonical interface dimension; the subgraph
    # cell's input_size is an internal wiring detail.
    # Bug: cb13bdc — read input_size from graph spec instead of hardcoding.
    # ------------------------------------------------------------------
    network_node = nodes[network_node_id]
    outer_params = network_node.get("params", {})
    try:
        result["input_size"] = int(
            outer_params.get("input_size", defaults["input_size"])
        )
    except (TypeError, ValueError):
        pass

    # ------------------------------------------------------------------
    # Prefer reading hidden/output architecture from the Network node's
    # internal subgraph.  The subgraph is the authoritative source of truth;
    # outer params are a legacy fallback for graphs without a subgraph yet.
    # ------------------------------------------------------------------
    subgraphs = graph_spec.get("subgraphs") or {}
    network_subgraph = subgraphs.get(network_node_id)

    if network_subgraph is not None:
        sub_nodes = network_subgraph.get("nodes", {})
        # Find the hidden cell node (GRU or LSTM)
        cell_node = next(
            (n for n in sub_nodes.values() if n.get("type") in ("GRU", "LSTM")),
            None,
        )
        # Find the readout/output projection node (Linear)
        readout_node = next(
            (n for n in sub_nodes.values() if n.get("type") == "Linear"),
            None,
        )

        if cell_node is not None:
            result["hidden_type"] = CELL_MAP.get(
                cell_node.get("type", "GRU"), eqx.nn.GRUCell
            )
            cell_params = cell_node.get("params", {})
            try:
                result["hidden_size"] = int(
                    cell_params.get("hidden_size", defaults["hidden_size"])
                )
            except (TypeError, ValueError):
                pass

        if readout_node is not None:
            readout_params = readout_node.get("params", {})
            try:
                result["out_size"] = int(
                    readout_params.get("output_size", defaults["out_size"])
                )
            except (TypeError, ValueError):
                pass
            nonlin_key = readout_params.get("activation", "identity")
            result["out_nonlinearity"] = NONLINEARITY_MAP.get(
                nonlin_key, lambda x: x
            )

        return result

    # Fallback: read hidden/output architecture from outer Network node params
    # (legacy / no-subgraph case).
    hidden_type_key = outer_params.get("hidden_type", "GRUCell")
    result["hidden_type"] = CELL_MAP.get(hidden_type_key, eqx.nn.GRUCell)

    try:
        result["hidden_size"] = int(
            outer_params.get("hidden_size", defaults["hidden_size"])
        )
    except (TypeError, ValueError):
        pass

    try:
        result["out_size"] = int(
            outer_params.get("out_size", defaults["out_size"])
        )
    except (TypeError, ValueError):
        pass

    nonlin_key = outer_params.get("out_nonlinearity", "sigmoid")
    result["out_nonlinearity"] = NONLINEARITY_MAP.get(nonlin_key, jax.nn.sigmoid)

    return result


# ---------------------------------------------------------------------------
# Real JAX training backend
# ---------------------------------------------------------------------------


def _run_training_real(job: _Job, cfg: "_TrainingCfg") -> None:
    """Real JAX training loop using AnalyticalMusculoskeletalPlant + GRU controller.

    Runs in a background thread. Streams training_progress, training_log,
    training_trajectory, and terminal (training_complete / training_error)
    events via job.event_queue.

    Imports JAX lazily so the worker process starts quickly even if JAX is
    slow to initialize.

    Args:
        job: The current _Job (used for stop_event, emit, and metadata).
        cfg: Parsed training configuration.
    """
    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        import optax

        from feedbax.mechanics.backend import DiffraxBackend, PhysicsState
        from feedbax.mechanics.body import (
            BodyPreset,
            default_2link_bounds,
        )
        from feedbax.mechanics.analytical_plant import AnalyticalMusculoskeletalPlant
        from feedbax.mechanics.model_builder import ChainConfig
        from feedbax.mechanics.muscle_config import default_6muscle_2link_topology
        from feedbax.nn import SimpleStagedNetwork

    except ImportError as exc:
        _emit(
            job,
            {
                "type": "training_error",
                "job_id": job.job_id,
                "error": f"Failed to import JAX/feedbax dependencies: {exc}",
            },
        )
        job.status = WorkerStatus.ERROR
        return

    # ------------------------------------------------------------------
    # Extract architecture and physics params from the graph spec.
    # Bug: cb13bdc — these values must come from the graph spec, not be
    # hardcoded. _extract_graph_params falls back to sensible defaults when
    # the graph spec is absent or incomplete.
    # ------------------------------------------------------------------

    graph_params = _extract_graph_params(job.graph_spec)

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    # Bug: cb13bdc — CONTROL_DT read from mechanics node dt param.
    CONTROL_DT = graph_params["dt"]
    N_STEPS = cfg.n_reach_steps  # control steps per episode
    N_MUSCLES = 6
    N_JOINTS = 2
    # Bug: cb13bdc — OBS_DIM read from Network node input_size param.
    OBS_DIM = graph_params["input_size"]

    # ------------------------------------------------------------------
    # Build plant (single canonical body preset at default parameters)
    # ------------------------------------------------------------------

    rng_key = jr.PRNGKey(0)
    preset_key, ctrl_key, rng_key = jr.split(rng_key, 3)

    bounds = default_2link_bounds()
    # Use the midpoint of the bounds to get a typical body.
    preset = BodyPreset(
        segment_lengths=0.5 * (bounds.segment_lengths_min + bounds.segment_lengths_max),
        segment_masses=0.5 * (bounds.segment_masses_min + bounds.segment_masses_max),
        joint_damping=0.5 * (bounds.joint_damping_min + bounds.joint_damping_max),
        joint_stiffness=0.5 * (bounds.joint_stiffness_min + bounds.joint_stiffness_max),
        muscle_pcsa=0.5 * (bounds.muscle_pcsa_min + bounds.muscle_pcsa_max),
        muscle_optimal_fiber_length=0.5 * (
            bounds.muscle_optimal_fiber_length_min
            + bounds.muscle_optimal_fiber_length_max
        ),
        muscle_tendon_slack_length=0.5 * (
            bounds.muscle_tendon_slack_length_min
            + bounds.muscle_tendon_slack_length_max
        ),
        muscle_moment_arm_magnitudes=0.5 * (
            bounds.muscle_moment_arm_magnitudes_min
            + bounds.muscle_moment_arm_magnitudes_max
        ),
    )

    topology = default_6muscle_2link_topology()
    chain_config = ChainConfig(n_joints=N_JOINTS, muscle_topology=topology)

    plant = AnalyticalMusculoskeletalPlant.from_body_preset(
        preset, chain_config, clip_states=True,
    )

    backend = DiffraxBackend(control_dt=CONTROL_DT)

    # ------------------------------------------------------------------
    # Build GRU controller (SimpleStagedNetwork with GRUCell hidden layer)
    # ------------------------------------------------------------------

    hidden_size = graph_params["hidden_size"]
    controller = SimpleStagedNetwork(
        input_size=OBS_DIM,
        hidden_size=hidden_size,
        out_size=graph_params["out_size"],
        hidden_type=graph_params["hidden_type"],
        out_nonlinearity=graph_params["out_nonlinearity"],
        key=ctrl_key,
    )

    # ------------------------------------------------------------------
    # Apply spec overrides BEFORE JIT (cfg mutations must precede any
    # jit-compiled functions that close over cfg values).
    # ------------------------------------------------------------------

    cfg.effort_weight = _extract_effort_weight_from_spec(
        job.training_spec, cfg.effort_weight
    )
    optimizer = _build_optimizer_from_spec(job.training_spec, cfg)

    # ------------------------------------------------------------------
    # Optimizer state
    # ------------------------------------------------------------------

    opt_state = optimizer.init(eqx.filter(controller, eqx.is_array))

    # ------------------------------------------------------------------
    # Observation helper
    # ------------------------------------------------------------------

    def _extract_obs(
        physics_state: PhysicsState,
        muscle_activations,
        target_pos,
        target_vel,
        phase,
    ):
        sk = physics_state.plant.skeleton
        effector = physics_state.effector
        return jnp.concatenate([
            sk.angle,
            sk.d_angle,
            muscle_activations,
            effector.pos,
            target_pos,
            target_vel,
            phase,
        ])

    # ------------------------------------------------------------------
    # Single-episode rollout through Diffrax (differentiable)
    # ------------------------------------------------------------------

    def _rollout(ctrl, target_pos_traj, target_vel_traj, episode_key):
        phys = backend.init_state(plant, key=episode_key)
        init_act = jnp.zeros(N_MUSCLES)
        init_phase = jnp.zeros(1)
        init_obs = _extract_obs(
            phys, init_act, target_pos_traj[0], target_vel_traj[0], init_phase,
        )
        # Controller hidden state: initialize to zeros
        init_hidden = jnp.zeros(hidden_size)

        scan_keys = jr.split(episode_key, N_STEPS)

        def _step(carry, inputs):
            t_idx, step_key = inputs
            phys_s, act, hidden, obs_prev = carry

            phase = jnp.array([t_idx / N_STEPS])
            obs = _extract_obs(
                phys_s, act,
                target_pos_traj[t_idx], target_vel_traj[t_idx],
                phase,
            )

            # GRU step: SimpleStagedNetwork wraps eqx.nn.GRUCell
            # We call the GRU cell directly to get new hidden state
            new_hidden = ctrl.hidden(obs, hidden)
            # Readout
            if ctrl.readout is not None:
                raw_out = ctrl.readout(new_hidden)
                action = ctrl.out_nonlinearity(raw_out)
            else:
                action = ctrl.out_nonlinearity(new_hidden)

            # Physics substep
            def _substep(ps, _):
                return backend.substep(plant, ps, action), None

            new_phys, _ = jax.lax.scan(_substep, phys_s, None, length=backend.n_substeps)

            # Update effector
            new_effector = backend.observe(plant, new_phys)
            new_phys = PhysicsState(
                plant=new_phys.plant,
                effector=new_effector,
                aux=new_phys.aux,
            )

            new_carry = (new_phys, action, new_hidden, obs)
            output = (new_effector.pos, action, new_hidden)
            return new_carry, output

        init_carry = (phys, init_act, init_hidden, init_obs)
        t_idxs = jnp.arange(N_STEPS)
        _, (eff_traj, act_traj, hidden_traj) = jax.lax.scan(
            _step, init_carry, (t_idxs, scan_keys),
        )
        return eff_traj, act_traj, hidden_traj

    # ------------------------------------------------------------------
    # Target sampling (random reach targets)
    # ------------------------------------------------------------------

    def _sample_targets(batch_key, batch_size):
        """Sample random 2D reach targets in the reachable workspace.

        Returns:
            target_pos_batch: shape (batch, N_STEPS, 2)
            target_vel_batch: shape (batch, N_STEPS, 2), zeros for reach
        """
        keys = jr.split(batch_key, batch_size)

        def _one_target(k):
            # Polar coordinates: r in [0.1, 0.4], theta in [0, pi/2]
            r = jr.uniform(k, minval=0.1, maxval=0.4)
            theta = jr.uniform(k, minval=0.0, maxval=jnp.pi / 2.0)
            tx = r * jnp.cos(theta)
            ty = r * jnp.sin(theta)
            target_pos = jnp.broadcast_to(jnp.array([tx, ty]), (N_STEPS, 2))
            target_vel = jnp.zeros((N_STEPS, 2))
            return target_pos, target_vel

        return jax.vmap(_one_target)(keys)

    # ------------------------------------------------------------------
    # Supervised loss
    # ------------------------------------------------------------------

    def _loss_fn(ctrl, target_pos_batch, target_vel_batch, batch_keys):
        """Mean supervised loss over a batch of episodes."""

        def _single(tgt_pos, tgt_vel, ep_key):
            eff_traj, act_traj, _ = _rollout(ctrl, tgt_pos, tgt_vel, ep_key)
            # Tracking: mean L1 distance, weighted by temporal ramp
            l1 = jnp.sum(jnp.abs(eff_traj - tgt_pos), axis=-1)  # (T,)
            time_w = jnp.linspace(0.5, 1.5, N_STEPS)
            tracking = jnp.mean(l1 * time_w)
            # Effort
            effort = jnp.mean(act_traj ** 2)
            # Smoothness (activation jerk)
            d_act = jnp.diff(act_traj, axis=0)
            dd_act = jnp.diff(d_act, axis=0)
            smoothness = jnp.mean(dd_act ** 2)
            total = tracking + cfg.effort_weight * effort + 0.001 * smoothness
            return total, (tracking, effort, smoothness)

        results = jax.vmap(_single)(target_pos_batch, target_vel_batch, batch_keys)
        totals, (trackings, efforts, smoothnesses) = results
        mean_total = jnp.mean(totals)
        mean_tracking = jnp.mean(trackings)
        mean_effort = jnp.mean(efforts)
        mean_smoothness = jnp.mean(smoothnesses)
        return mean_total, {
            "tracking": mean_tracking,
            "effort": mean_effort,
            "smoothness": mean_smoothness,
            "hidden_reg": jnp.float32(0.0),
        }

    # ------------------------------------------------------------------
    # JIT-compiled training step
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def _train_step(ctrl, opt_st, target_pos_batch, target_vel_batch, step_key):
        batch_keys = jr.split(step_key, cfg.batch_size)
        (loss, terms), grads = eqx.filter_value_and_grad(_loss_fn, has_aux=True)(
            ctrl, target_pos_batch, target_vel_batch, batch_keys,
        )
        grad_norm = optax.global_norm(grads)
        updates, new_opt_st = optimizer.update(grads, opt_st, ctrl)
        new_ctrl = eqx.apply_updates(ctrl, updates)
        return new_ctrl, new_opt_st, loss, terms, grad_norm

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    t_start = time.perf_counter()
    snapshot_interval = cfg.snapshot_interval

    for batch in range(job.total_batches):
        if job.stop_event.is_set():
            job.status = WorkerStatus.IDLE
            return

        rng_key, batch_key, step_key = jr.split(rng_key, 3)

        # Sample targets
        target_pos_batch, target_vel_batch = _sample_targets(batch_key, cfg.batch_size)

        step_t0 = time.perf_counter()
        try:
            controller, opt_state, loss_val, loss_terms, grad_norm = _train_step(
                controller, opt_state,
                target_pos_batch, target_vel_batch,
                step_key,
            )
            # Block until JAX computation is complete for accurate timing.
            loss_val = float(jax.block_until_ready(loss_val))
        except Exception as exc:
            job.status = WorkerStatus.ERROR
            _emit(
                job,
                {
                    "type": "training_error",
                    "job_id": job.job_id,
                    "error": f"JAX training error at batch {batch + 1}: {exc}",
                },
            )
            return

        step_time_ms = (time.perf_counter() - step_t0) * 1000.0

        job.last_loss = loss_val
        job.batch = batch + 1

        loss_terms_out = {
            "tracking": float(loss_terms["tracking"]),
            "effort": float(loss_terms["effort"]),
            "smoothness": float(loss_terms["smoothness"]),
            "hidden_reg": 0.0,
        }
        grad_norm_val = float(grad_norm)
        log_line = (
            f"Step {batch + 1} | loss={loss_val:.4f} | "
            f"grad_norm={grad_norm_val:.3f} | "
            f"{step_time_ms:.0f}ms"
        )

        # Progress event
        _emit(
            job,
            {
                "type": "training_progress",
                "job_id": job.job_id,
                "batch": batch + 1,
                "total_batches": job.total_batches,
                "loss": loss_val,
                "loss_terms": loss_terms_out,
                "grad_norm": grad_norm_val,
                "step_time_ms": step_time_ms,
                "status": "running",
            },
        )
        # Log event
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": batch + 1,
                "level": "info",
                "message": log_line,
            },
        )

        # Trajectory snapshot
        if (batch + 1) % snapshot_interval == 0:
            try:
                # Eval rollout: use a fixed target for visualization
                eval_key = jr.PRNGKey(batch)
                tgt_pos_const = jnp.broadcast_to(
                    jnp.array([0.25, 0.25]), (N_STEPS, 2),
                )
                tgt_vel_const = jnp.zeros((N_STEPS, 2))
                eff_traj, _, _ = _rollout(
                    controller, tgt_pos_const, tgt_vel_const, eval_key,
                )
                eff_traj_np = np.array(jax.block_until_ready(eff_traj))
                t_axis = np.linspace(0.0, N_STEPS * CONTROL_DT, N_STEPS).tolist()
                effector_list = eff_traj_np.tolist()
                _emit(
                    job,
                    {
                        "type": "training_trajectory",
                        "job_id": job.job_id,
                        "batch": batch + 1,
                        "trajectory": {
                            "effector": effector_list,
                            "target": [0.25, 0.25],
                            "t": t_axis,
                            "n_steps": N_STEPS,
                        },
                    },
                )
            except Exception:
                # Non-fatal: fall back to synthetic snapshot
                _emit(job, _make_trajectory_event(job, batch + 1, loss_val))

    job.status = WorkerStatus.COMPLETED

    # Serialize the trained controller to disk before emitting the terminal event.
    try:
        import os as _os
        import tempfile as _tmpfile

        _ckpt_dir = _tmpfile.mkdtemp(prefix="feedbax_ckpt_")
        _ckpt_path = _os.path.join(_ckpt_dir, f"{job.job_id}.eqx")
        ready_controller = jax.block_until_ready(controller)
        eqx.tree_serialise_leaves(_ckpt_path, ready_controller)
        job.checkpoint_path = _ckpt_path
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": job.total_batches,
                "level": "info",
                "message": "Checkpoint saved",
            },
        )
    except Exception as _exc:
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": job.total_batches,
                "level": "warning",
                "message": f"Failed to save checkpoint: {_exc}",
            },
        )

    _emit(
        job,
        {
            "type": "training_complete",
            "job_id": job.job_id,
            "batch": job.total_batches,
            "loss": job.last_loss,
        },
    )


# ---------------------------------------------------------------------------
# Stub training loop (fallback when real training is unavailable)
# ---------------------------------------------------------------------------


def _run_training_stub(job: _Job) -> None:
    """Synthetic training loop — runs in a background thread."""
    start_loss = 1.0
    for batch in range(job.total_batches):
        if job.stop_event.is_set():
            job.status = WorkerStatus.IDLE
            return

        time.sleep(0.05)

        decay = 0.98 ** batch
        loss = start_loss * decay
        job.last_loss = loss
        job.batch = batch + 1

        noise = lambda: random.uniform(-0.005, 0.005)
        loss_terms = {
            "tracking": max(0.0, 0.70 * loss + noise()),
            "effort": max(0.0, 0.20 * loss + noise()),
            "smoothness": max(0.0, 0.07 * loss + noise()),
            "hidden_reg": max(0.0, 0.03 * loss + noise()),
        }
        grad_norm = max(0.01, 1.0 * decay + random.uniform(-0.02, 0.02))
        step_time_ms = random.uniform(30.0, 60.0)
        log_line = f"Step {batch + 1} | loss={loss:.4f} | grad_norm={grad_norm:.3f}"

        _emit(
            job,
            {
                "type": "training_progress",
                "job_id": job.job_id,
                "batch": batch + 1,
                "total_batches": job.total_batches,
                "loss": loss,
                "loss_terms": loss_terms,
                "grad_norm": grad_norm,
                "step_time_ms": step_time_ms,
                "status": "running",
            },
        )
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": batch + 1,
                "level": "info",
                "message": log_line,
            },
        )

        if (batch + 1) % job.snapshot_interval == 0:
            _emit(job, _make_trajectory_event(job, batch + 1, loss))

    job.status = WorkerStatus.COMPLETED
    _emit(
        job,
        {
            "type": "training_complete",
            "job_id": job.job_id,
            "batch": job.total_batches,
            "loss": job.last_loss,
        },
    )


def _run_training(job: _Job) -> None:
    """Training entry point. Always attempts real JAX training.

    Only falls back to the synthetic stub on exception — so the worker never
    crashes the SSE stream. When ``training_config`` is ``None``, defaults from
    ``_TrainingCfg`` are used for real training.
    """
    try:
        cfg = _extract_training_cfg(job.training_config, job.task_spec)
        _run_training_real(job, cfg)
    except Exception as exc:
        # Real training raised an unexpected exception — fall back to stub
        # only if the stream hasn't terminated yet (status still RUNNING).
        if job.status == WorkerStatus.RUNNING:
            _emit(
                job,
                {
                    "type": "training_log",
                    "job_id": job.job_id,
                    "batch": job.batch,
                    "level": "warning",
                    "message": (
                        f"Real JAX training failed ({exc}); "
                        "falling back to synthetic stub."
                    ),
                },
            )
            _run_training_stub(job)
    finally:
        # Sentinel: tells SSE generator the stream is done.
        job.event_queue.put(None)


def _emit(job: _Job, event: dict) -> None:
    """Assign a seq number to *event*, buffer it, and enqueue it for SSE delivery."""
    seq = job.next_seq()
    event["seq"] = seq
    job.event_buffer.append((seq, event))
    job.event_queue.put(event)


def create_app(auth_token: Optional[str] = None) -> FastAPI:
    """Create and return the worker FastAPI application.

    Args:
        auth_token: Optional shared secret. When provided every request must
            include ``Authorization: Bearer <token>``; requests without it
            receive HTTP 401.
    """
    app = FastAPI(title="Feedbax Training Worker", version="0.1.0")

    # ------------------------------------------------------------------
    # Auth dependency
    # ------------------------------------------------------------------

    _bearer_scheme = HTTPBearer(auto_error=False)

    def _require_auth(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    ) -> None:
        """FastAPI dependency that enforces the bearer token when one is configured."""
        if auth_token is None:
            # Auth not configured — allow all requests.
            return
        if credentials is None or credentials.credentials != auth_token:
            raise HTTPException(status_code=401, detail="Unauthorized")

    # All routes share this dependency.
    _auth_dep = Depends(_require_auth)

    # ------------------------------------------------------------------
    # Module-level state for the single active job.
    # ------------------------------------------------------------------

    _state: Dict[str, Optional[_Job]] = {"current": None}

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", dependencies=[_auth_dep])
    def health():
        return {"status": "ok"}

    @app.post("/start", dependencies=[_auth_dep])
    def start(body: dict):
        total_batches = int(body.get("total_batches", 100))
        training_config: Optional[Dict[str, Any]] = body.get("training_config", None)
        training_spec: Optional[Dict[str, Any]] = body.get("training_spec", None)
        task_spec: Optional[Dict[str, Any]] = body.get("task_spec", None)
        graph_spec: Optional[Dict[str, Any]] = body.get("graph_spec", None)
        snapshot_interval = int(body.get("snapshot_interval", 100))

        job_id = str(uuid.uuid4())
        stop_event = threading.Event()
        event_queue: queue.Queue = queue.Queue()

        job = _Job(
            job_id=job_id,
            total_batches=total_batches,
            event_queue=event_queue,
            stop_event=stop_event,
            training_config=training_config,
            training_spec=training_spec,
            task_spec=task_spec,
            graph_spec=graph_spec,
            status=WorkerStatus.RUNNING,
            snapshot_interval=snapshot_interval,
        )
        thread = threading.Thread(target=_run_training, args=(job,), daemon=True)
        job.thread = thread
        _state["current"] = job
        thread.start()
        return {"job_id": job_id}

    @app.post("/stop", dependencies=[_auth_dep])
    def stop():
        job = _state.get("current")
        if job is not None:
            job.stop_event.set()
            job.status = WorkerStatus.IDLE
        return {"ok": True}

    @app.get("/status", dependencies=[_auth_dep])
    def status():
        job = _state.get("current")
        if job is None:
            return {
                "status": WorkerStatus.IDLE,
                "batch": 0,
                "total_batches": 0,
                "last_loss": 0.0,
            }
        return {
            "status": job.status,
            "batch": job.batch,
            "total_batches": job.total_batches,
            "last_loss": job.last_loss,
        }

    @app.get("/stream", dependencies=[_auth_dep])
    def stream(from_seq: Optional[int] = Query(default=None, alias="from_seq")):
        """SSE stream of training events for the current job.

        Args:
            from_seq: When provided, replay buffered events with seq >=
                *from_seq* before streaming live ones. Used by the client for
                reconnection.
        """
        job = _state.get("current")
        if job is None:
            # No job running; return an empty stream immediately.
            async def _empty():
                yield "data: {}\n\n"
            return StreamingResponse(_empty(), media_type="text/event-stream")

        # Collect any buffered events to replay before the live stream.
        replay_events: list[dict] = []
        if from_seq is not None:
            replay_events = [
                evt for seq, evt in job.event_buffer if seq >= from_seq
            ]

        async def _generate():
            loop = asyncio.get_running_loop()

            # --- Replay phase ---
            for event in replay_events:
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("training_complete", "training_error"):
                    return

            # --- Live streaming phase ---
            while True:
                try:
                    # Poll the thread-safe queue without blocking the event loop.
                    event = await loop.run_in_executor(
                        None, lambda: job.event_queue.get(timeout=1.0)
                    )
                except queue.Empty:
                    # Worker still alive; keep the connection open.
                    t = job.thread
                    if t is None or not t.is_alive():
                        break
                    continue

                if event is None:
                    # Sentinel: stream is finished.
                    break

                yield f"data: {json.dumps(event)}\n\n"

                # Stop streaming after the terminal events.
                if event.get("type") in ("training_complete", "training_error"):
                    break

        return StreamingResponse(_generate(), media_type="text/event-stream")

    @app.get("/checkpoint", dependencies=[_auth_dep])
    def checkpoint():
        """Return checkpoint metadata for the current job."""
        job = _state.get("current")
        if job is None:
            return {"batch": 0, "loss": 0.0, "weights_available": False}
        return {
            "batch": job.batch,
            "loss": job.last_loss,
            "weights_available": job.checkpoint_path is not None,
        }

    @app.get("/checkpoint/download", dependencies=[_auth_dep])
    def checkpoint_download():
        """Download the serialized checkpoint file for the current job."""
        import os

        job = _state.get("current")
        if job is None or job.checkpoint_path is None:
            raise HTTPException(status_code=404, detail="No checkpoint available")
        if not os.path.exists(job.checkpoint_path):
            raise HTTPException(status_code=410, detail="Checkpoint file gone")
        return FileResponse(
            job.checkpoint_path,
            media_type="application/octet-stream",
            filename=f"feedbax_checkpoint_{job.job_id}.eqx",
        )

    return app
