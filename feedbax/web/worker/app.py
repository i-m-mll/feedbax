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
    task_binding_spec: Optional[Dict[str, Any]] = None
    # Graph spec dict forwarded from the API layer for network param extraction.
    graph_spec: Optional[Dict[str, Any]] = None
    # Path to the serialized checkpoint file after training completes.
    checkpoint_path: Optional[str] = None
    # Path/payload for the durable manifest emitted after training completes.
    manifest_path: Optional[str] = None
    manifest_payload: Optional[Dict[str, Any]] = None
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


def _manifest_history_events(job: _Job) -> list[dict[str, Any]]:
    """Return compact event history suitable for a durable JSON artifact."""
    history_types = {"training_progress", "training_log", "training_error", "training_complete"}
    return [dict(event) for _, event in job.event_buffer if event.get("type") in history_types]


def _write_job_manifest(job: _Job) -> None:
    """Write a durable training-run manifest for a completed worker job."""
    try:
        from feedbax.manifest import write_training_run_manifest

        manifest, path = write_training_run_manifest(
            job_id=job.job_id,
            total_batches=job.total_batches,
            training_spec=job.training_spec,
            task_spec=job.task_spec,
            task_binding_spec=job.task_binding_spec,
            graph_spec=job.graph_spec,
            checkpoint_path=job.checkpoint_path,
            history_events=_manifest_history_events(job),
            status=job.status.value,
            final_loss=job.last_loss,
        )
        job.manifest_path = str(path)
        job.manifest_payload = manifest.model_dump(mode="json", exclude_none=True)
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": job.batch,
                "level": "info",
                "message": "Training manifest saved",
                "manifest_path": job.manifest_path,
                "manifest_id": manifest.id,
            },
        )
    except Exception as exc:
        _emit(
            job,
            {
                "type": "training_log",
                "job_id": job.job_id,
                "batch": job.batch,
                "level": "warning",
                "message": f"Failed to save training manifest: {exc}",
            },
        )


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


@dataclass(frozen=True)
class _TaskSamplingCfg:
    """Compact task-sampling settings derived from scenario-owned task_spec."""

    task_type: str
    workspace_min: tuple[float, float]
    workspace_max: tuple[float, float]
    train_endpoint_mode: str = "workspace"
    reach_length: float = 0.25
    p_catch_trial: float = 0.0
    epoch_len_ranges: tuple[tuple[int, int], ...] = ()
    hold_epochs: tuple[int, ...] = ()
    move_epochs: tuple[int, ...] = ()


def _as_mapping(name: str, value: Any) -> Dict[str, Any]:
    """Return *value* as a dict or raise a clear worker-spec error."""
    if not isinstance(value, dict):
        raise ValueError(f"Training worker requires {name} to be an object")
    return value


def _require_worker_specs(job: _Job) -> None:
    """Validate the Studio payload shape required by the real worker path."""
    _as_mapping("training_spec", job.training_spec)
    _as_mapping("task_spec", job.task_spec)
    _as_mapping("graph_spec", job.graph_spec)
    if job.task_binding_spec is None:
        raise ValueError(
            "Training worker requires scenario-owned task_binding_spec; "
            "task data bindings must not be inferred from graph task nodes"
        )
    task_binding_spec = _as_mapping("task_binding_spec", job.task_binding_spec)
    if task_binding_spec.get("schema_version") != "feedbax.studio.task_bindings.v2":
        raise ValueError("Training worker requires task_binding_spec schema v2")
    if "exposed_outputs" in task_binding_spec:
        raise ValueError("task_binding_spec.exposed_outputs is not accepted; use exposed_data")
    for index, binding in enumerate(task_binding_spec.get("bindings", [])):
        if not isinstance(binding, dict):
            raise ValueError(f"task_binding_spec.bindings[{index}] must be an object")
        if "source_output_id" in binding or "source_data_id" not in binding:
            raise ValueError(
                "task_binding_spec bindings must use source_data_id; "
                f"invalid binding at index {index}"
            )


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
        for key, attr, cast in [
            ("n_reach_steps", "n_reach_steps", int),
            ("n_steps", "n_reach_steps", int),
            ("effort_weight", "effort_weight", float),
        ]:
            if key in task_params:
                try:
                    setattr(cfg, attr, cast(task_params[key]))
                except (TypeError, ValueError):
                    pass

    return cfg


def _extract_task_sampling_cfg(task_spec: Dict[str, Any]) -> _TaskSamplingCfg:
    """Derive compact reach sampling settings from a Studio task spec.

    The worker consumes scenario-owned task params only. Dense trajectories are
    intentionally rejected at the provider boundary and are not accepted here.
    """
    task_type = str(task_spec.get("type", ""))
    params = _as_mapping("task_spec.params", task_spec.get("params", {}))
    normalized_type = task_type.removeprefix("feedbax.task.")
    supported = {"ReachingTask", "SimpleReaches", "DelayedReaches"}
    if normalized_type not in supported:
        raise ValueError(
            f"Training worker does not support task type {task_type!r}; "
            f"supported task types are {sorted(supported)}"
        )
    dense_keys = [
        key for key in ("targets", "target_pos", "target_vel", "validation_trials") if key in params
    ]
    if dense_keys:
        raise ValueError(
            "Training worker accepts compact task params only; remove dense "
            f"trajectory fields {dense_keys}"
        )

    workspace = params.get("workspace")
    if workspace is None:
        target_radius = params.get("target_radius")
        if target_radius is None:
            raise ValueError(
                "task_spec.params must include workspace or target_radius for worker sampling"
            )
        radius = float(target_radius)
        workspace = [[-radius, -radius], [radius, radius]]
    elif (
        not isinstance(workspace, list)
        or len(workspace) != 2
        or not all(isinstance(item, list) and len(item) == 2 for item in workspace)
    ):
        raise ValueError("task_spec.params.workspace must be [[xmin, ymin], [xmax, ymax]]")
    workspace_min = (float(workspace[0][0]), float(workspace[0][1]))
    workspace_max = (float(workspace[1][0]), float(workspace[1][1]))
    if workspace_min[0] >= workspace_max[0] or workspace_min[1] >= workspace_max[1]:
        raise ValueError("task_spec.params.workspace min bounds must be below max bounds")

    train_endpoint_mode = str(params.get("train_endpoint_mode", "workspace"))
    if train_endpoint_mode not in {"workspace", "center_out"}:
        raise ValueError("task_spec.params.train_endpoint_mode must be 'workspace' or 'center_out'")

    workspace_extent = min(
        workspace_max[0] - workspace_min[0],
        workspace_max[1] - workspace_min[1],
    )
    reach_length = params.get(
        "eval_reach_length",
        params.get("target_radius", 0.25 * workspace_extent),
    )
    p_catch_trial = params.get("p_catch_trial", 0.0)

    epoch_len_ranges: list[tuple[int, int]] = []
    raw_epoch_ranges = params.get("epoch_len_ranges", [])
    if raw_epoch_ranges:
        if not isinstance(raw_epoch_ranges, list):
            raise ValueError("task_spec.params.epoch_len_ranges must be a list")
        for index, item in enumerate(raw_epoch_ranges):
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"task_spec.params.epoch_len_ranges[{index}] must be [min, max]")
            lower = int(item[0])
            upper = int(item[1])
            if lower < 0 or upper < lower:
                raise ValueError(
                    "task_spec.params.epoch_len_ranges entries must satisfy 0 <= min <= max"
                )
            epoch_len_ranges.append((lower, upper))

    def _epoch_list(key: str) -> tuple[int, ...]:
        value = params.get(key, [])
        if not isinstance(value, list):
            raise ValueError(f"task_spec.params.{key} must be a list of epoch indexes")
        return tuple(int(item) for item in value)

    return _TaskSamplingCfg(
        task_type=normalized_type,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        train_endpoint_mode=train_endpoint_mode,
        reach_length=float(reach_length),
        p_catch_trial=float(p_catch_trial),
        epoch_len_ranges=tuple(epoch_len_ranges),
        hold_epochs=_epoch_list("hold_epochs"),
        move_epochs=_epoch_list("move_epochs"),
    )


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

    Reads from the Network node's internal subgraph (the authoritative source
    per the graph-as-model principle). Raises ValueError if the subgraph is
    absent — incomplete model state is an error, not a condition to work around.

    Also extracts ``input_size`` from the outer Network node params (the
    subgraph cell's input_size is an internal wiring detail; the outer
    Network param is the canonical interface dimension), and ``dt`` from the
    first mechanics/plant node found in the top-level graph.

    Returns a dict with keys:
        hidden_type: equinox cell class
        hidden_size: hidden state dimension
        out_size: output dimension
        out_nonlinearity: activation callable
        input_size: network input dimension
        dt: control timestep in seconds
        plant_type: mechanics node type string
    """
    import equinox as eqx
    import jax

    CELL_MAP = {
        "GRU": eqx.nn.GRUCell,
        "LSTM": eqx.nn.LSTMCell,
        "GRUCell": eqx.nn.GRUCell,
        "LSTMCell": eqx.nn.LSTMCell,
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
    _MECHANICS_NODE_TYPES = frozenset(
        {
            "TwoLinkArm",
            "PointMass",
            "Mechanics",
            "Arm6MuscleRigidTendon",
            "PointMass8MuscleRelu",
            "AcausalSystem",
        }
    )

    if graph_spec is None:
        raise ValueError("Training worker requires graph_spec")

    nodes = graph_spec.get("nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError("graph_spec.nodes must be an object")

    mechanics_node = next(
        (n for n in nodes.values() if n.get("type") in _MECHANICS_NODE_TYPES),
        None,
    )
    if mechanics_node is None:
        raise ValueError("graph_spec must include a supported mechanics node")
    mech_params = mechanics_node.get("params", {})
    plant_type = mechanics_node.get("type")
    if plant_type is None:
        raise ValueError("mechanics node is missing type")
    try:
        dt = float(mech_params["dt"])
    except KeyError as exc:
        raise ValueError("mechanics node params must include dt") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("mechanics node dt must be numeric") from exc

    network_node_id, network_node = next(
        (
            (nid, n)
            for nid, n in nodes.items()
            if n.get("type") in {"Network", "SimpleStagedNetwork"}
        ),
        (None, None),
    )
    if network_node_id is None or network_node is None:
        raise ValueError("graph_spec must include a Network or SimpleStagedNetwork node")

    result = {
        "hidden_type": eqx.nn.GRUCell,
        "hidden_size": 128,
        "out_size": 6,
        "out_nonlinearity": jax.nn.sigmoid,
        "input_size": 17,
        "dt": dt,
        "plant_type": plant_type,
    }

    outer_params = network_node.get("params", {})
    network_type = network_node.get("type")
    if network_type == "SimpleStagedNetwork":
        hidden_key = outer_params.get("hidden_type", "GRUCell")
        if hidden_key not in CELL_MAP:
            raise ValueError(f"Unsupported SimpleStagedNetwork hidden_type {hidden_key!r}")
        nonlin_key = outer_params.get("out_nonlinearity", "identity")
        if nonlin_key not in NONLINEARITY_MAP:
            raise ValueError(f"Unsupported SimpleStagedNetwork out_nonlinearity {nonlin_key!r}")
        try:
            result.update(
                {
                    "hidden_type": CELL_MAP[hidden_key],
                    "hidden_size": int(outer_params["hidden_size"]),
                    "out_size": int(outer_params["out_size"]),
                    "out_nonlinearity": NONLINEARITY_MAP[nonlin_key],
                    "input_size": int(outer_params["input_size"]),
                }
            )
        except KeyError as exc:
            raise ValueError(
                f"SimpleStagedNetwork node {network_node_id!r} is missing param {exc.args[0]!r}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("SimpleStagedNetwork size params must be integers") from exc
        return result

    # ------------------------------------------------------------------
    # Authoritative source: reading hidden/output architecture from the
    # Network node's internal subgraph. The subgraph IS the model; outer
    # params are UI defaults only and are ignored if the subgraph exists.
    # ------------------------------------------------------------------
    subgraphs = graph_spec.get("subgraphs") or {}
    network_subgraph = subgraphs.get(network_node_id)

    if network_subgraph is None:
        raise ValueError(
            f"Network node {network_node_id!r} has no subgraph. "
            "Open it in Studio to generate the internal architecture, then save again."
        )

    sub_nodes = network_subgraph.get("nodes", {})
    # Find the hidden cell node (GRU or LSTM)
    cell_node = next(
        (n for n in sub_nodes.values() if n.get("type") in ("GRU", "LSTM")),
        None,
    )
    if cell_node is None:
        raise ValueError(
            f"Network node {network_node_id!r} subgraph is missing a GRU/LSTM cell node"
        )
    output_bindings = network_subgraph.get("output_bindings", {})
    if "output" not in output_bindings:
        raise ValueError(
            f"Network node {network_node_id!r} subgraph must bind output to a readout node"
        )
    readout_node_name = output_bindings["output"][0]

    readout_node = sub_nodes.get(readout_node_name)
    if readout_node is None:
        raise ValueError(
            f"Network node {network_node_id!r} output binding references missing "
            f"node {readout_node_name!r}"
        )

    cell_type = cell_node.get("type", "GRU")
    if cell_type not in CELL_MAP:
        raise ValueError(f"Unsupported Network cell type {cell_type!r}")
    cell_params = cell_node.get("params", {})
    readout_params = readout_node.get("params", {})
    nonlin_key = readout_params.get("activation", "identity")
    if nonlin_key not in NONLINEARITY_MAP:
        raise ValueError(f"Unsupported Network readout activation {nonlin_key!r}")
    try:
        result["input_size"] = int(outer_params["input_size"])
        result["hidden_type"] = CELL_MAP[cell_type]
        result["hidden_size"] = int(cell_params["hidden_size"])
        result["out_size"] = int(readout_params["output_size"])
        result["out_nonlinearity"] = NONLINEARITY_MAP[nonlin_key]
    except KeyError as exc:
        raise ValueError(f"Network graph params are missing {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("Network graph size params must be integers") from exc

    return result


# ---------------------------------------------------------------------------
# Real JAX training backend
# ---------------------------------------------------------------------------


def _run_training_real(job: _Job, cfg: "_TrainingCfg") -> None:
    """Real JAX training loop dispatching on the graph spec's mechanics node type.

    Supports TwoLinkArm/Arm6MuscleRigidTendon (AnalyticalMusculoskeletalPlant)
    and PointMass (DirectForceInput(PointMass)).

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
        import jax.tree as jt
        import optax

        from feedbax.mechanics.backend import DiffraxBackend, PhysicsState
        from feedbax.mechanics.body import (
            BodyPreset,
            default_2link_bounds,
        )
        from feedbax.mechanics.analytical_plant import AnalyticalMusculoskeletalPlant
        from feedbax.mechanics.model_builder import ChainConfig
        from feedbax.mechanics.muscle_config import default_6muscle_2link_topology
        from feedbax.mechanics.plant import DirectForceInput
        from feedbax.mechanics.skeleton.pointmass import PointMass
        from feedbax.nn import SimpleStagedNetwork
        from feedbax.training.rl.tasks import reach_task_params, target_at_t

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

    graph_params = _extract_graph_params(job.graph_spec)
    task_sampling_cfg = _extract_task_sampling_cfg(_as_mapping("task_spec", job.task_spec))

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    CONTROL_DT = graph_params["dt"]
    N_STEPS = cfg.n_reach_steps  # control steps per episode
    OBS_DIM = graph_params["input_size"]

    plant_type = graph_params["plant_type"]

    # ------------------------------------------------------------------
    # Build plant — dispatch on mechanics node type
    # ------------------------------------------------------------------

    rng_key = jr.PRNGKey(0)
    preset_key, ctrl_key, rng_key = jr.split(rng_key, 3)

    _ARM_TYPES = {"TwoLinkArm", "Arm6MuscleRigidTendon"}
    _POINTMASS_TYPES = {"PointMass", "PointMass8MuscleRelu"}

    if plant_type in _ARM_TYPES:
        # 6-muscle, 2-joint analytical musculoskeletal plant (original path).
        N_MUSCLES = 6
        N_JOINTS = 2
        _is_pointmass = False

        bounds = default_2link_bounds()
        preset = BodyPreset(
            segment_lengths=0.5 * (bounds.segment_lengths_min + bounds.segment_lengths_max),
            segment_masses=0.5 * (bounds.segment_masses_min + bounds.segment_masses_max),
            joint_damping=0.5 * (bounds.joint_damping_min + bounds.joint_damping_max),
            joint_stiffness=0.5 * (bounds.joint_stiffness_min + bounds.joint_stiffness_max),
            muscle_pcsa=0.5 * (bounds.muscle_pcsa_min + bounds.muscle_pcsa_max),
            muscle_optimal_fiber_length=0.5
            * (bounds.muscle_optimal_fiber_length_min + bounds.muscle_optimal_fiber_length_max),
            muscle_tendon_slack_length=0.5
            * (bounds.muscle_tendon_slack_length_min + bounds.muscle_tendon_slack_length_max),
            muscle_moment_arm_magnitudes=0.5
            * (bounds.muscle_moment_arm_magnitudes_min + bounds.muscle_moment_arm_magnitudes_max),
        )

        topology = default_6muscle_2link_topology()
        chain_config = ChainConfig(n_joints=N_JOINTS, muscle_topology=topology)

        plant = AnalyticalMusculoskeletalPlant.from_body_preset(
            preset,
            chain_config,
            clip_states=True,
        )

    elif plant_type in _POINTMASS_TYPES:
        if plant_type == "PointMass8MuscleRelu":
            raise ValueError(
                "PointMass8MuscleRelu requires explicit muscle wiring and is not "
                "supported by the direct-force training bridge"
            )
        N_MUSCLES = 2
        N_JOINTS = 0
        _is_pointmass = True

        plant = DirectForceInput(PointMass(mass=1.0))
        # PointMass uses cached JAX arrays; warm them outside JIT so transformed
        # rollouts do not cache tracers on the skeleton instance.
        _ = plant.skeleton.A, plant.skeleton.B, plant.skeleton.C, plant.skeleton._lti_system

    else:
        raise ValueError(f"Unsupported mechanics node type {plant_type!r}")

    if graph_params["out_size"] != N_MUSCLES:
        raise ValueError(
            f"Network output size {graph_params['out_size']} does not match "
            f"{plant_type} action size {N_MUSCLES}"
        )
    if _is_pointmass and OBS_DIM not in {4, 13}:
        raise ValueError(
            f"PointMass worker bridge supports network input_size 4 or 13, got {OBS_DIM}"
        )
    if not _is_pointmass and OBS_DIM != 17:
        raise ValueError(f"Arm worker bridge supports network input_size 17, got {OBS_DIM}")

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

    cfg.effort_weight = _extract_effort_weight_from_spec(job.training_spec, cfg.effort_weight)
    optimizer = _build_optimizer_from_spec(job.training_spec, cfg)

    # ------------------------------------------------------------------
    # Optimizer state
    # ------------------------------------------------------------------

    opt_state = optimizer.init(eqx.filter(controller, eqx.is_array))

    # ------------------------------------------------------------------
    # Observation helper
    # ------------------------------------------------------------------

    # Bug: cb13bdc — observation layout differs between plant types.
    def _extract_obs(
        physics_state: PhysicsState,
        action_or_activation,
        target_pos,
        target_vel,
        phase,
    ):
        """Extract observation vector, dispatcher for PointMass vs articulated."""
        sk = physics_state.plant.skeleton
        effector = physics_state.effector
        if _is_pointmass:
            # PointMass skeleton state is CartesianState with pos(2), vel(2).
            # No joint angles or muscle activations.
            if OBS_DIM == 4:
                return jnp.concatenate([effector.pos, target_pos])
            return jnp.concatenate(
                [
                    sk.pos,  # config pos (2)
                    sk.vel,  # config vel (2)
                    action_or_activation,  # last action / 2D force (2)
                    effector.pos,  # effector pos (2)
                    target_pos,  # (2)
                    target_vel,  # (2)
                    phase,  # (1)
                ]
            )
        else:
            # Articulated plant: skeleton has angle, d_angle, etc.
            return jnp.concatenate(
                [
                    sk.angle,
                    sk.d_angle,
                    action_or_activation,  # muscle_activations
                    effector.pos,
                    target_pos,
                    target_vel,
                    phase,
                ]
            )

    # ------------------------------------------------------------------
    # Single-episode rollout through Diffrax (differentiable)
    # ------------------------------------------------------------------

    def _rollout(ctrl, task_params, episode_key):
        phys = backend.init_state(plant, key=episode_key)
        init_act = jnp.zeros(N_MUSCLES)
        init_phase = jnp.zeros(1)
        init_target_pos, init_target_vel = target_at_t(
            task_params,
            jnp.array(0, dtype=jnp.int32),
        )
        init_obs = _extract_obs(
            phys,
            init_act,
            init_target_pos,
            init_target_vel,
            init_phase,
        )
        # Controller hidden state: initialize to zeros
        init_hidden = jnp.zeros(hidden_size)

        scan_keys = jr.split(episode_key, N_STEPS)

        def _step(carry, inputs):
            t_idx, _step_key = inputs
            phys_s, act, hidden, _obs_prev = carry

            phase = jnp.array([t_idx / N_STEPS])
            target_pos, target_vel = target_at_t(task_params, t_idx)
            obs = _extract_obs(
                phys_s,
                act,
                target_pos,
                target_vel,
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
            output = (new_effector.pos, action, new_hidden, target_pos)
            return new_carry, output

        init_carry = (phys, init_act, init_hidden, init_obs)
        t_idxs = jnp.arange(N_STEPS)
        _, (eff_traj, act_traj, hidden_traj, target_pos_traj) = jax.lax.scan(
            _step,
            init_carry,
            (t_idxs, scan_keys),
        )
        return eff_traj, act_traj, hidden_traj, target_pos_traj

    # ------------------------------------------------------------------
    # Target sampling from scenario-owned compact task params
    # ------------------------------------------------------------------

    def _sample_task_params(batch_key, batch_size):
        """Sample compact reach task parameter records for a training batch.

        Returns:
            Batched TaskParams. Endpoint/timing records are stored eagerly;
            dense target trajectories are generated inside rollout scans.
        """
        keys = jr.split(batch_key, batch_size)
        workspace_min = jnp.asarray(task_sampling_cfg.workspace_min)
        workspace_max = jnp.asarray(task_sampling_cfg.workspace_max)
        workspace_center = 0.5 * (workspace_min + workspace_max)
        reach_length = jnp.asarray(task_sampling_cfg.reach_length)
        p_catch_trial = jnp.asarray(task_sampling_cfg.p_catch_trial)
        epoch_len_ranges = tuple(task_sampling_cfg.epoch_len_ranges)
        hold_epochs = set(task_sampling_cfg.hold_epochs)
        first_move_epoch = (
            min(task_sampling_cfg.move_epochs)
            if task_sampling_cfg.move_epochs
            else len(epoch_len_ranges)
        )

        def _one_task(k):
            k_endpoint, k_angle, k_catch, *epoch_keys = jr.split(
                k, max(4, len(epoch_len_ranges) + 3)
            )
            if task_sampling_cfg.train_endpoint_mode == "center_out":
                angle = jr.uniform(k_angle, minval=0.0, maxval=2.0 * jnp.pi)
                start = workspace_center
                end = start + reach_length * jnp.array([jnp.cos(angle), jnp.sin(angle)])
                end = jnp.clip(end, workspace_min, workspace_max)
            else:
                endpoint_pair = jr.uniform(
                    k_endpoint,
                    shape=(2, 2),
                    minval=workspace_min,
                    maxval=workspace_max,
                )
                start = endpoint_pair[0]
                end = endpoint_pair[1]

            is_catch = jr.uniform(k_catch) < p_catch_trial
            end = jnp.where(is_catch, start, end)

            delay_steps = 0
            for index, (lower, upper) in enumerate(epoch_len_ranges):
                if index in hold_epochs and index < first_move_epoch:
                    sampled = jr.randint(
                        epoch_keys[index],
                        shape=(),
                        minval=lower,
                        maxval=upper + 1,
                    )
                    delay_steps = delay_steps + sampled

            task_params = reach_task_params(start, end, N_STEPS, CONTROL_DT)
            if task_sampling_cfg.task_type == "DelayedReaches":
                delay_steps = jnp.minimum(delay_steps, jnp.maximum(N_STEPS - 2, 0))
                t0 = CONTROL_DT * delay_steps
                task_params = task_params._replace(t0=t0)
            return task_params

        return jax.vmap(_one_task)(keys)

    # ------------------------------------------------------------------
    # Supervised loss
    # ------------------------------------------------------------------

    def _loss_fn(ctrl, task_params_batch, batch_keys):
        """Mean supervised loss over a batch of episodes."""

        def _single(task_params, ep_key):
            eff_traj, act_traj, _, target_pos_traj = _rollout(ctrl, task_params, ep_key)
            # Tracking: mean L1 distance, weighted by temporal ramp
            l1 = jnp.sum(jnp.abs(eff_traj - target_pos_traj), axis=-1)  # (T,)
            time_w = jnp.linspace(0.5, 1.5, N_STEPS)
            tracking = jnp.mean(l1 * time_w)
            # Effort
            effort = jnp.mean(act_traj**2)
            # Smoothness (activation jerk)
            d_act = jnp.diff(act_traj, axis=0)
            dd_act = jnp.diff(d_act, axis=0)
            smoothness = jnp.mean(dd_act**2)
            total = tracking + cfg.effort_weight * effort + 0.001 * smoothness
            return total, (tracking, effort, smoothness)

        results = jax.vmap(_single)(task_params_batch, batch_keys)
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
    def _train_step(ctrl, opt_st, task_params_batch, step_key):
        batch_keys = jr.split(step_key, cfg.batch_size)
        (loss, terms), grads = eqx.filter_value_and_grad(_loss_fn, has_aux=True)(
            ctrl,
            task_params_batch,
            batch_keys,
        )
        grad_norm = optax.global_norm(grads)
        updates, new_opt_st = optimizer.update(grads, opt_st, ctrl)
        new_ctrl = eqx.apply_updates(ctrl, updates)
        return new_ctrl, new_opt_st, loss, terms, grad_norm

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    snapshot_interval = cfg.snapshot_interval

    for batch in range(job.total_batches):
        if job.stop_event.is_set():
            job.status = WorkerStatus.IDLE
            return

        rng_key, batch_key, step_key = jr.split(rng_key, 3)

        # Sample compact task params. Dense target trajectories are materialized
        # inside the per-episode rollout scan where each timestep needs them.
        task_params_batch = _sample_task_params(batch_key, cfg.batch_size)

        step_t0 = time.perf_counter()
        try:
            controller, opt_state, loss_val, loss_terms, grad_norm = _train_step(
                controller,
                opt_state,
                task_params_batch,
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
                eval_key = jr.PRNGKey(batch)
                eval_task = _sample_task_params(eval_key, 1)
                single_eval_task = jt.map(lambda x: x[0], eval_task)
                eff_traj, _, _, target_pos_traj = _rollout(
                    controller,
                    single_eval_task,
                    eval_key,
                )
                eff_traj_np = np.array(jax.block_until_ready(eff_traj))
                target_traj_np = np.array(jax.block_until_ready(target_pos_traj))
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
                            "target": target_traj_np[-1].tolist(),
                            "target_trajectory": target_traj_np.tolist(),
                            "t": t_axis,
                            "n_steps": N_STEPS,
                        },
                    },
                )
            except Exception as exc:
                _emit(
                    job,
                    {
                        "type": "training_log",
                        "job_id": job.job_id,
                        "batch": batch + 1,
                        "level": "warning",
                        "message": f"Failed to emit trajectory snapshot: {exc}",
                    },
                )

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

    _write_job_manifest(job)

    complete_event = {
        "type": "training_complete",
        "job_id": job.job_id,
        "batch": job.total_batches,
        "loss": job.last_loss,
    }
    if job.manifest_path is not None:
        complete_event["manifest_path"] = job.manifest_path
    if job.manifest_payload is not None:
        complete_event["manifest_id"] = job.manifest_payload.get("id")
    _emit(job, complete_event)


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

        decay = 0.98**batch
        loss = start_loss * decay
        job.last_loss = loss
        job.batch = batch + 1

        def noise() -> float:
            return random.uniform(-0.005, 0.005)

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
    _write_job_manifest(job)

    complete_event = {
        "type": "training_complete",
        "job_id": job.job_id,
        "batch": job.total_batches,
        "loss": job.last_loss,
    }
    if job.manifest_path is not None:
        complete_event["manifest_path"] = job.manifest_path
    if job.manifest_payload is not None:
        complete_event["manifest_id"] = job.manifest_payload.get("id")
    _emit(job, complete_event)


def _run_training(job: _Job) -> None:
    """Training entry point. Always attempts real JAX training.

    Invalid Studio payloads terminate with a ``training_error`` event instead
    of falling through to synthetic output.
    """
    try:
        _require_worker_specs(job)
        cfg = _extract_training_cfg(job.training_config, job.task_spec)
        _run_training_real(job, cfg)
    except Exception as exc:
        if job.status == WorkerStatus.RUNNING:
            job.status = WorkerStatus.ERROR
            _emit(
                job,
                {
                    "type": "training_error",
                    "job_id": job.job_id,
                    "batch": job.batch,
                    "error": str(exc),
                },
            )
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
        task_binding_spec: Optional[Dict[str, Any]] = body.get("task_binding_spec", None)
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
            task_binding_spec=task_binding_spec,
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
            "manifest_path": job.manifest_path,
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
            replay_events = [evt for seq, evt in job.event_buffer if seq >= from_seq]

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

    @app.get("/manifest", dependencies=[_auth_dep])
    def manifest():
        """Return the durable manifest for the current job."""
        job = _state.get("current")
        if job is None or job.manifest_payload is None:
            raise HTTPException(status_code=404, detail="No manifest available")
        return job.manifest_payload

    return app
