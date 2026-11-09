"""Graph and native model construction for configurable planar chains."""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import numpy as np

from feedbax.contracts.acausal import (
    AcausalConnectionSpec,
    AcausalGraphSpec,
    SolverConfigSpec,
    SolverType,
)
from feedbax.contracts.graph import ComponentSpec
from feedbax.mechanics.body import BodyPreset
from feedbax.mechanics.muscle_config import (
    MuscleConfig,
    MuscleTopology,
    default_muscle_config,
    default_monoarticular_topology,
)


class ChainConfig(eqx.Module):
    """Physical chain topology configuration.

    Attributes:
        n_joints: Number of rotational joints.
        muscle_topology: Static muscle-joint connectivity.  Defaults to a
            monoarticular layout with 2 muscles per joint if not provided.
    """

    n_joints: int = 3
    muscle_topology: Optional[MuscleTopology] = None

    def __post_init__(self):
        if self.muscle_topology is None:
            object.__setattr__(
                self,
                "muscle_topology",
                default_monoarticular_topology(self.n_joints),
            )

    @property
    def n_muscles(self) -> int:
        """Total number of muscles."""
        return self.muscle_topology.n_muscles


class SimConfig(eqx.Module):
    """MuJoCo simulation timing configuration.

    Attributes:
        dt: Physics timestep in seconds (default 0.002 = 500 Hz).
        episode_duration: Total episode length in seconds.
    """

    dt: float = 0.002
    episode_duration: float = 2.0


def planar_chain_graph(
    preset: BodyPreset,
    chain_config: ChainConfig,
    sim_config: SimConfig,
    muscle_config: MuscleConfig | None = None,
    *,
    solver_type: SolverType = "euler",
    joint_limits: tuple[float, float] | None = None,
) -> AcausalGraphSpec:
    """Author a planar chain from a morphology preset."""
    if chain_config.n_joints < 1:
        raise ValueError("A planar chain requires at least one joint")
    link_shape = (chain_config.n_joints,)
    if any(value.shape != link_shape for value in (
        preset.segment_lengths, preset.segment_masses, preset.joint_damping, preset.joint_stiffness,
    )):
        raise ValueError(f"Each physical preset vector must have shape {link_shape}")
    if muscle_config is None:
        muscle_config = default_muscle_config(preset, chain_config)
    muscle_shape = (muscle_config.n_muscles,)
    if muscle_config.n_joints != chain_config.n_joints or any(
        value.shape != muscle_shape for value in (
            preset.muscle_pcsa, preset.muscle_optimal_fiber_length, preset.muscle_tendon_slack_length,
        )
    ):
        raise ValueError("Muscle parameters and attachments must match the planar chain topology")

    def component(name: str, **params: object) -> ComponentSpec:
        return ComponentSpec(type=f"feedbax.{name}", params=params)

    nodes = {
        "world": component("WorldFrame"),
        "anchor": component("Anchor", world_frame="world.frame", frame="link0.proximal"),
    }
    edges = [(("world", "frame"), ("anchor", "world")),
             (("anchor", "frame"), ("link0", "proximal"))]
    for i in range(chain_config.n_joints):
        link, joint = f"link{i}", f"joint{i}"
        parent = ("world", "frame") if i == 0 else (f"link{i - 1}", "distal")
        length, mass = float(preset.segment_lengths[i]), float(preset.segment_masses[i])
        nodes[link] = component(
            "PlanarLink", length=length, mass=mass, com=0.5, inertia=mass * length**2 / 12,
        )
        limits = {"limit_min": joint_limits[0], "limit_max": joint_limits[1]} if joint_limits else {}
        nodes[joint] = component(
            "RevoluteJoint", parent_frame=".".join(parent), child_frame=f"{link}.proximal",
            damping=float(preset.joint_damping[i]), stiffness=float(preset.joint_stiffness[i]),
            **limits,
        )
        edges.extend(((parent, (joint, "parent")), ((joint, "child"), (link, "proximal"))))

    def point(body: str, position) -> dict:
        frame = "world.frame" if body == "world" else f"{body}.proximal"
        x, y, z = map(float, position)
        if z != 0:
            raise ValueError(f"Planar attachment on {body!r} requires zero Z offset")
        return {"frame": frame, "offset": [x, y]}

    for i in range(muscle_config.n_muscles):
        origin = point(muscle_config.origin_body[i], muscle_config.origin_pos[i])
        insertion = point(muscle_config.insertion_body[i], muscle_config.insertion_pos[i])
        optimal_length = float(preset.muscle_optimal_fiber_length[i])
        slack_length = float(preset.muscle_tendon_slack_length[i])
        nodes[f"muscle_{i}"] = component(
            "MusclePath", path_points=[origin, insertion],
            law="feedbax.RigidTendonHillMuscle", state_size=1,
            moment_arm=[float(value) for value in muscle_config.moment_arms[i]],
            reference_length=optimal_length + slack_length,
            max_isometric_force=float(preset.muscle_pcsa[i]) * 30,
            optimal_muscle_length=optimal_length, tendon_slack_length=slack_length,
            tau_act=preset.tau_act, tau_deact=preset.tau_deact,
        )

    nodes.update({
        "excitation": component(
            "ActuationInput", port_name="excitation", source_kind="muscle_excitation_vector",
        ),
        "effector": component("SensorOutput", port_name="effector", quantity="effector"),
        "state": component("SensorOutput", port_name="state", quantity="state"),
        "effector_marker": component(
            "PointMarker", frame=f"link{chain_config.n_joints - 1}.distal", marker_name="effector",
        ),
    })
    return AcausalGraphSpec(
        physical_domain="planar_multibody", nodes=nodes,
        connections=[AcausalConnectionSpec.model_validate({"a": a, "b": b}) for a, b in edges],
        solver=SolverConfigSpec(solver_type=solver_type, dt=sim_config.dt),
    )


def build_model(
    preset: BodyPreset,
    chain_config: ChainConfig,
    sim_config: SimConfig,
    muscle_config: MuscleConfig | None = None,
    differentiable: bool = False,
):
    """Build the native model through the same graph used for portable models."""
    from feedbax.mechanics.authored_model import compile_planar_model

    if differentiable:
        raise ValueError(
            "This preset has joint limits. For MJX reverse mode, author a constraint-free "
            "planar_chain_graph and retain its physical settings."
        )
    graph = planar_chain_graph(
        preset, chain_config, sim_config, muscle_config, joint_limits=(-np.pi, np.pi),
    )
    return compile_planar_model(graph, external_tendon_actuation=False).model


def to_mjx(model):
    """Transfer a compiled MuJoCo model to an MJX device model.

    Args:
        model: ``mujoco.MjModel`` instance (CPU).

    Returns:
        ``mjx.Model`` on the default JAX device.
    """
    from mujoco import mjx

    return mjx.put_model(model, impl="jax")


def get_site_id(model, name: str) -> int:
    """Look up a site index by name on a CPU MuJoCo model.

    Args:
        model: ``mujoco.MjModel`` instance.
        name: Site name.

    Returns:
        Integer site ID, or -1 if not found.
    """
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def get_body_id(model, name: str) -> int:
    """Look up a body index by name on a CPU MuJoCo model.

    Args:
        model: ``mujoco.MjModel`` instance.
        name: Body name.

    Returns:
        Integer body ID, or -1 if not found.
    """
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
