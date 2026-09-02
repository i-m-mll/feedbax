"""Executable mechanics-domain acausal graph templates."""

from __future__ import annotations

from typing import Any

from feedbax.contracts.acausal import ACAUSAL_GRAPH_SCHEMA_VERSION
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.graph import ComponentSpec
from feedbax.mechanics.body import default_2link_bounds
from feedbax.mechanics.muscle_config import (
    default_6muscle_2link_moment_arms,
    default_6muscle_2link_reference_lengths,
)


def point_mass_template_graph(params: dict[str, Any] | None = None) -> AcausalGraphSpec:
    """Build a driven translational point-mass mechanics interior."""

    params = dict(params or {})
    return AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="MechanicsGround"),
            "mass": ComponentSpec(
                type="MechanicsMass",
                params={"mass": float(params.get("mass", 1.0))},
            ),
            "damper": ComponentSpec(
                type="MechanicsLinearDamper",
                params={"damping": float(params.get("damping", 0.0))},
            ),
            "force": ComponentSpec(
                type="ActuationInput",
                params={"port_name": "force", "source_kind": "force"},
            ),
            "position": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "position", "quantity": "position"},
            ),
            "velocity": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "velocity", "quantity": "velocity"},
            ),
        },
        connections=[
            {"a": ("wall", "flange"), "b": ("damper", "flange_a")},
            {"a": ("damper", "flange_b"), "b": ("mass", "flange")},
            {"a": ("force", "flange"), "b": ("mass", "flange")},
            {"a": ("position", "flange"), "b": ("mass", "flange")},
            {"a": ("velocity", "flange"), "b": ("mass", "flange")},
        ],
        solver={
            "solver_type": str(params.get("solver_type", "euler")),
            "dt": float(params.get("dt", 0.001)),
        },
    )


def mass_spring_damper_template_graph(
    params: dict[str, Any] | None = None,
) -> AcausalGraphSpec:
    """Build a grounded mass-spring-damper mechanics interior."""

    params = dict(params or {})
    return AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="MechanicsGround"),
            "mass": ComponentSpec(
                type="MechanicsMass",
                params={"mass": float(params.get("mass", 1.0))},
            ),
            "spring": ComponentSpec(
                type="MechanicsLinearSpring",
                params={"stiffness": float(params.get("stiffness", 10.0))},
            ),
            "damper": ComponentSpec(
                type="MechanicsLinearDamper",
                params={"damping": float(params.get("damping", 0.5))},
            ),
            "force": ComponentSpec(
                type="ActuationInput",
                params={"port_name": "force", "source_kind": "force"},
            ),
            "position": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "position", "quantity": "position"},
            ),
            "velocity": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "velocity", "quantity": "velocity"},
            ),
        },
        connections=[
            {"a": ("wall", "flange"), "b": ("spring", "flange_a")},
            {"a": ("wall", "flange"), "b": ("damper", "flange_a")},
            {"a": ("spring", "flange_b"), "b": ("mass", "flange")},
            {"a": ("damper", "flange_b"), "b": ("mass", "flange")},
            {"a": ("force", "flange"), "b": ("mass", "flange")},
            {"a": ("position", "flange"), "b": ("mass", "flange")},
            {"a": ("velocity", "flange"), "b": ("mass", "flange")},
        ],
        solver={
            "solver_type": str(params.get("solver_type", "euler")),
            "dt": float(params.get("dt", 0.001)),
        },
    )


def point_mass_with_muscles_template_graph(
    params: dict[str, Any] | None = None,
) -> AcausalGraphSpec:
    """Build a point mass driven by one antagonistic muscle pair."""

    params = dict(params or {})
    fmax = float(params.get("max_isometric_force", 1.0))
    return AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="MechanicsGround"),
            "mass": ComponentSpec(
                type="MechanicsMass",
                params={"mass": float(params.get("mass", 1.0))},
            ),
            "damper": ComponentSpec(
                type="MechanicsLinearDamper",
                params={"damping": float(params.get("damping", 0.0))},
            ),
            "flexor": ComponentSpec(
                type="RigidTendonHillMuscle",
                params={"max_isometric_force": fmax, "direction": 1.0},
            ),
            "extensor": ComponentSpec(
                type="RigidTendonHillMuscle",
                params={"max_isometric_force": fmax, "direction": -1.0},
            ),
            "position": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "position", "quantity": "position"},
            ),
            "velocity": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "velocity", "quantity": "velocity"},
            ),
        },
        connections=[
            {"a": ("wall", "flange"), "b": ("damper", "flange_a")},
            {"a": ("damper", "flange_b"), "b": ("mass", "flange")},
            {"a": ("flexor", "flange"), "b": ("mass", "flange")},
            {"a": ("extensor", "flange"), "b": ("mass", "flange")},
            {"a": ("position", "flange"), "b": ("mass", "flange")},
            {"a": ("velocity", "flange"), "b": ("mass", "flange")},
        ],
        solver={
            "solver_type": str(params.get("solver_type", "euler")),
            "dt": float(params.get("dt", 0.001)),
        },
    )


def _float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def two_link_arm_6muscle_template_graph(
    params: dict[str, Any] | None = None,
) -> AcausalGraphSpec:
    """Build an editable planar multibody interior for the canonical arm."""

    params = dict(params or {})
    bounds = default_2link_bounds()
    lengths = (bounds.segment_lengths_min + bounds.segment_lengths_max) / 2.0
    masses = (bounds.segment_masses_min + bounds.segment_masses_max) / 2.0
    damping = (bounds.joint_damping_min + bounds.joint_damping_max) / 2.0
    moment_arms = default_6muscle_2link_moment_arms()
    reference_lengths = default_6muscle_2link_reference_lengths()
    fmax = float(params.get("max_isometric_force", 1.0))

    muscle_nodes = {
        f"muscle_{index}": ComponentSpec(
            type="MusclePath",
            params={
                "path_points": [
                    {"frame": "world.frame", "offset": [0.0, 0.0]},
                    {"frame": "upper.distal", "offset": [0.0, 0.0]},
                    {"frame": "forearm.distal", "offset": [0.0, 0.0]},
                ],
                "moment_arm": _float_list(moment_arms[index]),
                "reference_length": float(reference_lengths[index]),
                "max_isometric_force": fmax,
            },
        )
        for index in range(6)
    }

    return AcausalGraphSpec(
        schema_version=ACAUSAL_GRAPH_SCHEMA_VERSION,
        physical_domain="planar_multibody",
        nodes={
            "world": ComponentSpec(type="WorldFrame"),
            "anchor": ComponentSpec(
                type="Anchor",
                params={"world_frame": "world.frame", "frame": "upper.proximal"},
            ),
            "shoulder": ComponentSpec(
                type="RevoluteJoint",
                params={
                    "parent_frame": "world.frame",
                    "child_frame": "upper.proximal",
                    "damping": float(damping[0]),
                },
            ),
            "elbow": ComponentSpec(
                type="RevoluteJoint",
                params={
                    "parent_frame": "upper.distal",
                    "child_frame": "forearm.proximal",
                    "damping": float(damping[1]),
                },
            ),
            "upper": ComponentSpec(
                type="PlanarLink",
                params={
                    "length": float(lengths[0]),
                    "mass": float(masses[0]),
                    "com": 0.5,
                    "inertia": float((1.0 / 3.0) * masses[0] * lengths[0] ** 2),
                },
            ),
            "forearm": ComponentSpec(
                type="PlanarLink",
                params={
                    "length": float(lengths[1]),
                    "mass": float(masses[1]),
                    "com": 0.5,
                    "inertia": float((1.0 / 3.0) * masses[1] * lengths[1] ** 2),
                },
            ),
            **muscle_nodes,
            "excitation": ComponentSpec(
                type="ActuationInput",
                params={
                    "port_name": "excitation",
                    "source_kind": "muscle_excitation_vector",
                },
            ),
            "effector": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "effector", "quantity": "effector"},
            ),
            "state": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "state", "quantity": "state"},
            ),
            "effector_marker": ComponentSpec(
                type="PointMarker",
                params={
                    "frame": "forearm.distal",
                    "offset": [0.0, 0.0],
                    "marker_name": "effector",
                },
            ),
        },
        connections=[
            {"a": ("world", "frame"), "b": ("anchor", "world")},
            {"a": ("anchor", "frame"), "b": ("upper", "proximal")},
            {"a": ("world", "frame"), "b": ("shoulder", "parent")},
            {"a": ("shoulder", "child"), "b": ("upper", "proximal")},
            {"a": ("upper", "distal"), "b": ("elbow", "parent")},
            {"a": ("elbow", "child"), "b": ("forearm", "proximal")},
        ],
        solver={
            "solver_type": str(params.get("solver_type", "euler")),
            "dt": float(params.get("dt", 0.01)),
        },
    )


__all__ = [
    "mass_spring_damper_template_graph",
    "point_mass_template_graph",
    "point_mass_with_muscles_template_graph",
    "two_link_arm_6muscle_template_graph",
]
