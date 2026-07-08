"""Executable mechanics-domain acausal graph templates."""

from __future__ import annotations

from typing import Any

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.graph import ComponentSpec


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


__all__ = [
    "mass_spring_damper_template_graph",
    "point_mass_template_graph",
    "point_mass_with_muscles_template_graph",
]
