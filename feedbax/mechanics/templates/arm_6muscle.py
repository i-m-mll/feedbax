"""Python constructor for the native rigid-tendon arm muscle graph.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from feedbax.mechanics.geometry import TwoLinkArmMuscleGeometry
from feedbax.runtime.graph import Graph


def Arm6MuscleRigidTendon(
    dt: float = 0.01,
    max_isometric_force: float = 500.0,
    optimal_muscle_length: float = 0.1,
    tendon_slack_length: float = 0.1,
    geometry: TwoLinkArmMuscleGeometry | None = None,
) -> Graph:
    """Compile the arm force generator with the supplied geometry and parameters."""
    from feedbax.component_registry import ComponentRegistry
    from feedbax.compiler import GraphDocument, compile_graph
    from feedbax.compiler.muscle_templates import arm_6muscle_rigid_tendon_graph

    spec = arm_6muscle_rigid_tendon_graph(
        dt,
        max_isometric_force,
        optimal_muscle_length,
        tendon_slack_length,
        geometry,
    )
    return compile_graph(
        GraphDocument(graph=spec),
        ComponentRegistry(),
    ).graph
