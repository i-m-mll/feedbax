"""Python constructor for the native radial ReLU muscle graph.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from feedbax.runtime.graph import Graph


def PointMass8MuscleRelu(
    n_pairs: int = 4,
    max_isometric_force: float = 500.0,
    dt: float = 0.01,
    tau_activation: float = 0.015,
    tau_deactivation: float = 0.05,
) -> Graph:
    """Compile the radial force generator with the supplied actuator parameters."""
    from feedbax.component_registry import ComponentRegistry
    from feedbax.compiler import GraphDocument, compile_graph
    from feedbax.compiler.muscle_templates import point_mass_8muscle_relu_graph

    spec = point_mass_8muscle_relu_graph(
        n_pairs,
        max_isometric_force,
        dt,
        tau_activation,
        tau_deactivation,
    )
    return compile_graph(
        GraphDocument(graph=spec),
        ComponentRegistry(),
    ).graph
