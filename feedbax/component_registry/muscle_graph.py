"""Transitional adapter for the central built-in registration."""

from __future__ import annotations

from typing import Any, Protocol

from feedbax.mechanics.muscle_graph import JointMuscleGeometry, RadialMuscleGeometry, VectorElements

from .declarations import DeclaredComponent
from .discovery import compile_owner


class _Registry(Protocol):
    def register(self, meta: DeclaredComponent) -> None: ...


def register_muscle_graph_leaves(
    registry: _Registry, _default_joint_paths: list[dict[str, Any]]
) -> None:
    for owner in (VectorElements, JointMuscleGeometry, RadialMuscleGeometry):
        registry.register(compile_owner(owner)[0][0])
