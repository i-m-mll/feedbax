"""Causal geometry, routing, and force reduction leaves for muscle graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

from equinox import field
import jax.numpy as jnp

from feedbax.mechanics.geometry import (
    ConstantMomentArmGeometry,
    PointMassRadialGeometry,
    PolynomialMomentArmGeometry,
    TwoLinkArmMuscleGeometry,
    WrappingGeometry,
)
from feedbax.runtime.graph import Component


class VectorElements(Component):
    """Route each element of a vector to its own scalar output."""

    input_ports = ("input",)
    output_ports: tuple[str, ...]
    size: int = field(static=True)

    def __init__(self, size: int):
        self.size = size
        self.output_ports = tuple(f"out_{index}" for index in range(size))

    def __call__(self, inputs, state, *, key):
        vector = jnp.asarray(inputs.get("input", jnp.zeros(self.size)))
        if vector.shape != (self.size,):
            raise ValueError(f"Expected a vector of shape {(self.size,)}, got {vector.shape}")
        return dict(zip(self.output_ports, vector, strict=True)), state


class JointMuscleGeometry(Component):
    """Produce musculotendon kinematics and moment arms from joint kinematics."""

    input_ports = ("angles", "angular_velocities")
    output_ports = ("lengths", "velocities", "directions")

    geometry: TwoLinkArmMuscleGeometry

    def __init__(self, paths: Sequence[Mapping[str, Any]]):
        if not paths:
            raise ValueError("Joint muscle geometry requires at least one path")
        self.geometry = TwoLinkArmMuscleGeometry(
            tuple(_PATH_TYPES[path["kind"]](**path["params"]) for path in paths)
        )

    @property
    def paths(self) -> list[dict[str, Any]]:
        return muscle_path_parameters(self.geometry)

    def __call__(self, inputs, state, *, key):
        angles = inputs.get("angles", jnp.zeros(2))
        velocities = inputs.get("angular_velocities", jnp.zeros(2))
        return {
            "lengths": self.geometry.musculotendon_lengths(angles),
            "velocities": self.geometry.musculotendon_velocities(angles, velocities),
            "directions": self.geometry.moment_arms(angles),
        }, state


_PATH_TYPES = {
    "constant": ConstantMomentArmGeometry,
    "polynomial": PolynomialMomentArmGeometry,
    "wrapping": WrappingGeometry,
}


def muscle_path_parameters(geometry: TwoLinkArmMuscleGeometry) -> list[dict[str, Any]]:
    """Encode the existing geometry models as portable constructor parameters."""
    kinds = {geometry_type: kind for kind, geometry_type in _PATH_TYPES.items()}
    paths = []
    for path in geometry.geometries:
        kind = kinds.get(type(path))
        if kind is None:
            raise ValueError(f"Unsupported portable muscle geometry {type(path).__name__}")
        params = {}
        for parameter in fields(path):
            value = getattr(path, parameter.name)
            params[parameter.name] = value.tolist() if hasattr(value, "tolist") else value
        paths.append({"kind": kind, "params": params})
    return paths


class RadialMuscleGeometry(Component):
    """Produce the interleaved directions of radial antagonist pairs."""

    input_ports = ()
    output_ports = ("directions",)

    geometry: PointMassRadialGeometry

    def __init__(self, n_pairs: int):
        self.geometry = PointMassRadialGeometry(n_pairs)

    @property
    def n_pairs(self) -> int:
        return self.geometry.n_muscles // 2

    def __call__(self, inputs, state, *, key):
        return {"directions": self.geometry.directions}, state
