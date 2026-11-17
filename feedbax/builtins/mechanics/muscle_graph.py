"""Causal geometry, routing, and force reduction leaves for muscle graphs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from equinox import field
import jax.numpy as jnp

from feedbax.contracts.component import DynamicPortPolicy, PortType
from feedbax.mechanics.geometry import (
    ConstantMomentArmGeometry,
    PointMassRadialGeometry,
    PolynomialMomentArmGeometry,
    TwoLinkArmMuscleGeometry,
    WrappingGeometry,
)
from feedbax.runtime.graph import Component
from feedbax.contracts.portable import Ports, ComponentOptions
from typing import Annotated
from pydantic import Field


def _vector_elements_prototypes(params: Mapping[str, Any], inputs: Mapping[str, Any]):
    size = params["size"]
    vector = inputs.get("input")
    if vector is not None and getattr(vector, "shape", None) != (size,):
        raise ValueError(
            f"Expected a vector of shape {(size,)}, got {getattr(vector, 'shape', None)}"
        )
    dtype = getattr(vector, "dtype", None)
    return {f"out_{index}": jnp.zeros((), dtype=dtype) for index in range(size)}


def _joint_geometry_prototypes(params: Mapping[str, Any], inputs: Mapping[str, Any]):
    count = len(params["paths"])
    dtype = getattr(inputs.get("angles"), "dtype", None)
    return {
        "lengths": jnp.zeros(count, dtype=dtype),
        "velocities": jnp.zeros(count, dtype=dtype),
        "directions": jnp.zeros((count, 2), dtype=dtype),
    }


def _radial_geometry_prototypes(params: Mapping[str, Any], inputs: Mapping[str, Any]):
    return {"directions": jnp.zeros((2 * params["n_pairs"], 2))}


class VectorElements(Component):
    """Route each element of a vector to its own scalar output."""

    ports = Ports(
        inputs={"input": PortType(dtype="vector")},
        outputs={},
        dynamic=DynamicPortPolicy(
            count_param="size",
            count_mode="integer",
            direction="output",
            fixed_input_ports=("input",),
            generated_name_template="out_{index}",
            dynamic_port_type=PortType(dtype="scalar"),
        ),
    )
    component_options = ComponentOptions(output_prototype_fn=_vector_elements_prototypes)
    output_ports: tuple[str, ...]
    size: int = field(static=True)

    def __init__(self, size: Annotated[int, Field(ge=1)] = 2):
        self.size = size
        self.output_ports = tuple(f"out_{index}" for index in range(size))

    def __call__(self, inputs, state, *, key):
        vector = jnp.asarray(inputs.get("input", jnp.zeros(self.size)))
        if vector.shape != (self.size,):
            raise ValueError(f"Expected a vector of shape {(self.size,)}, got {vector.shape}")
        return dict(zip(self.output_ports, vector, strict=True)), state


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


class JointMuscleGeometry(Component):
    """Produce musculotendon kinematics and moment arms from joint kinematics."""

    ports = Ports(
        inputs={"angles": PortType(dtype="vector"), "angular_velocities": PortType(dtype="vector")},
        outputs={
            "lengths": PortType(dtype="vector"),
            "velocities": PortType(dtype="vector"),
            "directions": PortType(dtype="array"),
        },
    )
    component_options = ComponentOptions(output_prototype_fn=_joint_geometry_prototypes)

    geometry: TwoLinkArmMuscleGeometry

    def __init__(
        self,
        paths: Annotated[
            Any, Field(json_schema_extra={"feedbax": {"type": "array"}})
        ] = muscle_path_parameters(TwoLinkArmMuscleGeometry.default_six_muscle()),
    ):
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


class RadialMuscleGeometry(Component):
    """Produce the interleaved directions of radial antagonist pairs."""

    ports = Ports(inputs={}, outputs={"directions": PortType(dtype="array")})
    component_options = ComponentOptions(output_prototype_fn=_radial_geometry_prototypes)

    geometry: PointMassRadialGeometry

    def __init__(self, n_pairs: Annotated[int, Field(ge=1)] = 4):
        self.geometry = PointMassRadialGeometry(n_pairs)

    @property
    def n_pairs(self) -> int:
        return self.geometry.n_muscles // 2

    def __call__(self, inputs, state, *, key):
        return {"directions": self.geometry.directions}, state
