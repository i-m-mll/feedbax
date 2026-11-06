"""Registry declarations for the causal muscle force graph leaves."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import jax.numpy as jnp

from feedbax.contracts.component import DynamicPortPolicy, PortType, PortTypeSpec
from feedbax.mechanics.muscle_graph import (
    JointMuscleGeometry,
    RadialMuscleGeometry,
    VectorElements,
)

from .declarations import DeclaredComponent, ParameterField, declare_component


class _Registry(Protocol):
    def register(self, meta: DeclaredComponent) -> None: ...


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


def register_muscle_graph_leaves(
    registry: _Registry, default_joint_paths: list[dict[str, Any]]
) -> None:
    vector = PortType(dtype="vector")
    scalar = PortType(dtype="scalar")
    array = PortType(dtype="array")
    registry.register(
        declare_component(
            name="VectorElements",
            runtime_type=VectorElements,
            category="Signal Processing",
            description="Route each vector element to a scalar port.",
            parameter_fields=[ParameterField(name="size", type="int", default=2, min=1)],
            input_ports=VectorElements.input_ports,
            output_ports=["out_0", "out_1"],
            port_types=PortTypeSpec(
                inputs={"input": vector},
                outputs={"out_0": scalar, "out_1": scalar},
            ),
            dynamic_port_policy=DynamicPortPolicy(
                count_param="size",
                count_mode="integer",
                direction="output",
                fixed_input_ports=VectorElements.input_ports,
                generated_name_template="out_{index}",
                dynamic_port_type=scalar,
            ),
            output_prototype_fn=_vector_elements_prototypes,
        )
    )
    registry.register(
        declare_component(
            name="JointMuscleGeometry",
            runtime_type=JointMuscleGeometry,
            category="Mechanics",
            description="Compute muscle kinematics and moment arms from joint kinematics.",
            parameter_fields=[
                ParameterField(
                    name="paths", type="array", default=default_joint_paths, required=True
                ),
            ],
            input_ports=JointMuscleGeometry.input_ports,
            output_ports=JointMuscleGeometry.output_ports,
            port_types=PortTypeSpec(
                inputs={"angles": vector, "angular_velocities": vector},
                outputs={"lengths": vector, "velocities": vector, "directions": array},
            ),
            output_prototype_fn=_joint_geometry_prototypes,
        )
    )
    registry.register(
        declare_component(
            name="RadialMuscleGeometry",
            runtime_type=RadialMuscleGeometry,
            category="Mechanics",
            description="Generate direction rows for interleaved antagonist pairs.",
            parameter_fields=[ParameterField(name="n_pairs", type="int", default=4, min=1)],
            input_ports=RadialMuscleGeometry.input_ports,
            output_ports=RadialMuscleGeometry.output_ports,
            port_types=PortTypeSpec(inputs={}, outputs={"directions": array}),
            output_prototype_fn=_radial_geometry_prototypes,
        )
    )
