"""Component definitions for direct-force plant construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import Any

from feedbax.component_registry.declarations import (
    DeclaredComponent,
    ParameterField,
    declare_component,
)
from feedbax.component_registry.discovery import source_path
from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.representation import RepresentationSpec
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.arm import TwoLinkArm
from feedbax.mechanics.skeleton.pointmass import PointMass


def _build_two_link_arm(params: Mapping[str, Any]) -> Mechanics:
    plant = DirectForceInput(TwoLinkArm(l=params.get("link_lengths", (0.30, 0.33))))
    return Mechanics(plant=plant, dt=float(params.get("dt", 0.01)))


def _build_point_mass(params: Mapping[str, Any]) -> Mechanics:
    plant = DirectForceInput(
        PointMass(
            mass=float(params.get("mass", 1.0)),
            damping=float(params.get("damping", 0.0)),
        )
    )
    return Mechanics(plant=plant, dt=float(params.get("dt", 0.01)))


def mechanics_state_output_prototype(
    builder: Callable[[Mapping[str, Any]], Mechanics],
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the same port trees as the initialized mechanics component."""
    mechanics = builder(params)
    return mechanics.initial_outputs(mechanics.state_index.init)


def _point_mass_representation() -> RepresentationSpec:
    from feedbax.component_registry.representation import (
        _representation,
        _selector_binding,
    )

    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "anchors": [
                {
                    "id": "center",
                    "semantic_role": "center",
                    "interaction_roles": ["selectable", "hoverable"],
                    "label": "Effector",
                    "binding": _selector_binding(
                        "mechanics_object",
                        "output:effector",
                        target_id="effector",
                        path="pos",
                        anchor_subpath="position",
                        dim=2,
                    ),
                    "units": "m",
                    "dim": 2,
                }
            ],
            "elements": [
                {
                    "id": "point-body",
                    "archetype": "point_body",
                    "anchors": ["center"],
                    "bindings": {
                        "position": _selector_binding(
                            "mechanics_object",
                            "output:effector",
                            target_id="effector",
                            path="pos",
                            anchor_subpath="position",
                            dim=2,
                        )
                    },
                    "style": [
                        {"channel": "radius", "value": 5},
                        {"channel": "fill", "value": "var(--workspace-body-fill)"},
                    ],
                    "metadata": {
                        "body_kind": "point_mass",
                        "source_output_port": "effector",
                    },
                }
            ],
        }
    )


def _two_link_arm_representation() -> RepresentationSpec:
    from feedbax.component_registry.representation import (
        _literal_binding,
        _param_binding,
        _representation,
        _selector_binding,
    )

    link_lengths = _param_binding("link_lengths", expected_type="array", dim=2)
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "anchors": [
                {
                    "id": "shoulder",
                    "semantic_role": "joint",
                    "interaction_roles": ["selectable", "hoverable"],
                    "label": "Shoulder",
                    "binding": _literal_binding([0.0, 0.0], dim=2),
                    "units": "m",
                    "dim": 2,
                    "metadata": {"joint_index": 0},
                },
                {
                    "id": "elbow",
                    "semantic_role": "joint",
                    "interaction_roles": ["selectable", "hoverable"],
                    "label": "Elbow",
                    "units": "m",
                    "dim": 2,
                    "metadata": {
                        "computed_from": "joint_angles_and_link_lengths",
                        "joint_index": 1,
                    },
                },
                {
                    "id": "effector",
                    "semantic_role": "endpoint",
                    "interaction_roles": ["selectable", "hoverable"],
                    "label": "Effector",
                    "binding": _selector_binding(
                        "mechanics_object",
                        "output:effector",
                        target_id="effector",
                        path="pos",
                        anchor_subpath="position",
                        dim=2,
                    ),
                    "units": "m",
                    "dim": 2,
                },
            ],
            "elements": [
                {
                    "id": "links",
                    "archetype": "planar_chain",
                    "anchors": ["shoulder", "elbow", "effector"],
                    "bindings": {
                        "joint_angles": _selector_binding(
                            "mechanics_object",
                            "output:state",
                            target_id="state",
                            path="skeleton.angle",
                            anchor_subpath="orientation",
                            dim=2,
                        ),
                        "link_lengths": link_lengths,
                    },
                    "style": [
                        {"channel": "stroke", "value": "var(--workspace-body-stroke)"},
                        {"channel": "line_width", "value": 3},
                    ],
                    "metadata": {
                        "chain_kind": "two_link_arm",
                        "kinematics": "feedbax.mechanics.skeleton.arm.TwoLinkArm.forward_kinematics",
                        "link_length_param": "link_lengths",
                    },
                    "planar_chain": {
                        "frame_ids": ["world", "link0", "link1"],
                        "reference_pose": {
                            "coordinate_space": "configuration",
                            "values": [0.0, 0.0],
                        },
                        "pose_fallback": "zero",
                    },
                }
            ],
            "reachability": {
                "kind": "radial",
                "origin_anchor": "shoulder",
                "radius_binding": link_lengths,
                "radius_transform": "sum_abs",
                "label": "arm reach",
                "units": "m",
            },
        }
    )


def two_link_arm_definition() -> DeclaredComponent:
    definition = declare_component(
        name="TwoLinkArm",
        builder=_build_two_link_arm,
        attribute_paths={"link_lengths": "plant.skeleton.l"},
        override_reason="constructor wraps the skeleton in DirectForceInput and Mechanics",
        category="Mechanics",
        description="Two-link arm plant with direct force input.",
        parameter_fields=[
            ParameterField(name="dt", type="float", default=0.01, min=0.001, required=True),
            ParameterField(
                name="link_lengths",
                type="array",
                default=[0.30, 0.33],
                required=False,
            ),
        ],
        input_ports=Mechanics.input_ports,
        output_ports=Mechanics.output_ports,
        icon="Activity",
        port_types=PortTypeSpec(
            inputs={"force": PortType(dtype="vector")},
            outputs={
                "effector": PortType(dtype="state"),
                "state": PortType(dtype="state"),
            },
        ),
        output_prototype_fn=partial(mechanics_state_output_prototype, _build_two_link_arm),
        representation=_two_link_arm_representation(),
    )

    return replace(definition, source_paths=(source_path(TwoLinkArm),))


def point_mass_definition() -> DeclaredComponent:
    from feedbax.compiler.mechanics_templates import point_mass_template_graph

    definition = declare_component(
        name="PointMass",
        builder=_build_point_mass,
        attribute_paths={
            "mass": "plant.skeleton.mass",
            "damping": "plant.skeleton.damping",
        },
        override_reason="constructor wraps the skeleton in DirectForceInput and Mechanics",
        category="Mechanics",
        description="Point-mass plant with direct force input.",
        parameter_fields=[
            ParameterField(name="dt", type="float", default=0.01, min=0.001, required=True),
            ParameterField(name="mass", type="float", default=1.0, min=0.0, required=False),
            ParameterField(name="damping", type="float", default=0.0, min=0.0, required=False),
        ],
        input_ports=Mechanics.input_ports,
        output_ports=Mechanics.output_ports,
        icon="Activity",
        port_types=PortTypeSpec(
            inputs={"force": PortType(dtype="vector")},
            outputs={
                "effector": PortType(dtype="state"),
                "state": PortType(dtype="state"),
            },
        ),
        output_prototype_fn=partial(mechanics_state_output_prototype, _build_point_mass),
        representation=_point_mass_representation(),
        template_graph=point_mass_template_graph(),
        template_id="feedbax.templates.mechanics.point_mass",
        template_kind="executable",
    )
    return replace(
        definition, source_paths=(source_path(PointMass), source_path(point_mass_template_graph))
    )
