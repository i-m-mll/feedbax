"""Curated Studio representations for direct-force plants."""

from feedbax.contracts.representation import RepresentationSpec


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
