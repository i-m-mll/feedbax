"""Studio representations for the native muscle force graphs and plant."""

from collections.abc import Mapping
from typing import Any

from feedbax.contracts.representation import RepresentationSpec
from feedbax.mechanics.muscle_config import default_6muscle_2link_attachment_paths
from feedbax.studio.representation import (
    _representation,
    _selector_binding,
    _literal_binding,
    _param_binding,
    _compose_children_metadata,
)


def _default_two_link_muscle_path_geometry() -> dict[str, Any]:
    """Expose canonical provider-owned body-local attachment topology."""
    paths = default_6muscle_2link_attachment_paths()
    return {
        "paths": [
            {
                "id": f"muscle-{index}",
                "points": [
                    {
                        "frame": path.origin.body,
                        "position": [float(path.origin.pos[0]), float(path.origin.pos[1])],
                    },
                    {
                        "frame": path.insertion.body,
                        "position": [
                            float(path.insertion.pos[0]),
                            float(path.insertion.pos[1]),
                        ],
                    },
                ],
            }
            for index, path in enumerate(paths)
        ],
    }


def _two_link_muscle_representation(
    *,
    source: str,
    composite: bool,
    muscle_path_geometry: Mapping[str, Any] | None = None,
    frame_input_port: str | None = None,
    frame_element_id: str | None = None,
    self_link_lengths: list[float] | None = None,
) -> RepresentationSpec:
    if frame_input_port is not None and frame_element_id is not None:
        raise ValueError("muscle path representation accepts only one frame provider")
    metadata = {
        "geometry_source": source,
        "chain_source": "feedbax.mechanics.skeleton.arm.TwoLinkArm",
        "muscle_count": 6,
        **(_compose_children_metadata() if composite else {}),
    }
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "metadata": metadata,
            **(
                {"muscle_path_geometry": dict(muscle_path_geometry)}
                if muscle_path_geometry is not None
                else {}
            ),
            "anchors": [
                {
                    "id": "shoulder",
                    "semantic_role": "joint",
                    "label": "Shoulder",
                    "binding": _literal_binding([0.0, 0.0], dim=2),
                    "units": "m",
                    "dim": 2,
                    "metadata": {"joint_index": 0},
                },
                {
                    "id": "elbow",
                    "semantic_role": "joint",
                    "label": "Elbow",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"computed_from": "joint_angles_and_link_lengths"},
                },
                {
                    "id": "effector",
                    "semantic_role": "endpoint",
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
                {
                    "id": "muscle-origins",
                    "semantic_role": "origin",
                    "label": "Muscle origins",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"collection": "muscle_origins"},
                },
                {
                    "id": "muscle-insertions",
                    "semantic_role": "insertion",
                    "label": "Muscle insertions",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"collection": "muscle_insertions"},
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
                        **(
                            {"link_lengths": _literal_binding(self_link_lengths, dim=2)}
                            if self_link_lengths is not None
                            else {}
                        ),
                    },
                    "metadata": {"chain_kind": "two_link_arm"},
                    **(
                        {
                            "planar_chain": {
                                "frame_ids": ["world", "link0", "link1"],
                                "pose_fallback": "zero",
                            }
                        }
                        if self_link_lengths is not None
                        else {}
                    ),
                },
                {
                    "id": "muscle-paths",
                    "archetype": "muscle_path",
                    "anchors": ["muscle-origins", "muscle-insertions"],
                    "style": [
                        {
                            "channel": "opacity",
                            "binding": _selector_binding(
                                "biomechanics_object",
                                "output:activations",
                                target_id="activations",
                                role="activation",
                                metadata={"style_binding": "per_muscle"},
                            ),
                            "metadata": {
                                "maps": "muscle_activation",
                                "range": [0.15, 1.0],
                            },
                        },
                        {"channel": "stroke", "value": "var(--workspace-muscle-stroke)"},
                    ],
                    **(
                        {
                            "frame_provider": {
                                "kind": "from_input_port",
                                "input_port": frame_input_port,
                            }
                        }
                        if frame_input_port is not None
                        else (
                            {
                                "frame_provider": {
                                    "kind": "from_representation_element",
                                    "element_id": frame_element_id,
                                }
                            }
                            if frame_element_id is not None
                            else {}
                        )
                    ),
                    "metadata": {
                        "geometry_source": source,
                        "runtime_frame_paths": "resolver_required",
                        "activation_channel": "opacity",
                    },
                },
            ],
        }
    )


def _point_mass_muscle_representation() -> RepresentationSpec:
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "metadata": {
                "geometry_source": "feedbax.mechanics.geometry.PointMassRadialGeometry",
                "muscle_count_param": "n_pairs",
                **_compose_children_metadata(),
            },
            "anchors": [
                {
                    "id": "center",
                    "semantic_role": "center",
                    "label": "Point mass",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"source_output_port": "force_2d"},
                },
                {
                    "id": "radial-muscle-paths",
                    "semantic_role": "path",
                    "label": "Radial muscle paths",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"collection": "radial_muscle_paths"},
                },
            ],
            "elements": [
                {
                    "id": "point-body",
                    "archetype": "point_body",
                    "anchors": ["center"],
                    "metadata": {"body_kind": "point_mass"},
                },
                {
                    "id": "radial-muscles",
                    "archetype": "muscle_path",
                    "anchors": ["radial-muscle-paths"],
                    "bindings": {
                        "n_pairs": _param_binding("n_pairs", expected_type="int"),
                    },
                    "style": [
                        {
                            "channel": "opacity",
                            "binding": _selector_binding(
                                "biomechanics_object",
                                "output:activations",
                                target_id="activations",
                                role="activation",
                                metadata={"style_binding": "per_muscle"},
                            ),
                            "metadata": {
                                "maps": "muscle_activation",
                                "range": [0.15, 1.0],
                            },
                        }
                    ],
                    "metadata": {
                        "geometry_source": "feedbax.mechanics.geometry.PointMassRadialGeometry",
                        "runtime_frame_paths": "resolver_required",
                    },
                },
            ],
        }
    )
