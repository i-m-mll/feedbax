"""Spatial representations for acausal multibody catalogue owners."""

from typing import Any
from feedbax.contracts.representation import RepresentationSpec

def _literal_binding(value: Any, *, dim: int | None = None) -> dict[str, Any]:
    binding: dict[str, Any] = {"kind": "literal", "value": value}
    if dim is not None:
        binding["dim"] = dim
    return binding


def _param_binding(
    path: str, *, expected_type: str | None = None, dim: int | None = None
) -> dict[str, Any]:
    binding: dict[str, Any] = {"kind": "param_path", "path": path}
    if expected_type is not None:
        binding["expected_type"] = expected_type
    if dim is not None:
        binding["dim"] = dim
    return binding


def _representation(data: dict[str, Any]) -> RepresentationSpec:
    return RepresentationSpec.model_validate(data)


def _planar_link_representation() -> RepresentationSpec:
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "anchors": [
                {
                    "id": "proximal",
                    "semantic_role": "joint",
                    "binding": _literal_binding([0.0, 0.0], dim=2),
                    "units": "m",
                    "dim": 2,
                },
                {
                    "id": "distal",
                    "semantic_role": "endpoint",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"computed_from": "length_and_orientation"},
                },
            ],
            "elements": [
                {
                    "id": "link",
                    "archetype": "planar_chain",
                    "anchors": ["proximal", "distal"],
                    "bindings": {
                        "link_lengths": _param_binding("length", expected_type="float", dim=1)
                    },
                    "metadata": {"chain_kind": "planar_link_segment"},
                }
            ],
        }
    )


def _muscle_path_representation() -> RepresentationSpec:
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "anchors": [
                {
                    "id": "path",
                    "semantic_role": "path",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"collection": "path_points"},
                }
            ],
            "elements": [
                {
                    "id": "muscle-path",
                    "archetype": "muscle_path",
                    "anchors": ["path"],
                    "bindings": {
                        "path_points": _param_binding("path_points", expected_type="array")
                    },
                    "metadata": {"activation_input": "activation"},
                }
            ],
        }
    )


def _point_marker_representation() -> RepresentationSpec:
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "anchors": [
                {
                    "id": "point",
                    "semantic_role": "endpoint",
                    "binding": _param_binding("offset", expected_type="array", dim=2),
                    "units": "m",
                    "dim": 2,
                }
            ],
            "elements": [
                {
                    "id": "marker",
                    "archetype": "marker",
                    "anchors": ["point"],
                    "metadata": {"frame_param": "frame"},
                }
            ],
        }
    )


