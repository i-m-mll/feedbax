"""Authored project folders used by the Scene witnesses."""

import json
from pathlib import Path
import sys

from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project


_LINKS = ("upper", "forearm", "hand")
_JOINTS = ("shoulder", "elbow", "wrist")


def _model(count: int) -> dict:
    def node(kind: str, **fields) -> dict:
        return {"kind": "leaf", "name": kind, "type": f"feedbax.{kind}", "fields": fields}

    nodes = {
        "base": node("WorldFrame"),
        "mount": node("Anchor", frame="upper.proximal", world_frame="base.frame"),
        "drive": node(
            "ActuationInput", port_name="force", source_kind="actuator_control", size=count
        ),
        "sense": node("SensorOutput", port_name="effector", quantity="effector"),
        "tip": node("PointMarker", frame=f"{_LINKS[count - 1]}.distal", marker_name="effector"),
    }
    parents = ("base.frame", "upper.distal", "forearm.proximal")
    children = ("upper.proximal", "forearm.distal", "hand.proximal")
    lengths = (0.41, 0.32, 0.23)
    offsets = ([0.12, -0.08], [0.04, 0.07], [-0.03, 0.02])
    angles = (0.22, -0.36, 0.19)
    for index in range(count):
        nodes[_LINKS[index]] = node("PlanarLink", length=lengths[index])
        nodes[_JOINTS[index]] = node(
            "RevoluteJoint",
            parent_frame=parents[index],
            child_frame=children[index],
            frame_offset=offsets[index],
            frame_angle=angles[index],
            rest_angle=0.14,
        )
    if count == 3:
        nodes["cross_chain"] = node(
            "MusclePath",
            path_points=[
                {"frame": "base.frame", "offset": [-0.11, 0.14]},
                {"frame": "forearm.distal", "offset": [0.03, -0.09]},
                {"frame": "hand.distal", "offset": [-0.07, 0.05]},
            ],
        )

    for name, value in nodes.items():
        value["name"] = name

    def end(component, port):
        return {"owner": "component", "component": component, "port": port}

    def port(name, size):
        return {"id": name, "name": name, "type": {"base": "real", "shape": [size]}}

    return {
        "step_seconds": 0.002,
        "ports": {},
        "interior": {"components": {"mechanics": {
            "kind": "composite", "name": "Mechanics",
            "ports": {
                "inputs": [{**port("force", count), "default": {"value": [0.0] * count}}],
                "outputs": [port("effector", 2)],
            },
            "interior": {
                "physical_domain": "planar_multibody", "components": nodes,
                "wires": {
                    "force": {"source": {"owner": "inputs", "port": "force"}, "target": end("drive", "u")},
                    "effector": {"source": end("sense", "y"), "target": {"owner": "outputs", "port": "effector"}},
                },
            },
        }}},
    }


def scene_files():
    return [
        {"kind": "document", "id": name, "document": {
            "schema": MODEL_DOCUMENT_SCHEMA_ID, "payload": _model(count),
        }}
        for name, count in (("single", 1), ("chain", 3))
    ]


def seed_scene_project(root: Path, files: list[dict] | None = None) -> None:
    if files is None:
        files = scene_files()
    project = Project.model_validate({
        "documents": {file["id"]: file["document"] for file in files},
    })
    for name, content in dump_text_project(
        project, ProjectConfiguration(project=root.name, store="results")
    ).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


if __name__ == "__main__":
    action, *paths = sys.argv[1:]
    if action == "files":
        print(json.dumps(scene_files()))
    elif action == "seed":
        [path] = paths
        seed_scene_project(Path(path))
    else:
        raise ValueError(f"Unknown Scene fixture action {action!r}")
