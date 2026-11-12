"""Authored folders and native kinematics used by the Scene witnesses."""

import json
from pathlib import Path
import sys

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.graph import GRAPH_SPEC_SCHEMA_ID, ComponentSpec, GraphSpec
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project, load_text_project


_LINKS = ("upper", "forearm", "hand")
_JOINTS = ("shoulder", "elbow", "wrist")
_POSES = {
    1: [[0.32], [-0.61]],
    3: [[0.43, -0.67, 0.26], [-0.31, 0.58, -0.48], [0.64, 0.15, -0.72]],
}


def _interior(count: int, *, wrapped: bool = False) -> AcausalGraphSpec:
    def node(kind: str, **params) -> ComponentSpec:
        return ComponentSpec(type=f"feedbax.{kind}", params=params)

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
    wrap_geometries = {}
    if count == 3:
        nodes["cross_chain"] = node(
            "MusclePath",
            path_points=[
                {"frame": "base.frame", "offset": [-0.11, 0.14]},
                {"frame": "forearm.distal", "offset": [0.03, -0.09]},
                {"frame": "hand.distal", "offset": [-0.07, 0.05]},
            ],
        )
        if wrapped:
            nodes["side"] = node("PointMarker", frame="base.frame", offset=[0.0, 0.2])
            nodes["cross_chain"].params["wraps"] = [
                {"after_site": 0, "geom": "guide", "side_site": "side"}
            ]
            wrap_geometries["guide"] = {"kind": "cylinder", "frame": "base.frame", "radius": 0.03}
    return AcausalGraphSpec(
        physical_domain="planar_multibody",
        nodes=nodes,
        wrap_geometries=wrap_geometries,
        solver={"solver_type": "euler", "dt": 0.002},
    )


def _graph(interior: AcausalGraphSpec) -> GraphSpec:
    return GraphSpec(
        nodes={
            "mechanics": ComponentSpec(
                type="feedbax.AcausalSystem",
                input_ports=["force"],
                output_ports=["effector"],
            )
        },
        subgraphs={"mechanics": interior},
        input_ports=["force"],
        input_bindings={"force": ("mechanics", "force")},
        output_ports=["effector"],
        output_bindings={"effector": ("mechanics", "effector")},
    )


def seed_scene_project(root: Path) -> None:
    graphs = {
        "single": _graph(_interior(1)),
        "chain": _graph(_interior(3)),
        "wrapped": _graph(_interior(3, wrapped=True)),
    }
    project = Project.model_validate(
        {
            "documents": {
                name: {
                    "schema": GRAPH_SPEC_SCHEMA_ID,
                    "payload": graph.model_dump(mode="json", exclude_none=True),
                }
                for name, graph in graphs.items()
            }
        }
    )
    for name, content in dump_text_project(
        project, ProjectConfiguration(project=root.name, store="results")
    ).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def native_scene_samples(root: Path, document: str):
    import mujoco

    from feedbax.compiler import GraphDocument, compile_graph
    from feedbax.component_registry import ComponentRegistry

    saved = load_text_project(root)
    graph = GraphSpec.model_validate(saved.project.documents[document].payload)
    compiled = (
        compile_graph(GraphDocument(graph=graph), ComponentRegistry())
        .graph.nodes["mechanics"]
        .descriptor
    )
    model = compiled.model
    interior = graph.subgraphs["mechanics"]
    samples = []
    for qpos in _POSES[model.nq]:
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        bodies = {}
        for name in _LINKS[: len(qpos)]:
            body = model.body(name).id
            origin = data.xpos[body, :2]
            rotation = data.xmat[body].reshape(3, 3)[:2, :2]
            tip = origin + rotation[:, 0] * interior.nodes[name].params["length"]
            bodies[name] = {
                "start": origin.tolist(),
                "end": tip.tolist(),
                "rotation": rotation.flatten().tolist(),
            }
        samples.append(
            {
                "qpos": qpos,
                "bodies": bodies,
                "joints": {
                    name: data.xanchor[model.joint(name).id, :2].tolist()
                    for name in _JOINTS[: len(qpos)]
                },
                "muscles": {
                    path.name: [
                        data.site_xpos[model.site(site).id, :2].tolist() for site in path.sites
                    ]
                    for path in compiled.paths
                },
            }
        )
    return {"revision": saved.revision, "identity": compiled.identity, "samples": samples}


if __name__ == "__main__":
    action, path, *args = sys.argv[1:]
    if action == "seed":
        seed_scene_project(Path(path))
    elif action == "read":
        output, *documents = args
        Path(output).write_text(
            json.dumps(
                {document: native_scene_samples(Path(path), document) for document in documents}
            )
        )
    else:
        raise ValueError(f"Unknown Scene fixture action {action!r}")
