"""Lower authored acausal mechanics into the model shared by the backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from math import cos, isfinite, sin
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING
from xml.etree import ElementTree

import numpy as np

from feedbax.contracts.acausal import AcausalGraphSpec, SolverConfigSpec, acausal_interior_content_hash
from feedbax.contracts.acausal_interface import is_acausal_type
from feedbax.contracts.artifact_custody import ArtifactBlobProvider
from feedbax.mechanics.compiled_model import (
    CompiledPhysicalModel,
    MusclePathBinding,
    describe_model,
)

if TYPE_CHECKING:
    from mujoco import MjSpec, MjsBody


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _vector(value: object, dimension: int, label: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != dimension:
        raise ValueError(f"{label} requires {dimension} coordinates")
    return np.array([_number(item, label) for item in value])


def _element_groups(graph: AcausalGraphSpec, types: Sequence[str]) -> dict[str, dict]:
    groups: dict[str, dict] = {kind: {} for kind in types}
    if graph.subgraphs:
        raise ValueError("Nested interiors are unsupported in authored physical models")
    for name, node in sorted(graph.nodes.items()):
        for kind in types:
            if is_acausal_type(node.type, kind):
                groups[kind][name] = node.params
                break
        else:
            raise ValueError(f"Unsupported physical model element {name!r} ({node.type})")
    return groups


def _add_site(body: MjsBody, name: str, position: Sequence[float]) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("MuJoCo sites require a nonempty name")
    body.add_site(name=name, pos=position, size=[0.005, 0.005, 0.005])


def _add_muscle_path(
    spec: MjSpec, name: str, sites: tuple[str, ...], law: str, state_size: int,
    parameters: Mapping[str, object], *, external_tendon_actuation: bool = True,
    law_revision: int | None = None,
) -> MusclePathBinding:
    if len(sites) < 2:
        raise ValueError(f"MusclePath {name!r} requires at least two sites")
    if not isinstance(law, str) or not law or type(state_size) is not int or state_size < 0:
        raise ValueError(f"MusclePath {name!r} requires a law name and nonnegative state size")
    if law_revision is not None and (type(law_revision) is not int or law_revision < 1):
        raise ValueError(f"MusclePath {name!r} requires a positive law revision")
    import mujoco

    path = MusclePathBinding(
        name, sites, law, state_size, MappingProxyType(deepcopy(dict(parameters))),
        law_revision=law_revision,
    )
    if spec.actuator(path.actuator_name) is not None:
        raise ValueError(f"MusclePath {name!r} duplicates actuator {path.actuator_name!r}")
    tendon = spec.add_tendon(name=name)
    for site in sites:
        tendon.wrap_site(site)

    if not external_tendon_actuation:
        return path

    motor = spec.add_actuator(name=path.actuator_name)
    motor.set_to_motor()
    motor.trntype = mujoco.mjtTrn.mjTRN_TENDON
    motor.target = name
    motor.gear = [-1, 0, 0, 0, 0, 0]
    motor.ctrllimited = False
    motor.forcelimited = False
    return path


def _physical_identity(graph: AcausalGraphSpec, timestep: float) -> str:
    return acausal_interior_content_hash(graph.model_copy(update={
        "solver": SolverConfigSpec(solver_type="euler", dt=timestep),
        "metadata": None,
    }))


def compile_planar_model(
    graph: AcausalGraphSpec, *, external_tendon_actuation: bool = True,
) -> CompiledPhysicalModel:
    """Compile one rooted planar serial chain with explicit COM inertials.

    The graph's revolute joints supply the topology. Anchoring identifies the
    fixed root pivot and must agree with the first joint. A supplied set of
    frame connections must agree with the same topology.
    """
    import mujoco

    if graph.physical_domain != "planar_multibody":
        raise ValueError("Expected a planar multibody graph")
    groups = _element_groups(graph, (
        "WorldFrame", "Anchor", "PlanarLink", "RevoluteJoint",
        "PointMarker", "MusclePath", "ActuationInput", "SensorOutput",
    ))
    worlds, anchors, links, joints = (
        groups[kind] for kind in ("WorldFrame", "Anchor", "PlanarLink", "RevoluteJoint")
    )
    if len(worlds) != 1 or len(anchors) != 1 or not links or len(links) != len(joints):
        raise ValueError("A planar chain requires one world, one anchor and one joint per link")
    world = next(iter(worlds))
    anchor, anchor_params = next(iter(anchors.items()))
    world_frame = f"{world}.frame"
    lengths = {}
    for name, params in links.items():
        length = _number(params.get("length", 0.3), f"{name}.length")
        if length <= 0:
            raise ValueError(f"{name}.length must be positive")
        lengths[name] = length
    frames = {world_frame: (world, 0.0)}
    frames.update({
        f"{name}.{end}": (name, distance)
        for name, length in lengths.items()
        for end, distance in (("proximal", 0.0), ("distal", length))
    })
    root_frame = anchor_params.get("frame")
    if (not isinstance(root_frame, str) or root_frame not in frames
            or anchor_params.get("world_frame", "world.frame") != world_frame):
        raise ValueError(f"Anchor {anchor!r} must connect a link frame to {world_frame!r}")

    outgoing = {}
    expected_edges = {
        frozenset(((anchor, "world"), (world, "frame"))),
        frozenset(((anchor, "frame"), tuple(root_frame.rsplit(".", 1)))),
    }
    for name, params in joints.items():
        parent_frame, child_frame = params.get("parent_frame"), params.get("child_frame")
        if (not isinstance(parent_frame, str) or not isinstance(child_frame, str)
                or parent_frame not in frames or child_frame not in frames
                or child_frame == world_frame):
            raise ValueError(f"Joint {name!r} has invalid parent/child frame references")
        parent_name, parent_x = frames[parent_frame]
        child_name, child_x = frames[child_frame]
        if parent_name in outgoing:
            raise ValueError(f"Branching planar chain at {parent_frame!r}")
        if parent_name == world and child_frame != root_frame:
            raise ValueError(f"Root joint {name!r} disagrees with anchor {anchor!r}")
        outgoing[parent_name] = (name, params, parent_x, child_name, child_x)
        expected_edges.add(frozenset(((name, "parent"), tuple(parent_frame.rsplit(".", 1)))))
        expected_edges.add(frozenset(((name, "child"), tuple(child_frame.rsplit(".", 1)))))
    actual_edges = {frozenset((edge.a, edge.b)) for edge in graph.connections}
    if graph.connections and (
        actual_edges != expected_edges or len(actual_edges) != len(graph.connections)
    ):
        raise ValueError("Planar frame connections disagree with the authored joint or anchor frames")

    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = _number(graph.solver.dt, "solver.dt")
    if spec.option.timestep <= 0:
        raise ValueError("Planar MuJoCo timestep must be positive")
    spec.option.gravity = [0.0, -9.81, 0.0]
    bodies = {world: spec.worldbody}
    parent_name = world
    for _ in links:
        if parent_name not in outgoing:
            raise ValueError("Planar joints must form a single connected chain rooted at the world")
        joint_name, joint_params, parent_x, name, child_x = outgoing[parent_name]
        if name in bodies:
            raise ValueError(f"Planar chain repeats link {name!r}")
        params = links[name]
        mass = _number(params.get("mass", 1.0), f"{name}.mass")
        com = _number(params.get("com", 0.5), f"{name}.com")
        com_y = _number(params.get("com_y", 0.0), f"{name}.com_y")
        inertia = _number(params.get("inertia", 0.03), f"{name}.inertia")
        if mass <= 0 or inertia <= 0:
            raise ValueError(f"Link {name!r} requires positive mass and COM inertia")
        angle = _number(joint_params.get("frame_angle", 0.0), f"{joint_name}.frame_angle")
        dx, dy = _vector(joint_params.get("frame_offset", [0, 0]), 2, f"{joint_name}.frame_offset")
        body = bodies[parent_name].add_body(
            name=name,
            pos=[parent_x + dx - cos(angle) * child_x, dy - sin(angle) * child_x, 0],
            quat=[cos(angle / 2), 0, 0, sin(angle / 2)],
            ipos=[lengths[name] * com, com_y, 0],
            mass=mass,
            inertia=[inertia / 2, inertia / 2, inertia],
            explicitinertial=True,
        )
        damping = _number(joint_params.get("damping", 0.0), f"{joint_name}.damping")
        stiffness = _number(joint_params.get("stiffness", 0.0), f"{joint_name}.stiffness")
        if damping < 0 or stiffness < 0:
            raise ValueError(f"Joint {joint_name!r} requires nonnegative damping and stiffness")
        joint = body.add_joint(
            name=joint_name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
            pos=[child_x, 0, 0], damping=[damping, 0, 0], stiffness=[stiffness, 0, 0],
            springref=_number(joint_params.get("rest_angle", 0.0), f"{joint_name}.rest_angle"),
        )
        lo, hi = (joint_params.get(key) for key in ("limit_min", "limit_max"))
        if lo is not None or hi is not None:
            lo, hi = _number(lo, f"{joint_name}.limit_min"), _number(hi, f"{joint_name}.limit_max")
            if lo >= hi:
                raise ValueError(f"Joint {joint_name!r} requires an increasing angle range")
            joint.range = [lo, hi]
            joint.limited = True
        motor = spec.add_actuator(name=joint_name)
        motor.set_to_motor()
        motor.trntype = mujoco.mjtTrn.mjTRN_JOINT
        motor.target = joint_name
        bodies[name] = body
        parent_name = name

    def attach(site: str, point: Mapping[str, object]) -> None:
        frame = point.get("frame")
        if not isinstance(frame, str) or frame not in frames:
            raise ValueError(f"Site {site!r} references missing planar frame {frame!r}")
        body_name, x = frames[frame]
        dx, dy = _vector(point.get("offset", [0, 0]), 2, f"{site}.offset")
        _add_site(bodies[body_name], site, [x + dx, dy, 0])

    markers = set()
    for name, params in groups["PointMarker"].items():
        marker = params.get("marker_name") or name
        attach(marker, params)
        markers.add(marker)
    paths = []
    for name, params in groups["MusclePath"].items():
        points = params.get("path_points")
        if not isinstance(points, list) or not all(isinstance(point, dict) for point in points):
            raise ValueError(f"MusclePath {name!r} requires an ordered list of planar points")
        sites = tuple(f"{name}/point{i}" for i in range(len(points)))
        for site, point in zip(sites, points, strict=True):
            attach(site, point)
        law = params.get("law", "feedbax.RigidTendonHillMuscle")
        moment_arm = params.get("moment_arm")
        law_parameters = params.get("law_parameters")
        if moment_arm is not None and law_parameters is not None:
            raise ValueError(f"MusclePath {name!r} mixes a constant moment arm and nested law parameters")
        if law_parameters is None:
            law_parameters = (
                {key: value for key, value in params.items()
                 if key not in {"path_points", "law", "state_size", "law_revision",
                                "law_parameters", "moment_arm"}}
                if law == "feedbax.RigidTendonHillMuscle" else {}
            )
        if moment_arm is not None:
            law_parameters = {**law_parameters, "moment_arm": moment_arm}
        paths.append(_add_muscle_path(
            spec, name, sites, law, params.get("state_size", 1), law_parameters,
            law_revision=params.get("law_revision"),
            external_tendon_actuation=external_tendon_actuation,
        ))
    return describe_model(
        spec.compile(), identity=_physical_identity(graph, graph.solver.dt), dimension=2,
        effector_site="effector" if "effector" in markers else None, paths=tuple(paths),
        planar_segment_lengths=tuple(lengths[name] for name in bodies if name != world),
    )


def compile_spatial_model(
    graph: AcausalGraphSpec, artifact_provider: ArtifactBlobProvider | None,
) -> CompiledPhysicalModel:
    """Compile an exact, self-contained MJCF asset and its graph-owned overlays."""
    import mujoco

    binding = graph.mujoco_asset
    if graph.physical_domain != "spatial_multibody" or binding is None:
        raise ValueError("Expected a spatial multibody graph with a MuJoCo asset")
    if artifact_provider is None:
        raise ValueError(f"Spatial MuJoCo asset {binding.asset.object_id} needs an artifact provider")
    _element_groups(graph, ("ActuationInput", "SensorOutput"))
    if graph.connections:
        raise ValueError("Spatial anatomy and joints belong to the MuJoCo asset")

    raw = artifact_provider.get_object(binding.asset)
    if len(raw) != binding.asset.size_bytes or sha256(raw).hexdigest() != binding.asset.sha256:
        raise ValueError(f"MuJoCo asset bytes disagree with {binding.asset.object_id}")
    xml = raw.decode("utf-8")
    if "<!DOCTYPE" in xml:
        raise ValueError("MuJoCo assets cannot declare XML document types")
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise ValueError(f"Invalid MuJoCo asset {binding.asset.object_id}: {exc}") from exc
    if root.tag != "mujoco":
        raise ValueError("MuJoCo assets require a self-contained <mujoco> document")
    for element in root.iter():
        if element.tag in {"include", "plugin", "extension"} or element.attrib.keys() & {"file", "uri"}:
            raise ValueError(f"MuJoCo asset uses an external resource in <{element.tag}>")

    spec = mujoco.MjSpec.from_string(xml)
    for name, site in sorted(binding.sites.items()):
        if spec.site(name) is not None:
            raise ValueError(f"Overlay site {name!r} duplicates a site in the MuJoCo asset")
        body = spec.body(site.body)
        if body is None:
            raise ValueError(f"Overlay site {name!r} references unknown asset body {site.body!r}")
        _add_site(body, name, site.position)
    paths = []
    for name, path in sorted(binding.paths.items()):
        if spec.tendon(name) is not None:
            raise ValueError(f"Overlay path {name!r} duplicates a tendon in the MuJoCo asset")
        missing = [site for site in path.sites if spec.site(site) is None]
        if missing:
            raise ValueError(f"Overlay path {name!r} references unknown sites {missing!r}")
        paths.append(_add_muscle_path(
            spec, name, path.sites, path.law, path.state_size, path.parameters,
            law_revision=path.law_revision,
        ))
    model = spec.compile()
    return describe_model(
        model, identity=_physical_identity(graph, model.opt.timestep), dimension=3,
        asset_identity=binding.asset.object_id, effector_site=binding.effector_site,
        paths=tuple(paths),
    )
