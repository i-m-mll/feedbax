"""Project authored Model occurrences into world-space Scene geometry."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import pairwise
from math import atan2, cos, sin
from operator import attrgetter

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import JsonValue

from feedbax.compiler.acausal_compiler import compile_acausal_authoring_report
from feedbax.compiler.model_document import lower_planar_scene_interior
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.acausal_interface import is_acausal_type
from feedbax.contracts.domain import DomainDiagnostic
from feedbax.contracts.model_address import ComponentElement, FieldElement, FieldPath, ModelAddress
from feedbax.contracts.model_document import (
    MODEL_DOCUMENT_SCHEMA_ID, GraphInterior, LeafComponent, ModelDocument,
)
from feedbax.contracts.model_values import StateValue
from feedbax.contracts.project import Project
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document
from feedbax.contracts.representation import RepresentationMusclePathPointSpec
from feedbax.contracts.studio_scene import (
    SceneLinkGeometry,
    SceneMechanismGeometry,
    SceneMuscleGeometry,
    ScenePlanarGeometry,
    ScenePlanarLink,
    ScenePlanarModel,
    ScenePlanarMuscle,
    ScenePoint,
    ScenePointGeometry,
    StudioSceneGeometryRequest,
)
from feedbax.mechanics.authored_model import compile_planar_model
from feedbax.mechanics.compiled_model import CompiledPhysicalModel
from feedbax.mechanics.mechanics import Mechanics
from feedbax.runtime.graph import GraphNode, init_state_from_component


def _compiled_geometry(
    graph: AcausalGraphSpec,
    compiled: CompiledPhysicalModel,
) -> tuple[ScenePlanarModel, list[DomainDiagnostic]]:
    import mujoco

    native = compiled.model
    world_id = next(
        name for name, node in graph.nodes.items() if is_acausal_type(node.type, "WorldFrame")
    )
    world_frame = f"{world_id}.frame"
    body_names = {index: name for name, index in compiled.body_ids.items()}
    lengths_by_frame = {world_frame: 0.0}
    links = []
    for layout, length in zip(compiled.joints, compiled.planar_segment_lengths, strict=True):
        joint = compiled.joint_ids[layout.name]
        body = int(native.jnt_bodyid[joint])
        name = body_names[body]
        params = graph.nodes[layout.name].params
        w, _, _, z = native.body_quat[body]
        angle = 2 * atan2(z, w)
        x, y = (float(value) for value in native.jnt_pos[joint, :2])
        tx, ty = native.body_pos[body, :2]
        links.append(
            ScenePlanarLink(
                body_id=name,
                joint_id=layout.name,
                parent_frame=params["parent_frame"],
                child_frame=params["child_frame"],
                qpos_index=layout.qpos.start,
                parent_pivot=(
                    tx + cos(angle) * x - sin(angle) * y,
                    ty + sin(angle) * x + cos(angle) * y,
                ),
                body_pivot=(x, y),
                reference_angle=angle,
                length=length,
            )
        )
        lengths_by_frame.update({f"{name}.proximal": 0.0, f"{name}.distal": length})

    diagnostics = []
    muscles = []
    for binding in compiled.paths:
        start, count = native.tendon_adr[binding.tendon_id], native.tendon_num[binding.tendon_id]
        if any(
            kind != mujoco.mjtWrap.mjWRAP_SITE for kind in native.wrap_type[start : start + count]
        ):
            diagnostics.append(
                DomainDiagnostic(
                    code="scene.unsupported_wrap",
                    severity="warning",
                    node_ids=[binding.name],
                    message=f"Scene cannot display surface wrapping for muscle {binding.name!r}.",
                )
            )
            muscles.append(ScenePlanarMuscle(node_id=binding.name, points=None))
            continue
        points = []
        for authored, site in zip(
            graph.nodes[binding.name].params["path_points"], binding.sites, strict=True
        ):
            frame = authored["frame"]
            x, y = native.site_pos[compiled.site_ids[site], :2]
            points.append(
                RepresentationMusclePathPointSpec(
                    frame=frame,
                    position=[x - lengths_by_frame[frame], y],
                )
            )
        muscles.append(ScenePlanarMuscle(node_id=binding.name, points=points))

    return ScenePlanarModel(
        model_identity=compiled.identity,
        world_frame=world_frame,
        links=links,
        reference_qpos=native.qpos0.tolist(),
        muscles=muscles,
    ), diagnostics


def _read_model(project: Project, document_id: str) -> ModelDocument:
    document = compose_document(project.documents.get, document_id)
    if isinstance(document, ProjectDiagnostic) or document.schema_id != MODEL_DOCUMENT_SCHEMA_ID:
        raise ValueError(f"{document_id!r} must name an authored Model document.")
    return ModelDocument.model_validate(document.payload)


def _compile_planar(
    model: ModelDocument,
    path: tuple[str, ...],
    registry: ComponentRegistry,
) -> tuple[tuple[AcausalGraphSpec, CompiledPhysicalModel] | None, list[DomainDiagnostic]]:
    try:
        graph = lower_planar_scene_interior(model, path, registry)
        report = compile_acausal_authoring_report(
            graph, node_path=list(path), component_registry=registry,
        )
        if report.status == "error":
            return None, [item for item in report.diagnostics if item.severity == "error"]
        return (graph, compile_planar_model(graph)), []
    except ValueError as exc:
        return None, [_invalid_model(exc, path)]
    except ModuleNotFoundError as exc:
        if exc.name != "mujoco":
            raise
        return None, [
            DomainDiagnostic(
                code="scene.mujoco_unavailable",
                message="Planar Scene geometry requires the Feedbax MuJoCo extra.",
            )
        ]


def _invalid_model(error: ValueError, path: Sequence[str] = ()) -> DomainDiagnostic:
    return DomainDiagnostic(code="scene.invalid_model", node_ids=list(path[-1:]), message=str(error))


def prepare_project_scene(
    project: Project,
    document_id: str,
    node_path: Sequence[str],
    registry: ComponentRegistry,
) -> tuple[ScenePlanarModel | None, list[DomainDiagnostic]]:
    """Prepare geometry for one occurrence without realizing the outer graph or a run."""
    try:
        model = _read_model(project, document_id)
    except ValueError as exc:
        return None, [_invalid_model(exc, node_path)]
    compiled, diagnostics = _compile_planar(model, tuple(node_path), registry)
    if compiled is None:
        return None, diagnostics
    return _compiled_geometry(*compiled)


def _component_address(path: tuple[str, ...]) -> ModelAddress:
    return ModelAddress(
        composites=path[:-1], element=ComponentElement(kind="component", id=path[-1]),
    )


def _geometry_address(path: tuple[str, ...], field: FieldPath) -> ModelAddress:
    return ModelAddress(
        composites=path[:-1],
        element=FieldElement(
            kind="field", parent=ComponentElement(kind="component", id=path[-1]),
            path=("geometry", *field),
        ),
    )


def _state_value(
    path: tuple[str, ...], state: StateValue, request: StudioSceneGeometryRequest,
) -> JsonValue:
    address = ModelAddress.at_state(path[:-1], path[-1], state.symbol)
    return request.state_values.get(address.to_text())


def _point(value: Iterable[float]) -> ScenePoint:
    x, y = value
    return float(x), float(y)


def _pose_value(value: JsonValue, shape: tuple[int, ...]) -> jax.Array:
    pose = np.asarray(value)
    if pose.shape != shape or pose.dtype.kind not in "iuf" or not np.isfinite(pose).all():
        raise ValueError(f"Scene pose requires a finite numeric value of shape {shape}.")
    return jnp.asarray(pose)


def _portable_geometry(
    component: LeafComponent,
    path: tuple[str, ...],
    request: StudioSceneGeometryRequest,
    registry: ComponentRegistry,
) -> SceneMechanismGeometry | None:
    declaration = registry.get(component.type)
    owner_class = declaration.owner_class if declaration is not None else None
    if owner_class is None or not issubclass(owner_class, GraphNode):
        return None
    facts = owner_class.model_facts
    if facts is None or not facts.states:
        return None

    points = [state for state in facts.states if state.space == "point" and state.frame == "world"]
    writable_points = [state for state in points if state.reset is not None]
    preferred_points = writable_points if len(points) > 1 else points
    position = preferred_points[0] if len(preferred_points) == 1 else None
    configurations = [state for state in facts.states if state.configuration]
    representation = declaration.representation
    planar = representation is not None and any(
        element.archetype == "planar_chain" for element in representation.elements
    )
    if not planar and (position is None or configurations):
        return None

    resolved = registry.resolve_component_spec(
        component.type, component.fields, revision=component.revision,
    )
    owner = resolved.meta.owner_class(**resolved.params)
    state = init_state_from_component(owner)
    address = _component_address(path)
    supplied_position = _state_value(path, position, request) if position is not None else None
    if not planar and position is not None:
        value = (
            _pose_value(supplied_position, (2,)) if supplied_position is not None
            else attrgetter(".".join(position.fields))(owner.state_view(state))
        )
        return ScenePointGeometry(
            kind="point", address=address, point=_geometry_address(path, ("point",)),
            position=_point(value),
        )

    if not isinstance(owner, Mechanics) or not configurations:
        raise ValueError("Planar chain geometry requires a mechanics owner with declared configuration.")
    supplied_configuration = [
        (leaf, value) for leaf in configurations
        if (value := _state_value(path, leaf, request)) is not None
    ]
    if supplied_configuration:
        rest = owner.state_view(state)
        for leaf, value in supplied_configuration:
            original = attrgetter(".".join(leaf.fields))(rest)
            sample = _pose_value(value, original.shape)
            state = owner.replace_state_value(state, leaf.fields, sample.astype(original.dtype))
    elif position is not None and supplied_position is not None:
        state = owner.replace_state_value(state, position.fields, _pose_value(supplied_position, (2,)))

    skeleton = owner.state_view(state).plant.skeleton
    endpoints = owner.plant.skeleton.forward_kinematics(skeleton).pos
    posed_points = [(0.0, 0.0), *map(_point, endpoints)]
    return ScenePlanarGeometry(
        kind="planar", address=address, muscles=[],
        links=[
            SceneLinkGeometry(
                body=_geometry_address(path, ("body", index)),
                joint=_geometry_address(path, ("joint", index)),
                start=start, end=end, pivot=start,
            )
            for index, (start, end) in enumerate(pairwise(posed_points))
        ],
    )


def _native_geometry(
    graph: AcausalGraphSpec,
    compiled: CompiledPhysicalModel,
    path: tuple[str, ...],
) -> tuple[ScenePlanarGeometry, list[DomainDiagnostic]]:
    import mujoco

    layout, diagnostics = _compiled_geometry(graph, compiled)
    native = compiled.model
    data = mujoco.MjData(native)
    mujoco.mj_kinematics(native, data)
    links = []
    for link in layout.links:
        body = compiled.body_ids[link.body_id]
        start = data.xpos[body, :2]
        end = start + data.xmat[body].reshape(3, 3)[:2, 0] * link.length
        links.append(SceneLinkGeometry(
            body=_component_address((*path, link.body_id)),
            joint=_component_address((*path, link.joint_id)),
            start=_point(start),
            end=_point(end),
            pivot=_point(data.xanchor[compiled.joint_ids[link.joint_id], :2]),
        ))
    muscles = [
        SceneMuscleGeometry(
            address=_component_address((*path, binding.name)),
            points=[_point(data.site_xpos[compiled.site_ids[site], :2]) for site in binding.sites],
        )
        for binding, muscle in zip(compiled.paths, layout.muscles, strict=True)
        if muscle.points is not None
    ]
    return ScenePlanarGeometry(
        kind="planar", address=_component_address(path), links=links, muscles=muscles,
    ), diagnostics


def prepare_scene_geometry(
    project: Project,
    request: StudioSceneGeometryRequest,
    registry: ComponentRegistry,
) -> tuple[list[SceneMechanismGeometry], list[DomainDiagnostic]]:
    """Project the Model's drawable occurrences without realizing the surrounding graph."""
    try:
        model = _read_model(project, request.document_id)
    except ValueError as exc:
        return [], [_invalid_model(exc)]

    mechanisms = []
    diagnostics = []

    def visit(interior: GraphInterior, scope: tuple[str, ...]) -> None:
        for name, component in interior.components.items():
            path = (*scope, name)
            if isinstance(component, LeafComponent):
                try:
                    geometry = _portable_geometry(component, path, request, registry)
                except ValueError as exc:
                    diagnostics.append(_invalid_model(exc, path))
                else:
                    if geometry is not None:
                        mechanisms.append(geometry)
            elif isinstance(component.interior, GraphInterior):
                visit(component.interior, path)
            elif component.interior.physical_domain == "planar_multibody":
                compiled, problems = _compile_planar(model, path, registry)
                diagnostics.extend(problems)
                if compiled is not None:
                    geometry, problems = _native_geometry(*compiled, path)
                    mechanisms.append(geometry)
                    diagnostics.extend(problems)

    visit(model.interior, ())
    return mechanisms, diagnostics
