"""Project authored Model occurrences into world-space Scene geometry."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import pairwise
from operator import attrgetter

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import JsonValue

from feedbax.compiler.acausal_compiler import compile_acausal_authoring_report
from feedbax.compiler.model_document import lower_planar_scene_interior
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.domain import DomainDiagnostic
from feedbax.contracts.model_address import ComponentElement, FieldElement, FieldPath, ModelAddress
from feedbax.contracts.model_document import (
    MODEL_DOCUMENT_SCHEMA_ID, CompositeComponent, GraphInterior, LeafComponent, ModelDocument,
)
from feedbax.contracts.portable import StateSignal
from feedbax.contracts.project import Project
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document
from feedbax.contracts.representation import RepresentationStateAnchorSelectorBinding
from feedbax.contracts.studio_scene import (
    SceneLinkGeometry,
    SceneMechanismGeometry,
    SceneMuscleGeometry,
    ScenePlanarGeometry,
    ScenePoint,
    ScenePointGeometry,
    StudioSceneGeometryFrame,
)
from feedbax.mechanics.authored_model import compile_planar_model
from feedbax.mechanics.compiled_model import CompiledPhysicalModel
from feedbax.mechanics.mechanics import Mechanics
from feedbax.runtime.graph import GraphNode, init_state_from_component


type PoseValues = Mapping[str, JsonValue]
type MechanismPose = Callable[[PoseValues], SceneMechanismGeometry]


def _read_model(project: Project, document_id: str) -> ModelDocument:
    document = compose_document(project.documents.get, document_id)
    if isinstance(document, ProjectDiagnostic) or document.schema_id != MODEL_DOCUMENT_SCHEMA_ID:
        raise ValueError(f"{document_id!r} must name an authored Model document.")
    return ModelDocument.model_validate(document.payload)


def _compile_planar(
    model: ModelDocument,
    path: tuple[str, ...],
    registry: ComponentRegistry,
) -> tuple[CompiledPhysicalModel | None, list[DomainDiagnostic]]:
    try:
        graph = lower_planar_scene_interior(model, path, registry)
        report = compile_acausal_authoring_report(
            graph, node_path=list(path), component_registry=registry,
        )
        if report.status == "error":
            return None, [item for item in report.diagnostics if item.severity == "error"]
        return compile_planar_model(graph), []
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


def _state_key(path: tuple[str, ...], symbol: str) -> str:
    return ModelAddress.at_state(path[:-1], path[-1], symbol).to_text()


def _point(value: Iterable[float]) -> ScenePoint:
    x, y = value
    return float(x), float(y)


def _pose_value(value: JsonValue, target: jax.Array) -> jax.Array:
    pose = np.asarray(value)
    if pose.dtype.kind not in "iuf" or not np.isfinite(pose).all():
        raise ValueError("Scene pose requires a finite numeric value.")
    return jnp.broadcast_to(jnp.asarray(pose, dtype=target.dtype), target.shape)


def _prepare_portable_geometry(
    component: LeafComponent,
    path: tuple[str, ...],
    registry: ComponentRegistry,
) -> MechanismPose | None:
    declaration = registry.get(component.type)
    owner_class = declaration.owner_class if declaration is not None else None
    if owner_class is None or not issubclass(owner_class, GraphNode):
        return None
    facts = owner_class.model_facts
    if facts is None or not facts.states:
        return None

    representation = declaration.representation
    if representation is None:
        return None

    resolved = registry.resolve_component_spec(
        component.type, component.fields, revision=component.revision,
    )
    states_by_symbol = {state.symbol: state for state in facts.states}
    position = None
    for anchor in representation.anchors:
        binding = anchor.binding
        if not isinstance(binding, RepresentationStateAnchorSelectorBinding):
            continue
        selector = binding.selector
        if selector.target_id is None or selector.path is None:
            continue
        leaf = owner_class.ports.output_leaf(
            (selector.target_id, *selector.path.split(".")), resolved.params,
        )
        if isinstance(leaf, StateSignal):
            state = states_by_symbol[leaf.symbol]
            if state.space == "point":
                position = state
                break

    configurations = [state for state in facts.states if state.configuration]
    planar = any(element.archetype == "planar_chain" for element in representation.elements)
    if not planar and (position is None or configurations):
        return None

    owner = resolved.meta.owner_class(**resolved.params)
    rest = owner.state_view(init_state_from_component(owner))
    address = _component_address(path)
    position_key = _state_key(path, position.symbol) if position is not None else None
    reference_position = attrgetter(".".join(position.fields))(rest) if position is not None else None
    if not planar and position is not None:
        point = _geometry_address(path, ("point",))
        keys = (position_key, *(
            _state_key(path, companion.symbol) for companion in facts.states
            if position.fields in companion.aliases
        ))

        def pose_point(values: PoseValues) -> ScenePointGeometry:
            supplied = next(
                (value for key in keys if (value := values.get(key)) is not None),
                None,
            )
            value = _pose_value(supplied, reference_position) if supplied is not None else reference_position
            return ScenePointGeometry(
                kind="point", address=address, name=component.name, point=point, position=_point(value),
            )

        return pose_point

    if not isinstance(owner, Mechanics) or not configurations:
        raise ValueError("Planar chain geometry requires a mechanics owner with declared configuration.")
    layout = [
        (_state_key(path, leaf.symbol), leaf.fields, attrgetter(".".join(leaf.fields))(rest))
        for leaf in configurations
    ]
    skeleton_owner = owner.plant.skeleton
    rest_skeleton = rest.plant.skeleton
    rest_endpoints = skeleton_owner.forward_kinematics(rest_skeleton).pos
    links = tuple(
        (
            _geometry_address(path, ("body", index)), f"{component.name} link {index + 1}",
            _geometry_address(path, ("joint", index)), f"{component.name} joint {index + 1}",
        )
        for index in range(len(rest_endpoints))
    )

    def pose_chain(values: PoseValues) -> ScenePlanarGeometry:
        supplied = [
            (fields, reference, value) for key, fields, reference in layout
            if (value := values.get(key)) is not None
        ]
        supplied_position = values.get(position_key) if position_key is not None else None
        endpoints = rest_endpoints
        if supplied or supplied_position is not None:
            state = init_state_from_component(owner)
            if supplied:
                for fields, reference, value in supplied:
                    state = owner.replace_state_value(state, fields, _pose_value(value, reference))
            else:
                state = owner.replace_state_value(
                    state, position.fields, _pose_value(supplied_position, reference_position),
                )
            endpoints = skeleton_owner.forward_kinematics(owner.state_view(state).plant.skeleton).pos
        posed = [(0.0, 0.0), *map(_point, endpoints)]
        return ScenePlanarGeometry(
            kind="planar", address=address, name=component.name, muscles=[],
            links=[
                SceneLinkGeometry(
                    body=body, body_name=body_name, joint=joint, joint_name=joint_name,
                    start=start, end=end, pivot=start,
                )
                for (body, body_name, joint, joint_name), (start, end) in zip(
                    links, pairwise(posed), strict=True,
                )
            ],
        )

    return pose_chain


def _prepare_native_geometry(
    compiled: CompiledPhysicalModel,
    path: tuple[str, ...],
    component: CompositeComponent,
) -> tuple[MechanismPose, list[DomainDiagnostic]]:
    import mujoco

    children = component.interior.components
    address = _component_address(path)
    configuration_key = _state_key(path, "skeleton/qpos")
    native = compiled.model
    data = mujoco.MjData(native)
    body_names = {index: name for name, index in compiled.body_ids.items()}
    links = []
    for layout, length in zip(compiled.joints, compiled.planar_segment_lengths, strict=True):
        joint = compiled.joint_ids[layout.name]
        body = int(native.jnt_bodyid[joint])
        name = body_names[body]
        links.append((
            body, _component_address((*path, name)), children[name].name, length,
            joint, _component_address((*path, layout.name)), children[layout.name].name,
        ))

    diagnostics = []
    muscles = []
    for binding in compiled.paths:
        start, count = native.tendon_adr[binding.tendon_id], native.tendon_num[binding.tendon_id]
        if any(kind != mujoco.mjtWrap.mjWRAP_SITE for kind in native.wrap_type[start : start + count]):
            diagnostics.append(DomainDiagnostic(
                code="scene.unsupported_wrap", severity="warning", node_ids=[binding.name],
                message=f"Scene cannot display surface wrapping for muscle {binding.name!r}.",
            ))
        else:
            muscles.append((
                _component_address((*path, binding.name)), children[binding.name].name,
                [compiled.site_ids[site] for site in binding.sites],
            ))

    def pose_native(values: PoseValues) -> ScenePlanarGeometry:
        qpos = native.qpos0
        if configuration_key in values:
            supplied = np.asarray(values[configuration_key])
            if (supplied.shape != qpos.shape or supplied.dtype.kind not in "iuf"
                or not np.isfinite(supplied).all()):
                raise ValueError(f"Scene planar configuration requires {native.nq} finite joint coordinates.")
            qpos = supplied
        data.qpos[:] = qpos
        mujoco.mj_kinematics(native, data)
        posed_links = []
        for body, body_address, body_name, length, joint, joint_address, joint_name in links:
            start = data.xpos[body, :2]
            end = start + data.xmat[body].reshape(3, 3)[:2, 0] * length
            posed_links.append(SceneLinkGeometry(
                body=body_address, body_name=body_name, joint=joint_address, joint_name=joint_name,
                start=_point(start), end=_point(end), pivot=_point(data.xanchor[joint, :2]),
            ))
        return ScenePlanarGeometry(
            kind="planar", address=address, name=component.name, links=posed_links,
            muscles=[
                SceneMuscleGeometry(
                    address=muscle_address, name=name,
                    points=[_point(data.site_xpos[site, :2]) for site in sites],
                )
                for muscle_address, name, sites in muscles
            ],
        )

    return pose_native, diagnostics


def prepare_scene_geometry(
    project: Project,
    document_id: str,
    poses: Sequence[PoseValues],
    registry: ComponentRegistry,
) -> list[StudioSceneGeometryFrame]:
    """Prepare each model occurrence once and draw poses without realizing the surrounding graph."""
    try:
        model = _read_model(project, document_id)
    except ValueError as exc:
        return [StudioSceneGeometryFrame(mechanisms=[], diagnostics=[_invalid_model(exc)]) for _ in poses]

    mechanisms: list[tuple[tuple[str, ...], MechanismPose]] = []
    diagnostics = []

    def visit(interior: GraphInterior, scope: tuple[str, ...]) -> None:
        for name, component in interior.components.items():
            path = (*scope, name)
            if isinstance(component, LeafComponent):
                try:
                    pose = _prepare_portable_geometry(component, path, registry)
                except ValueError as exc:
                    diagnostics.append(_invalid_model(exc, path))
                else:
                    if pose is not None:
                        mechanisms.append((path, pose))
            elif isinstance(component.interior, GraphInterior):
                visit(component.interior, path)
            elif component.interior.physical_domain == "planar_multibody":
                compiled, problems = _compile_planar(model, path, registry)
                diagnostics.extend(problems)
                if compiled is not None:
                    pose, problems = _prepare_native_geometry(compiled, path, component)
                    diagnostics.extend(problems)
                    mechanisms.append((path, pose))

    visit(model.interior, ())

    frames = []
    for values in poses:
        frame = StudioSceneGeometryFrame(mechanisms=[], diagnostics=list(diagnostics))
        for path, pose in mechanisms:
            try:
                frame.mechanisms.append(pose(values))
            except ValueError as exc:
                frame.diagnostics.append(_invalid_model(exc, path))
        frames.append(frame)
    return frames
