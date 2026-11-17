"""Project native planar bindings into the Studio representation."""

from __future__ import annotations

from collections.abc import Sequence
from math import atan2, cos, sin

from feedbax.compiler.acausal_compiler import compile_acausal_authoring_report
from feedbax.compiler.model_document import lower_planar_scene_interior
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.acausal_interface import is_acausal_type
from feedbax.contracts.domain import DomainDiagnostic
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.project import Project
from feedbax.contracts.representation import RepresentationMusclePathPointSpec
from feedbax.contracts.studio_scene import ScenePlanarLink, ScenePlanarModel, ScenePlanarMuscle
from feedbax.mechanics.authored_model import compile_planar_model
from feedbax.mechanics.compiled_model import CompiledPhysicalModel


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


def prepare_project_scene(
    project: Project,
    document_id: str,
    node_path: Sequence[str],
    registry: ComponentRegistry,
) -> tuple[ScenePlanarModel | None, list[DomainDiagnostic]]:
    """Prepare geometry for one occurrence without realizing the outer graph or a run."""
    try:
        document = project.documents.get(document_id)
        if document is None or document.schema_id != MODEL_DOCUMENT_SCHEMA_ID:
            raise ValueError(f"{document_id!r} must name an authored Model document.")
        model = ModelDocument.model_validate(document.payload)
        graph = lower_planar_scene_interior(document_id, model, tuple(node_path), registry)
        report = compile_acausal_authoring_report(
            graph,
            node_path=list(node_path),
            component_registry=registry,
        )
        if report.status == "error":
            return None, [item for item in report.diagnostics if item.severity == "error"]
        compiled = compile_planar_model(graph)
    except ValueError as exc:
        return None, [
            DomainDiagnostic(
                code="scene.invalid_model",
                severity="error",
                node_ids=list(node_path[-1:]),
                message=str(exc),
            )
        ]
    except ModuleNotFoundError as exc:
        if exc.name != "mujoco":
            raise
        return None, [
            DomainDiagnostic(
                code="scene.mujoco_unavailable",
                severity="error",
                message="Planar Scene geometry requires the Feedbax MuJoCo extra.",
            )
        ]
    return _compiled_geometry(graph, compiled)
