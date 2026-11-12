from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List
import json

from feedbax.compiler.graph import GraphCompilationError
from feedbax.web.config import GRAPHS_DIR
from feedbax.web.model_assets import model_artifact_provider
from feedbax.compiler.normalization import (
    normalize_graph_for_studio_authoring,
    normalize_project_for_studio_authoring,
)
from feedbax.contracts.migrations import (
    migrate_graph_project_payload,
)
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.graph import (
    ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
    ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
    AnalysisCanvasLayoutDocument,
    AnalysisPageSpec,
    GraphProject,
    GraphSpec,
    GraphUIState,
    StudioWorkspaceSpec,
    SemanticAnchor,
    WorkspaceDocument,
    build_default_studio_workspace,
    studio_semantic_document_sha256,
)
from feedbax.compiler import GraphDocument, compile_graph


@dataclass
class GraphRecord:
    graph_id: str
    project: GraphProject


def _read_project_payload(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("The project document must be a JSON object")
    return payload


def _admit_project(
    payload: dict[str, Any], *, component_registry: object | None
) -> GraphProject:
    canonical_json_v2_bytes(payload)
    migrated = migrate_graph_project_payload(payload, component_registry=component_registry)
    return GraphProject.model_validate(migrated)


class GraphService:
    def __init__(self, storage_dir: Path = GRAPHS_DIR) -> None:
        self._storage_dir = storage_dir

    def get_graph(
        self,
        graph_id: str,
        *,
        component_registry: object | None = None,
    ) -> GraphRecord:
        project = self._load_project(
            self._path_for(graph_id),
            component_registry=component_registry,
        )
        return GraphRecord(graph_id=graph_id, project=project)

    def _path_for(self, graph_id: str) -> Path:
        return self._storage_dir / f"{graph_id}.json"

    def _load_project(
        self,
        path: Path,
        *,
        component_registry: object | None = None,
    ) -> GraphProject:
        project = normalize_project_for_studio_authoring(
            _admit_project(_read_project_payload(path), component_registry=component_registry),
            component_registry=component_registry,
            component_versions="historical",
        )
        self._ensure_workspace(project, component_registry=component_registry)
        return project

    def _ensure_workspace(
        self,
        project: GraphProject,
        *,
        component_registry: object | None = None,
    ) -> None:
        project.graph = normalize_graph_for_studio_authoring(
            project.graph,
            component_registry=component_registry,
        )
        current_workspace = project.workspace_document
        expected_root = self._workspace_document(
            project.graph,
            workspace=project.workspace,
            graph_ui_state=current_workspace.graph_ui_state,
            workspace_ui_state=current_workspace.workspace_ui_state,
            stage_ui_state=current_workspace.stage_ui_state,
            scenario_ui_state=current_workspace.scenario_ui_state,
            analysis_pages=current_workspace.analysis_pages,
            active_analysis_page_id=current_workspace.active_analysis_page_id,
            analysis_canvas_layout=current_workspace.analysis_canvas_layout,
            component_registry=component_registry,
        )
        if current_workspace.semantic_root != expected_root.semantic_root or (
            component_registry is not None
            and current_workspace.semantic_anchors != expected_root.semantic_anchors
        ):
            project.workspace_document = expected_root
        if project.workspace is not None:
            return
        project.workspace = build_default_studio_workspace(
            label=project.metadata.name,
            analysis_pages=project.workspace_document.analysis_pages,
            active_analysis_page_id=project.workspace_document.active_analysis_page_id,
        )

    def _workspace_document(
        self,
        graph: GraphSpec,
        *,
        workspace: StudioWorkspaceSpec | None = None,
        graph_ui_state: GraphUIState | None = None,
        workspace_ui_state: dict[str, object] | None = None,
        stage_ui_state: dict[str, dict[str, object]] | None = None,
        scenario_ui_state: dict[str, dict[str, object]] | None = None,
        analysis_pages: List[AnalysisPageSpec] | None = None,
        active_analysis_page_id: str | None = None,
        analysis_canvas_layout: AnalysisCanvasLayoutDocument | None = None,
        component_registry: object | None = None,
    ) -> WorkspaceDocument:
        document = GraphDocument(graph=graph)
        document_sha256 = studio_semantic_document_sha256(graph, workspace)
        semantic_anchors: dict[str, SemanticAnchor] = {}
        if component_registry is not None:
            try:
                compilation = compile_graph(
                    document, component_registry, artifact_provider=model_artifact_provider(graph),
                )
            except GraphCompilationError:
                # Incomplete drafts are documents even when no resolved model exists.
                pass
            else:
                semantic_anchors = {
                    entry.resolved_path: SemanticAnchor(
                        semantic_document_sha256=document_sha256,
                        authored_path=entry.authored_anchor.authored_path,
                    )
                    for entry in compilation.record.source_map.entries
                }
        return WorkspaceDocument(
            semantic_root=SemanticAnchor(
                semantic_document_sha256=document_sha256,
                authored_path="/graph",
            ),
            graph_ui_state=graph_ui_state or GraphUIState(),
            workspace_ui_state=workspace_ui_state or {},
            stage_ui_state=stage_ui_state or {},
            scenario_ui_state=scenario_ui_state or {},
            analysis_pages=analysis_pages or [],
            active_analysis_page_id=active_analysis_page_id,
            analysis_canvas_layout=analysis_canvas_layout
            or AnalysisCanvasLayoutDocument(
                schema_id=ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
                schema_version=ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
            ),
            semantic_anchors=semantic_anchors,
        )
