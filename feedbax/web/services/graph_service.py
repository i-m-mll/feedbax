from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import json
import os
import uuid

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.domain import DomainCompileReport
from feedbax.compiler.acausal_compiler import compile_acausal_authoring_report
from feedbax.compiler.penzai_compiler import compile_penzai_authoring_report
from feedbax.web.config import GRAPHS_DIR, ensure_dirs
from feedbax.compiler.normalization import (
    normalize_graph_for_studio_authoring,
    normalize_project_for_studio_authoring,
    normalize_workspace_for_studio_authoring,
)
from feedbax.contracts.migrations import migrate_graph_project_payload
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.domain import DomainDiagnostic
from feedbax.contracts.graph import (
    ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
    ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
    AnalysisCanvasLayoutDocument,
    AnalysisPageSpec,
    GraphProject,
    GraphSpec,
    GraphUIState,
    GraphMetadata,
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


class GraphSaveConflictError(RuntimeError):
    """Raised when a Studio save does not match the current project revision."""

    def __init__(
        self,
        *,
        graph_id: str,
        current_revision: int,
        expected_revision: Optional[int],
    ) -> None:
        self.graph_id = graph_id
        self.current_revision = current_revision
        self.expected_revision = expected_revision
        if expected_revision is None:
            message = (
                f"Graph {graph_id} save is missing an optimistic-concurrency revision; "
                f"current revision is {current_revision}."
            )
        else:
            message = (
                f"Graph {graph_id} save revision {expected_revision} is stale; "
                f"current revision is {current_revision}."
            )
        super().__init__(message)


class GraphService:
    def __init__(self, storage_dir: Path = GRAPHS_DIR) -> None:
        self._storage_dir = storage_dir
        ensure_dirs()

    def list_graphs(self, *, component_registry: object | None = None) -> List[dict]:
        ensure_dirs()
        results: List[dict] = []
        for path in self._storage_dir.glob("*.json"):
            project = self._load_project(path, component_registry=component_registry)
            results.append({"id": path.stem, "metadata": project.metadata})
        return results

    def create_graph(
        self,
        graph: GraphSpec,
        *,
        workspace: StudioWorkspaceSpec | None = None,
        workspace_document: WorkspaceDocument | None = None,
        component_registry: object | None = None,
    ) -> GraphRecord:
        ensure_dirs()
        graph = normalize_graph_for_studio_authoring(
            graph,
            component_registry=component_registry,
        )
        graph_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        metadata = graph.metadata or GraphMetadata(
            name="Untitled Graph",
            description=None,
            created_at=now,
            updated_at=now,
            version="1.0.0",
        )
        if graph.metadata is None:
            graph.metadata = metadata
        semantic_workspace = workspace or build_default_studio_workspace(label=metadata.name)
        semantic_workspace = normalize_workspace_for_studio_authoring(
            semantic_workspace,
            graph,
            component_registry=component_registry,
        )
        presentation = self._workspace_document(
            graph,
            workspace=semantic_workspace,
            graph_ui_state=(workspace_document.graph_ui_state if workspace_document else None),
            workspace_ui_state=(
                workspace_document.workspace_ui_state if workspace_document else None
            ),
            stage_ui_state=(workspace_document.stage_ui_state if workspace_document else None),
            scenario_ui_state=(
                workspace_document.scenario_ui_state if workspace_document else None
            ),
            analysis_pages=(workspace_document.analysis_pages if workspace_document else None),
            active_analysis_page_id=(
                workspace_document.active_analysis_page_id if workspace_document else None
            ),
            analysis_canvas_layout=(
                workspace_document.analysis_canvas_layout if workspace_document else None
            ),
            component_registry=component_registry,
        )
        project = GraphProject(
            metadata=metadata,
            graph=graph,
            workspace_document=presentation,
            workspace=semantic_workspace,
        )
        self._save_project(self._path_for(graph_id), project)
        return GraphRecord(graph_id=graph_id, project=project)

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

    def update_graph(
        self,
        graph_id: str,
        graph: Optional[GraphSpec],
        *,
        workspace: Optional[StudioWorkspaceSpec] = None,
        workspace_document: Optional[WorkspaceDocument] = None,
        expected_save_revision: Optional[int] = None,
        require_save_revision: bool = False,
        component_registry: object | None = None,
    ) -> GraphRecord:
        record = self.get_graph(graph_id, component_registry=component_registry)
        project = record.project
        current_revision = project.metadata.save_revision
        if require_save_revision and expected_save_revision is None:
            raise GraphSaveConflictError(
                graph_id=graph_id,
                current_revision=current_revision,
                expected_revision=None,
            )
        if expected_save_revision is not None and expected_save_revision != current_revision:
            raise GraphSaveConflictError(
                graph_id=graph_id,
                current_revision=current_revision,
                expected_revision=expected_save_revision,
            )
        graph_changed = False
        if graph is not None:
            normalized_graph = normalize_graph_for_studio_authoring(
                graph,
                component_registry=component_registry,
            )
            graph_changed = normalized_graph != project.graph
            project.graph = normalized_graph
        presentation = workspace_document or project.workspace_document
        if workspace is not None:
            project.workspace = normalize_workspace_for_studio_authoring(
                workspace,
                project.graph,
                component_registry=component_registry,
            )
        updated_at = datetime.now(timezone.utc).isoformat()
        next_revision = current_revision + 1
        project.metadata.updated_at = updated_at
        project.metadata.save_revision = next_revision
        if graph_changed and project.graph.metadata is not None:
            project.graph.metadata.updated_at = updated_at
            project.graph.metadata.save_revision = next_revision
        project.workspace_document = self._workspace_document(
            project.graph,
            workspace=project.workspace,
            graph_ui_state=presentation.graph_ui_state,
            workspace_ui_state=presentation.workspace_ui_state,
            stage_ui_state=presentation.stage_ui_state,
            scenario_ui_state=presentation.scenario_ui_state,
            analysis_pages=presentation.analysis_pages,
            active_analysis_page_id=presentation.active_analysis_page_id,
            analysis_canvas_layout=presentation.analysis_canvas_layout,
            component_registry=component_registry,
        )
        self._ensure_workspace(project, component_registry=component_registry)
        self._save_project(self._path_for(graph_id), project)
        return GraphRecord(graph_id=graph_id, project=project)

    def delete_graph(self, graph_id: str) -> None:
        path = self._path_for(graph_id)
        if path.exists():
            path.unlink()

    def validate_graph(self, graph: GraphSpec) -> list[DomainDiagnostic]:
        diagnostics: list[DomainDiagnostic] = []

        for node_name, node in graph.nodes.items():
            for input_port in node.input_ports:
                has_wire = any(
                    w.target_node == node_name and w.target_port == input_port for w in graph.wires
                )
                has_binding = any(
                    binding == (node_name, input_port) for binding in graph.input_bindings.values()
                )
                if not has_wire and not has_binding:
                    diagnostics.append(
                        DomainDiagnostic(
                            severity="error",
                            code="graph.missing_input",
                            message=f"Input port '{node_name}.{input_port}' is not connected",
                            node_ids=[node_name],
                            location={"node": node_name, "port": input_port},
                            details={"source_type": "missing_input"},
                        )
                    )

            for output_port in node.output_ports:
                has_wire = any(
                    w.source_node == node_name and w.source_port == output_port for w in graph.wires
                )
                has_binding = any(
                    binding == (node_name, output_port)
                    for binding in graph.output_bindings.values()
                )
                if not has_wire and not has_binding:
                    diagnostics.append(
                        DomainDiagnostic(
                            severity="warning",
                            code="graph.unconnected_output",
                            message=f"Output port '{node_name}.{output_port}' is not connected",
                            node_ids=[node_name],
                            location={"node": node_name, "port": output_port},
                            details={"source_type": "unconnected_output"},
                        )
                    )

        cycles = self._detect_cycles(graph)
        for cycle in cycles:
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="graph.same_step_cycle",
                    message=(
                        "Instant wires contain a same-step cycle; mark one cycle edge recurrent"
                    ),
                    node_ids=cycle,
                    details={"cycle": cycle},
                )
            )
        return diagnostics

    def compile_node(
        self,
        graph_id: str,
        *,
        node_path: list[str],
        interior: AcausalGraphSpec,
        component_registry: object,
    ) -> DomainCompileReport:
        record = self.get_graph(graph_id)
        report = compile_acausal_authoring_report(
            interior,
            node_path=node_path,
            component_registry=component_registry,
        )
        key = "/".join(node_path)
        record.project.compile_reports = {
            **(record.project.compile_reports or {}),
            key: report,
        }
        self._save_project(self._path_for(graph_id), record.project)
        return report

    def compile_penzai_node(
        self,
        graph_id: str,
        *,
        node_path: list[str],
        builder_name: str,
        params: dict[str, object],
        input_port: str,
        output_port: str,
    ) -> DomainCompileReport:
        record = self.get_graph(graph_id)
        report = compile_penzai_authoring_report(
            builder_name=builder_name,
            params=params,
            input_port=input_port,
            output_port=output_port,
            node_path=node_path,
        )
        key = "/".join(node_path)
        record.project.compile_reports = {
            **(record.project.compile_reports or {}),
            key: report,
        }
        self._save_project(self._path_for(graph_id), record.project)
        return report

    def export_graph(self, graph_id: str, export_format: str) -> dict:
        record = self.get_graph(graph_id)
        if export_format == "json":
            content = record.project.model_dump_json(indent=2)
            filename = f"{record.project.metadata.name}.json"
            return {"content": content, "filename": filename}
        if export_format == "python":
            content = (
                "# TODO: Implement export to Python once feedbax.runtime.graph is available.\n"
                f"# Graph id: {graph_id}\n"
            )
            filename = f"{record.project.metadata.name}.py"
            return {"content": content, "filename": filename}
        raise ValueError("Unsupported format")

    def _path_for(self, graph_id: str) -> Path:
        return self._storage_dir / f"{graph_id}.json"

    def _load_project(
        self,
        path: Path,
        *,
        component_registry: object | None = None,
    ) -> GraphProject:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        canonical_json_v2_bytes(data)
        data = migrate_graph_project_payload(data)
        project = normalize_project_for_studio_authoring(
            GraphProject.model_validate(data),
            component_registry=component_registry,
        )
        self._ensure_workspace(project, component_registry=component_registry)
        return project

    def _save_project(self, path: Path, project: GraphProject) -> None:
        normalized = project.model_dump(mode="json")
        canonical_json_v2_bytes(normalized)
        validated = GraphProject.model_validate(normalized)
        serialized = json.dumps(
            validated.model_dump(mode="json"),
            indent=2,
            allow_nan=False,
        )
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(temporary, "w", encoding="utf-8") as file:
                file.write(serialized)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

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
        project.workspace = normalize_workspace_for_studio_authoring(
            project.workspace,
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
            compilation = compile_graph(document, component_registry)
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

    def _detect_cycles(self, graph: GraphSpec) -> List[List[str]]:
        adjacency = {node_name: set() for node_name in graph.nodes}
        for wire in graph.wires:
            if wire.temporality == "recurrent":
                continue
            adjacency.setdefault(wire.source_node, set()).add(wire.target_node)

        cycles: List[List[str]] = []
        visited: set[str] = set()
        recursion_stack: set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            recursion_stack.add(node)
            path.append(node)

            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in recursion_stack:
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:])

            path.pop()
            recursion_stack.discard(node)

        for node in graph.nodes:
            if node not in visited:
                dfs(node)

        return cycles
