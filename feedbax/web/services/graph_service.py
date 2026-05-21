from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import json
import uuid

from feedbax.web.config import GRAPHS_DIR, ensure_dirs
from feedbax.web.graph_normalization import (
    normalize_graph_for_studio_authoring,
    normalize_project_for_studio_authoring,
    normalize_workspace_for_studio_authoring,
)
from feedbax.web.models.graph import (
    AnalysisPageSpec,
    GraphProject,
    GraphSpec,
    GraphUIState,
    GraphMetadata,
    StudioWorkspaceSpec,
    ValidationError,
    ValidationResult,
    ValidationWarning,
    build_default_studio_workspace,
)


@dataclass
class GraphRecord:
    graph_id: str
    project: GraphProject


class GraphService:
    def __init__(self, storage_dir: Path = GRAPHS_DIR) -> None:
        self._storage_dir = storage_dir
        ensure_dirs()

    def list_graphs(self) -> List[dict]:
        ensure_dirs()
        results: List[dict] = []
        for path in self._storage_dir.glob('*.json'):
            project = self._load_project(path)
            results.append({'id': path.stem, 'metadata': project.metadata})
        return results

    def create_graph(self, graph: GraphSpec, ui_state: Optional[GraphUIState]) -> GraphRecord:
        ensure_dirs()
        graph = normalize_graph_for_studio_authoring(graph)
        graph_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        metadata = graph.metadata or GraphMetadata(
            name='Untitled Graph',
            description=None,
            created_at=now,
            updated_at=now,
            version='1.0.0',
        )
        if graph.metadata is None:
            graph.metadata = metadata
        workspace = build_default_studio_workspace(
            label=metadata.name,
            graph=graph,
            ui_state=ui_state,
        )
        project = GraphProject(
            metadata=metadata,
            graph=graph,
            ui_state=ui_state,
            workspace=workspace,
        )
        self._save_project(self._path_for(graph_id), project)
        return GraphRecord(graph_id=graph_id, project=project)

    def get_graph(self, graph_id: str) -> GraphRecord:
        project = self._load_project(self._path_for(graph_id))
        return GraphRecord(graph_id=graph_id, project=project)

    def update_graph(
        self,
        graph_id: str,
        graph: Optional[GraphSpec],
        ui_state: Optional[GraphUIState],
        analysis_pages: Optional[List[AnalysisPageSpec]] = None,
        active_analysis_page_id: Optional[str] = None,
        workspace: Optional[StudioWorkspaceSpec] = None,
    ) -> GraphRecord:
        record = self.get_graph(graph_id)
        project = record.project
        if graph is not None:
            project.graph = normalize_graph_for_studio_authoring(graph)
        if ui_state is not None:
            project.ui_state = ui_state
        if analysis_pages is not None:
            project.analysis_pages = analysis_pages
        if active_analysis_page_id is not None:
            project.active_analysis_page_id = active_analysis_page_id
        if workspace is not None:
            project.workspace = normalize_workspace_for_studio_authoring(workspace)
        updated_at = datetime.now(timezone.utc).isoformat()
        project.metadata.updated_at = updated_at
        if project.graph.metadata is not None:
            project.graph.metadata.updated_at = updated_at
        self._ensure_workspace(project)
        self._save_project(self._path_for(graph_id), project)
        return GraphRecord(graph_id=graph_id, project=project)

    def delete_graph(self, graph_id: str) -> None:
        path = self._path_for(graph_id)
        if path.exists():
            path.unlink()

    def validate_graph(self, graph: GraphSpec) -> ValidationResult:
        errors: List[ValidationError] = []
        warnings: List[ValidationWarning] = []

        for node_name, node in graph.nodes.items():
            for input_port in node.input_ports:
                has_wire = any(
                    w.target_node == node_name and w.target_port == input_port
                    for w in graph.wires
                )
                has_binding = any(
                    binding == (node_name, input_port) for binding in graph.input_bindings.values()
                )
                if not has_wire and not has_binding:
                    errors.append(
                        ValidationError(
                            type='missing_input',
                            message=f"Input port '{node_name}.{input_port}' is not connected",
                            location={'node': node_name, 'port': input_port},
                        )
                    )

            for output_port in node.output_ports:
                has_wire = any(
                    w.source_node == node_name and w.source_port == output_port
                    for w in graph.wires
                )
                has_binding = any(
                    binding == (node_name, output_port) for binding in graph.output_bindings.values()
                )
                if not has_wire and not has_binding:
                    warnings.append(
                        ValidationWarning(
                            type='unconnected_output',
                            message=f"Output port '{node_name}.{output_port}' is not connected",
                            location={'node': node_name, 'port': output_port},
                        )
                    )

        cycles = self._detect_cycles(graph)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cycles=cycles,
        )

    def export_graph(self, graph_id: str, export_format: str) -> dict:
        record = self.get_graph(graph_id)
        if export_format == 'json':
            content = record.project.model_dump_json(indent=2)
            filename = f"{record.project.metadata.name}.json"
            return {'content': content, 'filename': filename}
        if export_format == 'python':
            content = (
                "# TODO: Implement export to Python once feedbax.graph is available.\n"
                f"# Graph id: {graph_id}\n"
            )
            filename = f"{record.project.metadata.name}.py"
            return {'content': content, 'filename': filename}
        raise ValueError('Unsupported format')

    def _path_for(self, graph_id: str) -> Path:
        return self._storage_dir / f"{graph_id}.json"

    def _load_project(self, path: Path) -> GraphProject:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        project = normalize_project_for_studio_authoring(GraphProject.model_validate(data))
        self._ensure_workspace(project)
        return project

    def _save_project(self, path: Path, project: GraphProject) -> None:
        self._ensure_workspace(project)
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(project.model_dump(), file, indent=2)

    def _ensure_workspace(self, project: GraphProject) -> None:
        project.graph = normalize_graph_for_studio_authoring(project.graph)
        project.workspace = normalize_workspace_for_studio_authoring(project.workspace)
        if project.workspace is not None:
            return
        project.workspace = build_default_studio_workspace(
            label=project.metadata.name,
            graph=project.graph,
            ui_state=project.ui_state,
            analysis_pages=project.analysis_pages,
            active_analysis_page_id=project.active_analysis_page_id,
        )

    def _detect_cycles(self, graph: GraphSpec) -> List[List[str]]:
        adjacency = {node_name: set() for node_name in graph.nodes}
        for wire in graph.wires:
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
