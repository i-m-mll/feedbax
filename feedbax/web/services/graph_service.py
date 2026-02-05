from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import json
import uuid

from feedbax.web.config import GRAPHS_DIR, ensure_dirs
from feedbax.web.models.graph import (
    GraphProject,
    GraphSpec,
    GraphUIState,
    GraphMetadata,
    ValidationError,
    ValidationResult,
    ValidationWarning,
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
        project = GraphProject(metadata=metadata, graph=graph, ui_state=ui_state)
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
    ) -> GraphRecord:
        record = self.get_graph(graph_id)
        project = record.project
        if graph is not None:
            project.graph = graph
        if ui_state is not None:
            project.ui_state = ui_state
        updated_at = datetime.now(timezone.utc).isoformat()
        project.metadata.updated_at = updated_at
        if project.graph.metadata is not None:
            project.graph.metadata.updated_at = updated_at
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
        return GraphProject.model_validate(data)

    def _save_project(self, path: Path, project: GraphProject) -> None:
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(project.model_dump(), file, indent=2)

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
