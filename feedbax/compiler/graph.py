"""One staged entry point from authored graph semantics to JAX execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from feedbax.contracts.authored_canonical import canonical_sha256
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.graphs.prototypes import (
    normalize_derived_dimensions,
    normalize_stateful_prototypes,
)
from feedbax.contracts.graphs.serialization import _instantiate_graph
from feedbax.contracts.migrations import migrate_graph_spec
from feedbax.runtime.graph import Graph
from feedbax.runtime.graph_channel_adapters import materialize_additive_channel_adapters


GRAPH_COMPILER_ID = "feedbax.graph.compiler"
GRAPH_COMPILER_VERSION = "1"
GRAPH_DOCUMENT_SCHEMA_ID = "feedbax.graph_document"
GRAPH_DOCUMENT_SCHEMA_VERSION = "1"
RESOLVED_GRAPH_SCHEMA_ID = "feedbax.resolved_graph"
RESOLVED_GRAPH_SCHEMA_VERSION = "1"
COMPILATION_RECORD_SCHEMA_ID = "feedbax.graph_compilation_record"
COMPILATION_RECORD_SCHEMA_VERSION = "1"
KEY_SCHEDULE_ID = "feedbax.graph_key_schedule.execution_order_split.v1"


class DocumentRoot(BaseModel):
    """Content-pinned semantic root bound to a graph document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: str
    schema_version: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GraphDocument(BaseModel):
    """Complete authored model semantics supplied to the graph compiler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal[GRAPH_DOCUMENT_SCHEMA_ID] = GRAPH_DOCUMENT_SCHEMA_ID
    schema_version: str = GRAPH_DOCUMENT_SCHEMA_VERSION
    graph: GraphSpec | dict[str, Any]
    trial_root: DocumentRoot | None = None
    objective_root: DocumentRoot | None = None

    @field_validator("schema_version")
    @classmethod
    def reject_unsupported_version(cls, value: str) -> str:
        return _require_version("GraphDocument", value, GRAPH_DOCUMENT_SCHEMA_VERSION)


class GraphSourceMapEntry(BaseModel):
    """Mapping from one resolved element to its authored JSON pointer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolved_path: str
    authored_path: str
    origin: Literal["authored", "compiler-generated"] = "authored"


class GraphSourceMap(BaseModel):
    """Complete deterministic source map for a resolved graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[GraphSourceMapEntry, ...]


class GraphKeySchedule(BaseModel):
    """Versioned contract for assigning per-step keys to graph nodes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schedule_id: Literal[KEY_SCHEDULE_ID] = KEY_SCHEDULE_ID
    node_order: tuple[str, ...]


class ResolvedGraph(BaseModel):
    """Normalized immutable semantic graph ready for process-local realization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal[RESOLVED_GRAPH_SCHEMA_ID] = RESOLVED_GRAPH_SCHEMA_ID
    schema_version: str = RESOLVED_GRAPH_SCHEMA_VERSION
    graph: GraphSpec
    source_map: GraphSourceMap
    key_schedule: GraphKeySchedule
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("schema_version")
    @classmethod
    def reject_unsupported_version(cls, value: str) -> str:
        return _require_version("ResolvedGraph", value, RESOLVED_GRAPH_SCHEMA_VERSION)

    @model_validator(mode="after")
    def validate_identity(self) -> "ResolvedGraph":
        expected = _resolved_digest(self.graph, self.source_map, self.key_schedule)
        if self.resolved_sha256 != expected:
            raise ValueError(
                "ResolvedGraph resolved_sha256 does not match its canonical semantic payload"
            )
        return self


class CompilationRecord(BaseModel):
    """Durable evidence describing one deterministic graph compilation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal[COMPILATION_RECORD_SCHEMA_ID] = COMPILATION_RECORD_SCHEMA_ID
    schema_version: str = COMPILATION_RECORD_SCHEMA_VERSION
    compiler_id: Literal[GRAPH_COMPILER_ID] = GRAPH_COMPILER_ID
    compiler_version: Literal[GRAPH_COMPILER_VERSION] = GRAPH_COMPILER_VERSION
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_map: GraphSourceMap
    key_schedule: GraphKeySchedule
    graph_schema_source_version: str
    graph_schema_target_version: str
    migration_record_count: int = Field(ge=0)

    @field_validator("schema_version")
    @classmethod
    def reject_unsupported_version(cls, value: str) -> str:
        return _require_version(
            "CompilationRecord", value, COMPILATION_RECORD_SCHEMA_VERSION
        )


@dataclass(frozen=True)
class ExecutableGraph:
    """Process-local graph realization plus its semantic compilation evidence."""

    graph: Graph
    resolved: ResolvedGraph
    record: CompilationRecord


def _require_version(family: str, value: str, current: str) -> str:
    if value != current:
        raise ValueError(
            f"Unsupported {family} schema version: source_version={value!r}; "
            f"current_version={current!r}; migration_intentionally_absent=yes"
        )
    return value


def _source_map(
    document: GraphDocument,
    authored_graph: GraphSpec,
    graph: GraphSpec,
) -> GraphSourceMap:
    authored_wires = [
        wire.model_dump(mode="json", exclude_none=True) for wire in authored_graph.wires
    ]
    entries = [
        GraphSourceMapEntry(resolved_path="/graph", authored_path="/graph"),
        *(
            GraphSourceMapEntry(
                resolved_path=f"/graph/nodes/{node_id}",
                authored_path=(
                    f"/graph/nodes/{node_id}"
                    if node_id in authored_graph.nodes
                    else "/graph"
                ),
                origin=(
                    "authored" if node_id in authored_graph.nodes else "compiler-generated"
                ),
            )
            for node_id in sorted(graph.nodes)
        ),
        *(
            GraphSourceMapEntry(
                resolved_path=f"/graph/wires/{index}",
                authored_path=(
                    f"/graph/wires/{authored_wires.index(wire_payload)}"
                    if wire_payload in authored_wires
                    else "/graph"
                ),
                origin=("authored" if wire_payload in authored_wires else "compiler-generated"),
            )
            for index, wire_payload in enumerate(
                wire.model_dump(mode="json", exclude_none=True) for wire in graph.wires
            )
        ),
    ]
    if document.trial_root is not None:
        entries.append(GraphSourceMapEntry(resolved_path="/trial_root", authored_path="/trial_root"))
    if document.objective_root is not None:
        entries.append(
            GraphSourceMapEntry(resolved_path="/objective_root", authored_path="/objective_root")
        )
    return GraphSourceMap(entries=tuple(entries))


def _resolved_digest(
    graph: GraphSpec,
    source_map: GraphSourceMap,
    key_schedule: GraphKeySchedule,
) -> str:
    return canonical_sha256(
        {
            "graph": graph.model_dump(mode="json", exclude_none=True),
            "source_map": source_map.model_dump(mode="json"),
            "key_schedule": key_schedule.model_dump(mode="json"),
        }
    )


def compile_graph(
    document: GraphDocument,
    component_registry: Any,
    input_prototypes: Mapping[tuple[str, str], Any] | None = None,
) -> ExecutableGraph:
    """Compile one exact graph document through resolution and realization."""
    migration = migrate_graph_spec(document.graph)
    authored_graph = GraphSpec.model_validate(migration.payload)
    graph = authored_graph
    graph = materialize_additive_channel_adapters(graph)
    graph = normalize_derived_dimensions(
        graph,
        input_prototypes,
        component_registry=component_registry,
    )
    graph = normalize_stateful_prototypes(
        graph,
        input_prototypes,
        component_registry=component_registry,
    )
    source_map = _source_map(document, authored_graph, graph)
    executable = _instantiate_graph(graph, component_registry, input_prototypes)
    node_order = tuple(executable._execution_order)
    key_schedule = GraphKeySchedule(node_order=node_order)
    document_sha256 = canonical_sha256(document.model_dump(mode="json", exclude_none=True))
    resolved_sha256 = _resolved_digest(graph, source_map, key_schedule)
    resolved = ResolvedGraph(
        graph=graph,
        source_map=source_map,
        key_schedule=key_schedule,
        document_sha256=document_sha256,
        resolved_sha256=resolved_sha256,
    )
    record = CompilationRecord(
        document_sha256=document_sha256,
        resolved_sha256=resolved_sha256,
        source_map=source_map,
        key_schedule=key_schedule,
        graph_schema_source_version=migration.source_version,
        graph_schema_target_version=migration.target_version,
        migration_record_count=len(migration.migration_records),
    )
    return ExecutableGraph(graph=executable, resolved=resolved, record=record)
