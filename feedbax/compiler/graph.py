"""One staged entry point from authored graph semantics to JAX execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Literal, Mapping, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from feedbax.contracts.authored_canonical import canonical_sha256
from feedbax.contracts.graph import GraphSpec, SemanticAnchor
from feedbax.contracts.scientific_compiler_schema import (
    COMPILATION_FAILURE_SCHEMA_ID,
    COMPILATION_FAILURE_SCHEMA_VERSION,
    COMPILATION_RECORD_SCHEMA_ID,
    COMPILATION_RECORD_SCHEMA_VERSION,
    GRAPH_DOCUMENT_SCHEMA_ID,
    GRAPH_DOCUMENT_SCHEMA_VERSION,
    RESOLVED_GRAPH_SCHEMA_ID,
    RESOLVED_GRAPH_SCHEMA_VERSION,
)
from feedbax.contracts.graphs.prototypes import (
    normalize_derived_dimensions,
    normalize_stateful_prototypes,
)
from feedbax.contracts.graphs.serialization import _instantiate_graph
from feedbax.runtime.graph import Graph
from feedbax.runtime.graph_channel_adapters import materialize_additive_channel_adapters


GRAPH_COMPILER_ID = "feedbax.graph.compiler"
GRAPH_COMPILER_VERSION = "1"
KEY_SCHEDULE_ID = "feedbax.graph_key_schedule.execution_order_split.v1"


class CompilerPhase(StrEnum):
    """Ordered semantic compiler phases named by the public compiler contract."""

    STRUCTURAL_PARSING = "structural_parsing"
    TYPE_RESOLUTION = "type_resolution"
    COMPOSITE_AND_ACAUSAL_LOWERING = "composite_and_acausal_lowering"
    CONSTRAINT_SOLVING = "constraint_solving"
    SEMANTIC_VALIDATION = "semantic_validation"
    SCHEDULING = "scheduling"
    RESOLVED_IR_EMISSION = "resolved_ir_emission"


class DiagnosticSeverity(StrEnum):
    """Closed compiler diagnostic severity taxonomy."""

    ERROR = "error"
    WARNING = "warning"


class CompilerDiagnostic(BaseModel):
    """Stable source-mapped explanation of one compiler condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^compiler\.[a-z0-9_]+\.[a-z0-9_]+$")
    phase: CompilerPhase
    severity: DiagnosticSeverity
    source_anchor: SemanticAnchor
    expected_condition: str = Field(min_length=1)
    observed_condition: str = Field(min_length=1)
    actionable_context: dict[str, str]

    @field_validator("actionable_context")
    @classmethod
    def require_action(cls, value: dict[str, str]) -> dict[str, str]:
        action = value.get("action")
        if not action:
            raise ValueError("compiler diagnostic actionable_context requires 'action'")
        return value


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
    authored_anchor: SemanticAnchor


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
    diagnostics: tuple[CompilerDiagnostic, ...] = ()

    @field_validator("schema_version")
    @classmethod
    def reject_unsupported_version(cls, value: str) -> str:
        return _require_version(
            "CompilationRecord", value, COMPILATION_RECORD_SCHEMA_VERSION
        )


class CompilationFailureRecord(BaseModel):
    """Durable compiler evidence for a compilation that produced no resolved IR."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal[COMPILATION_FAILURE_SCHEMA_ID] = COMPILATION_FAILURE_SCHEMA_ID
    schema_version: str = COMPILATION_FAILURE_SCHEMA_VERSION
    compiler_id: Literal[GRAPH_COMPILER_ID] = GRAPH_COMPILER_ID
    compiler_version: Literal[GRAPH_COMPILER_VERSION] = GRAPH_COMPILER_VERSION
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostics: tuple[CompilerDiagnostic, ...] = Field(min_length=1)

    @field_validator("schema_version")
    @classmethod
    def reject_unsupported_version(cls, value: str) -> str:
        return _require_version(
            "CompilationFailureRecord", value, COMPILATION_FAILURE_SCHEMA_VERSION
        )


class GraphCompilationError(ValueError):
    """Compilation refusal carrying the complete structured failure record."""

    def __init__(self, record: CompilationFailureRecord):
        self.record = record
        diagnostic = record.diagnostics[0]
        super().__init__(
            f"{diagnostic.code} at {diagnostic.source_anchor.authored_path}: "
            f"{diagnostic.observed_condition}"
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


def migrate_graph_spec(payload: GraphSpec | Mapping[str, Any]) -> Any:
    """Call the public migration boundary without creating an import cycle."""
    from feedbax.contracts.migrations import migrate_graph_spec as migrate

    return migrate(payload)


T = TypeVar("T")


def _compile_phase(
    *,
    phase: CompilerPhase,
    code: str,
    anchor: SemanticAnchor,
    document_sha256: str,
    expected: str,
    action: str,
    operation: Callable[[], T],
) -> T:
    try:
        return operation()
    except GraphCompilationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        diagnostic = CompilerDiagnostic(
            code=code,
            phase=phase,
            severity=DiagnosticSeverity.ERROR,
            source_anchor=anchor,
            expected_condition=expected,
            observed_condition=str(exc) or type(exc).__name__,
            actionable_context={
                "action": action,
                "exception_type": type(exc).__name__,
            },
        )
        raise GraphCompilationError(
            CompilationFailureRecord(
                document_sha256=document_sha256,
                diagnostics=(diagnostic,),
            )
        ) from exc


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _source_map(
    document: GraphDocument,
    authored_graph: GraphSpec,
    graph: GraphSpec,
    document_sha256: str,
) -> GraphSourceMap:
    authored_wires = [
        wire.model_dump(mode="json", exclude_none=True) for wire in authored_graph.wires
    ]
    entries = [
        GraphSourceMapEntry(
            resolved_path="/graph",
            authored_path="/graph",
            authored_anchor=SemanticAnchor(
                semantic_document_sha256=document_sha256,
                authored_path="/graph",
            ),
        ),
        *(
            GraphSourceMapEntry(
                resolved_path=f"/graph/nodes/{_json_pointer_token(node_id)}",
                authored_path=(
                    f"/graph/nodes/{_json_pointer_token(node_id)}"
                    if node_id in authored_graph.nodes
                    else "/graph"
                ),
                origin=(
                    "authored" if node_id in authored_graph.nodes else "compiler-generated"
                ),
                authored_anchor=SemanticAnchor(
                    semantic_document_sha256=document_sha256,
                    authored_path=(
                        f"/graph/nodes/{_json_pointer_token(node_id)}"
                        if node_id in authored_graph.nodes
                        else "/graph"
                    ),
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
                authored_anchor=SemanticAnchor(
                    semantic_document_sha256=document_sha256,
                    authored_path=(
                        f"/graph/wires/{authored_wires.index(wire_payload)}"
                        if wire_payload in authored_wires
                        else "/graph"
                    ),
                ),
            )
            for index, wire_payload in enumerate(
                wire.model_dump(mode="json", exclude_none=True) for wire in graph.wires
            )
        ),
    ]
    if document.trial_root is not None:
        entries.append(GraphSourceMapEntry(
            resolved_path="/trial_root",
            authored_path="/trial_root",
            authored_anchor=SemanticAnchor(
                semantic_document_sha256=document_sha256,
                authored_path="/trial_root",
            ),
        ))
    if document.objective_root is not None:
        entries.append(
            GraphSourceMapEntry(
                resolved_path="/objective_root",
                authored_path="/objective_root",
                authored_anchor=SemanticAnchor(
                    semantic_document_sha256=document_sha256,
                    authored_path="/objective_root",
                ),
            )
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


def _require_component_types(graph: GraphSpec, component_registry: Any) -> None:
    resolver = getattr(component_registry, "resolve_component_spec", None)
    should_resolve = getattr(component_registry, "should_resolve_component_spec", None)
    getter = getattr(component_registry, "get", None)
    for node_id, node in graph.nodes.items():
        if callable(resolver) and callable(should_resolve) and should_resolve(
            node.type,
            param_schema_version=node.param_schema_version,
        ):
            resolver(
                node.type,
                node.params,
                param_schema_version=node.param_schema_version,
            )
        elif callable(getter) and getter(node.type) is None:
            names = getattr(component_registry, "names", None)
            known = names() if callable(names) else []
            raise ValueError(
                f"node {node_id!r} names unregistered component type {node.type!r}. "
                f"Known component types: {known!r}"
            )


def compile_graph(
    document: GraphDocument,
    component_registry: Any,
    input_prototypes: Mapping[tuple[str, str], Any] | None = None,
) -> ExecutableGraph:
    """Compile one exact graph document through resolution and realization."""
    document_sha256 = canonical_sha256(document.model_dump(mode="json", exclude_none=True))
    graph_anchor = SemanticAnchor(
        semantic_document_sha256=document_sha256,
        authored_path="/graph",
    )

    def parse_graph() -> tuple[Any, GraphSpec]:
        migration = migrate_graph_spec(document.graph)
        return migration, GraphSpec.model_validate(migration.payload)

    migration, authored_graph = _compile_phase(
        phase=CompilerPhase.STRUCTURAL_PARSING,
        code="compiler.structural_parsing.invalid_graph_document",
        anchor=graph_anchor,
        document_sha256=document_sha256,
        expected="graph payload satisfies a supported explicit GraphSpec schema",
        action="correct the authored graph payload or migrate it through a registered schema",
        operation=parse_graph,
    )
    _compile_phase(
        phase=CompilerPhase.TYPE_RESOLUTION,
        code="compiler.type_resolution.unresolved_component_type",
        anchor=graph_anchor,
        document_sha256=document_sha256,
        expected="every authored component type resolves exactly once in the supplied registry",
        action="register the missing component declaration or correct its authored type identity",
        operation=lambda: _require_component_types(authored_graph, component_registry),
    )
    graph = _compile_phase(
        phase=CompilerPhase.COMPOSITE_AND_ACAUSAL_LOWERING,
        code="compiler.composite_and_acausal_lowering.invalid_lowering",
        anchor=graph_anchor,
        document_sha256=document_sha256,
        expected="every authored lowering rule produces one valid literal semantic graph",
        action="correct the authored composite, acausal, or channel-adapter declaration",
        operation=lambda: materialize_additive_channel_adapters(authored_graph),
    )

    def solve_constraints() -> GraphSpec:
        normalized = normalize_derived_dimensions(
            graph,
            input_prototypes,
            component_registry=component_registry,
        )
        return normalize_stateful_prototypes(
            normalized,
            input_prototypes,
            component_registry=component_registry,
        )

    graph = _compile_phase(
        phase=CompilerPhase.CONSTRAINT_SOLVING,
        code="compiler.constraint_solving.inconsistent_graph_constraints",
        anchor=graph_anchor,
        document_sha256=document_sha256,
        expected="derived dimensions and state prototypes have one consistent solution",
        action="correct the conflicting port, dimension, parameter, or prototype declarations",
        operation=solve_constraints,
    )
    graph = _compile_phase(
        phase=CompilerPhase.SEMANTIC_VALIDATION,
        code="compiler.semantic_validation.invalid_resolved_semantics",
        anchor=graph_anchor,
        document_sha256=document_sha256,
        expected="the lowered graph satisfies every cross-reference and semantic invariant",
        action="correct the authored semantic reference named by the diagnostic",
        operation=lambda: GraphSpec.model_validate(
            graph.model_dump(mode="json", exclude_none=True)
        ),
    )

    def build_schedule() -> tuple[Graph, tuple[str, ...]]:
        executable = _instantiate_graph(graph, component_registry, input_prototypes)
        return executable, tuple(executable._execution_order)

    executable, node_order = _compile_phase(
        phase=CompilerPhase.SCHEDULING,
        code="compiler.scheduling.invalid_executable_schedule",
        anchor=graph_anchor,
        document_sha256=document_sha256,
        expected=(
            "every resolved component realizes and the graph has one deterministic "
            "executable node order"
        ),
        action=(
            "correct the component runtime facet, its resolved parameters, or the "
            "scheduling dependency"
        ),
        operation=build_schedule,
    )

    def emit_resolved_ir() -> tuple[ResolvedGraph, CompilationRecord]:
        source_map = _source_map(document, authored_graph, graph, document_sha256)
        key_schedule = GraphKeySchedule(node_order=node_order)
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
            diagnostics=(),
        )
        return resolved, record

    resolved, record = _compile_phase(
        phase=CompilerPhase.RESOLVED_IR_EMISSION,
        code="compiler.resolved_ir_emission.invalid_compilation_record",
        anchor=graph_anchor,
        document_sha256=document_sha256,
        expected="resolved IR, source map, key schedule, and record have canonical identities",
        action="report the compiler defect with the authored document revision",
        operation=emit_resolved_ir,
    )
    return ExecutableGraph(graph=executable, resolved=resolved, record=record)
