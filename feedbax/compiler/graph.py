"""One staged entry point from authored graph semantics to JAX execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Literal, Mapping, TypeVar

import jax.random as jr
from jaxtyping import PRNGKeyArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.artifact_custody import ArtifactBlobProvider
from feedbax.contracts._causal_inputs import InputDriverConflict
from feedbax.contracts.authored_canonical import canonical_sha256
from feedbax.contracts.graph import (
    ComponentSpec,
    GraphSpec,
    SemanticAnchor,
    graph_input_driver_conflicts,
)
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
from feedbax.compiler.feedback_migration import lower_legacy_feedback
from feedbax.compiler.resolution import (
    GraphParameterResolution,
    ParameterResolutionError,
    resolve_component_node,
    resolve_graph_parameters,
)
from feedbax.compiler.studio_taps import lower_studio_taps
from feedbax.component_registry.declarations import ComponentParamValidationError
from feedbax.compiler.serialization import (
    _instantiate_graph,
)
from feedbax.runtime._graph import instant_execution_order
from feedbax.runtime._graph_key_schedule import KEY_SCHEDULE_ID
from feedbax.runtime._signal_schedule import SignalSchedule, needs_signal_schedule, resolve_signal_schedule
from feedbax.runtime.graph import Graph
from feedbax.runtime.graph_channel_adapters import materialize_additive_channel_adapters
from feedbax.runtime.parameter_constraints import apply_parameter_constraints


GRAPH_COMPILER_ID = "feedbax.graph.compiler"
GRAPH_COMPILER_VERSION = "6"


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
    """Random schedule identity and root execution order as compilation evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schedule_id: Literal[KEY_SCHEDULE_ID]
    node_order: tuple[str, ...]
    scoped_node_order: tuple[tuple[str, ...], ...] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @field_validator("schedule_id", mode="before")
    @classmethod
    def reject_unsupported_schedule(cls, value: str) -> str:
        if value != KEY_SCHEDULE_ID:
            raise ValueError(
                f"Unsupported graph key schedule: source_schedule={value!r}; "
                f"current_schedule={KEY_SCHEDULE_ID!r}; "
                "recompile the authored GraphDocument to obtain current compiler evidence; "
                "exact historical replay requires the original producer revision and environment"
            )
        return value


class ResolvedGraph(BaseModel):
    """Normalized immutable semantic graph ready for process-local realization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal[RESOLVED_GRAPH_SCHEMA_ID]
    schema_version: Literal[RESOLVED_GRAPH_SCHEMA_VERSION]
    graph: GraphSpec
    source_map: GraphSourceMap
    key_schedule: GraphKeySchedule
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("schema_version", mode="before")
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

    schema_id: Literal[COMPILATION_RECORD_SCHEMA_ID]
    schema_version: Literal[COMPILATION_RECORD_SCHEMA_VERSION]
    compiler_id: Literal[GRAPH_COMPILER_ID]
    compiler_version: Literal[GRAPH_COMPILER_VERSION]
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_map: GraphSourceMap
    key_schedule: GraphKeySchedule
    graph_schema_source_version: str
    graph_schema_target_version: str
    migration_record_count: int = Field(ge=0)
    diagnostics: tuple[CompilerDiagnostic, ...] = ()

    @field_validator("schema_version", mode="before")
    @classmethod
    def reject_unsupported_version(cls, value: str) -> str:
        return _require_version("CompilationRecord", value, COMPILATION_RECORD_SCHEMA_VERSION)

    @field_validator("compiler_version", mode="before")
    @classmethod
    def reject_unsupported_compiler(cls, value: str) -> str:
        return _require_compiler_version(value)


class CompilationFailureRecord(BaseModel):
    """Durable compiler evidence for a compilation that produced no resolved IR."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal[COMPILATION_FAILURE_SCHEMA_ID]
    schema_version: Literal[COMPILATION_FAILURE_SCHEMA_VERSION]
    compiler_id: Literal[GRAPH_COMPILER_ID]
    compiler_version: Literal[GRAPH_COMPILER_VERSION]
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostics: tuple[CompilerDiagnostic, ...] = Field(min_length=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def reject_unsupported_version(cls, value: str) -> str:
        return _require_version(
            "CompilationFailureRecord", value, COMPILATION_FAILURE_SCHEMA_VERSION
        )

    @field_validator("compiler_version", mode="before")
    @classmethod
    def reject_unsupported_compiler(cls, value: str) -> str:
        return _require_compiler_version(value)


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


@dataclass(frozen=True)
class GraphResolution:
    """Process-local graph semantics and construction inputs before realization."""

    document: GraphDocument
    component_registry: Any
    authored_graph: GraphSpec
    graph: GraphSpec
    parameters: GraphParameterResolution
    tap_origins: dict[tuple[str, str], str]
    anchor: SemanticAnchor
    schema_source_version: str
    schema_target_version: str
    migration_record_count: int


def _require_compiler_version(value: str) -> str:
    if value != GRAPH_COMPILER_VERSION:
        raise ValueError(
            f"Unsupported graph compiler version: source_version={value!r}; "
            f"current_version={GRAPH_COMPILER_VERSION!r}; "
            "recompile the authored GraphDocument to obtain current compiler evidence; "
            "exact historical replay requires the original producer revision and environment"
        )
    return value


def _require_version(family: str, value: str, current: str) -> str:
    if value != current:
        raise ValueError(
            f"Unsupported {family} schema version: source_version={value!r}; "
            f"current_version={current!r}; migration_intentionally_absent=yes; "
            "recompile the authored GraphDocument for current evidence; "
            "exact historical replay requires the original producer revision and environment"
        )
    return value


def migrate_graph_spec(
    payload: GraphSpec | Mapping[str, Any], *, component_registry: Any | None = None
) -> Any:
    """Call the public migration boundary without creating an import cycle."""
    from feedbax.contracts.migrations import migrate_graph_spec as migrate

    return migrate(payload, component_registry=component_registry)


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
        cause = exc.cause if isinstance(exc, ParameterResolutionError) else exc
        context = {"action": action, "exception_type": type(exc).__name__}
        if isinstance(exc, InputDriverConflict):
            context["action"] = "remove a driver or route both through an explicit merge"
            context["occupied_by"] = exc.occupied_by
        diagnostic = CompilerDiagnostic(
            code=(
                "compiler.constraint_solving.invalid_component_parameters"
                if phase is CompilerPhase.CONSTRAINT_SOLVING
                and isinstance(cause, ComponentParamValidationError)
                else code
            ),
            phase=phase,
            severity=DiagnosticSeverity.ERROR,
            source_anchor=(
                anchor.model_copy(update={"authored_path": exc.authored_path})
                if getattr(exc, "authored_path", None)
                else anchor
            ),
            expected_condition=expected,
            observed_condition=str(exc) or type(exc).__name__,
            actionable_context=context,
        )
        raise GraphCompilationError(
            CompilationFailureRecord(
                schema_id=COMPILATION_FAILURE_SCHEMA_ID,
                schema_version=COMPILATION_FAILURE_SCHEMA_VERSION,
                compiler_id=GRAPH_COMPILER_ID,
                compiler_version=GRAPH_COMPILER_VERSION,
                document_sha256=document_sha256,
                diagnostics=(diagnostic,),
            )
        ) from cause


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _source_map(
    document: GraphDocument,
    authored_graph: GraphSpec,
    graph: GraphSpec,
    document_sha256: str,
    tap_origins: Mapping[tuple[str, str], str],
    graph_path: str = "/graph",
) -> GraphSourceMap:
    authored_wires = [
        wire.model_dump(mode="json", exclude_none=True) for wire in authored_graph.wires
    ]
    wire_indices = {wire.id: index for index, wire in enumerate(authored_graph.wires) if wire.id}

    def entry(resolved_path: str, authored_path: str, *, generated: bool) -> GraphSourceMapEntry:
        return GraphSourceMapEntry(
            resolved_path=resolved_path,
            authored_path=authored_path,
            origin="compiler-generated" if generated else "authored",
            authored_anchor=SemanticAnchor(
                semantic_document_sha256=document_sha256, authored_path=authored_path
            ),
        )

    def tap_path(node_id: str) -> str | None:
        origin = tap_origins.get((graph_path, node_id))
        return "/graph" + origin.removeprefix(graph_path) if origin else None

    entries = [entry("/graph", "/graph", generated=False)]
    for node_id in sorted(graph.nodes):
        path = f"/graph/nodes/{_json_pointer_token(node_id)}"
        authored = node_id in authored_graph.nodes
        entries.append(
            entry(path, path if authored else tap_path(node_id) or "/graph", generated=not authored)
        )
    for index, wire in enumerate(graph.wires):
        wire_payload = wire.model_dump(mode="json", exclude_none=True)
        authored_index = wire_indices.get(wire.id) if wire.id else None
        if authored_index is None and wire_payload in authored_wires:
            authored_index = authored_wires.index(wire_payload)
        authored_path = (
            f"/graph/wires/{authored_index}"
            if authored_index is not None
            else tap_path(wire.source_node) or tap_path(wire.target_node) or "/graph"
        )
        entries.append(
            entry(f"/graph/wires/{index}", authored_path, generated=authored_index is None)
        )
    for node_id, subgraph in sorted((graph.subgraphs or {}).items()):
        authored_subgraph = (authored_graph.subgraphs or {}).get(node_id)
        if not isinstance(subgraph, GraphSpec) or not isinstance(authored_subgraph, GraphSpec):
            continue
        prefix = f"/graph/subgraphs/{_json_pointer_token(node_id)}"
        nested_map = _source_map(
            GraphDocument(graph=authored_subgraph),
            authored_subgraph,
            subgraph,
            document_sha256,
            tap_origins,
            graph_path + prefix.removeprefix("/graph"),
        )
        for entry in nested_map.entries:
            authored_path = prefix + entry.authored_path.removeprefix("/graph")
            entries.append(
                entry.model_copy(
                    update={
                        "resolved_path": prefix + entry.resolved_path.removeprefix("/graph"),
                        "authored_path": authored_path,
                        "authored_anchor": entry.authored_anchor.model_copy(
                            update={
                                "authored_path": authored_path,
                            }
                        ),
                    }
                )
            )
    if document.trial_root is not None:
        entries.append(
            GraphSourceMapEntry(
                resolved_path="/trial_root",
                authored_path="/trial_root",
                authored_anchor=SemanticAnchor(
                    semantic_document_sha256=document_sha256,
                    authored_path="/trial_root",
                ),
            )
        )
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


_ComponentGraph = TypeVar("_ComponentGraph", GraphSpec, AcausalGraphSpec)


def _map_component_nodes(
    graph: _ComponentGraph,
    operation: Callable[[str, ComponentSpec], ComponentSpec],
    *,
    document_sha256: str,
    path: str = "/graph",
    migration_path: str = "graph",
    phase: CompilerPhase = CompilerPhase.TYPE_RESOLUTION,
) -> _ComponentGraph:
    nodes: dict[str, ComponentSpec] = {}
    for node_id, node in graph.nodes.items():
        anchor = SemanticAnchor(
            semantic_document_sha256=document_sha256,
            authored_path=f"{path}/nodes/{_json_pointer_token(node_id)}",
        )
        nodes[node_id] = _compile_phase(
            phase=phase,
            code=(
                "compiler.type_resolution.unresolved_component_type"
                if phase is CompilerPhase.TYPE_RESOLUTION
                else "compiler.constraint_solving.invalid_component_parameters"
            ),
            anchor=anchor,
            document_sha256=document_sha256,
            expected="the component has a registered canonical type and valid effective parameters",
            action="correct the component identity, version, or parameters at the authored node",
            operation=lambda node=node: operation(migration_path, node),
        )
    subgraphs = {
        node_id: (
            _map_component_nodes(
                subgraph,
                operation,
                document_sha256=document_sha256,
                path=f"{path}/subgraphs/{_json_pointer_token(node_id)}",
                migration_path=f"{migration_path}.subgraphs[{node_id!r}]",
                phase=phase,
            )
            if isinstance(subgraph, (GraphSpec, AcausalGraphSpec))
            else subgraph
        )
        for node_id, subgraph in (graph.subgraphs or {}).items()
    }
    return graph.model_copy(update={"nodes": nodes, "subgraphs": subgraphs or None})


def compile_graph(
    document: GraphDocument,
    component_registry: Any,
    input_prototypes: Mapping[tuple[str, str], Any] | None = None,
    *,
    initialization_key: PRNGKeyArray | None = None,
    initialization_keys: Mapping[str | tuple[str, ...], PRNGKeyArray] | None = None,
    artifact_provider: ArtifactBlobProvider | None = None,
) -> ExecutableGraph:
    """Compile one exact graph document through resolution and realization.

    Stochastic builders receive a key derived from the root initialization key
    (seed zero by default) and their scoped occurrence ID. An explicit
    ``initialization_keys`` entry overrides derivation for one leaf. Use a node
    ID for a top-level leaf or a tuple of IDs for a nested leaf. These keys
    do not change the resolved topology.

    Supply the artifact provider for content-addressed model assets. The
    provider is construction context and stays outside the saved document.
    """
    return realize_graph(
        resolve_graph(document, component_registry, input_prototypes),
        initialization_key=initialization_key,
        initialization_keys=initialization_keys,
        artifact_provider=artifact_provider,
    )


def resolve_graph(
    document: GraphDocument,
    component_registry: Any,
    input_prototypes: Mapping[tuple[str, str], Any] | None = None,
) -> GraphResolution:
    """Resolve the semantic graph without constructing a seeded model."""
    document_sha256 = canonical_sha256(document.model_dump(mode="json", exclude_none=True))
    graph_anchor = SemanticAnchor(
        semantic_document_sha256=document_sha256,
        authored_path="/graph",
    )

    def parse_graph() -> tuple[Any, GraphSpec]:
        migration = migrate_graph_spec(document.graph, component_registry=component_registry)
        graph = GraphSpec.model_validate(migration.payload)
        if conflict := next(graph_input_driver_conflicts(graph, "/graph"), None):
            raise conflict
        return migration, graph

    migration, authored_graph = _compile_phase(
        phase=CompilerPhase.STRUCTURAL_PARSING,
        code="compiler.structural_parsing.invalid_graph_document",
        anchor=graph_anchor,
        document_sha256=document_sha256,
        expected="graph payload satisfies a supported explicit GraphSpec schema",
        action="correct the authored graph payload or migrate it through a registered schema",
        operation=parse_graph,
    )
    resolved_inputs = dict(input_prototypes or {})
    feedback_graph = _compile_phase(
        phase=CompilerPhase.STRUCTURAL_PARSING,
        code="compiler.structural_parsing.invalid_legacy_feedback",
        anchor=graph_anchor,
        document_sha256=document_sha256,
        expected="legacy feedback nodes declare recoverable mechanics selector paths",
        action="replace the legacy wrapper with SelectState and Channel nodes",
        operation=lambda: lower_legacy_feedback(authored_graph),
    )
    historical_paths = {
        record.metadata.get("graph_path") for record in migration.migration_records
    }
    typed_graph = _map_component_nodes(
        feedback_graph,
        lambda graph_path, node: resolve_component_node(
            node,
            component_registry,
            historical_unversioned=graph_path in historical_paths,
        ),
        document_sha256=document_sha256,
    )

    def lower_graph() -> tuple[GraphSpec, dict[tuple[str, str], str]]:
        lowered, tap_origins = lower_studio_taps(typed_graph)
        lowered = materialize_additive_channel_adapters(lowered)
        if conflict := next(graph_input_driver_conflicts(lowered, "/graph"), None):
            raise conflict
        graph = _map_component_nodes(
            lowered,
            lambda _graph_path, node: (
                resolve_component_node(node, component_registry)
                if node.revision is None
                else node
            ),
            document_sha256=document_sha256,
        )

        return graph, tap_origins

    graph, tap_origins = _compile_phase(
        phase=CompilerPhase.COMPOSITE_AND_ACAUSAL_LOWERING,
        code="compiler.composite_and_acausal_lowering.invalid_lowering",
        anchor=graph_anchor,
        document_sha256=document_sha256,
        expected="every authored lowering rule produces one valid literal semantic graph",
        action="correct the authored composite, acausal, channel-adapter, or intervention tap declaration",
        operation=lower_graph,
    )

    def solve_constraints() -> GraphParameterResolution:
        return resolve_graph_parameters(
            graph,
            resolved_inputs,
            component_registry,
            identities_resolved=True,
        )

    resolution = _compile_phase(
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
            resolution.graph.model_dump(mode="json", exclude_none=True)
        ),
    )

    return GraphResolution(
        document, component_registry, authored_graph, graph, resolution, tap_origins, graph_anchor,
        migration.source_version, migration.target_version, len(migration.migration_records),
    )


def resolve_graph_schedule(graph: GraphSpec) -> tuple[SignalSchedule | None, GraphKeySchedule]:
    """Compute runtime and PRNG order from the resolved topology alone."""
    scoped = resolve_signal_schedule(graph) if needs_signal_schedule(graph) else None
    return scoped, GraphKeySchedule(
        schedule_id=KEY_SCHEDULE_ID,
        node_order=() if scoped is not None else instant_execution_order(graph.nodes, (
            (wire.source_node, wire.target_node) for wire in graph.wires if wire.temporality != "recurrent"
        )),
        scoped_node_order=scoped.order if scoped is not None else None,
    )


def realize_graph(
    resolved: GraphResolution,
    *,
    initialization_key: PRNGKeyArray | None = None,
    initialization_keys: Mapping[str | tuple[str, ...], PRNGKeyArray] | None = None,
    artifact_provider: ArtifactBlobProvider | None = None,
) -> ExecutableGraph:
    """Construct and schedule a model from the exact resolved graph."""
    graph = resolved.graph
    document_sha256 = resolved.anchor.semantic_document_sha256

    def build_schedule() -> tuple[Graph, GraphKeySchedule]:
        leaf_keys = {}
        for name, key in (initialization_keys or {}).items():
            path = (name,) if isinstance(name, str) else name
            if not path or path in leaf_keys:
                raise ValueError(f"Invalid or duplicate initialization key path {path!r}")
            leaf_keys[path] = key
        scoped_schedule, key_schedule = resolve_graph_schedule(graph)
        instantiated = _instantiate_graph(
            graph,
            resolved.component_registry,
            resolved.parameters,
            construction_key=jr.PRNGKey(0) if initialization_key is None else initialization_key,
            initialization_keys=leaf_keys,
            artifact_provider=artifact_provider,
            scoped_schedule=scoped_schedule,
        )
        executable = apply_parameter_constraints(instantiated).prepare()
        return executable, key_schedule

    executable, key_schedule = _compile_phase(
        phase=CompilerPhase.SCHEDULING,
        code="compiler.scheduling.invalid_executable_schedule",
        anchor=resolved.anchor,
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
        source_map = _source_map(
            resolved.document, resolved.authored_graph, graph, document_sha256, resolved.tap_origins,
        )
        resolved_sha256 = _resolved_digest(graph, source_map, key_schedule)
        evidence = ResolvedGraph(
            schema_id=RESOLVED_GRAPH_SCHEMA_ID,
            schema_version=RESOLVED_GRAPH_SCHEMA_VERSION,
            graph=graph,
            source_map=source_map,
            key_schedule=key_schedule,
            document_sha256=document_sha256,
            resolved_sha256=resolved_sha256,
        )
        record = CompilationRecord(
            schema_id=COMPILATION_RECORD_SCHEMA_ID,
            schema_version=COMPILATION_RECORD_SCHEMA_VERSION,
            compiler_id=GRAPH_COMPILER_ID,
            compiler_version=GRAPH_COMPILER_VERSION,
            document_sha256=document_sha256,
            resolved_sha256=resolved_sha256,
            source_map=source_map,
            key_schedule=key_schedule,
            graph_schema_source_version=resolved.schema_source_version,
            graph_schema_target_version=resolved.schema_target_version,
            migration_record_count=resolved.migration_record_count,
            diagnostics=(),
        )
        return evidence, record

    evidence, record = _compile_phase(
        phase=CompilerPhase.RESOLVED_IR_EMISSION,
        code="compiler.resolved_ir_emission.invalid_compilation_record",
        anchor=resolved.anchor,
        document_sha256=document_sha256,
        expected="resolved IR, source map, key schedule, and record have canonical identities",
        action="report the compiler defect with the authored document revision",
        operation=emit_resolved_ir,
    )
    return ExecutableGraph(graph=executable, resolved=evidence, record=record)
