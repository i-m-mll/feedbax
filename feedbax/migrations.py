"""Versioned schema migration registries for durable Feedbax artifacts and specs."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from feedbax.contracts.graph import (
    GRAPH_SPEC_SCHEMA_ID,
    GRAPH_SPEC_SCHEMA_VERSION,
    LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
    GraphSpec,
)
from feedbax.manifest import ArtifactMigrationRecord, SCHEMA_VERSION as MANIFEST_SCHEMA_VERSION
from feedbax.retention_artifact_schema import (
    LOSS_TERM_PLAN_SCHEMA_ID,
    LOSS_TERM_PLAN_SCHEMA_VERSION,
    RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_ID,
    RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_VERSION,
    RETAINED_OBSERVABLE_PLAN_SCHEMA_ID,
    RETAINED_OBSERVABLE_PLAN_SCHEMA_VERSION,
    RETENTION_PLAN_SCHEMA_ID,
    RETENTION_PLAN_SCHEMA_VERSION,
    RETENTION_POLICY_PLAN_SCHEMA_ID,
    RETENTION_POLICY_PLAN_SCHEMA_VERSION,
)
from feedbax.schema_namespace import (
    SchemaNamespaceKind,
    validate_schema_identity,
    validate_schema_version,
)


MigrationPayload = Mapping[str, Any]
MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]
ComponentParamMigrationFn = Callable[[dict[str, Any]], dict[str, Any]]


class MigrationError(ValueError):
    """Base class for Feedbax artifact migration failures."""


class UnsupportedMigrationPath(MigrationError):
    """Raised when no deterministic schema migration path is registered."""


class UnknownSpecFamily(MigrationError):
    """Raised when a structured spec kind is not registered."""


class UnsupportedSpecVersion(MigrationError):
    """Raised when a structured spec version cannot be accepted or migrated."""


class UnsupportedComponentMigration(MigrationError):
    """Raised when a component ID or parameter schema cannot be migrated."""


class MissingComponentOwner(UnsupportedComponentMigration):
    """Raised when a durable component ID does not name a loadable owner."""


STUDIO_TASK_BINDING_LEGACY_V1 = "feedbax.studio.task_bindings.v1"


@dataclass(frozen=True)
class SchemaMigration:
    """One deterministic migration edge between schema versions."""

    source_version: str
    target_version: str
    migration_id: str
    migrate: MigrationFn
    description: str = ""

    def apply(self, payload: MigrationPayload) -> tuple[dict[str, Any], ArtifactMigrationRecord]:
        """Apply this migration edge and return the migrated payload plus record."""
        migrated = self.migrate(dict(payload))
        migrated["schema_version"] = self.target_version
        record = ArtifactMigrationRecord(
            migration_id=self.migration_id,
            source_schema_version=self.source_version,
            target_schema_version=self.target_version,
            metadata={"description": self.description} if self.description else {},
        )
        return migrated, record


@dataclass(frozen=True)
class ComponentMigration:
    """One deterministic component type or parameter-schema migration edge."""

    source_type: str
    target_type: str
    migration_id: str
    owner: str
    source_param_schema_version: str | None = None
    target_param_schema_version: str | None = None
    migrate_params: ComponentParamMigrationFn | None = None
    description: str = ""

    def apply(
        self,
        params: Mapping[str, Any],
        *,
        source_param_schema_version: str | None = None,
    ) -> tuple[str, dict[str, Any], str | None]:
        """Apply this component edge to a parameter payload."""
        if source_param_schema_version != self.source_param_schema_version:
            raise UnsupportedComponentMigration(
                "Unsupported component parameter schema version: "
                f"type={self.source_type!r}, owner={self.owner!r}, "
                f"source_version={source_param_schema_version!r}, "
                f"expected_source_version={self.source_param_schema_version!r}, "
                f"target_type={self.target_type!r}, "
                f"target_version={self.target_param_schema_version!r}"
            )
        migrated = dict(params)
        if self.migrate_params is not None:
            migrated = self.migrate_params(migrated)
        return self.target_type, migrated, self.target_param_schema_version


@dataclass(frozen=True)
class ComponentMigrationPack:
    """Migration pack contributed by a component owner package."""

    owner: str
    migrations: tuple[ComponentMigration, ...]
    package: str | None = None
    version: str | None = None
    description: str = ""


class MigrationRegistry:
    """Small deterministic registry for schema-to-schema migration paths."""

    def __init__(self) -> None:
        self._edges: dict[tuple[str, str], SchemaMigration] = {}

    def register(self, migration: SchemaMigration) -> None:
        """Register a migration edge.

        Raises:
            ValueError: If an edge for the same source/target pair already exists.
        """
        key = (migration.source_version, migration.target_version)
        if key in self._edges:
            raise ValueError(
                "Migration edge already registered: "
                f"{migration.source_version!r} -> {migration.target_version!r}"
            )
        self._edges[key] = migration

    def edges(self) -> tuple[SchemaMigration, ...]:
        """Return registered migration edges sorted by source/target."""
        return tuple(self._edges[key] for key in sorted(self._edges))

    def path(self, source_version: str, target_version: str) -> list[SchemaMigration]:
        """Return the shortest deterministic migration path."""
        if source_version == target_version:
            return []

        adjacency: dict[str, list[SchemaMigration]] = {}
        for (source, _target), migration in sorted(self._edges.items()):
            adjacency.setdefault(source, []).append(migration)

        queue: deque[tuple[str, list[SchemaMigration]]] = deque([(source_version, [])])
        visited = {source_version}
        while queue:
            version, path = queue.popleft()
            for migration in adjacency.get(version, []):
                if migration.target_version in visited:
                    continue
                next_path = [*path, migration]
                if migration.target_version == target_version:
                    return next_path
                visited.add(migration.target_version)
                queue.append((migration.target_version, next_path))

        known_sources = sorted({source for source, _target in self._edges})
        known_targets = sorted({target for _source, target in self._edges})
        raise UnsupportedMigrationPath(
            "No Feedbax schema migration path registered: "
            f"{source_version!r} -> {target_version!r}; "
            f"known_sources={known_sources}, known_targets={known_targets}"
        )

    def migrate(
        self,
        payload: MigrationPayload,
        *,
        source_version: str,
        target_version: str,
    ) -> tuple[dict[str, Any], list[ArtifactMigrationRecord]]:
        """Migrate a payload through a registered deterministic schema path."""
        migrated = dict(payload)
        records: list[ArtifactMigrationRecord] = []
        for migration in self.path(source_version, target_version):
            migrated, record = migration.apply(migrated)
            records.append(record)
        return migrated, records


@dataclass(frozen=True)
class SpecSchemaFamily:
    """Identity and current schema version for one structured spec family.

    Attributes:
        kind: Stable public kind, usually the provider schema/model name.
        current_version: Current accepted version for this family.
        schema_id: Stable namespaced schema identity. Defaults to ``kind``.
        durable: Whether payloads in this family are intended as saved artifacts.
        emitted: Whether Feedbax emits this family through provider/manifest surfaces.
        description: Short human-facing context for registry consumers.
        policy: Explicit migration/rejection policy for this family.
        namespace: Resolved governed namespace for this family.
    """

    kind: str
    current_version: str
    schema_id: str | None = None
    durable: bool = True
    emitted: bool = True
    description: str = ""
    policy: "SpecFamilyMigrationPolicy | None" = None
    namespace: SchemaNamespaceKind | None = None

    @property
    def identity(self) -> str:
        """Return the stable schema identity for this family."""
        return self.schema_id or self.kind


@dataclass(frozen=True)
class SpecFamilyMigrationPolicy:
    """Explicit old-version behavior for one emitted structured spec family."""

    owner_module: str
    emitted_by: tuple[str, ...]
    consumed_by: tuple[str, ...]
    stance: Literal["migrate", "reject"]
    supported_old_versions: tuple[str, ...] = ()
    rejected_old_versions: tuple[str, ...] = ()
    rejection_message: str = (
        "Unsupported-version errors must include family, schema_id, source_version, "
        "current_version, migration_intentionally_absent, and the policy reason."
    )
    required_tests: tuple[str, ...] = ()
    notes: str = ""
    covers: str | None = None


@dataclass(frozen=True)
class UnsupportedSpecVersionPolicy:
    """Explicit rejection policy for an unsupported old structured spec version."""

    kind: str
    version: str
    reason: str
    migration_intentionally_absent: bool = True


@dataclass(frozen=True)
class SpecMigrationResult:
    """Result from accepting or migrating one structured spec payload."""

    kind: str
    schema_id: str
    source_version: str
    target_version: str
    payload: dict[str, Any]
    migration_records: list[ArtifactMigrationRecord]

    @property
    def migrated(self) -> bool:
        """Whether any migration edge was applied."""
        return bool(self.migration_records)


class SpecSchemaRegistry:
    """Registry for structured spec schema identity and migration policy.

    This registry is family-scoped: migration edges for one spec kind are not
    considered for another kind even if their version strings happen to match.
    Current payloads are accepted as no-ops. Old payloads either follow a
    registered migration path or fail through an explicit unsupported-version
    policy or a clear missing-path diagnostic.
    """

    def __init__(self) -> None:
        self._families: dict[str, SpecSchemaFamily] = {}
        self._migrations: dict[str, MigrationRegistry] = {}
        self._unsupported_versions: dict[tuple[str, str], UnsupportedSpecVersionPolicy] = {}

    def register_family(self, family: SpecSchemaFamily) -> None:
        """Register one structured spec family.

        Raises:
            ValueError: If the kind or current version is empty, or already registered.
        """
        if not family.kind:
            raise ValueError("Structured spec family kind must be non-empty")
        if not family.current_version:
            raise ValueError("Structured spec family current_version must be non-empty")
        if family.kind in self._families:
            raise ValueError(f"Structured spec family already registered: {family.kind!r}")
        namespace = validate_schema_identity(family.identity, family=family.kind)
        validate_schema_version(family.current_version, family=family.kind)
        if family.namespace is not None and namespace != family.namespace:
            raise ValueError(
                "Structured spec family namespace mismatch: "
                f"kind={family.kind!r}, schema_id={family.identity!r}, "
                f"declared={family.namespace.value!r}, resolved={namespace.value!r}"
            )
        self._families[family.kind] = (
            family if family.namespace is not None else replace(family, namespace=namespace)
        )

    def families(self) -> tuple[SpecSchemaFamily, ...]:
        """Return registered families sorted by kind."""
        return tuple(self._families[kind] for kind in sorted(self._families))

    def policy_matrix(self) -> dict[str, SpecFamilyMigrationPolicy]:
        """Return explicit migration/rejection policies keyed by family kind."""
        return {
            kind: family.policy
            for kind, family in sorted(self._families.items())
            if family.policy is not None
        }

    def families_missing_policy(self) -> tuple[str, ...]:
        """Return registered emitted family kinds that do not have policy rows."""
        return tuple(
            kind
            for kind, family in sorted(self._families.items())
            if family.emitted and family.policy is None
        )

    def resolve(self, kind: str) -> SpecSchemaFamily:
        """Return the schema family for ``kind`` or raise a clear diagnostic."""
        try:
            return self._families[kind]
        except KeyError as exc:
            known = ", ".join(sorted(self._families)) or "<none>"
            raise UnknownSpecFamily(
                f"Unknown Feedbax structured spec family {kind!r}; known families: {known}"
            ) from exc

    def current_version(self, kind: str) -> str:
        """Return the current version registered for a structured spec kind."""
        return self.resolve(kind).current_version

    def register_migration(self, kind: str, migration: SchemaMigration) -> None:
        """Register one migration edge for a structured spec family."""
        self.resolve(kind)
        self._migrations.setdefault(kind, MigrationRegistry()).register(migration)

    def available_migrations(self, kind: str) -> tuple[SchemaMigration, ...]:
        """Return registered migrations for ``kind`` sorted by source/target."""
        self.resolve(kind)
        registry = self._migrations.get(kind)
        return () if registry is None else registry.edges()

    def reject_version(
        self,
        kind: str,
        version: str,
        *,
        reason: str,
        migration_intentionally_absent: bool = True,
    ) -> None:
        """Register an explicit unsupported-version rejection policy."""
        self.resolve(kind)
        if not version:
            raise ValueError("Unsupported structured spec version must be non-empty")
        if not reason:
            raise ValueError("Unsupported structured spec version reason must be non-empty")
        key = (kind, version)
        if key in self._unsupported_versions:
            raise ValueError(
                f"Unsupported structured spec version already registered: {kind!r} {version!r}"
            )
        self._unsupported_versions[key] = UnsupportedSpecVersionPolicy(
            kind=kind,
            version=version,
            reason=reason,
            migration_intentionally_absent=migration_intentionally_absent,
        )

    def migrate(
        self,
        kind: str,
        payload: MigrationPayload,
        *,
        source_version: str | None = None,
        target_version: str | None = None,
    ) -> SpecMigrationResult:
        """Accept or migrate a structured spec payload for ``kind``.

        If neither ``source_version`` nor ``payload["schema_version"]`` is
        present, the payload is treated as current and returned unchanged. This
        preserves existing versionless in-memory specs while still allowing
        durable callers to opt into explicit version checks.
        """
        family = self.resolve(kind)
        payload_dict = dict(payload)
        resolved_source = (
            source_version
            or _payload_schema_version(payload_dict)
            or family.current_version
        )
        resolved_target = target_version or family.current_version

        if resolved_source == resolved_target:
            return SpecMigrationResult(
                kind=family.kind,
                schema_id=family.identity,
                source_version=resolved_source,
                target_version=resolved_target,
                payload=payload_dict,
                migration_records=[],
            )

        policy = self._unsupported_versions.get((kind, resolved_source))
        if policy is not None:
            raise UnsupportedSpecVersion(
                _unsupported_version_message(family, policy, resolved_target)
            )

        registry = self._migrations.get(kind)
        if registry is None:
            raise UnsupportedSpecVersion(
                _missing_path_message(family, resolved_source, resolved_target, ())
            )

        try:
            migrated, records = registry.migrate(
                payload_dict,
                source_version=resolved_source,
                target_version=resolved_target,
            )
        except UnsupportedMigrationPath as exc:
            raise UnsupportedSpecVersion(
                _missing_path_message(
                    family,
                    resolved_source,
                    resolved_target,
                    registry.edges(),
                )
            ) from exc

        return SpecMigrationResult(
            kind=family.kind,
            schema_id=family.identity,
            source_version=resolved_source,
            target_version=resolved_target,
            payload=migrated,
            migration_records=records,
        )


def _payload_schema_version(payload: Mapping[str, Any]) -> str | None:
    schema_version = payload.get("schema_version")
    return schema_version if isinstance(schema_version, str) and schema_version else None


def _unsupported_version_message(
    family: SpecSchemaFamily,
    policy: UnsupportedSpecVersionPolicy,
    target_version: str,
) -> str:
    absent = "yes" if policy.migration_intentionally_absent else "no"
    return (
        "Unsupported Feedbax structured spec version: "
        f"family={family.kind!r}, schema_id={family.identity!r}, "
        f"source_version={policy.version!r}, current_version={target_version!r}, "
        f"migration_intentionally_absent={absent}; reason: {policy.reason}"
    )


def _missing_path_message(
    family: SpecSchemaFamily,
    source_version: str,
    target_version: str,
    available_migrations: tuple[SchemaMigration, ...],
) -> str:
    available = [
        f"{migration.source_version}->{migration.target_version} ({migration.migration_id})"
        for migration in available_migrations
    ]
    available_text = ", ".join(available) if available else "<none>"
    return (
        "No Feedbax structured spec migration path registered: "
        f"family={family.kind!r}, schema_id={family.identity!r}, "
        f"source_version={source_version!r}, current_version={target_version!r}; "
        "no explicit unsupported-version policy is registered for this source version; "
        f"available_migrations=[{available_text}]"
    )


def _mapping_payload(payload: Mapping[str, Any] | GraphSpec) -> dict[str, Any]:
    if isinstance(payload, GraphSpec):
        return payload.model_dump(mode="json")
    return dict(payload)


def _payload_metadata_version(payload: Mapping[str, Any]) -> str | None:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        version = metadata.get("version")
        if isinstance(version, str) and version:
            return version
    return None


def _graph_spec_source_version(
    payload: Mapping[str, Any],
    explicit_source_version: str | None,
) -> str:
    if explicit_source_version:
        return explicit_source_version
    schema_version = _payload_schema_version(payload)
    if schema_version:
        return schema_version
    metadata_version = _payload_metadata_version(payload)
    if metadata_version:
        return metadata_version
    return GRAPH_SPEC_SCHEMA_VERSION


def _validate_graph_spec_schema_id(payload: Mapping[str, Any], *, path: str) -> None:
    schema_id = payload.get("schema_id")
    if schema_id is not None and schema_id != GRAPH_SPEC_SCHEMA_ID:
        raise UnsupportedSpecVersion(
            "Unsupported Feedbax GraphSpec schema identity: "
            f"path={path!r}, schema_id={schema_id!r}, expected={GRAPH_SPEC_SCHEMA_ID!r}"
        )


def _record_with_graph_path(record: ArtifactMigrationRecord, path: str) -> ArtifactMigrationRecord:
    return record.model_copy(
        update={"metadata": {**record.metadata, "graph_path": path}},
    )


def _record_with_spec_path(record: ArtifactMigrationRecord, path: str) -> ArtifactMigrationRecord:
    return record.model_copy(
        update={"metadata": {**record.metadata, "spec_path": path}},
    )


def _migrate_legacy_graph_spec_payload(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    migrated["schema_id"] = GRAPH_SPEC_SCHEMA_ID

    nodes: dict[str, Any] = {}
    for node_id, raw_node in dict(payload.get("nodes") or {}).items():
        node = dict(raw_node)
        next_type = node.get("type")
        if next_type == "SimpleStagedNetwork":
            next_type = "Network"
        if next_type == "FeedbackChannel":
            next_type = "Channel"
        if next_type == "PenzaiSubgraph":
            next_type = "PenzaiAdapter"

        params = dict(node.get("params") or {})
        if next_type == "Network" and "output_size" in params and "out_size" not in params:
            params["out_size"] = params.get("output_size")

        input_ports = list(node.get("input_ports") or [])
        if next_type == "Network":
            input_ports = ["input" if port == "target" else port for port in input_ports]

        node["type"] = next_type
        node["params"] = params
        node["input_ports"] = input_ports
        node["output_ports"] = list(node.get("output_ports") or [])
        nodes[str(node_id)] = node

    def _rename_port(node_name: str, port: str) -> str:
        node = nodes.get(node_name)
        if isinstance(node, Mapping) and node.get("type") == "Network" and port == "target":
            return "input"
        return port

    wires: list[dict[str, Any]] = []
    for raw_wire in list(payload.get("wires") or []):
        wire = dict(raw_wire)
        source_node = str(wire.get("source_node"))
        target_node = str(wire.get("target_node"))
        wire["source_port"] = _rename_port(source_node, str(wire.get("source_port")))
        wire["target_port"] = _rename_port(target_node, str(wire.get("target_port")))
        wires.append(wire)

    input_bindings: dict[str, tuple[str, str]] = {}
    for name, raw_binding in dict(payload.get("input_bindings") or {}).items():
        try:
            node, port = raw_binding
        except (TypeError, ValueError):
            input_bindings[str(name)] = raw_binding
            continue
        input_bindings[str(name)] = (str(node), _rename_port(str(node), str(port)))

    migrated["nodes"] = nodes
    migrated["wires"] = wires
    migrated["input_bindings"] = input_bindings
    migrated.setdefault("output_bindings", dict(payload.get("output_bindings") or {}))
    return migrated


def _migrate_studio_task_binding_v1_payload(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    if "exposed_outputs" in migrated and "exposed_data" not in migrated:
        migrated["exposed_data"] = migrated.pop("exposed_outputs")
    else:
        migrated.pop("exposed_outputs", None)

    bindings: list[Any] = []
    for raw_binding in list(migrated.get("bindings") or []):
        if not isinstance(raw_binding, Mapping):
            bindings.append(raw_binding)
            continue
        binding = dict(raw_binding)
        if "source_output_id" in binding and "source_data_id" not in binding:
            binding["source_data_id"] = binding.pop("source_output_id")
        else:
            binding.pop("source_output_id", None)
        bindings.append(binding)
    migrated["bindings"] = bindings
    return migrated


def migrate_graph_spec(
    payload: Mapping[str, Any] | GraphSpec,
    *,
    source_version: str | None = None,
    target_version: str | None = None,
    path: str = "graph",
    registry: SpecSchemaRegistry | None = None,
) -> SpecMigrationResult:
    """Migrate a GraphSpec payload through the public registered schema path.

    Nested subgraphs are migrated recursively through the same registry policy.
    Migration records are returned in deterministic parent-before-child order
    and include ``metadata["graph_path"]`` for provenance.
    """
    registry = registry or default_spec_registry
    payload_dict = _mapping_payload(payload)
    _validate_graph_spec_schema_id(payload_dict, path=path)
    resolved_source = _graph_spec_source_version(payload_dict, source_version)
    resolved_target = target_version or registry.current_version("GraphSpec")

    result = registry.migrate(
        "GraphSpec",
        payload_dict,
        source_version=resolved_source,
        target_version=resolved_target,
    )
    migrated_payload = dict(result.payload)
    migrated_payload.setdefault("schema_id", GRAPH_SPEC_SCHEMA_ID)
    migrated_payload.setdefault("schema_version", resolved_target)
    records = [_record_with_graph_path(record, path) for record in result.migration_records]

    subgraphs = migrated_payload.get("subgraphs")
    if isinstance(subgraphs, Mapping):
        migrated_subgraphs: dict[str, Any] = {}
        for node_id in sorted(subgraphs):
            raw_subgraph = subgraphs[node_id]
            if not isinstance(raw_subgraph, Mapping) and not isinstance(raw_subgraph, GraphSpec):
                migrated_subgraphs[str(node_id)] = raw_subgraph
                continue
            nested_source = None
            if isinstance(raw_subgraph, Mapping):
                nested_source = (
                    _payload_schema_version(raw_subgraph)
                    or _payload_metadata_version(raw_subgraph)
                    or resolved_source
                )
            nested = migrate_graph_spec(
                raw_subgraph,
                source_version=nested_source,
                target_version=resolved_target,
                path=f"{path}.subgraphs[{node_id!r}]",
                registry=registry,
            )
            migrated_subgraphs[str(node_id)] = nested.payload
            records.extend(nested.migration_records)
        migrated_payload["subgraphs"] = migrated_subgraphs

    return SpecMigrationResult(
        kind=result.kind,
        schema_id=result.schema_id,
        source_version=result.source_version,
        target_version=result.target_version,
        payload=migrated_payload,
        migration_records=records,
    )


def migrate_studio_task_binding_spec(
    payload: Mapping[str, Any],
    *,
    source_version: str | None = None,
    target_version: str | None = None,
    path: str = "task_binding_spec",
    registry: SpecSchemaRegistry | None = None,
) -> SpecMigrationResult:
    """Migrate a Studio task-binding payload through the public schema path."""
    registry = registry or default_spec_registry
    result = registry.migrate(
        "StudioTaskBindingSpec",
        payload,
        source_version=source_version,
        target_version=target_version,
    )
    return SpecMigrationResult(
        kind=result.kind,
        schema_id=result.schema_id,
        source_version=result.source_version,
        target_version=result.target_version,
        payload=result.payload,
        migration_records=[
            _record_with_spec_path(record, path) for record in result.migration_records
        ],
    )


def migrate_structured_spec_payload(
    kind: str,
    payload: Mapping[str, Any],
    *,
    source_version: str | None = None,
    target_version: str | None = None,
    path: str = "spec",
    registry: SpecSchemaRegistry | None = None,
) -> SpecMigrationResult:
    """Migrate or explicitly reject one registered structured spec payload."""
    registry = registry or default_spec_registry
    if kind == "GraphSpec":
        return migrate_graph_spec(
            payload,
            source_version=source_version,
            target_version=target_version,
            path=path,
            registry=registry,
        )
    if kind == "StudioTaskBindingSpec":
        return migrate_studio_task_binding_spec(
            payload,
            source_version=source_version,
            target_version=target_version,
            path=path,
            registry=registry,
        )
    result = registry.migrate(
        kind,
        payload,
        source_version=source_version,
        target_version=target_version,
    )
    return SpecMigrationResult(
        kind=result.kind,
        schema_id=result.schema_id,
        source_version=result.source_version,
        target_version=result.target_version,
        payload=result.payload,
        migration_records=[
            _record_with_spec_path(record, path) for record in result.migration_records
        ],
    )


_SCENARIO_STRUCTURED_FIELDS = {
    "training_spec": "TrainingSpec",
    "task_spec": "TaskSpec",
    "objective_spec": "ObjectiveSpec",
    "temporal_spec": "StudioTaskTimelineSpec",
    "analysis_spec": "AnalysisRunSpec",
    "report_spec": "ReportSpec",
}


def migrate_studio_scenario_spec(
    payload: Mapping[str, Any],
    *,
    source_version: str | None = None,
    target_version: str | None = None,
    path: str = "scenario",
    registry: SpecSchemaRegistry | None = None,
) -> SpecMigrationResult:
    """Migrate a Studio scenario and nested durable spec payloads."""
    registry = registry or default_spec_registry
    result = registry.migrate(
        "StudioScenarioSpec",
        payload,
        source_version=source_version,
        target_version=target_version,
    )
    migrated_payload = dict(result.payload)
    records = [_record_with_spec_path(record, path) for record in result.migration_records]

    graph_payload = migrated_payload.get("graph")
    if isinstance(graph_payload, Mapping) or isinstance(graph_payload, GraphSpec):
        graph_result = migrate_graph_spec(
            graph_payload,
            path=f"{path}/graph",
            registry=registry,
        )
        migrated_payload["graph"] = graph_result.payload
        records.extend(graph_result.migration_records)

    task_binding_payload = migrated_payload.get("task_binding_spec")
    if isinstance(task_binding_payload, Mapping):
        task_binding_result = migrate_studio_task_binding_spec(
            task_binding_payload,
            path=f"{path}/task_binding_spec",
            registry=registry,
        )
        migrated_payload["task_binding_spec"] = task_binding_result.payload
        records.extend(task_binding_result.migration_records)

    for field_name, kind in _SCENARIO_STRUCTURED_FIELDS.items():
        field_payload = migrated_payload.get(field_name)
        if isinstance(field_payload, Mapping):
            field_result = migrate_structured_spec_payload(
                kind,
                field_payload,
                path=f"{path}/{field_name}",
                registry=registry,
            )
            migrated_payload[field_name] = field_result.payload
            records.extend(field_result.migration_records)

    probe_specs = migrated_payload.get("probe_specs")
    if isinstance(probe_specs, list):
        migrated_probes: list[Any] = []
        for index, probe_payload in enumerate(probe_specs):
            if not isinstance(probe_payload, Mapping):
                migrated_probes.append(probe_payload)
                continue
            probe_result = migrate_structured_spec_payload(
                "RetainedObservableSpec",
                probe_payload,
                path=f"{path}/probe_specs/{index}",
                registry=registry,
            )
            migrated_probes.append(probe_result.payload)
            records.extend(probe_result.migration_records)
        migrated_payload["probe_specs"] = migrated_probes

    return SpecMigrationResult(
        kind=result.kind,
        schema_id=result.schema_id,
        source_version=result.source_version,
        target_version=result.target_version,
        payload=migrated_payload,
        migration_records=records,
    )


def migrate_studio_stage_spec(
    payload: Mapping[str, Any],
    *,
    source_version: str | None = None,
    target_version: str | None = None,
    path: str = "stage",
    registry: SpecSchemaRegistry | None = None,
) -> SpecMigrationResult:
    """Migrate or explicitly reject a Studio pipeline stage payload."""
    return migrate_structured_spec_payload(
        "StudioStageSpec",
        payload,
        source_version=source_version,
        target_version=target_version,
        path=path,
        registry=registry,
    )


def migrate_studio_workspace_spec(
    payload: Mapping[str, Any],
    *,
    source_version: str | None = None,
    target_version: str | None = None,
    path: str = "workspace",
    registry: SpecSchemaRegistry | None = None,
) -> SpecMigrationResult:
    """Migrate a durable Studio workspace and nested scenario/stage payloads."""
    registry = registry or default_spec_registry
    result = registry.migrate(
        "StudioWorkspaceSpec",
        payload,
        source_version=source_version,
        target_version=target_version,
    )
    migrated_payload = dict(result.payload)
    records = [_record_with_spec_path(record, path) for record in result.migration_records]

    scenarios = migrated_payload.get("scenarios")
    if isinstance(scenarios, Mapping):
        migrated_scenarios: dict[str, Any] = {}
        for scenario_id in sorted(scenarios):
            scenario_payload = scenarios[scenario_id]
            if not isinstance(scenario_payload, Mapping):
                migrated_scenarios[str(scenario_id)] = scenario_payload
                continue
            scenario_result = migrate_studio_scenario_spec(
                scenario_payload,
                path=f"{path}/scenarios/{scenario_id}",
                registry=registry,
            )
            migrated_scenarios[str(scenario_id)] = scenario_result.payload
            records.extend(scenario_result.migration_records)
        migrated_payload["scenarios"] = migrated_scenarios

    stages = migrated_payload.get("stages")
    if isinstance(stages, list):
        migrated_stages: list[Any] = []
        for index, stage_payload in enumerate(stages):
            if not isinstance(stage_payload, Mapping):
                migrated_stages.append(stage_payload)
                continue
            stage_result = migrate_studio_stage_spec(
                stage_payload,
                path=f"{path}/stages/{index}",
                registry=registry,
            )
            migrated_stages.append(stage_result.payload)
            records.extend(stage_result.migration_records)
        migrated_payload["stages"] = migrated_stages

    return SpecMigrationResult(
        kind=result.kind,
        schema_id=result.schema_id,
        source_version=result.source_version,
        target_version=result.target_version,
        payload=migrated_payload,
        migration_records=records,
    )


def migrate_graph_project_payload(
    payload: Mapping[str, Any],
    *,
    registry: SpecSchemaRegistry | None = None,
) -> dict[str, Any]:
    """Migrate durable project-level graph and workspace payloads before validation."""
    registry = registry or default_spec_registry
    migrated = dict(payload)
    graph_payload = migrated.get("graph")
    if isinstance(graph_payload, Mapping) or isinstance(graph_payload, GraphSpec):
        migrated["graph"] = migrate_graph_spec(
            graph_payload,
            path="graph",
            registry=registry,
        ).payload
    workspace_payload = migrated.get("workspace")
    if isinstance(workspace_payload, Mapping):
        migrated["workspace"] = migrate_studio_workspace_spec(
            workspace_payload,
            path="workspace",
            registry=registry,
        ).payload
    return migrated


def _register_default_spec_families(registry: SpecSchemaRegistry) -> None:
    """Populate schema identities for emitted Feedbax spec families."""
    def _old(schema_id: str) -> str:
        return f"{schema_id}.v0"

    def _policy(
        *,
        owner_module: str,
        emitted_by: tuple[str, ...],
        consumed_by: tuple[str, ...],
        stance: Literal["migrate", "reject"] = "reject",
        supported_old_versions: tuple[str, ...] = (),
        rejected_old_versions: tuple[str, ...] | None = None,
        required_tests: tuple[str, ...] = ("tests/test_structured_spec_migrations.py",),
        notes: str = "",
        covers: str | None = None,
    ) -> SpecFamilyMigrationPolicy:
        return SpecFamilyMigrationPolicy(
            owner_module=owner_module,
            emitted_by=emitted_by,
            consumed_by=consumed_by,
            stance=stance,
            supported_old_versions=supported_old_versions,
            rejected_old_versions=tuple(rejected_old_versions or ()),
            required_tests=required_tests,
            notes=notes,
            covers=covers,
        )

    def _family(
        kind: str,
        schema_id: str,
        current_version: str,
        *,
        owner_module: str,
        emitted_by: tuple[str, ...],
        consumed_by: tuple[str, ...],
        durable: bool = True,
        description: str,
        stance: Literal["migrate", "reject"] = "reject",
        supported_old_versions: tuple[str, ...] = (),
        rejected_old_versions: tuple[str, ...] | None = None,
        required_tests: tuple[str, ...] = ("tests/test_structured_spec_migrations.py",),
        notes: str = "",
        covers: str | None = None,
    ) -> SpecSchemaFamily:
        rejected_versions = rejected_old_versions
        if stance == "reject" and rejected_versions is None:
            rejected_versions = (_old(schema_id),)
        return SpecSchemaFamily(
            kind=kind,
            schema_id=schema_id,
            current_version=current_version,
            durable=durable,
            description=description,
            policy=_policy(
                owner_module=owner_module,
                emitted_by=emitted_by,
                consumed_by=consumed_by,
                stance=stance,
                supported_old_versions=supported_old_versions,
                rejected_old_versions=rejected_versions,
                required_tests=required_tests,
                notes=notes,
                covers=covers,
            ),
        )

    manifest_emitters = ("feedbax.manifest", "feedbax.provider")
    studio_schema_emitters = ("feedbax.studio_schema", "feedbax.provider")
    studio_execution_emitters = ("feedbax.studio_execution", "feedbax.provider")
    objective_emitters = ("feedbax.objective_spec", "feedbax.provider")
    execution_emitters = ("feedbax.execution_models", "feedbax.provider")

    families = [
        _family(
            "GraphSpec",
            GRAPH_SPEC_SCHEMA_ID,
            GRAPH_SPEC_SCHEMA_VERSION,
            owner_module="feedbax.contracts.graph",
            emitted_by=("Studio canvas save/load", "provider_manifest.schemas"),
            consumed_by=("feedbax.serialization.spec_to_graph", "Studio backend", "worker"),
            description="Canvas-authored executable graph specification.",
            stance="migrate",
            supported_old_versions=(LEGACY_GRAPH_SPEC_SCHEMA_VERSION,),
            rejected_old_versions=(),
            required_tests=("tests/test_graphspec_schema_migrations.py",),
        ),
        _family(
            "AdditiveGraphChannelAdapterSpec",
            "feedbax.spec.graph.additive_channel_adapter",
            GRAPH_SPEC_SCHEMA_VERSION,
            owner_module="feedbax.contracts.graph",
            emitted_by=("GraphSpec.additive_channel_adapters", "provider_manifest.schemas"),
            consumed_by=("feedbax.graph_channel_adapters",),
            description="Graph-embedded external additive channel adapter.",
        ),
        _family(
            "AdditiveGraphChannelTargetSpec",
            "feedbax.spec.graph.additive_channel_target",
            GRAPH_SPEC_SCHEMA_VERSION,
            owner_module="feedbax.contracts.graph",
            emitted_by=("AdditiveGraphChannelAdapterSpec.target", "provider_manifest.schemas"),
            consumed_by=("feedbax.graph_channel_adapters",),
            description="Target address for an additive graph-channel adapter.",
        ),
        _family(
            "PopulationStructureSpec",
            "feedbax.spec.population_structure",
            "feedbax.spec.population_structure.v1",
            owner_module="feedbax.nn",
            emitted_by=("PopulationStructure.to_spec", "GraphSpec node params"),
            consumed_by=("population_structure_from_spec", "serialization_builders"),
            description="Reusable nested spec for hidden-unit population assignments.",
            rejected_old_versions=("feedbax.population_structure.v1",),
            required_tests=(
                "tests/test_parameter_constraints.py",
                "tests/test_structured_spec_migrations.py",
            ),
        ),
        _family(
            "AnalysisInputRequirement",
            "feedbax.spec.analysis.input_requirement",
            "feedbax.spec.analysis.input_requirement.v1",
            owner_module="feedbax.contracts.graph",
            emitted_by=("GraphSpec retained analysis inputs", "provider_manifest.schemas"),
            consumed_by=("feedbax.analysis", "Studio schema enumeration"),
            description="Observable requirement declared by an analysis consumer.",
        ),
        _family(
            "RetainedObservableSpec",
            "feedbax.spec.graph.retained_observable",
            "feedbax.spec.graph.retained_observable.v1",
            owner_module="feedbax.contracts.graph",
            emitted_by=("GraphSpec.retained_observables",),
            consumed_by=("rollout retention planning", "analysis materialization"),
            description="Graph-embedded retained-observable request.",
        ),
        _family(
            "TrainingSpec",
            "feedbax.spec.training",
            "feedbax.spec.training.v1",
            owner_module="feedbax.contracts.training",
            emitted_by=("TrainingRunManifest.training_spec", "provider_manifest.schemas"),
            consumed_by=("training service", "worker"),
            description="Training optimizer, loss, and run-shape specification.",
        ),
        _family(
            "TaskSpec",
            "feedbax.spec.task",
            "feedbax.spec.task.v1",
            owner_module="feedbax.contracts.training",
            emitted_by=("TrainingRunManifest.task_spec", "provider_manifest.schemas"),
            consumed_by=("task preset lowering", "worker"),
            description="Task family and task parameter specification.",
        ),
        _family(
            "LossTermSpec",
            "feedbax.spec.training.loss_term",
            "feedbax.spec.training.loss_term.v1",
            owner_module="feedbax.contracts.training",
            emitted_by=("TrainingSpec.loss", "provider_manifest.schemas"),
            consumed_by=("training loss lowering",),
            description="Legacy structured loss-term specification.",
        ),
        _family(
            "ObjectiveSpec",
            "feedbax.spec.objective",
            "feedbax.spec.objective.v1",
            owner_module="feedbax.objective_spec",
            emitted_by=("StudioScenarioSpec.objective_spec", "provider_manifest.schemas"),
            consumed_by=("future objective lowering",),
            description="Durable selector-addressed objective specification.",
            rejected_old_versions=("feedbax.objective.v0",),
        ),
        _family(
            "EvaluationRunSpec",
            "feedbax.spec.evaluation_run",
            "feedbax.spec.evaluation_run.v1",
            owner_module="feedbax.manifest",
            emitted_by=("EvaluationRunManifest.evaluation_spec", "provider_manifest.schemas"),
            consumed_by=("feedbax.analysis.evaluation",),
            description="Declarative evaluation run request.",
        ),
        _family(
            "AnalysisRunSpec",
            "feedbax.spec.analysis_run",
            "feedbax.spec.analysis_run.v1",
            owner_module="feedbax.manifest",
            emitted_by=("AnalysisRunManifest.analysis_spec", "provider_manifest.schemas"),
            consumed_by=("feedbax.analysis.specs",),
            description="Declarative analysis run request.",
        ),
        _family(
            "ReportSpec",
            "feedbax.spec.report",
            "feedbax.spec.report.v1",
            owner_module="feedbax.manifest",
            emitted_by=("ReportManifest.report_spec", "provider_manifest.schemas"),
            consumed_by=("Studio report materialization",),
            description="Declarative report request.",
        ),
        _family(
            "CheckpointSelectionSpec",
            "feedbax.spec.checkpoint_selection",
            "feedbax.spec.checkpoint_selection.v1",
            owner_module="feedbax.manifest",
            emitted_by=("CheckpointSelectionManifest.selection_spec", "provider_manifest.schemas"),
            consumed_by=("checkpoint-selection materializers", "downstream scorer plug-ins"),
            description="Declarative generic checkpoint-selection request.",
        ),
    ]

    for kind in (
        "TargetStateLossSpec",
        "FiniteDifferenceLossSpec",
        "SelectorAddressSpec",
        "TargetValueSpec",
        "EpochMaskSpec",
        "ConstantScheduleSpec",
        "PowerLawScheduleSpec",
        "MovementEpochRampScheduleSpec",
        "MetricSpec",
        "ReductionSpec",
        "TaskTimelineSpec",
        "TimelineEpochSpec",
        "TimelineEventSpec",
    ):
        families.append(
            _family(
                kind,
                f"feedbax.spec.objective.{kind.removesuffix('Spec').lower()}",
                "feedbax.spec.objective.v1",
                owner_module="feedbax.objective_spec",
                emitted_by=objective_emitters,
                consumed_by=("ObjectiveSpec",),
                description=f"Provider-exported ObjectiveSpec submodel {kind}.",
                covers="ObjectiveSpec",
            )
        )

    for kind, schema_id, description in (
        ("ArtifactRef", "feedbax.manifest.artifact_ref", "Manifest artifact reference."),
        (
            "ArtifactValidationRecord",
            "feedbax.manifest.artifact_validation_record",
            "Artifact validation provenance record.",
        ),
        (
            "ArtifactMigrationRecord",
            "feedbax.manifest.artifact_migration_record",
            "Artifact migration provenance record.",
        ),
        ("SpecPayload", "feedbax.manifest.spec_payload", "Manifest-embedded inline spec wrapper."),
        ("ArrayStoreRef", "feedbax.manifest.array_store_ref", "Manifest array-store ref."),
        (
            "CheckpointScorerIdentity",
            "feedbax.manifest.checkpoint_selection.scorer",
            "Checkpoint-selection scorer identity record.",
        ),
        (
            "CheckpointSelectionBank",
            "feedbax.manifest.checkpoint_selection.bank",
            "Checkpoint-selection validation/evaluation bank record.",
        ),
        (
            "CheckpointCandidateRef",
            "feedbax.manifest.checkpoint_selection.candidate",
            "Checkpoint-selection candidate reference.",
        ),
        (
            "CheckpointScoreSummary",
            "feedbax.manifest.checkpoint_selection.score_summary",
            "Checkpoint-selection score summary.",
        ),
        (
            "CheckpointSelectionGroup",
            "feedbax.manifest.checkpoint_selection.group",
            "Checkpoint-selection run or replicate group.",
        ),
        ("GraphSpecManifest", "feedbax.manifest.graph_spec", "Durable graph-spec manifest."),
        (
            "GraphSpecLoadResult",
            "feedbax.manifest.graph_spec_load_result",
            "GraphSpec manifest load and migration custody result.",
        ),
        (
            "ModelArtifactManifest",
            "feedbax.manifest.model_artifact",
            "Durable model-artifact manifest.",
        ),
        (
            "TrainingRunSetManifest",
            "feedbax.manifest.training_run_set",
            "Durable training-run collection manifest.",
        ),
        ("TrainingRunManifest", "feedbax.manifest.training_run", "Durable training-run manifest."),
        (
            "EvaluationRunManifest",
            "feedbax.manifest.evaluation_run",
            "Durable evaluation-run manifest.",
        ),
        (
            "CheckpointSelectionManifest",
            "feedbax.manifest.checkpoint_selection",
            "Durable checkpoint-selection manifest.",
        ),
        ("AnalysisRunManifest", "feedbax.manifest.analysis_run", "Durable analysis-run manifest."),
        ("ReportManifest", "feedbax.manifest.report", "Durable report manifest."),
    ):
        families.append(
            _family(
                kind,
                schema_id,
                MANIFEST_SCHEMA_VERSION,
                owner_module="feedbax.manifest",
                emitted_by=manifest_emitters,
                consumed_by=("manifest load/write", "provider handoff"),
                description=description,
            )
        )

    for kind, schema_id, version, description in (
        (
            "RetentionPlan",
            RETENTION_PLAN_SCHEMA_ID,
            RETENTION_PLAN_SCHEMA_VERSION,
            "Produced training-run retention-plan artifact payload.",
        ),
        (
            "RetainedObservablesArtifact",
            RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_ID,
            RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_VERSION,
            "Produced retained-observable values artifact payload.",
        ),
        (
            "RetainedObservablePlan",
            RETAINED_OBSERVABLE_PLAN_SCHEMA_ID,
            RETAINED_OBSERVABLE_PLAN_SCHEMA_VERSION,
            "Executable retained-observable plan subrecord.",
        ),
        (
            "RetentionPolicyPlan",
            RETENTION_POLICY_PLAN_SCHEMA_ID,
            RETENTION_POLICY_PLAN_SCHEMA_VERSION,
            "Executable retention policy subrecord.",
        ),
        (
            "LossTermPlan",
            LOSS_TERM_PLAN_SCHEMA_ID,
            LOSS_TERM_PLAN_SCHEMA_VERSION,
            "Executable lowered loss term subrecord.",
        ),
    ):
        families.append(
            _family(
                kind,
                schema_id,
                version,
                owner_module="feedbax.retained_observables",
                emitted_by=(
                    "feedbax.retained_observables.retention_plan_to_json",
                    "TrainingRunManifest retention artifacts",
                ),
                consumed_by=("training-run manifest write/load", "analysis materialization"),
                description=description,
                required_tests=("tests/test_retained_observables.py",),
            )
        )

    for kind, schema_id, version, description in (
        (
            "ArrayStorePayload",
            "feedbax.manifest.array_store",
            "feedbax.manifest.array_store.v1",
            "Portable role-addressed array-store metadata payload.",
        ),
        (
            "ArrayRecord",
            "feedbax.manifest.array_record",
            "feedbax.manifest.array_roles.v1",
            "Per-array role metadata embedded in array stores.",
        ),
    ):
        families.append(
            _family(
                kind,
                schema_id,
                version,
                owner_module="feedbax.artifact_schema",
                emitted_by=("feedbax.artifact_schema", "provider_manifest.schemas"),
                consumed_by=("feedbax.artifact_schema.read_npz_array_store", "artifact materializer"),
                description=description,
            )
        )

    for kind, schema_id, description in (
        ("ExecutionSpec", "feedbax.spec.execution", "Provider-neutral execution request."),
        ("ExecutionPlan", "feedbax.manifest.execution_plan", "Inspectable concrete execution plan."),
        ("LocalExecutionResult", "feedbax.manifest.local_execution_result", "Local execution result."),
    ):
        families.append(
            _family(
                kind,
                schema_id,
                (
                    "feedbax.spec.execution.v1"
                    if kind == "ExecutionSpec"
                    else "feedbax.manifest.execution.v1"
                ),
                owner_module="feedbax.execution_models",
                emitted_by=execution_emitters,
                consumed_by=("execution planning", "Studio execution"),
                description=description,
            )
        )

    for kind, schema_id, description in (
        (
            "StudioWorkspaceSpec",
            "feedbax.spec.studio.workspace",
            "Durable Studio workspace/pipeline state.",
        ),
        (
            "StudioScenarioSpec",
            "feedbax.spec.studio.scenario",
            "Durable Studio scenario draft state.",
        ),
        (
            "StudioStageSpec",
            "feedbax.spec.studio.stage",
            "Durable Studio pipeline stage state.",
        ),
        (
            "StudioTaskBindingSpec",
            "feedbax.spec.studio.task_bindings",
            "Scenario task-data to graph binding specification.",
        ),
        (
            "StudioTaskTimelineSpec",
            "feedbax.spec.studio.task_timeline",
            "Structured Studio-authored task timeline.",
        ),
        ("StudioValueSpec", "feedbax.spec.studio.value", "Structured Studio-authored value."),
    ):
        supported = (STUDIO_TASK_BINDING_LEGACY_V1,) if kind == "StudioTaskBindingSpec" else None
        rejected = ("feedbax.studio.task_bindings.v0",) if kind == "StudioTaskBindingSpec" else None
        families.append(
            _family(
                kind,
                schema_id,
                f"{schema_id}.v2" if kind == "StudioTaskBindingSpec" else f"{schema_id}.v1",
                owner_module="feedbax.contracts.graph",
                emitted_by=("Studio save/load", "provider_manifest.schemas"),
                consumed_by=("Studio backend", "worker"),
                description=description,
                stance="migrate" if kind == "StudioTaskBindingSpec" else "reject",
                supported_old_versions=supported,
                rejected_old_versions=rejected,
            )
        )

    for kind, schema_id, description in (
        (
            "StudioTrainingExecutionRequest",
            "feedbax.spec.studio.training_execution_request",
            "Request to lower a Studio train stage into an execution plan.",
        ),
        (
            "StudioTrainingExecutionPreparation",
            "feedbax.spec.studio.training_execution_preparation",
            "Prepared Studio training execution plan.",
        ),
        (
            "StudioTrainingLocalRunRequest",
            "feedbax.spec.studio.training_local_run_request",
            "Request to execute Studio training locally.",
        ),
        (
            "StudioTrainingLocalRunResult",
            "feedbax.manifest.studio.training_local_run_result",
            "Result from local Studio training execution.",
        ),
        (
            "StudioPipelineMaterializationRequest",
            "feedbax.spec.studio.pipeline_materialization_request",
            "Request to materialize eval/analysis/report Studio stages.",
        ),
        (
            "StudioPipelineMaterializationResult",
            "feedbax.manifest.studio.pipeline_materialization_result",
            "Result from Studio pipeline materialization.",
        ),
    ):
        families.append(
            _family(
                kind,
                schema_id,
                (
                    "feedbax.manifest.studio.execution.v1"
                    if kind.endswith("Result")
                    else "feedbax.spec.studio.execution.v1"
                ),
                owner_module="feedbax.studio_execution",
                emitted_by=studio_execution_emitters,
                consumed_by=("provider HTTP API", "Studio backend"),
                description=description,
            )
        )

    for kind, schema_id, description in (
        ("ValueSchema", "feedbax.spec.studio.schema.value", "Provider-owned value schema record."),
        ("PortSchema", "feedbax.spec.studio.schema.port", "Provider-owned graph port schema."),
        ("TaskDataSchema", "feedbax.spec.studio.schema.task_data", "Provider-owned task data schema."),
        (
            "SelectorTargetSchema",
            "feedbax.spec.studio.schema.selector_target",
            "Provider-owned selectable target schema.",
        ),
        (
            "SchemaValidationIssue",
            "feedbax.spec.studio.schema.validation_issue",
            "Studio schema validation issue.",
        ),
        (
            "RuntimeIntrospectionOptions",
            "feedbax.spec.studio.runtime_introspection_options",
            "Bounded runtime introspection request options.",
        ),
        (
            "RuntimeSampleLeafSchema",
            "feedbax.spec.studio.runtime_sample_leaf",
            "Runtime sample leaf schema record.",
        ),
        (
            "RuntimeIntrospectionResult",
            "feedbax.manifest.studio.runtime_introspection",
            "Validation/runtime sample response, not a saved artifact format.",
        ),
        (
            "StudioSchemaEnumerationRequest",
            "feedbax.spec.studio.schema_enumeration_request",
            "Request to enumerate Studio schema surfaces.",
        ),
        (
            "StudioSchemaRegistry",
            "feedbax.manifest.studio.schema_registry",
            "Provider-emitted Studio schema enumeration.",
        ),
    ):
        families.append(
            _family(
                kind,
                schema_id,
                MANIFEST_SCHEMA_VERSION,
                owner_module="feedbax.studio_schema",
                emitted_by=studio_schema_emitters,
                consumed_by=("Studio frontend", "provider HTTP API"),
                durable=kind not in {
                    "StudioSchemaRegistry",
                    "RuntimeIntrospectionResult",
                    "RuntimeIntrospectionOptions",
                    "RuntimeSampleLeafSchema",
                    "StudioSchemaEnumerationRequest",
                },
                description=description,
            )
        )

    for kind, schema_id, durable, description in (
        ("ProviderManifest", "feedbax.manifest.provider", True, "Provider capability manifest."),
        ("ProviderHealth", "feedbax.manifest.provider_health", False, "Provider health response."),
        (
            "ProviderValidationResult",
            "feedbax.manifest.provider_validation_result",
            False,
            "Provider validation response.",
        ),
        ("CapabilitySpec", "feedbax.manifest.provider_capability", False, "Provider capability record."),
        (
            "MandibleArtifactMapping",
            "feedbax.manifest.mandible_artifact_mapping",
            False,
            "Mandible artifact mapping metadata.",
        ),
        (
            "MandibleManifestMapping",
            "feedbax.manifest.mandible_manifest_mapping",
            False,
            "Mandible manifest mapping metadata.",
        ),
        ("RegistrySnapshot", "feedbax.manifest.registry_snapshot", True, "Provider registry snapshot."),
        (
            "RegistryEntry",
            "feedbax.manifest.registry_entry",
            False,
            "Registry entry embedded in registry snapshots.",
        ),
        (
            "ComponentRegistrySnapshot",
            "feedbax.manifest.registry_snapshot.component",
            True,
            "Component registry snapshot capability alias.",
        ),
        (
            "TaskRegistrySnapshot",
            "feedbax.manifest.registry_snapshot.task",
            True,
            "Task registry snapshot capability alias.",
        ),
        (
            "LossRegistrySnapshot",
            "feedbax.manifest.registry_snapshot.loss",
            True,
            "Loss registry snapshot capability alias.",
        ),
        (
            "ProtocolRegistrySnapshot",
            "feedbax.manifest.registry_snapshot.protocol",
            True,
            "Protocol registry snapshot capability alias.",
        ),
        (
            "AnalysisRegistrySnapshot",
            "feedbax.manifest.registry_snapshot.analysis",
            True,
            "Analysis registry snapshot capability alias.",
        ),
    ):
        families.append(
            _family(
                kind,
                schema_id,
                MANIFEST_SCHEMA_VERSION,
                owner_module="feedbax.provider",
                emitted_by=("feedbax.provider.provider_manifest",),
                consumed_by=("Mandible provider integration",),
                durable=durable,
                description=description,
                covers="RegistrySnapshot" if kind.endswith("RegistrySnapshot") else None,
            )
        )

    for family in families:
        registry.register_family(family)

    for family in registry.families():
        if family.policy is None:
            continue
        for old_version in family.policy.rejected_old_versions:
            registry.reject_version(
                family.kind,
                old_version,
                reason=(
                    f"{family.kind} has no registered migration from {old_version!r}; "
                    f"{family.policy.owner_module} owns this schema and current-version "
                    "recreation or an explicit new migration is required."
                ),
            )


default_registry = MigrationRegistry()
default_spec_registry = SpecSchemaRegistry()
_register_default_spec_families(default_spec_registry)
default_spec_registry.register_migration(
    "GraphSpec",
    SchemaMigration(
        source_version=LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
        target_version=GRAPH_SPEC_SCHEMA_VERSION,
        migration_id="graph-spec-legacy-v1-to-v2",
        migrate=_migrate_legacy_graph_spec_payload,
        description=(
            "Promote legacy GraphSpec payloads to the explicit schema identity and "
            "rename built-in node types and Network input ports."
        ),
    ),
)
default_spec_registry.register_migration(
    "StudioTaskBindingSpec",
    SchemaMigration(
        source_version=STUDIO_TASK_BINDING_LEGACY_V1,
        target_version="feedbax.spec.studio.task_bindings.v2",
        migration_id="studio-task-bindings-v1-to-v2",
        migrate=_migrate_studio_task_binding_v1_payload,
        description=(
            "Rename exposed_outputs to exposed_data and source_output_id to "
            "source_data_id for scenario-owned task data bindings."
        ),
    ),
)
