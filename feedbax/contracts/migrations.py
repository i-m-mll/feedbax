"""Versioned schema migration registries for durable Feedbax artifacts and specs."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Literal

from feedbax.contracts.artifact_custody import (
    CONTENT_BINDING_SCHEMA_ID,
    CONTENT_BINDING_SCHEMA_VERSION,
    CONTENT_GC_RECEIPT_SCHEMA_ID,
    CONTENT_GC_RECEIPT_SCHEMA_VERSION,
    CONTENT_HOLD_RECORD_SCHEMA_ID,
    CONTENT_HOLD_RECORD_SCHEMA_VERSION,
    CONTENT_LEASE_RECORD_SCHEMA_ID,
    CONTENT_LEASE_RECORD_SCHEMA_VERSION,
    CONTENT_LEASE_RECORD_SCHEMA_VERSION_V1,
    CONTENT_MANIFEST_REF_SCHEMA_ID,
    CONTENT_MANIFEST_REF_SCHEMA_VERSION,
    CONTENT_OBJECT_REF_SCHEMA_ID,
    CONTENT_OBJECT_REF_SCHEMA_VERSION,
    CONTENT_RETENTION_INDEX_SCHEMA_ID,
    CONTENT_RETENTION_INDEX_SCHEMA_VERSION,
    CONTENT_RETENTION_INDEX_SCHEMA_VERSION_V1,
    CONTENT_ROOT_RECORD_SCHEMA_ID,
    CONTENT_ROOT_RECORD_SCHEMA_VERSION,
    CONTENT_TREE_REF_SCHEMA_ID,
    CONTENT_TREE_REF_SCHEMA_VERSION,
    CONTENT_TREE_SCHEMA_ID,
    CONTENT_TREE_SCHEMA_VERSION,
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID,
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION,
)
from feedbax.contracts.component import (
    COMPONENT_DEFINITION_DYNAMIC_PORT_POLICY_MIGRATION_ID,
    COMPONENT_DEFINITION_PORT_KIND_MIGRATION_ID,
    COMPONENT_DEFINITION_SCHEMA_ID,
    COMPONENT_DEFINITION_SCHEMA_VERSION,
    COMPONENT_DEFINITION_SCHEMA_VERSION_V1,
    COMPONENT_DEFINITION_SCHEMA_VERSION_V2,
    COMPONENT_DEFINITION_SCHEMA_VERSION_V3,
    migrate_component_definition_v1_to_v2_payload,
    migrate_component_definition_v2_to_v3_payload,
    migrate_component_definition_v3_to_v4_payload,
)
from feedbax.contracts.descriptors import (
    COMPONENT_DESCRIPTOR_SCHEMA_ID,
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    COMPONENT_SELECTOR_SYNTAX_SCHEMA_ID,
    COMPONENT_SELECTOR_SYNTAX_SCHEMA_VERSION,
    DESCRIPTOR_BASIS_SCHEMA_ID,
    DESCRIPTOR_BASIS_SCHEMA_VERSION,
    SELECTOR_FALLBACK_POLICY_SCHEMA_ID,
    SELECTOR_FALLBACK_POLICY_SCHEMA_VERSION,
    SELECTOR_ROLE_IDENTITY_SCHEMA_ID,
    SELECTOR_ROLE_IDENTITY_SCHEMA_VERSION,
    VARIABLE_DESCRIPTOR_SCHEMA_ID,
    VARIABLE_DESCRIPTOR_SCHEMA_VERSION,
)
from feedbax.contracts.expressions import (
    PATH_EXPRESSION_SCHEMA_ID,
    PATH_EXPRESSION_SCHEMA_VERSION,
)
from feedbax.contracts.execution_schema import (
    REPO_REALIZATION_PLAN_SCHEMA_ID,
    REPO_REALIZATION_PLAN_SCHEMA_VERSION,
    REPO_SNAPSHOT_MANIFEST_SCHEMA_ID,
    REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION_V1,
    REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION_V2,
)
from feedbax.contracts.domain import (
    DOMAIN_COMPILE_REPORT_SCHEMA_ID,
    DOMAIN_COMPILE_REPORT_SCHEMA_VERSION,
    DOMAIN_REGISTRY_PAYLOAD_SCHEMA_ID,
    DOMAIN_REGISTRY_PAYLOAD_SCHEMA_VERSION,
)
from feedbax.contracts.figure_layout import (
    DATA_DERIVED_AXIS_RANGE_SCHEMA_ID,
    DATA_DERIVED_AXIS_RANGE_SCHEMA_VERSION,
)
from feedbax.contracts.graph import (
    GRAPH_SPEC_SCHEMA_ID,
    GRAPH_SPEC_SCHEMA_VERSION,
    GRAPH_SPEC_SCHEMA_VERSION_V2,
    GRAPH_SPEC_SCHEMA_VERSION_V3,
    GRAPH_SPEC_SCHEMA_VERSION_V4,
    GRAPH_SPEC_SCHEMA_VERSION_V5,
    LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
    STUDIO_EPOCH_VALUE_SPEC_SCHEMA_ID,
    STUDIO_EPOCH_VALUE_SPEC_SCHEMA_VERSION,
    STUDIO_TASK_TIMELINE_SCHEMA_ID,
    STUDIO_TASK_TIMELINE_SCHEMA_VERSION,
    STUDIO_TASK_TIMELINE_SCHEMA_VERSION_V1,
    STUDIO_BIOMECHANICS_SCHEMA_ID,
    STUDIO_BIOMECHANICS_SCHEMA_VERSION,
    GraphSpec,
)
from feedbax.contracts.array_values import (
    ARRAY_VALUE_SCHEMA_ID,
    ARRAY_VALUE_SCHEMA_VERSION,
    SparseCooArrayValueSpec,
    SparseCooEntrySpec,
)
from feedbax.contracts.acausal import (
    ACAUSAL_GRAPH_SCHEMA_ID,
    ACAUSAL_GRAPH_SCHEMA_VERSION,
    ACAUSAL_GRAPH_SCHEMA_VERSION_V1,
    AcausalGraphSpec,
)
from feedbax.contracts.representation import (
    REPRESENTATION_SCHEMA_ID,
    REPRESENTATION_SCHEMA_VERSION,
    REPRESENTATION_SCHEMA_VERSION_V4,
    REPRESENTATION_SCHEMA_VERSION_V3,
    REPRESENTATION_SCHEMA_VERSION_V2,
    REPRESENTATION_SCHEMA_VERSION_V1,
    REPRESENTATION_SCHEMA_VERSION_V0,
)
from feedbax.contracts.base import (
    ArtifactMigrationRecord,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.manifest import (
    EVALUATION_RUN_SPEC_SCHEMA_ID,
    EVALUATION_RUN_SPEC_SCHEMA_VERSION,
    EVALUATION_RUN_MANIFEST_SCHEMA_VERSION,
)
from feedbax.contracts.manifest import (
    SCHEMA_VERSION as MANIFEST_SCHEMA_VERSION,
)
from feedbax.contracts.schema_namespace import (
    SchemaNamespaceKind,
    validate_schema_identity,
    validate_schema_version,
)
from feedbax.contracts.selection import (
    SELECTION_SPEC_SCHEMA_ID,
    SELECTION_SPEC_SCHEMA_VERSION,
    SELECTION_SPEC_SCHEMA_VERSION_V1,
    migrate_selection_spec_payload,
)
from feedbax.contracts.studio_api import (
    STUDIO_API_TRANSPORT_SCHEMA_ID,
    STUDIO_API_TRANSPORT_SCHEMA_VERSION,
)
from feedbax.contracts.workspace_replay import (
    WORKSPACE_REPLAY_SCHEMA_ID,
    WORKSPACE_REPLAY_SCHEMA_VERSION,
    WORKSPACE_REPLAY_SCHEMA_VERSION_V0,
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

RUN_CONFORMANCE_SCHEMA_ID = "feedbax.run_conformance"
RUN_CONFORMANCE_SCHEMA_VERSION_V1 = "feedbax.run_conformance.v1"
RUN_CONFORMANCE_SCHEMA_VERSION = "feedbax.run_conformance.v2"
REALIZED_DEPLOYMENT_RECORD_SCHEMA_ID = "feedbax.manifest.realized_deployment"
REALIZED_DEPLOYMENT_RECORD_SCHEMA_VERSION = "feedbax.manifest.realized_deployment.v1"
RUN_ASSEMBLY_REQUEST_SCHEMA_ID = "feedbax.spec.run_assembly_request"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V1 = "feedbax.spec.run_assembly_request.v1"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V2 = "feedbax.spec.run_assembly_request.v2"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V3 = "feedbax.spec.run_assembly_request.v3"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V4 = "feedbax.spec.run_assembly_request.v4"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V5 = "feedbax.spec.run_assembly_request.v5"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V6 = "feedbax.spec.run_assembly_request.v6"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V7 = "feedbax.spec.run_assembly_request.v7"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V8 = "feedbax.spec.run_assembly_request.v8"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION = "feedbax.spec.run_assembly_request.v9"
STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID = "feedbax.spec.studio.training_assembly"
STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION = "feedbax.spec.studio.training_assembly.v1"
TRAINING_DIAGNOSTICS_SCHEMA_ID = "feedbax.manifest.training_diagnostics"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V1 = "feedbax.manifest.training_diagnostics.v1"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V2 = "feedbax.manifest.training_diagnostics.v2"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3 = "feedbax.manifest.training_diagnostics.v3"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION = "feedbax.manifest.training_diagnostics.v4"

MigrationPayload = Mapping[str, Any]
MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]
MigrationRecordMetadataFn = Callable[[MigrationPayload], Mapping[str, Any]]
ComponentParamMigrationFn = Callable[[dict[str, Any]], dict[str, Any]]
_SCHEMA_MIGRATION_RESERVED_METADATA_KEYS = frozenset({"description"})


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
    record_metadata: MigrationRecordMetadataFn | None = None

    def apply(self, payload: MigrationPayload) -> tuple[dict[str, Any], ArtifactMigrationRecord]:
        """Apply this migration edge and return the migrated payload plus record."""
        source_payload = deepcopy(dict(payload))
        metadata = {"description": self.description} if self.description else {}
        if self.record_metadata is not None:
            record_metadata = dict(self.record_metadata(deepcopy(source_payload)))
            reserved = sorted(_SCHEMA_MIGRATION_RESERVED_METADATA_KEYS & record_metadata.keys())
            if reserved:
                raise ValueError(
                    f"SchemaMigration record_metadata returned reserved metadata keys: {reserved!r}"
                )
            metadata.update(record_metadata)
        migrated = self.migrate(deepcopy(source_payload))
        migrated["schema_version"] = self.target_version
        record = ArtifactMigrationRecord(
            migration_id=self.migration_id,
            source_schema_version=self.source_version,
            target_schema_version=self.target_version,
            metadata=metadata,
        )
        return migrated, record


@dataclass(frozen=True)
class ComponentMigration:
    """One deterministic component type or parameter-schema migration edge."""

    source_type: str
    target_type: str
    migration_id: str
    owner: str
    target_param_schema_version: str
    source_param_schema_version: str | None = None
    migrate_params: ComponentParamMigrationFn | None = None
    description: str = ""
    matches_unversioned: Callable[[Mapping[str, Any]], bool] | None = None

    def __post_init__(self) -> None:
        if not self.target_param_schema_version:
            raise ValueError("Component migration requires an explicit target parameter version")

    def apply(
        self,
        params: Mapping[str, Any],
        *,
        source_param_schema_version: str | None = None,
    ) -> tuple[str, dict[str, Any], str]:
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
        assume_current: bool = False,
    ) -> SpecMigrationResult:
        """Accept or migrate a structured spec payload for ``kind``.

        Versionless payloads fail closed unless ``assume_current`` is set by a
        caller that is deliberately stamping a new current spec payload.
        """
        family = self.resolve(kind)
        payload_dict = dict(payload)
        resolved_target = target_version or family.current_version
        resolved_source = source_version or _payload_schema_version(payload_dict)
        if resolved_source is None:
            if not assume_current:
                raise UnsupportedSpecVersion(
                    "Structured spec payload is missing schema_version: "
                    f"kind={family.kind!r}, schema_id={family.identity!r}, "
                    f"target_version={resolved_target!r}"
                )
            resolved_source = family.current_version

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
    if isinstance(payload, (GraphSpec, AcausalGraphSpec)):
        return payload.model_dump(mode="json")
    return dict(payload)


def _payload_metadata_version(payload: Mapping[str, Any]) -> str | None:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        version = metadata.get("version")
        if isinstance(version, str) and version:
            return version
    return None


def _canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


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


def _migrate_graph_spec_v2_to_v3_payload(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    migrated.setdefault("schema_id", GRAPH_SPEC_SCHEMA_ID)
    migrated.setdefault("derived_dimensions", [])
    return migrated


def _migrate_graph_spec_v3_to_v4_payload(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    migrated.setdefault("schema_id", GRAPH_SPEC_SCHEMA_ID)
    return migrated


def _migrate_graph_spec_v4_to_v5_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Tag only the legacy StructuralLinearStateSpace sparse delta_A form."""
    migrated = dict(payload)
    migrated.setdefault("schema_id", GRAPH_SPEC_SCHEMA_ID)
    raw_nodes = migrated.get("nodes")
    if not isinstance(raw_nodes, Mapping):
        return migrated

    nodes = dict(raw_nodes)
    for node_id, raw_node in raw_nodes.items():
        if (
            not isinstance(raw_node, Mapping)
            or raw_node.get("type") != "StructuralLinearStateSpace"
        ):
            continue
        raw_params = raw_node.get("params")
        if not isinstance(raw_params, Mapping):
            continue
        raw_delta = raw_params.get("delta_A")
        if not isinstance(raw_delta, Mapping) or set(raw_delta) != {"shape", "entries"}:
            continue
        raw_entries = raw_delta.get("entries")
        if not isinstance(raw_entries, list | tuple):
            raise TypeError("legacy StructuralLinearStateSpace delta_A.entries must be a sequence")
        entries: list[SparseCooEntrySpec] = []
        for index, raw_entry in enumerate(raw_entries):
            if isinstance(raw_entry, Mapping) and set(raw_entry) == {"row", "column", "value"}:
                row, column, value = (
                    raw_entry["row"],
                    raw_entry["column"],
                    raw_entry["value"],
                )
            elif isinstance(raw_entry, list | tuple) and len(raw_entry) == 3:
                row, column, value = raw_entry
            else:
                raise ValueError(
                    "legacy StructuralLinearStateSpace "
                    f"delta_A.entries[{index}] must contain row, column, and value"
                )
            entries.append(SparseCooEntrySpec(coordinate=(row, column), value=value))
        tagged = SparseCooArrayValueSpec(
            schema_id=ARRAY_VALUE_SCHEMA_ID,
            schema_version=ARRAY_VALUE_SCHEMA_VERSION,
            encoding="sparse_coo",
            shape=raw_delta.get("shape"),
            dtype="float64",
            nonfinite="forbid",
            fill=0.0,
            entries=entries,
        ).model_dump(mode="json")
        nodes[str(node_id)] = {
            **raw_node,
            "params": {**raw_params, "delta_A": tagged},
        }
    migrated["nodes"] = nodes
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


def _migrate_studio_task_binding_v2_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate legacy task data already routed to a concrete graph input."""

    source_data = payload.get("exposed_data") or []
    data_by_id = {data["id"]: data for data in source_data}
    bindings = []
    for raw_binding in payload.get("bindings") or []:
        binding = dict(raw_binding)
        data = data_by_id.get(binding.get("source_data_id"), {})
        metadata = data.get("metadata") or {}
        role = binding.get("role") or data.get("role") or metadata.get("task_data_role")
        if role == "component_parameter":
            temporal = metadata.get("temporality") or metadata.get("temporal_support")
            temporal = temporal if isinstance(temporal, str) else None
            shape = data.get("expected_shape")
            if shape is None:
                shape = (data.get("value_spec") or {}).get("shape")
            time_varying = temporal in {
                "trajectory",
                "materialized_trajectory",
                "epoch_masked_signal",
                "time_varying",
                "per_step",
            } or (
                temporal not in {"constant", "trial_constant", "initial", "per_trial"}
                and isinstance(shape, (list, tuple))
                and bool(shape)
                and shape[0] == "time"
            )
            if not time_varying:
                raise ValueError(
                    f"Legacy constant task binding {binding['id']!r} selected a global "
                    "state label. Open it in the original producer to identify the "
                    "owning component, then reauthor it for that component's input."
                )
            binding["role"] = "graph_input"
            binding["metadata"] = dict(binding.get("metadata", {}))
            binding["metadata"].pop("task_parameter_label", None)
        bindings.append(binding)

    exposed_data = []
    for raw_data in source_data:
        data = dict(raw_data)
        if (
            data.get("kind") == "intervention"
            and data.get("id") == "intervene"
            and data.get("path") == "intervene"
            and not data.get("bindable")
        ):
            continue
        metadata = dict(data.get("metadata", {}))
        role = data.get("role") or metadata.get("task_data_role")
        if role == "component_parameter" or data.get("kind") == "intervention":
            if data.get("bindable"):
                data.update(role="graph_input", kind="signal")
            else:
                data.update(role="protocol_value", kind="protocol_value")
            path = data.get("path", "")
            if path.startswith("intervene."):
                data["path"] = f"inputs.{data['id']}"
            metadata.pop("task_parameter_label", None)
            if "task_data_role" in metadata:
                metadata["task_data_role"] = data["role"]
            data["metadata"] = metadata
        exposed_data.append(data)
    return {**payload, "exposed_data": exposed_data, "bindings": bindings}


def _migrate_studio_value_spec_v1_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy Studio ValueSpec v1 payloads to the v2 envelope."""
    from feedbax.contracts.graph import StudioValueSpec

    return StudioValueSpec.model_validate(payload).model_dump(mode="json", exclude_none=True)


def _migrate_studio_task_timeline_v1_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Move legacy per-signal epoch membership into explicit epoch-value entries."""
    from feedbax.contracts.graph import StudioTaskTimelineSpec

    return StudioTaskTimelineSpec.model_validate(payload).model_dump(mode="json", exclude_none=True)


def _migrate_representation_spec_v1_to_v2_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Promote v1 representations to the capability-aware v2 envelope."""
    migrated = dict(payload)
    migrated.setdefault("schema_id", REPRESENTATION_SCHEMA_ID)
    return migrated


def _migrate_representation_spec_v2_to_v3_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Promote v2 representations to graph-bound muscle-path topology."""
    migrated = dict(payload)
    migrated.setdefault("schema_id", REPRESENTATION_SCHEMA_ID)
    geometry = migrated.get("muscle_path_geometry")
    if isinstance(geometry, Mapping):
        migrated["muscle_path_geometry"] = {
            key: value for key, value in geometry.items() if key != "frames"
        }
    return migrated


def _migrate_representation_spec_v3_to_v4_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Promote v3 representations to same-entity frame-provider support."""
    migrated = dict(payload)
    migrated.setdefault("schema_id", REPRESENTATION_SCHEMA_ID)
    return migrated


def _migrate_representation_spec_v4_to_v5_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Promote v4 representations to provider-declared reference poses."""
    migrated = dict(payload)
    migrated.setdefault("schema_id", REPRESENTATION_SCHEMA_ID)
    return migrated


def _decode_component_revisions(
    payload: dict[str, Any],
    component_registry: Any | None,
    *,
    historical: bool = False,
    resolve_current: bool = False,
) -> dict[str, Any]:
    nodes = payload.get("nodes")
    if not isinstance(nodes, Mapping):
        return payload
    decoded = {}
    resolver_registry = component_registry
    for name, raw_node in nodes.items():
        if not isinstance(raw_node, Mapping) or (
            "param_schema_version" not in raw_node and not historical and not resolve_current
        ):
            decoded[name] = raw_node
            continue
        node = dict(raw_node)
        token = node.pop("param_schema_version", None)
        if "revision" in node and "param_schema_version" in raw_node:
            raise ValueError(
                f"Component node {name!r} declares both revision and legacy param_schema_version"
            )
        if component_registry is None and token is None:
            decoded[name] = node
            continue
        if token is not None or historical or resolve_current:
            if resolver_registry is None:
                from feedbax.component_registry import ComponentRegistry

                resolver_registry = ComponentRegistry()
            resolution = resolver_registry.resolve_component_spec(
                node["type"],
                node.get("params", {}),
                legacy_token=token,
                revision=node.get("revision"),
                historical_unversioned=historical and token is None,
                complete_params=False,
            )
            node.update(
                type=resolution.type_id, params=resolution.params, revision=resolution.revision
            )
        decoded[name] = node
    return {**payload, "nodes": decoded}


def migrate_graph_spec(
    payload: Mapping[str, Any] | GraphSpec,
    *,
    source_version: str | None = None,
    target_version: str | None = None,
    path: str = "graph",
    registry: SpecSchemaRegistry | None = None,
    component_registry: Any | None = None,
    historical_unversioned: bool = False,
) -> SpecMigrationResult:
    """Migrate a GraphSpec payload through the public registered schema path.

    Nested subgraphs are migrated recursively through the same registry policy.
    Migration records are returned in deterministic parent-before-child order
    and include ``metadata["graph_path"]`` for provenance. Supplying a component
    registry resolves node histories in the caller's authoring or persisted-read
    context. Without one, current-format nodes remain opaque; older formats and
    legacy token fields require conversion through the built-in registry.
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
    if resolved_target == GRAPH_SPEC_SCHEMA_VERSION:
        if (
            component_registry is None
            and resolved_source != GRAPH_SPEC_SCHEMA_VERSION
            and migrated_payload.get("nodes")
        ):
            from feedbax.component_registry import ComponentRegistry

            component_registry = ComponentRegistry()
        migrated_payload = _decode_component_revisions(
            migrated_payload,
            component_registry,
            historical=historical_unversioned or resolved_source != GRAPH_SPEC_SCHEMA_VERSION,
        )
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
            nested_path = f"{path}.subgraphs[{node_id!r}]"
            nested_schema_id = None
            if isinstance(raw_subgraph, AcausalGraphSpec):
                nested_schema_id = ACAUSAL_GRAPH_SCHEMA_ID
            elif isinstance(raw_subgraph, GraphSpec):
                nested_schema_id = GRAPH_SPEC_SCHEMA_ID
            elif isinstance(raw_subgraph, Mapping):
                raw_schema_id = raw_subgraph.get("schema_id")
                nested_schema_id = raw_schema_id if isinstance(raw_schema_id, str) else None
            if nested_schema_id == ACAUSAL_GRAPH_SCHEMA_ID:
                nested = migrate_acausal_graph_spec(
                    raw_subgraph,
                    source_version=(
                        _payload_schema_version(raw_subgraph)
                        or _payload_metadata_version(raw_subgraph)
                        or (
                            ACAUSAL_GRAPH_SCHEMA_VERSION_V1
                            if resolved_source != GRAPH_SPEC_SCHEMA_VERSION
                            else ACAUSAL_GRAPH_SCHEMA_VERSION
                        )
                    )
                    if isinstance(raw_subgraph, Mapping)
                    else None,
                    path=nested_path,
                    registry=registry,
                    component_registry=component_registry,
                    historical_unversioned=historical_unversioned,
                )
            else:
                nested = migrate_graph_spec(
                    raw_subgraph,
                    source_version=nested_source,
                    target_version=resolved_target,
                    path=nested_path,
                    registry=registry,
                    component_registry=component_registry,
                    historical_unversioned=historical_unversioned,
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


def _validate_acausal_graph_spec_schema_id(payload: Mapping[str, Any], *, path: str) -> None:
    schema_id = payload.get("schema_id")
    if schema_id is not None and schema_id != ACAUSAL_GRAPH_SCHEMA_ID:
        raise UnsupportedSpecVersion(
            "Unsupported Feedbax AcausalGraphSpec schema identity: "
            f"path={path!r}, schema_id={schema_id!r}, expected={ACAUSAL_GRAPH_SCHEMA_ID!r}"
        )


def migrate_acausal_graph_spec(
    payload: Mapping[str, Any] | AcausalGraphSpec,
    *,
    source_version: str | None = None,
    target_version: str | None = None,
    path: str = "graph",
    registry: SpecSchemaRegistry | None = None,
    component_registry: Any | None = None,
    historical_unversioned: bool = False,
) -> SpecMigrationResult:
    """Read acausal graph formats with explicit component interpretation context.

    A supplied registry resolves histories; persisted readers pass
    ``historical_unversioned=True`` independently of the document format.
    Current-format nodes remain opaque when no registry is supplied. Old v1
    nodes require historical conversion through the built-in registry by default.
    """
    registry = registry or default_spec_registry
    payload_dict = _mapping_payload(payload)
    _validate_acausal_graph_spec_schema_id(payload_dict, path=path)
    resolved_source = (
        source_version
        or _payload_schema_version(payload_dict)
        or _payload_metadata_version(payload_dict)
        or ACAUSAL_GRAPH_SCHEMA_VERSION
    )
    resolved_target = target_version or registry.current_version("AcausalGraphSpec")

    result = registry.migrate(
        "AcausalGraphSpec",
        payload_dict,
        source_version=resolved_source,
        target_version=resolved_target,
    )
    migrated_payload = dict(result.payload)
    if resolved_target == ACAUSAL_GRAPH_SCHEMA_VERSION:
        if (
            component_registry is None
            and resolved_source == ACAUSAL_GRAPH_SCHEMA_VERSION_V1
            and migrated_payload.get("nodes")
        ):
            from feedbax.component_registry import ComponentRegistry

            component_registry = ComponentRegistry()
        migrated_payload = _decode_component_revisions(
            migrated_payload,
            component_registry,
            historical=historical_unversioned or resolved_source == ACAUSAL_GRAPH_SCHEMA_VERSION_V1,
            resolve_current=True,
        )
    migrated_payload.setdefault("schema_id", ACAUSAL_GRAPH_SCHEMA_ID)
    migrated_payload.setdefault("schema_version", resolved_target)
    records = [_record_with_graph_path(record, path) for record in result.migration_records]

    subgraphs = migrated_payload.get("subgraphs")
    if isinstance(subgraphs, Mapping):
        migrated_subgraphs: dict[str, Any] = {}
        for node_id in sorted(subgraphs):
            raw_subgraph = subgraphs[node_id]
            if not isinstance(raw_subgraph, Mapping) and not isinstance(
                raw_subgraph,
                AcausalGraphSpec,
            ):
                migrated_subgraphs[str(node_id)] = raw_subgraph
                continue
            nested_source = None
            if isinstance(raw_subgraph, Mapping):
                nested_source = _payload_schema_version(raw_subgraph) or resolved_source
            nested = migrate_acausal_graph_spec(
                raw_subgraph,
                source_version=nested_source,
                target_version=resolved_target,
                path=f"{path}.subgraphs[{node_id!r}]",
                registry=registry,
                component_registry=component_registry,
                historical_unversioned=historical_unversioned,
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
    assume_current: bool = False,
    path: str = "spec",
    registry: SpecSchemaRegistry | None = None,
    component_registry: Any | None = None,
) -> SpecMigrationResult:
    """Migrate or explicitly reject one registered structured spec payload."""
    registry = registry or default_spec_registry
    if kind in {"GraphSpec", "AcausalGraphSpec"}:
        migrate_graph = migrate_graph_spec if kind == "GraphSpec" else migrate_acausal_graph_spec
        return migrate_graph(
            payload,
            source_version=source_version,
            target_version=target_version,
            path=path,
            registry=registry,
            component_registry=component_registry,
            historical_unversioned=not assume_current,
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
        assume_current=assume_current,
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

    manifest_emitters = ("feedbax.contracts.manifest",)

    families = [
        _family(
            "GraphDocument",
            GRAPH_DOCUMENT_SCHEMA_ID,
            GRAPH_DOCUMENT_SCHEMA_VERSION,
            owner_module="feedbax.compiler",
            emitted_by=("feedbax.compiler.GraphDocument",),
            consumed_by=("feedbax.compiler.compile_graph",),
            description="Complete authored semantic graph document.",
            required_tests=("tests/test_graph_compiler.py",),
        ),
        *(
            _family(
                kind,
                schema_id,
                schema_version,
                owner_module="feedbax.contracts.artifact_custody",
                emitted_by=(emitter,),
                consumed_by=("canonical content-store retention and manual GC",),
                description=description,
                stance=(
                    "migrate"
                    if kind in {"ContentLeaseRecord", "ContentRetentionIndex"}
                    else "reject"
                ),
                supported_old_versions=(
                    (CONTENT_LEASE_RECORD_SCHEMA_VERSION_V1,)
                    if kind == "ContentLeaseRecord"
                    else (
                        (CONTENT_RETENTION_INDEX_SCHEMA_VERSION_V1,)
                        if kind == "ContentRetentionIndex"
                        else ()
                    )
                ),
                rejected_old_versions=(f"{schema_id}.v0",),
                required_tests=(
                    "tests/test_content_retention.py",
                    "tests/test_structured_spec_migrations.py",
                ),
            )
            for kind, schema_id, schema_version, emitter, description in (
                (
                    "ContentRootRecord",
                    CONTENT_ROOT_RECORD_SCHEMA_ID,
                    CONTENT_ROOT_RECORD_SCHEMA_VERSION,
                    "ImmutableArtifactBlobProvider.register_root",
                    "Explicit logical, run, export, import, or migration retention root.",
                ),
                (
                    "ContentLeaseRecord",
                    CONTENT_LEASE_RECORD_SCHEMA_ID,
                    CONTENT_LEASE_RECORD_SCHEMA_VERSION,
                    "ImmutableArtifactBlobProvider.acquire_lease",
                    "Explicit renewable lease with authored blocking or release expiry.",
                ),
                (
                    "ContentHoldRecord",
                    CONTENT_HOLD_RECORD_SCHEMA_ID,
                    CONTENT_HOLD_RECORD_SCHEMA_VERSION,
                    "ImmutableArtifactBlobProvider.place_hold",
                    "Explicit scientific or owner retention hold.",
                ),
                (
                    "ContentRetentionIndex",
                    CONTENT_RETENTION_INDEX_SCHEMA_ID,
                    CONTENT_RETENTION_INDEX_SCHEMA_VERSION,
                    "feedbax.persistence.content_retention",
                    "Generation-guarded canonical retention state.",
                ),
                (
                    "ContentGcReceipt",
                    CONTENT_GC_RECEIPT_SCHEMA_ID,
                    CONTENT_GC_RECEIPT_SCHEMA_VERSION,
                    "ImmutableArtifactBlobProvider.collect_garbage",
                    "Exact manual candidate, quarantine, and sweep receipt.",
                ),
            )
        ),
        _family(
            "ResolvedGraph",
            RESOLVED_GRAPH_SCHEMA_ID,
            RESOLVED_GRAPH_SCHEMA_VERSION,
            owner_module="feedbax.compiler",
            emitted_by=("feedbax.compiler.compile_graph",),
            consumed_by=("feedbax.compiler.ExecutableGraph",),
            description="Canonical executable semantic graph with source and key schedules.",
            rejected_old_versions=("1", "2", "3", "4", "5"),
            required_tests=("tests/test_graph_compiler.py", "tests/test_graph_key_schedule.py"),
        ),
        _family(
            "CompilationRecord",
            COMPILATION_RECORD_SCHEMA_ID,
            COMPILATION_RECORD_SCHEMA_VERSION,
            owner_module="feedbax.compiler",
            emitted_by=("feedbax.compiler.compile_graph",),
            consumed_by=("compiler evidence readers",),
            description="Successful graph compilation evidence with structured diagnostics.",
            rejected_old_versions=("1", "2", "3", "4", "5", "6"),
            required_tests=("tests/test_graph_compiler.py", "tests/test_graph_key_schedule.py"),
        ),
        _family(
            "CompilationFailureRecord",
            COMPILATION_FAILURE_SCHEMA_ID,
            COMPILATION_FAILURE_SCHEMA_VERSION,
            owner_module="feedbax.compiler",
            emitted_by=("feedbax.compiler.GraphCompilationError",),
            consumed_by=("Studio and CLI compiler diagnostic renderers",),
            description="Failed graph compilation evidence with source-mapped diagnostics.",
            required_tests=("tests/test_graph_compiler.py",),
        ),
        _family(
            "ImmutableArtifactBlobProviderSpec",
            IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID,
            IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION,
            owner_module="feedbax.contracts.artifact_custody",
            emitted_by=("feedbax.persistence.artifact_custody",),
            consumed_by=(
                "feedbax.persistence.open_immutable_artifact_blob_provider",
                "downstream staged execution",
            ),
            description=(
                "Portable root-free selection of the immutable local SHA-256 artifact CAS."
            ),
            rejected_old_versions=(f"{IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID}.v0",),
            required_tests=(
                "tests/test_artifact_custody.py",
                "tests/test_structured_spec_migrations.py",
            ),
        ),
        _family(
            "ImmutableArtifactBlobProviderConfig",
            IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID,
            IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION,
            owner_module="feedbax.contracts.artifact_custody",
            emitted_by=("feedbax.persistence.artifact_custody",),
            consumed_by=(
                "feedbax.persistence.open_immutable_artifact_blob_provider",
                "downstream staged execution",
            ),
            durable=False,
            description="Nested fixed-backend config covered by the portable provider spec.",
            rejected_old_versions=(f"{IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID}.v0",),
            required_tests=(
                "tests/test_artifact_custody.py",
                "tests/test_structured_spec_migrations.py",
            ),
            covers="ImmutableArtifactBlobProviderSpec",
        ),
        *(
            _family(
                kind,
                schema_id,
                schema_version,
                owner_module="feedbax.contracts.artifact_custody",
                emitted_by=("feedbax.persistence.ImmutableArtifactBlobProvider",),
                consumed_by=(
                    "feedbax.persistence.ImmutableArtifactBlobProvider",
                    "Feedbax orchestration content projection",
                ),
                description=description,
                rejected_old_versions=(f"{schema_id}.v0",),
                required_tests=(
                    "tests/test_artifact_custody.py",
                    "tests/test_structured_spec_migrations.py",
                ),
            )
            for kind, schema_id, schema_version, description in (
                (
                    "ContentObjectRef",
                    CONTENT_OBJECT_REF_SCHEMA_ID,
                    CONTENT_OBJECT_REF_SCHEMA_VERSION,
                    "Role-free exact-byte object reference in the canonical store.",
                ),
                (
                    "ContentTree",
                    CONTENT_TREE_SCHEMA_ID,
                    CONTENT_TREE_SCHEMA_VERSION,
                    "Canonical role-free directory-tree descriptor.",
                ),
                (
                    "ContentTreeRef",
                    CONTENT_TREE_REF_SCHEMA_ID,
                    CONTENT_TREE_REF_SCHEMA_VERSION,
                    "Pinned content-tree descriptor reference.",
                ),
                (
                    "ContentManifestRef",
                    CONTENT_MANIFEST_REF_SCHEMA_ID,
                    CONTENT_MANIFEST_REF_SCHEMA_VERSION,
                    "Logical manifest kind/id pinned to exact bytes.",
                ),
                (
                    "ContentBinding",
                    CONTENT_BINDING_SCHEMA_ID,
                    CONTENT_BINDING_SCHEMA_VERSION,
                    "Runtime role binding kept separate from content-tree identity.",
                ),
            )
        ),
        _family(
            "GraphSpec",
            GRAPH_SPEC_SCHEMA_ID,
            GRAPH_SPEC_SCHEMA_VERSION,
            owner_module="feedbax.contracts.graph",
            emitted_by=("Studio canvas save/load"),
            consumed_by=(
                "feedbax.compiler.compile_graph",
                "Studio backend",
                "worker",
            ),
            description="Canvas-authored executable graph specification.",
            stance="migrate",
            supported_old_versions=(
                LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
                GRAPH_SPEC_SCHEMA_VERSION_V2,
                GRAPH_SPEC_SCHEMA_VERSION_V3,
                GRAPH_SPEC_SCHEMA_VERSION_V4,
                GRAPH_SPEC_SCHEMA_VERSION_V5,
            ),
            rejected_old_versions=(),
            required_tests=("tests/test_graphspec_schema_migrations.py",),
        ),
        _family(
            "AcausalGraphSpec",
            ACAUSAL_GRAPH_SCHEMA_ID,
            ACAUSAL_GRAPH_SCHEMA_VERSION,
            owner_module="feedbax.contracts.acausal",
            emitted_by=("GraphSpec.subgraphs"),
            consumed_by=("Studio backend", "acausal domain compiler"),
            description="Durable acausal graph interior specification.",
            stance="migrate",
            supported_old_versions=(ACAUSAL_GRAPH_SCHEMA_VERSION_V1,),
            rejected_old_versions=("feedbax.spec.acausal_graph.v0",),
            required_tests=("tests/test_graphspec_schema_migrations.py",),
        ),
        _family(
            "DataDerivedAxisRange",
            DATA_DERIVED_AXIS_RANGE_SCHEMA_ID,
            DATA_DERIVED_AXIS_RANGE_SCHEMA_VERSION,
            owner_module="feedbax.contracts.figure_layout",
            emitted_by=("feedbax.contracts.figure_document.FigurePanel",),
            consumed_by=("feedbax.plot.dataset_figure",),
            description="Finite rendered-data range shared by named Cartesian panel axes.",
            rejected_old_versions=("feedbax.spec.data_derived_axis_range.v0",),
            required_tests=("tests/test_dataset_figure.py",),
        ),
        _family(
            "ComponentDefinition",
            COMPONENT_DEFINITION_SCHEMA_ID,
            COMPONENT_DEFINITION_SCHEMA_VERSION,
            owner_module="feedbax.contracts.component",
            emitted_by=("GET /api/components", "Studio component registry"),
            consumed_by=("Studio frontend", "component registry clients"),
            description="Discoverable component metadata and port typing contract.",
            stance="migrate",
            supported_old_versions=(
                COMPONENT_DEFINITION_SCHEMA_VERSION_V1,
                COMPONENT_DEFINITION_SCHEMA_VERSION_V2,
                COMPONENT_DEFINITION_SCHEMA_VERSION_V3,
            ),
            rejected_old_versions=(),
            required_tests=("tests/test_component_registration.py",),
        ),
        _family(
            "AdditiveGraphChannelAdapterSpec",
            "feedbax.spec.graph.additive_channel_adapter",
            GRAPH_SPEC_SCHEMA_VERSION,
            owner_module="feedbax.contracts.graph",
            emitted_by=("GraphSpec.additive_channel_adapters"),
            consumed_by=("feedbax.graph_channel_adapters",),
            description="Graph-embedded external additive channel adapter.",
        ),
        _family(
            "AdditiveGraphChannelTargetSpec",
            "feedbax.spec.graph.additive_channel_target",
            GRAPH_SPEC_SCHEMA_VERSION,
            owner_module="feedbax.contracts.graph",
            emitted_by=("AdditiveGraphChannelAdapterSpec.target"),
            consumed_by=("feedbax.graph_channel_adapters",),
            description="Target address for an additive graph-channel adapter.",
        ),
        _family(
            "PopulationStructureSpec",
            "feedbax.spec.population_structure",
            "feedbax.spec.population_structure.v1",
            owner_module="feedbax.builtins.models.networks",
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
            "RetainedObservableSpec",
            "feedbax.spec.graph.retained_observable",
            "feedbax.spec.graph.retained_observable.v1",
            owner_module="feedbax.contracts.graph",
            emitted_by=("GraphSpec.retained_observables",),
            consumed_by=("rollout retention planning", "analysis materialization"),
            description="Graph-embedded retained-observable request.",
        ),
        _family(
            "DomainRegistryPayload",
            DOMAIN_REGISTRY_PAYLOAD_SCHEMA_ID,
            DOMAIN_REGISTRY_PAYLOAD_SCHEMA_VERSION,
            owner_module="feedbax.contracts.domain",
            emitted_by=("GET /api/domains",),
            consumed_by=("Studio domain registry clients",),
            description="Backend registry payload for graph-domain metadata.",
            rejected_old_versions=("feedbax.spec.domain.v0",),
            required_tests=("tests/test_domain_registry.py",),
        ),
        _family(
            "DomainCompileReport",
            DOMAIN_COMPILE_REPORT_SCHEMA_ID,
            DOMAIN_COMPILE_REPORT_SCHEMA_VERSION,
            owner_module="feedbax.contracts.domain",
            emitted_by=("POST /api/graphs/{graph_id}/nodes/compile", "Studio save/load"),
            consumed_by=("Studio frontend", "Studio backend"),
            description=(
                "Derived authoring compile report cache. Unknown or old versions are "
                "dropped on project load and the node reverts to never_compiled."
            ),
            rejected_old_versions=("feedbax.spec.domain_compile_report.v0",),
            required_tests=("tests/test_acausal_compile_reports.py",),
        ),
        _family(
            "RepoSnapshotManifest",
            REPO_SNAPSHOT_MANIFEST_SCHEMA_ID,
            REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
            owner_module="feedbax.orchestration.repo_snapshot",
            emitted_by=("feedbax.orchestration.repo_snapshot.seal_repo_snapshots",),
            consumed_by=("feedbax.orchestration.drivers.runpod.RunPodOrchestrationDriver",),
            description="Sealed source transfer authority with generated distribution provenance.",
            rejected_old_versions=(
                "feedbax.orchestration.repo_snapshot_manifest.v0",
                REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION_V1,
                REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION_V2,
            ),
            required_tests=(
                "tests/test_repo_snapshot.py",
                "tests/test_structured_spec_migrations.py",
            ),
        ),
        _family(
            "RepoRealizationPlan",
            REPO_REALIZATION_PLAN_SCHEMA_ID,
            REPO_REALIZATION_PLAN_SCHEMA_VERSION,
            owner_module="feedbax.orchestration.repo_realization",
            emitted_by=("feedbax.orchestration.drivers.runpod.build_runpod_repo_realization_plan",),
            consumed_by=("feedbax.orchestration.drivers.runpod.RunPodOrchestrationDriver",),
            description="Content-addressed multi-repository realization authority.",
            rejected_old_versions=(f"{REPO_REALIZATION_PLAN_SCHEMA_ID}.v0",),
            required_tests=(
                "tests/test_repo_realization.py",
                "tests/test_structured_spec_migrations.py",
            ),
        ),
        _family(
            "EvaluationRunSpec",
            EVALUATION_RUN_SPEC_SCHEMA_ID,
            EVALUATION_RUN_SPEC_SCHEMA_VERSION,
            owner_module="feedbax.contracts.manifest",
            emitted_by=("feedbax.training.evaluate_worker",),
            consumed_by=("feedbax.studio.canonical_evaluate", "feedbax.persistence.evaluation_subject"),
            description="Managed rollout specification projected into its evaluation manifest.",
            rejected_old_versions=(f"{EVALUATION_RUN_SPEC_SCHEMA_ID}.v0",),
            required_tests=("tests/test_canonical_evaluate_admission.py",),
        ),
        _family(
            "SelectorRoleIdentity",
            SELECTOR_ROLE_IDENTITY_SCHEMA_ID,
            SELECTOR_ROLE_IDENTITY_SCHEMA_VERSION,
            owner_module="feedbax.contracts.descriptors",
            emitted_by=("VariableDescriptor.role"),
            consumed_by=("descriptor resolution", "downstream selector validation"),
            description="Namespaced selector-role identity used by descriptor consumers.",
            required_tests=(
                "tests/test_descriptor_schema.py",
                "tests/test_structured_spec_migrations.py",
            ),
        ),
        _family(
            "ComponentSelectorSyntax",
            COMPONENT_SELECTOR_SYNTAX_SCHEMA_ID,
            COMPONENT_SELECTOR_SYNTAX_SCHEMA_VERSION,
            owner_module="feedbax.contracts.descriptors",
            emitted_by=("VariableDescriptor.selector_syntax"),
            consumed_by=("descriptor resolution", "downstream selector validation"),
            description="Namespaced component-selector syntax identity.",
            required_tests=(
                "tests/test_descriptor_schema.py",
                "tests/test_structured_spec_migrations.py",
            ),
        ),
        _family(
            "SelectorFallbackPolicyIdentity",
            SELECTOR_FALLBACK_POLICY_SCHEMA_ID,
            SELECTOR_FALLBACK_POLICY_SCHEMA_VERSION,
            owner_module="feedbax.contracts.descriptors",
            emitted_by=("VariableDescriptor.fallback_policy"),
            consumed_by=("descriptor resolution", "downstream selector validation"),
            description="Namespaced selector fallback-policy identity.",
            required_tests=(
                "tests/test_descriptor_schema.py",
                "tests/test_structured_spec_migrations.py",
            ),
        ),
        _family(
            "VariableDescriptor",
            VARIABLE_DESCRIPTOR_SCHEMA_ID,
            VARIABLE_DESCRIPTOR_SCHEMA_VERSION,
            owner_module="feedbax.contracts.descriptors",
            emitted_by=("GraphSpec/training/run metadata"),
            consumed_by=("descriptor resolution", "downstream selector validation"),
            description="Selectable variable descriptor contract.",
            required_tests=(
                "tests/test_descriptor_schema.py",
                "tests/test_structured_spec_migrations.py",
            ),
        ),
        _family(
            "ComponentDescriptor",
            COMPONENT_DESCRIPTOR_SCHEMA_ID,
            COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
            owner_module="feedbax.contracts.descriptors",
            emitted_by=("VariableDescriptor components"),
            consumed_by=("descriptor resolution", "downstream selector validation"),
            description="Selectable scalar or slice component descriptor contract.",
            required_tests=(
                "tests/test_descriptor_schema.py",
                "tests/test_structured_spec_migrations.py",
            ),
        ),
        _family(
            "DescriptorBasisIdentity",
            DESCRIPTOR_BASIS_SCHEMA_ID,
            DESCRIPTOR_BASIS_SCHEMA_VERSION,
            owner_module="feedbax.contracts.descriptors",
            emitted_by=("descriptor-bearing specs"),
            consumed_by=("descriptor resolution", "analysis data-product basis pins"),
            description="Whole descriptor-basis identity and hash contract.",
            required_tests=(
                "tests/test_descriptor_schema.py",
                "tests/test_structured_spec_migrations.py",
            ),
        ),
        _family(
            "PathExpression",
            PATH_EXPRESSION_SCHEMA_ID,
            PATH_EXPRESSION_SCHEMA_VERSION,
            owner_module="feedbax.contracts.expressions",
            emitted_by=("bundle conditions", "data-product extraction", "report predicates"),
            consumed_by=(
                "feedbax.contracts.expressions.evaluate_expr",
                "feedbax.contracts.expressions.evaluate_query",
            ),
            description="Composable manifest path-expression and value-query AST.",
            required_tests=(
                "tests/test_path_expressions.py",
                "tests/test_structured_spec_migrations.py",
            ),
        ),
        _family(
            "CheckpointSelectionSpec",
            "feedbax.spec.checkpoint_selection",
            "feedbax.spec.checkpoint_selection.v1",
            owner_module="feedbax.contracts.manifest",
            emitted_by=("CheckpointSelectionManifest.selection_spec"),
            consumed_by=("checkpoint-selection materializers", "downstream scorer plug-ins"),
            description="Declarative generic checkpoint-selection request.",
        ),
        _family(
            "SelectionSpec",
            SELECTION_SPEC_SCHEMA_ID,
            SELECTION_SPEC_SCHEMA_VERSION,
            owner_module="feedbax.contracts.selection",
            emitted_by=("Studio stage selection_spec"),
            consumed_by=(
                "Studio pipeline staging",
                "evaluation/analysis/queue/lineage consumers",
            ),
            description="Explicit, query, and frozen manifest input selection.",
            stance="migrate",
            supported_old_versions=(SELECTION_SPEC_SCHEMA_VERSION_V1,),
            rejected_old_versions=(),
            required_tests=(
                "tests/test_selection_spec.py",
                "tests/test_structured_spec_migrations.py",
            ),
        ),
    ]

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
    ):
        families.append(
            _family(
                kind,
                schema_id,
                EVALUATION_RUN_MANIFEST_SCHEMA_VERSION
                if kind == "EvaluationRunManifest"
                else MANIFEST_SCHEMA_VERSION,
                owner_module="feedbax.contracts.manifest",
                emitted_by=manifest_emitters,
                consumed_by=("manifest load/write", "provider handoff"),
                description=description,
                required_tests=("tests/test_structured_spec_migrations.py",),
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
                owner_module="feedbax.contracts.artifact_schema",
                emitted_by=("feedbax.contracts.artifact_schema"),
                consumed_by=(
                    "feedbax.contracts.artifact_schema.read_npz_array_store",
                    "artifact materializer",
                ),
                description=description,
            )
        )

    families.append(
        _family(
            "StudioApiTransport",
            STUDIO_API_TRANSPORT_SCHEMA_ID,
            STUDIO_API_TRANSPORT_SCHEMA_VERSION,
            owner_module="feedbax.contracts.studio_api",
            emitted_by=("feedbax.contracts.studio_api", "scripts.generate_studio_contracts"),
            consumed_by=("Studio frontend", "Studio HTTP API"),
            description="Shared schema identity for Studio HTTP transport models.",
            rejected_old_versions=("feedbax.spec.studio.api_transport.v0",),
            required_tests=(
                "tests/test_studio_api_contracts.py",
                "tests/test_structured_spec_migrations.py",
            ),
        )
    )
    families.append(
        _family(
            "RepresentationSpec",
            REPRESENTATION_SCHEMA_ID,
            REPRESENTATION_SCHEMA_VERSION,
            owner_module="feedbax.contracts.representation",
            emitted_by=(
                "feedbax.component_registry.registry.ComponentRegistry",
                "scripts.generate_studio_contracts",
            ),
            consumed_by=("Studio frontend", "workspace renderer"),
            description="Component-owned workspace representation declaration.",
            stance="migrate",
            supported_old_versions=(
                REPRESENTATION_SCHEMA_VERSION_V1,
                REPRESENTATION_SCHEMA_VERSION_V2,
                REPRESENTATION_SCHEMA_VERSION_V3,
                REPRESENTATION_SCHEMA_VERSION_V4,
            ),
            rejected_old_versions=(REPRESENTATION_SCHEMA_VERSION_V0,),
            required_tests=(
                "tests/test_component_registration.py",
                "tests/test_structured_spec_migrations.py",
                "web/src/generated/studioContracts.ts",
            ),
        )
    )

    for kind, schema_id, description in (
        (
            "StudioBiomechanicsSpec",
            STUDIO_BIOMECHANICS_SCHEMA_ID,
            "Scenario-local biomechanics metadata boundary.",
        ),
        (
            "StudioTaskBindingSpec",
            "feedbax.spec.studio.task_bindings",
            "Scenario task-data to graph binding specification.",
        ),
        (
            "StudioTaskTimelineSpec",
            STUDIO_TASK_TIMELINE_SCHEMA_ID,
            "Structured Studio-authored task timeline.",
        ),
        (
            "StudioEpochValueSpec",
            STUDIO_EPOCH_VALUE_SPEC_SCHEMA_ID,
            "One authored value over one named task epoch interval.",
        ),
        ("StudioValueSpec", "feedbax.spec.studio.value", "Structured Studio-authored value."),
    ):
        if kind == "StudioBiomechanicsSpec":
            current_version = STUDIO_BIOMECHANICS_SCHEMA_VERSION
            stance = "reject"
            supported = None
            rejected = ("feedbax.spec.studio.biomechanics.v0",)
        elif kind == "StudioTaskBindingSpec":
            current_version = f"{schema_id}.v3"
            stance = "migrate"
            supported = (STUDIO_TASK_BINDING_LEGACY_V1, f"{schema_id}.v2")
            rejected = ("feedbax.studio.task_bindings.v0",)
        elif kind == "StudioValueSpec":
            current_version = f"{schema_id}.v2"
            stance = "migrate"
            supported = ("feedbax.spec.studio.value.v1", "feedbax.studio.value.v1")
            rejected = ("feedbax.spec.studio.value.v0",)
        elif kind == "StudioTaskTimelineSpec":
            current_version = STUDIO_TASK_TIMELINE_SCHEMA_VERSION
            stance = "migrate"
            supported = (
                STUDIO_TASK_TIMELINE_SCHEMA_VERSION_V1,
                "feedbax.studio.task_timeline.v1",
            )
            rejected = ("feedbax.spec.studio.task_timeline.v0",)
        elif kind == "StudioEpochValueSpec":
            current_version = STUDIO_EPOCH_VALUE_SPEC_SCHEMA_VERSION
            stance = "reject"
            supported = None
            rejected = ("feedbax.spec.studio.epoch_value.v0",)
        else:
            current_version = f"{schema_id}.v1"
            stance = "reject"
            supported = None
            rejected = None
        families.append(
            _family(
                kind,
                schema_id,
                current_version,
                owner_module="feedbax.contracts.graph",
                emitted_by=("Studio save/load"),
                consumed_by=("Studio backend", "worker"),
                description=description,
                stance=stance,
                supported_old_versions=supported,
                rejected_old_versions=rejected,
            )
        )

    families.append(
        _family(
            "WorkspaceReplayProduct",
            WORKSPACE_REPLAY_SCHEMA_ID,
            WORKSPACE_REPLAY_SCHEMA_VERSION,
            owner_module="feedbax.contracts.workspace_replay",
            emitted_by=("eval/validation replay materialization"),
            consumed_by=(
                "Studio project playback",
                "Mandible provider integration",
                "analysis materialization",
            ),
            description="Manifest-linked replay product for authored Studio geometry.",
            rejected_old_versions=(WORKSPACE_REPLAY_SCHEMA_VERSION_V0,),
            required_tests=(
                "tests/test_workspace_replay_contract.py",
                "tests/test_structured_spec_migrations.py",
            ),
        )
    )


    for family in families:
        registry.register_family(family)

    for family in registry.families():
        if family.policy is None:
            continue
        for old_version in family.policy.rejected_old_versions:
            remediation = (
                f"{family.kind} has no registered migration from {old_version!r}; "
                f"{family.policy.owner_module} owns this schema and current-version "
                "recreation or an explicit new migration is required."
            )
            if family.kind in {"ResolvedGraph", "CompilationRecord"}:
                remediation = (
                    "Historical compiler evidence cannot establish current resolved "
                    "semantics; recompile the authored GraphDocument."
                )
            registry.reject_version(
                family.kind,
                old_version,
                reason=remediation,
            )


def _parameter_free_acknowledgement_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(dict(payload))
    migrated.setdefault("parameters", {})
    parameters_sha256 = sha256_bytes(canonical_json_bytes(migrated["parameters"]))
    for field_name in ("fragment", "merge_state"):
        artifact = migrated.get(field_name)
        if isinstance(artifact, dict):
            metadata = dict(artifact.get("metadata", {}))
            metadata.setdefault("consumer_parameters", migrated["parameters"])
            metadata.setdefault("consumer_parameters_sha256", parameters_sha256)
            artifact["metadata"] = metadata
    return migrated


def _migrate_content_lease_record_v1(payload: dict[str, Any]) -> dict[str, Any]:
    """Preserve historical leases as explicit-release safety barriers."""
    migrated = deepcopy(payload)
    migrated["schema_version"] = CONTENT_LEASE_RECORD_SCHEMA_VERSION
    migrated["expiry_disposition"] = "block"
    return migrated


def _migrate_content_retention_index_v1(payload: dict[str, Any]) -> dict[str, Any]:
    """Preserve v1 retention state while typing every nested lease expiry."""
    migrated = deepcopy(payload)
    migrated["schema_version"] = CONTENT_RETENTION_INDEX_SCHEMA_VERSION
    migrated["leases"] = [
        _migrate_content_lease_record_v1(lease) for lease in migrated.get("leases", [])
    ]
    return migrated


default_spec_registry = SpecSchemaRegistry()
_register_default_spec_families(default_spec_registry)
default_spec_registry.register_migration(
    "ContentLeaseRecord",
    SchemaMigration(
        source_version=CONTENT_LEASE_RECORD_SCHEMA_VERSION_V1,
        target_version=CONTENT_LEASE_RECORD_SCHEMA_VERSION,
        migration_id="content-lease-record-v1-to-v2-expiry-disposition",
        migrate=_migrate_content_lease_record_v1,
        description="Preserve historical lease expiry as an explicit-release safety barrier.",
    ),
)
default_spec_registry.register_migration(
    "ContentRetentionIndex",
    SchemaMigration(
        source_version=CONTENT_RETENTION_INDEX_SCHEMA_VERSION_V1,
        target_version=CONTENT_RETENTION_INDEX_SCHEMA_VERSION,
        migration_id="content-retention-index-v1-to-v2-expiry-disposition",
        migrate=_migrate_content_retention_index_v1,
        description="Type nested lease expiry without changing historical retention meaning.",
    ),
)
default_spec_registry.register_migration(
    "RepresentationSpec",
    SchemaMigration(
        source_version=REPRESENTATION_SCHEMA_VERSION_V1,
        target_version=REPRESENTATION_SCHEMA_VERSION_V2,
        migration_id="representation-spec-v1-to-v2-reachability-capability",
        migrate=_migrate_representation_spec_v1_to_v2_payload,
        description="Add the optional provider-declared reachability capability envelope.",
    ),
)
default_spec_registry.register_migration(
    "RepresentationSpec",
    SchemaMigration(
        source_version=REPRESENTATION_SCHEMA_VERSION_V2,
        target_version=REPRESENTATION_SCHEMA_VERSION_V3,
        migration_id="representation-spec-v2-to-v3-muscle-path-geometry",
        migrate=_migrate_representation_spec_v2_to_v3_payload,
        description=(
            "Add provider-owned muscle-path topology; frame transforms are resolved from graph "
            "wiring and are never persisted in the geometry payload."
        ),
    ),
)
default_spec_registry.register_migration(
    "RepresentationSpec",
    SchemaMigration(
        source_version=REPRESENTATION_SCHEMA_VERSION_V3,
        target_version=REPRESENTATION_SCHEMA_VERSION_V4,
        migration_id="representation-spec-v3-to-v4-same-entity-frame-provider",
        migrate=_migrate_representation_spec_v3_to_v4_payload,
        description=(
            "Add typed same-entity planar-chain frame providers for self-contained components."
        ),
    ),
)
default_spec_registry.register_migration(
    "RepresentationSpec",
    SchemaMigration(
        source_version=REPRESENTATION_SCHEMA_VERSION_V4,
        target_version=REPRESENTATION_SCHEMA_VERSION,
        migration_id="representation-spec-v4-to-v5-reference-pose",
        migrate=_migrate_representation_spec_v4_to_v5_payload,
        description=(
            "Add optional provider-declared planar-chain reference poses in configuration "
            "coordinates."
        ),
    ),
)
default_spec_registry.register_migration(
    "SelectionSpec",
    SchemaMigration(
        source_version=SELECTION_SPEC_SCHEMA_VERSION_V1,
        target_version=SELECTION_SPEC_SCHEMA_VERSION,
        migration_id="selection-spec-v1-to-v2-query-forms",
        migrate=migrate_selection_spec_payload,
        description=("Promote legacy id-list selection payloads to explicit SelectionSpec v2."),
    ),
)
default_spec_registry.register_migration(
    "AcausalGraphSpec",
    SchemaMigration(
        source_version=ACAUSAL_GRAPH_SCHEMA_VERSION_V1,
        target_version=ACAUSAL_GRAPH_SCHEMA_VERSION,
        migration_id="acausal-graph-v1-to-v2-component-revisions",
        migrate=dict,
        description="Interpret unversioned historical components before canonical revision output.",
    ),
)
default_spec_registry.register_migration(
    "GraphSpec",
    SchemaMigration(
        source_version=LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
        target_version=GRAPH_SPEC_SCHEMA_VERSION_V2,
        migration_id="graph-spec-legacy-v1-to-v2",
        migrate=_migrate_legacy_graph_spec_payload,
        description=(
            "Promote legacy GraphSpec payloads to the explicit schema identity and "
            "rename built-in node types and Network input ports."
        ),
    ),
)
default_spec_registry.register_migration(
    "GraphSpec",
    SchemaMigration(
        source_version=GRAPH_SPEC_SCHEMA_VERSION_V2,
        target_version=GRAPH_SPEC_SCHEMA_VERSION_V3,
        migration_id="graph-spec-v2-to-v3-derived-dimensions",
        migrate=_migrate_graph_spec_v2_to_v3_payload,
        description="Add explicit derived_dimensions rules to GraphSpec.",
    ),
)
default_spec_registry.register_migration(
    "GraphSpec",
    SchemaMigration(
        source_version=GRAPH_SPEC_SCHEMA_VERSION_V3,
        target_version=GRAPH_SPEC_SCHEMA_VERSION_V4,
        migration_id="graph-spec-v3-to-v4-discriminated-subgraphs",
        migrate=_migrate_graph_spec_v3_to_v4_payload,
        description="Allow discriminated causal/acausal subgraph payloads.",
    ),
)
default_spec_registry.register_migration(
    "GraphSpec",
    SchemaMigration(
        source_version=GRAPH_SPEC_SCHEMA_VERSION_V4,
        target_version=GRAPH_SPEC_SCHEMA_VERSION_V5,
        migration_id="graph-spec-v4-to-v5-component-param-array-values",
        migrate=_migrate_graph_spec_v4_to_v5_payload,
        description=(
            "Tag the legacy StructuralLinearStateSpace sparse delta_A form while "
            "leaving dense and unrelated component parameters unchanged."
        ),
    ),
)
default_spec_registry.register_migration(
    "GraphSpec",
    SchemaMigration(
        source_version=GRAPH_SPEC_SCHEMA_VERSION_V5,
        target_version=GRAPH_SPEC_SCHEMA_VERSION,
        migration_id="graph-spec-v5-to-v6-component-revisions",
        migrate=dict,
        description="Use canonical integer component revisions.",
    ),
)


default_spec_registry.register_migration(
    "ComponentDefinition",
    SchemaMigration(
        source_version=COMPONENT_DEFINITION_SCHEMA_VERSION_V1,
        target_version=COMPONENT_DEFINITION_SCHEMA_VERSION_V2,
        migration_id=COMPONENT_DEFINITION_PORT_KIND_MIGRATION_ID,
        migrate=migrate_component_definition_v1_to_v2_payload,
        description="Default legacy component port metadata to explicit signal ports.",
    ),
)
default_spec_registry.register_migration(
    "ComponentDefinition",
    SchemaMigration(
        source_version=COMPONENT_DEFINITION_SCHEMA_VERSION_V2,
        target_version=COMPONENT_DEFINITION_SCHEMA_VERSION_V3,
        migration_id=COMPONENT_DEFINITION_DYNAMIC_PORT_POLICY_MIGRATION_ID,
        migrate=migrate_component_definition_v2_to_v3_payload,
        description="Add the optional declarative dynamic-port policy field.",
    ),
)
default_spec_registry.register_migration(
    "ComponentDefinition",
    SchemaMigration(
        source_version=COMPONENT_DEFINITION_SCHEMA_VERSION_V3,
        target_version=COMPONENT_DEFINITION_SCHEMA_VERSION,
        migration_id="component-definition-v3-to-v4-component-revisions",
        migrate=migrate_component_definition_v3_to_v4_payload,
        description="Resolve exact historical component tokens to canonical revisions.",
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
default_spec_registry.register_migration(
    "StudioTaskBindingSpec",
    SchemaMigration(
        source_version="feedbax.spec.studio.task_bindings.v2",
        target_version="feedbax.spec.studio.task_bindings.v3",
        migration_id="studio-task-bindings-v2-to-v3",
        migrate=_migrate_studio_task_binding_v2_payload,
        description="Bind task controls directly to graph input ports.",
    ),
)
default_spec_registry.register_migration(
    "StudioValueSpec",
    SchemaMigration(
        source_version="feedbax.spec.studio.value.v1",
        target_version="feedbax.spec.studio.value.v2",
        migration_id="studio-value-spec-v1-to-v2",
        migrate=_migrate_studio_value_spec_v1_payload,
        description="Split legacy mode/sampling_scope into value_form and variation.",
    ),
)
default_spec_registry.register_migration(
    "StudioTaskTimelineSpec",
    SchemaMigration(
        source_version=STUDIO_TASK_TIMELINE_SCHEMA_VERSION_V1,
        target_version=STUDIO_TASK_TIMELINE_SCHEMA_VERSION,
        migration_id="studio-task-timeline-v1-to-v2-epoch-values",
        migrate=_migrate_studio_task_timeline_v1_payload,
        description="Move per-signal epoch membership into explicit epoch-value entries.",
    ),
)
default_spec_registry.register_migration(
    "StudioTaskTimelineSpec",
    SchemaMigration(
        source_version="feedbax.studio.task_timeline.v1",
        target_version=STUDIO_TASK_TIMELINE_SCHEMA_VERSION,
        migration_id="studio-task-timeline-frontend-v1-to-v2-epoch-values",
        migrate=_migrate_studio_task_timeline_v1_payload,
        description="Normalize the legacy frontend timeline spelling and epoch values.",
    ),
)
default_spec_registry.register_migration(
    "StudioValueSpec",
    SchemaMigration(
        source_version="feedbax.studio.value.v1",
        target_version="feedbax.spec.studio.value.v2",
        migration_id="studio-value-spec-frontend-v1-to-v2",
        migrate=_migrate_studio_value_spec_v1_payload,
        description="Normalize frontend-emitted legacy ValueSpec v1 spelling.",
    ),
)
