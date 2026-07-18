"""Typed authored-request to executable RunBundle assembly."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol

from pydantic import Field, JsonValue, model_validator

from feedbax.contracts.manifest import StrictModel
from feedbax.contracts.run_matrix import TrainingRowProvenance
from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot
from feedbax.contracts.spec_storage import (
    build_resolved_semantics_snapshot,
    canonicalize_immutable_input_identities,
    store_canonical_json_artifact,
    training_run_execution_hash,
    training_spec_canonical_bytes,
)
from feedbax.orchestration.bundle import (
    EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID,
    EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION,
    AuthoredIntentRef,
    BudgetPolicy,
    DeploymentPolicy,
    EnvironmentDeclaration,
    ExecutionCapsuleRef,
    ExecutionIdentityEnvelope,
    ImmutableInputIdentity,
    LaunchPolicy,
    ResolvedSnapshotRef,
    RowLaunchSpec,
    RunBundle,
    RunRowSpec,
    SchemaArtifactRef,
)

if TYPE_CHECKING:
    from feedbax.contracts.migrations import SpecMigrationResult, SpecSchemaRegistry


RUN_ASSEMBLY_REQUEST_SCHEMA_ID = "feedbax.spec.run_assembly_request"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V1 = f"{RUN_ASSEMBLY_REQUEST_SCHEMA_ID}.v1"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION = f"{RUN_ASSEMBLY_REQUEST_SCHEMA_ID}.v2"


class CompilerIdentity(StrictModel):
    """Explicit compiler dispatch identity."""

    compiler_id: str = Field(min_length=1)
    compiler_version: str = Field(min_length=1)


class AssemblyInputDeclaration(StrictModel):
    """Unresolved external input declaration supplied by an author."""

    role: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    schema_id: str | None = None
    schema_version: str | None = None

    @model_validator(mode="after")
    def _validate_schema_pair(self) -> "AssemblyInputDeclaration":
        if (self.schema_id is None) != (self.schema_version is None):
            raise ValueError("schema_id and schema_version must be supplied together")
        return self


class RunAssemblyRequest(StrictModel):
    """Durable authored input consumed by the persisted ASSEMBLE stage."""

    schema_id: str = RUN_ASSEMBLY_REQUEST_SCHEMA_ID
    schema_version: str = RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION
    authored: SchemaArtifactRef
    compiler: CompilerIdentity
    inputs: list[AssemblyInputDeclaration] = Field(default_factory=list)
    deployment_policy: DeploymentPolicy
    environment: EnvironmentDeclaration
    launch_policy: LaunchPolicy = Field(default_factory=LaunchPolicy)
    budget: BudgetPolicy
    orchestration_root: str | None = None
    keep_alive: bool = False
    deadman_enabled: bool = False
    deadman_silence_seconds: int = Field(default=1800, ge=60)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schema(self) -> "RunAssemblyRequest":
        if self.schema_id != RUN_ASSEMBLY_REQUEST_SCHEMA_ID:
            raise ValueError("unsupported run assembly request schema_id")
        if self.schema_version != RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION:
            raise ValueError("unsupported run assembly request schema_version")
        return self


class CompiledExecutionRow(StrictModel):
    """Pure compiler output before ASSEMBLE writes custody artifacts."""

    row_id: str = Field(min_length=1)
    payload: dict[str, JsonValue]
    resolved_semantics: dict[str, JsonValue]
    provenance: TrainingRowProvenance | None = None
    immutable_inputs: list[ImmutableInputIdentity] = Field(default_factory=list)
    launch: RowLaunchSpec

    @model_validator(mode="after")
    def _require_registered_payload(self) -> "CompiledExecutionRow":
        if not isinstance(self.payload.get("schema_id"), str) or not isinstance(
            self.payload.get("schema_version"), str
        ):
            raise ValueError("compiled payload requires schema_id and schema_version")
        if self.provenance is not None and self.provenance.row_id != self.row_id:
            raise ValueError("compiled row provenance row_id must match row_id")
        return self


class CompiledRunSet(StrictModel):
    """Pure compiler output for one authored request."""

    rows: list[CompiledExecutionRow] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_rows(self) -> "CompiledRunSet":
        ids = [row.row_id for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("compiled run set contains duplicate row_id values")
        return self


class RowIdentities(StrictModel):
    """Generic identities fixed before a family-specific capsule is built."""

    intent_hash: str
    resolved_root_hash: str
    immutable_inputs: list[dict[str, JsonValue]]
    execution_hash: str


class ExecutionIdentityAdapter(Protocol):
    """Family-specific semantic binding hook used by generic ASSEMBLE."""

    def intent_hash(self, authored: Mapping[str, Any]) -> str: ...

    def build_capsule(
        self,
        row: CompiledExecutionRow,
        *,
        identities: RowIdentities,
        context: "AssemblyContext",
    ) -> Mapping[str, Any]: ...

    def capsule_identities(self, capsule: Mapping[str, Any]) -> RowIdentities: ...


class AssemblyCompiler(Protocol):
    """Compiler from a verified authored payload to pure typed rows."""

    def compile(
        self,
        *,
        authored: Mapping[str, Any],
        run_set_id: str,
        context: "AssemblyContext",
    ) -> CompiledRunSet: ...


ArtifactResolver = Callable[[SchemaArtifactRef], bytes]
InputResolver = Callable[[AssemblyInputDeclaration], ImmutableInputIdentity]


@dataclass(frozen=True)
class AssemblyContext:
    """Machine-local assembly dependencies excluded from durable identity."""

    custody_root: Path
    repo_root: Path | None = None
    schema_registry: "SpecSchemaRegistry | None" = None
    artifact_resolver: ArtifactResolver | None = None
    input_resolver: InputResolver | None = None
    materializer_commit: str = "unknown"
    dependency_lock_digest: str = "0" * 64
    environment_digest: str | None = None
    authored_ref: SchemaArtifactRef | None = None
    resolved_inputs: tuple[ImmutableInputIdentity, ...] = ()


@dataclass(frozen=True)
class _CompilerRegistration:
    schema_id: str
    compiler_id: str
    compiler_version: str
    compiler: AssemblyCompiler
    identity_adapter: ExecutionIdentityAdapter


class AssemblyCompilerRegistry:
    """Exact, ambiguity-free compiler and identity-adapter registry."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], _CompilerRegistration] = {}

    def register(
        self,
        *,
        schema_id: str,
        compiler_id: str,
        compiler_version: str,
        compiler: AssemblyCompiler,
        identity_adapter: ExecutionIdentityAdapter,
    ) -> None:
        key = (schema_id, compiler_id, compiler_version)
        if key in self._entries:
            raise ValueError(f"assembly compiler already registered: {key!r}")
        self._entries[key] = _CompilerRegistration(
            schema_id, compiler_id, compiler_version, compiler, identity_adapter
        )

    def resolve(self, request: RunAssemblyRequest) -> _CompilerRegistration:
        key = (
            request.authored.schema_id,
            request.compiler.compiler_id,
            request.compiler.compiler_version,
        )
        try:
            return self._entries[key]
        except KeyError as exc:
            known = ", ".join(repr(item) for item in sorted(self._entries)) or "<none>"
            raise ValueError(f"no assembly compiler registered for {key!r}; known: {known}") from exc


def build_default_assembly_registry() -> AssemblyCompilerRegistry:
    """Return Feedbax's built-in training-matrix and Studio compiler registry."""
    from feedbax.contracts.studio_training import register_studio_training_compiler
    from feedbax.training.spec_storage import register_training_run_matrix_compiler

    registry = AssemblyCompilerRegistry()
    register_training_run_matrix_compiler(registry)
    register_studio_training_compiler(registry)
    return registry


class CompiledRowStorageResult(StrictModel):
    """Custody-backed row and its complete execution-identity envelope."""

    row_id: str
    execution: ExecutionIdentityEnvelope
    launch: RowLaunchSpec


def load_schema_artifact(
    ref: SchemaArtifactRef,
    *,
    context: AssemblyContext,
) -> "SpecMigrationResult":
    """Dereference, digest-check, migrate, and validate a governed JSON artifact."""
    if context.artifact_resolver is not None:
        data = context.artifact_resolver(ref)
    elif ref.uri is not None:
        data = Path(ref.uri).read_bytes()
    else:
        raise ValueError(f"artifact {ref.artifact_id!r} has no resolver or materialization URI")
    actual = hashlib.sha256(data).hexdigest()
    if actual != ref.sha256:
        raise ValueError(
            f"artifact byte digest mismatch for {ref.artifact_id!r}: expected={ref.sha256} actual={actual}"
        )
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("registered structured artifact must contain a JSON object")
    if payload.get("schema_id") != ref.schema_id or payload.get("schema_version") != ref.schema_version:
        raise ValueError("artifact schema identity/version does not match its SchemaArtifactRef")
    registry = context.schema_registry
    if registry is None:
        from feedbax.contracts.migrations import default_spec_registry

        registry = default_spec_registry
    family = next(
        (item for item in registry.families() if item.identity == ref.schema_id),
        None,
    )
    if family is None:
        raise ValueError(f"unknown registered artifact schema_id: {ref.schema_id!r}")
    return registry.migrate(family.kind, payload)


def persist_compiled_row(
    row: CompiledExecutionRow,
    *,
    authored: Mapping[str, Any],
    identity_adapter: ExecutionIdentityAdapter,
    context: AssemblyContext,
) -> CompiledRowStorageResult:
    """Persist one pure compiled row and bind its independently verified identities."""
    if context.authored_ref is None:
        raise ValueError("AssemblyContext.authored_ref is required while persisting rows")
    canonical_inputs = canonicalize_immutable_input_identities(row.immutable_inputs)
    snapshot = build_resolved_semantics_snapshot(row.resolved_semantics)
    decode_resolved_snapshot(snapshot)
    intent_hash = identity_adapter.intent_hash(authored)
    execution_hash = training_run_execution_hash(snapshot["root_hash"], canonical_inputs)
    identities = RowIdentities(
        intent_hash=intent_hash,
        resolved_root_hash=snapshot["root_hash"],
        immutable_inputs=canonical_inputs,
        execution_hash=execution_hash,
    )
    capsule = dict(
        identity_adapter.build_capsule(row, identities=identities, context=context)
    )
    if not isinstance(capsule.get("schema_id"), str) or not isinstance(
        capsule.get("schema_version"), str
    ):
        raise ValueError("identity adapter capsule requires schema_id and schema_version")
    observed = identity_adapter.capsule_identities(capsule)
    if observed != identities:
        raise ValueError(
            "execution capsule identity does not bind authored intent, resolved root, "
            "immutable inputs, and execution hash"
        )

    payload_artifact = store_canonical_json_artifact(
        row.payload,
        root=context.custody_root,
        role="compiled_execution_payload",
        logical_name=f"{row.row_id}.payload.json",
    )
    snapshot_artifact = store_canonical_json_artifact(
        snapshot,
        root=context.custody_root,
        role="resolved_semantics_snapshot",
        logical_name=f"{row.row_id}.resolved.json",
    )
    capsule_artifact = store_canonical_json_artifact(
        capsule,
        root=context.custody_root,
        role="execution_identity_capsule",
        logical_name=f"{row.row_id}.execution.json",
    )
    authored_ref = context.authored_ref
    execution = ExecutionIdentityEnvelope(
        schema_id=EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID,
        schema_version=EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION,
        payload=_schema_ref(row.payload, payload_artifact),
        authored_intent=AuthoredIntentRef(**authored_ref.model_dump(), intent_hash=intent_hash),
        resolved_snapshot=ResolvedSnapshotRef(
            **_schema_ref(snapshot, snapshot_artifact).model_dump(), root_hash=snapshot["root_hash"]
        ),
        execution_capsule=ExecutionCapsuleRef(
            **_schema_ref(capsule, capsule_artifact).model_dump(), execution_hash=execution_hash
        ),
        immutable_inputs=canonical_inputs,
        row_provenance=row.provenance,
    )
    return CompiledRowStorageResult(
        row_id=row.row_id,
        execution=execution,
        launch=row.launch,
    )


def assemble_run_bundle(
    request: RunAssemblyRequest,
    *,
    run_set_id: str,
    context: AssemblyContext,
    registry: AssemblyCompilerRegistry,
) -> RunBundle:
    """Compile a verified authored request and persist a current RunBundle."""
    authored_result = load_schema_artifact(request.authored, context=context)
    authored = authored_result.payload
    registration = registry.resolve(request)
    resolved_inputs = [
        _resolve_input(item, context=context) for item in request.inputs
    ]
    compiler_context = replace(context, resolved_inputs=tuple(resolved_inputs))
    compiled = registration.compiler.compile(
        authored=authored,
        run_set_id=run_set_id,
        context=compiler_context,
    )
    if resolved_inputs:
        declared = canonicalize_immutable_input_identities(resolved_inputs)
        for row in compiled.rows:
            row_inputs = canonicalize_immutable_input_identities(row.immutable_inputs)
            if row_inputs != declared:
                raise ValueError(
                    f"compiled row {row.row_id!r} immutable inputs do not match resolved request inputs"
                )
    row_context = replace(context, authored_ref=request.authored)
    stored = [
        persist_compiled_row(
            row,
            authored=authored,
            identity_adapter=registration.identity_adapter,
            context=row_context,
        )
        for row in compiled.rows
    ]
    return RunBundle(
        run_set_id=run_set_id,
        deployment_policy=request.deployment_policy,
        migration_evidence=authored_result.migration_records,
        rows=[
            RunRowSpec(
                row_id=item.row_id,
                execution=item.execution,
                launch=item.launch,
            )
            for item in stored
        ],
        environment=request.environment,
        launch_policy=request.launch_policy,
        budget=request.budget,
        orchestration_root=request.orchestration_root,
        keep_alive=request.keep_alive,
        deadman_enabled=request.deadman_enabled,
        deadman_silence_seconds=request.deadman_silence_seconds,
        metadata=request.metadata,
    )


def _resolve_input(
    declaration: AssemblyInputDeclaration,
    *,
    context: AssemblyContext,
) -> ImmutableInputIdentity:
    if context.input_resolver is None:
        raise ValueError(
            f"input declaration {declaration.role!r} requires AssemblyContext.input_resolver"
        )
    return context.input_resolver(declaration)


def _schema_ref(payload: Mapping[str, Any], artifact: Any) -> SchemaArtifactRef:
    return SchemaArtifactRef(
        schema_id=str(payload["schema_id"]),
        schema_version=str(payload["schema_version"]),
        artifact_id=artifact.artifact_id,
        sha256=artifact.sha256,
        uri=artifact.uri,
    )


def persist_assembly_request(request: RunAssemblyRequest, path: Path) -> str:
    """Atomically persist a canonical assembly request and return its SHA-256."""
    data = training_spec_canonical_bytes(request)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return hashlib.sha256(data).hexdigest()
