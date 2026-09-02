"""Typed authored-request to executable RunBundle assembly."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Callable, Mapping, Protocol

from pydantic import Field, JsonValue, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.evaluation_lifecycle import EvaluationMatrixBatchPlan
from feedbax.contracts.evaluation_preflight import (
    EvaluationOutputPreflightEvidence,
    EvaluationOutputPreflightPolicy,
)
from feedbax.contracts.run_composition import (
    AuthoredIntentParent,
    ResolvedOutputParent,
    authored_envelope_hash,
    parse_composition_node,
)
from feedbax.contracts.run_matrix import TrainingRowParentProvenance, TrainingRowProvenance
from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot
from feedbax.contracts.spec_storage import (
    build_resolved_semantics_snapshot,
    canonicalize_immutable_input_identities,
    store_canonical_json_artifact,
    training_run_execution_hash,
    training_spec_canonical_bytes,
    training_spec_sha256,
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
    ExecutionFamily,
    ImmutableInputIdentity,
    LaunchPolicy,
    ResolvedSnapshotRef,
    ResolvedAssemblyInput,
    RowLaunchSpec,
    RunBundle,
    RunRowSpec,
    SchemaArtifactRef,
    default_orchestration_root,
)
from feedbax.orchestration.revision import FeedbaxRevisionError, check_feedbax_provenance
from feedbax.orchestration.staged_root_custody import StagedRootCustody
from feedbax.training.row_lowering import (
    GovernedTrainingRowParent,
    TrainingRowLoweringContext,
)

if TYPE_CHECKING:
    from feedbax.contracts.migrations import SpecMigrationResult, SpecSchemaRegistry


RUN_ASSEMBLY_REQUEST_SCHEMA_ID = "feedbax.spec.run_assembly_request"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V1 = f"{RUN_ASSEMBLY_REQUEST_SCHEMA_ID}.v1"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V2 = f"{RUN_ASSEMBLY_REQUEST_SCHEMA_ID}.v2"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V3 = f"{RUN_ASSEMBLY_REQUEST_SCHEMA_ID}.v3"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V4 = f"{RUN_ASSEMBLY_REQUEST_SCHEMA_ID}.v4"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V5 = f"{RUN_ASSEMBLY_REQUEST_SCHEMA_ID}.v5"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V6 = f"{RUN_ASSEMBLY_REQUEST_SCHEMA_ID}.v6"
RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION = f"{RUN_ASSEMBLY_REQUEST_SCHEMA_ID}.v7"


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


class GovernedTrainingRowParentDeclaration(StrictModel):
    """One content-pinned composition parent available during row lowering."""

    role: str = Field(min_length=1)
    parent: Annotated[
        AuthoredIntentParent | ResolvedOutputParent,
        Field(discriminator="kind"),
    ]
    artifact: SchemaArtifactRef


class RunAssemblyRequest(StrictModel):
    """Durable authored input consumed by the persisted ASSEMBLE stage.

    ``feedbax_revision`` is the authored Feedbax revision authority. It is the
    commit the author intends this request to be assembled and executed against,
    and it is checked against the provenance of the package that actually
    supplied ``import feedbax`` before anything is compiled or written. The
    value is then copied verbatim into ``RunBundle.feedbax_revision`` rather
    than minted from the imported package, so a stale editable install can no
    longer produce a self-consistent bundle that passes its own preflight.
    """

    schema_id: str = RUN_ASSEMBLY_REQUEST_SCHEMA_ID
    schema_version: str = RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION
    authored: SchemaArtifactRef
    compiler: CompilerIdentity
    feedbax_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    inputs: list[AssemblyInputDeclaration] = Field(default_factory=list)
    training_row_parents: list[GovernedTrainingRowParentDeclaration] = Field(default_factory=list)
    staged_roots: list[StagedRootCustody] = Field(default_factory=list)
    evaluation_batch_plan: EvaluationMatrixBatchPlan | None = None
    evaluation_output_preflight: EvaluationOutputPreflightPolicy | None = None
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
        keys = [(item.parent.kind, item.parent.ref) for item in self.training_row_parents]
        if len(keys) != len(set(keys)):
            raise ValueError("training_row_parents contain ambiguous kind/ref declarations")
        staged_root_keys = [(item.root_kind, item.binding_name) for item in self.staged_roots]
        if staged_root_keys != sorted(staged_root_keys):
            raise ValueError("staged_roots must be in canonical (root_kind, binding_name) order")
        if len(staged_root_keys) != len(set(staged_root_keys)):
            raise ValueError("staged_roots contain duplicate typed binding names")
        return self


class CompiledExecutionRow(StrictModel):
    """Pure compiler output before ASSEMBLE writes custody artifacts."""

    row_id: str = Field(min_length=1)
    execution_family: ExecutionFamily = "native-training"
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
InputResolver = Callable[[AssemblyInputDeclaration], ResolvedAssemblyInput]


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
    staged_roots: tuple[StagedRootCustody, ...] = ()
    evaluation_batch_plan: EvaluationMatrixBatchPlan | None = None
    training_row_lowering_context: Any | None = None


@dataclass(frozen=True)
class _CompilerRegistration:
    schema_id: str
    compiler_id: str
    compiler_version: str
    compiler: AssemblyCompiler
    identity_adapter: ExecutionIdentityAdapter


@dataclass(frozen=True)
class _PreparedRunAssembly:
    """Pure assembly result retained before any compiled row custody writes."""

    request_sha256: str
    run_set_id: str
    authored_result: "SpecMigrationResult"
    authored: dict[str, Any]
    registration: _CompilerRegistration
    resolved_input_records: tuple[ResolvedAssemblyInput, ...]
    compiled: CompiledRunSet
    evaluation_output_preflight: EvaluationOutputPreflightEvidence | None


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
            raise ValueError(
                f"no assembly compiler registered for {key!r}; known: {known}"
            ) from exc


def build_default_assembly_registry(
    *, method_registry: Any, row_lowerer_registry: Any, evaluation_registry: Any
) -> AssemblyCompilerRegistry:
    """Return Feedbax's built-in training-matrix and Studio compiler registry."""
    from feedbax.analysis.evaluation_orchestration import (
        register_evaluation_run_matrix_compiler,
    )
    from feedbax.contracts.studio_training import register_studio_training_compiler
    from feedbax.training.run_matrix import _validate_training_payload
    from feedbax.training.spec_storage import register_training_run_matrix_compiler

    registry = AssemblyCompilerRegistry()
    register_training_run_matrix_compiler(
        registry,
        method_registry=method_registry,
        row_validator=lambda payload, row_id: _validate_training_payload(
            payload, row_id=row_id, method_registry=method_registry
        ),
        row_lowerer=row_lowerer_registry.lower,
    )
    register_evaluation_run_matrix_compiler(registry, evaluation_registry=evaluation_registry)
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
    if (
        payload.get("schema_id") != ref.schema_id
        or payload.get("schema_version") != ref.schema_version
    ):
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
    capsule = dict(identity_adapter.build_capsule(row, identities=identities, context=context))
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


def assert_authored_feedbax_revision(request: RunAssemblyRequest) -> str:
    """Fail closed unless the imported package matches the request's revision authority.

    This is the stale-install gate. ``RunBundle.feedbax_revision`` used to be
    minted from whichever package happened to be imported, so preflight compared
    a stale editable install against itself and passed. The authored request now
    carries the authority, and it is verified here — before any compilation or
    output — against the real provenance of the imported package, including the
    cleanliness of the checkout that supplied it.

    Returns:
        The authored revision, which callers copy into the assembled bundle.

    Raises:
        FeedbaxRevisionError: If the imported package is a different revision, is
            supplied by a dirty checkout, or cannot be verified at all.
    """
    try:
        check_feedbax_provenance(request.feedbax_revision)
    except FeedbaxRevisionError as exc:
        raise FeedbaxRevisionError(
            "the imported Feedbax package does not satisfy the assembly request's "
            f"feedbax_revision authority {request.feedbax_revision}: {exc}; re-author "
            "the request against the intended revision, or install that revision "
            "(a stale editable install is the usual cause)"
        ) from exc
    return request.feedbax_revision


def assemble_run_bundle(
    request: RunAssemblyRequest,
    *,
    run_set_id: str,
    context: AssemblyContext,
    registry: AssemblyCompilerRegistry,
) -> RunBundle:
    """Compile a verified authored request and persist a current RunBundle."""
    prepared = _prepare_run_assembly(
        request,
        run_set_id=run_set_id,
        context=context,
        registry=registry,
    )
    return _persist_prepared_run_bundle(
        request,
        prepared=prepared,
        run_set_id=run_set_id,
        context=context,
    )


def _persist_prepared_run_bundle(
    request: RunAssemblyRequest,
    *,
    prepared: _PreparedRunAssembly,
    run_set_id: str,
    context: AssemblyContext,
) -> RunBundle:
    """Persist rows from one already checked pure assembly result."""
    request_sha256 = hashlib.sha256(training_spec_canonical_bytes(request)).hexdigest()
    if prepared.request_sha256 != request_sha256 or prepared.run_set_id != run_set_id:
        raise ValueError("prepared run assembly does not match the request and run_set_id")
    authored = prepared.authored
    registration = prepared.registration
    compiled = prepared.compiled
    resolved_input_records = list(prepared.resolved_input_records)
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
    execution_family = compiled.rows[0].execution_family
    return RunBundle(
        run_set_id=run_set_id,
        feedbax_revision=request.feedbax_revision,
        deployment_policy=request.deployment_policy,
        execution_family=execution_family,
        migration_evidence=prepared.authored_result.migration_records,
        rows=[
            RunRowSpec(
                row_id=item.row_id,
                execution_family=compiled.rows[index].execution_family,
                execution=item.execution,
                launch=item.launch,
            )
            for index, item in enumerate(stored)
        ],
        environment=request.environment,
        launch_policy=request.launch_policy,
        budget=request.budget,
        resolved_inputs=resolved_input_records,
        staged_roots=request.staged_roots,
        evaluation_output_preflight=prepared.evaluation_output_preflight,
        orchestration_root=request.orchestration_root,
        keep_alive=request.keep_alive,
        deadman_enabled=request.deadman_enabled,
        deadman_silence_seconds=request.deadman_silence_seconds,
        metadata=request.metadata,
    )


def _prepare_run_assembly(
    request: RunAssemblyRequest,
    *,
    run_set_id: str,
    context: AssemblyContext,
    registry: AssemblyCompilerRegistry,
) -> _PreparedRunAssembly:
    """Resolve and compile one request without persisting compiled row artifacts."""
    assert_authored_feedbax_revision(request)
    authored_result = load_schema_artifact(request.authored, context=context)
    authored = dict(authored_result.payload)
    registration = registry.resolve(request)
    resolved_input_records = tuple(
        sorted(
            (_resolve_input(item, context=context) for item in request.inputs),
            key=lambda item: (item.identity.role, item.identity.kind, item.identity.identifier),
        )
    )
    resolved_inputs = [item.identity for item in resolved_input_records]
    lowering_context = _resolve_training_row_parents(
        request.training_row_parents,
        context=context,
    )
    compiler_context = replace(
        context,
        resolved_inputs=tuple(resolved_inputs),
        staged_roots=tuple(request.staged_roots),
        evaluation_batch_plan=request.evaluation_batch_plan,
        training_row_lowering_context=lowering_context,
    )
    compiled = registration.compiler.compile(
        authored=authored,
        run_set_id=run_set_id,
        context=compiler_context,
    )
    evaluation_output_preflight = _preflight_evaluation_output(
        request,
        compiled=compiled,
        run_set_id=run_set_id,
    )
    if resolved_inputs:
        declared = canonicalize_immutable_input_identities(resolved_inputs)
        for row in compiled.rows:
            row_inputs = canonicalize_immutable_input_identities(row.immutable_inputs)
            if row_inputs != declared:
                raise ValueError(
                    f"compiled row {row.row_id!r} immutable inputs do not match resolved request inputs"
                )
    execution_families = {row.execution_family for row in compiled.rows}
    if len(execution_families) != 1:
        raise ValueError("compiled run set must contain exactly one execution family")
    return _PreparedRunAssembly(
        request_sha256=hashlib.sha256(training_spec_canonical_bytes(request)).hexdigest(),
        run_set_id=run_set_id,
        authored_result=authored_result,
        authored=authored,
        registration=registration,
        resolved_input_records=resolved_input_records,
        compiled=compiled,
        evaluation_output_preflight=evaluation_output_preflight,
    )


def _preflight_evaluation_output(
    request: RunAssemblyRequest,
    *,
    compiled: CompiledRunSet,
    run_set_id: str,
) -> EvaluationOutputPreflightEvidence | None:
    """Refuse cardinality or disk-budget drift before compiled rows are persisted."""
    policy = request.evaluation_output_preflight
    execution_families = {row.execution_family for row in compiled.rows}
    if execution_families == {"evaluation-matrix"}:
        if len(compiled.rows) != 1:
            raise ValueError("evaluation output preflight requires one compiled evaluation matrix")
        if policy is None:
            raise ValueError(
                "evaluation-matrix assembly requires authored evaluation_output_preflight "
                "with an explicit storage_mode choice"
            )
    elif policy is None:
        return None
    else:
        raise ValueError(
            "evaluation_output_preflight is only valid for one compiled evaluation matrix"
        )
    ordered_row_ids = compiled.rows[0].resolved_semantics.get("ordered_row_ids")
    if not isinstance(ordered_row_ids, list) or not ordered_row_ids:
        raise ValueError("evaluation output preflight requires canonical resolved ordered_row_ids")
    if any(not isinstance(row_id, str) or not row_id for row_id in ordered_row_ids):
        raise ValueError("evaluation resolved ordered_row_ids must contain non-empty strings")
    resolved_row_count = len(ordered_row_ids)
    if resolved_row_count != policy.expected_resolved_row_count:
        raise ValueError(
            "evaluation resolved row count does not match authored preflight expectation: "
            f"expected={policy.expected_resolved_row_count} resolved={resolved_row_count}"
        )

    output_root = _evaluation_output_root(request, run_set_id=run_set_id)
    filesystem_path = _nearest_existing_path(output_root)
    observed_free_bytes = shutil.disk_usage(filesystem_path).free
    active_batch_count = 0
    max_rows_per_active_batch = 0
    if policy.storage_mode == "batch_reclamation":
        if request.evaluation_batch_plan is None:
            raise ValueError(
                "evaluation_batch_plan is required when "
                "evaluation_output_preflight.storage_mode='batch_reclamation'"
            )
        plan = EvaluationMatrixBatchPlan.model_validate(
            compiled.rows[0].launch.metadata.get("batch_plan")
        )
        if not plan.consumers:
            raise ValueError(
                "batch-reclamation output preflight requires declared terminal consumers"
            )
        active_batch_count = min(request.launch_policy.max_parallel_rows, len(plan.batches))
        max_rows_per_active_batch = max(len(batch.ordered_row_ids) for batch in plan.batches)
        raw_row_capacity = active_batch_count * max_rows_per_active_batch
    else:
        raw_row_capacity = resolved_row_count
    estimated_retained_bytes = (
        raw_row_capacity * policy.retained_bytes_per_resolved_row * policy.planned_repetitions
        + policy.estimated_compact_retained_bytes
    )
    required_free_bytes = estimated_retained_bytes + policy.required_free_space_reserve_bytes
    if observed_free_bytes < required_free_bytes:
        raise ValueError(
            "evaluation output preflight requires more free space than observed: "
            f"required_free_bytes={required_free_bytes} "
            f"observed_free_bytes={observed_free_bytes} "
            f"output_root={output_root}"
        )
    evidence = EvaluationOutputPreflightEvidence(
        expected_resolved_row_count=policy.expected_resolved_row_count,
        resolved_row_count=resolved_row_count,
        retained_bytes_per_resolved_row=policy.retained_bytes_per_resolved_row,
        retained_bytes_per_resolved_row_source=(policy.retained_bytes_per_resolved_row_source),
        planned_repetitions=policy.planned_repetitions,
        storage_mode=policy.storage_mode,
        active_batch_count=active_batch_count,
        max_rows_per_active_batch=max_rows_per_active_batch,
        estimated_compact_retained_bytes=policy.estimated_compact_retained_bytes,
        estimated_retained_bytes=estimated_retained_bytes,
        required_free_space_reserve_bytes=policy.required_free_space_reserve_bytes,
        required_free_bytes=required_free_bytes,
        observed_free_bytes=observed_free_bytes,
        output_root=str(output_root),
        observed_filesystem_path=str(filesystem_path),
        observed_filesystem_device=os.stat(filesystem_path).st_dev,
    )
    return evidence


def _evaluation_output_root(request: RunAssemblyRequest, *, run_set_id: str) -> Path:
    if request.orchestration_root:
        root = Path(request.orchestration_root).expanduser()
        return root if root.name == run_set_id else root / run_set_id
    return default_orchestration_root(run_set_id)


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise ValueError(f"no existing filesystem ancestor for evaluation output: {path}")
        candidate = parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate.resolve()


def _resolve_training_row_parents(
    declarations: list[GovernedTrainingRowParentDeclaration],
    *,
    context: AssemblyContext,
) -> Any:
    parents: list[GovernedTrainingRowParent] = []
    for declaration in declarations:
        payload, artifact_sha256 = _load_training_row_parent_artifact(
            declaration.artifact,
            context=context,
        )
        parent = declaration.parent
        if isinstance(parent, AuthoredIntentParent):
            observed_hash = authored_envelope_hash(parse_composition_node(payload))
            semantic_hash = parent.content_hash
        else:
            observed_hash = training_spec_sha256(payload)
            semantic_hash = parent.resolved_root_hash
        if observed_hash != semantic_hash:
            raise ValueError(
                f"training-row parent {parent.kind}:{parent.ref!r} semantic hash drifted"
            )
        parents.append(
            GovernedTrainingRowParent(
                provenance=TrainingRowParentProvenance(
                    role=declaration.role,
                    parent_kind=parent.kind,
                    ref=parent.ref,
                    semantic_hash=semantic_hash,
                    artifact_id=declaration.artifact.artifact_id,
                    artifact_sha256=artifact_sha256,
                    schema_id=declaration.artifact.schema_id,
                    schema_version=declaration.artifact.schema_version,
                ),
                parent=parent,
                payload=payload,
            )
        )
    return TrainingRowLoweringContext(tuple(parents))


def _load_training_row_parent_artifact(
    ref: SchemaArtifactRef,
    *,
    context: AssemblyContext,
) -> tuple[dict[str, Any], str]:
    if context.artifact_resolver is not None:
        data = context.artifact_resolver(ref)
    elif ref.uri is not None:
        data = Path(ref.uri).read_bytes()
    else:
        raise ValueError(f"training-row parent artifact {ref.artifact_id!r} has no resolver or URI")
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != ref.sha256:
        raise ValueError(
            f"artifact byte digest mismatch for {ref.artifact_id!r}: "
            f"expected={ref.sha256} actual={actual_sha256}"
        )
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"training-row parent artifact {ref.artifact_id!r} is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("training-row parent artifact must contain a JSON object")
    if (payload.get("schema_id"), payload.get("schema_version")) != (
        ref.schema_id,
        ref.schema_version,
    ):
        raise ValueError("training-row parent schema identity does not match its declaration")
    return payload, actual_sha256


def _resolve_input(
    declaration: AssemblyInputDeclaration,
    *,
    context: AssemblyContext,
) -> ResolvedAssemblyInput:
    if context.input_resolver is None:
        resolved = ResolvedAssemblyInput.model_validate_json(
            Path(declaration.locator).read_text(encoding="utf-8")
        )
    else:
        resolved = context.input_resolver(declaration)
    identity = resolved.identity
    if identity.role != declaration.role:
        raise ValueError("resolved input role does not match its declaration")
    if identity.kind != declaration.kind:
        raise ValueError("resolved input kind does not match its declaration")
    if (
        identity.schema_id != declaration.schema_id
        or identity.schema_version != declaration.schema_version
    ):
        raise ValueError("resolved input schema identity does not match its declaration")
    return resolved


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
