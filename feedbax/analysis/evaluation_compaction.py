"""Registered batch-scoped evaluation compaction and reclamation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from feedbax.analysis.execution_context import StagedExecutionContext
from feedbax.contracts.evaluation_lifecycle import (
    EvaluationBatchConsumerDeclaration,
    EvaluationBatchLeafAcknowledgement,
    EvaluationBatchMergeCheckpoint,
    EvaluationBatchReclamationEvidence,
    EvaluationLifecycleRowOutcome,
    EvaluationMatrixBatchUnit,
)
from feedbax.contracts.manifest import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    AnalysisRunSpec,
    ArtifactRef,
    ParentRef,
    canonical_json_bytes,
    evaluation_states_cache_path,
    sha256_bytes,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


@dataclass(frozen=True)
class EvaluationBatchConsumerInput:
    """Authenticated inputs supplied to one registered compact leaf."""

    matrix_intent_hash: str
    batch: EvaluationMatrixBatchUnit
    outcomes: tuple[EvaluationLifecycleRowOutcome, ...]
    manifests: tuple[Mapping[str, Any], ...]
    states: tuple[Any, ...]
    parent_authorities: tuple[ParentRef, ...]
    parameters: Mapping[str, JsonValue]
    execution_context: StagedExecutionContext


@dataclass(frozen=True)
class EvaluationBatchFragment:
    """Opaque, schema-identified compact fragment returned by a consumer."""

    payload: Any
    schema_id: str
    schema_version: str
    role: str


@dataclass(frozen=True)
class EvaluationBatchMergeInput:
    """One verified fragment plus prior merge state in authored batch order."""

    matrix_intent_hash: str
    batch: EvaluationMatrixBatchUnit
    fragment: Any
    prior_merge_state: Any | None
    parameters: Mapping[str, JsonValue]
    execution_context: StagedExecutionContext


@dataclass(frozen=True)
class EvaluationBatchMergeState:
    """Opaque next merge state returned by a registered consumer."""

    payload: Any
    schema_id: str
    schema_version: str


@dataclass(frozen=True)
class EvaluationBatchFinalizeInput:
    """Verified terminal merge state supplied for product publication."""

    matrix_intent_hash: str
    terminal_merge_state: Any
    parameters: Mapping[str, JsonValue]
    execution_context: StagedExecutionContext


@dataclass(frozen=True)
class EvaluationBatchConsumer:
    """Registered consumer callbacks for concurrent compaction and ordered merge."""

    compact: Callable[[EvaluationBatchConsumerInput], EvaluationBatchFragment]
    merge: Callable[[EvaluationBatchMergeInput], EvaluationBatchMergeState]
    finalize: Callable[[EvaluationBatchFinalizeInput], EvaluationBatchFragment]


_CONSUMERS: dict[tuple[str, str], EvaluationBatchConsumer] = {}


def register_evaluation_batch_consumer(
    consumer_id: str,
    consumer_version: str,
    *,
    compact: Callable[[EvaluationBatchConsumerInput], EvaluationBatchFragment],
    merge: Callable[[EvaluationBatchMergeInput], EvaluationBatchMergeState],
    finalize: Callable[[EvaluationBatchFinalizeInput], EvaluationBatchFragment],
    replace: bool = False,
) -> None:
    """Register a downstream compact leaf through the public plugin surface."""
    key = (consumer_id, consumer_version)
    if not consumer_id or not consumer_version:
        raise ValueError("evaluation batch consumer identity and version must be non-empty")
    if key in _CONSUMERS and not replace:
        raise ValueError(f"evaluation batch consumer {key!r} is already registered")
    _CONSUMERS[key] = EvaluationBatchConsumer(
        compact=compact,
        merge=merge,
        finalize=finalize,
    )


def compact_evaluation_batch(
    declaration: EvaluationBatchConsumerDeclaration,
    consumer_input: EvaluationBatchConsumerInput,
    *,
    custody_root: Path,
) -> ArtifactRef:
    """Run one compact callback and verify its content-addressed fragment."""
    _validate_callback_input(declaration, consumer_input)
    observed_schemas = {
        schema for outcome in consumer_input.outcomes for schema in outcome.diagnostic_schema_ids
    }
    unexpected = observed_schemas - set(declaration.accepted_evaluation_state_schema_ids)
    if unexpected or any(not outcome.diagnostic_schema_ids for outcome in consumer_input.outcomes):
        raise ValueError(f"consumer leaf {declaration.leaf_id!r} rejected evaluation-state schemas")
    consumer = _resolve_consumer(declaration)
    fragment = consumer.compact(consumer_input)
    if (
        fragment.schema_id != declaration.compact_product_schema_id
        or fragment.schema_version != declaration.compact_product_schema_version
        or fragment.role != declaration.compact_product_role
    ):
        raise ValueError(f"consumer leaf {declaration.leaf_id!r} returned wrong fragment contract")
    provider = ImmutableArtifactBlobProvider(custody_root)
    ref = provider.store_bytes(
        canonical_json_bytes(fragment.payload),
        role=fragment.role,
        logical_name=f"{declaration.leaf_id}-{consumer_input.batch.batch_id}.json",
        media_type="application/json",
        metadata={
            "schema_id": fragment.schema_id,
            "schema_version": fragment.schema_version,
            "batch_id": consumer_input.batch.batch_id,
            "leaf_id": declaration.leaf_id,
            "consumer_parameters": declaration.parameters,
            "consumer_parameters_sha256": _parameters_sha256(declaration),
        },
    )
    provider.get_bytes(ref)
    return ref


def merge_evaluation_batch_fragment(
    declaration: EvaluationBatchConsumerDeclaration,
    *,
    matrix_intent_hash: str,
    batch: EvaluationMatrixBatchUnit,
    parent_authorities: Sequence[ParentRef],
    fragment: ArtifactRef,
    prior_merge_state: ArtifactRef | None,
    custody_root: Path,
    execution_context: StagedExecutionContext,
) -> EvaluationBatchLeafAcknowledgement:
    """Verify and apply one fragment exactly once in authored batch order."""
    _require_execution_context(execution_context)
    provider = ImmutableArtifactBlobProvider(custody_root)
    if (
        fragment.role != declaration.compact_product_role
        or fragment.metadata.get("schema_id") != declaration.compact_product_schema_id
        or fragment.metadata.get("schema_version") != declaration.compact_product_schema_version
        or fragment.metadata.get("leaf_id") != declaration.leaf_id
        or fragment.metadata.get("batch_id") != batch.batch_id
        or fragment.metadata.get("consumer_parameters") != declaration.parameters
        or fragment.metadata.get("consumer_parameters_sha256") != _parameters_sha256(declaration)
    ):
        raise ValueError(f"consumer leaf {declaration.leaf_id!r} fragment identity drifted")
    fragment_payload = json.loads(provider.get_bytes(fragment))
    checkpoint = custody_root / "merge-checkpoints" / declaration.leaf_id / f"{batch.batch_id}.json"
    expected_prior = prior_merge_state.sha256 if prior_merge_state is not None else None
    if checkpoint.is_file():
        from feedbax.contracts.migrations import migrate_structured_spec_payload

        persisted = EvaluationBatchMergeCheckpoint.model_validate(
            migrate_structured_spec_payload(
                "EvaluationBatchMergeCheckpoint",
                json.loads(checkpoint.read_text(encoding="utf-8")),
                path=str(checkpoint),
            ).payload
        )
        expected_identity = {
            "matrix_intent_hash": matrix_intent_hash,
            "batch": batch.model_dump(mode="json"),
            "declaration": declaration.model_dump(mode="json"),
            "parent_authorities": [item.model_dump(mode="json") for item in parent_authorities],
        }
        persisted_identity = {
            "matrix_intent_hash": persisted.matrix_intent_hash,
            "batch": persisted.batch.model_dump(mode="json"),
            "declaration": persisted.declaration.model_dump(mode="json"),
            "parent_authorities": [
                item.model_dump(mode="json") for item in persisted.parent_authorities
            ],
        }
        if canonical_json_bytes(persisted_identity) != canonical_json_bytes(expected_identity):
            raise ValueError(
                f"consumer leaf {declaration.leaf_id!r} merge checkpoint identity drifted"
            )
        acknowledgement = persisted.acknowledgement
        if (
            acknowledgement.fragment.sha256 != fragment.sha256
            or acknowledgement.prior_merge_state_sha256 != expected_prior
            or acknowledgement.leaf_id != declaration.leaf_id
            or acknowledgement.consumer_id != declaration.consumer_id
            or acknowledgement.consumer_version != declaration.consumer_version
            or canonical_json_bytes(acknowledgement.parameters)
            != canonical_json_bytes(declaration.parameters)
            or acknowledgement.compact_product_role != declaration.compact_product_role
        ):
            raise ValueError(
                f"consumer leaf {declaration.leaf_id!r} merge checkpoint identity drifted"
            )
        _validate_merge_state_ref(
            declaration,
            acknowledgement.merge_state,
            matrix_intent_hash=matrix_intent_hash,
        )
        provider.get_bytes(acknowledgement.merge_state)
        return acknowledgement.model_copy(update={"reused_verified_fragment": True})
    if prior_merge_state is not None:
        _validate_merge_state_ref(
            declaration,
            prior_merge_state,
            matrix_intent_hash=matrix_intent_hash,
        )
    prior_payload = (
        json.loads(provider.get_bytes(prior_merge_state)) if prior_merge_state is not None else None
    )
    consumer = _resolve_consumer(declaration)
    next_state = consumer.merge(
        EvaluationBatchMergeInput(
            matrix_intent_hash=matrix_intent_hash,
            batch=batch,
            fragment=fragment_payload,
            prior_merge_state=prior_payload,
            parameters=_callback_parameters(declaration),
            execution_context=_require_execution_context(execution_context),
        )
    )
    if (
        next_state.schema_id != declaration.merge_state_schema_id
        or next_state.schema_version != declaration.merge_state_schema_version
    ):
        raise ValueError(
            f"consumer leaf {declaration.leaf_id!r} returned wrong merge-state contract"
        )
    state_ref = provider.store_bytes(
        canonical_json_bytes(next_state.payload),
        role=f"{declaration.compact_product_role}_merge_state",
        logical_name=f"{declaration.leaf_id}-{batch.batch_id}-merge-state.json",
        media_type="application/json",
        metadata={
            "schema_id": next_state.schema_id,
            "schema_version": next_state.schema_version,
            "batch_id": batch.batch_id,
            "leaf_id": declaration.leaf_id,
            "matrix_intent_hash": matrix_intent_hash,
            "consumer_parameters": declaration.parameters,
            "consumer_parameters_sha256": _parameters_sha256(declaration),
            "prior_merge_state_sha256": (
                prior_merge_state.sha256 if prior_merge_state is not None else None
            ),
        },
    )
    provider.get_bytes(state_ref)
    acknowledgement = EvaluationBatchLeafAcknowledgement(
        leaf_id=declaration.leaf_id,
        consumer_id=declaration.consumer_id,
        consumer_version=declaration.consumer_version,
        parameters=declaration.parameters,
        compact_product_role=declaration.compact_product_role,
        fragment=fragment,
        prior_merge_state_sha256=(
            prior_merge_state.sha256 if prior_merge_state is not None else None
        ),
        merge_state=state_ref,
    )
    persisted = EvaluationBatchMergeCheckpoint(
        matrix_intent_hash=matrix_intent_hash,
        batch=batch,
        declaration=declaration,
        parent_authorities=tuple(parent_authorities),
        acknowledgement=acknowledgement,
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(checkpoint, persisted.model_dump(mode="json"))
    return acknowledgement


def reclaim_evaluation_batch_caches(
    batch: EvaluationMatrixBatchUnit,
    *,
    batch_index: int,
    outcomes: Sequence[EvaluationLifecycleRowOutcome],
    acknowledgements: Sequence[EvaluationBatchLeafAcknowledgement],
    required_declarations: Sequence[EvaluationBatchConsumerDeclaration],
    custody_root: Path,
    execution_context: StagedExecutionContext,
) -> EvaluationBatchReclamationEvidence:
    """Remove only authenticated row state caches after every leaf acknowledged."""
    _require_execution_context(execution_context)
    manifest_ids = tuple(outcome.manifest_id for outcome in outcomes)
    if tuple(outcome.row_id for outcome in outcomes) != batch.ordered_row_ids:
        raise ValueError("reclamation outcomes drifted from the authored batch")
    declarations_by_leaf = {item.leaf_id: item for item in required_declarations}
    if len(declarations_by_leaf) != len(required_declarations):
        raise ValueError("raw cache reclamation requires unique consumer declarations")
    observed_leaf_ids = [item.leaf_id for item in acknowledgements]
    if (
        len(observed_leaf_ids) != len(set(observed_leaf_ids))
        or set(observed_leaf_ids) != set(declarations_by_leaf)
        or any(
            acknowledgement.consumer_id != declarations_by_leaf[acknowledgement.leaf_id].consumer_id
            or acknowledgement.consumer_version
            != declarations_by_leaf[acknowledgement.leaf_id].consumer_version
            or canonical_json_bytes(acknowledgement.parameters)
            != canonical_json_bytes(declarations_by_leaf[acknowledgement.leaf_id].parameters)
            or acknowledgement.compact_product_role
            != declarations_by_leaf[acknowledgement.leaf_id].compact_product_role
            for acknowledgement in acknowledgements
        )
    ):
        raise ValueError("raw cache reclamation requires every declared leaf acknowledgement")
    checkpoint = custody_root / "reclamation-checkpoints" / f"{batch.batch_id}.json"
    if checkpoint.is_file():
        evidence = EvaluationBatchReclamationEvidence.model_validate(
            _normalize_legacy_reclamation_checkpoint(
                json.loads(checkpoint.read_text(encoding="utf-8"))
            )
        )
        if (
            evidence.batch_index != batch_index
            or evidence.ordered_row_ids != batch.ordered_row_ids
            or canonical_json_bytes(
                [item.model_dump(mode="json") for item in evidence.leaf_acknowledgements]
            )
            != canonical_json_bytes([item.model_dump(mode="json") for item in acknowledgements])
        ):
            raise ValueError("evaluation batch reclamation checkpoint identity drifted")
        return evidence
    intent_path = custody_root / "reclamation-intents" / f"{batch.batch_id}.json"
    acknowledgement_hashes = [item.merge_state.sha256 for item in acknowledgements]
    if intent_path.is_file():
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        if (
            intent.get("batch") != batch.model_dump(mode="json", exclude_none=True)
            or intent.get("batch_index") != batch_index
            or canonical_json_bytes(intent.get("consumer_declarations"))
            != canonical_json_bytes(
                [item.model_dump(mode="json") for item in required_declarations]
            )
            or intent.get("acknowledgement_hashes") != acknowledgement_hashes
            or [entry.get("manifest_id") for entry in intent.get("entries", [])]
            != list(manifest_ids)
        ):
            raise ValueError("evaluation batch reclamation intent identity drifted")
    else:
        entries = []
        for outcome in outcomes:
            manifest = json.loads(Path(outcome.manifest_path).read_text(encoding="utf-8"))
            cache = manifest.get("metadata", {}).get("cache", {})
            states_path = cache.get("states_path")
            if not isinstance(states_path, str) or not states_path:
                raise ValueError(
                    f"evaluation row {outcome.row_id!r} lacks a declared raw state cache"
                )
            path = Path(states_path)
            if not path.is_file():
                raise ValueError(
                    f"evaluation row {outcome.row_id!r} raw state cache is unmaterialized"
                )
            if (
                not path.is_absolute()
                or path.resolve(strict=True) != path
                or path
                != evaluation_states_cache_path(
                    outcome.manifest_id,
                    root=path.parents[2],
                )
            ):
                raise ValueError(
                    f"evaluation row {outcome.row_id!r} raw state cache locator is not canonical"
                )
            data = path.read_bytes()
            entries.append(
                {
                    "manifest_id": outcome.manifest_id,
                    "path": str(path),
                    "size_bytes": len(data),
                    "sha256": sha256_bytes(data),
                    "status": "pending",
                }
            )
        intent = {
            "batch": batch.model_dump(mode="json", exclude_none=True),
            "batch_index": batch_index,
            "consumer_declarations": [
                item.model_dump(mode="json") for item in required_declarations
            ],
            "acknowledgement_hashes": acknowledgement_hashes,
            "entries": entries,
        }
        _atomic_write_json(intent_path, intent)
    for entry in intent["entries"]:
        path = Path(entry["path"])
        if entry["status"] == "completed":
            if path.exists():
                raise ValueError("reclaimed evaluation raw cache unexpectedly reappeared")
            continue
        if entry["status"] == "pending":
            if not path.is_file():
                raise ValueError("pending evaluation raw cache is unmaterialized")
            data = path.read_bytes()
            if len(data) != entry["size_bytes"] or sha256_bytes(data) != entry["sha256"]:
                raise ValueError("pending evaluation raw cache identity drifted")
            entry["status"] = "deleting"
            _atomic_write_json(intent_path, intent)
        if entry["status"] == "deleting":
            if path.exists():
                data = path.read_bytes()
                if len(data) != entry["size_bytes"] or sha256_bytes(data) != entry["sha256"]:
                    raise ValueError("deleting evaluation raw cache identity drifted")
                path.unlink()
            entry["status"] = "completed"
            _atomic_write_json(intent_path, intent)
    removed_bytes = sum(entry["size_bytes"] for entry in intent["entries"])
    evidence = EvaluationBatchReclamationEvidence(
        batch_id=batch.batch_id,
        batch_index=batch_index,
        ordered_row_ids=batch.ordered_row_ids,
        leaf_acknowledgements=tuple(acknowledgements),
        removed_cache_manifest_ids=manifest_ids,
        removed_cache_bytes=removed_bytes,
    )
    _atomic_write_json(checkpoint, evidence.model_dump(mode="json"))
    return evidence


def publish_evaluation_compaction_products(
    declarations: Sequence[EvaluationBatchConsumerDeclaration],
    terminal_states: Mapping[str, ArtifactRef],
    outcomes: Sequence[EvaluationLifecycleRowOutcome],
    *,
    custody_root: Path,
    execution_context: StagedExecutionContext,
) -> tuple[ArtifactRef, ...]:
    """Publish terminal compact leaves through ``AnalysisRunManifest`` custody."""
    _require_execution_context(execution_context)
    from feedbax.analysis.context import AnalysisRunContext

    parents = []
    for outcome in outcomes:
        data = Path(outcome.manifest_path).read_bytes()
        parents.append(
            ParentRef(
                kind="EvaluationRunManifest",
                id=outcome.manifest_id,
                role="evaluation_run",
                metadata={
                    "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
                    "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
                    "manifest_sha256": sha256_bytes(data),
                    "size_bytes": len(data),
                },
            )
        )
    provider = ImmutableArtifactBlobProvider(custody_root)
    finalized = []
    for declaration in declarations:
        try:
            state_ref = terminal_states[declaration.leaf_id]
        except KeyError as exc:
            raise ValueError(
                f"consumer leaf {declaration.leaf_id!r} terminal merge state is missing"
            ) from exc
        matrix_intent_hash = state_ref.metadata.get("matrix_intent_hash")
        if not isinstance(matrix_intent_hash, str):
            raise ValueError(
                f"consumer leaf {declaration.leaf_id!r} terminal merge identity is missing"
            )
        _validate_merge_state_ref(
            declaration,
            state_ref,
            matrix_intent_hash=matrix_intent_hash,
        )
        terminal_merge_state = json.loads(provider.get_bytes(state_ref))
        consumer = _resolve_consumer(declaration)
        finalized.append(
            (
                declaration,
                state_ref,
                consumer.finalize(
                    EvaluationBatchFinalizeInput(
                        matrix_intent_hash=matrix_intent_hash,
                        terminal_merge_state=terminal_merge_state,
                        parameters=_callback_parameters(declaration),
                        execution_context=execution_context,
                    )
                ),
            )
        )
    for declaration, _state_ref, terminal in finalized:
        if (
            terminal.schema_id != declaration.compact_product_schema_id
            or terminal.schema_version != declaration.compact_product_schema_version
            or terminal.role != declaration.compact_product_role
        ):
            raise ValueError(
                f"consumer leaf {declaration.leaf_id!r} returned wrong terminal product contract"
            )
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(
            analysis_type="feedbax.evaluation.batch_compaction",
            inputs=parents,
        ),
        root=custody_root / "analysis",
        index_manifest=False,
    )
    for declaration, state_ref, terminal in finalized:
        context.record_data_product(
            terminal.payload,
            product_schema_id=terminal.schema_id,
            product_schema_version=terminal.schema_version,
            role=terminal.role,
            logical_name=declaration.leaf_id,
            materialization={
                "merge_state_schema_id": declaration.merge_state_schema_id,
                "merge_state_schema_version": declaration.merge_state_schema_version,
                "merge_state_sha256": state_ref.sha256,
                "consumer_parameters": declaration.parameters,
                "consumer_parameters_sha256": _parameters_sha256(declaration),
            },
        )
    manifest, manifest_path = context.finalize()
    manifest_provider = ImmutableArtifactBlobProvider(custody_root)
    return (
        manifest_provider.store_bytes(
            manifest_path.read_bytes(),
            role="terminal_analysis_manifest",
            logical_name=f"{manifest.id}.json",
            media_type="application/json",
            metadata={"manifest_id": manifest.id, "kind": manifest.kind},
        ),
    )


def _resolve_consumer(
    declaration: EvaluationBatchConsumerDeclaration,
) -> EvaluationBatchConsumer:
    try:
        return _CONSUMERS[(declaration.consumer_id, declaration.consumer_version)]
    except KeyError as exc:
        raise ValueError(
            "no registered evaluation batch consumer for "
            f"{declaration.consumer_id!r}@{declaration.consumer_version!r}"
        ) from exc


def _callback_parameters(
    declaration: EvaluationBatchConsumerDeclaration,
) -> Mapping[str, JsonValue]:
    return json.loads(canonical_json_bytes(declaration.parameters))


def _parameters_sha256(declaration: EvaluationBatchConsumerDeclaration) -> str:
    return sha256_bytes(canonical_json_bytes(declaration.parameters))


def _require_execution_context(value: Any) -> StagedExecutionContext:
    if not isinstance(value, StagedExecutionContext):
        raise ValueError("evaluation batch callback requires a resolved StagedExecutionContext")
    return value


def _validate_callback_input(
    declaration: EvaluationBatchConsumerDeclaration,
    value: EvaluationBatchConsumerInput,
) -> None:
    if canonical_json_bytes(dict(value.parameters)) != canonical_json_bytes(declaration.parameters):
        raise ValueError(f"consumer leaf {declaration.leaf_id!r} callback parameters drifted")
    _require_execution_context(value.execution_context)


def _validate_merge_state_ref(
    declaration: EvaluationBatchConsumerDeclaration,
    value: ArtifactRef,
    *,
    matrix_intent_hash: str,
) -> None:
    if (
        value.role != f"{declaration.compact_product_role}_merge_state"
        or value.metadata.get("schema_id") != declaration.merge_state_schema_id
        or value.metadata.get("schema_version") != declaration.merge_state_schema_version
        or value.metadata.get("leaf_id") != declaration.leaf_id
        or value.metadata.get("matrix_intent_hash") != matrix_intent_hash
        or value.metadata.get("consumer_parameters") != declaration.parameters
        or value.metadata.get("consumer_parameters_sha256") != _parameters_sha256(declaration)
    ):
        raise ValueError(f"consumer leaf {declaration.leaf_id!r} merge-state identity drifted")


def _normalize_legacy_reclamation_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Bind unversioned historical acknowledgements to parameter-free identity."""
    normalized = json.loads(canonical_json_bytes(payload))
    for acknowledgement in normalized.get("leaf_acknowledgements", []):
        parameters = acknowledgement.setdefault("parameters", {})
        parameters_sha256 = sha256_bytes(canonical_json_bytes(parameters))
        for artifact_name in ("fragment", "merge_state"):
            metadata = acknowledgement.get(artifact_name, {}).setdefault("metadata", {})
            metadata.setdefault("consumer_parameters", parameters)
            metadata.setdefault("consumer_parameters_sha256", parameters_sha256)
    return normalized


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "EvaluationBatchConsumer",
    "EvaluationBatchConsumerInput",
    "EvaluationBatchFragment",
    "EvaluationBatchFinalizeInput",
    "EvaluationBatchMergeInput",
    "EvaluationBatchMergeState",
    "compact_evaluation_batch",
    "merge_evaluation_batch_fragment",
    "publish_evaluation_compaction_products",
    "reclaim_evaluation_batch_caches",
    "register_evaluation_batch_consumer",
]
