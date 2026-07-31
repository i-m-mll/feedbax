"""Finalize already-published compact products across governed matrix identities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from feedbax.analysis.evaluation_compaction import EvaluationBatchFragment
from feedbax.contracts.evaluation_lifecycle import (
    EvaluationBatchCompactionEvidence,
    EvaluationBatchMergeCheckpoint,
)
from feedbax.contracts.evaluation_product_union import (
    EvaluationCompactProductUnion,
    EvaluationCompactProductUnionEvidence,
    EvaluationCompactProductUnionSource,
    EvaluationCompactProductUnionSourceEvidence,
)
from feedbax.contracts.manifest import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    AnalysisDataProduct,
    AnalysisRunManifest,
    AnalysisRunSpec,
    ArtifactRef,
    ParentRef,
    analysis_run_manifest_id,
    canonical_json_bytes,
    safe_manifest_key,
    sha256_bytes,
)
from feedbax.contracts.migrations import migrate_structured_spec_payload
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.registry_errors import RegistryCollisionError


@dataclass(frozen=True)
class EvaluationCompactProductUnionBinding:
    """Runtime locations for one declared source; locations are not durable identity."""

    cohort_key: str
    custody_root: Path
    compaction_evidence_path: Path
    terminal_checkpoint_path: Path
    terminal_manifest: ArtifactRef


@dataclass(frozen=True)
class EvaluationCompactProductUnionValue:
    """One authenticated compact payload supplied to a union finalizer."""

    cohort_key: str
    matrix_intent_hash: str
    ordered_row_ids: tuple[str, ...]
    payload: Any


@dataclass(frozen=True)
class EvaluationCompactProductUnionInput:
    """Authenticated compact products supplied in explicitly declared order."""

    declaration: EvaluationCompactProductUnion
    sources: tuple[EvaluationCompactProductUnionValue, ...]


class EvaluationCompactProductUnionFinalizerRegistry:
    """Caller-owned finalizers for governed compact-product unions."""

    def __init__(self) -> None:
        self._sealed = False
        self._finalizers: dict[
            tuple[str, str],
            Callable[[EvaluationCompactProductUnionInput], EvaluationBatchFragment],
        ] = {}

    def register(
        self,
        consumer_id: str,
        consumer_version: str,
        finalize: Callable[[EvaluationCompactProductUnionInput], EvaluationBatchFragment],
    ) -> None:
        if self._sealed:
            raise RuntimeError("compact product union finalizer registry is sealed")
        if not consumer_id or not consumer_version or not callable(finalize):
            raise ValueError("compact product union finalizer declaration is invalid")
        key = (consumer_id, consumer_version)
        if key in self._finalizers:
            raise RegistryCollisionError(
                f"compact product union finalizer {key!r} is already registered"
            )
        self._finalizers[key] = finalize

    def get(
        self,
        consumer_id: str,
        consumer_version: str,
    ) -> Callable[[EvaluationCompactProductUnionInput], EvaluationBatchFragment]:
        try:
            return self._finalizers[(consumer_id, consumer_version)]
        except KeyError as exc:
            raise ValueError(
                "no registered compact product union finalizer for "
                f"{consumer_id!r}@{consumer_version!r}"
            ) from exc

    def keys(self) -> tuple[str, ...]:
        return tuple(
            f"{consumer_id}@{version}" for consumer_id, version in sorted(self._finalizers)
        )

    def seal(self) -> None:
        self._sealed = True


def finalize_evaluation_compact_product_union(
    declaration: EvaluationCompactProductUnion,
    bindings: Sequence[EvaluationCompactProductUnionBinding],
    *,
    custody_root: Path,
    finalizer_registry: EvaluationCompactProductUnionFinalizerRegistry,
) -> EvaluationCompactProductUnionEvidence:
    """Authenticate, union, publish, certify, and tear down without provider action."""
    declaration = EvaluationCompactProductUnion.model_validate(declaration.model_dump(mode="json"))
    union_sha256 = sha256_bytes(canonical_json_bytes(declaration.model_dump(mode="json")))
    expected_keys = tuple(source.cohort_key for source in declaration.sources)
    observed_keys = tuple(binding.cohort_key for binding in bindings)
    if observed_keys != expected_keys:
        raise ValueError(
            "compact product union bindings are missing, duplicated, or reordered; "
            f"expected={expected_keys!r}, observed={observed_keys!r}"
        )

    verified = tuple(
        _verify_source(source, binding)
        for source, binding in zip(declaration.sources, bindings, strict=True)
    )
    checkpoint_path = custody_root / "union-checkpoints" / f"{union_sha256}.json"
    if checkpoint_path.is_file():
        persisted = EvaluationCompactProductUnionEvidence.model_validate_json(
            checkpoint_path.read_text(encoding="utf-8")
        )
        _validate_persisted_union(declaration, persisted, verified, custody_root=custody_root)
        return persisted

    from feedbax.analysis.context import AnalysisRunContext

    parents = _union_parents(declaration, verified)
    spec = AnalysisRunSpec(
        analysis_type="feedbax.evaluation.compact_product_union",
        inputs=parents,
        params={"union_sha256": union_sha256},
    )
    context = AnalysisRunContext(
        spec=spec,
        root=custody_root / "analysis",
        index_manifest=False,
    )
    manifest_path = (
        context.root_path
        / "manifests"
        / "analysis_runs"
        / f"{safe_manifest_key(context.manifest_id)}.json"
    )
    if manifest_path.is_file():
        evidence = _recover_published_union(
            declaration,
            verified,
            union_sha256=union_sha256,
            spec=spec,
            manifest_path=manifest_path,
            custody_root=custody_root,
        )
        _atomic_write_json(checkpoint_path, evidence.model_dump(mode="json"))
        return evidence

    finalizer = finalizer_registry.get(
        declaration.consumer_id,
        declaration.consumer_version,
    )
    terminal = finalizer(
        EvaluationCompactProductUnionInput(
            declaration=declaration,
            sources=tuple(item.value for item in verified),
        )
    )
    if (
        terminal.schema_id != declaration.output_schema_id
        or terminal.schema_version != declaration.output_schema_version
        or terminal.role != declaration.output_role
    ):
        raise ValueError("compact product union finalizer returned the wrong output contract")
    product = context.record_data_product(
        terminal.payload,
        product_schema_id=terminal.schema_id,
        product_schema_version=terminal.schema_version,
        role=terminal.role,
        logical_name=declaration.output_logical_name,
        materialization=_union_materialization(declaration, union_sha256),
    )
    manifest, manifest_path = context.finalize()
    terminal_manifest = ImmutableArtifactBlobProvider(custody_root).store_bytes(
        manifest_path.read_bytes(),
        role="terminal_analysis_manifest",
        logical_name=f"{manifest.id}.json",
        media_type="application/json",
        metadata={"manifest_id": manifest.id, "kind": manifest.kind},
    )
    evidence = EvaluationCompactProductUnionEvidence(
        union_sha256=union_sha256,
        sources=tuple(item.evidence for item in verified),
        terminal_product=product.artifacts[0],
        terminal_manifest=terminal_manifest,
    )
    _validate_persisted_union(declaration, evidence, verified, custody_root=custody_root)
    _atomic_write_json(checkpoint_path, evidence.model_dump(mode="json"))
    return evidence


def _union_parents(
    declaration: EvaluationCompactProductUnion,
    verified: Sequence[_VerifiedSource],
) -> list[ParentRef]:
    return [
        ParentRef(
            kind="AnalysisRunManifest",
            id=item.manifest.id,
            role="terminal_compact_product",
            metadata={
                "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
                "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
                "manifest_sha256": source.terminal_manifest_sha256,
                "size_bytes": item.manifest_bytes_size,
                "cohort_key": source.cohort_key,
                "matrix_intent_hash": source.matrix_intent_hash,
            },
        )
        for source, item in zip(declaration.sources, verified, strict=True)
    ]


def _union_materialization(
    declaration: EvaluationCompactProductUnion,
    union_sha256: str,
) -> dict[str, Any]:
    return {
        "union_sha256": union_sha256,
        "declared_source_order": [source.cohort_key for source in declaration.sources],
        "source_terminal_product_sha256": [
            source.terminal_product_sha256 for source in declaration.sources
        ],
        "source_terminal_checkpoint_sha256": [
            source.terminal_checkpoint_sha256 for source in declaration.sources
        ],
    }


def _recover_published_union(
    declaration: EvaluationCompactProductUnion,
    verified: Sequence[_VerifiedSource],
    *,
    union_sha256: str,
    spec: AnalysisRunSpec,
    manifest_path: Path,
    custody_root: Path,
) -> EvaluationCompactProductUnionEvidence:
    manifest_bytes = manifest_path.read_bytes()
    manifest = AnalysisRunManifest.model_validate_json(manifest_bytes)
    if (
        manifest.id != analysis_run_manifest_id(spec)
        or manifest.inputs != spec.inputs
        or canonical_json_bytes(manifest.analysis_spec.inline)
        != canonical_json_bytes(spec.model_dump(mode="json", exclude_none=True))
    ):
        raise ValueError("persisted compact product union manifest identity drifted")
    product = _union_product(declaration, manifest, union_sha256=union_sha256)
    terminal_manifest = ImmutableArtifactBlobProvider(custody_root).store_bytes(
        manifest_bytes,
        role="terminal_analysis_manifest",
        logical_name=f"{manifest.id}.json",
        media_type="application/json",
        metadata={"manifest_id": manifest.id, "kind": manifest.kind},
    )
    evidence = EvaluationCompactProductUnionEvidence(
        union_sha256=union_sha256,
        sources=tuple(item.evidence for item in verified),
        terminal_product=product.artifacts[0],
        terminal_manifest=terminal_manifest,
    )
    _validate_persisted_union(declaration, evidence, verified, custody_root=custody_root)
    return evidence


@dataclass(frozen=True)
class _VerifiedSource:
    value: EvaluationCompactProductUnionValue
    evidence: EvaluationCompactProductUnionSourceEvidence
    manifest: AnalysisRunManifest
    manifest_bytes_size: int


def _verify_source(
    source: EvaluationCompactProductUnionSource,
    binding: EvaluationCompactProductUnionBinding,
) -> _VerifiedSource:
    if binding.cohort_key != source.cohort_key:
        raise ValueError("compact product union source cohort key drifted")
    compaction_bytes = _read_materialized(binding.compaction_evidence_path, "compaction evidence")
    if sha256_bytes(compaction_bytes) != source.compaction_evidence_sha256:
        raise ValueError("compact product union compaction evidence was tampered")
    compaction = EvaluationBatchCompactionEvidence.model_validate(
        migrate_structured_spec_payload(
            "EvaluationBatchCompactionEvidence",
            json.loads(compaction_bytes),
            path=str(binding.compaction_evidence_path),
        ).payload
    )
    if compaction.matrix_intent_hash != source.matrix_intent_hash:
        raise ValueError("compact product union source matrix identity drifted")
    terminal_manifests = [
        item
        for item in compaction.terminal_products
        if item.sha256 == binding.terminal_manifest.sha256
    ]
    if len(terminal_manifests) != 1:
        raise ValueError(
            "compact product union terminal manifest is missing or duplicated in compaction evidence"
        )
    relevant = []
    for reclamation in compaction.reclamations:
        acknowledgements = [
            item for item in reclamation.leaf_acknowledgements if item.leaf_id == source.leaf_id
        ]
        if len(acknowledgements) > 1:
            raise ValueError("compact product union source has duplicate leaf acknowledgements")
        if acknowledgements:
            relevant.append((reclamation, acknowledgements[0]))
    ordered_row_ids = tuple(
        row_id for reclamation, _ in relevant for row_id in reclamation.ordered_row_ids
    )
    if ordered_row_ids != source.ordered_row_ids:
        raise ValueError("compact product union source ordered row coverage drifted")
    if not relevant:
        raise ValueError("compact product union source has no terminal leaf coverage")

    checkpoint_bytes = _read_materialized(
        binding.terminal_checkpoint_path, "terminal merge checkpoint"
    )
    if sha256_bytes(checkpoint_bytes) != source.terminal_checkpoint_sha256:
        raise ValueError("compact product union terminal checkpoint was tampered")
    checkpoint_payload = json.loads(checkpoint_bytes)
    if (
        checkpoint_payload.get("schema_id") != source.terminal_checkpoint_schema_id
        or checkpoint_payload.get("schema_version") != source.terminal_checkpoint_schema_version
    ):
        raise ValueError("compact product union terminal checkpoint schema identity drifted")
    checkpoint = EvaluationBatchMergeCheckpoint.model_validate(
        migrate_structured_spec_payload(
            "EvaluationBatchMergeCheckpoint",
            checkpoint_payload,
            path=str(binding.terminal_checkpoint_path),
        ).payload
    )
    terminal_reclamation, terminal_acknowledgement = relevant[-1]
    if (
        checkpoint.matrix_intent_hash != source.matrix_intent_hash
        or checkpoint.batch.batch_id != terminal_reclamation.batch_id
        or checkpoint.declaration.leaf_id != source.leaf_id
        or checkpoint.declaration.consumer_id != source.consumer_id
        or checkpoint.declaration.consumer_version != source.consumer_version
        or checkpoint.declaration.compact_product_schema_id != source.compact_product_schema_id
        or checkpoint.declaration.compact_product_schema_version
        != source.compact_product_schema_version
        or checkpoint.declaration.compact_product_role != source.compact_product_role
        or checkpoint.acknowledgement.merge_state.sha256
        != terminal_acknowledgement.merge_state.sha256
    ):
        raise ValueError("compact product union terminal checkpoint identity drifted")

    manifest_provider = ImmutableArtifactBlobProvider(binding.custody_root)
    manifest_bytes = _get_materialized(
        manifest_provider,
        binding.terminal_manifest,
        "terminal manifest",
    )
    if (
        binding.terminal_manifest.sha256 != source.terminal_manifest_sha256
        or sha256_bytes(manifest_bytes) != source.terminal_manifest_sha256
    ):
        raise ValueError("compact product union terminal manifest identity drifted")
    manifest = AnalysisRunManifest.model_validate_json(manifest_bytes)
    terminal_analysis_type = checkpoint.declaration.terminal_analysis_type
    if (
        binding.terminal_manifest.metadata.get("analysis_type") != terminal_analysis_type
        or manifest.analysis_spec.inline.get("analysis_type") != terminal_analysis_type
    ):
        raise ValueError("compact product union terminal analysis identity drifted")
    products = [
        product
        for product in manifest.produced_data
        if product.logical_name == source.leaf_id
        and product.role == source.compact_product_role
        and product.product_schema_id == source.compact_product_schema_id
        and product.product_schema_version == source.compact_product_schema_version
    ]
    if len(products) != 1 or len(products[0].artifacts) != 1:
        raise ValueError("compact product union terminal product is missing or duplicated")
    product_ref = products[0].artifacts[0]
    if (
        product_ref.sha256 != source.terminal_product_sha256
        or products[0].materialization.get("merge_state_sha256")
        != checkpoint.acknowledgement.merge_state.sha256
    ):
        raise ValueError("compact product union terminal product identity drifted")
    product_bytes = _get_materialized(
        ImmutableArtifactBlobProvider(binding.custody_root / "analysis"),
        product_ref,
        "terminal product",
    )
    if sha256_bytes(product_bytes) != source.terminal_product_sha256:
        raise ValueError("compact product union terminal product was tampered")
    return _VerifiedSource(
        value=EvaluationCompactProductUnionValue(
            cohort_key=source.cohort_key,
            matrix_intent_hash=source.matrix_intent_hash,
            ordered_row_ids=source.ordered_row_ids,
            payload=json.loads(product_bytes),
        ),
        evidence=EvaluationCompactProductUnionSourceEvidence(
            cohort_key=source.cohort_key,
            matrix_intent_hash=source.matrix_intent_hash,
            ordered_row_ids=source.ordered_row_ids,
            terminal_checkpoint_sha256=source.terminal_checkpoint_sha256,
            terminal_product=product_ref,
        ),
        manifest=manifest,
        manifest_bytes_size=len(manifest_bytes),
    )


def _validate_persisted_union(
    declaration: EvaluationCompactProductUnion,
    evidence: EvaluationCompactProductUnionEvidence,
    sources: Sequence[_VerifiedSource],
    *,
    custody_root: Path,
) -> None:
    expected_hash = sha256_bytes(canonical_json_bytes(declaration.model_dump(mode="json")))
    if evidence.union_sha256 != expected_hash or evidence.sources != tuple(
        source.evidence for source in sources
    ):
        raise ValueError("persisted compact product union identity drifted")
    manifest_bytes = _get_materialized(
        ImmutableArtifactBlobProvider(custody_root),
        evidence.terminal_manifest,
        "persisted terminal manifest",
    )
    manifest = AnalysisRunManifest.model_validate_json(manifest_bytes)
    product = _union_product(declaration, manifest, union_sha256=expected_hash)
    if product.artifacts != [evidence.terminal_product]:
        raise ValueError("persisted compact product union publication drifted")
    _get_materialized(
        ImmutableArtifactBlobProvider(custody_root / "analysis"),
        evidence.terminal_product,
        "persisted terminal product",
    )


def _union_product(
    declaration: EvaluationCompactProductUnion,
    manifest: AnalysisRunManifest,
    *,
    union_sha256: str,
) -> AnalysisDataProduct:
    products = [
        product
        for product in manifest.produced_data
        if product.logical_name == declaration.output_logical_name
        and product.role == declaration.output_role
        and product.product_schema_id == declaration.output_schema_id
        and product.product_schema_version == declaration.output_schema_version
    ]
    if (
        len(products) != 1
        or len(products[0].artifacts) != 1
        or canonical_json_bytes(products[0].materialization)
        != canonical_json_bytes(_union_materialization(declaration, union_sha256))
    ):
        raise ValueError("persisted compact product union publication drifted")
    return products[0]


def _read_materialized(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise ValueError(f"compact product union {label} is unmaterialized")
    return path.read_bytes()


def _get_materialized(
    provider: ImmutableArtifactBlobProvider,
    artifact: ArtifactRef,
    label: str,
) -> bytes:
    try:
        return provider.get_bytes(artifact)
    except FileNotFoundError as exc:
        raise ValueError(f"compact product union {label} is unmaterialized") from exc


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "EvaluationCompactProductUnionBinding",
    "EvaluationCompactProductUnionFinalizerRegistry",
    "EvaluationCompactProductUnionInput",
    "EvaluationCompactProductUnionValue",
    "finalize_evaluation_compact_product_union",
]
