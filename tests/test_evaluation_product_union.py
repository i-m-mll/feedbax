from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from feedbax.analysis.evaluation_compaction import (
    EvaluationBatchConsumerInput,
    EvaluationBatchConsumerRegistry,
    EvaluationBatchFragment,
    EvaluationBatchMergeInput,
    EvaluationBatchMergeState,
    compact_evaluation_batch,
    merge_evaluation_batch_fragment,
    publish_evaluation_compaction_products,
    reclaim_evaluation_batch_caches,
)
from feedbax.analysis.evaluation_product_union import (
    EvaluationCompactProductUnionBinding,
    EvaluationCompactProductUnionFinalizerRegistry,
    EvaluationCompactProductUnionInput,
    finalize_evaluation_compact_product_union,
)
from feedbax.analysis.execution_context import EMPTY_STAGED_EXECUTION_CONTEXT
from feedbax.contracts.evaluation_lifecycle import (
    EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_ID,
    EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_VERSION,
    EvaluationBatchCompactionEvidence,
    EvaluationBatchConsumerDeclaration,
    EvaluationLifecycleRowOutcome,
    EvaluationMatrixBatchUnit,
)
from feedbax.contracts.evaluation_product_union import (
    EVALUATION_COMPACT_PRODUCT_UNION_EVIDENCE_SCHEMA_ID,
    EVALUATION_COMPACT_PRODUCT_UNION_SCHEMA_ID,
    EvaluationCompactProductUnion,
    EvaluationCompactProductUnionSource,
)
from feedbax.contracts.manifest import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    AnalysisRunManifest,
    ParentRef,
    evaluation_states_cache_path,
    sha256_bytes,
)
from feedbax.contracts.migrations import migrate_structured_spec_payload
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


_CONSUMER_ID = "tests.cross-matrix"
_CONSUMER_VERSION = "v1"
_LEAF_ID = "cohort"


@dataclass(frozen=True)
class _SourceFixture:
    source: EvaluationCompactProductUnionSource
    binding: EvaluationCompactProductUnionBinding
    cache_paths: tuple[Path, ...]


def _declaration() -> EvaluationBatchConsumerDeclaration:
    return EvaluationBatchConsumerDeclaration(
        leaf_id=_LEAF_ID,
        consumer_id=_CONSUMER_ID,
        consumer_version=_CONSUMER_VERSION,
        terminal_analysis_type="tests.cohort.analysis",
        accepted_evaluation_state_schema_ids=("tests.states.v1",),
        compact_product_schema_id="tests.cohort.product",
        compact_product_schema_version="tests.cohort.product.v1",
        compact_product_role="cohort_product",
        merge_state_schema_id="tests.cohort.merge",
        merge_state_schema_version="tests.cohort.merge.v1",
    )


def _register_consumers(
    union_calls: list[tuple[str, ...]],
) -> tuple[EvaluationBatchConsumerRegistry, EvaluationCompactProductUnionFinalizerRegistry]:
    declaration = _declaration()

    def compact(value: EvaluationBatchConsumerInput) -> EvaluationBatchFragment:
        return EvaluationBatchFragment(
            payload={"rows": list(value.batch.ordered_row_ids)},
            schema_id=declaration.compact_product_schema_id,
            schema_version=declaration.compact_product_schema_version,
            role=declaration.compact_product_role,
        )

    def merge(value: EvaluationBatchMergeInput) -> EvaluationBatchMergeState:
        prior = [] if value.prior_merge_state is None else value.prior_merge_state["rows"]
        return EvaluationBatchMergeState(
            payload={"rows": [*prior, *value.fragment["rows"]]},
            schema_id=declaration.merge_state_schema_id,
            schema_version=declaration.merge_state_schema_version,
        )

    consumer_registry = EvaluationBatchConsumerRegistry()
    consumer_registry.register(
        _CONSUMER_ID,
        _CONSUMER_VERSION,
        compact=compact,
        merge=merge,
        finalize=lambda value: EvaluationBatchFragment(
            payload=value.terminal_merge_state,
            schema_id=declaration.compact_product_schema_id,
            schema_version=declaration.compact_product_schema_version,
            role=declaration.compact_product_role,
        ),
    )

    def finalize_union(value: EvaluationCompactProductUnionInput) -> EvaluationBatchFragment:
        union_calls.append(tuple(source.cohort_key for source in value.sources))
        return EvaluationBatchFragment(
            payload={
                "cohorts": [
                    {
                        "cohort_key": source.cohort_key,
                        "matrix_intent_hash": source.matrix_intent_hash,
                        "rows": source.payload["rows"],
                    }
                    for source in value.sources
                ]
            },
            schema_id=value.declaration.output_schema_id,
            schema_version=value.declaration.output_schema_version,
            role=value.declaration.output_role,
        )

    registry = EvaluationCompactProductUnionFinalizerRegistry()
    registry.register(
        _CONSUMER_ID,
        _CONSUMER_VERSION,
        finalize_union,
    )
    return consumer_registry, registry


def _source_fixture(
    root: Path,
    *,
    consumer_registry: EvaluationBatchConsumerRegistry,
    cohort_key: str,
    matrix_intent_hash: str,
    row_ids: tuple[str, ...],
) -> _SourceFixture:
    declaration = _declaration()
    batch = EvaluationMatrixBatchUnit(
        batch_id=f"{cohort_key}-batch",
        ordered_row_ids=row_ids,
        required_leaf_ids=(_LEAF_ID,),
    )
    outcomes = []
    manifests = []
    cache_paths = []
    for row_id in row_ids:
        row_root = root / "raw" / row_id
        manifest_id = f"evaluation:{cohort_key}:{row_id}"
        cache_path = evaluation_states_cache_path(manifest_id, root=row_root)
        cache_path.parent.mkdir(parents=True)
        cache_path.write_bytes(f"raw:{cohort_key}:{row_id}".encode())
        manifest_path = row_root / "manifest.json"
        manifest = {
            "id": manifest_id,
            "metadata": {"cache": {"states_path": str(cache_path)}},
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        outcomes.append(
            EvaluationLifecycleRowOutcome(
                row_id=row_id,
                manifest_id=manifest_id,
                manifest_path=str(manifest_path),
                diagnostic_schema_ids=("tests.states.v1",),
            )
        )
        manifests.append(manifest)
        cache_paths.append(cache_path)
    authorities = []
    for outcome in outcomes:
        manifest_bytes = Path(outcome.manifest_path).read_bytes()
        authorities.append(
            ParentRef(
                kind="EvaluationRunManifest",
                id=outcome.manifest_id,
                role="evaluation_run",
                metadata={
                    "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
                    "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
                    "manifest_sha256": sha256_bytes(manifest_bytes),
                    "size_bytes": len(manifest_bytes),
                },
            )
        )
    custody_root = root / "custody"
    fragment = compact_evaluation_batch(
        declaration,
        EvaluationBatchConsumerInput(
            matrix_intent_hash=matrix_intent_hash,
            batch=batch,
            outcomes=tuple(outcomes),
            manifests=tuple(manifests),
            states=tuple({"row": row_id} for row_id in row_ids),
            parent_authorities=tuple(authorities),
            parameters=declaration.parameters,
            execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
        ),
        registry=consumer_registry,
        custody_root=custody_root,
    )
    acknowledgement = merge_evaluation_batch_fragment(
        declaration,
        registry=consumer_registry,
        matrix_intent_hash=matrix_intent_hash,
        batch=batch,
        parent_authorities=authorities,
        fragment=fragment,
        prior_merge_state=None,
        custody_root=custody_root,
        execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
    )
    reclamation = reclaim_evaluation_batch_caches(
        batch,
        registry=consumer_registry,
        matrix_intent_hash=matrix_intent_hash,
        batch_index=0,
        outcomes=outcomes,
        acknowledgements=(acknowledgement,),
        required_declarations=(declaration,),
        custody_root=custody_root,
        execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
    )
    terminal_manifest = publish_evaluation_compaction_products(
        (declaration,),
        {_LEAF_ID: acknowledgement.merge_state},
        outcomes,
        registry=consumer_registry,
        custody_root=custody_root,
        execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
    )[0]
    compaction = EvaluationBatchCompactionEvidence(
        matrix_intent_hash=matrix_intent_hash,
        ordered_batch_ids=(batch.batch_id,),
        declared_leaf_ids=(_LEAF_ID,),
        required_leaf_ids_by_batch={batch.batch_id: (_LEAF_ID,)},
        reclamations=(reclamation,),
        terminal_products=(terminal_manifest,),
    )
    compaction_path = root / "evaluation-batch-compaction.json"
    compaction_path.write_text(
        json.dumps(compaction.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = custody_root / "merge-checkpoints" / _LEAF_ID / f"{batch.batch_id}.json"
    terminal_manifest_bytes = ImmutableArtifactBlobProvider(custody_root).get_bytes(
        terminal_manifest
    )
    manifest = AnalysisRunManifest.model_validate_json(terminal_manifest_bytes)
    product_ref = manifest.produced_data[0].artifacts[0]
    source = EvaluationCompactProductUnionSource(
        cohort_key=cohort_key,
        matrix_intent_hash=matrix_intent_hash,
        consumer_id=_CONSUMER_ID,
        consumer_version=_CONSUMER_VERSION,
        leaf_id=_LEAF_ID,
        compact_product_schema_id=declaration.compact_product_schema_id,
        compact_product_schema_version=declaration.compact_product_schema_version,
        compact_product_role=declaration.compact_product_role,
        ordered_row_ids=row_ids,
        compaction_evidence_sha256=sha256_bytes(compaction_path.read_bytes()),
        terminal_checkpoint_schema_id=EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_ID,
        terminal_checkpoint_schema_version=EVALUATION_BATCH_MERGE_CHECKPOINT_SCHEMA_VERSION,
        terminal_checkpoint_sha256=sha256_bytes(checkpoint_path.read_bytes()),
        terminal_manifest_sha256=terminal_manifest.sha256,
        terminal_product_sha256=product_ref.sha256,
    )
    return _SourceFixture(
        source=source,
        binding=EvaluationCompactProductUnionBinding(
            cohort_key=cohort_key,
            custody_root=custody_root,
            compaction_evidence_path=compaction_path,
            terminal_checkpoint_path=checkpoint_path,
            terminal_manifest=terminal_manifest,
        ),
        cache_paths=tuple(cache_paths),
    )


def _union(
    sources: tuple[EvaluationCompactProductUnionSource, ...],
) -> EvaluationCompactProductUnion:
    return EvaluationCompactProductUnion(
        consumer_id=_CONSUMER_ID,
        consumer_version=_CONSUMER_VERSION,
        output_schema_id="tests.cross-matrix.product",
        output_schema_version="tests.cross-matrix.product.v1",
        output_role="cross_matrix_product",
        output_logical_name="cross-matrix",
        sources=sources,
    )


def _two_sources(
    tmp_path: Path,
    union_calls: list[tuple[str, ...]],
) -> tuple[
    tuple[_SourceFixture, _SourceFixture],
    EvaluationCompactProductUnionFinalizerRegistry,
]:
    consumer_registry, finalizer_registry = _register_consumers(union_calls)
    fixtures = (
        _source_fixture(
            tmp_path / "discrete",
            consumer_registry=consumer_registry,
            cohort_key="discrete",
            matrix_intent_hash="a" * 64,
            row_ids=("d-0", "d-1"),
        ),
        _source_fixture(
            tmp_path / "continuous",
            consumer_registry=consumer_registry,
            cohort_key="continuous",
            matrix_intent_hash="b" * 64,
            row_ids=("c-0", "c-1"),
        ),
    )
    return fixtures, finalizer_registry


def test_provider_free_shadow_unions_two_matrices_and_resumes_with_identical_terminal_bytes(
    tmp_path: Path,
) -> None:
    union_calls: list[tuple[str, ...]] = []
    fixtures, finalizer_registry = _two_sources(tmp_path, union_calls)
    declaration = _union(tuple(item.source for item in fixtures))
    bindings = tuple(item.binding for item in fixtures)
    custody_root = tmp_path / "union"

    first = finalize_evaluation_compact_product_union(
        declaration,
        bindings,
        custody_root=custody_root,
        finalizer_registry=finalizer_registry,
    )
    first_manifest_bytes = ImmutableArtifactBlobProvider(custody_root).get_bytes(
        first.terminal_manifest
    )
    product_bytes = ImmutableArtifactBlobProvider(custody_root / "analysis").get_bytes(
        first.terminal_product
    )
    (custody_root / "union-checkpoints" / f"{first.union_sha256}.json").unlink()
    recovered = finalize_evaluation_compact_product_union(
        declaration,
        bindings,
        custody_root=custody_root,
        finalizer_registry=finalizer_registry,
    )
    resumed = finalize_evaluation_compact_product_union(
        declaration,
        bindings,
        custody_root=custody_root,
        finalizer_registry=finalizer_registry,
    )

    assert first == recovered == resumed
    assert first_manifest_bytes == ImmutableArtifactBlobProvider(custody_root).get_bytes(
        resumed.terminal_manifest
    )
    assert json.loads(product_bytes)["cohorts"] == [
        {
            "cohort_key": "discrete",
            "matrix_intent_hash": "a" * 64,
            "rows": ["d-0", "d-1"],
        },
        {
            "cohort_key": "continuous",
            "matrix_intent_hash": "b" * 64,
            "rows": ["c-0", "c-1"],
        },
    ]
    assert union_calls == [("discrete", "continuous")]
    assert first.provider_readiness == "not_evaluated"
    assert first.completed_stages == ("UNION", "COLLECT", "CERTIFY", "TEARDOWN")
    assert all(not path.exists() for item in fixtures for path in item.cache_paths)


@pytest.mark.parametrize(
    "invalid",
    [
        "duplicate_matrix",
        "duplicate_cohort",
        "consumer_mismatch",
        "duplicate_rows",
        "unversioned_source_product",
    ],
)
def test_union_runtime_revalidates_model_copy_bypasses(tmp_path: Path, invalid: str) -> None:
    fixtures, finalizer_registry = _two_sources(tmp_path, [])
    sources = [item.source for item in fixtures]
    if invalid == "duplicate_matrix":
        sources[1] = sources[1].model_copy(
            update={"matrix_intent_hash": sources[0].matrix_intent_hash}
        )
    elif invalid == "duplicate_cohort":
        sources[1] = sources[1].model_copy(update={"cohort_key": sources[0].cohort_key})
    elif invalid == "consumer_mismatch":
        sources[1] = sources[1].model_copy(update={"consumer_id": "tests.wrong"})
    elif invalid == "duplicate_rows":
        sources[0] = sources[0].model_copy(update={"ordered_row_ids": ("d-0", "d-0")})
    else:
        sources[0] = sources[0].model_copy(
            update={
                "compact_product_schema_version": sources[0].compact_product_schema_id,
            }
        )
    declaration = _union(tuple(item.source for item in fixtures)).model_copy(
        update={"sources": tuple(sources)}
    )

    with pytest.raises(ValueError):
        finalize_evaluation_compact_product_union(
            declaration,
            tuple(item.binding for item in fixtures),
            custody_root=tmp_path / "union",
            finalizer_registry=finalizer_registry,
        )


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("missing", "missing, duplicated, or reordered"),
        ("duplicate", "missing, duplicated, or reordered"),
        ("reordered", "missing, duplicated, or reordered"),
        ("wrong_consumer", "checkpoint identity drifted"),
        ("wrong_matrix", "checkpoint identity drifted"),
        ("wrong_analysis", "terminal analysis identity drifted"),
        ("wrong_coverage", "ordered row coverage drifted"),
        ("tampered", "checkpoint was tampered"),
        ("unmaterialized", "terminal product is unmaterialized"),
    ],
)
def test_union_sources_fail_closed(tmp_path: Path, failure: str, match: str) -> None:
    source_fixtures, finalizer_registry = _two_sources(tmp_path, [])
    fixtures = list(source_fixtures)
    sources = [item.source for item in fixtures]
    bindings = [item.binding for item in fixtures]
    if failure == "missing":
        bindings = bindings[:1]
    elif failure == "duplicate":
        bindings[1] = bindings[0]
    elif failure == "reordered":
        bindings.reverse()
    elif failure in {"wrong_consumer", "wrong_matrix", "wrong_analysis"}:
        path = bindings[0].terminal_checkpoint_path
        payload = json.loads(path.read_text())
        if failure == "wrong_consumer":
            payload["declaration"]["consumer_id"] = "tests.wrong"
            payload["acknowledgement"]["consumer_id"] = "tests.wrong"
        elif failure == "wrong_matrix":
            payload["matrix_intent_hash"] = "f" * 64
        else:
            payload["declaration"]["terminal_analysis_type"] = "tests.wrong.analysis"
            payload["acknowledgement"]["terminal_analysis_type"] = "tests.wrong.analysis"
            binding = bindings[0]
            bindings[0] = EvaluationCompactProductUnionBinding(
                cohort_key=binding.cohort_key,
                custody_root=binding.custody_root,
                compaction_evidence_path=binding.compaction_evidence_path,
                terminal_checkpoint_path=binding.terminal_checkpoint_path,
                terminal_manifest=binding.terminal_manifest.model_copy(
                    update={
                        "metadata": {
                            **binding.terminal_manifest.metadata,
                            "analysis_type": "tests.wrong.analysis",
                        }
                    }
                ),
            )
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        sources[0] = sources[0].model_copy(
            update={"terminal_checkpoint_sha256": sha256_bytes(path.read_bytes())}
        )
    elif failure == "wrong_coverage":
        sources[0] = sources[0].model_copy(update={"ordered_row_ids": ("d-1", "d-0")})
    elif failure == "tampered":
        bindings[0].terminal_checkpoint_path.write_bytes(
            bindings[0].terminal_checkpoint_path.read_bytes() + b" "
        )
    elif failure == "unmaterialized":
        manifest_bytes = ImmutableArtifactBlobProvider(bindings[0].custody_root).get_bytes(
            bindings[0].terminal_manifest
        )
        product = AnalysisRunManifest.model_validate_json(manifest_bytes).produced_data[0]
        provider = ImmutableArtifactBlobProvider(bindings[0].custody_root / "analysis")
        (provider.root / provider.canonical_relative_path(product.artifacts[0])).unlink()

    with pytest.raises(ValueError, match=match):
        finalize_evaluation_compact_product_union(
            _union(tuple(sources)),
            tuple(bindings),
            custody_root=tmp_path / "union",
            finalizer_registry=finalizer_registry,
        )


def test_union_structured_specs_reject_unversioned_predecessors() -> None:
    for family, schema_id in (
        ("EvaluationCompactProductUnion", EVALUATION_COMPACT_PRODUCT_UNION_SCHEMA_ID),
        (
            "EvaluationCompactProductUnionEvidence",
            EVALUATION_COMPACT_PRODUCT_UNION_EVIDENCE_SCHEMA_ID,
        ),
    ):
        with pytest.raises(ValueError, match="Unsupported Feedbax structured spec version"):
            migrate_structured_spec_payload(
                family,
                {"schema_id": schema_id, "schema_version": f"{schema_id}.v0"},
            )
