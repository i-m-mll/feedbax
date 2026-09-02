from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.analysis.context import (
    AnalysisArtifactFile,
    AnalysisRunContext,
    parent_ref_from_evaluation_manifest,
)
from feedbax.analysis.analysis import IdentityNode, SinglePort
from feedbax.analysis.execution import (
    AnalysisModuleTransformSpec,
    run_analyses_with_context,
    run_evaluation,
)
from feedbax.analysis.materialization import (
    AnalysisArtifactGroup,
    ContextMaterializationPending,
    ContextMaterializer,
    ExistingAnalysisArtifact,
    MaterializationResult,
)
from feedbax.contracts.base import (
    ArtifactRef,
    ParentRef,
)
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    DataProductParentRef,
    REGENERATION_SPEC_SCHEMA_ID,
    RegenerationCommand,
    RegenerationSpec,
    analysis_run_manifest_id,
    load_manifest,
)
from feedbax.persistence.manifest_index import rebuild_manifest_index
from feedbax.persistence.artifact_custody import (
    ArtifactBlobIntegrityError,
    ImmutableArtifactBlobProvider,
)
from feedbax.analysis.types import AnalysisInputData
from tests.analysis_fixtures import (
    ARTIFACT_PRODUCER_CALLS,
    ToyAnalysis,
    ToyArtifactProducer,
    build_toy_artifact_analyses,
    build_toy_analysis_data,
    execute_toy_evaluation,
    reset_artifact_producer_calls,
)


def test_headless_analysis_context_writes_manifest_figures_and_rebuildable_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    eval_manifest, eval_path = execute_toy_evaluation(tmp_path)
    eval_ref = parent_ref_from_evaluation_manifest(
        eval_manifest.id,
        uri=str(eval_path),
    )
    spec = AnalysisRunSpec(
        analysis_type="toy_analysis",
        inputs=[eval_ref],
        params={"outputs": ["toy"]},
    )
    context = AnalysisRunContext(
        spec=spec,
        root=tmp_path,
        fig_dump_formats=("json",),
        issues=["52c7728"],
    )

    _all_analyses, all_results, _all_figs = run_analyses_with_context(
        {"toy": ToyAnalysis(variant="toy", cache_result=True)},
        build_toy_analysis_data(),
        context,
        fig_dump_formats=["json"],
    )

    assert all_results["toy"]["value"] == 3
    assert context.manifest_id == analysis_run_manifest_id(spec)
    assert context.manifest_path is not None
    manifest = load_manifest(context.manifest_path)
    assert manifest.kind == "AnalysisRunManifest"
    assert manifest.status == "completed"
    assert manifest.inputs == [eval_ref]
    assert manifest.provenance.parents == [eval_ref]
    assert manifest.provenance.issues == ["52c7728"]
    assert manifest.analysis_spec.inline["inputs"][0]["id"] == eval_manifest.id
    assert manifest.summary_metrics["figure_count"] == 1
    assert manifest.summary_metrics["artifact_count"] == 1
    assert manifest.summary_metrics["analysis_count"] == 1
    assert len(manifest.artifacts) == 1
    assert manifest.artifacts[0].role == "figure"
    assert manifest.artifacts[0].media_type == "application/json"
    assert Path(manifest.artifacts[0].uri).exists()
    assert list(context.results_cache_dir.glob("*.pkl"))

    index_path = rebuild_manifest_index(tmp_path)
    with sqlite3.connect(index_path) as conn:
        manifest_row = conn.execute(
            "SELECT kind, status FROM manifests WHERE id = ?",
            (context.manifest_id,),
        ).fetchone()
        edge_row = conn.execute(
            """
            SELECT parent_kind, parent_id, role
            FROM lineage_edges
            WHERE child_id = ?
            """,
            (context.manifest_id,),
        ).fetchone()
        artifact_row = conn.execute(
            """
            SELECT role, logical_name, media_type
            FROM artifacts
            WHERE manifest_id = ?
            """,
            (context.manifest_id,),
        ).fetchone()

    assert manifest_row == ("AnalysisRunManifest", "completed")
    assert edge_row == ("EvaluationRunManifest", eval_manifest.id, "evaluation_run")
    assert artifact_row == ("figure", "toy/toy_toy_analysis_0.json", "application/json")


def test_analysis_result_cache_corruption_recomputes_with_versioned_payload(
    tmp_path: Path,
) -> None:
    reset_artifact_producer_calls()
    spec = AnalysisRunSpec(
        analysis_type="toy_artifact_analysis",
        params={"outputs": ["artifact_producer"]},
    )
    context = AnalysisRunContext(spec=spec, root=tmp_path)

    run_analyses_with_context(
        {"artifact_producer": ToyArtifactProducer(variant="toy", cache_result=True)},
        build_toy_analysis_data(),
        context,
        requested_outputs={"artifact_producer"},
    )

    cache_files = list(context.results_cache_dir.glob("*.pkl"))
    assert len(cache_files) == 1
    cache_files[0].write_bytes(b"not a valid result cache")

    rerun_context = AnalysisRunContext(spec=spec, root=tmp_path)
    _all_analyses, cached_results, _all_figs = run_analyses_with_context(
        {"artifact_producer": ToyArtifactProducer(variant="toy", cache_result=True)},
        build_toy_analysis_data(),
        rerun_context,
        requested_outputs={"artifact_producer"},
    )

    assert ARTIFACT_PRODUCER_CALLS["count"] == 2
    assert cached_results["artifact_producer"]["value"] == 4
    assert b"schema_version" in cache_files[0].read_bytes()


def test_analysis_md5_identity_includes_dependency_wiring() -> None:
    left = IdentityNode(inputs=SinglePort(input="left_dependency"))
    right = IdentityNode(inputs=SinglePort(input="right_dependency"))

    assert left.md5_str != right.md5_str


def test_legacy_evaluation_state_cache_key_includes_prng_key(tmp_path: Path) -> None:
    def eval_fn(key, hps, model, task):
        del hps, model, task
        return jnp.asarray(key[1])

    data = AnalysisInputData(
        models={"toy": jnp.asarray(1)},
        tasks={"toy": jnp.asarray(1)},
        states=None,
        hps={"toy": SimpleNamespace()},
        extras={},
    )

    values = []
    for seed in (0, 1):
        result = run_evaluation(
            SimpleNamespace(eval_fn=eval_fn),
            data,
            common_inputs={},
            transforms=AnalysisModuleTransformSpec(),
            eval_info=SimpleNamespace(hash="same-eval-hash"),
            states_pkl_dir=tmp_path,
            key=jax.random.PRNGKey(seed),
        )
        values.append(int(result.states["toy"]))

    assert values == [0, 1]
    assert len(list(tmp_path.glob("same-eval-hash_*.pkl"))) == 2


def test_requested_outputs_empty_intersection_raises_clear_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    eval_manifest, eval_path = execute_toy_evaluation(tmp_path)
    eval_ref = parent_ref_from_evaluation_manifest(
        eval_manifest.id,
        uri=str(eval_path),
    )
    spec = AnalysisRunSpec(
        analysis_type="toy_analysis",
        inputs=[eval_ref],
        params={"requested_outputs": ["missing"]},
    )
    context = AnalysisRunContext(
        spec=spec,
        root=tmp_path,
        fig_dump_formats=("json",),
    )

    with pytest.raises(ValueError, match="requested_outputs=\\['missing'\\]") as excinfo:
        run_analyses_with_context(
            {"toy": ToyAnalysis(variant="toy", cache_result=True)},
            build_toy_analysis_data(),
            context,
            requested_outputs={"missing"},
        )

    assert "available_analysis_keys=['toy']" in str(excinfo.value)
    assert context.manifest_path is None


def test_requested_outputs_partial_intersection_runs_matching_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    eval_manifest, eval_path = execute_toy_evaluation(tmp_path)
    eval_ref = parent_ref_from_evaluation_manifest(
        eval_manifest.id,
        uri=str(eval_path),
    )
    spec = AnalysisRunSpec(
        analysis_type="toy_analysis",
        inputs=[eval_ref],
        params={"requested_outputs": ["toy", "missing"]},
    )
    context = AnalysisRunContext(
        spec=spec,
        root=tmp_path,
        fig_dump_formats=("json",),
    )

    all_analyses, all_results, _all_figs = run_analyses_with_context(
        {"toy": ToyAnalysis(variant="toy", cache_result=True)},
        build_toy_analysis_data(),
        context,
        fig_dump_formats=["json"],
        requested_outputs={"toy", "missing"},
    )

    assert set(all_analyses) == {"toy"}
    assert all_results["toy"]["value"] == 3
    manifest = load_manifest(context.manifest_path)
    assert manifest.status == "completed"
    assert manifest.summary_metrics["analysis_count"] == 1


def test_analysis_context_records_grouped_artifacts_cache_and_downstream_consumption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    eval_manifest, eval_path = execute_toy_evaluation(tmp_path)
    eval_ref = parent_ref_from_evaluation_manifest(
        eval_manifest.id,
        uri=str(eval_path),
    )
    spec = AnalysisRunSpec(
        analysis_type="toy_artifact_analysis",
        inputs=[eval_ref],
        params={"outputs": ["artifact_consumer"]},
    )
    reset_artifact_producer_calls()

    context = AnalysisRunContext(
        spec=spec,
        root=tmp_path,
        issues=["c3bce34"],
    )
    _all_analyses, all_results, _all_figs = run_analyses_with_context(
        build_toy_artifact_analyses(),
        build_toy_analysis_data(),
        context,
    )

    assert ARTIFACT_PRODUCER_CALLS["count"] == 1
    assert all_results["artifact_consumer"] == {
        "consumed_value": 4,
        "array_total": 15,
        "group_id": f"{context.manifest_id}:toy-artifact-group",
    }
    assert context.manifest_path is not None
    manifest = load_manifest(context.manifest_path)
    assert manifest.kind == "AnalysisRunManifest"
    assert manifest.summary_metrics["artifact_count"] == 2
    assert manifest.summary_metrics["figure_count"] == 0
    assert manifest.summary_metrics["analysis_count"] == 2

    artifacts_by_role = {artifact.role: artifact for artifact in manifest.artifacts}
    summary_ref = artifacts_by_role["analysis_summary"]
    arrays_ref = artifacts_by_role["analysis_arrays"]
    assert summary_ref.logical_name == "toy/summary.json"
    assert summary_ref.media_type == "application/json"
    assert arrays_ref.logical_name == "toy/arrays.npz"
    assert arrays_ref.media_type == "application/x-npz"
    assert (
        summary_ref.metadata["artifact_group"]["id"] == arrays_ref.metadata["artifact_group"]["id"]
    )
    assert summary_ref.metadata["artifact_group"]["member_role"] == "summary"
    assert arrays_ref.metadata["artifact_group"]["member_role"] == "bulk_arrays"
    assert arrays_ref.metadata["arrays"]["values"]["role"] == "toy_value_series"
    assert (tmp_path / summary_ref.metadata["relative_path"]).exists()
    assert (tmp_path / arrays_ref.metadata["relative_path"]).exists()
    assert Path(summary_ref.uri).exists()
    assert Path(arrays_ref.uri).exists()
    assert list(context.results_cache_dir.glob("*.pkl"))

    index_path = rebuild_manifest_index(tmp_path)
    with sqlite3.connect(index_path) as conn:
        artifact_rows = conn.execute(
            """
            SELECT role, logical_name, media_type
            FROM artifacts
            WHERE manifest_id = ?
            ORDER BY logical_name
            """,
            (context.manifest_id,),
        ).fetchall()

    assert artifact_rows == [
        ("analysis_arrays", "toy/arrays.npz", "application/x-npz"),
        ("analysis_summary", "toy/summary.json", "application/json"),
    ]

    cached_context = AnalysisRunContext(
        spec=spec,
        root=tmp_path,
        issues=["c3bce34"],
    )
    _all_analyses, cached_results, _all_figs = run_analyses_with_context(
        build_toy_artifact_analyses(),
        build_toy_analysis_data(),
        cached_context,
    )

    assert ARTIFACT_PRODUCER_CALLS["count"] == 1
    assert cached_results["artifact_consumer"] == all_results["artifact_consumer"]
    cached_manifest = load_manifest(cached_context.manifest_path)
    assert cached_manifest.summary_metrics["artifact_count"] == 2
    assert {artifact.artifact_id for artifact in cached_manifest.artifacts} == {
        summary_ref.artifact_id,
        arrays_ref.artifact_id,
    }


def test_context_materializer_emits_json_payload_with_explicit_compute_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    spec = AnalysisRunSpec(
        analysis_type="toy_context_materializer",
        params={"requested_outputs": ["materializer"]},
    )
    context = AnalysisRunContext(spec=spec, root=tmp_path)

    def materialize(run_context: AnalysisRunContext) -> dict[str, object]:
        return {
            "kind": "toy.materialized.v1",
            "manifest_id": run_context.manifest_id,
            "value": 17,
        }

    analysis = ContextMaterializer(
        materializer=materialize,
        artifact_role="toy_materialized_payload",
        logical_name="toy/materialized.json",
        schema_boundary="toy-owned payload",
    )
    pending = analysis.compute(build_toy_analysis_data())

    assert isinstance(pending, ContextMaterializationPending)
    assert pending.status == "pending_context_artifact_emission"

    _all_analyses, all_results, _all_figs = run_analyses_with_context(
        {"materializer": analysis},
        build_toy_analysis_data(),
        context,
    )

    assert all_results["materializer"]["kind"] == "toy.materialized.v1"
    manifest = load_manifest(context.manifest_path)
    assert manifest.summary_metrics["artifact_count"] == 1
    payload_ref = manifest.artifacts[0]
    assert payload_ref.role == "toy_materialized_payload"
    assert payload_ref.logical_name == "toy/materialized.json"
    assert payload_ref.metadata["schema_boundary"] == "toy-owned payload"
    payload = json.loads(ImmutableArtifactBlobProvider(tmp_path).get_bytes(payload_ref))
    assert payload == all_results["materializer"]


def test_record_json_artifact_ingestion_failure_does_not_record_manifest_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(analysis_type="toy_failed_json_ingestion"),
        root=tmp_path,
        index_manifest=False,
    )

    def fail_ingestion(*_args, **_kwargs):
        raise OSError("simulated CAS ingestion failure")

    monkeypatch.setattr(ImmutableArtifactBlobProvider, "store_bytes", fail_ingestion)

    with pytest.raises(OSError, match="simulated CAS ingestion failure"):
        context.record_json_artifact(
            {"value": 23},
            role="toy_payload",
            logical_name="toy/payload.json",
        )

    assert context.artifacts == ()
    manifest, _path = context.finalize(status="failed")
    assert manifest.artifacts == []


def test_record_data_product_custodies_payload_and_finalizes_exact_record(
    tmp_path: Path,
) -> None:
    parent_metadata = {
        "ref_schema_id": "feedbax.ref.authenticated_manifest",
        "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
        "manifest_sha256": "a" * 64,
        "size_bytes": 123,
    }
    parent = ParentRef(
        kind="AnalysisRunManifest",
        id="analysis-run:retained-authority",
        role="retained_analysis",
        metadata=parent_metadata,
    )
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(
            analysis_type="toy_scalar_projection",
            inputs=[parent],
        ),
        root=tmp_path,
        index_manifest=False,
    )

    product = context.record_data_product(
        {"value": np.float64(1.25), "path": "metrics.peak"},
        product_schema_id="downstream.scalar_projection",
        product_schema_version="downstream.scalar_projection.v1",
        role="peak_velocity_scalar",
        logical_name="peak_velocity_scalar",
        artifact_role="scalar_projection_payload",
        artifact_logical_name="scalars/peak_velocity.json",
        parameters={"cohort": "discrete"},
        materialization={"scalar_path": "value"},
    )

    assert context.produced_data == (product,)
    assert context.artifacts == tuple(product.artifacts)
    assert product.producer_manifest_id == context.manifest_id
    assert product.parent_manifests == [
        DataProductParentRef(
            kind=parent.kind,
            id=parent.id,
            role=parent.role,
            manifest_hash="a" * 64,
            metadata=parent_metadata,
        )
    ]
    payload_ref = product.artifacts[0]
    assert payload_ref.role == "scalar_projection_payload"
    assert payload_ref.uri == payload_ref.artifact_id
    assert json.loads(ImmutableArtifactBlobProvider(tmp_path).get_bytes(payload_ref)) == {
        "path": "metrics.peak",
        "value": 1.25,
    }

    manifest, path = context.finalize()
    loaded = load_manifest(path)
    assert manifest.produced_data == [product]
    assert loaded.produced_data == [product]
    assert loaded.produced_data[0].artifacts == [payload_ref]


def test_record_data_product_rejects_partial_authenticated_parent_profile(
    tmp_path: Path,
) -> None:
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(
            analysis_type="toy_scalar_projection",
            inputs=[
                ParentRef(
                    kind="AnalysisRunManifest",
                    id="analysis-run:partial-authority",
                    metadata={
                        "ref_schema_id": "feedbax.ref.authenticated_manifest",
                        "manifest_sha256": "a" * 64,
                    },
                )
            ],
        ),
        root=tmp_path,
        index_manifest=False,
    )

    with pytest.raises(ValueError, match="Authenticated manifest ref .* is incomplete"):
        context.record_data_product(
            {"value": 1.25},
            product_schema_id="downstream.scalar_projection",
            product_schema_version="downstream.scalar_projection.v1",
            role="peak_velocity_scalar",
            logical_name="peak_velocity_scalar",
        )

    assert context.produced_data == ()
    assert context.artifacts == ()


def test_record_data_product_rejects_duplicate_identity_and_role_schema(
    tmp_path: Path,
) -> None:
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(analysis_type="toy_scalar_projection"),
        root=tmp_path,
        index_manifest=False,
    )
    common = {
        "product_schema_id": "downstream.scalar_projection",
        "product_schema_version": "downstream.scalar_projection.v1",
        "role": "peak_velocity_scalar",
        "logical_name": "peak_velocity_scalar",
    }
    context.record_data_product({"value": 1.25}, **common)

    with pytest.raises(ValueError, match="duplicate AnalysisDataProduct"):
        context.record_data_product({"value": 1.25}, **common)
    with pytest.raises(ValueError, match="duplicate AnalysisDataProduct role/schema binding"):
        context.record_data_product(
            {"value": 2.5},
            **{
                **common,
                "logical_name": "peak_velocity_scalar_alternate",
            },
        )

    second = context.record_data_product(
        {"value": 2.5},
        **{
            **common,
            "role": "pulse_response_scalar",
            "logical_name": "pulse_response_scalar",
        },
    )
    assert context.produced_data == (context.produced_data[0], second)
    assert len(context.artifacts) == 2


def test_record_data_product_finalization_fails_closed_on_tampered_payload(
    tmp_path: Path,
) -> None:
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(analysis_type="toy_scalar_projection"),
        root=tmp_path,
        index_manifest=False,
    )
    product = context.record_data_product(
        {"value": 1.25},
        product_schema_id="downstream.scalar_projection",
        product_schema_version="downstream.scalar_projection.v1",
        role="peak_velocity_scalar",
        logical_name="peak_velocity_scalar",
    )
    payload_path = tmp_path / product.artifacts[0].metadata["local_relative_path"]
    payload_path.write_bytes(b'{"value": 9.5}\n')

    with pytest.raises(ArtifactBlobIntegrityError, match="artifact size mismatch|sha256 mismatch"):
        context.finalize()

    assert context.manifest_path is None


def test_analysis_context_finalizes_without_data_products(tmp_path: Path) -> None:
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(analysis_type="toy_zero_products"),
        root=tmp_path,
        index_manifest=False,
    )

    manifest, path = context.finalize()

    assert context.produced_data == ()
    assert manifest.produced_data == []
    assert load_manifest(path).produced_data == []


def test_context_materializer_records_embedded_refs_groups_and_regeneration_specs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    spec = AnalysisRunSpec(
        analysis_type="toy_context_materializer_rich",
        inputs=[ParentRef(kind="EvaluationRunManifest", id="eval-rich")],
    )
    context = AnalysisRunContext(spec=spec, root=tmp_path)

    def materialize(run_context: AnalysisRunContext) -> MaterializationResult:
        existing_path = run_context.results_cache_dir / "existing-summary.json"
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text('{"summary": true}\n', encoding="utf-8")

        group_dir = run_context.results_cache_dir / "bulk"
        group_dir.mkdir(parents=True, exist_ok=True)
        bulk_path = group_dir / "unit.npz"
        np.savez_compressed(bulk_path, values=np.asarray([1, 2, 3], dtype=np.int64))

        external_ref = ArtifactRef(
            role="downstream_existing_ref",
            logical_name="external/ref.json",
            artifact_id="artifact://external/ref",
            media_type="application/json",
            uri="https://example.invalid/ref.json",
        )
        regeneration = RegenerationSpec(
            command=RegenerationCommand(argv=["python", "make_payload.py"]),
            parameters={"analysis_type": run_context.spec.analysis_type},
            inputs=list(run_context.spec.inputs),
            outputs=[
                ArtifactRef(
                    role="toy_materialized_payload",
                    logical_name="toy/rich-materialized.json",
                )
            ],
        )
        return MaterializationResult(
            payload={
                "kind": "toy.rich-materialized.v1",
                "nested": {
                    "refs": [external_ref],
                },
            },
            payload_metadata={"payload_schema": "toy.rich-materialized.v1"},
            existing_artifacts=[
                ExistingAnalysisArtifact(
                    path=existing_path,
                    role="toy_existing_summary",
                    logical_name="toy/existing-summary.json",
                    media_type="application/json",
                )
            ],
            artifact_groups=[
                AnalysisArtifactGroup(
                    group_id="toy_bulk_group",
                    metadata={"description": "opaque toy bulk group"},
                    members=[
                        AnalysisArtifactFile(
                            path=bulk_path,
                            role="toy_bulk_arrays",
                            logical_name="toy/bulk/unit.npz",
                            media_type="application/x-npz",
                            group_role="bulk_arrays",
                            metadata={"arrays": {"values": {"role": "toy_series"}}},
                        )
                    ],
                )
            ],
            regeneration_specs=[regeneration],
        )

    analysis = ContextMaterializer(
        materializer=materialize,
        artifact_role="toy_materialized_payload",
        logical_name="toy/rich-materialized.json",
        schema_boundary="toy-owned payload",
    )
    _all_analyses, all_results, _all_figs = run_analyses_with_context(
        {"materializer": analysis},
        build_toy_analysis_data(),
        context,
    )

    manifest = load_manifest(context.manifest_path)
    assert all_results["materializer"]["nested"]["refs"][0]["artifact_id"] == (
        "artifact://external/ref"
    )
    assert manifest.summary_metrics["artifact_count"] == 4

    artifacts_by_role = {artifact.role: artifact for artifact in manifest.artifacts}
    payload_ref = artifacts_by_role["toy_materialized_payload"]
    external_ref = artifacts_by_role["downstream_existing_ref"]
    existing_ref = artifacts_by_role["toy_existing_summary"]
    bulk_ref = artifacts_by_role["toy_bulk_arrays"]

    assert json.loads(ImmutableArtifactBlobProvider(tmp_path).get_bytes(payload_ref))["nested"][
        "refs"
    ][0]["artifact_id"] == external_ref.artifact_id
    assert Path(existing_ref.uri).exists()
    assert Path(bulk_ref.uri).exists()
    assert bulk_ref.metadata["artifact_group"]["id"] == "toy_bulk_group"
    assert bulk_ref.metadata["artifact_group"]["member_role"] == "bulk_arrays"
    assert bulk_ref.metadata["artifact_group"]["metadata"] == {
        "description": "opaque toy bulk group"
    }
    assert bulk_ref.metadata["arrays"]["values"]["role"] == "toy_series"

    assert len(manifest.regeneration_specs) == 1
    regeneration_payload = manifest.regeneration_specs[0]
    assert regeneration_payload.kind == "RegenerationSpec"
    assert regeneration_payload.schema_id == REGENERATION_SPEC_SCHEMA_ID
    assert regeneration_payload.inline["parameters"] == {
        "analysis_type": "toy_context_materializer_rich"
    }
    assert regeneration_payload.inline["outputs"][0]["logical_name"] == (
        "toy/rich-materialized.json"
    )
