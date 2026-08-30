from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.analysis import (
    AnalysisBundleSpec,
    BundleStageSpec,
    ORDERED_FIGURE_REPORT_COMPOSITE_SCALAR_SCHEMA_ID,
    ORDERED_FIGURE_REPORT_COMPOSITE_SCALAR_SCHEMA_VERSION,
    ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
    ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
    ORDERED_FIGURE_REPORT_SCALAR_PROJECTION_SCHEMA_ID,
    ORDERED_FIGURE_REPORT_SCALAR_PROJECTION_SCHEMA_VERSION,
    ORDERED_FIGURE_REPORT_TYPE,
    OrderedFigureReportCompositeScalarCell,
    OrderedFigureReportParams,
    OrderedFigureReportScalarProjection,
)
from feedbax.analysis.execution_context import StagedArtifactProviderRootBinding
from feedbax.analysis.exact_parents import (
    STAGED_EXACT_PARENTS_SCHEMA_ID,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION,
    StagedExactParentEntry,
    StagedExactParents,
)
from feedbax.analysis.specs import find_manifest_by_id
from feedbax.bin import analysis as analysis_cli
from feedbax.contracts.selection import ManifestPredicate
from feedbax.analysis.reports import (
    ReportRecipeExecutionError,
    execute_report_spec,
)
from feedbax.contracts.manifest import (
    AnalysisDataProduct,
    AnalysisRunManifest,
    FigureManifest,
    ParentRef,
    ReportSpec,
    canonical_json_bytes,
    load_manifest,
    sha256_bytes,
    spec_payload,
    store_bytes_artifact,
    store_json_artifact,
    write_manifest,
)
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.staged_execution import (
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedCheckpointCustodySpec,
    StagedExecutionDescriptor,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


def _figure_spec_payload(name: str):
    return spec_payload(
        "FigureSpec",
        {"name": name, "assembler": "tests.test_ordered_figure_report:assemble"},
    )


def _figure_spec_sha256(name: str) -> str:
    digest = _figure_spec_payload(name).sha256
    assert digest is not None
    return digest


def _write_figure_manifest(
    root: Path,
    *,
    name: str,
    media_type: str = "image/png",
    retain_plotly_json: bool = False,
) -> ParentRef:
    suffix = {
        "image/png": ".png",
        "text/html": ".html",
    }[media_type]
    artifact = store_bytes_artifact(
        f"{name}-render".encode(),
        root=root,
        role="figure_render",
        logical_name=f"{name}{suffix}",
        media_type=media_type,
    )
    artifacts = [artifact]
    if retain_plotly_json:
        artifacts.append(
            store_json_artifact(
                {
                    "data": [
                        {
                            "type": "scatter",
                            "x": [0, 1],
                            "y": [1, 2],
                            "name": name,
                        }
                    ],
                    "layout": {"title": {"text": f"{name} plot"}},
                },
                root=root,
                role="figure_render",
                logical_name=f"{name}.plotly.json",
                metadata={"figure": name, "format": "plotly-json"},
            )
        )
    manifest = FigureManifest(
        id=f"feedbax-figure:{name}",
        status="completed",
        figure_spec=_figure_spec_payload(name),
        artifacts=artifacts,
    )
    path = write_manifest(manifest, root=root)
    return ParentRef(
        kind="FigureManifest",
        id=manifest.id,
        role=name,
        uri=str(path.relative_to(root)),
    )


def _params() -> dict[str, object]:
    return {
        "schema_id": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
        "schema_version": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
        "title": "Ordered evidence",
        "introduction": "Authored introduction.",
        "sections": [
            {
                "title": "Primary",
                "framing": "Authored framing.",
                "figures": [
                    {
                        "input_role": "first",
                        "figure_spec_sha256": _figure_spec_sha256("first"),
                        "caption": "First caption",
                    },
                    {
                        "caption": "Structural panel",
                        "applicability": "not_applicable",
                        "not_applicable_reason": "This binding has no such panel.",
                    },
                    {
                        "input_role": "second",
                        "figure_spec_sha256": _figure_spec_sha256("second"),
                        "caption": "Second caption",
                    },
                ],
                "tables": [
                    {
                        "title": "Scalars",
                        "columns": ["Name", "Enabled", "Value"],
                        "rows": [["alpha|beta", True, None], ["gamma", False, 2.5]],
                    }
                ],
            },
            {
                "title": "Unavailable structure",
                "applicability": "not_applicable",
                "not_applicable_reason": "The structure is absent by design.",
            },
        ],
    }


def _authenticated_provider_manifest_ref(
    provider: ImmutableArtifactBlobProvider,
    manifest: AnalysisRunManifest | FigureManifest,
    *,
    role: str,
) -> ParentRef:
    raw = canonical_json_bytes(manifest)
    provider.store_bytes(
        raw,
        role="manifest",
        logical_name=f"{manifest.id}.json",
        media_type="application/json",
    )
    return ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role=role,
        metadata={
            "ref_schema_id": "feedbax.ref.authenticated_manifest",
            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
            "manifest_sha256": sha256_bytes(raw),
            "size_bytes": len(raw),
        },
    )


def _provider_descriptor(*names: str) -> StagedExecutionDescriptor:
    return StagedExecutionDescriptor(
        schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        artifact_providers={name: ImmutableArtifactBlobProviderSpec() for name in names},
        checkpoint_custody={},
    )


def _write_exact_manifest_parent(
    root: Path,
    manifest: AnalysisRunManifest | FigureManifest,
    *,
    role: str,
    name: str,
) -> StagedExactParentEntry:
    raw = canonical_json_bytes(manifest)
    relative = Path("exact-inputs") / f"{name}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return StagedExactParentEntry(
        parent=ParentRef(
            kind=manifest.kind,
            id=manifest.id,
            role=role,
            metadata={
                "ref_schema_id": "feedbax.ref.authenticated_manifest",
                "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
                "manifest_sha256": sha256_bytes(raw),
                "size_bytes": len(raw),
            },
        ),
        execution_uri=relative.as_posix(),
    )


def test_ordered_figure_report_is_public_registered_and_serialisable(
    application_registry_bundle,
) -> None:
    params = OrderedFigureReportParams.model_validate(_params())

    assert ORDERED_FIGURE_REPORT_TYPE in application_registry_bundle.report_recipes.keys()
    assert OrderedFigureReportParams.model_validate_json(params.model_dump_json()) == params
    assert params.model_dump(mode="json")["sections"][0]["tables"][0]["rows"] == [
        ["alpha|beta", True, None],
        ["gamma", False, 2.5],
    ]


def test_ordered_figure_report_requires_authored_figure_spec_pin() -> None:
    payload = _params()
    del payload["sections"][0]["figures"][0]["figure_spec_sha256"]  # type: ignore[index]

    with pytest.raises(ValidationError, match="requires figure_spec_sha256"):
        OrderedFigureReportParams.model_validate(payload)


def test_composite_scalar_cell_is_public_versioned_and_serialisable() -> None:
    projection = OrderedFigureReportScalarProjection(
        input_role="summary",
        product_role="result",
        product_schema_id="science.result",
        product_schema_version="science.result.v1",
        artifact_role="result",
        artifact_provider="evidence",
        path=["value"],
    )
    cell = OrderedFigureReportCompositeScalarCell(
        format="{value:.3g}",
        projections={"value": projection},
    )

    assert cell.schema_id == ORDERED_FIGURE_REPORT_COMPOSITE_SCALAR_SCHEMA_ID
    assert cell.schema_version == ORDERED_FIGURE_REPORT_COMPOSITE_SCALAR_SCHEMA_VERSION
    assert (
        OrderedFigureReportCompositeScalarCell.model_validate_json(cell.model_dump_json()) == cell
    )


@pytest.mark.parametrize(
    "authored_format",
    ["{missing:.3g}", "{value:.2f}", "{value:.18g}", "{value!r:.3g}"],
)
def test_composite_scalar_cell_rejects_unsafe_or_non_significant_formats(
    authored_format: str,
) -> None:
    projection = OrderedFigureReportScalarProjection(
        input_role="summary",
        product_role="result",
        product_schema_id="science.result",
        product_schema_version="science.result.v1",
        artifact_role="result",
        artifact_provider="evidence",
        path=["value"],
    )

    with pytest.raises(ValidationError):
        OrderedFigureReportCompositeScalarCell(
            format=authored_format,
            projections={"value": projection},
        )


def test_authored_report_cli_executes_exact_parents_without_reauthoring(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "report-root"
    provider_root = tmp_path / "provider"
    checkpoint_root = tmp_path / "checkpoints"
    root.mkdir()
    checkpoint_root.mkdir()
    provider = ImmutableArtifactBlobProvider(provider_root)

    plotly_json = store_json_artifact(
        {
            "data": [{"type": "scatter", "x": [0, 1], "y": [1, 3]}],
            "layout": {"title": {"text": "Exact authored figure"}},
        },
        root=root,
        role="figure_render",
        logical_name="exact.plotly.json",
        metadata={"figure": "exact", "format": "plotly-json"},
    )
    figure = FigureManifest(
        id="feedbax-figure:exact-authored",
        status="completed",
        figure_spec=_figure_spec_payload("exact-authored"),
        artifacts=[plotly_json],
    )
    figure_entry = _write_exact_manifest_parent(
        root,
        figure,
        role="figure",
        name="figure",
    )

    scalar_artifact = provider.store_bytes(
        canonical_json_bytes({"metrics": {"accepted": 11}}),
        role="compact_result",
        logical_name="compact-result.json",
        media_type="application/json",
    )
    product = AnalysisDataProduct(
        product_schema_id="science.compact_result",
        product_schema_version="science.compact_result.v1",
        role="summary",
        logical_name="summary",
        producer_manifest_id="feedbax-analysis-run:exact-authored",
        artifacts=[scalar_artifact],
    )
    analysis = AnalysisRunManifest(
        id=product.producer_manifest_id,
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "science.compact_summary"},
        ),
        produced_data=[product],
    )
    analysis_entry = _write_exact_manifest_parent(
        root,
        analysis,
        role="summary",
        name="analysis",
    )
    terminal_plotly_json = store_json_artifact(
        {
            "data": [{"type": "scatter", "x": [0, 1], "y": [2, 4]}],
            "layout": {"title": {"text": "Terminal exact figure"}},
        },
        root=root,
        role="figure_render",
        logical_name="terminal.plotly.json",
        metadata={"figure": "terminal", "format": "plotly-json"},
    )
    terminal_entry = _write_exact_manifest_parent(
        root,
        FigureManifest(
            id="feedbax-figure:terminal-extension",
            status="completed",
            figure_spec=_figure_spec_payload("terminal-extension"),
            artifacts=[terminal_plotly_json],
        ),
        role="terminal_figure",
        name="terminal",
    )
    projection = OrderedFigureReportScalarProjection(
        schema_id=ORDERED_FIGURE_REPORT_SCALAR_PROJECTION_SCHEMA_ID,
        schema_version=ORDERED_FIGURE_REPORT_SCALAR_PROJECTION_SCHEMA_VERSION,
        input_role="summary",
        product_role="summary",
        product_schema_id="science.compact_result",
        product_schema_version="science.compact_result.v1",
        artifact_role="compact_result",
        artifact_provider="evidence",
        path=["metrics", "accepted"],
    )
    spec = ReportSpec(
        report_type=ORDERED_FIGURE_REPORT_TYPE,
        inputs=[figure_entry.parent, analysis_entry.parent],
        params={
            "schema_id": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
            "schema_version": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
            "title": "Exact authored report",
            "output_name": "exact-authored.html",
            "sections": [
                {
                    "title": "Evidence",
                    "figures": [
                        {
                            "input_role": "figure",
                            "figure_spec_sha256": _figure_spec_sha256("exact-authored"),
                            "caption": "Authored exact figure",
                        },
                        {
                            "input_role": "terminal_figure",
                            "figure_spec_sha256": _figure_spec_sha256("terminal-extension"),
                            "caption": "Terminal exact figure",
                        },
                    ],
                    "tables": [
                        {
                            "columns": ["metric", "value"],
                            "rows": [["accepted", projection.model_dump(mode="json")]],
                        }
                    ],
                }
            ],
        },
    )
    spec_path = tmp_path / "report.json"
    spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    exact = StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=[figure_entry, analysis_entry, terminal_entry],
    )
    exact_path = tmp_path / "exact-parents.json"
    exact_path.write_text(exact.model_dump_json(indent=2), encoding="utf-8")
    descriptor = StagedExecutionDescriptor(
        schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        artifact_providers={"evidence": ImmutableArtifactBlobProviderSpec()},
        checkpoint_custody={
            "capture": StagedCheckpointCustodySpec(backend="feedbax-checkpoint-transaction-tree")
        },
    )
    descriptor_path = tmp_path / "execution.json"
    descriptor_path.write_text(descriptor.model_dump_json(indent=2), encoding="utf-8")

    analysis_cli.main(
        [
            "report",
            str(spec_path),
            "--exact-parents",
            str(exact_path),
            "--root",
            str(root),
            "--execution-descriptor",
            str(descriptor_path),
            "--artifact-provider",
            f"evidence={provider_root}",
            "--checkpoint-custody",
            f"capture={checkpoint_root}",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    manifest = load_manifest(payload["manifest_path"])
    assert manifest.report_spec.inline["params"] == spec.params
    assert manifest.inputs == [entry.parent for entry in exact.parents]
    rendered = Path(manifest.artifacts[0].uri or "").read_text(encoding="utf-8")
    assert "Authored exact figure" in rendered
    assert "Terminal exact figure" in rendered
    assert "<td>11</td>" in rendered
    assert rendered.count("plotly.js v") == 1


@pytest.mark.parametrize(
    ("output_name", "expected_render"),
    [
        ("projected-evidence.md", "| accepted | 7 |"),
        ("projected-evidence.html", "<td>7</td>"),
    ],
)
def test_ordered_figure_report_projects_authenticated_product_scalar(
    tmp_path: Path,
    output_name: str,
    expected_render: str,
    application_registry_bundle,
) -> None:
    provider_root = tmp_path / "retained"
    provider = ImmutableArtifactBlobProvider(provider_root)
    payload = {
        "schema_id": "science.compact_result",
        "schema_version": "science.compact_result.v1",
        "metrics": {"accepted": 7, "ratio": 0.25},
    }
    artifact = provider.store_bytes(
        canonical_json_bytes(payload),
        role="compact_result",
        logical_name="compact-result.json",
        media_type="application/json",
    )
    auxiliary = provider.store_bytes(
        canonical_json_bytes({"note": "not the projected result"}),
        role="auxiliary",
        logical_name="auxiliary.json",
        media_type="application/json",
    )
    product = AnalysisDataProduct(
        product_schema_id="science.compact_result",
        product_schema_version="science.compact_result.v1",
        role="stability_summary",
        logical_name="stability-summary",
        producer_manifest_id="feedbax-analysis-run:summary",
        artifacts=[auxiliary, artifact],
    )
    analysis = AnalysisRunManifest(
        id=product.producer_manifest_id,
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "science.compact_summary"},
        ),
        produced_data=[product],
    )
    parent = _authenticated_provider_manifest_ref(
        provider,
        analysis,
        role="summary",
    )
    projection = OrderedFigureReportScalarProjection(
        schema_id=ORDERED_FIGURE_REPORT_SCALAR_PROJECTION_SCHEMA_ID,
        schema_version=ORDERED_FIGURE_REPORT_SCALAR_PROJECTION_SCHEMA_VERSION,
        input_role="summary",
        product_role="stability_summary",
        product_schema_id="science.compact_result",
        product_schema_version="science.compact_result.v1",
        artifact_role="compact_result",
        artifact_provider="evidence",
        path=["metrics", "accepted"],
    )
    spec = ReportSpec(
        report_type=ORDERED_FIGURE_REPORT_TYPE,
        inputs=[parent],
        params={
            "schema_id": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
            "schema_version": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
            "title": "Projected evidence",
            "output_name": output_name,
            "sections": [
                {
                    "title": "Scalars",
                    "tables": [
                        {
                            "columns": ["metric", "value"],
                            "rows": [["accepted", projection.model_dump(mode="json")]],
                        }
                    ],
                }
            ],
        },
    )

    manifest, _ = execute_report_spec(
        spec,
        root=tmp_path / "output",
        execution_descriptor=_provider_descriptor("evidence"),
        artifact_provider_bindings=[StagedArtifactProviderRootBinding("evidence", provider_root)],
        registry=application_registry_bundle.report_recipes,
    )

    rendered = Path(manifest.artifacts[0].uri or "").read_text(encoding="utf-8")
    assert expected_render in rendered
    assert manifest.regeneration_specs == [artifact]
    assert manifest.metadata["ordered_figure_report"]["scalar_projection_artifact_ids"] == [
        artifact.artifact_id
    ]


def _composite_scalar_report(
    tmp_path: Path,
    *,
    standard_deviation: object = 0.00804,
) -> tuple[ReportSpec, Path, list[str]]:
    provider_root = tmp_path / "retained"
    provider = ImmutableArtifactBlobProvider(provider_root)
    parents: list[ParentRef] = []
    projections: dict[str, OrderedFigureReportScalarProjection] = {}
    artifact_ids: list[str] = []
    for name, value in (("mean", 0.7351), ("standard_deviation", standard_deviation)):
        artifact = provider.store_bytes(
            canonical_json_bytes({"value": value}),
            role="scalar_result",
            logical_name=f"{name}.json",
            media_type="application/json",
        )
        artifact_ids.append(artifact.artifact_id)
        product = AnalysisDataProduct(
            product_schema_id="science.scalar_result",
            product_schema_version="science.scalar_result.v1",
            role=name,
            logical_name=name,
            producer_manifest_id=f"feedbax-analysis-run:{name}",
            artifacts=[artifact],
        )
        manifest = AnalysisRunManifest(
            id=product.producer_manifest_id,
            status="completed",
            analysis_spec=spec_payload(
                "AnalysisRunSpec",
                {"analysis_type": "science.scalar_summary"},
            ),
            produced_data=[product],
        )
        parents.append(_authenticated_provider_manifest_ref(provider, manifest, role=name))
        projections[name] = OrderedFigureReportScalarProjection(
            input_role=name,
            product_role=name,
            product_schema_id="science.scalar_result",
            product_schema_version="science.scalar_result.v1",
            artifact_role="scalar_result",
            artifact_provider="evidence",
            path=["value"],
        )
    cell = OrderedFigureReportCompositeScalarCell(
        schema_id=ORDERED_FIGURE_REPORT_COMPOSITE_SCALAR_SCHEMA_ID,
        schema_version=ORDERED_FIGURE_REPORT_COMPOSITE_SCALAR_SCHEMA_VERSION,
        format="{mean:.3g} ± {standard_deviation:.1g}",
        projections=projections,
    )
    spec = ReportSpec(
        report_type=ORDERED_FIGURE_REPORT_TYPE,
        inputs=parents,
        params={
            "schema_id": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
            "schema_version": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
            "title": "Composite scalars",
            "output_name": "composite-scalars.html",
            "sections": [
                {
                    "title": "Scalars",
                    "tables": [
                        {
                            "columns": ["metric", "value"],
                            "rows": [["score", cell.model_dump(mode="json")]],
                        }
                    ],
                }
            ],
        },
    )
    return spec, provider_root, artifact_ids


def test_ordered_figure_report_composes_formatted_authenticated_scalars(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    spec, provider_root, artifact_ids = _composite_scalar_report(tmp_path)

    manifest, _ = execute_report_spec(
        spec,
        root=tmp_path / "output",
        execution_descriptor=_provider_descriptor("evidence"),
        artifact_provider_bindings=[StagedArtifactProviderRootBinding("evidence", provider_root)],
        registry=application_registry_bundle.report_recipes,
    )

    rendered = Path(manifest.artifacts[0].uri or "").read_text(encoding="utf-8")
    assert "<td>0.735 ± 0.008</td>" in rendered
    assert {artifact.artifact_id for artifact in manifest.regeneration_specs} == set(artifact_ids)
    assert manifest.inputs == spec.inputs
    assert manifest.metadata["ordered_figure_report"]["scalar_projection_artifact_ids"] == sorted(
        artifact_ids
    )


def test_composite_projection_mapping_order_is_manifest_invariant(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    spec, provider_root, _ = _composite_scalar_report(tmp_path)
    reversed_params = json.loads(json.dumps(spec.params))
    projections = reversed_params["sections"][0]["tables"][0]["rows"][0][1]["projections"]
    reversed_params["sections"][0]["tables"][0]["rows"][0][1]["projections"] = dict(
        reversed(list(projections.items()))
    )
    reversed_spec = spec.model_copy(update={"params": reversed_params})
    kwargs = {
        "execution_descriptor": _provider_descriptor("evidence"),
        "artifact_provider_bindings": [
            StagedArtifactProviderRootBinding("evidence", provider_root)
        ],
    }

    first, _ = execute_report_spec(
        spec, root=tmp_path / "first", **kwargs, registry=application_registry_bundle.report_recipes
    )
    second, _ = execute_report_spec(
        reversed_spec,
        root=tmp_path / "second",
        **kwargs,
        registry=application_registry_bundle.report_recipes,
    )

    assert first.id == second.id
    assert first.artifacts[0].sha256 == second.artifacts[0].sha256
    assert first.regeneration_specs == second.regeneration_specs
    assert first.metadata["ordered_figure_report"] == second.metadata["ordered_figure_report"]


@pytest.mark.parametrize(
    ("missing_role", "standard_deviation", "expected"),
    [
        ("standard_deviation", 0.00804, "missing required input roles"),
        (None, {"nested": 0.00804}, "resolved non-scalar dict"),
    ],
)
def test_ordered_figure_report_composite_scalar_fails_closed(
    tmp_path: Path,
    missing_role: str | None,
    standard_deviation: object,
    expected: str,
    application_registry_bundle,
) -> None:
    spec, provider_root, _ = _composite_scalar_report(
        tmp_path,
        standard_deviation=standard_deviation,
    )
    if missing_role is not None:
        spec = spec.model_copy(
            update={"inputs": [parent for parent in spec.inputs if parent.role != missing_role]}
        )

    with pytest.raises(ReportRecipeExecutionError) as excinfo:
        execute_report_spec(
            spec,
            root=tmp_path / "output",
            execution_descriptor=_provider_descriptor("evidence"),
            artifact_provider_bindings=[
                StagedArtifactProviderRootBinding("evidence", provider_root)
            ],
            registry=application_registry_bundle.report_recipes,
        )

    assert expected in str(excinfo.value.__cause__)
    assert excinfo.value.manifest.status == "failed"
    assert excinfo.value.manifest.artifacts == []


def test_ordered_figure_report_single_value_cell_absence_parity(
    tmp_path: Path, application_registry_bundle
) -> None:
    spec, provider_root, _ = _composite_scalar_report(tmp_path)
    single_projection = next(
        cell
        for cell in OrderedFigureReportParams.model_validate(spec.params)
        .sections[0]
        .tables[0]
        .rows[0]
        if isinstance(cell, OrderedFigureReportCompositeScalarCell)
    ).projections["mean"]
    params = dict(spec.params)
    params["sections"] = [
        {
            "title": "Scalars",
            "tables": [
                {
                    "columns": ["metric", "value"],
                    "rows": [["score", single_projection.model_dump(mode="json")]],
                }
            ],
        }
    ]
    params["output_name"] = "single-scalar.md"

    manifest, _ = execute_report_spec(
        ReportSpec(
            report_type=ORDERED_FIGURE_REPORT_TYPE,
            inputs=[parent for parent in spec.inputs if parent.role == "mean"],
            params=params,
        ),
        root=tmp_path / "output",
        execution_descriptor=_provider_descriptor("evidence"),
        artifact_provider_bindings=[StagedArtifactProviderRootBinding("evidence", provider_root)],
        registry=application_registry_bundle.report_recipes,
    )

    rendered = Path(manifest.artifacts[0].uri or "").read_text(encoding="utf-8")
    assert rendered == (
        "# Composite scalars\n\n"
        "## Scalars\n\n"
        "| metric | value |\n"
        "| --- | --- |\n"
        "| score | 0.7351 |\n"
    )


def test_report_inputs_resolve_once_across_distinct_retained_roots(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    refs: list[ParentRef] = []
    bindings: list[StagedArtifactProviderRootBinding] = []
    for name in ("first", "second"):
        root = tmp_path / name
        provider = ImmutableArtifactBlobProvider(root)
        render = store_bytes_artifact(
            name.encode(),
            root=tmp_path / "renders",
            role="figure_render",
            logical_name=f"{name}.png",
            media_type="image/png",
        )
        manifest = FigureManifest(
            id=f"feedbax-figure:{name}",
            status="completed",
            figure_spec=_figure_spec_payload(name),
            artifacts=[render],
        )
        refs.append(_authenticated_provider_manifest_ref(provider, manifest, role=name))
        bindings.append(StagedArtifactProviderRootBinding(name, root))
    spec = ReportSpec(
        report_type=ORDERED_FIGURE_REPORT_TYPE,
        inputs=refs,
        params=_params(),
    )

    manifest, _ = execute_report_spec(
        spec,
        root=tmp_path / "output",
        execution_descriptor=_provider_descriptor("first", "second"),
        artifact_provider_bindings=bindings,
        registry=application_registry_bundle.report_recipes,
    )

    assert manifest.status == "completed"
    assert not (tmp_path / "output" / "manifests" / "FigureManifest").exists()


def test_staged_bundle_cli_preserves_report_projection_authority(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outputs"
    provider_root = tmp_path / "provider"
    provider = ImmutableArtifactBlobProvider(provider_root)
    artifact = provider.store_bytes(
        canonical_json_bytes({"metrics": [3]}),
        role="compact",
        logical_name="compact.json",
        media_type="application/json",
    )
    product = AnalysisDataProduct(
        product_schema_id="science.compact",
        product_schema_version="science.compact.v1",
        role="summary",
        logical_name="summary",
        producer_manifest_id="feedbax-analysis-run:cli",
        artifacts=[artifact],
    )
    write_manifest(
        AnalysisRunManifest(
            id=product.producer_manifest_id,
            status="completed",
            analysis_spec=spec_payload(
                "AnalysisRunSpec",
                {"analysis_type": "science.summary"},
            ),
            produced_data=[product],
        ),
        root=root,
    )
    projection = {
        "kind": "custody_projection",
        "input_role": "analysis_run",
        "product_role": "summary",
        "product_schema_id": "science.compact",
        "product_schema_version": "science.compact.v1",
        "artifact_role": "compact",
        "artifact_provider": "evidence",
        "path": ["metrics", 0],
    }
    bundle = AnalysisBundleSpec(
        name="report-cli",
        predicate=ManifestPredicate(manifest_kind="AnalysisRunManifest"),
        stages=[
            BundleStageSpec(
                name="report",
                kind="report",
                include_bundle_inputs=True,
                report_type=ORDERED_FIGURE_REPORT_TYPE,
                local_params={
                    "schema_id": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
                    "schema_version": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
                    "title": "CLI report",
                    "sections": [
                        {
                            "title": "Table",
                            "tables": [{"columns": ["value"], "rows": [[projection]]}],
                        }
                    ],
                },
            )
        ],
    )
    descriptor_path = tmp_path / "execution.json"
    descriptor_path.write_text(
        _provider_descriptor("evidence").model_dump_json(indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr(analysis_cli, "load_analysis_bundle", lambda *_args, **_kwargs: bundle)

    analysis_cli.main(
        [
            "--bundle",
            "test/report-cli",
            "--manifest-root",
            str(root),
            "--execution-descriptor",
            str(descriptor_path),
            "--artifact-provider",
            f"evidence={provider_root}",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert result["stages"][0]["status"] == "materialized"
    report, _ = find_manifest_by_id(
        result["stages"][0]["manifest_refs"][0]["id"],
        root=root,
    )
    assert "| 3 |" in Path(report.artifacts[0].uri or "").read_text(encoding="utf-8")


def test_report_input_duplicated_across_retained_roots_fails_before_outputs(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    providers = [ImmutableArtifactBlobProvider(root) for root in roots]
    render = store_bytes_artifact(
        b"figure",
        root=tmp_path / "render",
        role="figure_render",
        logical_name="figure.png",
        media_type="image/png",
    )
    figure = FigureManifest(
        id="feedbax-figure:duplicated",
        status="completed",
        figure_spec=_figure_spec_payload("duplicated"),
        artifacts=[render],
    )
    refs = [
        _authenticated_provider_manifest_ref(provider, figure, role="first")
        for provider in providers
    ]
    assert refs[0] == refs[1]
    output = tmp_path / "output"
    write_manifest(figure, root=output)

    with pytest.raises(ValueError, match="duplicated across authorities"):
        execute_report_spec(
            ReportSpec(
                report_type=ORDERED_FIGURE_REPORT_TYPE,
                inputs=[refs[0]],
                params={
                    **_params(),
                    "sections": [
                        {
                            "title": "One",
                            "figures": [
                                {
                                    "input_role": "first",
                                    "figure_spec_sha256": _figure_spec_sha256("duplicated"),
                                    "caption": "First",
                                }
                            ],
                        }
                    ],
                },
            ),
            root=output,
            execution_descriptor=_provider_descriptor("first", "second"),
            artifact_provider_bindings=[
                StagedArtifactProviderRootBinding(name, root)
                for name, root in zip(("first", "second"), roots, strict=True)
            ],
            registry=application_registry_bundle.report_recipes,
        )
    assert not (output / "manifests" / "reports").exists()
    assert not (output / "artifacts").exists()


@pytest.mark.parametrize(
    ("path", "artifact_role", "duplicate_artifact", "message"),
    [
        (["metrics", "missing"], "compact_result", False, "missing mapping key"),
        (["metrics"], "compact_result", False, "resolved non-scalar"),
        (["metrics", "accepted"], "missing", False, "resolved 0 JSON artifacts"),
        (["metrics", "accepted"], "compact_result", True, "resolved 2 JSON artifacts"),
    ],
)
def test_scalar_projection_path_never_substitutes_missing_or_structured_values(
    tmp_path: Path,
    path: list[str | int],
    artifact_role: str,
    duplicate_artifact: bool,
    message: str,
    application_registry_bundle,
) -> None:
    provider_root = tmp_path / "retained"
    provider = ImmutableArtifactBlobProvider(provider_root)
    artifact = provider.store_bytes(
        canonical_json_bytes({"metrics": {"accepted": 7}}),
        role="compact_result",
        logical_name="compact-result.json",
        media_type="application/json",
    )
    product = AnalysisDataProduct(
        product_schema_id="science.compact_result",
        product_schema_version="science.compact_result.v1",
        role="summary",
        logical_name="summary",
        producer_manifest_id="feedbax-analysis-run:summary",
        artifacts=[
            artifact,
            *(
                [artifact.model_copy(update={"logical_name": "duplicate.json"})]
                if duplicate_artifact
                else []
            ),
        ],
    )
    analysis = AnalysisRunManifest(
        id=product.producer_manifest_id,
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "science.compact_summary"},
        ),
        produced_data=[product],
    )
    parent = _authenticated_provider_manifest_ref(provider, analysis, role="analysis")
    output = tmp_path / "output"
    params = {
        "schema_id": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
        "schema_version": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
        "title": "Fail closed",
        "sections": [
            {
                "title": "Scalars",
                "tables": [
                    {
                        "columns": ["value"],
                        "rows": [
                            [
                                {
                                    "kind": "custody_projection",
                                    "input_role": "analysis",
                                    "product_role": "summary",
                                    "product_schema_id": "science.compact_result",
                                    "product_schema_version": "science.compact_result.v1",
                                    "artifact_role": artifact_role,
                                    "artifact_provider": "evidence",
                                    "path": path,
                                }
                            ]
                        ],
                    }
                ],
            }
        ],
    }

    with pytest.raises(ReportRecipeExecutionError) as excinfo:
        execute_report_spec(
            ReportSpec(
                report_type=ORDERED_FIGURE_REPORT_TYPE,
                inputs=[parent],
                params=params,
            ),
            root=output,
            execution_descriptor=_provider_descriptor("evidence"),
            artifact_provider_bindings=[
                StagedArtifactProviderRootBinding("evidence", provider_root)
            ],
            registry=application_registry_bundle.report_recipes,
        )
    assert message in str(excinfo.value.__cause__)
    assert not (output / "artifacts").exists()


def test_ordered_figure_report_renders_authored_order_roles_and_scalar_tables(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    second = _write_figure_manifest(tmp_path, name="second", media_type="text/html")
    first = _write_figure_manifest(tmp_path, name="first")
    spec = ReportSpec(
        report_type=ORDERED_FIGURE_REPORT_TYPE,
        inputs=[second, first],
        params=_params(),
    )

    manifest, path = execute_report_spec(
        spec, root=tmp_path, registry=application_registry_bundle.report_recipes
    )

    assert load_manifest(path) == manifest
    assert manifest.status == "completed"
    render = manifest.artifacts[0]
    markdown = Path(render.uri or "").read_text(encoding="utf-8")
    authored_markers = [
        "# Ordered evidence",
        "Authored introduction.",
        "## Primary",
        "Authored framing.",
        "First caption",
        "**Structural panel**",
        "This binding has no such panel.",
        "Second caption",
        "### Scalars",
        "alpha\\|beta",
        "## Unavailable structure",
        "The structure is absent by design.",
    ]
    positions = [markdown.index(marker) for marker in authored_markers]
    assert positions == sorted(positions)
    assert "![First caption]" in markdown
    assert "[second.html]" in markdown
    assert "| alpha\\|beta | true | null |" in markdown
    assert manifest.metadata["summary"] == {
        "sections": 2,
        "included_figures": 2,
        "not_applicable_items": 2,
        "scalar_tables": 1,
    }


def test_ordered_figure_report_accepts_exact_embedded_figure_spec_pin(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    provider_root = tmp_path / "provider"
    provider = ImmutableArtifactBlobProvider(provider_root)
    render = store_bytes_artifact(
        b"exact-render",
        root=tmp_path / "render",
        role="figure_render",
        logical_name="exact.png",
        media_type="image/png",
    )
    figure_spec = _figure_spec_payload("exact")
    parent = _authenticated_provider_manifest_ref(
        provider,
        FigureManifest(
            id="feedbax-figure:exact",
            status="completed",
            figure_spec=figure_spec,
            artifacts=[render],
        ),
        role="figure",
    )
    output = tmp_path / "output"

    manifest, _ = execute_report_spec(
        ReportSpec(
            report_type=ORDERED_FIGURE_REPORT_TYPE,
            inputs=[parent],
            params={
                "schema_id": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
                "schema_version": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
                "title": "Pinned figure",
                "sections": [
                    {
                        "title": "Evidence",
                        "figures": [
                            {
                                "input_role": "figure",
                                "figure_spec_sha256": figure_spec.sha256,
                                "caption": "Exact figure",
                            }
                        ],
                    }
                ],
            },
        ),
        root=output,
        execution_descriptor=_provider_descriptor("evidence"),
        artifact_provider_bindings=[StagedArtifactProviderRootBinding("evidence", provider_root)],
        registry=application_registry_bundle.report_recipes,
    )

    assert manifest.status == "completed"
    assert len(manifest.artifacts) == 1


@pytest.mark.parametrize("failure", ["mismatch", "wrong_kind", "malformed"])
def test_ordered_figure_report_rejects_unpinned_embedded_figure_spec_without_output(
    tmp_path: Path,
    failure: str,
    application_registry_bundle,
) -> None:
    provider_root = tmp_path / "provider"
    provider = ImmutableArtifactBlobProvider(provider_root)
    render = store_json_artifact(
        {"data": [], "layout": {}},
        root=tmp_path / "render",
        role="figure_render",
        logical_name="untrusted.plotly.json",
        metadata={"format": "plotly-json"},
    )
    Path(str(render.uri)).unlink()
    if failure == "malformed":
        figure_spec = spec_payload("FigureSpec", {"name": "malformed"})
    elif failure == "wrong_kind":
        figure_spec = spec_payload("AnalysisRunSpec", {"analysis_type": "wrong-kind"})
    else:
        figure_spec = _figure_spec_payload("mismatch")
    parent = _authenticated_provider_manifest_ref(
        provider,
        FigureManifest(
            id=f"feedbax-figure:{failure}",
            status="completed",
            figure_spec=figure_spec,
            artifacts=[render],
        ),
        role="figure",
    )
    expected_sha256 = figure_spec.sha256 if failure != "mismatch" else "0" * 64
    output = tmp_path / "output"

    with pytest.raises(ReportRecipeExecutionError) as excinfo:
        execute_report_spec(
            ReportSpec(
                report_type=ORDERED_FIGURE_REPORT_TYPE,
                inputs=[parent],
                params={
                    "schema_id": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
                    "schema_version": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
                    "title": "Rejected figure",
                    "output_name": "rejected.html",
                    "sections": [
                        {
                            "title": "Evidence",
                            "figures": [
                                {
                                    "input_role": "figure",
                                    "figure_spec_sha256": expected_sha256,
                                    "caption": "Untrusted figure",
                                }
                            ],
                        }
                    ],
                },
            ),
            root=output,
            execution_descriptor=_provider_descriptor("evidence"),
            artifact_provider_bindings=[
                StagedArtifactProviderRootBinding("evidence", provider_root)
            ],
            registry=application_registry_bundle.report_recipes,
        )

    cause = str(excinfo.value.__cause__)
    expected = {
        "malformed": "malformed embedded FigureSpec",
        "wrong_kind": "embedded spec kind 'AnalysisRunSpec'",
        "mismatch": "SHA-256 mismatch",
    }[failure]
    assert expected in cause
    assert excinfo.value.manifest.status == "failed"
    assert excinfo.value.manifest.artifacts == []
    assert not (output / "artifacts").exists()


def test_ordered_figure_report_html_is_self_contained_interactive_and_deterministic(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    second = _write_figure_manifest(
        tmp_path,
        name="second",
        retain_plotly_json=True,
    )
    first = _write_figure_manifest(
        tmp_path,
        name="first",
        retain_plotly_json=True,
    )
    params = _params()
    params["output_name"] = "ordered-evidence.html"
    spec = ReportSpec(
        report_type=ORDERED_FIGURE_REPORT_TYPE,
        inputs=[second, first],
        params=params,
    )

    first_manifest, _ = execute_report_spec(
        spec, root=tmp_path, registry=application_registry_bundle.report_recipes
    )
    second_manifest, _ = execute_report_spec(
        spec, root=tmp_path, registry=application_registry_bundle.report_recipes
    )

    render = first_manifest.artifacts[0]
    report_html = Path(render.uri or "").read_text(encoding="utf-8")
    authored_markers = [
        "<h1>Ordered evidence</h1>",
        "Authored introduction.",
        "<h2>Primary</h2>",
        "Authored framing.",
        "first plot",
        "<strong>Structural panel</strong>",
        "This binding has no such panel.",
        "second plot",
        "<h3>Scalars</h3>",
        "alpha|beta",
        "<h2>Unavailable structure</h2>",
        "The structure is absent by design.",
    ]
    positions = [report_html.index(marker) for marker in authored_markers]
    assert positions == sorted(positions)
    assert render.logical_name == "ordered-evidence.html"
    assert render.media_type == "text/html"
    assert report_html.startswith("<!DOCTYPE html>")
    assert report_html.count("plotly.js v") == 1
    assert "<script src=" not in report_html
    assert report_html.count('class="plotly-graph-div"') == 2
    assert 'id="ordered-figure-1"' in report_html
    assert 'id="ordered-figure-2"' in report_html
    assert "<table>" in report_html
    assert "<td>true</td>" in report_html
    assert "<td>null</td>" in report_html
    assert all(
        str(artifact.uri) not in report_html
        for ref in (first, second)
        for artifact in load_manifest(tmp_path / str(ref.uri)).artifacts
        if artifact.uri is not None
    )
    assert second_manifest.artifacts[0].sha256 == render.sha256


def test_ordered_figure_report_html_requires_retained_plotly_json(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    first = _write_figure_manifest(tmp_path, name="first")
    second = _write_figure_manifest(tmp_path, name="second")
    params = _params()
    params["output_name"] = "ordered-evidence.html"
    spec = ReportSpec(
        report_type=ORDERED_FIGURE_REPORT_TYPE,
        inputs=[first, second],
        params=params,
    )

    with pytest.raises(ReportRecipeExecutionError) as excinfo:
        execute_report_spec(
            spec, root=tmp_path, registry=application_registry_bundle.report_recipes
        )

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "input role 'first' has no materialized plotly-json" in str(excinfo.value.__cause__)
    assert excinfo.value.manifest.status == "failed"


def test_ordered_figure_report_html_verifies_plotly_json_digest(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    first = _write_figure_manifest(
        tmp_path,
        name="first",
        retain_plotly_json=True,
    )
    second = _write_figure_manifest(
        tmp_path,
        name="second",
        retain_plotly_json=True,
    )
    first_manifest = load_manifest(tmp_path / str(first.uri))
    plotly_artifact = next(
        artifact
        for artifact in first_manifest.artifacts
        if artifact.metadata.get("format") == "plotly-json"
    )
    Path(str(plotly_artifact.uri)).write_text("{}", encoding="utf-8")
    params = _params()
    params["output_name"] = "ordered-evidence.html"

    with pytest.raises(ReportRecipeExecutionError) as excinfo:
        execute_report_spec(
            ReportSpec(
                report_type=ORDERED_FIGURE_REPORT_TYPE,
                inputs=[first, second],
                params=params,
            ),
            root=tmp_path,
            registry=application_registry_bundle.report_recipes,
        )

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "Plotly JSON SHA-256 mismatch" in str(excinfo.value.__cause__)


def test_ordered_figure_report_missing_required_role_fails_closed(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    first = _write_figure_manifest(tmp_path, name="first")
    spec = ReportSpec(
        report_type=ORDERED_FIGURE_REPORT_TYPE,
        inputs=[first],
        params=_params(),
    )

    with pytest.raises(ReportRecipeExecutionError) as excinfo:
        execute_report_spec(
            spec, root=tmp_path, registry=application_registry_bundle.report_recipes
        )

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "missing required input roles: 'second'" in str(excinfo.value.__cause__)
    assert excinfo.value.manifest.status == "failed"
    assert load_manifest(excinfo.value.path) == excinfo.value.manifest


def test_ordered_figure_report_explicit_not_applicable_requires_no_input(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    spec = ReportSpec(
        report_type=ORDERED_FIGURE_REPORT_TYPE,
        params={
            "schema_id": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
            "schema_version": ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
            "title": "Structural report",
            "sections": [
                {
                    "title": "Optional structure",
                    "figures": [
                        {
                            "caption": "Optional panel",
                            "applicability": "not_applicable",
                            "not_applicable_reason": "Not part of this binding.",
                        }
                    ],
                }
            ],
        },
    )

    manifest, _ = execute_report_spec(
        spec, root=tmp_path, registry=application_registry_bundle.report_recipes
    )

    markdown = Path(manifest.artifacts[0].uri or "").read_text(encoding="utf-8")
    assert manifest.status == "completed"
    assert "Not applicable: Not part of this binding." in markdown
    assert manifest.metadata["summary"]["included_figures"] == 0


@pytest.mark.parametrize(
    ("remove", "update"),
    [
        ({"schema_id"}, {}),
        ({"schema_version"}, {}),
        (set(), {"schema_version": "feedbax.spec.report.ordered_figure.v0"}),
        (set(), {"schema_version": "feedbax.spec.report.ordered_figure.v1"}),
        (set(), {"schema_version": "feedbax.spec.report.ordered_figure.v2"}),
        (
            set(),
            {
                "sections": [
                    {
                        "title": "Bad table",
                        "tables": [{"columns": ["a", "b"], "rows": [[1]]}],
                    }
                ]
            },
        ),
        (
            set(),
            {
                "sections": [
                    {
                        "title": "Bad applicability",
                        "figures": [
                            {
                                "input_role": "unexpected",
                                "caption": "Panel",
                                "applicability": "not_applicable",
                                "not_applicable_reason": "Absent.",
                            }
                        ],
                    }
                ]
            },
        ),
    ],
)
def test_ordered_figure_report_rejects_invalid_authored_params(
    remove: set[str],
    update: dict[str, object],
) -> None:
    payload = _params()
    for key in remove:
        payload.pop(key)
    payload.update(update)

    with pytest.raises(ValidationError):
        OrderedFigureReportParams.model_validate(payload)


@pytest.mark.parametrize(
    "output_name",
    ["ordered-report.txt", "reports/ordered-report.html", "ordered-report.HTML"],
)
def test_ordered_figure_report_rejects_unsupported_output_name(output_name: str) -> None:
    payload = _params()
    payload["output_name"] = output_name

    with pytest.raises(ValidationError, match="Markdown or HTML filename"):
        OrderedFigureReportParams.model_validate(payload)


def test_ordered_figure_report_html_params_round_trip() -> None:
    payload = _params()
    payload["output_name"] = "ordered-report.html"

    params = OrderedFigureReportParams.model_validate(payload)

    assert OrderedFigureReportParams.model_validate_json(params.model_dump_json()) == params
