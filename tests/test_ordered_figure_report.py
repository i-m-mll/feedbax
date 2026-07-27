from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.analysis import (
    AnalysisBundleSpec,
    BundleStageSpec,
    ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
    ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
    ORDERED_FIGURE_REPORT_SCALAR_PROJECTION_SCHEMA_ID,
    ORDERED_FIGURE_REPORT_SCALAR_PROJECTION_SCHEMA_VERSION,
    ORDERED_FIGURE_REPORT_TYPE,
    OrderedFigureReportParams,
    OrderedFigureReportScalarProjection,
)
from feedbax.analysis.execution_context import StagedArtifactProviderRootBinding
from feedbax.analysis.specs import find_manifest_by_id
from feedbax.bin import analysis as analysis_cli
from feedbax.contracts.selection import ManifestPredicate
from feedbax.analysis.reports import (
    ReportRecipeExecutionError,
    execute_report_spec,
    registered_report_types,
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
    write_manifest,
)
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.staged_execution import (
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedExecutionDescriptor,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


def _write_figure_manifest(
    root: Path,
    *,
    name: str,
    media_type: str = "image/png",
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
        suffix=suffix,
    )
    manifest = FigureManifest(
        id=f"feedbax-figure:{name}",
        status="completed",
        figure_spec=spec_payload("FigureSpec", {"name": name}),
        artifacts=[artifact],
    )
    write_manifest(manifest, root=root)
    return ParentRef(
        kind="FigureManifest",
        id=manifest.id,
        role=name,
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
                    {"input_role": "first", "caption": "First caption"},
                    {
                        "caption": "Structural panel",
                        "applicability": "not_applicable",
                        "not_applicable_reason": "This binding has no such panel.",
                    },
                    {"input_role": "second", "caption": "Second caption"},
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
        artifact_providers={
            name: ImmutableArtifactBlobProviderSpec() for name in names
        },
        checkpoint_custody={},
    )


def test_ordered_figure_report_is_public_registered_and_serialisable() -> None:
    params = OrderedFigureReportParams.model_validate(_params())

    assert ORDERED_FIGURE_REPORT_TYPE in registered_report_types()
    assert OrderedFigureReportParams.model_validate_json(params.model_dump_json()) == params
    assert params.model_dump(mode="json")["sections"][0]["tables"][0]["rows"] == [
        ["alpha|beta", True, None],
        ["gamma", False, 2.5],
    ]


def test_ordered_figure_report_projects_authenticated_product_scalar(
    tmp_path: Path,
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
        artifact_provider_bindings=[
            StagedArtifactProviderRootBinding("evidence", provider_root)
        ],
    )

    markdown = Path(manifest.artifacts[0].uri or "").read_text(encoding="utf-8")
    assert "| accepted | 7 |" in markdown
    assert manifest.regeneration_specs == [artifact]
    assert manifest.metadata["ordered_figure_report"][
        "scalar_projection_artifact_ids"
    ] == [artifact.artifact_id]


def test_report_inputs_resolve_once_across_distinct_retained_roots(
    tmp_path: Path,
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
            suffix=".png",
        )
        manifest = FigureManifest(
            id=f"feedbax-figure:{name}",
            status="completed",
            figure_spec=spec_payload("FigureSpec", {"name": name}),
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
                            "tables": [
                                {"columns": ["value"], "rows": [[projection]]}
                            ],
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
) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    providers = [ImmutableArtifactBlobProvider(root) for root in roots]
    render = store_bytes_artifact(
        b"figure",
        root=tmp_path / "render",
        role="figure_render",
        logical_name="figure.png",
        media_type="image/png",
        suffix=".png",
    )
    figure = FigureManifest(
        id="feedbax-figure:duplicated",
        status="completed",
        figure_spec=spec_payload("FigureSpec", {"name": "duplicated"}),
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
                                {"input_role": "first", "caption": "First"}
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
        )
    assert message in str(excinfo.value.__cause__)
    assert not (output / "artifacts").exists()


def test_ordered_figure_report_renders_authored_order_roles_and_scalar_tables(
    tmp_path: Path,
) -> None:
    second = _write_figure_manifest(tmp_path, name="second", media_type="text/html")
    first = _write_figure_manifest(tmp_path, name="first")
    spec = ReportSpec(
        report_type=ORDERED_FIGURE_REPORT_TYPE,
        inputs=[second, first],
        params=_params(),
    )

    manifest, path = execute_report_spec(spec, root=tmp_path)

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


def test_ordered_figure_report_missing_required_role_fails_closed(
    tmp_path: Path,
) -> None:
    first = _write_figure_manifest(tmp_path, name="first")
    spec = ReportSpec(
        report_type=ORDERED_FIGURE_REPORT_TYPE,
        inputs=[first],
        params=_params(),
    )

    with pytest.raises(ReportRecipeExecutionError) as excinfo:
        execute_report_spec(spec, root=tmp_path)

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "missing required input roles: 'second'" in str(excinfo.value.__cause__)
    assert excinfo.value.manifest.status == "failed"
    assert load_manifest(excinfo.value.path) == excinfo.value.manifest


def test_ordered_figure_report_explicit_not_applicable_requires_no_input(
    tmp_path: Path,
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

    manifest, _ = execute_report_spec(spec, root=tmp_path)

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
