from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.analysis import (
    ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_ID,
    ORDERED_FIGURE_REPORT_PARAMS_SCHEMA_VERSION,
    ORDERED_FIGURE_REPORT_TYPE,
    OrderedFigureReportParams,
)
from feedbax.analysis.reports import (
    ReportRecipeExecutionError,
    execute_report_spec,
    registered_report_types,
)
from feedbax.contracts.manifest import (
    FigureManifest,
    ParentRef,
    ReportSpec,
    load_manifest,
    spec_payload,
    store_bytes_artifact,
    store_json_artifact,
    write_manifest,
)


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
        suffix=suffix,
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
        figure_spec=spec_payload("FigureSpec", {"name": name}),
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


def test_ordered_figure_report_is_public_registered_and_serialisable() -> None:
    params = OrderedFigureReportParams.model_validate(_params())

    assert ORDERED_FIGURE_REPORT_TYPE in registered_report_types()
    assert OrderedFigureReportParams.model_validate_json(params.model_dump_json()) == params
    assert params.model_dump(mode="json")["sections"][0]["tables"][0]["rows"] == [
        ["alpha|beta", True, None],
        ["gamma", False, 2.5],
    ]


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


def test_ordered_figure_report_html_is_self_contained_interactive_and_deterministic(
    tmp_path: Path,
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

    first_manifest, _ = execute_report_spec(spec, root=tmp_path)
    second_manifest, _ = execute_report_spec(spec, root=tmp_path)

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
        execute_report_spec(spec, root=tmp_path)

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "input role 'first' has no materialized plotly-json" in str(
        excinfo.value.__cause__
    )
    assert excinfo.value.manifest.status == "failed"


def test_ordered_figure_report_html_verifies_plotly_json_digest(
    tmp_path: Path,
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
        )

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "Plotly JSON SHA-256 mismatch" in str(excinfo.value.__cause__)


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
