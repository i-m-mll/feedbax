"""Render the first walkthrough figure directly from a stored dataset."""

from pathlib import Path

import numpy as np
import plotly.io as pio
import pytest
from pydantic import ValidationError

from feedbax.analysis.builtin_operations import register_builtin_operations
from feedbax.analysis.operations import OperationRegistry
from feedbax.analysis.report_document import render as render_report
from feedbax.contracts.figure_document import (
    FigureDocument,
    FigureLayerOmission,
    PlotlyFigureProduct,
    validate_figure_bindings,
)
from feedbax.contracts.report_document import ReportDocument
from feedbax.contracts.request_refs import ContentRef
from feedbax.dataset import Dataset, Variable
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_dataset, publish_dataset
from feedbax.plot.dataset_figure import render


_FIXTURES = Path(__file__).resolve().parents[1] / "web/e2e/studio/fixtures"


def velocity_summary_figure() -> FigureDocument:
    return FigureDocument.model_validate_json(
        (_FIXTURES / "velocity-summary.figure.json").read_bytes()
    )


def test_velocity_report_references_the_saved_band_with_condition_order_and_physical_time(
    tmp_path,
    monkeypatch,
):
    document = velocity_summary_figure()
    registry = OperationRegistry()
    register_builtin_operations(registry)
    port = registry.resolve("feedbax.summarize_velocity").definition.outputs["summary"]
    validate_figure_bindings(document, {"summary": port})

    time = np.array([0.0, 0.031, 0.11], dtype=np.float64)
    mean = np.array([[[2, 3, 4], [6, 7, 8]], [[30, 40, 50], [60, 70, 80]]])
    lower = mean - [0.2, 0.7, 1.1]
    upper = mean + [1.5, 0.4, 0.9]
    dataset = Dataset.from_arrays(
        {
            name: Variable(values, port.variables[name].dims, units="m/s", frame="reach-aligned")
            for name, values in (("mean", mean), ("lower", lower), ("upper", upper))
        },
        coords={
            "checkpoint": Variable(np.array([31, 7]), ("checkpoint",)),
            "condition": Variable(np.array(["perturbed", "control"]), ("condition",)),
            "state_time": Variable(time, ("state_time",), units="s"),
        },
        schema_id=port.schema_ids[0],
    )
    store = ImmutableArtifactBlobProvider(tmp_path)
    ref = publish_dataset(store, dataset)
    monkeypatch.setattr(pio.templates, "default", "plotly_dark")
    with (
        store.open_owned_object(ref.object_ref) as path,
        open_dataset(path, expected_descriptor_sha256=ref.descriptor_sha256) as reopened,
    ):
        figure = render(document, {"summary": reopened})
    assert pio.templates.default == "plotly_dark"
    assert (
        figure.layout.template.layout.paper_bgcolor
        == pio.templates["plotly_white"].layout.paper_bgcolor
    )
    assert figure.layout.xaxis.title.text == "Time (s)"
    assert figure.layout.yaxis.title.text == "Velocity (m/s)"
    assert len(figure.data) == 6
    centers = figure.data[::3]
    assert [trace.name for trace in centers] == ["perturbed", "control"]
    assert centers[0].line.color != centers[1].line.color
    for index, start in enumerate((0, 3)):
        center, high, low = figure.data[start : start + 3]
        for trace, values in ((center, mean), (high, upper), (low, lower)):
            np.testing.assert_array_equal(trace.x, time)
            np.testing.assert_array_equal(trace.y, values[0, index])
        assert low.fill == "tonexty"
        assert high.showlegend is False and low.showlegend is False

    figure_bytes = figure.to_json().encode()
    saved = store.store_object(figure_bytes)
    content = ContentRef(sha256=saved.sha256, size=saved.size_bytes, media_type="application/json")
    document = ReportDocument.model_validate_json(
        (_FIXTURES / "velocity-summary.report.json").read_bytes()
    )
    result = render_report(document, {"velocity_summary": PlotlyFigureProduct(content=content)})
    ((path, product),) = result.figures.items()
    assert product == content
    assert store.get_object(product.sha256, size_bytes=product.size) == figure_bytes
    assert result.html.count(path) == 1
    assert result.html.count('src="plotly.min.js"') == 1
    assert figure_bytes.decode() not in result.html
    assert result.html.index("<em>reach-aligned velocity</em>") < result.html.index("<figure ")
    assert result.html.index("</figure>") < result.html.index("The band gives the lower")
    assert (
        "Mean velocity and trial bounds by condition.](report.html#report-figure-2)"
        in result.markdown
    )


def test_report_shares_repeated_figure_content_and_checks_omissions_before_rendering():
    document = ReportDocument.model_validate(
        {
            "title": "Two placements &copy;",
            "blocks": [
                {"markdown": "First placement <script>alert(1)</script>."},
                {"figure": "velocity"},
                {"figure": "comparison", "caption": "Comparison &#65; ~~baseline~~"},
            ],
        }
    )
    content = ContentRef(sha256="a" * 64, size=100, media_type="application/json")
    figures = {
        "velocity": PlotlyFigureProduct(content=content),
        "comparison": PlotlyFigureProduct(
            content=content,
            omitted_layers=(FigureLayerOmission(layer="baseline", reason="No matched trials."),),
        ),
    }
    with pytest.raises(ValueError, match=r"comparison/baseline: No matched trials\."):
        render_report(document, figures)

    permissive = document.model_copy(update={"completeness": "allow_omissions"})
    result = render_report(permissive, figures)
    assert len(result.figures) == 1
    assert result.html.count(next(iter(result.figures))) == 1
    assert result.html.count('data-report-source="0"') == 2
    assert "<script>alert(1)</script>" not in result.html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in result.html
    assert "# Two placements \\&copy;" in result.markdown
    assert "[Comparison \\&\\#65; \\~\\~baseline\\~\\~]" in result.markdown
    assert "Omitted layer baseline: No matched trials." in result.html
    assert "Omitted layer baseline: No matched trials." in result.markdown
    for report in (document, permissive):
        with pytest.raises(ValueError, match="Missing saved report figures: comparison"):
            render_report(report, {"velocity": figures["velocity"]})

    with pytest.raises(ValidationError):
        ReportDocument.model_validate(
            {
                "title": "Invalid block",
                "blocks": [{"markdown": "Text", "figure": "velocity"}],
            }
        )


def test_unknown_band_channel_is_refused_against_the_operation_declaration():
    document = velocity_summary_figure().model_dump(mode="json")
    document["layers"][0]["encode"]["upper"] = "missing_upper"
    registry = OperationRegistry()
    register_builtin_operations(registry)
    port = registry.resolve("feedbax.summarize_velocity").definition.outputs["summary"]
    with pytest.raises(ValueError, match="trial-summary.*upper.*missing_upper"):
        validate_figure_bindings(FigureDocument.model_validate(document), {"summary": port})
