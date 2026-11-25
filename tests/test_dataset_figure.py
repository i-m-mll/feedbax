"""Render the first walkthrough figure directly from a stored dataset."""

from hashlib import sha256

import numpy as np
import plotly.io as pio
from plotly.offline import get_plotlyjs
import pytest
from pydantic import ValidationError

from feedbax.builtins.figure_templates import VelocitySummary, VelocitySummarySettings
from feedbax.application.composition import load_application
from feedbax.analysis.report_document import render as render_report
from feedbax.contracts.dataset import DATASET_MEDIA_TYPE, TrialFigureTime
from feedbax.contracts.effective_success import BoundMember, success_content_ref
from feedbax.contracts.figure_document import (
    FigureDocument,
    FigureLayerOmission,
    PlotlyFigureProduct,
    validate_figure_bindings,
)
from feedbax.contracts.project_sweep import SweepCoordinate, SweepInputMember
from feedbax.contracts.report_document import ReportDocument
from feedbax.contracts.request_identity import decode_request, encode_request
from feedbax.contracts.request_refs import ContentRef, InputMember, ProductRef
from feedbax.contracts.request_specs import RenderRequestSpec
from feedbax.contracts.rollout import RAW_STATES_SCHEMA
from feedbax.contracts.studio_figure import FigureProduct
from feedbax.dataset import Dataset, Variable
from feedbax.dataset_sweep import stack_dataset_members
from feedbax.host.light_worker import _compute
from feedbax.host.light_jobs import validate_light_request
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_dataset, publish_dataset
from feedbax.plot.dataset_figure import render


def test_published_speed_figure_keeps_trial_order_and_checkpoint_samples(tmp_path):
    bundle = load_application().bundle
    template = bundle.figure_templates.resolve("feedbax.SpeedByTrial")
    document = template.function(template.settings_model.model_validate({"checkpoint_index": 1}))
    operation = bundle.operations.resolve("feedbax.trial_speed")
    port = operation.value.definition.outputs["speed"]
    validate_figure_bindings(document, {"speed": port})

    time = np.array([0.0, 0.02, 0.07])
    velocity = np.array([
        [[[[0, 0], [1, 0], [2, 0]], [[0, 1], [0, 2], [0, 3]]]],
        [[[[0, 0], [-3, 4], [5, -12]], [[-8, 15], [0, -3], [0, 0]]]],
    ], dtype=np.float32)
    condition = "d" * 64
    trials = ('{"target":[0.2,0]}', '{"target":[0.1,0]}')
    states = Dataset.from_arrays(
        {"velocity": Variable(
            velocity, ("checkpoint", "condition", "trial", "state_time", "xy"), units="m/s", frame="world",
        )},
        coords={
            "checkpoint": Variable(np.array([31, 7]), ("checkpoint",)),
            "condition": Variable(np.array([condition]), ("condition",)),
            "trial": Variable(np.array(trials), ("trial",)),
            "state_time": Variable(time, ("state_time",), units="s", trial_time=TrialFigureTime()),
        },
        schema_id=RAW_STATES_SCHEMA,
    )
    speed = operation.defining_object(states, operation.value.parameter_model.model_validate({"velocity": "velocity"}))
    names = {"condition": {condition: "plant.mass=2"}}
    figure = render(document, {"speed": speed}, names=names)
    assert [trace.name for trace in figure.data] == ["plant.mass=2 · Trial 1", "plant.mass=2 · Trial 2"]
    for trace, samples in zip(figure.data, ([0, 5, 13], [17, 3, 0]), strict=True):
        np.testing.assert_array_equal(trace.x, time)
        np.testing.assert_array_equal(trace.y, samples)

    stacked = stack_dataset_members([
        (SweepInputMember("original", {}), speed), (SweepInputMember("duplicate", {}), speed),
    ])
    duplicate = render(document, {"speed": stacked}, names=names)
    assert [(trace.meta["group"]["member"], trace.meta["group"]["trial"]) for trace in duplicate.data] == [
        (member, trial) for member in ("original", "duplicate") for trial in trials
    ]
    assert [trace.name for trace in duplicate.data] == [
        f"plant.mass=2 · Trial {trial} · Run {run}" for run in (1, 2) for trial in (1, 2)
    ]
    assert len({trace.legendgroup for trace in duplicate.data}) == 4

    layer, = document.layers
    selected = layer.model_copy(update={"select": layer.select | {"trial": trials[1]}})
    single = render(document.model_copy(update={"layers": (selected,)}), {"speed": speed}, names=names)
    trace, = single.data
    assert trace.name == "plant.mass=2 · Trial 2"
    assert trace.meta["group"]["trial"] == trials[1]

    store = ImmutableArtifactBlobProvider(tmp_path)
    ref = publish_dataset(store, speed)
    encoded = encode_request(
        RenderRequestSpec(document=document, names=names),
        {"speed": (InputMember(
            key="speed", reference=ProductRef(request_id="analyze:" + "a" * 64, role="speed"),
        ),)},
    )
    request = decode_request(encoded.data, encoded.documents.__getitem__)
    bindings = {"speed": (BoundMember(
        key="speed", content=ref.content.model_copy(update={"media_type": DATASET_MEDIA_TYPE}),
        descriptor={"descriptor_sha256": ref.descriptor_sha256},
    ),)}
    saved = _compute(store, request, bindings, "render-attempt", *validate_light_request(request))
    product = store.get_object(success_content_ref(saved.content))
    assert product == figure.to_json().encode()
    parsed = FigureProduct.model_validate_json(product)
    panel = parsed.layout.meta.panels["speed"]
    assert panel.model_dump(exclude_none=True) == {"x_axis": "x", "y_axis": "y", "time": {}}
    assert [trace.meta.model_dump() for trace in parsed.data] == [
        {"layer": "trial-speed", "group": {"condition": condition, "trial": trial}}
        for trial in trials
    ]


def test_velocity_report_references_the_saved_band_with_condition_order_and_physical_time(
    tmp_path,
    monkeypatch,
):
    document = VelocitySummary(VelocitySummarySettings())
    registry = load_application().bundle.operations
    port = registry.resolve("feedbax.summarize_velocity").value.definition.outputs["summary"]
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
    assert "time" not in figure.layout.meta["panels"]["velocity"]
    encoded = encode_request(
        RenderRequestSpec(document=document),
        {"summary": (InputMember(
            key="summary", reference=ProductRef(request_id="analyze:" + "a" * 64, role="summary"),
        ),)},
    )
    request = decode_request(encoded.data, encoded.documents.__getitem__)
    bindings = {"summary": (BoundMember(
        key="summary",
        content=ref.content.model_copy(update={"media_type": DATASET_MEDIA_TYPE}),
        descriptor={"descriptor_sha256": ref.descriptor_sha256},
    ),)}
    saved = _compute(store, request, bindings, "render-attempt", *validate_light_request(request))
    content = saved.content
    report = ReportDocument.model_validate({
        "title": "Saved figure",
        "blocks": [
            {"markdown": "*Before figure*"},
            {"figure": "velocity_summary"},
            {"markdown": "*After figure*"},
        ],
    })
    result = render_report(
        report, {"velocity_summary": PlotlyFigureProduct(content=content)},
        plotly_script_sha256=sha256(get_plotlyjs().encode()).hexdigest(),
    )
    ((path, product),) = result.figures.items()
    assert product == content
    assert store.get_object(success_content_ref(content)) == figure_bytes
    assert result.html.count(path) == 1
    assert result.html.count('src="plotly.min.js"') == 1
    assert figure_bytes.decode() not in result.html
    assert result.html.index("<em>Before figure</em>") < result.html.index("<figure ")
    assert result.html.index("</figure>") < result.html.index("<em>After figure</em>")
    assert "](report.html#report-figure-2)" in result.markdown


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
    script_digest = sha256(get_plotlyjs().encode()).hexdigest()
    with pytest.raises(ValueError, match=r"comparison/baseline: No matched trials\."):
        render_report(document, figures, plotly_script_sha256=script_digest)

    permissive = document.model_copy(update={"completeness": "allow_omissions"})
    result = render_report(permissive, figures, plotly_script_sha256=script_digest)
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
            render_report(report, {"velocity": figures["velocity"]}, plotly_script_sha256=script_digest)

    with pytest.raises(ValidationError):
        ReportDocument.model_validate(
            {
                "title": "Invalid block",
                "blocks": [{"markdown": "Text", "figure": "velocity"}],
            }
        )


def test_unknown_band_channel_is_refused_against_the_operation_declaration():
    document = VelocitySummary(VelocitySummarySettings()).model_dump(mode="json")
    document["layers"][0]["encode"]["upper"] = "missing_upper"
    registry = load_application().bundle.operations
    port = registry.resolve("feedbax.summarize_velocity").value.definition.outputs["summary"]
    with pytest.raises(ValueError, match="trial-summary.*upper.*missing_upper"):
        validate_figure_bindings(FigureDocument.model_validate(document), {"summary": port})


def test_sparse_panels_keep_authored_positions_and_share_their_rendered_range():
    values = Dataset.from_arrays(
        {
            "earlier": Variable(np.array([1.0, 3.0]), ("time",)),
            "later": Variable(np.array([7.0, 11.0]), ("time",)),
        },
        coords={"time": Variable(np.array([0.0, 1.0]), ("time",), units="s")},
        schema_id="figure.panel_values.v1",
    )
    range_group = {"derived_range": {"group": "amplitude", "padding_fraction": 0.1}}
    document = FigureDocument.model_validate({
        "data": ["values"],
        "layout": [
            {"name": "later", "row": 1, "col": 3, "y_axis": range_group},
            {
                "name": "earlier",
                "row": 1,
                "col": 1,
                "y_axis": range_group,
                "x_axis": {"range": [-1, 9]},
                "axes_labels": {"x": "Elapsed time"},
            },
        ],
        "layers": [
            {
                "name": name, "role": "values", "panel": name,
                "mark": "line", "encode": {"x": "time", "y": name},
            }
            for name in ("earlier", "later")
        ],
        "view": {
            "shared_yaxes": False,
            "width": 600,
            "margin": {"l": 40, "r": 60},
            "legend_gutter": 50,
        },
    })

    figure = render(document, {"values": values})

    assert [trace.xaxis for trace in figure.data] == ["x3", "x"]
    left, right = figure.layout.annotations
    assert left.text == "earlier" and left.x < 0.5
    assert right.text == "later" and right.x > 0.5
    assert figure.layout.xaxis.title.text == "Elapsed time"
    assert figure.layout.xaxis.range == (-1, 9)
    assert figure.layout.yaxis.range == figure.layout.yaxis3.range == (0, 12)
    assert figure.layout.legend.x == pytest.approx(1.1)


def test_points_rules_and_spans_draw_at_their_authored_positions():
    """Catches error bars drawn as offsets instead of bounds, and references drawn on the wrong axis."""
    time = np.array([0.0, 0.02, 0.04])
    summary = Dataset.from_arrays(
        {
            "mean": Variable(np.array([[0.0, 0.3, 0.5], [0.0, 0.2, 0.4]]), ("condition", "state_time"), units="m/s"),
            "lower": Variable(np.array([[0.0, 0.25, 0.4], [0.0, 0.1, 0.3]]), ("condition", "state_time"), units="m/s"),
            "upper": Variable(np.array([[0.0, 0.4, 0.6], [0.0, 0.3, 0.45]]), ("condition", "state_time"), units="m/s"),
            "onset": Variable(np.array([0.01, 0.015]), ("condition",), units="s"),
        },
        coords={
            "condition": Variable(np.array(["light", "heavy"]), ("condition",)),
            "state_time": Variable(time, ("state_time",), units="s", trial_time=TrialFigureTime()),
        },
    )
    document = FigureDocument.model_validate({
        "data": ["summary"],
        "layout": [{"name": "velocity", "row": 1, "col": 2}, {"name": "spare", "row": 1, "col": 1}],
        "layers": [
            {
                "name": "peak", "mark": "points", "panel": "velocity", "role": "summary",
                "select": {"state_time": {"index": 2}}, "group": ["condition"],
                "encode": {"x": "condition", "y": "mean", "error_y": {"lower": "lower", "upper": "upper"}},
                "style": {"label": "{condition}", "marker_size": 9},
            },
            {"name": "target", "mark": "rule", "panel": "velocity", "role": "summary", "encode": {"axis": "y", "at": 0.5}},
            {
                "name": "onset", "mark": "rule", "panel": "velocity", "role": "summary", "group": ["condition"],
                "encode": {"axis": "x", "at": "onset"},
            },
            {
                "name": "window", "mark": "span", "panel": "velocity", "role": "summary",
                "encode": {"axis": "x", "start": 0.02, "end": 0.04}, "style": {"color": "#000000", "fill_alpha": 0.25},
            },
        ],
    })
    figure = render(document, {"summary": summary})

    assert [(trace.name, list(trace.x), list(trace.y)) for trace in figure.data] == [
        ("light", ["light"], [0.5]), ("heavy", ["heavy"], [0.4]),
    ]
    for trace, (lower, upper) in zip(figure.data, ((0.4, 0.6), (0.3, 0.45)), strict=True):
        assert trace.mode == "markers" and trace.xaxis == "x2"
        np.testing.assert_allclose(trace.y[0] + trace.error_y.array[0], upper)
        np.testing.assert_allclose(trace.y[0] - trace.error_y.arrayminus[0], lower)

    target, light, heavy, window = figure.layout.shapes
    assert (target.type, target.yref, target.y0, target.y1, target.xref, target.x0, target.x1) == (
        "line", "y2", 0.5, 0.5, "x2 domain", 0, 1,
    )
    assert [(shape.xref, shape.x0, shape.yref) for shape in (light, heavy)] == [
        ("x2", 0.01, "y2 domain"), ("x2", 0.015, "y2 domain"),
    ]
    assert (window.type, window.x0, window.x1, window.yref, window.fillcolor) == (
        "rect", 0.02, 0.04, "y2 domain", "rgba(0, 0, 0, 0.25)",
    )
    assert [shape.showlegend for shape in figure.layout.shapes] == [True, True, False, True]


def test_error_bounds_that_exclude_their_point_are_refused():
    """Catches bounds given as offsets (or swapped), which would otherwise draw inverted bars."""
    values = Dataset.from_arrays(
        {name: Variable(np.array([value]), ("condition",), units="m/s")
         for name, value in (("mean", 3.0), ("lower", 0.5), ("upper", 2.0))},
        coords={"condition": Variable(np.array(["control"]), ("condition",))},
    )
    document = FigureDocument.model_validate({
        "data": ["summary"], "layout": [{"name": "panel", "row": 1, "col": 1}],
        "layers": [{
            "name": "estimate", "role": "summary", "panel": "panel", "mark": "points", "group": ["condition"],
            "encode": {"x": "condition", "y": "mean", "error_y": {"lower": "lower", "upper": "upper"}},
        }],
    })
    with pytest.raises(ValueError, match="enclose their point"):
        render(document, {"summary": values})


def test_trajectory_scales_preserve_spatial_samples_and_group_values():
    hand = np.array([[[0, 0], [1, 2], [2, 5]], [[0, 0], [2, 2], [3, 6]]])
    pca = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[10, 20, 30], [9, 8, 7], [5, 3, 1]]])
    data = stack_dataset_members([
        (
            SweepInputMember(name, {"mass": SweepCoordinate(name, value=mass)}),
            Dataset.from_arrays(
                {
                    "hand": Variable(hand[index], ("sample", "xy"), units="m"),
                    "embedding": Variable(pca[index], ("sample", "pc")),
                },
                coords={
                    "xy": Variable(np.array(["x", "y"]), ("xy",)),
                    "pc": Variable(np.array(["PC 1", "PC 2", "PC 3"]), ("pc",)),
                    "time": Variable(np.array([0, 0.01, 0.02]), ("sample",), trial_time=TrialFigureTime()),
                },
            ),
        )
        for index, (name, mass) in enumerate((("nominal", 0.5), ("loaded", 2.0)))
    ])
    document = FigureDocument.model_validate({
        "data": ["trials"],
        "layout": [
            {"name": "hand", "row": 1, "col": 1},
            {"name": "pca", "row": 1, "col": 2, "z_axis": {"range": [-1, 31]},
             "axes_labels": {"x": "PC 1", "y": "PC 2", "z": "PC 3"}},
        ],
        "color_scales": {
            "mass": {"kind": "continuous", "palette": "cividis", "title": "Arm mass (kg)"},
            "condition": {"kind": "ordinal", "palette": "category10", "domain": ["loaded", "nominal"]},
        },
        "layers": [
            {"name": "reach", "role": "trials", "panel": "hand", "group": ["member"], "mark": "trajectory",
             "encode": {
                 "x": {"variable": "hand", "select": {"xy": "x"}},
                 "y": {"variable": "hand", "select": {"xy": "y"}},
                 "color": {"field": "mass.value", "scale": "mass"},
             }},
            {"name": "hidden", "role": "trials", "panel": "pca", "group": ["member"], "mark": "trajectory",
             "encode": {
                 **{axis: {"variable": "embedding", "select": {"pc": name}}
                    for axis, name in zip("xyz", ("PC 1", "PC 2", "PC 3"), strict=True)},
                 "color": {"field": "member", "scale": "condition"},
             }},
        ],
    })
    result = render(document, {"trials": data}, names={"member": {"nominal": "Nominal", "loaded": "Loaded"}})
    hand_traces, pca_traces, guides = result.data[:2], result.data[2:4], result.data[4:]
    assert [trace.type for trace in (*hand_traces, *pca_traces)] == ["scatter", "scatter", "scatter3d", "scatter3d"]
    for index, (reach, hidden) in enumerate(zip(hand_traces, pca_traces, strict=True)):
        np.testing.assert_array_equal(reach.x, hand[index, :, 0])
        np.testing.assert_array_equal(reach.y, hand[index, :, 1])
        for component, axis in enumerate("xyz"):
            np.testing.assert_array_equal(getattr(hidden, axis), pca[index, :, component])
    assert hand_traces[0].line.color != hand_traces[1].line.color
    assert pca_traces[0].line.color != pca_traces[1].line.color
    assert result.layout.coloraxis.colorbar.title.text == "Arm mass (kg)"
    assert (result.layout.coloraxis.cmin, result.layout.coloraxis.cmax) == (0.5, 2.0)
    assert result.layout.coloraxis2.colorbar.ticktext == ("Loaded", "Nominal")
    assert [trace.marker.coloraxis for trace in guides] == ["coloraxis", "coloraxis2"]
    assert result.layout.scene.zaxis.range == (-1.0, 31.0)
    assert result.layout.scene.zaxis.title.text == "PC 3"
    product = FigureProduct.model_validate_json(result.to_json())
    assert product.layout.meta.panels["pca"].z_axis == "scene.zaxis"
    assert product.layout.meta.panels["hand"].time is None

    wrong = document.model_dump(mode="json")
    wrong["layers"][0]["encode"]["color"]["field"] = "time"
    with pytest.raises(ValueError, match="one scalar value per plotted group"):
        render(FigureDocument.model_validate(wrong), {"trials": data})
