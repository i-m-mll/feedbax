"""Declared-colorbar render, trace-family composition, and fail-closed tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.analysis.figures import (
    FigureSpecExecutionError,
    execute_figure_spec,
    figure_manifest_plotly_json,
    resolve_figure_colorbar,
)
from feedbax.contracts.figures import (
    COLORBAR_PANEL_PLACEMENT_SCHEMA_ID,
    COLORBAR_PANEL_PLACEMENT_SCHEMA_VERSION,
    ColorbarPanelPlacement,
    FigureColorbar,
    FigureSpec,
    TraceBinding,
    TraceFamily,
    TraceFamilyIndex,
    TraceFamilyRange,
)
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    ParentRef,
    canonical_json_bytes,
    figure_manifest_id,
    safe_manifest_key,
    spec_payload,
    write_manifest,
)
from feedbax.plot.colors import sample_colorscale_at, sample_colorscale_unique
from feedbax.plot.constructors import (
    GridFigureParams,
    get_figure_constructor,
    register_figure_constructor,
)

pytestmark = [pytest.mark.feedbax_contract]

COLORSCALE = "Viridis"
KNOT_COUNT = 3


def _analysis_input(root: Path) -> ParentRef:
    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:colorbar",
        status="completed",
        metadata={
            "x": [0, 1, 2],
            "series": [
                [[1, 2, 3], [2, 3, 4]],
                [[4, 3, 2], [3, 2, 1]],
                [[2, 2, 2], [3, 3, 3]],
            ],
        },
        analysis_spec=spec_payload("AnalysisRunSpec", {"analysis_type": "feedbax.test"}),
    )
    write_manifest(manifest, root=root)
    return ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role="analysis",
        uri=f"manifests/analysis_runs/{safe_manifest_key(manifest.id)}.json",
    )


def _knot_family(**overrides: Any) -> TraceFamily:
    declaration: dict[str, Any] = {
        "name": "knots",
        "index": TraceFamilyIndex(range=TraceFamilyRange(stop=KNOT_COUNT)),
        "colorscale": COLORSCALE,
        "trace": TraceBinding(
            name="knot {index}",
            constructor="feedbax.profile_band",
            panel="main",
            required=True,
            data={
                "x": {"item": "analysis", "path": "metadata.x"},
                "y": {"item": "analysis", "path": "metadata.series.{index}"},
            },
            params={"label": "knot {index}", "showlegend": False},
        ),
    }
    return TraceFamily(**{**declaration, **overrides})


def _plain_trace() -> TraceBinding:
    return TraceBinding(
        name="baseline",
        constructor="feedbax.profile_band",
        panel="main",
        data={"y": [[1, 2, 3], [2, 3, 4]]},
    )


def _family_colors() -> list[str]:
    return list(sample_colorscale_unique(COLORSCALE, KNOT_COUNT, colortype="rgb"))


def _render(spec: FigureSpec, root: Path) -> dict[str, Any]:
    manifest, _path = execute_figure_spec(spec, root=root)
    rendered = figure_manifest_plotly_json(manifest)
    assert rendered is not None
    return rendered


def _colorbar_traces(rendered: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        trace
        for trace in rendered["data"]
        if trace.get("marker", {}).get("showscale") is True
    ]


def test_declared_colorbar_renders_a_key_without_legend_entries(tmp_path: Path) -> None:
    spec = FigureSpec(
        name="standalone-colorbar",
        assembler="feedbax.grid_figure",
        panels=[{"name": "main"}],
        traces=[_plain_trace()],
        colorbar=FigureColorbar(title="s", colorscale=COLORSCALE, range=(0.0, 1.0)),
    )

    rendered = _render(spec, tmp_path)

    (carrier,) = _colorbar_traces(rendered)
    assert carrier["marker"]["colorscale"] is not None
    assert (carrier["marker"]["cmin"], carrier["marker"]["cmax"]) == (0.0, 1.0)
    assert carrier["marker"]["colorbar"]["title"] == {"text": "s"}
    assert carrier["showlegend"] is False
    assert carrier["hoverinfo"] == "skip"
    assert carrier["x"] == [None] and carrier["y"] == [None]


def test_figures_without_a_colorbar_render_and_hash_unchanged(tmp_path: Path) -> None:
    spec = FigureSpec(
        name="unkeyed-figure",
        assembler="feedbax.grid_figure",
        panels=[{"name": "main"}],
        traces=[_plain_trace()],
    )
    keyed = FigureSpec(
        **{
            **spec.model_dump(exclude_none=True),
            "colorbar": FigureColorbar(colorscale=COLORSCALE, range=(0.0, 1.0)),
        }
    )

    rendered = _render(spec, tmp_path)
    keyed_rendered = _render(keyed, tmp_path)

    assert _colorbar_traces(rendered) == []
    assert b"colorbar" not in canonical_json_bytes(spec)
    assert "colorbar" not in spec.model_dump(mode="json", exclude_none=True)
    assert figure_manifest_id(spec) != figure_manifest_id(keyed)
    assert len(keyed_rendered["data"]) == len(rendered["data"]) + 1
    assert keyed_rendered["data"][: len(rendered["data"])] == rendered["data"]


def test_colorbar_panel_placement_has_explicit_schema_identity() -> None:
    placement = ColorbarPanelPlacement(panel="main", length_fraction=0.5)
    colorbar = FigureColorbar(
        colorscale=COLORSCALE,
        range=(0.0, 1.0),
    )

    assert placement.schema_id == COLORBAR_PANEL_PLACEMENT_SCHEMA_ID
    assert placement.schema_version == COLORBAR_PANEL_PLACEMENT_SCHEMA_VERSION
    assert placement.center_fraction == 0.5
    assert placement.side == "right"
    assert placement.offset_fraction == 0.0
    assert (
        default_spec_registry.current_version("ColorbarPanelPlacement")
        == COLORBAR_PANEL_PLACEMENT_SCHEMA_VERSION
    )
    assert "placement" not in colorbar.model_dump(mode="json", exclude_none=True)

    with pytest.raises(
        ValidationError,
        match="unsupported ColorbarPanelPlacement schema_version",
    ):
        ColorbarPanelPlacement(
            schema_version="feedbax.spec.colorbar_panel_placement.v0",
            panel="main",
            length_fraction=0.5,
        )


def test_colorbar_placement_resolves_against_selected_panel_domain(
    tmp_path: Path,
) -> None:
    spec = FigureSpec(
        name="panel-relative-colorbar",
        assembler="feedbax.grid_figure",
        panels=[
            {"name": "upper", "row": 1, "col": 1},
            {"name": "lower", "row": 2, "col": 1},
        ],
        traces=[_plain_trace()],
        colorbar=FigureColorbar(
            title="s",
            colorscale=COLORSCALE,
            range=(0.0, 1.0),
            placement=ColorbarPanelPlacement(
                panel="lower",
                length_fraction=0.5,
                center_fraction=0.5,
                side="left",
                offset_fraction=0.1,
            ),
        ),
    )

    rendered = _render(spec, tmp_path)

    (carrier,) = _colorbar_traces(rendered)
    colorbar = carrier["marker"]["colorbar"]
    x_low, x_high = rendered["layout"]["xaxis2"]["domain"]
    y_low, y_high = rendered["layout"]["yaxis2"]["domain"]
    assert colorbar["lenmode"] == "fraction"
    assert colorbar["len"] == pytest.approx((y_high - y_low) * 0.5)
    assert colorbar["y"] == pytest.approx(y_low + (y_high - y_low) * 0.5)
    assert colorbar["yanchor"] == "middle"
    assert colorbar["x"] == pytest.approx(x_low - (x_high - x_low) * 0.1)
    assert colorbar["xanchor"] == "right"
    assert colorbar["title"] == {"text": "s"}


@pytest.mark.parametrize(
    ("placement", "match"),
    [
        ({"panel": "", "length_fraction": 0.5}, "panel"),
        ({"panel": "main", "length_fraction": 0.0}, "length_fraction"),
        ({"panel": "main", "length_fraction": 1.1}, "length_fraction"),
        (
            {
                "panel": "main",
                "length_fraction": 0.6,
                "center_fraction": 0.2,
            },
            "extend outside",
        ),
        (
            {
                "panel": "main",
                "length_fraction": 0.5,
                "offset_fraction": 1.1,
            },
            "offset_fraction",
        ),
    ],
)
def test_invalid_colorbar_panel_placement_fails_closed(
    placement: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        ColorbarPanelPlacement.model_validate(placement)


def test_colorbar_placement_rejects_unknown_panel_without_artifact(
    tmp_path: Path,
) -> None:
    spec = FigureSpec(
        name="unknown-colorbar-panel",
        assembler="feedbax.grid_figure",
        panels=[{"name": "main"}],
        colorbar=FigureColorbar(
            colorscale=COLORSCALE,
            range=(0.0, 1.0),
            placement=ColorbarPanelPlacement(
                panel="missing",
                length_fraction=0.5,
            ),
        ),
    )

    with pytest.raises(FigureSpecExecutionError) as exc_info:
        execute_figure_spec(spec, root=tmp_path)

    assert "exactly one panel named 'missing'" in exc_info.value.manifest.failure["message"]
    assert exc_info.value.manifest.artifacts == []
    assert not (tmp_path / "figures").exists()


def test_colorbar_placement_rejects_non_grid_figure_assembler(
    tmp_path: Path,
) -> None:
    grid = get_figure_constructor("feedbax.grid_figure", tier="figure")
    register_figure_constructor(
        "feedbax.test_grid_figure_without_panel_placement",
        tier="figure",
        constructor=grid.callable,
        params_model=GridFigureParams,
        description="Grid-compatible test finalizer without placement authority.",
        replace=True,
    )
    spec = FigureSpec(
        name="unsupported-colorbar-placement",
        assembler="feedbax.test_grid_figure_without_panel_placement",
        panels=[{"name": "main"}],
        colorbar=FigureColorbar(
            colorscale=COLORSCALE,
            range=(0.0, 1.0),
            placement=ColorbarPanelPlacement(
                panel="main",
                length_fraction=0.5,
            ),
        ),
    )

    with pytest.raises(FigureSpecExecutionError) as exc_info:
        execute_figure_spec(spec, root=tmp_path)

    assert "requires the feedbax.comparison_grid/feedbax.grid_figure" in (
        exc_info.value.manifest.failure["message"]
    )
    assert exc_info.value.manifest.artifacts == []
    assert not (tmp_path / "figures").exists()


def test_family_bound_colorbar_reads_the_family_assigned_colors(tmp_path: Path) -> None:
    spec = FigureSpec(
        name="family-colorbar",
        assembler="feedbax.grid_figure",
        inputs=[_analysis_input(tmp_path)],
        panels=[{"name": "main"}],
        trace_families=[_knot_family()],
        colorbar=FigureColorbar(title="knot", family="knots"),
    )

    resolved = resolve_figure_colorbar(spec)
    rendered = _render(spec, tmp_path)

    assert resolved is not None and resolved.family is None
    assert resolved.colorscale == [
        (0.0, _family_colors()[0]),
        (0.5, _family_colors()[1]),
        (1.0, _family_colors()[2]),
    ]
    assert resolved.range == (0.0, 2.0)
    (carrier,) = _colorbar_traces(rendered)
    assert [stop[1] for stop in carrier["marker"]["colorscale"]] == _family_colors()
    assert (carrier["marker"]["cmin"], carrier["marker"]["cmax"]) == (0.0, 2.0)


def test_family_bound_colorbar_relabels_the_index_domain(tmp_path: Path) -> None:
    spec = FigureSpec(
        name="relabelled-colorbar",
        assembler="feedbax.grid_figure",
        inputs=[_analysis_input(tmp_path)],
        panels=[{"name": "main"}],
        trace_families=[_knot_family()],
        colorbar=FigureColorbar(title="s", family="knots", range=(0.0, 1.0)),
    )

    resolved = resolve_figure_colorbar(spec)

    assert resolved is not None
    assert resolved.range == (0.0, 1.0)
    assert [stop[0] for stop in resolved.colorscale] == [0.0, 0.5, 1.0]
    assert [stop[1] for stop in resolved.colorscale] == _family_colors()


def test_family_bound_colorbar_places_uneven_indices_at_their_positions() -> None:
    spec = FigureSpec(
        name="uneven-colorbar",
        assembler="feedbax.grid_figure",
        trace_families=[_knot_family(index=TraceFamilyIndex(values=[0, 1, 3]))],
        colorbar=FigureColorbar(family="knots"),
    )

    resolved = resolve_figure_colorbar(spec)

    assert resolved is not None
    assert [stop[0] for stop in resolved.colorscale] == [0.0, 1 / 3, 1.0]
    assert [stop[1] for stop in resolved.colorscale] == _family_colors()
    assert resolved.range == (0.0, 3.0)


def test_family_bound_colorbar_keys_the_colors_its_traces_were_given(tmp_path: Path) -> None:
    spec = FigureSpec(
        name="agreeing-colorbar",
        assembler="feedbax.grid_figure",
        inputs=[_analysis_input(tmp_path)],
        panels=[{"name": "main"}],
        trace_families=[_knot_family()],
        colorbar=FigureColorbar(family="knots"),
    )

    rendered = _render(spec, tmp_path)

    (carrier,) = _colorbar_traces(rendered)
    key_colors = [stop[1] for stop in carrier["marker"]["colorscale"]]
    trace_colors = [
        trace["line"]["color"]
        for trace in rendered["data"]
        if trace.get("name", "").startswith("knot ")
    ]
    assert key_colors == trace_colors


@pytest.mark.parametrize(
    ("declaration", "match"),
    [
        ({"title": "s"}, "either a trace family or both"),
        ({"colorscale": COLORSCALE}, "either a trace family or both"),
        ({"range": (0.0, 1.0)}, "either a trace family or both"),
        (
            {"family": "knots", "colorscale": COLORSCALE},
            "must not declare its own colorscale",
        ),
        ({"colorscale": COLORSCALE, "range": (1.0, 0.0)}, "range must increase"),
        ({"colorscale": COLORSCALE, "range": (1.0, 1.0)}, "range must increase"),
        (
            {
                "schema_version": "feedbax.spec.figure_colorbar.v0",
                "colorscale": COLORSCALE,
                "range": (0.0, 1.0),
            },
            "unsupported FigureColorbar schema_version",
        ),
    ],
)
def test_malformed_colorbar_declarations_fail_closed(
    declaration: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        FigureColorbar.model_validate(declaration)


@pytest.mark.parametrize(
    ("families", "colorbar", "match"),
    [
        (None, {"family": "knots"}, "undeclared trace family"),
        (
            [_knot_family(colorscale=None)],
            {"family": "knots"},
            "declares no colorscale",
        ),
        (
            [_knot_family(index=TraceFamilyIndex(values=["slow", "fast"]))],
            {"family": "knots"},
            "non-numeric indices",
        ),
        (
            [_knot_family(index=TraceFamilyIndex(values=[2]))],
            {"family": "knots"},
            "enumerates one index",
        ),
    ],
)
def test_malformed_colorbar_family_bindings_fail_closed(
    families: list[TraceFamily] | None, colorbar: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        FigureSpec(
            name="bad-binding",
            assembler="feedbax.grid_figure",
            trace_families=families,
            colorbar=FigureColorbar.model_validate(colorbar),
        )


def test_colorbar_authored_as_an_assembler_param_fails_closed(tmp_path: Path) -> None:
    spec = FigureSpec(
        name="misplaced-colorbar",
        assembler="feedbax.grid_figure",
        panels=[{"name": "main"}],
        traces=[_plain_trace()],
        assembler_params={"colorbar": {"colorscale": COLORSCALE, "range": (0.0, 1.0)}},
    )

    with pytest.raises(FigureSpecExecutionError) as exc_info:
        execute_figure_spec(spec, root=tmp_path)

    assert "not by assembler_params" in exc_info.value.manifest.failure["message"]


#: A geometrically spaced sweep whose interior knots no linear relabeling of an
#: index domain can place correctly.
SWEEP_VALUES = [0.2, 0.4, 1.0]


def _sweep_family(**overrides: Any) -> TraceFamily:
    return _knot_family(values=SWEEP_VALUES, **overrides)


def _sweep_colors() -> list[str]:
    low, high = SWEEP_VALUES[0], SWEEP_VALUES[-1]
    positions = [(value - low) / (high - low) for value in SWEEP_VALUES]
    return list(sample_colorscale_at(COLORSCALE, positions, colortype="rgb"))


def test_value_keyed_colorbar_stops_at_the_member_values(tmp_path: Path) -> None:
    spec = FigureSpec(
        name="value-colorbar",
        assembler="feedbax.grid_figure",
        inputs=[_analysis_input(tmp_path)],
        panels=[{"name": "main"}],
        trace_families=[_sweep_family()],
        colorbar=FigureColorbar(title="s", family="knots"),
    )

    resolved = resolve_figure_colorbar(spec)
    rendered = _render(spec, tmp_path)

    assert resolved is not None and resolved.family is None
    # Stops sit where the values sit, not at even index spacing.
    assert [stop[0] for stop in resolved.colorscale] == [0.0, 0.25, 1.0]
    assert [stop[1] for stop in resolved.colorscale] == _sweep_colors()
    assert resolved.range == (0.2, 1.0)
    (carrier,) = _colorbar_traces(rendered)
    assert (carrier["marker"]["cmin"], carrier["marker"]["cmax"]) == (0.2, 1.0)
    assert carrier["marker"]["colorbar"]["title"] == {"text": "s"}


def test_value_keyed_colorbar_agrees_with_the_colors_its_traces_were_given(
    tmp_path: Path,
) -> None:
    spec = FigureSpec(
        name="agreeing-value-colorbar",
        assembler="feedbax.grid_figure",
        inputs=[_analysis_input(tmp_path)],
        panels=[{"name": "main"}],
        trace_families=[_sweep_family()],
        colorbar=FigureColorbar(family="knots"),
    )

    rendered = _render(spec, tmp_path)

    (carrier,) = _colorbar_traces(rendered)
    trace_colors = [
        trace["line"]["color"]
        for trace in rendered["data"]
        if trace.get("name", "").startswith("knot ")
    ]
    assert [stop[1] for stop in carrier["marker"]["colorscale"]] == trace_colors
    assert trace_colors == _sweep_colors()


def test_value_keyed_colorbar_accepts_named_indices() -> None:
    spec = FigureSpec(
        name="named-index-value-colorbar",
        assembler="feedbax.grid_figure",
        trace_families=[
            _sweep_family(index=TraceFamilyIndex(values=["slow", "mid", "fast"]))
        ],
        colorbar=FigureColorbar(family="knots"),
    )

    resolved = resolve_figure_colorbar(spec)

    assert resolved is not None
    assert resolved.range == (0.2, 1.0)
    assert [stop[0] for stop in resolved.colorscale] == [0.0, 0.25, 1.0]


def test_value_keyed_colorbar_rejects_an_explicit_range() -> None:
    with pytest.raises(ValidationError, match="an explicit range would relabel them"):
        FigureSpec(
            name="relabelled-value-colorbar",
            assembler="feedbax.grid_figure",
            trace_families=[_sweep_family()],
            colorbar=FigureColorbar(family="knots", range=(0.0, 1.0)),
        )


def test_colorbar_on_an_assembler_without_one_fails_closed(tmp_path: Path) -> None:
    spec = FigureSpec(
        name="unsupported-assembler-colorbar",
        assembler="feedbax.trajectories_2d_row",
        panels=[{"name": "main"}],
        traces=[_plain_trace()],
        colorbar=FigureColorbar(colorscale=COLORSCALE, range=(0.0, 1.0)),
    )

    with pytest.raises(FigureSpecExecutionError) as exc_info:
        execute_figure_spec(spec, root=tmp_path)

    assert "declares no colorbar parameter" in exc_info.value.manifest.failure["message"]
