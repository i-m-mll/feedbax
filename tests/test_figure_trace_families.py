"""Trace-family declaration, expansion, and fail-closed contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.analysis.figures import (
    FigureSpecExecutionError,
    TraceFamilyExpansion,
    execute_figure_spec,
    expand_trace_families,
    figure_manifest_plotly_json,
)
from feedbax.contracts.figures import (
    FigureSpec,
    TraceBinding,
    TraceFamily,
    TraceFamilyIndex,
    TraceFamilyRange,
)
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

pytestmark = [pytest.mark.feedbax_contract]

COLORSCALE = "Viridis"
CYCLICAL_COLORSCALE = "Phase"
KNOT_COUNT = 3

#: A geometrically spaced conditioning sweep, and the Viridis colors it is
#: given when each knot is placed at its own fraction of the swept range. The
#: colors are pinned because reproducing exactly this hand-authored assignment
#: is what value positioning is for.
SWEEP_VALUES = [0.2, 0.239, 0.286, 0.342, 0.409, 0.489, 0.585, 0.699, 0.836, 1.0]
SWEEP_COLORS = [
    "rgb(68, 1, 84)",
    "rgb(70, 18, 100)",
    "rgb(72, 39, 119)",
    "rgb(66, 60, 130)",
    "rgb(57, 84, 139)",
    "rgb(46, 111, 142)",
    "rgb(36, 139, 140)",
    "rgb(45, 173, 127)",
    "rgb(121, 208, 81)",
    "rgb(253, 231, 37)",
]


def _analysis_input(root: Path) -> ParentRef:
    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:trace-families",
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


def _knot_family(prefix: str = "knot", **overrides: Any) -> TraceFamily:
    declaration: dict[str, Any] = {
        "name": f"{prefix}s",
        "index": TraceFamilyIndex(range=TraceFamilyRange(stop=KNOT_COUNT)),
        "colorscale": COLORSCALE,
        "trace": TraceBinding(
            name=prefix + " {index}",
            constructor="feedbax.profile_band",
            panel="main",
            required=True,
            data={
                "x": {"item": "analysis", "path": "metadata.x"},
                "y": {"item": "analysis", "path": "metadata.series.{index}"},
            },
            params={"label": prefix + " {index}"},
        ),
    }
    return TraceFamily(**{**declaration, **overrides})


def _enumerated_knot_traces(prefix: str = "knot") -> list[TraceBinding]:
    colors = list(sample_colorscale_unique(COLORSCALE, KNOT_COUNT, colortype="rgb"))
    return [
        TraceBinding(
            name=f"{prefix} {index}",
            constructor="feedbax.profile_band",
            panel="main",
            required=True,
            data={
                "x": {"item": "analysis", "path": "metadata.x"},
                "y": {"item": "analysis", "path": f"metadata.series.{index}"},
            },
            params={"label": f"{prefix} {index}", "color": color},
        )
        for index, color in enumerate(colors)
    ]


def _rendered(
    spec: FigureSpec, root: Path, *, figure_registry
) -> tuple[Any, list[tuple[str, str | None, str]]]:
    manifest, _path = execute_figure_spec(spec, root=root, registry=figure_registry)
    records = [(record.name, record.panel, record.status) for record in manifest.binding_records]
    return figure_manifest_plotly_json(manifest), records


def test_family_expansion_equals_hand_enumerated_spec(
    tmp_path: Path, application_registry_bundle
) -> None:
    analysis = _analysis_input(tmp_path)
    panels = [{"name": "main", "title": "Knots"}]
    family_spec = FigureSpec(
        name="knot-family",
        assembler="feedbax.grid_figure",
        inputs=[analysis],
        panels=panels,
        trace_families=[_knot_family()],
    )
    enumerated_spec = FigureSpec(
        name="knot-enumerated",
        assembler="feedbax.grid_figure",
        inputs=[analysis],
        panels=panels,
        traces=_enumerated_knot_traces(),
    )

    family_render, family_records = _rendered(
        family_spec, tmp_path, figure_registry=application_registry_bundle.figures
    )
    enumerated_render, enumerated_records = _rendered(
        enumerated_spec, tmp_path, figure_registry=application_registry_bundle.figures
    )

    assert family_render is not None
    assert canonical_json_bytes(family_render) == canonical_json_bytes(enumerated_render)
    assert family_records == enumerated_records
    assert family_records == [
        ("knot 0", "main", "included"),
        ("knot 1", "main", "included"),
        ("knot 2", "main", "included"),
    ]


def test_family_expansion_reports_deterministic_index_color_pairs() -> None:
    spec = FigureSpec(
        name="knot-family",
        assembler="feedbax.grid_figure",
        trace_families=[_knot_family()],
    )

    (expansion,) = expand_trace_families(spec)
    repeated = expand_trace_families(spec)[0]

    expected_colors = list(sample_colorscale_unique(COLORSCALE, KNOT_COUNT, colortype="rgb"))
    assert [member.index for member in expansion.members] == [0, 1, 2]
    assert [member.color for member in expansion.members] == expected_colors
    assert len(set(expected_colors)) == KNOT_COUNT
    assert [member.binding for member in expansion.members] == [
        member.binding for member in repeated.members
    ]
    assert expansion.family.name == "knots"


def test_family_expansion_orders_declared_traces_before_families(
    tmp_path: Path, application_registry_bundle
) -> None:
    analysis = _analysis_input(tmp_path)
    spec = FigureSpec(
        name="mixed-order",
        assembler="feedbax.grid_figure",
        inputs=[analysis],
        panels=[{"name": "main"}],
        traces=[
            TraceBinding(
                name="baseline",
                constructor="feedbax.profile_band",
                panel="main",
                data={"y": [[1, 2, 3]]},
            )
        ],
        trace_families=[_knot_family()],
    )

    _render, records = _rendered(
        spec, tmp_path, figure_registry=application_registry_bundle.figures
    )

    assert [name for name, _panel, _status in records] == [
        "baseline",
        "knot 0",
        "knot 1",
        "knot 2",
    ]


def test_family_index_values_form_supports_named_indices(
    tmp_path: Path, application_registry_bundle
) -> None:
    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:named-indices",
        status="completed",
        metadata={"series": {"slow": [[1, 2, 3]], "fast": [[3, 2, 1]]}},
        analysis_spec=spec_payload("AnalysisRunSpec", {"analysis_type": "feedbax.test"}),
    )
    write_manifest(manifest, root=tmp_path)
    analysis = ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role="analysis",
        uri=f"manifests/analysis_runs/{safe_manifest_key(manifest.id)}.json",
    )
    spec = FigureSpec(
        name="named-index-family",
        assembler="feedbax.grid_figure",
        inputs=[analysis],
        panels=[{"name": "main"}],
        trace_families=[
            TraceFamily(
                name="conditions",
                index=TraceFamilyIndex(values=["slow", "fast"]),
                trace=TraceBinding(
                    name="{index}",
                    constructor="feedbax.profile_band",
                    panel="main",
                    required=True,
                    data={"y": {"item": "analysis", "path": "metadata.series.{index}"}},
                ),
            )
        ],
    )

    _render, records = _rendered(
        spec, tmp_path, figure_registry=application_registry_bundle.figures
    )

    assert [name for name, _panel, _status in records] == ["slow", "fast"]


def test_per_trace_specs_keep_their_figure_manifest_identity() -> None:
    spec = FigureSpec(
        name="family-baseline",
        assembler="feedbax.grid_figure",
        traces=[
            TraceBinding(
                name="demo",
                constructor="feedbax.profile_band",
                data={"y": [[1, 2, 3], [2, 3, 4]]},
            )
        ],
    )

    payload = spec.model_dump(mode="json", exclude_none=True)

    assert "trace_families" not in payload
    assert figure_manifest_id(spec) == "feedbax-figure:01720643a02b69c259ff863ce129fde3"


def test_family_required_indexed_path_fails_closed(
    tmp_path: Path, application_registry_bundle
) -> None:
    analysis = _analysis_input(tmp_path)
    spec = FigureSpec(
        name="unresolvable-index",
        assembler="feedbax.grid_figure",
        inputs=[analysis],
        trace_families=[
            _knot_family(index=TraceFamilyIndex(range=TraceFamilyRange(stop=KNOT_COUNT + 1)))
        ],
    )

    with pytest.raises(FigureSpecExecutionError) as exc_info:
        execute_figure_spec(spec, root=tmp_path, registry=application_registry_bundle.figures)

    assert exc_info.value.manifest.status == "failed"
    assert exc_info.value.manifest.failure["type"] == "ExpressionPathMissing"


@pytest.mark.parametrize(
    ("index", "match"),
    [
        ({"values": [1, 1]}, "indices collide"),
        ({"values": [1, "1"]}, "indices collide"),
        ({"values": []}, "at least one index"),
        ({"values": [0], "range": {"stop": 2}}, "exactly one of values or range"),
        ({}, "exactly one of values or range"),
        ({"range": {"stop": 4, "step": 0}}, "step must be nonzero"),
        ({"range": {"start": 4, "stop": 0}}, "enumerates no indices"),
    ],
)
def test_malformed_index_sets_fail_closed(index: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        TraceFamilyIndex.model_validate(index)


@pytest.mark.parametrize(
    ("family", "match"),
    [
        (
            {
                "name": "constant",
                "index": {"range": {"stop": 3}},
                "trace": {"name": "knot", "constructor": "feedbax.profile_band"},
            },
            "would not vary with the index",
        ),
        (
            {
                "name": "templated",
                "index": {"range": {"stop": 3}},
                "trace": {
                    "name": "knot {index}",
                    "constructor": "feedbax.profile_band",
                    "params": {"label": "knot {index:02d}"},
                },
            },
            "not a templating language",
        ),
        (
            {
                "name": "styled",
                "index": {"range": {"stop": 3}},
                "colorscale": "Viridis",
                "trace": {
                    "name": "knot {index}",
                    "constructor": "feedbax.profile_band",
                    "params": {"color": "rgb(1,2,3)"},
                },
            },
            "both a colorscale and an explicit",
        ),
        (
            {
                "name": "collides",
                "index": {"values": ["a", "b"]},
                "trace": {
                    "name": "knot",
                    "constructor": "feedbax.profile_band",
                    "data": {"y": {"item": "analysis", "path": "metadata.{index}"}},
                },
            },
            "colliding trace names",
        ),
        (
            {
                "name": "wrong-version",
                "schema_version": "feedbax.spec.figure_trace_family.v0",
                "index": {"range": {"stop": 3}},
                "trace": {"name": "knot {index}", "constructor": "feedbax.profile_band"},
            },
            "unsupported TraceFamily schema_version",
        ),
    ],
)
def test_malformed_family_declarations_fail_closed(family: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        TraceFamily.model_validate(family)


def test_spec_rejects_family_name_and_expansion_collisions() -> None:
    with pytest.raises(ValidationError, match="trace_families names collide"):
        FigureSpec(
            name="duplicate-families",
            assembler="feedbax.grid_figure",
            trace_families=[_knot_family(), _knot_family()],
        )

    with pytest.raises(ValidationError, match="collides with other trace names"):
        FigureSpec(
            name="family-collides-with-trace",
            assembler="feedbax.grid_figure",
            traces=[TraceBinding(name="knot 1", constructor="feedbax.profile_band")],
            trace_families=[_knot_family()],
        )


def _sweep_family(**overrides: Any) -> TraceFamily:
    declaration: dict[str, Any] = {
        "name": "sweep",
        "index": TraceFamilyIndex(range=TraceFamilyRange(stop=len(SWEEP_VALUES))),
        "values": SWEEP_VALUES,
        "colorscale": COLORSCALE,
        "trace": TraceBinding(
            name="s={value:.3f}",
            constructor="feedbax.profile_band",
            panel="main",
            required=True,
            data={"y": {"item": "analysis", "path": "metadata.series.{index}"}},
            params={"label": "s={value:.3f}"},
        ),
    }
    return TraceFamily(**{**declaration, **overrides})


def _family_expansion(family: TraceFamily) -> TraceFamilyExpansion:
    spec = FigureSpec(
        name="expansion-only",
        assembler="feedbax.grid_figure",
        trace_families=[family],
    )
    (expansion,) = expand_trace_families(spec)
    return expansion


def test_declared_values_position_member_colors_by_value() -> None:
    expansion = _family_expansion(_sweep_family())

    assert [member.value for member in expansion.members] == SWEEP_VALUES
    assert [member.color for member in expansion.members] == SWEEP_COLORS
    # The pinned colors are exactly the scale sampled at each value's fraction
    # of the swept range, which is what a hand-authored sweep computes.
    positions = [
        (value - SWEEP_VALUES[0]) / (SWEEP_VALUES[-1] - SWEEP_VALUES[0]) for value in SWEEP_VALUES
    ]
    assert SWEEP_COLORS == list(sample_colorscale_at(COLORSCALE, positions, colortype="rgb"))


def test_declared_values_may_be_given_in_any_order() -> None:
    reversed_family = _sweep_family(
        values=list(reversed(SWEEP_VALUES)),
        index=TraceFamilyIndex(values=list(reversed(range(len(SWEEP_VALUES))))),
    )

    reverse = _family_expansion(reversed_family)

    # Position follows the value, not the position in the declaration list.
    assert [member.value for member in reverse.members] == list(reversed(SWEEP_VALUES))
    assert [member.color for member in reverse.members] == list(reversed(SWEEP_COLORS))


def test_uniformly_spaced_values_reproduce_index_spaced_colors() -> None:
    plain = _family_expansion(_knot_family())
    valued = _family_expansion(_knot_family(values=[0.0, 0.5, 1.0]))

    assert [member.color for member in valued.members] == [member.color for member in plain.members]


def test_positioned_sampling_matches_the_unique_sampler_on_cyclical_scales() -> None:
    count = 4
    uniform = [position / (count - 1) for position in range(count)]

    assert list(sample_colorscale_at(CYCLICAL_COLORSCALE, uniform, colortype="rgb")) == list(
        sample_colorscale_unique(CYCLICAL_COLORSCALE, count, colortype="rgb")
    )


def test_value_token_renders_values_alongside_index_keyed_paths() -> None:
    expansion = _family_expansion(_sweep_family())

    assert [member.binding.name for member in expansion.members] == [
        f"s={value:.3f}" for value in SWEEP_VALUES
    ]
    assert [member.binding.params["label"] for member in expansion.members] == [
        f"s={value:.3f}" for value in SWEEP_VALUES
    ]
    assert [member.binding.data["y"].path for member in expansion.members] == [
        f"metadata.series.{index}" for index in range(len(SWEEP_VALUES))
    ]


def test_value_token_without_a_format_spec_renders_the_bare_value() -> None:
    expansion = _family_expansion(
        _sweep_family(
            trace=TraceBinding(
                name="{index}: {value}",
                constructor="feedbax.profile_band",
                data={"y": {"item": "analysis", "path": "metadata.series.{index}"}},
            )
        )
    )

    assert expansion.members[0].binding.name == "0: 0.2"


def test_legend_index_gives_exactly_one_member_the_legend_entry(
    tmp_path: Path, application_registry_bundle
) -> None:
    spec = FigureSpec(
        name="legend-representative",
        assembler="feedbax.grid_figure",
        inputs=[_analysis_input(tmp_path)],
        panels=[{"name": "main"}],
        trace_families=[_knot_family(legend_index=1)],
    )

    (expansion,) = expand_trace_families(spec)
    rendered, _records = _rendered(
        spec, tmp_path, figure_registry=application_registry_bundle.figures
    )

    assert [member.binding.params["showlegend"] for member in expansion.members] == [
        False,
        True,
        False,
    ]
    # The color assignment is untouched by the legend flag.
    assert [member.color for member in expansion.members] == list(
        sample_colorscale_unique(COLORSCALE, KNOT_COUNT, colortype="rgb")
    )
    assert rendered is not None
    assert [trace["name"] for trace in rendered["data"] if trace.get("showlegend") is True] == [
        "knot 1"
    ]


def _trajectory_family(**overrides: Any) -> TraceFamily:
    declaration: dict[str, Any] = {
        "name": "trajectory-arms",
        "index": TraceFamilyIndex(range=TraceFamilyRange(stop=3)),
        "trace": TraceBinding(
            name="arm {index}",
            constructor="feedbax.trajectory_2d",
            panel="main",
            data={
                "trajectories": [
                    [[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]],
                    [[0.0, 0.0], [0.5, 0.5], [1.0, 0.75]],
                ]
            },
            params={"label": "arm {index}"},
        ),
    }
    return TraceFamily(**{**declaration, **overrides})


def test_trajectory_family_showlegend_false_suppresses_all_entries(
    tmp_path: Path, application_registry_bundle
) -> None:
    family = _trajectory_family(
        trace=TraceBinding(
            name="arm {index}",
            constructor="feedbax.trajectory_2d",
            panel="main",
            data={
                "trajectories": [
                    [[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]],
                    [[0.0, 0.0], [0.5, 0.5], [1.0, 0.75]],
                ]
            },
            params={"label": "arm {index}", "showlegend": False},
        )
    )
    spec = FigureSpec(
        name="trajectory-family-no-legend",
        assembler="feedbax.grid_figure",
        panels=[{"name": "main"}],
        trace_families=[family],
    )

    rendered, _records = _rendered(
        spec, tmp_path, figure_registry=application_registry_bundle.figures
    )

    assert rendered is not None
    assert not any(trace.get("showlegend") is True for trace in rendered["data"])


def test_trajectory_family_legend_index_selects_only_one_member(
    tmp_path: Path, application_registry_bundle
) -> None:
    spec = FigureSpec(
        name="trajectory-family-legend-representative",
        assembler="feedbax.grid_figure",
        panels=[{"name": "main"}],
        trace_families=[_trajectory_family(legend_index=1)],
    )

    (expansion,) = expand_trace_families(spec)
    rendered, _records = _rendered(
        spec, tmp_path, figure_registry=application_registry_bundle.figures
    )

    assert [member.binding.params["showlegend"] for member in expansion.members] == [
        False,
        True,
        False,
    ]
    assert rendered is not None
    assert [trace["name"] for trace in rendered["data"] if trace.get("showlegend") is True] == [
        "arm 1",
        "arm 1 mean",
    ]


def test_legend_index_accepts_a_named_index() -> None:
    expansion = _family_expansion(
        _knot_family(index=TraceFamilyIndex(values=["slow", "fast"]), legend_index="fast")
    )

    assert [member.binding.params["showlegend"] for member in expansion.members] == [
        False,
        True,
    ]


def _interleaved_spec(analysis: ParentRef, group: str | None) -> FigureSpec:
    return FigureSpec(
        name="interleaved" if group else "blocked",
        assembler="feedbax.grid_figure",
        inputs=[analysis],
        panels=[{"name": "main"}],
        trace_families=[
            _knot_family("knot", interleave_group=group),
            _knot_family("mark", interleave_group=group),
        ],
    )


def test_interleave_group_expands_families_position_by_position(
    tmp_path: Path, application_registry_bundle
) -> None:
    analysis = _analysis_input(tmp_path)
    knot_traces = _enumerated_knot_traces("knot")
    mark_traces = _enumerated_knot_traces("mark")
    enumerated_spec = FigureSpec(
        name="hand-interleaved",
        assembler="feedbax.grid_figure",
        inputs=[analysis],
        panels=[{"name": "main"}],
        traces=[trace for pair in zip(knot_traces, mark_traces, strict=True) for trace in pair],
    )

    interleaved_render, interleaved_records = _rendered(
        _interleaved_spec(analysis, "position"),
        tmp_path,
        figure_registry=application_registry_bundle.figures,
    )
    enumerated_render, enumerated_records = _rendered(
        enumerated_spec, tmp_path, figure_registry=application_registry_bundle.figures
    )

    assert interleaved_records == enumerated_records
    assert [name for name, _panel, _status in interleaved_records] == [
        "knot 0",
        "mark 0",
        "knot 1",
        "mark 1",
        "knot 2",
        "mark 2",
    ]
    assert interleaved_render is not None
    assert canonical_json_bytes(interleaved_render) == canonical_json_bytes(enumerated_render)


def test_families_without_an_interleave_group_stay_blocked(
    tmp_path: Path, application_registry_bundle
) -> None:
    analysis = _analysis_input(tmp_path)

    _render, records = _rendered(
        _interleaved_spec(analysis, None),
        tmp_path,
        figure_registry=application_registry_bundle.figures,
    )

    assert [name for name, _panel, _status in records] == [
        "knot 0",
        "knot 1",
        "knot 2",
        "mark 0",
        "mark 1",
        "mark 2",
    ]


def test_families_without_the_new_fields_keep_their_declaration_bytes() -> None:
    spec = FigureSpec(
        name="knot-family",
        assembler="feedbax.grid_figure",
        trace_families=[_knot_family()],
    )

    payload = spec.model_dump(mode="json", exclude_none=True)

    assert set(payload["trace_families"][0]) == {
        "schema_id",
        "schema_version",
        "name",
        "index",
        "trace",
        "colorscale",
    }
    assert figure_manifest_id(spec) == "feedbax-figure:ba391491fff6c7f5cfc30bf2abe713cb"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"values": [0.0, 0.5]}, "declares 2 values for 3 indices"),
        ({"values": [0.0, 0.5, float("inf")]}, "values must be finite"),
        ({"values": [0.0, float("nan"), 1.0]}, "values must be finite"),
        ({"values": [0.0, 0.5, 0.0]}, "values must be distinct"),
        (
            {"index": TraceFamilyIndex(values=[7]), "values": [0.5]},
            "spans no value domain",
        ),
        ({"legend_index": 9}, "is not one of"),
        ({"legend_index": "knots"}, "is not one of"),
    ],
)
def test_malformed_value_and_legend_declarations_fail_closed(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        _knot_family(**overrides)


def test_legend_index_conflicting_with_an_explicit_showlegend_fails_closed() -> None:
    with pytest.raises(ValidationError, match="legend_index and an explicit"):
        _knot_family(
            legend_index=0,
            trace=TraceBinding(
                name="knot {index}",
                constructor="feedbax.profile_band",
                params={"showlegend": False},
            ),
        )


def test_value_token_without_declared_values_fails_closed() -> None:
    with pytest.raises(ValidationError, match="only when the family declares per-member values"):
        _knot_family(
            trace=TraceBinding(
                name="s={value:.3f}",
                constructor="feedbax.profile_band",
            )
        )


def test_unusable_value_format_spec_fails_closed() -> None:
    with pytest.raises(ValidationError, match="does not format a value"):
        _sweep_family(
            trace=TraceBinding(
                name="{index} s={value:qq}",
                constructor="feedbax.profile_band",
            )
        )


@pytest.mark.parametrize(
    ("families", "match"),
    [
        (
            [_knot_family("knot", interleave_group="position")],
            "interleaves with nothing",
        ),
        (
            [
                _knot_family("knot", interleave_group="position"),
                _knot_family(
                    "mark",
                    interleave_group="position",
                    index=TraceFamilyIndex(range=TraceFamilyRange(stop=KNOT_COUNT + 1)),
                ),
            ],
            "different index counts",
        ),
    ],
)
def test_malformed_interleave_groups_fail_closed(families: list[TraceFamily], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        FigureSpec(
            name="bad-interleave",
            assembler="feedbax.grid_figure",
            trace_families=families,
        )
