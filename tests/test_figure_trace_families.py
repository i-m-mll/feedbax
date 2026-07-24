"""Trace-family declaration, expansion, and fail-closed contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.analysis.figures import (
    FigureSpecExecutionError,
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
from feedbax.plot.colors import sample_colorscale_unique

pytestmark = [pytest.mark.feedbax_contract]

COLORSCALE = "Viridis"
KNOT_COUNT = 3


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
            params={"label": "knot {index}"},
        ),
    }
    return TraceFamily(**{**declaration, **overrides})


def _enumerated_knot_traces() -> list[TraceBinding]:
    colors = list(sample_colorscale_unique(COLORSCALE, KNOT_COUNT, colortype="rgb"))
    return [
        TraceBinding(
            name=f"knot {index}",
            constructor="feedbax.profile_band",
            panel="main",
            required=True,
            data={
                "x": {"item": "analysis", "path": "metadata.x"},
                "y": {"item": "analysis", "path": f"metadata.series.{index}"},
            },
            params={"label": f"knot {index}", "color": color},
        )
        for index, color in enumerate(colors)
    ]


def _rendered(spec: FigureSpec, root: Path) -> tuple[Any, list[tuple[str, str | None, str]]]:
    manifest, _path = execute_figure_spec(spec, root=root)
    records = [
        (record.name, record.panel, record.status) for record in manifest.binding_records
    ]
    return figure_manifest_plotly_json(manifest), records


def test_family_expansion_equals_hand_enumerated_spec(tmp_path: Path) -> None:
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

    family_render, family_records = _rendered(family_spec, tmp_path)
    enumerated_render, enumerated_records = _rendered(enumerated_spec, tmp_path)

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


def test_family_expansion_orders_declared_traces_before_families(tmp_path: Path) -> None:
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

    _render, records = _rendered(spec, tmp_path)

    assert [name for name, _panel, _status in records] == [
        "baseline",
        "knot 0",
        "knot 1",
        "knot 2",
    ]


def test_family_index_values_form_supports_named_indices(tmp_path: Path) -> None:
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

    _render, records = _rendered(spec, tmp_path)

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


def test_family_required_indexed_path_fails_closed(tmp_path: Path) -> None:
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
        execute_figure_spec(spec, root=tmp_path)

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
