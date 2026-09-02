"""Compact row-family bindings for required figure-template slots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.analysis.figures import (
    FigureSpecExecutionError,
    execute_figure_spec,
    figure_manifest_plotly_json,
    resolve_figure_trace_bindings,
)
from feedbax.contracts.figures import (
    FIGURE_SLOT_FAMILY_SCHEMA_ID,
    FIGURE_SLOT_FAMILY_SCHEMA_VERSION,
    FigureSlotFamily,
    FigureSpec,
    FigureTemplate,
    SlotSpec,
    TraceBinding,
)
from feedbax.contracts.base import (
    ParentRef,
    canonical_json_bytes,
)
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    safe_manifest_key,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.migrations import default_spec_registry
from feedbax.plot.constructors import register_figure_template

pytestmark = [pytest.mark.feedbax_contract]


def _template(*, multiplicity: str = "many") -> FigureTemplate:
    return FigureTemplate(
        name="feedbax.test_slot_family",
        description="Required compact row family.",
        assembler="feedbax.grid_figure",
        assembler_params={"shared_xaxes": True},
        slots=[
            SlotSpec(
                name="profiles",
                constructor="feedbax.scalar_scatter",
                params_defaults={"marker_size": 9.0},
                multiplicity=multiplicity,
            )
        ],
    )


def _analysis(root: Path) -> ParentRef:
    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:slot-family",
        status="completed",
        metadata={
            "x": [0, 1, 2],
            "series": {
                "left": [1, 2, 3],
                "right": [3, 2, 1],
            },
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


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "name": "left profile",
            "panel": "left",
            "data_paths": {
                "y": {"item": "analysis", "path": "metadata.series.left"},
            },
            "label": "Left",
            "color": "red",
            "marker": "circle",
        },
        {
            "name": "right profile",
            "panel": "right",
            "data_paths": {
                "y": {"item": "analysis", "path": "metadata.series.right"},
            },
            "label": "Right",
            "color": "blue",
            "marker": "square",
        },
    ]


def _family() -> FigureSlotFamily:
    return FigureSlotFamily(
        name="profile rows",
        slot="profiles",
        data={"x": {"item": "analysis", "path": "metadata.x"}},
        rows=_rows(),
        required=True,
    )


def test_slot_family_has_registered_versioned_identity_and_preserves_figure_identity() -> None:
    family = _family()
    assert family.schema_id == FIGURE_SLOT_FAMILY_SCHEMA_ID
    assert family.schema_version == FIGURE_SLOT_FAMILY_SCHEMA_VERSION
    assert (
        default_spec_registry.current_version("FigureSlotFamily")
        == FIGURE_SLOT_FAMILY_SCHEMA_VERSION
    )

    legacy_shape = FigureSpec(name="unchanged", assembler="feedbax.grid_figure")
    dumped = legacy_shape.model_dump(mode="json", exclude_none=True)
    assert "slot_families" not in dumped
    assert FigureSpec.model_validate(dumped).model_dump(mode="json", exclude_none=True) == dumped

    with pytest.raises(ValidationError, match="unsupported FigureSlotFamily schema_version"):
        FigureSlotFamily.model_validate(
            {
                **family.model_dump(mode="json"),
                "schema_version": "feedbax.spec.figure_slot_family.v0",
            }
        )


def test_public_resolution_matches_hand_enumerated_bindings(application_registry_bundle) -> None:
    template = _template()
    compact = FigureSpec(
        name="compact",
        template=template.name,
        slot_families=[_family()],
    )
    enumerated = FigureSpec(
        name="enumerated",
        template=template.name,
        slot_bindings={
            "profiles": [
                TraceBinding(
                    name="left profile",
                    constructor="",
                    panel="left",
                    required=True,
                    data={
                        "x": {"item": "analysis", "path": "metadata.x"},
                        "y": {"item": "analysis", "path": "metadata.series.left"},
                    },
                    params={"label": "Left", "color": "red", "marker_symbol": "circle"},
                ),
                TraceBinding(
                    name="right profile",
                    constructor="",
                    panel="right",
                    required=True,
                    data={
                        "x": {"item": "analysis", "path": "metadata.x"},
                        "y": {"item": "analysis", "path": "metadata.series.right"},
                    },
                    params={"label": "Right", "color": "blue", "marker_symbol": "square"},
                ),
            ]
        },
    )

    compact_plans = resolve_figure_trace_bindings(
        compact, template, registry=application_registry_bundle.figures
    )
    enumerated_plans = resolve_figure_trace_bindings(
        enumerated, template, registry=application_registry_bundle.figures
    )
    assert compact_plans == enumerated_plans
    assert [plan.binding.constructor for plan in compact_plans] == [
        "feedbax.scalar_scatter",
        "feedbax.scalar_scatter",
    ]
    assert [plan.binding.params["marker_size"] for plan in compact_plans] == [9.0, 9.0]


def test_slot_family_execution_equals_hand_enumerated_figure(
    tmp_path: Path, application_registry_bundle
) -> None:
    template = _template()
    register_figure_template(template, registry=application_registry_bundle.figures)
    analysis = _analysis(tmp_path)
    common = {
        "template": template.name,
        "inputs": [analysis],
        "panels": [{"name": "left"}, {"name": "right"}],
    }
    compact = FigureSpec(name="same", slot_families=[_family()], **common)
    enumerated = FigureSpec(
        name="same",
        slot_bindings={
            "profiles": [
                TraceBinding(
                    name=row["name"],
                    constructor="",
                    panel=row["panel"],
                    required=True,
                    data={
                        "x": {"item": "analysis", "path": "metadata.x"},
                        **row["data_paths"],
                    },
                    params={
                        "label": row["label"],
                        "color": row["color"],
                        "marker_symbol": row["marker"],
                    },
                )
                for row in _rows()
            ]
        },
        **common,
    )

    compact_manifest, _ = execute_figure_spec(
        compact, root=tmp_path, registry=application_registry_bundle.figures
    )
    enumerated_manifest, _ = execute_figure_spec(
        enumerated, root=tmp_path, registry=application_registry_bundle.figures
    )
    compact_render = figure_manifest_plotly_json(compact_manifest)
    enumerated_render = figure_manifest_plotly_json(enumerated_manifest)

    assert compact_render is not None
    assert canonical_json_bytes(compact_render) == canonical_json_bytes(enumerated_render)
    assert [record.name for record in compact_manifest.binding_records] == [
        "left profile",
        "right profile",
    ]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["rows"][0].pop("panel"), "panel"),
        (lambda value: value["rows"][0].pop("data_paths"), "data_paths"),
        (lambda value: value["rows"][0].pop("label"), "label"),
        (lambda value: value["rows"][0].pop("color"), "color"),
        (lambda value: value["rows"][0].pop("marker"), "marker"),
        (
            lambda value: value["rows"].__setitem__(
                1, {**value["rows"][1], "name": "left profile"}
            ),
            "duplicate trace names",
        ),
        (
            lambda value: value.update({"schema_version": "feedbax.spec.figure_slot_family.v0"}),
            "unsupported FigureSlotFamily schema_version",
        ),
    ],
)
def test_slot_family_rows_fail_closed(
    mutation: Any,
    match: str,
) -> None:
    value = _family().model_dump(mode="json")
    mutation(value)
    with pytest.raises(ValidationError, match=match):
        FigureSlotFamily.model_validate(value)


def test_slot_family_rejects_field_collisions() -> None:
    with pytest.raises(ValidationError, match="data paths collide"):
        FigureSlotFamily(
            name="data collision",
            slot="profiles",
            data={"y": {"item": "analysis", "path": "metadata.series.left"}},
            rows=_rows(),
        )
    with pytest.raises(ValidationError, match="fields collide"):
        FigureSlotFamily(
            name="param collision",
            slot="profiles",
            params={"color": "black"},
            rows=_rows(),
        )


def test_slot_family_slot_and_cardinality_validation_fail_closed(
    application_registry_bundle,
) -> None:
    family = _family()
    with pytest.raises(ValueError, match="unknown template slots"):
        resolve_figure_trace_bindings(
            FigureSpec(
                name="unknown",
                template="template",
                slot_families=[family.model_copy(update={"slot": "unknown"})],
            ),
            _template(),
            registry=application_registry_bundle.figures,
        )
    with pytest.raises(ValueError, match="requires exactly one TraceBinding"):
        resolve_figure_trace_bindings(
            FigureSpec(name="one", template="template", slot_families=[family]),
            _template(multiplicity="one"),
            registry=application_registry_bundle.figures,
        )
    with pytest.raises(ValidationError, match="bound by both"):
        FigureSpec(
            name="collision",
            template="template",
            slot_bindings={
                "profiles": TraceBinding(name="concrete", constructor="feedbax.scalar_scatter")
            },
            slot_families=[family],
        )


def test_required_slot_and_resolved_trace_name_collisions_fail_closed(
    application_registry_bundle,
) -> None:
    template = _template()
    with pytest.raises(ValueError, match="missing required template slot"):
        resolve_figure_trace_bindings(
            FigureSpec(name="missing", template=template.name),
            template,
            registry=application_registry_bundle.figures,
        )
    with pytest.raises(ValidationError, match="collides with other trace names"):
        FigureSpec(
            name="trace collision",
            template=template.name,
            traces=[TraceBinding(name="left profile", constructor="feedbax.scalar_scatter")],
            slot_families=[_family()],
        )


def test_unknown_slot_execution_writes_failed_manifest(
    tmp_path: Path, application_registry_bundle
) -> None:
    template = _template()
    register_figure_template(template, registry=application_registry_bundle.figures)
    family = _family().model_copy(update={"slot": "unknown"})
    spec = FigureSpec(name="unknown", template=template.name, slot_families=[family])

    with pytest.raises(FigureSpecExecutionError) as exc_info:
        execute_figure_spec(spec, root=tmp_path, registry=application_registry_bundle.figures)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "unknown template slots" in str(exc_info.value.__cause__)
