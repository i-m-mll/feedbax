from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.contracts.expressions import Coalesce, Coerce, Compare, Select, ValueQuery
from feedbax.contracts.extraction import (
    EXTRACTION_PRODUCT_SPEC_SCHEMA_ID,
    EXTRACTION_PRODUCT_SPEC_SCHEMA_VERSION,
    DataProductDrift,
    ExtractionProductIdentityMismatch,
    ExtractionProductSpec,
    FieldMapping,
    SourceBinding,
    materialize_extraction_product,
    verify_extraction_product,
)
from feedbax.contracts.manifest import AnalysisDataProduct


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "extraction" / "broad_epsilon"
BROAD_EPSILON_PRODUCT_PATH = (
    FIXTURE_ROOT
    / "results"
    / "ea6ccb4"
    / "data_products"
    / "broad_epsilon_budget_anchors.json"
)
BROAD_EPSILON_HASH = "4e5d319c4848ef19d25ddf9dc8d21a6230cc0d336c5f565fe1a0b63516332542"
BROAD_EPSILON_ADOPTION_NOTE = (
    "Historical broad-epsilon runs used the identical baked values; the adoption record "
    "makes that provenance explicit and reproducible."
)


def _frontier_query(path: str, factor: float) -> ValueQuery:
    return ValueQuery(
        item="source",
        path=path,
        select=Select(
            where=Compare(
                item="entry",
                path="factor",
                op="approx_eq",
                value=factor,
                tolerance=1e-12,
            )
        ),
    )


def _field(
    output_path: str,
    *,
    item: str,
    source_path: str,
    adopts_source_field: str | None = None,
) -> FieldMapping:
    return FieldMapping(
        output_path=output_path,
        query=ValueQuery(
            item=item,
            path=source_path,
            coerce=Coerce(to="float"),
        ),
        adopts_source_field=adopts_source_field,
    )


def _broad_epsilon_spec(
    *,
    expected_identity_hash: str | None = BROAD_EPSILON_HASH,
) -> ExtractionProductSpec:
    # Calibration remains a builder in v1: its caller supplies pre-extracted values, so
    # Feedbax cannot byte-reproduce a fully source-extracted product until those upstream
    # extraction outputs are themselves durable source manifests.
    return ExtractionProductSpec(
        schema_id=EXTRACTION_PRODUCT_SPEC_SCHEMA_ID,
        schema_version=EXTRACTION_PRODUCT_SPEC_SCHEMA_VERSION,
        product_schema_id="rlrmp.broad_epsilon_budget_anchors",
        product_schema_version="rlrmp.broad_epsilon_budget_anchors.v1",
        role="broad_epsilon_budget_anchors",
        logical_name="cs_broad_epsilon_budget_anchors",
        producer_manifest_id=(
            "rlrmp.data_products.broad_epsilon.build_broad_epsilon_budget_anchors_product"
        ),
        sources=[
            SourceBinding(
                alias="moderate",
                kind="manifest",
                uri="results/cb98e58/notes/analytical_game_card_manifest.json",
                payload_query=_frontier_query("frontier", 1.4),
                adoption_note=BROAD_EPSILON_ADOPTION_NOTE,
            ),
            SourceBinding(
                alias="strong",
                kind="manifest",
                uri="results/a7dad8a/notes/adversary_equivalence_manifest.json",
                payload_query=_frontier_query("game_card_summary.frontier", 1.05),
                adoption_note=BROAD_EPSILON_ADOPTION_NOTE,
            ),
        ],
        fields=[
            _field("levels.moderate.gamma_factor", item="moderate", source_path="factor"),
            _field(
                "levels.moderate.closed_loop_epsilon_energy_15cm",
                item="moderate",
                source_path="closed_loop_epsilon_energy",
                adopts_source_field="closed_loop_epsilon_energy",
            ),
            _field(
                "levels.moderate.closed_loop_epsilon_l2_15cm",
                item="moderate",
                source_path="closed_loop_epsilon_l2",
                adopts_source_field="closed_loop_epsilon_l2",
            ),
            _field(
                "levels.moderate.delta_v_percent",
                item="moderate",
                source_path="delta_v_percent",
                adopts_source_field="delta_v_percent",
            ),
            _field("levels.strong.gamma_factor", item="strong", source_path="factor"),
            _field(
                "levels.strong.closed_loop_epsilon_energy_15cm",
                item="strong",
                source_path="closed_loop_epsilon_energy",
                adopts_source_field="closed_loop_epsilon_energy",
            ),
            _field(
                "levels.strong.closed_loop_epsilon_l2_15cm",
                item="strong",
                source_path="closed_loop_epsilon_l2",
                adopts_source_field="closed_loop_epsilon_l2",
            ),
            _field(
                "levels.strong.delta_v_percent",
                item="strong",
                source_path="delta_v_percent",
                adopts_source_field="delta_v_percent",
            ),
        ],
        static_parameters={
            "reference_reach_m": 0.15,
            "levels": {
                "moderate": {
                    "gamma_factor": None,
                    "closed_loop_epsilon_energy_15cm": None,
                    "closed_loop_epsilon_l2_15cm": None,
                    "delta_v_percent": None,
                    "source_issue": "cb98e58",
                    "source_note": "results/cb98e58/notes/analytical_game_card_manifest.json",
                },
                "strong": {
                    "gamma_factor": None,
                    "closed_loop_epsilon_energy_15cm": None,
                    "closed_loop_epsilon_l2_15cm": None,
                    "delta_v_percent": None,
                    "source_issue": "a7dad8a",
                    "source_note": "results/a7dad8a/notes/adversary_equivalence_manifest.json",
                },
            },
        },
        materialization={
            "materializer": (
                "rlrmp.data_products.broad_epsilon.build_broad_epsilon_budget_anchors_product"
            ),
            "adoption_mode": "read_analytical_frontier_entries",
        },
        metadata={
            "issue": "ea6ccb4",
            "note": "Broad-epsilon per-level closed-loop epsilon budgets adopted from "
            "analytical game-card / adversary-equivalence manifests.",
        },
        expected_identity_hash=expected_identity_hash,
    )


def _serialized_product(product: AnalysisDataProduct) -> str:
    return product.model_dump_json(indent=2, exclude_none=True) + "\n"


def _minimal_spec(
    *,
    sources: list[SourceBinding],
    fields: list[FieldMapping],
    static_parameters: dict[str, Any] | None = None,
) -> ExtractionProductSpec:
    return ExtractionProductSpec(
        product_schema_id="feedbax.spec.test_product",
        product_schema_version="feedbax.spec.test_product.v1",
        role="test_product",
        logical_name="test_product",
        producer_manifest_id="feedbax.tests",
        sources=sources,
        fields=fields,
        static_parameters=static_parameters or {},
    )


def _persisted_payload() -> dict[str, Any]:
    return json.loads(BROAD_EPSILON_PRODUCT_PATH.read_text(encoding="utf-8"))


def test_broad_epsilon_extraction_spec_reproduces_tracked_product_bytes() -> None:
    product = materialize_extraction_product(_broad_epsilon_spec(), FIXTURE_ROOT)

    assert product.product_identity_hash == BROAD_EPSILON_HASH
    assert _serialized_product(product) == BROAD_EPSILON_PRODUCT_PATH.read_text(
        encoding="utf-8"
    )


def test_verify_extraction_product_reports_typed_parameter_drift() -> None:
    payload = _persisted_payload()
    payload["parameters"]["levels"]["moderate"]["delta_v_percent"] = 0.0
    payload.pop("product_identity_hash")
    persisted = AnalysisDataProduct.model_validate(payload)

    with pytest.raises(DataProductDrift) as excinfo:
        verify_extraction_product(_broad_epsilon_spec(), persisted, FIXTURE_ROOT)

    drift = excinfo.value
    assert drift.output_path == "levels.moderate.delta_v_percent"
    assert drift.source_uri == "results/cb98e58/notes/analytical_game_card_manifest.json"
    assert drift.persisted_value == 0.0
    assert drift.source_value == 4.041729916548296


def test_expected_identity_hash_fails_closed() -> None:
    with pytest.raises(ExtractionProductIdentityMismatch) as excinfo:
        materialize_extraction_product(
            _broad_epsilon_spec(expected_identity_hash="sha256:not-the-product"),
            FIXTURE_ROOT,
        )

    assert BROAD_EPSILON_HASH in str(excinfo.value)


def test_optional_source_uses_present_payload_query_and_absent_placeholder(
    tmp_path: Path,
) -> None:
    (tmp_path / "present.json").write_text(
        json.dumps({"wrapped": {"value": 4}}),
        encoding="utf-8",
    )

    present = _minimal_spec(
        sources=[
            SourceBinding(
                alias="src",
                kind="manifest",
                uri="present.json",
                payload_query=ValueQuery(item="source", path="wrapped"),
                optional=True,
                missing_payload={"value": 7},
            )
        ],
        fields=[
            FieldMapping(
                output_path="value",
                query=ValueQuery(item="src", path="value"),
            )
        ],
    )
    absent = present.model_copy(
        update={
            "sources": [
                SourceBinding(
                    alias="src",
                    kind="manifest",
                    uri="absent.json",
                    payload_query=ValueQuery(item="source", path="wrapped"),
                    optional=True,
                    missing_payload={"value": 7},
                )
            ]
        }
    )

    assert materialize_extraction_product(present, tmp_path).parameters["value"] == 4
    # The placeholder is already the bound payload; payload_query is not applied to it.
    assert materialize_extraction_product(absent, tmp_path).parameters["value"] == 7


def test_optional_source_validation_and_nonoptional_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        SourceBinding(alias="src", kind="manifest", uri="missing.json", optional=True)
    with pytest.raises(ValidationError):
        SourceBinding(
            alias="src",
            kind="manifest",
            uri="missing.json",
            missing_payload=None,
        )

    spec = _minimal_spec(
        sources=[SourceBinding(alias="src", kind="manifest", uri="missing.json")],
        fields=[FieldMapping(output_path="value", query=ValueQuery(item="src"))],
    )
    with pytest.raises(FileNotFoundError, match="Extraction source is missing: missing.json"):
        materialize_extraction_product(spec, tmp_path)


def test_residual_exclude_deep_copies_mapping_minus_top_level_keys(tmp_path: Path) -> None:
    payload = {
        "block": {
            "keep": {"nested": [1, 2]},
            "drop": {"secret": True},
        }
    }
    (tmp_path / "source.json").write_text(json.dumps(payload), encoding="utf-8")
    spec = _minimal_spec(
        sources=[SourceBinding(alias="src", kind="manifest", uri="source.json")],
        fields=[
            FieldMapping(
                output_path="residual",
                query=ValueQuery(item="src", path="block"),
                residual_exclude=["drop"],
            ),
            FieldMapping(
                output_path="full",
                query=ValueQuery(item="src", path="block"),
                residual_exclude=[],
            ),
        ],
    )

    product = materialize_extraction_product(spec, tmp_path)
    assert product.parameters["residual"] == {"keep": {"nested": [1, 2]}}
    assert product.parameters["full"] == payload["block"]
    product.parameters["residual"]["keep"]["nested"].append(3)
    assert payload["block"]["keep"]["nested"] == [1, 2]


def test_residual_exclude_validation_and_mapping_requirement(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        FieldMapping(
            output_path="value",
            query=ValueQuery(item="src"),
            adopts_source_field="value",
            residual_exclude=[],
        )

    (tmp_path / "source.json").write_text(json.dumps({"value": 1}), encoding="utf-8")
    spec = _minimal_spec(
        sources=[SourceBinding(alias="src", kind="manifest", uri="source.json")],
        fields=[
            FieldMapping(
                output_path="value",
                query=ValueQuery(item="src", path="value"),
                residual_exclude=[],
            )
        ],
    )
    with pytest.raises(ValueError, match="residual_exclude requires a Mapping"):
        materialize_extraction_product(spec, tmp_path)


def test_coalesce_field_mapping_across_sources_and_drift_reporting(tmp_path: Path) -> None:
    (tmp_path / "primary.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "fallback.json").write_text(json.dumps({"value": 5}), encoding="utf-8")
    spec = _minimal_spec(
        sources=[
            SourceBinding(alias="primary", kind="manifest", uri="primary.json"),
            SourceBinding(alias="fallback", kind="manifest", uri="fallback.json"),
        ],
        fields=[
            FieldMapping(
                output_path="value",
                query=Coalesce(
                    queries=[
                        ValueQuery(item="primary", path="value"),
                        ValueQuery(item="fallback", path="value"),
                    ]
                ),
            )
        ],
    )

    product = materialize_extraction_product(spec, tmp_path)
    assert product.parameters["value"] == 5

    persisted_payload = product.model_dump(mode="json")
    persisted_payload["parameters"]["value"] = 6
    persisted_payload.pop("product_identity_hash")
    with pytest.raises(DataProductDrift) as excinfo:
        verify_extraction_product(spec, persisted_payload, tmp_path)
    drift = excinfo.value
    assert drift.output_path == "value"
    assert drift.persisted_value == 6
    assert drift.source_value == 5


def test_extraction_contracts_remain_strict_for_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SourceBinding.model_validate(
            {
                "alias": "src",
                "kind": "manifest",
                "uri": "source.json",
                "unknown": True,
            }
        )
    with pytest.raises(ValidationError):
        FieldMapping.model_validate(
            {
                "output_path": "value",
                "query": {"item": "src"},
                "unknown": True,
            }
        )
