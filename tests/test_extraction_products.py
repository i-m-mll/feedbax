from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from feedbax.contracts.expressions import Coerce, Compare, Select, ValueQuery
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
