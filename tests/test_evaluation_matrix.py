from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    EvaluationRunMatrixSpec,
    execute_evaluation_run_matrix,
    materialize_evaluation_run_matrix,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.contracts.expressions import ValueQuery
from feedbax.contracts.manifest import (
    EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_ID,
    EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1,
    EvaluationRunSpec,
    OverridePatch,
    TrainingSweepAxisGroup,
    canonical_json_bytes,
    load_manifest,
    migrate_spec_payload,
    sha256_bytes,
    spec_payload,
)
from feedbax.contracts.matrix_core import (
    MatrixAxis,
    MatrixAxisValue,
    MatrixRow,
    RowDerivation,
    derive_row_path,
    expand_matrix_axes,
    ordered_index_product,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry


def _axis_matrix_payload(tmp_path: Path) -> dict:
    base = {
        "evaluation_type": "example.evaluate",
        "training_run_ids": ["train-a"],
        "inputs": [],
        "params": {"gain": 0.0, "mode": "unset"},
    }
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    return {
        "schema_id": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "base": {"ref": "base.json", "sha256": sha256_bytes(canonical_json_bytes(base))},
        "axes": [
            {
                "id": "gain",
                "values": [
                    {"id": "low", "deltas": [{"path": "params.gain", "value": 1.0}]},
                    {"id": "high", "deltas": [{"path": "params.gain", "value": 2.0}]},
                ],
            },
            {
                "id": "mode",
                "values": [
                    {"id": "a", "deltas": [{"path": "params.mode", "value": "a"}]},
                    {"id": "b", "deltas": [{"path": "params.mode", "value": "b"}]},
                ],
            },
        ],
    }


def _matrix() -> EvaluationRunMatrixSpec:
    return EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type="example.evaluate",
            training_run_ids=["train-a"],
            params={"gain": 1.0, "derived_gain": 0.0},
        ),
        rows=[
            MatrixRow(
                row_id="control",
                deltas=[OverridePatch(path="params.gain", value=2.0)],
                derivations=[
                    RowDerivation(
                        output_path="params.derived_gain",
                        query=ValueQuery(item="row", path="params.gain"),
                    )
                ],
            ),
            MatrixRow(
                row_id="treatment",
                deltas=[OverridePatch(path="params.gain", value=3.0)],
                output_path="custom/result.json",
                spec_path="custom/request.json",
            ),
        ],
    )


def test_evaluation_matrix_applies_deltas_before_per_row_derivation() -> None:
    rows = materialize_evaluation_run_matrix(_matrix())

    assert [row.row_id for row in rows] == ["control", "treatment"]
    assert rows[0].payload.params == {"gain": 2.0, "derived_gain": 2.0}
    assert rows[1].payload.params == {"gain": 3.0, "derived_gain": 0.0}
    assert rows[0].output_path == "control/output.json"
    assert rows[0].spec_path == "control/spec.json"
    assert rows[1].output_path == "custom/result.json"
    assert rows[1].spec_path == "custom/request.json"


def test_evaluation_matrix_requires_unique_path_safe_rows() -> None:
    with pytest.raises(ValidationError, match="row_id values must be unique"):
        EvaluationRunMatrixSpec(
            base=EvaluationRunSpec(evaluation_type="example.evaluate"),
            rows=[MatrixRow(row_id="same"), MatrixRow(row_id="same")],
        )
    with pytest.raises(ValidationError, match="not path-safe"):
        MatrixRow(row_id="not/a/row")
    with pytest.raises(ValueError, match="must be relative"):
        derive_row_path("row", explicit_path="/tmp/output.json")


def test_evaluation_matrix_schema_accepts_current_and_rejects_legacy() -> None:
    payload = _matrix().model_dump(mode="json")
    result = default_spec_registry.migrate("EvaluationRunMatrixSpec", payload)

    assert result.schema_id == EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID
    assert result.target_version == EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION
    assert not result.migrated

    legacy = {**payload, "schema_version": "feedbax.spec.evaluation_run_matrix.v0"}
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent"):
        default_spec_registry.migrate("EvaluationRunMatrixSpec", legacy)


def test_evaluation_matrix_v1_explicit_rows_migrate_to_v2_unchanged() -> None:
    payload = _matrix().model_dump(mode="json")
    payload["schema_version"] = EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1

    result = default_spec_registry.migrate("EvaluationRunMatrixSpec", payload)

    assert result.migrated
    assert result.target_version == EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION
    assert result.payload["base"] == payload["base"]
    assert result.payload["rows"] == payload["rows"]
    assert EvaluationRunMatrixSpec.model_validate(result.payload).rows == _matrix().rows

    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent"):
        default_spec_registry.migrate(
            "EvaluationAxisExpansionProvenance",
            {
                "schema_id": EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_ID,
                "schema_version": "feedbax.manifest.evaluation_axis_expansion_provenance.v0",
            },
        )


def test_generic_spec_payload_transports_v1_and_axis_authored_v2(tmp_path: Path) -> None:
    explicit_v1 = _matrix().model_dump(mode="json")
    explicit_v1["schema_version"] = EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1
    migrated = migrate_spec_payload(
        {
            "kind": "EvaluationRunMatrixSpec",
            "inline": explicit_v1,
            "schema_id": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
            "schema_version": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1,
        }
    )
    axis_v2 = _axis_matrix_payload(tmp_path)
    transported = spec_payload("EvaluationRunMatrixSpec", axis_v2)

    assert migrated.schema_version == EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION
    assert migrated.inline["base"] == explicit_v1["base"]
    assert migrated.inline["rows"] == explicit_v1["rows"]
    assert transported.inline["base"] == axis_v2["base"]
    assert transported.inline["axes"] == axis_v2["axes"]


def test_evaluation_matrix_schema_identity_is_pinned() -> None:
    payload = _matrix().model_dump(mode="json")
    payload["schema_id"] = "example.spec.evaluation_matrix"
    with pytest.raises(ValidationError, match="unsupported EvaluationRunMatrixSpec schema_id"):
        EvaluationRunMatrixSpec.model_validate(payload)


def test_evaluation_matrix_executes_through_harness(tmp_path: Path) -> None:
    def recipe(spec, _root, _states_path, _execution_context):
        return EvaluationRecipeResult(summary_metrics={"gain": spec.params["gain"]})

    register_evaluation_recipe("example.evaluate", recipe)
    try:
        payload = _matrix().model_dump(mode="json")
        payload["schema_version"] = EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1
        result = execute_evaluation_run_matrix(payload, root=tmp_path)
    finally:
        unregister_evaluation_recipe("example.evaluate")

    assert [row.row_id for row in result.rows] == ["control", "treatment"]
    assert all(row.manifest_path is not None and row.manifest_path.exists() for row in result.rows)
    assert result.rows[0].result.metadata["matrix_harness"]["row_id"] == "control"
    assert "regeneration_spec" in result.rows[0].result.metadata["matrix_harness"]
    assert {artifact.role for artifact in result.rows[0].result.artifacts} >= {
        "regeneration_spec",
        "resolved_row_spec",
    }


def test_axis_product_matches_equivalent_explicit_rows_and_hashes(tmp_path: Path) -> None:
    axis_rows = materialize_evaluation_run_matrix(
        _axis_matrix_payload(tmp_path), repo_root=tmp_path
    )
    explicit = EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type="example.evaluate",
            training_run_ids=["train-a"],
            params={"gain": 0.0, "mode": "unset"},
        ),
        rows=[
            MatrixRow(
                row_id=f"gain-{gain_id}--mode-{mode_id}",
                deltas=[
                    OverridePatch(path="params.gain", value=gain),
                    OverridePatch(path="params.mode", value=mode_id),
                ],
            )
            for gain_id, gain in (("low", 1.0), ("high", 2.0))
            for mode_id in ("a", "b")
        ],
    )
    explicit_rows = materialize_evaluation_run_matrix(explicit)

    assert [row.row_id for row in axis_rows] == [row.row_id for row in explicit_rows]
    assert [row.payload for row in axis_rows] == [row.payload for row in explicit_rows]
    assert [
        sha256_bytes(canonical_json_bytes(row.payload)) for row in axis_rows
    ] == [sha256_bytes(canonical_json_bytes(row.payload)) for row in explicit_rows]


def test_axis_product_manifest_records_canonical_expansion_provenance(
    tmp_path: Path,
) -> None:
    def recipe(_spec, _root, _states_path, _execution_context):
        return EvaluationRecipeResult()

    register_evaluation_recipe("example.evaluate", recipe)
    try:
        result = execute_evaluation_run_matrix(
            _axis_matrix_payload(tmp_path),
            root=tmp_path / "runs",
            repo_root=tmp_path,
        )
        explicit_result = execute_evaluation_run_matrix(
            _matrix(), root=tmp_path / "explicit"
        )
    finally:
        unregister_evaluation_recipe("example.evaluate")

    expected_order = ["gain-low--mode-a", "gain-low--mode-b", "gain-high--mode-a", "gain-high--mode-b"]
    for row in result.rows:
        manifest = load_manifest(row.manifest_path)
        provenance = manifest.metadata["matrix_harness"]["axis_expansion"]
        assert provenance["schema_id"] == EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_ID
        assert provenance["schema_version"] == EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_VERSION
        assert provenance["canonical_row_order"] == expected_order
        assert provenance["coordinates"][0]["value_ids"] == {"gain": "low", "mode": "a"}
        assert provenance["canonical_payload_sha256"][row.row_id] == sha256_bytes(
            canonical_json_bytes(row.resolved)
        )
    assert "axis_expansion" not in explicit_result.rows[0].result.metadata["matrix_harness"]


def test_axis_contract_rejects_duplicate_ids_and_incomplete_products(monkeypatch) -> None:
    with pytest.raises(ValidationError, match="value ids must be unique"):
        MatrixAxis(id="axis", values=[MatrixAxisValue(id="same"), MatrixAxisValue(id="same")])
    with pytest.raises(ValidationError, match="axes id values must be unique"):
        EvaluationRunMatrixSpec(
            base={"ref": "base.json", "sha256": "0" * 64},
            axes=[
                MatrixAxis(id="same", values=[MatrixAxisValue(id="a")]),
                MatrixAxis(id="same", values=[MatrixAxisValue(id="b")]),
            ],
        )
    with pytest.raises(ValidationError, match="at least 1 item"):
        MatrixAxis(id="empty", values=[])
    with pytest.raises(ValidationError, match="requires axes"):
        EvaluationRunMatrixSpec(base={"ref": "base.json", "sha256": "0" * 64})

    import feedbax.contracts.matrix_core as matrix_core

    monkeypatch.setattr(matrix_core, "ordered_index_product", lambda _lengths: [{"a": 0}])
    with pytest.raises(ValueError, match="incomplete coordinate"):
        matrix_core.expand_matrix_axes(
            [
                MatrixAxis(id="a", values=[MatrixAxisValue(id="x")]),
                MatrixAxis(id="b", values=[MatrixAxisValue(id="y")]),
            ]
        )


def test_axis_contract_rejects_collisions_deltas_and_non_json_values() -> None:
    collision_axes = [
        MatrixAxis(
            id="a",
            values=[MatrixAxisValue(id="x"), MatrixAxisValue(id="x--b-y")],
        ),
        MatrixAxis(
            id="b",
            values=[MatrixAxisValue(id="y--b-z"), MatrixAxisValue(id="z")],
        ),
    ]
    with pytest.raises(ValueError, match="row_id collision"):
        expand_matrix_axes(collision_axes)

    duplicate_path_axes = [
        MatrixAxis(
            id="a",
            values=[MatrixAxisValue(id="x", deltas=[OverridePatch(path="params.gain", value=1)])],
        ),
        MatrixAxis(
            id="b",
            values=[MatrixAxisValue(id="y", deltas=[OverridePatch(path="params.gain", value=2)])],
        ),
    ]
    with pytest.raises(ValueError, match="duplicate delta paths"):
        expand_matrix_axes(duplicate_path_axes)
    with pytest.raises(ValidationError, match="finite JSON numbers"):
        MatrixAxisValue(id="bad", deltas=[OverridePatch(path="params.gain", value=float("nan"))])
    with pytest.raises(ValidationError, match="non-JSON value"):
        MatrixAxisValue(id="bad", deltas=[OverridePatch(path="params.gain", value=object())])


@pytest.mark.parametrize(
    ("ref", "digest", "message"),
    [
        ("missing.json", "0" * 64, "cannot load content-pinned JSON base"),
        ("../outside.json", "0" * 64, "escapes repo_root"),
        ("base.json", "0" * 64, "hash mismatch"),
    ],
)
def test_axis_matrix_rejects_unavailable_or_untrusted_pinned_base(
    tmp_path: Path, ref: str, digest: str, message: str
) -> None:
    (tmp_path / "base.json").write_text(
        json.dumps({"evaluation_type": "example.evaluate"}), encoding="utf-8"
    )
    payload = _axis_matrix_payload(tmp_path)
    payload["base"] = {"ref": ref, "sha256": digest}

    with pytest.raises(ValueError, match=message):
        materialize_evaluation_run_matrix(payload, repo_root=tmp_path)


def test_axis_matrix_requires_explicit_repo_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="content-pinned JSON base requires repo_root"):
        materialize_evaluation_run_matrix(_axis_matrix_payload(tmp_path))


def test_axis_matrix_rejects_delta_to_missing_path(tmp_path: Path) -> None:
    payload = _axis_matrix_payload(tmp_path)
    payload["axes"][0]["values"][0]["deltas"][0]["path"] = "params.missing"

    with pytest.raises(ValueError, match="replace delta path is missing"):
        materialize_evaluation_run_matrix(payload, repo_root=tmp_path)


def test_training_cross_group_delegates_to_matrix_core_and_matches_axis_order(
    monkeypatch,
) -> None:
    import feedbax.training.run_matrix as training_matrix

    calls: list[list[tuple[str, int]]] = []

    def traced(lengths):
        calls.append(list(lengths))
        return ordered_index_product(lengths)

    monkeypatch.setattr(training_matrix, "ordered_index_product", traced)
    training = training_matrix._expand_group(
        TrainingSweepAxisGroup(id="all", axes=["gain", "mode"], mode="cross"),
        {"gain": 2, "mode": 2},
    )
    evaluation = [
        coordinate.value_indices
        for coordinate in expand_matrix_axes(
            [
                MatrixAxis(id="gain", values=[MatrixAxisValue(id="low"), MatrixAxisValue(id="high")]),
                MatrixAxis(id="mode", values=[MatrixAxisValue(id="a"), MatrixAxisValue(id="b")]),
            ]
        )
    ]

    assert calls == [[("gain", 2), ("mode", 2)]]
    assert training == evaluation


def test_public_exports_include_matrix_and_harness_apis() -> None:
    import feedbax.analysis as analysis
    import feedbax.contracts as contracts

    assert {
        "EvaluationRunMatrixSpec",
        "MatrixMaterializerHarness",
        "execute_evaluation_run_matrix",
    } <= set(analysis.__all__)
    assert {"MatrixRow", "RowDerivation", "RowMatrixSpec"} <= set(contracts.__all__)
