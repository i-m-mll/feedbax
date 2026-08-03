from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from feedbax.analysis.evaluation import (
    EvaluationAuthoringSchema,
    EvaluationRecipeResult,
    EvaluationRecipeExecutionError,
    EvaluationRunMatrixSpec,
    compile_evaluation_run_matrix,
    execute_evaluation_run_matrix,
    execute_evaluation_run_spec,
    materialize_evaluation_run_matrix,
    resolve_staged_evaluation_prerequisite,
)
from feedbax.analysis.evaluation_inputs import resolve_evaluation_inputs
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedArtifactProviderRootBinding,
    StagedExecutionContextError,
)
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.evaluation_states import (
    EvaluationStatesProvenanceMismatch,
    EvaluationStatesSchemaMismatch,
    store_evaluation_states_artifact,
)
from feedbax.contracts.expressions import ValueQuery
from feedbax.contracts.manifest import (
    EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_ID,
    EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V2,
    EvaluationRunManifest,
    EvaluationRunSpec,
    OverridePatch,
    ParentRef,
    SpecPayload,
    StagedEvaluationPrerequisite,
    TrainingSweepAxisGroup,
    TrainingRunManifest,
    canonical_json_bytes,
    load_manifest,
    migrate_spec_payload,
    sha256_bytes,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.matrix_core import (
    MATRIX_AXIS_VALUE_GENERATOR_SCHEMA_VERSION,
    MatrixAxis,
    MatrixAxisValue,
    MatrixAxisValueGenerator,
    MatrixRow,
    RowDerivation,
    derive_row_path,
    expand_matrix_axes,
    ordered_index_product,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.staged_execution import (
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedExecutionDescriptor,
)
from feedbax.persistence.artifact_custody import open_immutable_artifact_blob_provider


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
                    ),
                    RowDerivation(
                        output_path="params.derived_gain_copy",
                        query=ValueQuery(item="row", path="params.derived_gain"),
                    ),
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


def test_evaluation_matrix_applies_deltas_before_per_row_derivation(evaluation_registry) -> None:
    rows = materialize_evaluation_run_matrix(_matrix(), registry=evaluation_registry)

    assert [row.row_id for row in rows] == ["control", "treatment"]
    assert rows[0].payload.params == {
        "gain": 2.0,
        "derived_gain": 2.0,
        "derived_gain_copy": 2.0,
    }
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

    v1 = {key: value for key, value in payload.items() if key != "staged_parents"}
    v1["schema_version"] = EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1
    migrated = default_spec_registry.migrate("EvaluationRunMatrixSpec", v1)
    assert migrated.migrated
    assert migrated.payload["staged_parents"] == {}
    assert migrated.target_version == EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION

    legacy = {**payload, "schema_version": "feedbax.spec.evaluation_run_matrix.v0"}
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent"):
        default_spec_registry.migrate("EvaluationRunMatrixSpec", legacy)


def test_evaluation_matrix_v1_explicit_rows_migrate_to_v3_unchanged() -> None:
    payload = _matrix().model_dump(mode="json")
    payload["schema_version"] = EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1

    result = default_spec_registry.migrate("EvaluationRunMatrixSpec", payload)

    assert result.migrated
    assert result.target_version == EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION
    assert result.payload["base"] == payload["base"]
    assert result.payload["rows"] == payload["rows"]
    assert result.payload["staged_parents"] == {}
    assert [record.target_schema_version for record in result.migration_records] == [
        EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V2,
        EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    ]
    assert EvaluationRunMatrixSpec.model_validate(result.payload).rows == _matrix().rows

    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent"):
        default_spec_registry.migrate(
            "EvaluationAxisExpansionProvenance",
            {
                "schema_id": EVALUATION_AXIS_EXPANSION_PROVENANCE_SCHEMA_ID,
                "schema_version": "feedbax.manifest.evaluation_axis_expansion_provenance.v0",
            },
        )


def test_evaluation_matrix_v2_variants_migrate_to_v3_unchanged(tmp_path: Path) -> None:
    explicit = _matrix().model_dump(mode="json")
    explicit["schema_version"] = EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V2
    axis = _axis_matrix_payload(tmp_path)
    axis["schema_version"] = EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V2
    axis["staged_parents"] = {}

    for payload in (explicit, axis):
        result = default_spec_registry.migrate("EvaluationRunMatrixSpec", payload)

        assert [record.target_schema_version for record in result.migration_records] == [
            EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION
        ]
        assert result.payload["base"] == payload["base"]
        assert result.payload.get("rows", []) == payload.get("rows", [])
        assert result.payload.get("axes", []) == payload.get("axes", [])
        assert result.payload["staged_parents"] == payload["staged_parents"]


def test_generic_spec_payload_transports_v1_and_axis_authored_v3(tmp_path: Path) -> None:
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
    axis_v3 = _axis_matrix_payload(tmp_path)
    transported = spec_payload("EvaluationRunMatrixSpec", axis_v3)

    assert migrated.schema_version == EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION
    assert migrated.inline["base"] == explicit_v1["base"]
    assert migrated.inline["rows"] == explicit_v1["rows"]
    assert transported.inline["base"] == axis_v3["base"]
    assert transported.inline["axes"] == axis_v3["axes"]


def test_public_materializer_migrates_serialized_v1_context_free_matrix(
    evaluation_registry,
) -> None:
    payload = _matrix().model_dump(mode="json")
    payload.pop("staged_parents")
    payload["schema_version"] = EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1

    rows = materialize_evaluation_run_matrix(payload, registry=evaluation_registry)

    assert [row.row_id for row in rows] == ["control", "treatment"]
    assert rows[0].payload.params["gain"] == 2.0


def test_evaluation_matrix_schema_identity_is_pinned() -> None:
    payload = _matrix().model_dump(mode="json")
    payload["schema_id"] = "example.spec.evaluation_matrix"
    with pytest.raises(ValidationError, match="unsupported EvaluationRunMatrixSpec schema_id"):
        EvaluationRunMatrixSpec.model_validate(payload)


def test_evaluation_matrix_executes_through_harness(tmp_path: Path, evaluation_registry) -> None:
    def recipe(spec, _root, _states_path, _execution_context):
        assert not _execution_context.parent_execution_locations
        return EvaluationRecipeResult(summary_metrics={"gain": spec.params["gain"]})

    evaluation_registry.register("example.evaluate", recipe)
    payload = _matrix().model_dump(mode="json")
    payload["schema_version"] = EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1
    result = execute_evaluation_run_matrix(payload, registry=evaluation_registry, root=tmp_path)

    assert [row.row_id for row in result.rows] == ["control", "treatment"]
    assert all(row.manifest_path is not None and row.manifest_path.exists() for row in result.rows)
    assert result.rows[0].result.metadata["matrix_harness"]["row_id"] == "control"
    assert "regeneration_spec" in result.rows[0].result.metadata["matrix_harness"]
    assert {artifact.role for artifact in result.rows[0].result.artifacts} >= {
        "regeneration_spec",
        "resolved_row_spec",
    }


def _generator_axis_base(tmp_path: Path) -> dict:
    base = {
        "evaluation_type": "example.evaluate",
        "training_run_ids": ["train-a"],
        "inputs": [],
        "params": {"target_index": -1},
    }
    (tmp_path / "generator-base.json").write_text(json.dumps(base), encoding="utf-8")
    return base


def _generator_axis_matrix(tmp_path: Path, *, generated: bool) -> EvaluationRunMatrixSpec:
    base = _generator_axis_base(tmp_path)
    if generated:
        axis = MatrixAxis(
            id="target",
            generator=MatrixAxisValueGenerator(
                path="params.target_index",
                start=0,
                stop=4,
                step=1,
                id_format="{value:03d}",
            ),
        )
    else:
        axis = MatrixAxis(
            id="target",
            values=[
                MatrixAxisValue(
                    id=f"{index:03d}",
                    deltas=[OverridePatch(path="params.target_index", value=index)],
                )
                for index in range(4)
            ],
        )
    return EvaluationRunMatrixSpec(
        base={
            "ref": "generator-base.json",
            "sha256": sha256_bytes(canonical_json_bytes(base)),
        },
        axes=[axis],
    )


def test_generator_axis_matches_hand_enumerated_axis_rows_and_hashes(
    tmp_path: Path, evaluation_registry
) -> None:
    generated = _generator_axis_matrix(tmp_path, generated=True)
    enumerated = _generator_axis_matrix(tmp_path, generated=False)

    generated_compiled = compile_evaluation_run_matrix(
        generated, repo_root=tmp_path, registry=evaluation_registry
    )
    enumerated_compiled = compile_evaluation_run_matrix(
        enumerated, repo_root=tmp_path, registry=evaluation_registry
    )
    assert generated_compiled.model_dump(mode="json") == enumerated_compiled.model_dump(mode="json")

    generated_rows = materialize_evaluation_run_matrix(
        generated, repo_root=tmp_path, registry=evaluation_registry
    )
    enumerated_rows = materialize_evaluation_run_matrix(
        enumerated, repo_root=tmp_path, registry=evaluation_registry
    )
    assert [row.row_id for row in generated_rows] == [
        "target-000",
        "target-001",
        "target-002",
        "target-003",
    ]
    assert [row.row_id for row in generated_rows] == [row.row_id for row in enumerated_rows]
    assert [
        sha256_bytes(canonical_json_bytes(row.payload.model_dump(mode="json")))
        for row in generated_rows
    ] == [
        sha256_bytes(canonical_json_bytes(row.payload.model_dump(mode="json")))
        for row in enumerated_rows
    ]

    enumerated_axis = enumerated.model_dump(mode="json", exclude_none=True)["axes"][0]
    assert "generator" not in enumerated_axis
    assert expand_matrix_axes(generated.axes) == expand_matrix_axes(enumerated.axes)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"step": 0}, "step must be non-zero"),
        ({"start": 4, "stop": 0}, "produces no values"),
        ({"id_format": "{bogus}"}, "is not one of"),
        ({"id_format": "fixed"}, "must reference at least one"),
        ({"id_format": "{value:{index}d}"}, "must not nest replacement fields"),
        ({"id_format": "{value:+d}"}, "not path-safe"),
        ({"id_format": "target-{"}, "not a valid format string"),
        ({"path": "params..index"}, "not dotted-path-like"),
        (
            {"schema_version": "feedbax.spec.matrix_axis_value_generator.v0"},
            "unsupported axis value generator schema_version",
        ),
        (
            {"schema_id": "feedbax.spec.other_generator"},
            "unsupported axis value generator schema_id",
        ),
        ({"unknown_field": 1}, "Extra inputs are not permitted"),
    ],
)
def test_axis_value_generator_fails_closed_on_malformed_declarations(
    overrides: dict, message: str
) -> None:
    declaration = {
        "path": "params.target_index",
        "start": 0,
        "stop": 4,
        "step": 1,
        "id_format": "{value:03d}",
        **overrides,
    }
    with pytest.raises(ValidationError, match=message):
        MatrixAxisValueGenerator(**declaration)


def test_axis_rejects_both_or_neither_value_declaration() -> None:
    generator = MatrixAxisValueGenerator(
        path="params.target_index", start=0, stop=2, id_format="{index}"
    )
    with pytest.raises(ValidationError, match="cannot declare both values and a generator"):
        MatrixAxis(id="target", values=[MatrixAxisValue(id="000")], generator=generator)
    with pytest.raises(ValidationError, match="requires enumerated values or a generator"):
        MatrixAxis(id="target")


def test_generator_axis_manifest_records_generator_as_expansion_authority(
    tmp_path: Path,
    evaluation_registry,
) -> None:
    def recipe(_spec, _root, _states_path, _execution_context):
        return EvaluationRecipeResult()

    matrix = _generator_axis_matrix(tmp_path, generated=True)
    evaluation_registry.register("example.evaluate", recipe)
    result = execute_evaluation_run_matrix(
        matrix, registry=evaluation_registry, root=tmp_path / "runs", repo_root=tmp_path
    )

    for row in result.rows:
        provenance = load_manifest(row.manifest_path).metadata["matrix_harness"]["axis_expansion"]
        assert provenance["ordered_axes"] == [
            {
                "axis_id": "target",
                "value_ids": ["000", "001", "002", "003"],
                "generator": matrix.axes[0].generator.model_dump(mode="json"),
            }
        ]
        assert provenance["ordered_axes"][0]["generator"]["schema_version"] == (
            MATRIX_AXIS_VALUE_GENERATOR_SCHEMA_VERSION
        )
        assert provenance["canonical_row_order"] == [
            "target-000",
            "target-001",
            "target-002",
            "target-003",
        ]


def test_axis_product_matches_equivalent_explicit_rows_and_hashes(
    tmp_path: Path, evaluation_registry
) -> None:
    axis_rows = materialize_evaluation_run_matrix(
        _axis_matrix_payload(tmp_path), registry=evaluation_registry, repo_root=tmp_path
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
    explicit_rows = materialize_evaluation_run_matrix(explicit, registry=evaluation_registry)

    assert [row.row_id for row in axis_rows] == [row.row_id for row in explicit_rows]
    assert [row.payload for row in axis_rows] == [row.payload for row in explicit_rows]
    assert [sha256_bytes(canonical_json_bytes(row.payload)) for row in axis_rows] == [
        sha256_bytes(canonical_json_bytes(row.payload)) for row in explicit_rows
    ]


def test_axis_product_manifest_records_canonical_expansion_provenance(
    tmp_path: Path,
    evaluation_registry,
) -> None:
    def recipe(_spec, _root, _states_path, _execution_context):
        return EvaluationRecipeResult()

    evaluation_registry.register("example.evaluate", recipe)
    result = execute_evaluation_run_matrix(
        _axis_matrix_payload(tmp_path),
        registry=evaluation_registry,
        root=tmp_path / "runs",
        repo_root=tmp_path,
    )
    explicit_result = execute_evaluation_run_matrix(
        _matrix(), registry=evaluation_registry, root=tmp_path / "explicit"
    )

    expected_order = [
        "gain-low--mode-a",
        "gain-low--mode-b",
        "gain-high--mode-a",
        "gain-high--mode-b",
    ]
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
    with pytest.raises(ValidationError, match="requires enumerated values or a generator"):
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


class _ChannelTaxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sensory_standard: Literal["feedback_disturbance"]
    process_standard: Literal["disturbance"]


class _ExpectedZeros(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    command_disturbance: Literal["task.plant.command_dim"]
    feedback_disturbance: Literal["task.plant.observation_dim"]


class _ComparatorAuthoringParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    arm_id: Literal["trained", "extlqg", "hinf"]
    target_index: int
    channels: _ChannelTaxonomy
    expected_zeros: _ExpectedZeros


def _schema_matrix(
    tmp_path: Path,
    *,
    command_dimension: str = "task.plant.command_dim",
    arms: tuple[str, ...] = ("extlqg", "hinf"),
) -> EvaluationRunMatrixSpec:
    base = EvaluationRunSpec(
        evaluation_type="example.schema_validated",
        params={
            "arm_id": "extlqg",
            "target_index": 0,
            "channels": {
                "sensory_standard": "feedback_disturbance",
                "process_standard": "disturbance",
            },
            "expected_zeros": {
                "command_disturbance": command_dimension,
                "feedback_disturbance": "task.plant.observation_dim",
            },
        },
    ).model_dump(mode="json", exclude_none=True)
    base_path = tmp_path / f"schema-base-{len(arms)}-{command_dimension.rsplit('.', 1)[-1]}.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    return EvaluationRunMatrixSpec(
        base={
            "ref": base_path.name,
            "sha256": sha256_bytes(canonical_json_bytes(base)),
        },
        rows=[],
        axes=[
            MatrixAxis(
                id="arm",
                values=[
                    MatrixAxisValue(
                        id=arm,
                        deltas=[OverridePatch(path="params.arm_id", value=arm)],
                    )
                    for arm in arms
                ],
            ),
            MatrixAxis(
                id="target",
                values=[
                    MatrixAxisValue(
                        id=str(index),
                        deltas=[OverridePatch(path="params.target_index", value=index)],
                    )
                    for index in range(2)
                ],
            ),
        ],
    )


def test_runtime_authoring_schema_validates_taxonomy_grid_without_changing_hashes(
    tmp_path: Path,
    evaluation_registry,
) -> None:
    matrix = _schema_matrix(tmp_path)
    before = compile_evaluation_run_matrix(matrix, registry=evaluation_registry, repo_root=tmp_path)
    authored_bytes = canonical_json_bytes(matrix.model_dump(mode="json", exclude_none=True))
    compiled_bytes = canonical_json_bytes(before.model_dump(mode="json", exclude_none=True))
    schema = EvaluationAuthoringSchema(
        schema_id="example.spec.evaluation.schema_validated",
        schema_version="example.spec.evaluation.schema_validated.v1",
        params_model=_ComparatorAuthoringParams,
        axis_profiles=(
            {"arm": ("extlqg", "hinf"), "target": ("0", "1")},
            {"arm": ("trained", "extlqg", "hinf"), "target": ("0", "1")},
        ),
    )
    evaluation_registry.register_authoring_schema("example.schema_validated", schema)
    after = compile_evaluation_run_matrix(matrix, registry=evaluation_registry, repo_root=tmp_path)
    assert canonical_json_bytes(matrix.model_dump(mode="json", exclude_none=True)) == authored_bytes
    assert canonical_json_bytes(after.model_dump(mode="json", exclude_none=True)) == compiled_bytes

    invalid = _schema_matrix(tmp_path, command_dimension="wrong")
    with pytest.raises(ValueError, match="do not match schema"):
        compile_evaluation_run_matrix(invalid, registry=evaluation_registry, repo_root=tmp_path)

    wrong_grid = matrix.model_copy(deep=True)
    wrong_grid.axes[1].values.pop()
    with pytest.raises(ValueError, match="do not match schema"):
        compile_evaluation_run_matrix(wrong_grid, registry=evaluation_registry, repo_root=tmp_path)


def test_authoring_schema_registration_rejects_duplicate_and_conflicting_claims(
    tmp_path: Path, evaluation_registry
) -> None:
    profiles = (
        {"arm": ("extlqg", "hinf"), "target": ("0", "1")},
        {"arm": ("trained", "extlqg", "hinf"), "target": ("0", "1")},
    )

    def register_hook(*, reverse_profiles: bool) -> None:
        evaluation_registry.register_authoring_schema(
            "example.schema_validated",
            EvaluationAuthoringSchema(
                schema_id="example.spec.evaluation.schema_validated",
                schema_version="example.spec.evaluation.schema_validated.v1",
                params_model=_ComparatorAuthoringParams,
                axis_profiles=profiles[::-1] if reverse_profiles else profiles,
            ),
        )

    register_hook(reverse_profiles=False)
    assert (
        len(
            compile_evaluation_run_matrix(
                _schema_matrix(tmp_path), registry=evaluation_registry, repo_root=tmp_path
            ).rows
        )
        == 4
    )
    assert (
        len(
            compile_evaluation_run_matrix(
                _schema_matrix(tmp_path, arms=("trained", "extlqg", "hinf")),
                registry=evaluation_registry,
                repo_root=tmp_path,
            ).rows
        )
        == 6
    )
    with pytest.raises(ValueError, match="already registered"):
        register_hook(reverse_profiles=True)


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
    tmp_path: Path,
    ref: str,
    digest: str,
    message: str,
    evaluation_registry,
) -> None:
    (tmp_path / "base.json").write_text(
        json.dumps({"evaluation_type": "example.evaluate"}), encoding="utf-8"
    )
    payload = _axis_matrix_payload(tmp_path)
    payload["base"] = {"ref": ref, "sha256": digest}

    with pytest.raises(ValueError, match=message):
        materialize_evaluation_run_matrix(payload, repo_root=tmp_path, registry=evaluation_registry)


def test_axis_matrix_requires_explicit_repo_root(tmp_path: Path, evaluation_registry) -> None:
    with pytest.raises(ValueError, match="content-pinned JSON base requires repo_root"):
        materialize_evaluation_run_matrix(
            _axis_matrix_payload(tmp_path), registry=evaluation_registry
        )


def test_axis_matrix_rejects_delta_to_missing_path(tmp_path: Path, evaluation_registry) -> None:
    payload = _axis_matrix_payload(tmp_path)
    payload["axes"][0]["values"][0]["deltas"][0]["path"] = "params.missing"

    with pytest.raises(ValueError, match="replace delta path is missing"):
        materialize_evaluation_run_matrix(payload, repo_root=tmp_path, registry=evaluation_registry)


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
                MatrixAxis(
                    id="gain", values=[MatrixAxisValue(id="low"), MatrixAxisValue(id="high")]
                ),
                MatrixAxis(id="mode", values=[MatrixAxisValue(id="a"), MatrixAxisValue(id="b")]),
            ]
        )
    ]

    assert calls == [[("gain", 2), ("mode", 2)]]
    assert training == evaluation


def test_direct_single_run_keeps_empty_staged_context(tmp_path: Path, evaluation_registry) -> None:
    observed = []

    def recipe(_spec, _root, _states_path, execution_context):
        observed.append(execution_context)
        return EvaluationRecipeResult(summary_metrics={"direct": True})

    evaluation_registry.register("example.direct", recipe)
    manifest, path = execute_evaluation_run_spec(
        EvaluationRunSpec(evaluation_type="example.direct"),
        registry=evaluation_registry,
        root=tmp_path,
    )

    assert observed == [EMPTY_STAGED_EXECUTION_CONTEXT]
    assert manifest.status == "completed"
    assert path.exists()


def test_public_exports_include_matrix_and_harness_apis() -> None:
    import feedbax.analysis as analysis
    import feedbax.contracts as contracts

    assert {
        "EvaluationRunMatrixSpec",
        "MatrixMaterializerHarness",
        "execute_evaluation_run_matrix",
        "resolve_staged_evaluation_prerequisite",
    } <= set(analysis.__all__)
    assert {"MatrixRow", "RowDerivation", "RowMatrixSpec"} <= set(contracts.__all__)


def _evaluation_manifest(artifact) -> EvaluationRunManifest:
    return EvaluationRunManifest(
        id="feedbax-evaluation-run:paired-bank",
        status="completed",
        evaluation_spec=SpecPayload(
            kind="EvaluationRunSpec",
            schema_id="feedbax.spec.evaluation_run",
            schema_version="feedbax.spec.evaluation_run.v1",
            inline={
                "schema_version": "feedbax.spec.evaluation_run.v1",
                "evaluation_type": "example.bank",
                "training_run_ids": [],
                "inputs": [],
                "params": {},
            },
        ),
        artifacts=[artifact],
    )


def _staged_matrix(
    training: ParentRef,
    bank: ParentRef,
    *,
    artifact_provider: str | None = None,
) -> EvaluationRunMatrixSpec:
    bank_prerequisite = StagedEvaluationPrerequisite(
        parent=bank,
        artifact_provider=artifact_provider,
    )
    return EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type="example.staged_matrix",
            inputs=[training],
            params={"staged_prerequisites": {"paired_bank": bank_prerequisite}},
        ),
        rows=[MatrixRow(row_id="row-a"), MatrixRow(row_id="row-b")],
        staged_parents={
            "training": StagedEvaluationPrerequisite(
                parent=training,
                artifact_provider=artifact_provider,
            ),
            "paired_bank": bank_prerequisite,
        },
    )


def _run_staged_matrix(
    matrix: EvaluationRunMatrixSpec,
    *,
    evaluation_registry,
    output_root: Path,
    **kwargs,
):
    observed: list[tuple[Path, str, list[int]]] = []

    def recipe(spec, root, _states_path, execution_context):
        training = resolve_evaluation_inputs(
            spec,
            manifest_root=root,
            execution_context=execution_context,
        )[0]
        states = resolve_staged_evaluation_prerequisite(
            spec.params["staged_prerequisites"]["paired_bank"],
            execution_context=execution_context,
        )
        observed.append((root, training.id, states["pair"].tolist()))
        return EvaluationRecipeResult(summary_metrics={"pair_count": len(states["pair"])})

    evaluation_registry.register("example.staged_matrix", recipe)
    result = execute_evaluation_run_matrix(
        matrix, registry=evaluation_registry, root=output_root, **kwargs
    )
    return result, observed


def test_matrix_resolves_shared_local_parents_before_distinct_row_roots(
    tmp_path: Path,
    evaluation_registry,
) -> None:
    parent_root = tmp_path / "parents"
    parent_root.mkdir()
    training = TrainingRunManifest(id="feedbax-training-run:shared", status="completed")
    training_path = write_manifest(training, root=parent_root, index=False)
    training_ref = authenticated_manifest_ref(training, training_path, "training_run")
    artifact = store_evaluation_states_artifact(
        {"pair": np.asarray([3, 5])},
        root=parent_root,
        manifest_id="feedbax-evaluation-run:paired-bank",
    )
    artifact = artifact.model_copy(update={"uri": artifact.metadata["relative_path"]})
    bank = _evaluation_manifest(artifact)
    bank_path = write_manifest(bank, root=parent_root, index=False)
    bank_ref = authenticated_manifest_ref(bank, bank_path, "evaluation_run")
    explicit = _staged_matrix(training_ref, bank_ref)
    base = explicit.base.model_dump(mode="json", exclude_none=True)
    (tmp_path / "staged-base.json").write_text(json.dumps(base), encoding="utf-8")
    matrix = EvaluationRunMatrixSpec.model_validate(
        {
            **explicit.model_dump(mode="json", exclude_none=True),
            "base": {
                "ref": "staged-base.json",
                "sha256": sha256_bytes(canonical_json_bytes(base)),
            },
            "rows": [],
            "axes": [
                {
                    "id": "row",
                    "values": [{"id": "a"}, {"id": "b"}],
                }
            ],
        }
    )

    result, observed = _run_staged_matrix(
        matrix,
        evaluation_registry=evaluation_registry,
        output_root=tmp_path / "rows",
        parent_manifest_root=parent_root,
        repo_root=tmp_path,
    )

    assert observed == [
        (tmp_path / "rows" / "row-a", training.id, [3, 5]),
        (tmp_path / "rows" / "row-b", training.id, [3, 5]),
    ]
    assert result.metadata["staged_parents"]["training"]["parent"] == (
        training_ref.model_dump(mode="json", exclude_none=True)
    )
    for row in result.rows:
        assert row.result.provenance.parents == [training_ref, bank_ref]
        assert (
            row.result.metadata["matrix_harness"]["staged_parents"]
            == (result.metadata["staged_parents"])
        )
        axis_provenance = row.result.metadata["matrix_harness"]["axis_expansion"]
        assert axis_provenance["authored_matrix_sha256"] == sha256_bytes(
            canonical_json_bytes(matrix.model_dump(mode="json", exclude_none=True))
        )
        assert row.regeneration is not None
        assert row.regeneration.parameters["executable_spec"] == matrix.model_dump(
            mode="json", exclude_none=True
        )
        assert row.regeneration.parameters["manifest_root"] == str(tmp_path / "rows")
        assert row.regeneration.parameters["repo_root"] == str(tmp_path)
        assert row.regeneration.parameters["parent_manifest_root"] == str(parent_root)
        assert row.regeneration.parameters["execution_descriptor"] is None
        assert row.regeneration.parameters["artifact_provider_bindings"] == []
        assert row.regeneration.parameters["checkpoint_custody_bindings"] == []
        assert row.regeneration.parameters["staged_parents"] == result.metadata["staged_parents"]
        assert row.result.metadata["matrix_harness"]["regeneration_spec"] == (
            row.regeneration.model_dump(mode="json", exclude_none=True)
        )


@pytest.mark.parametrize(
    ("metadata_update", "error_type"),
    [
        ({}, None),
        (
            {"schema_version": "feedbax.spec.evaluation_states_container.v999"},
            EvaluationStatesSchemaMismatch,
        ),
        ({"manifest_id": "feedbax-evaluation-run:tampered"}, EvaluationStatesProvenanceMismatch),
    ],
)
def test_matrix_resolves_shared_provider_parents_and_validates_durable_bank(
    tmp_path: Path,
    metadata_update: dict[str, str],
    error_type: type[ValueError] | None,
    evaluation_registry,
) -> None:
    provider_root = tmp_path / "provider"
    provider_root.mkdir()
    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(), explicit_root=provider_root
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    training = TrainingRunManifest(id="feedbax-training-run:provider", status="completed")
    training_path = write_manifest(training, root=source_root, index=False)
    training_bytes = training_path.read_bytes()
    provider.store_bytes(training_bytes, role="training_run", logical_name="training.json")
    training_ref = authenticated_manifest_ref(training, training_path, "training_run")
    source_artifact = store_evaluation_states_artifact(
        {"pair": np.asarray([8, 13])},
        root=source_root,
        manifest_id="feedbax-evaluation-run:paired-bank",
    )
    state_bytes = (source_root / source_artifact.metadata["relative_path"]).read_bytes()
    provider_artifact = provider.store_bytes(
        state_bytes,
        role="evaluation_states",
        logical_name="states.npz",
        media_type=source_artifact.media_type,
        metadata={
            key: value
            for key, value in source_artifact.metadata.items()
            if key not in {"relative_path", "storage_backend"}
        },
    ).model_copy(
        update={
            "metadata": {
                key: value
                for key, value in {**source_artifact.metadata, **metadata_update}.items()
                if key != "relative_path"
            }
        }
    )
    bank = _evaluation_manifest(provider_artifact)
    bank_path = write_manifest(bank, root=source_root, index=False)
    bank_bytes = bank_path.read_bytes()
    provider.store_bytes(bank_bytes, role="evaluation_run", logical_name="bank.json")
    bank_ref = authenticated_manifest_ref(bank, bank_path, "evaluation_run")
    descriptor = StagedExecutionDescriptor(
        schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        artifact_providers={"shared": ImmutableArtifactBlobProviderSpec()},
        checkpoint_custody={},
    )

    kwargs = {
        "evaluation_registry": evaluation_registry,
        "output_root": tmp_path / "rows",
        "execution_descriptor": descriptor,
        "artifact_provider_bindings": [StagedArtifactProviderRootBinding("shared", provider_root)],
    }
    matrix = _staged_matrix(training_ref, bank_ref, artifact_provider="shared")
    if error_type is not None:
        with pytest.raises(EvaluationRecipeExecutionError) as exc_info:
            _run_staged_matrix(matrix, **kwargs)
        assert isinstance(exc_info.value.__cause__, error_type)
        return

    result, observed = _run_staged_matrix(matrix, **kwargs)

    assert [item[2] for item in observed] == [[8, 13], [8, 13]]
    assert result.metadata["staged_parents"]["paired_bank"]["artifact_provider"] == "shared"
    replay = result.metadata["regeneration_parameters"]
    assert replay["execution_descriptor"] == descriptor.model_dump(mode="json", exclude_none=True)
    assert replay["artifact_provider_bindings"] == [{"name": "shared", "root": str(provider_root)}]


def test_matrix_staged_parent_contract_fails_closed_before_row_creation(
    tmp_path: Path,
    evaluation_registry,
) -> None:
    parent = ParentRef(kind="TrainingRunManifest", id="unauthenticated", role="training_run")
    with pytest.raises(ValidationError, match="authenticated ParentRef"):
        _staged_matrix(parent, parent)

    parent_root = tmp_path / "parents"
    parent_root.mkdir()
    training = TrainingRunManifest(id="feedbax-training-run:wrong-size", status="completed")
    path = write_manifest(training, root=parent_root, index=False)
    exact = authenticated_manifest_ref(training, path, "training_run")
    unreferenced = EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(evaluation_type="example.staged_matrix"),
        rows=[MatrixRow(row_id="row-a")],
        staged_parents={"training": StagedEvaluationPrerequisite(parent=exact)},
    )
    with pytest.raises(ValueError, match="does not reference staged parent"):
        materialize_evaluation_run_matrix(unreferenced, registry=evaluation_registry)

    wrong_size = exact.model_copy(
        update={"metadata": {**exact.metadata, "size_bytes": exact.metadata["size_bytes"] + 1}}
    )
    matrix = EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(evaluation_type="example.staged_matrix", inputs=[wrong_size]),
        rows=[MatrixRow(row_id="row-a")],
        staged_parents={"training": StagedEvaluationPrerequisite(parent=wrong_size)},
    )
    with pytest.raises(ValueError, match="byte size"):
        execute_evaluation_run_matrix(
            matrix,
            registry=evaluation_registry,
            root=tmp_path / "rows",
            parent_manifest_root=parent_root,
        )
    assert not (tmp_path / "rows").exists()

    with pytest.raises(StagedExecutionContextError, match="parent_manifest_root"):
        execute_evaluation_run_matrix(matrix, root=tmp_path / "rows", registry=evaluation_registry)
    with pytest.raises(StagedExecutionContextError, match="must be absolute"):
        execute_evaluation_run_matrix(
            matrix,
            registry=evaluation_registry,
            root=tmp_path / "rows",
            parent_manifest_root="relative/parents",
        )


def test_matrix_staged_parent_input_matches_material_identity_not_consumer_role(
    evaluation_registry,
) -> None:
    profile = {
        "ref_schema_id": "feedbax.ref.authenticated_manifest",
        "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
        "manifest_sha256": "a" * 64,
        "size_bytes": 17,
    }
    executable_input = ParentRef(
        kind="TrainingRunManifest",
        id="feedbax-training-run:staged-subject",
        role="training_run",
        metadata=profile,
    )
    staged_parent = executable_input.model_copy(update={"role": "trained"})
    matrix = EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type="example.staged_matrix",
            inputs=[executable_input],
        ),
        rows=[MatrixRow(row_id="row-a")],
        staged_parents={
            "trained": StagedEvaluationPrerequisite(parent=staged_parent),
        },
    )

    [row] = materialize_evaluation_run_matrix(matrix, registry=evaluation_registry)

    assert row.payload.inputs == [executable_input]
    assert row.payload.inputs[0].role == "training_run"
    assert matrix.staged_parents["trained"].parent == staged_parent
    assert matrix.staged_parents["trained"].parent.role == "trained"


@pytest.mark.parametrize(
    "staged_update",
    [
        {"kind": "EvaluationRunManifest"},
        {"id": "feedbax-training-run:other"},
        {
            "metadata": {
                "ref_schema_id": "feedbax.ref.authenticated_manifest",
                "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
                "manifest_sha256": "b" * 64,
                "size_bytes": 17,
            }
        },
        {
            "metadata": {
                "ref_schema_id": "feedbax.ref.authenticated_manifest",
                "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
                "manifest_sha256": "a" * 64,
                "size_bytes": 18,
            }
        },
    ],
    ids=["kind", "id", "digest", "size"],
)
def test_matrix_staged_parent_input_refuses_different_material_identity(
    staged_update: dict[str, object],
    evaluation_registry,
) -> None:
    executable_input = ParentRef(
        kind="TrainingRunManifest",
        id="feedbax-training-run:staged-subject",
        role="training_run",
        metadata={
            "ref_schema_id": "feedbax.ref.authenticated_manifest",
            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
            "manifest_sha256": "a" * 64,
            "size_bytes": 17,
        },
    )
    staged_parent = executable_input.model_copy(update={"role": "trained", **staged_update})
    matrix = EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type="example.staged_matrix",
            inputs=[executable_input],
        ),
        rows=[MatrixRow(row_id="row-a")],
        staged_parents={
            "trained": StagedEvaluationPrerequisite(parent=staged_parent),
        },
    )

    with pytest.raises(ValueError, match="does not reference staged parent"):
        materialize_evaluation_run_matrix(matrix, registry=evaluation_registry)
