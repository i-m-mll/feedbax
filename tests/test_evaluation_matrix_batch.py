from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import jax.tree as jt
import numpy as np
import pytest

import feedbax.analysis.evaluation as evaluation_module
import feedbax.analysis.execution_context as execution_context_module
from feedbax.analysis.evaluation import (
    EvaluationBatchExecution,
    EvaluationBatchRowError,
    EvaluationRecipeResult,
    EvaluationRunMatrixSpec,
    execute_evaluation_run_matrix,
    load_evaluation_states,
    register_evaluation_recipe,
    resolve_staged_evaluation_prerequisite,
    unregister_evaluation_recipe,
)
from feedbax.analysis.execution_context import StagedExecutionContextError
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.analysis.specs import (
    register_analysis_recipe,
    resolve_analysis_inputs,
    unregister_analysis_recipe,
)
from feedbax.analysis.validation import RecipeValidationError
from feedbax.contracts.evaluation_states import (
    EvaluationStatesCustodyUnavailable,
    store_evaluation_states_artifact,
)
from feedbax.contracts.manifest import (
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION,
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3,
    AnalysisRunSpec,
    EvaluationRunManifest,
    EvaluationRunSpec,
    OverridePatch,
    StagedEvaluationPrerequisite,
    load_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.matrix_core import MatrixRow


EVALUATION_TYPE = "feedbax.test.batched_matrix"
ANALYSIS_TYPE = "feedbax.test.batched_matrix_analysis"


def _matrix() -> EvaluationRunMatrixSpec:
    return EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type=EVALUATION_TYPE,
            params={"gain": 1.0, "states_custody": "durable"},
        ),
        rows=[
            MatrixRow(
                row_id=f"row-{index}",
                deltas=[OverridePatch(path="params.gain", value=float(index + 1))],
            )
            for index in range(3)
        ],
    )


def _result(gain: float) -> EvaluationRecipeResult:
    states = {"value": np.asarray([gain, gain + 1], dtype=np.float32)}
    return EvaluationRecipeResult(
        states=states,
        summary_metrics={"total": float(states["value"].sum())},
        metadata={"states_schema": "feedbax.test.batched_matrix_states.v1"},
    )


class TypedRowStates(NamedTuple):
    value: object


def _typed_result(gain: float) -> EvaluationRecipeResult:
    states = TypedRowStates(np.asarray([gain, gain + 1], dtype=np.float32))
    return EvaluationRecipeResult(states=states, metadata={"states_schema": "typed.v1"})


def _execute_typed_batch(matrix, root: Path, *, parent_root: Path | None = None):
    register_evaluation_recipe(
        EVALUATION_TYPE,
        lambda spec, *_args: _typed_result(spec.params["gain"]),
        batch_recipe=lambda items, _: [_typed_result(item.spec.params["gain"]) for item in items],
        replace=True,
    )
    try:
        return execute_evaluation_run_matrix(
            matrix,
            root=root,
            parent_manifest_root=parent_root,
            batch=EvaluationBatchExecution(),
        )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)


def _resolve_typed_rows(batch, root: Path) -> list[TypedRowStates]:
    register_analysis_recipe(
        ANALYSIS_TYPE,
        lambda *_args: None,
        evaluation_states_structure=lambda _: jt.structure(_typed_result(0).states),
        replace=True,
    )
    try:
        return [
            resolve_analysis_inputs(
                AnalysisRunSpec(
                    analysis_type=ANALYSIS_TYPE,
                    inputs=[authenticated_manifest_ref(row.result, row.manifest_path, "evaluation_run")],
                    evaluation_states_policy="require_durable",
                ),
                root=root / row.row_id,
            )[0].states
            for row in batch.rows
        ]
    finally:
        unregister_analysis_recipe(ANALYSIS_TYPE)


def _staged_matrix(tmp_path: Path, states):
    parent_root = tmp_path / "parents"
    parent_id = "feedbax-evaluation-run:batch-prerequisite"
    artifact = store_evaluation_states_artifact(states, root=parent_root, manifest_id=parent_id)
    parent = EvaluationRunManifest(
        id=parent_id,
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec", EvaluationRunSpec(evaluation_type=EVALUATION_TYPE).model_dump(
                mode="json")),
        artifacts=[artifact],
    )
    parent_path = write_manifest(parent, root=parent_root, index=False)
    prerequisite = StagedEvaluationPrerequisite(
        parent=authenticated_manifest_ref(parent, parent_path, "evaluation_run")
    )
    matrix = EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type=EVALUATION_TYPE,
            params={
                "gain": 3.0, "states_custody": "durable",
                "staged_prerequisites": {"parent": prerequisite}},
        ),
        rows=[MatrixRow(row_id="row")],
        staged_parents={"parent": prerequisite},
    )
    return parent_root, artifact, matrix


def test_batched_typed_v3_outputs_round_trip_through_require_durable(tmp_path: Path) -> None:
    root = tmp_path / "batched"
    batch = _execute_typed_batch(_matrix(), root)
    restored = _resolve_typed_rows(batch, root)
    assert all(isinstance(states, TypedRowStates) for states in restored)
    for gain, row, states in zip((1, 2, 3), batch.rows, restored, strict=True):
        artifact = next(a for a in row.result.artifacts if a.role == "evaluation_states")
        assert artifact.metadata["schema_version"] == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3
        np.testing.assert_array_equal(states.value, np.asarray([gain, gain + 1]))


def test_batched_typed_v3_outputs_accept_normalized_staged_v2_parent(tmp_path: Path) -> None:
    parent_root, parent_artifact, matrix = _staged_matrix(
        tmp_path, {"value": np.asarray([3, 5], dtype=np.int32)}
    )
    root = tmp_path / "batched"
    batch = _execute_typed_batch(matrix, root, parent_root=parent_root)
    output_artifact = next(
        a for a in batch.rows[0].result.artifacts if a.role == "evaluation_states"
    )
    assert parent_artifact.uri is None
    assert parent_artifact.metadata["schema_version"] == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION
    assert output_artifact.metadata["schema_version"] == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3
    restored = _resolve_typed_rows(batch, root)
    assert isinstance(restored[0], TypedRowStates)
    np.testing.assert_array_equal(restored[0].value, np.asarray([3, 4]))


def test_batch_preflight_primes_transaction_memo_for_recipe_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_root, _artifact, matrix = _staged_matrix(
        tmp_path,
        {"value": np.asarray([3, 5], dtype=np.int32)},
    )
    matrix = matrix.model_copy(
        update={
            "rows": [
                MatrixRow(
                    row_id=f"row-{index}",
                    deltas=[OverridePatch(path="params.gain", value=float(index + 1))],
                )
                for index in range(3)
            ]
        }
    )
    original_load = execution_context_module.load_authenticated_evaluation_states_artifact
    loads = 0

    def counted_load(*args, **kwargs):
        nonlocal loads
        loads += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(
        execution_context_module,
        "load_authenticated_evaluation_states_artifact",
        counted_load,
    )

    def batch_recipe(items, execution_context):
        for item in items:
            prerequisite = item.spec.params["staged_prerequisites"]["parent"]
            resolve_staged_evaluation_prerequisite(
                prerequisite,
                execution_context=execution_context,
            )
        return [_result(item.spec.params["gain"]) for item in items]

    register_evaluation_recipe(
        EVALUATION_TYPE,
        lambda *_args: pytest.fail("scalar recipe must not run"),
        batch_recipe=batch_recipe,
        replace=True,
    )
    try:
        execute_evaluation_run_matrix(
            matrix,
            root=tmp_path / "batched",
            parent_manifest_root=parent_root,
            batch=EvaluationBatchExecution(),
        )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)

    assert loads == 1


def test_batched_preflight_rejects_typed_v3_staged_parent(tmp_path: Path) -> None:
    parent_root, artifact, matrix = _staged_matrix(tmp_path, _typed_result(9).states)
    output = tmp_path / "batched"
    register_evaluation_recipe(
        EVALUATION_TYPE,
        lambda spec, *_args: _typed_result(spec.params["gain"]),
        batch_recipe=lambda *_args: pytest.fail("batch recipe ran before preflight"),
        replace=True,
    )
    try:
        with pytest.raises(EvaluationBatchRowError) as exc_info:
            execute_evaluation_run_matrix(
                matrix,
                root=output,
                parent_manifest_root=parent_root,
                batch=EvaluationBatchExecution(),
            )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)

    assert artifact.metadata["schema_version"] == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3
    assert exc_info.value.row_id == "row"
    assert isinstance(exc_info.value.__cause__, EvaluationStatesCustodyUnavailable)
    assert not output.exists()


def test_batched_matrix_matches_default_and_round_trips_require_durable(
    tmp_path: Path,
) -> None:
    scalar_calls: list[float] = []
    batch_calls: list[tuple[str, ...]] = []

    def scalar_recipe(spec, _root, _states_path, _execution_context):
        scalar_calls.append(spec.params["gain"])
        return _result(spec.params["gain"])

    def batch_recipe(items, _execution_context):
        batch_calls.append(tuple(item.row_id for item in items))
        gains = np.asarray([item.spec.params["gain"] for item in items])
        return [_result(float(gain)) for gain in gains]

    register_evaluation_recipe(
        EVALUATION_TYPE,
        scalar_recipe,
        batch_recipe=batch_recipe,
        replace=True,
    )
    staging_paths: list[Path] = []
    real_mkdtemp = evaluation_module.tempfile.mkdtemp

    def capture_staging(**kwargs):
        staging = Path(real_mkdtemp(**kwargs))
        staging_paths.append(staging)
        return str(staging)

    try:
        default = execute_evaluation_run_matrix(_matrix(), root=tmp_path / "default")
        with patch.object(evaluation_module.tempfile, "mkdtemp", side_effect=capture_staging):
            batched = execute_evaluation_run_matrix(
                _matrix(),
                root=tmp_path / "batched",
                batch=EvaluationBatchExecution(),
            )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)

    assert scalar_calls == [1.0, 2.0, 3.0]
    assert batch_calls == [("row-0", "row-1", "row-2")]
    assert len(staging_paths) == 1 and not staging_paths[0].exists()
    for default_row, batched_row in zip(default.rows, batched.rows, strict=True):
        assert default_row.row_id == batched_row.row_id
        assert default_row.result.status == batched_row.result.status == "completed"
        assert default_row.result.evaluation_spec == batched_row.result.evaluation_spec
        assert default_row.result.input_training_runs == batched_row.result.input_training_runs
        assert default_row.result.provenance == batched_row.result.provenance
        assert default_row.result.summary_metrics == batched_row.result.summary_metrics
        default_artifacts = {
            artifact.role: artifact for artifact in default_row.result.artifacts
        }
        batch_artifacts = {artifact.role: artifact for artifact in batched_row.result.artifacts}
        assert set(default_artifacts) == set(batch_artifacts)
        for role in set(batch_artifacts) - {"regeneration_spec"}:
            assert default_artifacts[role].artifact_id == batch_artifacts[role].artifact_id
            assert default_artifacts[role].metadata == batch_artifacts[role].metadata
        assert default_artifacts["regeneration_spec"].artifact_id != (
            batch_artifacts["regeneration_spec"].artifact_id
        )
        for artifact in batch_artifacts.values():
            if artifact.uri is not None:
                assert Path(artifact.uri).exists()
        default_cache = default_row.result.metadata["cache"]
        batch_cache = batched_row.result.metadata["cache"]
        assert {key: value for key, value in default_cache.items() if key != "states_path"} == {
            key: value for key, value in batch_cache.items() if key != "states_path"
        }
        assert Path(batch_cache["states_path"]).is_relative_to(tmp_path / "batched")
        default_harness = dict(default_row.result.metadata["matrix_harness"])
        batch_harness = dict(batched_row.result.metadata["matrix_harness"])
        assert batch_harness.pop("batch_execution") == {
            "row_ids": ["row-0", "row-1", "row-2"]
        }
        default_regeneration = default_harness.pop("regeneration_spec")
        batch_regeneration = batch_harness.pop("regeneration_spec")
        assert default_harness == batch_harness
        assert default_regeneration["parameters"]["resolved"] == (
            batch_regeneration["parameters"]["resolved"]
        )
        assert load_manifest(batched_row.manifest_path) == batched_row.result
        default_states = load_evaluation_states(
            default_row.result, root=tmp_path / "default" / default_row.row_id
        )
        batched_states = load_evaluation_states(
            batched_row.result, root=tmp_path / "batched" / batched_row.row_id
        )
        np.testing.assert_array_equal(default_states["value"], batched_states["value"])

    register_analysis_recipe(
        ANALYSIS_TYPE,
        lambda *_args: None,
        replace=True,
        evaluation_states_structure=lambda _: jt.structure(
            {"value": np.asarray([0.0, 0.0], dtype=np.float32)}
        ),
    )
    try:
        for row in batched.rows:
            authority = authenticated_manifest_ref(
                row.result,
                row.manifest_path,
                "evaluation_run",
            )
            resolved = resolve_analysis_inputs(
                AnalysisRunSpec(
                    analysis_type=ANALYSIS_TYPE,
                    inputs=[authority],
                    evaluation_states_policy="require_durable",
                ),
                root=tmp_path / "batched" / row.row_id,
            )[0]
            np.testing.assert_array_equal(
                resolved.states["value"],
                load_evaluation_states(
                    row.result, root=tmp_path / "batched" / row.row_id
                )["value"],
            )
    finally:
        unregister_analysis_recipe(ANALYSIS_TYPE)

    staging_bytes = str(staging_paths[0]).encode()
    assert str(staging_paths[0]) not in repr(batched)
    assert all(
        staging_bytes not in path.read_bytes()
        for path in (tmp_path / "batched").rglob("*")
        if path.is_file()
    )


def test_batched_matrix_fails_closed_with_typed_row_diagnostic(tmp_path: Path) -> None:
    def scalar_recipe(spec, _root, _states_path, _execution_context):
        return _result(spec.params["gain"])

    def failing_batch(items, _execution_context):
        raise EvaluationBatchRowError(items[1].row_id, ValueError("injected failure"))

    register_evaluation_recipe(
        EVALUATION_TYPE,
        scalar_recipe,
        batch_recipe=failing_batch,
        replace=True,
    )
    output = tmp_path / "batched"
    try:
        with pytest.raises(EvaluationBatchRowError, match="row-1.*injected failure") as exc_info:
            execute_evaluation_run_matrix(
                _matrix(),
                root=output,
                batch=EvaluationBatchExecution(),
            )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)

    assert exc_info.value.row_id == "row-1"
    assert not output.exists()


def test_batched_matrix_fails_closed_when_staging_index_discard_is_no_op(
    tmp_path: Path,
) -> None:
    real_rmtree = evaluation_module.shutil.rmtree

    def no_op_index_discard(path, *args, **kwargs):
        if Path(path).name == "index":
            if kwargs.get("ignore_errors") is True:
                raise AssertionError("index discard must not ignore errors")
            return None
        return real_rmtree(path, *args, **kwargs)

    register_evaluation_recipe(
        EVALUATION_TYPE,
        lambda spec, *_args: _result(spec.params["gain"]),
        batch_recipe=lambda items, _context: [
            _result(item.spec.params["gain"]) for item in items
        ],
        replace=True,
    )
    output = tmp_path / "batched"
    try:
        with (
            patch.object(evaluation_module.shutil, "rmtree", side_effect=no_op_index_discard),
            pytest.raises(RuntimeError, match="staging index discard did not remove"),
        ):
            execute_evaluation_run_matrix(
                _matrix(),
                root=output,
                batch=EvaluationBatchExecution(),
            )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)

    assert not output.exists()


def test_batched_matrix_authenticates_staged_prerequisites_before_publish(
    tmp_path: Path,
) -> None:
    parent_root = tmp_path / "parents"
    parent_id = "feedbax-evaluation-run:batch-prerequisite"
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([3, 5], dtype=np.int32)},
        root=parent_root,
        manifest_id=parent_id,
    )
    parent = EvaluationRunManifest(
        id=parent_id,
        status="completed",
        evaluation_spec=spec_payload("EvaluationRunSpec", EvaluationRunSpec(evaluation_type=EVALUATION_TYPE).model_dump(mode="json")),
        artifacts=[artifact],
    )
    parent_path = write_manifest(parent, root=parent_root, index=False)
    parent_ref = authenticated_manifest_ref(parent, parent_path, "evaluation_run")
    prerequisite = StagedEvaluationPrerequisite(parent=parent_ref)
    matrix = EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type=EVALUATION_TYPE,
            params={"staged_prerequisites": {"parent": prerequisite}},
        ),
        rows=[MatrixRow(row_id="row")],
        staged_parents={"parent": prerequisite},
    )
    artifact_path = parent_root / artifact.metadata["relative_path"]
    artifact_path.write_bytes(b"x" * artifact_path.stat().st_size)

    register_evaluation_recipe(
        EVALUATION_TYPE,
        lambda *_args: _result(1.0),
        batch_recipe=lambda items, _context: [_result(1.0) for _ in items],
        replace=True,
    )
    output = tmp_path / "batch"
    try:
        with pytest.raises(EvaluationBatchRowError) as exc_info:
            execute_evaluation_run_matrix(
                matrix,
                root=output,
                parent_manifest_root=parent_root,
                batch=EvaluationBatchExecution(),
            )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)

    assert exc_info.value.row_id == "row"
    assert isinstance(exc_info.value.__cause__, StagedExecutionContextError)
    assert "sha256 mismatch" in str(exc_info.value.__cause__)
    assert not output.exists()


def test_batch_validation_matches_scalar_reserved_field_and_callback_contract(
    tmp_path: Path,
) -> None:
    bad_matrix = EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type=EVALUATION_TYPE,
            params={"staged_prerequisites": []},
        ),
        rows=[MatrixRow(row_id="row")],
    )
    with pytest.raises(RecipeValidationError, match="must accept two positional arguments"):
        register_evaluation_recipe(
            EVALUATION_TYPE,
            lambda *_args: _result(1.0),
            batch_recipe=lambda _items: [],
            replace=True,
        )

    register_evaluation_recipe(
        EVALUATION_TYPE,
        lambda *_args: _result(1.0),
        batch_recipe=lambda items, _context: [_result(1.0) for _ in items],
        replace=True,
    )
    try:
        with pytest.raises(TypeError, match="EvaluationBatchExecution"):
            execute_evaluation_run_matrix(bad_matrix, root=tmp_path / "marker", batch=object())
        with pytest.raises(EvaluationBatchRowError) as exc_info:
            execute_evaluation_run_matrix(
                bad_matrix,
                root=tmp_path / "batch",
                batch=EvaluationBatchExecution(),
            )
        with pytest.raises(TypeError) as scalar_exc:
            execute_evaluation_run_matrix(bad_matrix, root=tmp_path / "default")
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)

    assert isinstance(exc_info.value.__cause__, TypeError)
    assert str(exc_info.value.__cause__) == str(scalar_exc.value)
    assert not (tmp_path / "batch").exists()
