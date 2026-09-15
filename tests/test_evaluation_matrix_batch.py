from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
import multiprocessing
from pathlib import Path
from typing import Literal, NamedTuple
from unittest.mock import patch

import jax.tree as jt
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

import feedbax.analysis.evaluation as evaluation_module
import feedbax.analysis.execution_context as execution_context_module
from feedbax.analysis.evaluation import (
    EvaluationBatchExecution,
    EvaluationBatchRowError,
    EvaluationRecipeRegistry,
    EvaluationRecipeResult,
    EvaluationRunMatrixSpec,
    execute_evaluation_run_matrix,
    finalize_evaluation_batch_retention,
    load_evaluation_states,
    resolve_staged_evaluation_prerequisite,
)
from feedbax.analysis.execution_context import StagedExecutionContextError
from feedbax.analysis.harness import resolve_manifest_regeneration
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.analysis.specs import (
    resolve_analysis_inputs,
)
from feedbax.analysis.validation import RecipeValidationError
from feedbax.contracts.evaluation_states import (
    EvaluationStatesCustodyUnavailable,
    store_evaluation_states_artifact,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.manifest import (
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION,
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3,
    AnalysisRunSpec,
    EvaluationRunManifest,
    EvaluationRunSpec,
    OverridePatch,
    StagedEvaluationPrerequisite,
    TrainingRunManifest,
    load_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.matrix_core import MatrixRow


EVALUATION_TYPE = "feedbax.test.batched_matrix"
ANALYSIS_TYPE = "feedbax.test.batched_matrix_analysis"


def _raise_pickled_evaluation_batch_row_error() -> None:
    raise EvaluationBatchRowError("spawned-row", ValueError("spawned failure"))


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


class TypedPerturbationSource(BaseModel):
    """Plugin-owned typed source retained inside an authenticated training parent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["example.spec.perturbation_source"]
    schema_version: Literal["example.spec.perturbation_source.v1"]
    trial_ids: tuple[str, ...]
    draw: tuple[float, ...]


def _typed_result(gain: float) -> EvaluationRecipeResult:
    states = TypedRowStates(np.asarray([gain, gain + 1], dtype=np.float32))
    return EvaluationRecipeResult(states=states, metadata={"states_schema": "typed.v1"})


def _execute_typed_batch(
    matrix, root: Path, *, parent_root: Path | None = None, application_registry_bundle
):
    application_registry_bundle.evaluation_recipes.register(
        EVALUATION_TYPE,
        lambda spec, *_args: _typed_result(spec.params["gain"]),
        batch_recipe=lambda items, _: [_typed_result(item.spec.params["gain"]) for item in items],
    )
    return execute_evaluation_run_matrix(
        matrix,
        root=root,
        parent_manifest_root=parent_root,
        batch=EvaluationBatchExecution(),
        registry=application_registry_bundle.evaluation_recipes,
    )


def _resolve_typed_rows(batch, root: Path, application_registry_bundle) -> list[TypedRowStates]:
    application_registry_bundle.analysis_recipes.register(
        ANALYSIS_TYPE,
        lambda *_args: None,
        evaluation_states_structure=lambda _: jt.structure(_typed_result(0).states),
    )
    return [
        resolve_analysis_inputs(
            AnalysisRunSpec(
                analysis_type=ANALYSIS_TYPE,
                inputs=[
                    authenticated_manifest_ref(row.result, row.manifest_path, "evaluation_run")
                ],
                evaluation_states_policy="require_durable",
            ),
            root=root,
            evaluation_registry=application_registry_bundle.evaluation_recipes,
            registry=application_registry_bundle.analysis_recipes,
        )[0].states
        for row in batch.rows
    ]


def test_matrix_recipes_receive_one_resolved_repo_root_for_scalar_and_batch(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    scalar_roots: list[Path | None] = []
    batch_roots: list[Path | None] = []

    def scalar_recipe(spec, _root, _states_path, execution_context):
        scalar_roots.append(execution_context.repo_root)
        return _result(spec.params["gain"])

    def batch_recipe(items, execution_context):
        batch_roots.append(execution_context.repo_root)
        return [_result(item.spec.params["gain"]) for item in items]

    application_registry_bundle.evaluation_recipes.register(
        EVALUATION_TYPE,
        scalar_recipe,
        batch_recipe=batch_recipe,
    )
    execute_evaluation_run_matrix(
        _matrix(),
        root=tmp_path / "scalar",
        repo_root=repo_root,
        registry=application_registry_bundle.evaluation_recipes,
    )
    execute_evaluation_run_matrix(
        _matrix(),
        root=tmp_path / "batch",
        repo_root=repo_root,
        batch=EvaluationBatchExecution(),
        registry=application_registry_bundle.evaluation_recipes,
    )

    assert scalar_roots == [repo_root.resolve()] * 3
    assert batch_roots == [repo_root.resolve()]


def test_batch_recipe_reuses_typed_source_from_authenticated_parent(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    parent_root = tmp_path / "parents"
    parent_root.mkdir()
    source_payload = {
        "schema_id": "example.spec.perturbation_source",
        "schema_version": "example.spec.perturbation_source.v1",
        "trial_ids": ["trial-0", "trial-1"],
        "draw": [0.25, -0.5],
    }
    training = TrainingRunManifest(
        id="feedbax-training-run:typed-perturbation-source",
        status="completed",
        training_spec=spec_payload(
            "TrainingRunSpec",
            {"task": {"params": {"perturbation": source_payload}}},
        ),
    )
    training_path = write_manifest(training, root=parent_root, index=False)
    training_ref = authenticated_manifest_ref(training, training_path, "training_run")
    prerequisite = StagedEvaluationPrerequisite(parent=training_ref)
    conditions = ("clean", "bank", "sisu", "bank-plus-sisu")
    matrix = EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type=EVALUATION_TYPE,
            inputs=[training_ref],
            params={"condition": "clean", "states_custody": "durable"},
        ),
        rows=[
            MatrixRow(
                row_id=condition,
                deltas=[OverridePatch(path="params.condition", value=condition)],
            )
            for condition in conditions
        ],
        staged_parents={"training": prerequisite},
    )
    parsed_sources: list[TypedPerturbationSource] = []
    source_object_ids: list[int] = []

    def batch_recipe(items, context):
        resolved = context.resolve_manifest_input(items[0].spec.inputs[0])
        source = TypedPerturbationSource.model_validate(
            resolved.manifest.training_spec.inline["task"]["params"]["perturbation"]
        )
        parsed_sources.append(source)
        results = []
        for item in items:
            source_object_ids.append(id(source))
            states = {"trial_ids": source.trial_ids}
            if item.spec.params["condition"] in {"bank", "bank-plus-sisu"}:
                states["ordinary_bank_draw"] = source.draw
            results.append(
                EvaluationRecipeResult(
                    states=states,
                    metadata={"states_schema": "example.evaluation.interaction_cells.v1"},
                )
            )
        return results

    application_registry_bundle.evaluation_recipes.register(
        EVALUATION_TYPE,
        lambda *_args: pytest.fail("typed shared source requires batch execution"),
        batch_recipe=batch_recipe,
    )
    root = tmp_path / "evaluation"
    batch = execute_evaluation_run_matrix(
        matrix,
        root=root,
        parent_manifest_root=parent_root,
        batch=EvaluationBatchExecution(),
        registry=application_registry_bundle.evaluation_recipes,
    )

    assert len(parsed_sources) == 1
    assert len(set(source_object_ids)) == 1
    assert [row.row_id for row in batch.rows] == list(conditions)
    for row in batch.rows:
        assert row.result.input_training_runs == [training_ref]
        assert row.result.provenance.parents == [training_ref]
        states = load_evaluation_states(row.result, root=root)
        assert states["trial_ids"] == tuple(source_payload["trial_ids"])
        if row.row_id in {"bank", "bank-plus-sisu"}:
            assert states["ordinary_bank_draw"] == tuple(source_payload["draw"])
        else:
            assert "ordinary_bank_draw" not in states


def _staged_matrix(tmp_path: Path, states):
    parent_root = tmp_path / "parents"
    parent_id = "feedbax-evaluation-run:batch-prerequisite"
    artifact = store_evaluation_states_artifact(states, root=parent_root, manifest_id=parent_id)
    parent = EvaluationRunManifest(
        id=parent_id,
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            EvaluationRunSpec(evaluation_type=EVALUATION_TYPE).model_dump(mode="json"),
        ),
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
                "gain": 3.0,
                "states_custody": "durable",
                "staged_prerequisites": {"parent": prerequisite},
            },
        ),
        rows=[MatrixRow(row_id="row")],
        staged_parents={"parent": prerequisite},
    )
    return parent_root, artifact, matrix


def test_batched_typed_v3_outputs_round_trip_through_require_durable(
    tmp_path: Path, application_registry_bundle
) -> None:
    root = tmp_path / "batched"
    batch = _execute_typed_batch(
        _matrix(), root, application_registry_bundle=application_registry_bundle
    )
    restored = _resolve_typed_rows(
        batch, root, application_registry_bundle=application_registry_bundle
    )
    assert all(isinstance(states, TypedRowStates) for states in restored)
    for gain, row, states in zip((1, 2, 3), batch.rows, restored, strict=True):
        artifact = next(a for a in row.result.artifacts if a.role == "evaluation_states")
        assert artifact.metadata["schema_version"] == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3
        np.testing.assert_array_equal(states.value, np.asarray([gain, gain + 1]))


def test_consumed_durable_batch_states_become_bounded_gc_candidates(
    tmp_path: Path, application_registry_bundle
) -> None:
    root = tmp_path / "batched"
    batch = _execute_typed_batch(
        _matrix(), root, application_registry_bundle=application_registry_bundle
    )
    now = datetime.now(timezone.utc)
    record_ids = finalize_evaluation_batch_retention(
        [row.result for row in batch.rows],
        root=root,
        retain_intermediates=False,
        grace_expires_at=now + timedelta(minutes=5),
        manifest_sha256s=[sha256_bytes(row.manifest_path.read_bytes()) for row in batch.rows],
    )

    assert len(record_ids) == 1
    assert record_ids[0].startswith("evaluation-intermediates-release:")
    provider = ImmutableArtifactBlobProvider(root)
    before = provider.collect_garbage(now=now)
    assert before.stage == "reported"
    assert all(decision.decision == "live" for decision in before.decisions)
    after = provider.collect_garbage(now=now + timedelta(minutes=6))
    assert after.stage == "reported"
    state_digests = {
        artifact.sha256
        for row in batch.rows
        for artifact in row.result.artifacts
        if artifact.role == "evaluation_states"
    }
    assert {
        decision.sha256 for decision in after.decisions if decision.decision == "candidate"
    } == state_digests


def test_batched_typed_v3_outputs_accept_normalized_staged_v2_parent(
    tmp_path: Path, application_registry_bundle
) -> None:
    parent_root, parent_artifact, matrix = _staged_matrix(
        tmp_path, {"value": np.asarray([3, 5], dtype=np.int32)}
    )
    root = tmp_path / "batched"
    batch = _execute_typed_batch(
        matrix,
        root,
        parent_root=parent_root,
        application_registry_bundle=application_registry_bundle,
    )
    output_artifact = next(
        a for a in batch.rows[0].result.artifacts if a.role == "evaluation_states"
    )
    assert parent_artifact.uri is None
    assert parent_artifact.metadata["schema_version"] == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION
    assert (
        output_artifact.metadata["schema_version"] == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3
    )
    restored = _resolve_typed_rows(
        batch, root, application_registry_bundle=application_registry_bundle
    )
    assert isinstance(restored[0], TypedRowStates)
    np.testing.assert_array_equal(restored[0].value, np.asarray([3, 4]))


def test_batch_preflight_primes_transaction_memo_for_recipe_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    application_registry_bundle,
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

    application_registry_bundle.evaluation_recipes.register(
        EVALUATION_TYPE,
        lambda *_args: pytest.fail("scalar recipe must not run"),
        batch_recipe=batch_recipe,
    )
    execute_evaluation_run_matrix(
        matrix,
        root=tmp_path / "batched",
        parent_manifest_root=parent_root,
        batch=EvaluationBatchExecution(),
        registry=application_registry_bundle.evaluation_recipes,
    )

    assert loads == 1


def test_batched_preflight_rejects_typed_v3_staged_parent(
    tmp_path: Path, application_registry_bundle
) -> None:
    parent_root, artifact, matrix = _staged_matrix(tmp_path, _typed_result(9).states)
    output = tmp_path / "batched"
    application_registry_bundle.evaluation_recipes.register(
        EVALUATION_TYPE,
        lambda spec, *_args: _typed_result(spec.params["gain"]),
        batch_recipe=lambda *_args: pytest.fail("batch recipe ran before preflight"),
    )
    with pytest.raises(EvaluationBatchRowError) as exc_info:
        execute_evaluation_run_matrix(
            matrix,
            root=output,
            parent_manifest_root=parent_root,
            batch=EvaluationBatchExecution(),
            registry=application_registry_bundle.evaluation_recipes,
        )

    assert artifact.metadata["schema_version"] == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3
    assert exc_info.value.row_id == "row"
    assert isinstance(exc_info.value.__cause__, EvaluationStatesCustodyUnavailable)
    assert not output.exists()


def test_batched_matrix_matches_default_and_round_trips_require_durable(
    tmp_path: Path,
    application_registry_bundle,
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

    application_registry_bundle.evaluation_recipes.register(
        EVALUATION_TYPE,
        scalar_recipe,
        batch_recipe=batch_recipe,
    )
    staging_paths: list[Path] = []
    real_mkdtemp = evaluation_module.tempfile.mkdtemp

    def capture_staging(**kwargs):
        staging = Path(real_mkdtemp(**kwargs))
        staging_paths.append(staging)
        return str(staging)

    default = execute_evaluation_run_matrix(
        _matrix(),
        root=tmp_path / "default",
        registry=application_registry_bundle.evaluation_recipes,
    )
    with patch.object(evaluation_module.tempfile, "mkdtemp", side_effect=capture_staging):
        batched = execute_evaluation_run_matrix(
            _matrix(),
            root=tmp_path / "batched",
            batch=EvaluationBatchExecution(),
            registry=application_registry_bundle.evaluation_recipes,
        )

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
        default_artifacts = {artifact.role: artifact for artifact in default_row.result.artifacts}
        batch_artifacts = {artifact.role: artifact for artifact in batched_row.result.artifacts}
        assert set(default_artifacts) == set(batch_artifacts)
        for role in set(batch_artifacts) - {
            "regeneration_spec",
            "shared_regeneration_context",
        }:
            assert default_artifacts[role].artifact_id == batch_artifacts[role].artifact_id
            assert default_artifacts[role].metadata == batch_artifacts[role].metadata
        assert default_artifacts["regeneration_spec"].artifact_id != (
            batch_artifacts["regeneration_spec"].artifact_id
        )
        for artifact in batch_artifacts.values():
            if artifact.uri is not None:
                if artifact.uri.startswith("artifact://sha256/"):
                    assert artifact.sha256 is not None
                    assert (
                        tmp_path
                        / "batched"
                        / "artifacts"
                        / "sha256"
                        / artifact.sha256[:2]
                        / artifact.sha256
                    ).exists()
                else:
                    assert Path(artifact.uri).exists()
        default_cache = default_row.result.metadata["cache"]
        batch_cache = batched_row.result.metadata["cache"]
        assert {key: value for key, value in default_cache.items() if key != "states_path"} == {
            key: value for key, value in batch_cache.items() if key != "states_path"
        }
        assert Path(batch_cache["states_path"]).is_relative_to(tmp_path / "batched")
        # Shared store: every row's durable state cache lives in the one store
        # root, not a per-row subtree.
        assert Path(batch_cache["states_path"]).is_relative_to(tmp_path / "batched" / "cache")
        default_harness = default_row.result.metadata["matrix_harness"]
        batch_harness = batched_row.result.metadata["matrix_harness"]
        assert {key: value for key, value in default_harness.items() if value is not None} == (
            {key: value for key, value in batch_harness.items() if value is not None}
        )
        assert "axis_expansion" not in batch_harness
        assert "regeneration_spec" not in batch_harness
        assert (
            resolve_manifest_regeneration(batched_row.result, root=tmp_path / "batched")
            == batched_row.regeneration
        )
        assert load_manifest(batched_row.manifest_path) == batched_row.result
        default_states = load_evaluation_states(
            default_row.result, root=tmp_path / "default" / default_row.row_id
        )
        batched_states = load_evaluation_states(batched_row.result, root=tmp_path / "batched")
        np.testing.assert_array_equal(default_states["value"], batched_states["value"])

    application_registry_bundle.analysis_recipes.register(
        ANALYSIS_TYPE,
        lambda *_args: None,
        evaluation_states_structure=lambda _: jt.structure(
            {"value": np.asarray([0.0, 0.0], dtype=np.float32)}
        ),
    )
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
            root=tmp_path / "batched",
            evaluation_registry=application_registry_bundle.evaluation_recipes,
            registry=application_registry_bundle.analysis_recipes,
        )[0]
        np.testing.assert_array_equal(
            resolved.states["value"],
            load_evaluation_states(row.result, root=tmp_path / "batched")["value"],
        )

    staging_bytes = str(staging_paths[0]).encode()
    shared_ids = {
        artifact.artifact_id
        for row in batched.rows
        for artifact in row.result.artifacts
        if artifact.role == "shared_regeneration_context"
    }
    assert len(shared_ids) == 1
    gc = ImmutableArtifactBlobProvider((tmp_path / "batched").absolute()).collect_garbage()
    assert gc.stage == "reported"
    assert all(decision.decision == "live" for decision in gc.decisions)
    assert str(staging_paths[0]) not in repr(batched)
    assert all(
        staging_bytes not in path.read_bytes()
        for path in (tmp_path / "batched").rglob("*")
        if path.is_file()
    )


def test_batched_matrix_fails_closed_with_typed_row_diagnostic(
    tmp_path: Path, application_registry_bundle
) -> None:
    def scalar_recipe(spec, _root, _states_path, _execution_context):
        return _result(spec.params["gain"])

    def failing_batch(items, _execution_context):
        raise EvaluationBatchRowError(items[1].row_id, ValueError("injected failure"))

    application_registry_bundle.evaluation_recipes.register(
        EVALUATION_TYPE,
        scalar_recipe,
        batch_recipe=failing_batch,
    )
    output = tmp_path / "batched"
    with pytest.raises(EvaluationBatchRowError, match="row-1.*injected failure") as exc_info:
        execute_evaluation_run_matrix(
            _matrix(),
            root=output,
            batch=EvaluationBatchExecution(),
            registry=application_registry_bundle.evaluation_recipes,
        )

    assert exc_info.value.row_id == "row-1"
    assert not output.exists()


def test_batch_row_error_survives_spawned_process_boundary() -> None:
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        future = pool.submit(_raise_pickled_evaluation_batch_row_error)
        with pytest.raises(
            EvaluationBatchRowError,
            match="spawned-row.*spawned failure",
        ) as exc_info:
            future.result()

    assert exc_info.value.row_id == "spawned-row"
    assert isinstance(exc_info.value.cause, ValueError)
    assert str(exc_info.value.cause) == "spawned failure"


def test_batched_matrix_rejects_content_identical_row_manifest_ids(
    tmp_path: Path, application_registry_bundle
) -> None:
    """Two byte-identical row specs would collapse last-write-wins; fail closed instead."""
    matrix = EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type=EVALUATION_TYPE,
            params={"gain": 1.0, "states_custody": "durable"},
        ),
        rows=[
            MatrixRow(row_id="row-a", deltas=[OverridePatch(path="params.gain", value=2.0)]),
            MatrixRow(row_id="row-b", deltas=[OverridePatch(path="params.gain", value=2.0)]),
        ],
    )
    output = tmp_path / "batched"

    _register_scalar_batch_recipe(application_registry_bundle)
    with pytest.raises(EvaluationBatchRowError) as exc_info:
        execute_evaluation_run_matrix(
            matrix,
            root=output,
            batch=EvaluationBatchExecution(),
            registry=application_registry_bundle.evaluation_recipes,
        )

    message = str(exc_info.value)
    assert "row-a" in message
    assert "row-b" in message
    assert "evaluation_run_manifest_id" in message
    assert exc_info.value.row_id in {"row-a", "row-b"}
    # Fail-closed staging cleanup: nothing is published for either row.
    assert not output.exists()


def _register_scalar_batch_recipe(application_registry_bundle) -> None:
    application_registry_bundle.evaluation_recipes.register(
        EVALUATION_TYPE,
        lambda spec, *_args: _result(spec.params["gain"]),
        batch_recipe=lambda items, _context: [_result(item.spec.params["gain"]) for item in items],
    )


def _sized_matrix(row_count: int) -> EvaluationRunMatrixSpec:
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
            for index in range(row_count)
        ],
    )


def test_batched_matrix_fails_closed_when_shared_index_build_fails(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    # The batch path imports index_manifest lazily from its source module, so
    # patching the source binding intercepts the shared-index build.
    def failing_index_manifest(*_args, **_kwargs):
        raise RuntimeError("injected shared index failure")

    _register_scalar_batch_recipe(application_registry_bundle)
    output = tmp_path / "batched"
    with (
        patch(
            "feedbax.persistence.manifest_index.index_manifest",
            side_effect=failing_index_manifest,
        ),
        pytest.raises(RuntimeError, match="injected shared index failure"),
    ):
        execute_evaluation_run_matrix(
            _matrix(),
            root=output,
            batch=EvaluationBatchExecution(),
            registry=application_registry_bundle.evaluation_recipes,
        )

    assert not output.exists()
    # The staging store was cleaned up rather than left as a partial sibling.
    assert not any(tmp_path.glob(".batched.batch-*"))


def test_batched_matrix_publishes_single_shared_store(
    tmp_path: Path, application_registry_bundle
) -> None:
    _register_scalar_batch_recipe(application_registry_bundle)
    output = tmp_path / "batched"
    batch = execute_evaluation_run_matrix(
        _sized_matrix(4),
        root=output,
        batch=EvaluationBatchExecution(),
        registry=application_registry_bundle.evaluation_recipes,
    )

    # Exactly one SQLite index for the whole matrix, at the store root.
    indexes = list(output.rglob("feedbax.sqlite"))
    assert indexes == [output / "index" / "feedbax.sqlite"]
    # No per-row store subtrees were minted.
    assert not any((output / row.row_id).exists() for row in batch.rows)
    # One shared manifest tree carries every row, keyed by manifest id.
    manifests = sorted((output / "manifests" / "evaluation_runs").glob("*.json"))
    assert len(manifests) == 4
    # One shared state cache tree carries every row's cached trajectory.
    caches = sorted((output / "cache" / "states").glob("*.pkl"))
    assert len(caches) == 4
    # Row identity and order are preserved through the shared store.
    assert [row.row_id for row in batch.rows] == ["row-0", "row-1", "row-2", "row-3"]
    for row in batch.rows:
        assert row.manifest_path.is_relative_to(output / "manifests")


def test_batched_matrix_fixed_store_overhead_independent_of_row_count(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    small_root = tmp_path / "small"
    large_root = tmp_path / "large"
    _register_scalar_batch_recipe(application_registry_bundle)
    small = execute_evaluation_run_matrix(
        _sized_matrix(2),
        root=small_root,
        batch=EvaluationBatchExecution(),
        registry=application_registry_bundle.evaluation_recipes,
    )
    large = execute_evaluation_run_matrix(
        _sized_matrix(16),
        root=large_root,
        batch=EvaluationBatchExecution(),
        registry=application_registry_bundle.evaluation_recipes,
    )

    assert len(small.rows) == 2 and len(large.rows) == 16
    # The heavy fixed machinery (the SQLite index) is created once per store,
    # not once per row: an 8x larger matrix still has exactly one index.
    small_indexes = list(small_root.rglob("feedbax.sqlite"))
    large_indexes = list(large_root.rglob("feedbax.sqlite"))
    assert len(small_indexes) == len(large_indexes) == 1
    # Per-row mode would have paid one index per row (2 vs 16). The shared store
    # holds total index overhead essentially flat: the single index grows only
    # by compact per-row metadata, nowhere near a whole extra index per row. Assert
    # the total index bytes for the 16-row store stay well under what even two
    # per-row indexes of the 2-row store would cost.
    small_index_bytes = small_indexes[0].stat().st_size
    large_index_bytes = large_indexes[0].stat().st_size
    per_row_16_lower_bound = 16 * small_index_bytes
    assert large_index_bytes < per_row_16_lower_bound / 2, (
        f"shared index for 16 rows ({large_index_bytes} B) is not clearly below "
        f"the per-row-store overhead 16 x {small_index_bytes} B"
    )


def test_batched_matrix_authenticates_staged_prerequisites_before_publish(
    tmp_path: Path,
    application_registry_bundle,
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
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            EvaluationRunSpec(evaluation_type=EVALUATION_TYPE).model_dump(mode="json"),
        ),
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

    application_registry_bundle.evaluation_recipes.register(
        EVALUATION_TYPE,
        lambda *_args: _result(1.0),
        batch_recipe=lambda items, _context: [_result(1.0) for _ in items],
    )
    output = tmp_path / "batch"
    with pytest.raises(EvaluationBatchRowError) as exc_info:
        execute_evaluation_run_matrix(
            matrix,
            root=output,
            parent_manifest_root=parent_root,
            batch=EvaluationBatchExecution(),
            registry=application_registry_bundle.evaluation_recipes,
        )

    assert exc_info.value.row_id == "row"
    assert isinstance(exc_info.value.__cause__, StagedExecutionContextError)
    assert "sha256 mismatch" in str(exc_info.value.__cause__)
    assert not output.exists()


def test_batch_validation_matches_scalar_reserved_field_and_callback_contract(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    bad_matrix = EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type=EVALUATION_TYPE,
            params={"staged_prerequisites": []},
        ),
        rows=[MatrixRow(row_id="row")],
    )
    failure_probe_registry = EvaluationRecipeRegistry()
    with pytest.raises(RecipeValidationError, match="must accept two positional arguments"):
        failure_probe_registry.register(
            EVALUATION_TYPE,
            lambda *_args: _result(1.0),
            batch_recipe=lambda _items: [],
        )

    application_registry_bundle.evaluation_recipes.register(
        EVALUATION_TYPE,
        lambda *_args: _result(1.0),
        batch_recipe=lambda items, _context: [_result(1.0) for _ in items],
    )
    with pytest.raises(TypeError, match="EvaluationBatchExecution"):
        execute_evaluation_run_matrix(
            bad_matrix,
            root=tmp_path / "marker",
            batch=object(),
            registry=application_registry_bundle.evaluation_recipes,
        )
    with pytest.raises(EvaluationBatchRowError) as exc_info:
        execute_evaluation_run_matrix(
            bad_matrix,
            root=tmp_path / "batch",
            batch=EvaluationBatchExecution(),
            registry=application_registry_bundle.evaluation_recipes,
        )
    with pytest.raises(TypeError) as scalar_exc:
        execute_evaluation_run_matrix(
            bad_matrix,
            root=tmp_path / "default",
            registry=application_registry_bundle.evaluation_recipes,
        )

    assert isinstance(exc_info.value.__cause__, TypeError)
    assert str(exc_info.value.__cause__) == str(scalar_exc.value)
    assert not (tmp_path / "batch").exists()
