from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    EvaluationRunMatrixSpec,
    execute_evaluation_run_matrix,
    execute_evaluation_run_spec,
    materialize_evaluation_run_matrix,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedArtifactProviderRootBinding,
    StagedExecutionContextError,
)
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.evaluation_states import store_evaluation_states_artifact
from feedbax.contracts.expressions import ValueQuery
from feedbax.contracts.manifest import (
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1,
    EvaluationRunManifest,
    EvaluationRunSpec,
    OverridePatch,
    ParentRef,
    SpecPayload,
    StagedEvaluationPrerequisite,
    TrainingRunManifest,
    write_manifest,
)
from feedbax.contracts.matrix_core import MatrixRow, RowDerivation, derive_row_path
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.staged_execution import (
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedExecutionDescriptor,
)
from feedbax.persistence.artifact_custody import open_immutable_artifact_blob_provider


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

    v1 = {
        key: value
        for key, value in payload.items()
        if key != "staged_parents"
    }
    v1["schema_version"] = EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1
    migrated = default_spec_registry.migrate("EvaluationRunMatrixSpec", v1)
    assert migrated.migrated
    assert migrated.payload["staged_parents"] == {}
    assert migrated.target_version == EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION

    legacy = {**payload, "schema_version": "feedbax.spec.evaluation_run_matrix.v0"}
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent"):
        default_spec_registry.migrate("EvaluationRunMatrixSpec", legacy)


def test_public_materializer_migrates_serialized_v1_context_free_matrix() -> None:
    payload = _matrix().model_dump(mode="json")
    payload.pop("staged_parents")
    payload["schema_version"] = EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1

    rows = materialize_evaluation_run_matrix(payload)

    assert [row.row_id for row in rows] == ["control", "treatment"]
    assert rows[0].payload.params["gain"] == 2.0


def test_evaluation_matrix_schema_identity_is_pinned() -> None:
    payload = _matrix().model_dump(mode="json")
    payload["schema_id"] = "example.spec.evaluation_matrix"
    with pytest.raises(ValidationError, match="unsupported EvaluationRunMatrixSpec schema_id"):
        EvaluationRunMatrixSpec.model_validate(payload)


def test_evaluation_matrix_executes_through_harness(tmp_path: Path) -> None:
    def recipe(spec, _root, _states_path, _execution_context):
        assert not _execution_context.parent_execution_locations
        return EvaluationRecipeResult(summary_metrics={"gain": spec.params["gain"]})

    register_evaluation_recipe("example.evaluate", recipe)
    try:
        result = execute_evaluation_run_matrix(_matrix(), root=tmp_path)
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


def test_direct_single_run_keeps_empty_staged_context(tmp_path: Path) -> None:
    observed = []

    def recipe(_spec, _root, _states_path, execution_context):
        observed.append(execution_context)
        return EvaluationRecipeResult(summary_metrics={"direct": True})

    register_evaluation_recipe("example.direct", recipe)
    try:
        manifest, path = execute_evaluation_run_spec(
            EvaluationRunSpec(evaluation_type="example.direct"),
            root=tmp_path,
        )
    finally:
        unregister_evaluation_recipe("example.direct")

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
    output_root: Path,
    **kwargs,
):
    observed: list[tuple[Path, str, list[int]]] = []

    def recipe(spec, root, _states_path, execution_context):
        training = execution_context.resolve_manifest_input(spec.inputs[0])
        bank = StagedEvaluationPrerequisite.model_validate(
            spec.params["staged_prerequisites"]["paired_bank"]
        )
        states = execution_context.load_evaluation_states(bank.parent)
        observed.append((root, training.manifest.id, states["pair"].tolist()))
        return EvaluationRecipeResult(summary_metrics={"pair_count": len(states["pair"])})

    register_evaluation_recipe("example.staged_matrix", recipe)
    try:
        result = execute_evaluation_run_matrix(matrix, root=output_root, **kwargs)
    finally:
        unregister_evaluation_recipe("example.staged_matrix")
    return result, observed


def test_matrix_resolves_shared_local_parents_before_distinct_row_roots(
    tmp_path: Path,
) -> None:
    parent_root = tmp_path / "parents"
    parent_root.mkdir()
    training = TrainingRunManifest(id="feedbax-training-run:shared", status="completed")
    training_path = write_manifest(training, root=parent_root, index=False)
    training_ref = authenticated_manifest_ref(training, training_path, "training_run")
    artifact = store_evaluation_states_artifact(
        {"pair": np.asarray([3, 5])},
        root=parent_root,
        manifest_id="paired-bank",
    )
    artifact = artifact.model_copy(update={"uri": artifact.metadata["relative_path"]})
    bank = _evaluation_manifest(artifact)
    bank_path = write_manifest(bank, root=parent_root, index=False)
    bank_ref = authenticated_manifest_ref(bank, bank_path, "evaluation_run")
    matrix = _staged_matrix(training_ref, bank_ref)

    result, observed = _run_staged_matrix(
        matrix,
        output_root=tmp_path / "rows",
        parent_manifest_root=parent_root,
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
        assert row.result.metadata["matrix_harness"]["staged_parents"] == (
            result.metadata["staged_parents"]
        )


def test_matrix_resolves_shared_provider_parents_and_durable_bank(tmp_path: Path) -> None:
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
        {"pair": np.asarray([8, 13])}, root=source_root, manifest_id="provider-bank"
    )
    state_bytes = (source_root / source_artifact.metadata["relative_path"]).read_bytes()
    provider_artifact = provider.store_bytes(
        state_bytes, role="evaluation_states", logical_name="states.npz"
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

    result, observed = _run_staged_matrix(
        _staged_matrix(training_ref, bank_ref, artifact_provider="shared"),
        output_root=tmp_path / "rows",
        execution_descriptor=descriptor,
        artifact_provider_bindings=[
            StagedArtifactProviderRootBinding("shared", provider_root)
        ],
    )

    assert [item[2] for item in observed] == [[8, 13], [8, 13]]
    assert result.metadata["staged_parents"]["paired_bank"]["artifact_provider"] == "shared"


def test_matrix_staged_parent_contract_fails_closed_before_row_creation(
    tmp_path: Path,
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
        materialize_evaluation_run_matrix(unreferenced)

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
            root=tmp_path / "rows",
            parent_manifest_root=parent_root,
        )
    assert not (tmp_path / "rows").exists()

    with pytest.raises(StagedExecutionContextError, match="parent_manifest_root"):
        execute_evaluation_run_matrix(matrix, root=tmp_path / "rows")
    with pytest.raises(StagedExecutionContextError, match="must be absolute"):
        execute_evaluation_run_matrix(
            matrix,
            root=tmp_path / "rows",
            parent_manifest_root="relative/parents",
        )
