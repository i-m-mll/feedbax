from __future__ import annotations

import io
import json
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pytest

from feedbax.analysis.evaluation import (
    EvaluationRecipeDiagnosticError,
    EvaluationRecipeResult,
    EvaluationRecipeExecutionError,
    load_evaluation_states,
    execute_evaluation_run_spec,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.contracts.evaluation_states import (
    EVALUATION_STATES_ARTIFACT_ROLE,
    EVALUATION_STATES_MEDIA_TYPE,
    EVALUATION_STATES_METADATA_KEY,
    EVALUATION_STATES_METADATA_VALUES_KEY,
    EvaluationStatesContainerError,
    EvaluationStatesHashMismatch,
    EvaluationStatesLeafError,
    evaluation_states_container_bytes,
    evaluation_states_container_bytes_v1,
    load_evaluation_states_container_bytes,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion
from feedbax.contracts.manifest import (
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION,
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V1,
    EvaluationRunSpec,
    EvaluationRunManifest,
    ParentRef,
    Provenance,
    evaluation_run_manifest_id,
    evaluation_states_cache_path,
    load_manifest,
    sha256_file,
    spec_payload,
    store_bytes_artifact,
)
from feedbax.persistence.manifest_index import rebuild_manifest_index


def _assert_tree_arrays_equal(left, right) -> None:
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_tree_arrays_equal(left[key], right[key])
        return
    if isinstance(left, tuple):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_tree_arrays_equal(left_item, right_item)
        return
    if isinstance(left, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_tree_arrays_equal(left_item, right_item)
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        return
    assert left == right


def test_evaluation_run_spec_executes_headless_and_reuses_manifest_cache(tmp_path: Path):
    calls: list[str] = []
    parent = ParentRef(
        kind="TrainingRunManifest",
        id="feedbax-training-run:toy",
        role="training_run",
    )
    spec = EvaluationRunSpec(
        evaluation_type="testpkg.toy_eval",
        inputs=[parent],
        params={"n_trials": 3},
    )
    spec_path = tmp_path / "evaluation-spec.json"
    spec_path.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def recipe(
        run_spec: EvaluationRunSpec,
        root: Path,
        states_path: Path,
        _execution_context,
    ) -> EvaluationRecipeResult:
        calls.append(str(root))
        return EvaluationRecipeResult(
            states={"training_run_ids": [ref.id for ref in run_spec.inputs]},
            summary_metrics={"n_trials": run_spec.params["n_trials"]},
            metadata={"states_path_seen": str(states_path)},
        )

    register_evaluation_recipe("testpkg.toy_eval", recipe, replace=True)
    try:
        manifest, path = execute_evaluation_run_spec(
            spec_path,
            root=tmp_path,
            issues=["8f40e2d"],
        )
        assert manifest.status == "completed"
        assert path.exists()
        assert manifest.id == evaluation_run_manifest_id(spec)
        assert manifest.evaluation_spec.inline["evaluation_type"] == "testpkg.toy_eval"
        assert manifest.input_training_runs == [parent]
        assert manifest.provenance.parents == [parent]
        assert manifest.provenance.issues == ["8f40e2d"]
        assert manifest.summary_metrics["n_trials"] == 3

        cache_path = evaluation_states_cache_path(manifest.id, root=tmp_path)
        assert cache_path.exists()
        assert manifest.metadata["cache"]["states_path"] == str(cache_path)
        assert not [
            artifact
            for artifact in manifest.artifacts
            if artifact.role == EVALUATION_STATES_ARTIFACT_ROLE
        ]

        loaded = load_manifest(path)
        assert loaded.id == manifest.id

        rerun_manifest, rerun_path = execute_evaluation_run_spec(spec, root=tmp_path)
        assert rerun_path == path
        assert rerun_manifest.id == manifest.id
        assert rerun_manifest.summary_metrics["n_trials"] == 3
        assert rerun_manifest.summary_metrics["input_training_runs"] == 1
        assert "states_cache_hit" not in rerun_manifest.summary_metrics
        assert rerun_manifest.metadata["cache"]["states_cache_hit"] is True
        assert calls == [str(tmp_path)]

        index_path = rebuild_manifest_index(tmp_path)
        with sqlite3.connect(index_path) as conn:
            row = conn.execute(
                "SELECT kind, status FROM manifests WHERE id = ?",
                (manifest.id,),
            ).fetchone()
            edge = conn.execute(
                """
                SELECT parent_kind, parent_id, role
                FROM lineage_edges
                WHERE child_id = ?
                """,
                (manifest.id,),
            ).fetchone()
        assert row == ("EvaluationRunManifest", "completed")
        assert edge == ("TrainingRunManifest", parent.id, "training_run")
    finally:
        unregister_evaluation_recipe("testpkg.toy_eval")


def test_evaluation_states_durable_custody_round_trips(tmp_path: Path):
    expected_states = {
        "float_batch": np.asarray([[1.0, 2.0], [3.5, 4.5]], dtype=np.float32),
        "metadata": {
            "bridge": "rlrmp",
            "enabled": True,
            "n_trials": 3,
            "tags": ["bridge", "certificate"],
            "threshold": 0.125,
        },
        "nested": (
            np.asarray([1, 2, 3], dtype=np.int32),
            {"sample": np.asarray(7, dtype=np.int64), "label": "scalar-metadata"},
        ),
    }
    spec = EvaluationRunSpec(
        evaluation_type="testpkg.durable_eval",
        params={"states_custody": "durable"},
    )

    def recipe(
        _run_spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
        _execution_context,
    ) -> EvaluationRecipeResult:
        return EvaluationRecipeResult(states=expected_states)

    register_evaluation_recipe("testpkg.durable_eval", recipe, replace=True)
    try:
        manifest, path = execute_evaluation_run_spec(spec, root=tmp_path)

        assert path.exists()
        artifacts = [
            artifact
            for artifact in manifest.artifacts
            if artifact.role == EVALUATION_STATES_ARTIFACT_ROLE
        ]
        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert artifact.media_type == EVALUATION_STATES_MEDIA_TYPE
        assert artifact.sha256 == sha256_file(Path(artifact.uri))
        assert artifact.size_bytes == Path(artifact.uri).stat().st_size
        assert artifact.metadata["schema_version"] == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION

        loaded_states = load_evaluation_states(manifest, root=tmp_path)
        _assert_tree_arrays_equal(expected_states, loaded_states)

        loaded_manifest = load_manifest(path)
        assert loaded_manifest.artifacts[0].sha256 == artifact.sha256

        data_a, payload_a = evaluation_states_container_bytes(expected_states)
        data_b, payload_b = evaluation_states_container_bytes(expected_states)
        assert payload_a.schema_version == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION
        assert payload_a.metadata_sha256 is not None
        assert data_a == data_b
        assert payload_a == payload_b
    finally:
        unregister_evaluation_recipe("testpkg.durable_eval")


def test_evaluation_states_tamper_fails_closed(tmp_path: Path):
    spec = EvaluationRunSpec(
        evaluation_type="testpkg.tamper_eval",
        params={"states_custody": "durable"},
    )

    def recipe(
        _run_spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
        _execution_context,
    ) -> EvaluationRecipeResult:
        return EvaluationRecipeResult(states={"value": np.asarray([1], dtype=np.int32)})

    register_evaluation_recipe("testpkg.tamper_eval", recipe, replace=True)
    try:
        manifest, _path = execute_evaluation_run_spec(spec, root=tmp_path)
        artifact = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == EVALUATION_STATES_ARTIFACT_ROLE
        )
        Path(artifact.uri).write_bytes(b"tampered")

        with pytest.raises(EvaluationStatesHashMismatch):
            load_evaluation_states(manifest, root=tmp_path)
    finally:
        unregister_evaluation_recipe("testpkg.tamper_eval")


def test_evaluation_states_unknown_container_version_rejected(tmp_path: Path):
    data, _payload = evaluation_states_container_bytes(
        {"value": np.asarray([1], dtype=np.int32)}
    )
    bad_data = _with_container_schema_version(
        data,
        "feedbax.manifest.evaluation_states_container.v0",
    )
    artifact = store_bytes_artifact(
        bad_data,
        root=tmp_path,
        role=EVALUATION_STATES_ARTIFACT_ROLE,
        logical_name="bad-version.states.npz",
        media_type=EVALUATION_STATES_MEDIA_TYPE,
        suffix=".npz",
        metadata={
            "schema_id": "feedbax.manifest.evaluation_states_container",
            "schema_version": "feedbax.manifest.evaluation_states_container.v0",
        },
    )
    manifest = EvaluationRunManifest(
        id="feedbax-evaluation-run:bad-version",
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            {"evaluation_type": "version_eval"},
        ),
        artifacts=[artifact],
    )

    with pytest.raises(UnsupportedSpecVersion, match="EvaluationStatesContainer"):
        load_evaluation_states(manifest, root=tmp_path)


def test_evaluation_states_v2_metadata_section_tamper_fails_closed() -> None:
    data, _payload = evaluation_states_container_bytes(
        {
            "array": np.asarray([1], dtype=np.int32),
            "metadata": {"label": "clean"},
        }
    )
    source = zipfile.ZipFile(io.BytesIO(data), mode="r")
    output = io.BytesIO()
    with source, zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as dest:
        for name in source.namelist():
            member_data = source.read(name)
            if name == EVALUATION_STATES_METADATA_VALUES_KEY:
                values = json.loads(member_data.decode("utf-8"))
                values[0]["value"] = "tampered"
                member_data = (
                    json.dumps(values, separators=(",", ":"), sort_keys=True).encode("utf-8")
                    + b"\n"
                )
            info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            dest.writestr(info, member_data)

    with pytest.raises(EvaluationStatesContainerError, match="metadata section digest"):
        load_evaluation_states_container_bytes(output.getvalue())


def test_evaluation_states_v1_loads_and_rejects_non_array_leaf_path() -> None:
    states = {
        "float_batch": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "nested": (np.asarray([1, 2, 3], dtype=np.int32),),
    }
    data, payload = evaluation_states_container_bytes_v1(states)

    assert payload.schema_version == EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V1
    _assert_tree_arrays_equal(states, load_evaluation_states_container_bytes(data))

    with pytest.raises(EvaluationStatesLeafError) as excinfo:
        evaluation_states_container_bytes_v1(
            {
                "ok": np.asarray([1], dtype=np.int32),
                "bad": "metadata",
            }
        )
    message = str(excinfo.value)
    assert "['bad']" in message
    assert "str" in message


def test_evaluation_states_durable_rejects_non_json_metadata_leaf_path(tmp_path: Path):
    spec = EvaluationRunSpec(
        evaluation_type="testpkg.exotic_eval",
        params={"states_custody": "durable"},
    )

    def recipe(
        _run_spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
        _execution_context,
    ) -> EvaluationRecipeResult:
        return EvaluationRecipeResult(
            states={
                "ok": np.asarray([1], dtype=np.int32),
                "bad": object(),
            }
        )

    register_evaluation_recipe("testpkg.exotic_eval", recipe, replace=True)
    try:
        with pytest.raises(EvaluationRecipeExecutionError) as excinfo:
            execute_evaluation_run_spec(spec, root=tmp_path)
        assert isinstance(excinfo.value.__cause__, EvaluationStatesLeafError)
        message = str(excinfo.value.__cause__)
        assert "['bad']" in message
        assert "object" in message
    finally:
        unregister_evaluation_recipe("testpkg.exotic_eval")


def test_evaluation_run_spec_copies_caller_provenance_before_stamping(tmp_path: Path):
    parent = ParentRef(
        kind="TrainingRunManifest",
        id="feedbax-training-run:copy-provenance",
        role="training_run",
    )
    spec = EvaluationRunSpec(
        evaluation_type="testpkg.copy_provenance_eval",
        inputs=[parent],
        params={},
    )
    caller_provenance = Provenance(
        source_commit="abc123",
        dirty=False,
        issues=["existing"],
    )

    def recipe(
        _run_spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
        _execution_context,
    ) -> EvaluationRecipeResult:
        return EvaluationRecipeResult()

    register_evaluation_recipe("testpkg.copy_provenance_eval", recipe, replace=True)
    try:
        manifest, _path = execute_evaluation_run_spec(
            spec,
            root=tmp_path,
            provenance=caller_provenance,
            issues=["new"],
        )

        assert manifest.provenance is not caller_provenance
        assert manifest.provenance.source_commit == "abc123"
        assert manifest.provenance.parents == [parent]
        assert manifest.provenance.issues == ["existing", "new"]
        assert manifest.provenance.entrypoint is not None
        assert caller_provenance.parents == []
        assert caller_provenance.issues == ["existing"]
        assert caller_provenance.entrypoint is None
    finally:
        unregister_evaluation_recipe("testpkg.copy_provenance_eval")


def test_evaluation_failure_diagnostics_round_trip_with_recipe_provenance(tmp_path: Path):
    evaluation_type = "testpkg.diagnostic_eval"
    spec = EvaluationRunSpec(evaluation_type=evaluation_type)

    def recipe(*_args) -> EvaluationRecipeResult:
        raise EvaluationRecipeDiagnosticError(
            "solver rejected the condition",
            {
                "schema_id": "testpkg.solver_failure",
                "schema_version": "testpkg.solver_failure.v1",
                "values": {"residual": 0.25, "iterations": 7},
            },
        )

    register_evaluation_recipe(evaluation_type, recipe, replace=True)
    try:
        with pytest.raises(EvaluationRecipeExecutionError) as excinfo:
            execute_evaluation_run_spec(spec, root=tmp_path)
        manifest = excinfo.value.manifest
        diagnostics = manifest.metadata["error"]["diagnostics"]
        assert diagnostics["schema_id"] == "testpkg.solver_failure"
        assert diagnostics["schema_version"] == "testpkg.solver_failure.v1"
        assert diagnostics["values"] == {"residual": 0.25, "iterations": 7}
        assert diagnostics["recipe"] == {
            "evaluation_type": evaluation_type,
            "entrypoint": {
                "kind": "feedbax-evaluation-recipe",
                "name": evaluation_type,
                "metadata": {},
            },
        }
        assert load_manifest(excinfo.value.path).metadata["error"] == manifest.metadata["error"]
    finally:
        unregister_evaluation_recipe(evaluation_type)


def test_evaluation_failure_without_diagnostics_keeps_ordinary_error(tmp_path: Path):
    evaluation_type = "testpkg.ordinary_failure"

    def recipe(*_args) -> EvaluationRecipeResult:
        raise RuntimeError("ordinary failure")

    register_evaluation_recipe(evaluation_type, recipe, replace=True)
    try:
        with pytest.raises(EvaluationRecipeExecutionError) as excinfo:
            execute_evaluation_run_spec(
                EvaluationRunSpec(evaluation_type=evaluation_type),
                root=tmp_path,
            )
        assert excinfo.value.manifest.metadata["error"] == {
            "type": "RuntimeError",
            "message": "ordinary failure",
        }
    finally:
        unregister_evaluation_recipe(evaluation_type)


@pytest.mark.parametrize(
    "diagnostics",
    [
        {"schema_version": "testpkg.failure.v1", "values": {}},
        {
            "schema_id": "testpkg.failure",
            "schema_version": "testpkg.failure.v1",
            "values": {"path": Path("/private/diagnostic-secret")},
        },
        {
            "schema_id": "testpkg.failure",
            "schema_version": "testpkg.failure.v1",
            "values": {"oversized": "x" * 17_000},
        },
        {
            "schema_id": "testpkg.failure",
            "schema_version": "testpkg.failure.v1",
            "values": {},
            "recipe": {"evaluation_type": "forged.recipe"},
        },
    ],
)
def test_invalid_evaluation_failure_diagnostics_are_rejected_safely(
    tmp_path: Path,
    diagnostics: dict[str, object],
):
    evaluation_type = "testpkg.invalid_diagnostics"

    def recipe(*_args) -> EvaluationRecipeResult:
        raise EvaluationRecipeDiagnosticError("scientific failure", diagnostics)

    register_evaluation_recipe(evaluation_type, recipe, replace=True)
    try:
        with pytest.raises(EvaluationRecipeExecutionError) as excinfo:
            execute_evaluation_run_spec(
                EvaluationRunSpec(evaluation_type=evaluation_type),
                root=tmp_path,
            )
        error = excinfo.value.manifest.metadata["error"]
        assert error == {
            "type": "ValueError",
            "message": "evaluation failure diagnostics payload is invalid",
        }
        assert "diagnostic-secret" not in excinfo.value.path.read_text(encoding="utf-8")
    finally:
        unregister_evaluation_recipe(evaluation_type)


def _with_container_schema_version(data: bytes, schema_version: str) -> bytes:
    source = zipfile.ZipFile(io.BytesIO(data), mode="r")
    output = io.BytesIO()
    with source, zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as dest:
        for name in source.namelist():
            member_data = source.read(name)
            if name == EVALUATION_STATES_METADATA_KEY:
                payload = json.loads(member_data.decode("utf-8"))
                payload["schema_version"] = schema_version
                member_data = (
                    json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
                )
            info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            dest.writestr(info, member_data)
    return output.getvalue()
