from __future__ import annotations

import json
import io
import zipfile
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import jax.tree as jt
import numpy as np
import pytest
from pydantic import ValidationError

from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    execute_evaluation_run_spec,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.analysis.specs import (
    AnalysisEvaluationStatesResolutionError,
    AnalysisRecipeExecutionError,
    AnalysisRecipeResult,
    coerce_analysis_run_spec,
    execute_analysis_run_spec,
    find_manifest_by_id,
    register_analysis_recipe,
    resolve_analysis_inputs,
    unregister_analysis_recipe,
)
from feedbax.analysis.manifest_inputs import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    authenticated_manifest_ref,
    is_authenticated_manifest_ref,
)
from feedbax.contracts.evaluation_states import (
    EVALUATION_STATES_METADATA_KEY,
    EvaluationStatesCustodyUnavailable,
    EvaluationStatesSizeMismatch,
)
from feedbax.contracts.manifest import (
    ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION,
    ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION_V1,
    ANALYSIS_EVALUATION_STATE_SOURCE_SCHEMA_VERSION_V1,
    ANALYSIS_RUN_SPEC_SCHEMA_VERSION,
    ANALYSIS_RUN_SPEC_SCHEMA_VERSION_V1,
    EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3,
    AnalysisEvaluationStateSource,
    AnalysisRunSpec,
    EvaluationRunSpec,
    ParentRef,
    evaluation_states_cache_path,
    load_manifest,
    load_manifest_bytes,
    store_bytes_artifact,
    write_manifest,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from tests.analysis_fixtures import ToyAnalysis, build_toy_analysis_data


pytestmark = [pytest.mark.feedbax_contract, pytest.mark.analysis_recipe_contract]

EVALUATION_TYPE = "feedbax.test.analysis_states_policy_evaluation"
ANALYSIS_TYPE = "feedbax.test.analysis_states_policy_analysis"


def _register_evaluation() -> None:
    def recipe(run_spec, _root, _states_path, _execution_context):
        return EvaluationRecipeResult(
            states={"value": np.asarray(run_spec.params["value"], dtype=np.int32)},
        )

    register_evaluation_recipe(EVALUATION_TYPE, recipe, replace=True)


def _register_analysis() -> None:
    def recipe(_spec, _root, inputs, _execution_context):
        value = sum(int(resolved.states["value"]) for resolved in inputs)
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="policy", cache_result=True)},
            data=build_toy_analysis_data(value=value),
        )

    register_analysis_recipe(ANALYSIS_TYPE, recipe, replace=True)


def _evaluation(
    root: Path,
    *,
    value: int = 3,
    durable: bool = False,
):
    spec = EvaluationRunSpec(
        evaluation_type=EVALUATION_TYPE,
        inputs=[
            ParentRef(
                kind="TrainingRunManifest",
                id=f"feedbax-training-run:analysis-policy-{value}-{durable}",
                role="training_run",
            )
        ],
        params={
            "value": value,
            **({"states_custody": "durable"} if durable else {}),
        },
    )
    return execute_evaluation_run_spec(spec, root=root, force=True)


def _analysis_spec(
    manifest_id: str,
    *,
    policy: str,
    ref: ParentRef | None = None,
    root: Path | None = None,
) -> AnalysisRunSpec:
    if policy == "require_durable" and ref is None and root is not None:
        manifest, path = find_manifest_by_id(manifest_id, root=root)
        ref = authenticated_manifest_ref(manifest, path, "evaluation_run")
    return AnalysisRunSpec(
        analysis_type=ANALYSIS_TYPE,
        inputs=[
            ref
            or ParentRef(
                kind="EvaluationRunManifest",
                id=manifest_id,
                role="evaluation_run",
            )
        ],
        evaluation_states_policy=policy,
    )


def _authenticated_evaluation_authority() -> ParentRef:
    return ParentRef(
        kind="EvaluationRunManifest",
        id="feedbax-evaluation-run:source-model-authority",
        role="evaluation_run",
        metadata={
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": "a" * 64,
            "size_bytes": 123,
        },
    )


def _evaluation_state_source(
    source_kind: str,
    authority: ParentRef,
) -> AnalysisEvaluationStateSource:
    if source_kind == "durable":
        return AnalysisEvaluationStateSource(
            source_kind="durable",
            requested_evaluation_manifest_id=authority.id,
            evaluation_manifest_authority=authority,
            supplying_evaluation_manifest_id=authority.id,
            artifact_id="artifact:sha256:states",
            artifact_sha256="b" * 64,
            artifact_size_bytes=456,
            artifact_storage_backend="feedbax-local",
            container_schema_id="feedbax.manifest.evaluation_states_container",
            container_schema_version="feedbax.manifest.evaluation_states_container.v2",
            container_storage_backend="npz.v2",
        )
    return AnalysisEvaluationStateSource(
        source_kind="analysis_time_recompute",
        requested_evaluation_manifest_id=authority.id,
        resulting_evaluation_manifest_id=authority.id,
        resulting_evaluation_manifest_authority=authority,
    )


def _invalid_evaluation_authority(profile_case: str) -> ParentRef:
    authority = _authenticated_evaluation_authority()
    metadata = dict(authority.metadata)
    uri = None
    if profile_case == "id-only":
        metadata = {}
    elif profile_case == "partial":
        metadata = {"ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID}
    elif profile_case == "malformed":
        metadata = {"manifest_sha256": "a" * 64, "size_bytes": 123}
    elif profile_case == "uri-bearing":
        uri = "/machine/local/evaluation.json"
    elif profile_case == "wrong-schema":
        metadata["ref_schema_id"] = "feedbax.ref.other"
    elif profile_case == "wrong-version":
        metadata["ref_schema_version"] = "feedbax.ref.authenticated_manifest.v0"
    elif profile_case == "invalid-hash":
        metadata["manifest_sha256"] = "not-a-sha256"
    elif profile_case == "invalid-size":
        metadata["size_bytes"] = True
    else:  # pragma: no cover - parametrization controls this helper.
        raise AssertionError(f"unknown profile case: {profile_case}")
    return authority.model_copy(update={"metadata": metadata, "uri": uri})


@pytest.mark.parametrize("source_kind", ["durable", "analysis_time_recompute"])
def test_evaluation_state_source_accepts_complete_authenticated_authorities(
    source_kind: str,
) -> None:
    source = _evaluation_state_source(source_kind, _authenticated_evaluation_authority())
    assert source.source_kind == source_kind


@pytest.mark.parametrize("source_kind", ["durable", "analysis_time_recompute"])
@pytest.mark.parametrize(
    "profile_case",
    [
        "id-only",
        "partial",
        "malformed",
        "uri-bearing",
        "wrong-schema",
        "wrong-version",
        "invalid-hash",
        "invalid-size",
    ],
)
def test_evaluation_state_source_rejects_incomplete_authenticated_authorities(
    source_kind: str,
    profile_case: str,
) -> None:
    with pytest.raises(ValidationError):
        _evaluation_state_source(source_kind, _invalid_evaluation_authority(profile_case))


def _write_mutated_artifact_metadata(root: Path, manifest, **updates: object) -> None:
    artifact = manifest.artifacts[0]
    metadata = {**artifact.metadata, **updates}
    mutated = manifest.model_copy(
        update={"artifacts": [artifact.model_copy(update={"metadata": metadata})]}
    )
    write_manifest(mutated, root=root)


def _artifact_with_mutated_container_storage(root: Path, artifact):
    source = root / artifact.metadata["relative_path"]
    with zipfile.ZipFile(source, mode="r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    header = json.loads(members[EVALUATION_STATES_METADATA_KEY])
    header["storage_backend"] = "npz.v1"
    members[EVALUATION_STATES_METADATA_KEY] = (
        json.dumps(header, indent=2, sort_keys=True).encode() + b"\n"
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    metadata = {
        key: value for key, value in artifact.metadata.items() if key != "relative_path"
    }
    return store_bytes_artifact(
        output.getvalue(),
        root=root,
        role=artifact.role,
        logical_name=artifact.logical_name,
        media_type=artifact.media_type,
        suffix=".npz",
        metadata=metadata,
    )


def test_require_durable_bypasses_cache_and_never_recomputes(tmp_path: Path) -> None:
    _register_evaluation()
    try:
        manifest, _ = _evaluation(tmp_path)
        assert evaluation_states_cache_path(manifest.id, root=tmp_path).exists()

        with (
            patch(
                "feedbax.analysis.specs.execute_evaluation_run_spec",
                side_effect=AssertionError("require_durable must not recompute"),
            ) as execute,
            pytest.raises(AnalysisEvaluationStatesResolutionError) as excinfo,
        ):
            resolve_analysis_inputs(
                _analysis_spec(manifest.id, policy="require_durable", root=tmp_path),
                root=tmp_path,
            )

        assert excinfo.value.diagnostic.code == "missing_durable_states"
        execute.assert_not_called()
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_require_durable_threads_typed_structure_provider_end_to_end(tmp_path: Path) -> None:
    class TypedStates(NamedTuple):
        value: object

    def evaluation_recipe(run_spec, _root, _states_path, _execution_context):
        return EvaluationRecipeResult(
            states=TypedStates(np.asarray(run_spec.params["value"], dtype=np.int32))
        )

    register_evaluation_recipe(EVALUATION_TYPE, evaluation_recipe, replace=True)
    try:
        manifest, manifest_path = _evaluation(tmp_path, durable=True)
        register_analysis_recipe(
            ANALYSIS_TYPE,
            lambda *_args: None,
            replace=True,
            evaluation_states_structure=lambda _: jt.structure(TypedStates(np.asarray(0))),
        )
        resolved = resolve_analysis_inputs(
            _analysis_spec(manifest.id, policy="require_durable", root=tmp_path), root=tmp_path
        )[0]
        assert isinstance(resolved.states, TypedStates)
        assert resolved.manifest_input is not None
        assert resolved.manifest_input.ref == authenticated_manifest_ref(
            manifest,
            manifest_path,
            "evaluation_run",
        )
        assert resolved.manifest_input.raw_bytes == manifest_path.read_bytes()
        assert resolved.evaluation_state_source.container_schema_version == (
            EVALUATION_STATES_CONTAINER_SCHEMA_VERSION_V3
        )
        assert resolved.evaluation_state_source.container_storage_backend == "npz.v3"

        register_analysis_recipe(ANALYSIS_TYPE, lambda *_args: None, replace=True)
        with pytest.raises(AnalysisEvaluationStatesResolutionError) as excinfo:
            resolve_analysis_inputs(
                _analysis_spec(manifest.id, policy="require_durable", root=tmp_path), root=tmp_path
            )
        assert excinfo.value.diagnostic.code == "custody_unavailable"
    finally:
        unregister_analysis_recipe(ANALYSIS_TYPE)
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_require_durable_classifies_structure_provider_failure(tmp_path: Path) -> None:
    _register_evaluation()
    try:
        manifest, _ = _evaluation(tmp_path, durable=True)

        def fail_provider(_manifest):
            raise RuntimeError("provider exploded")

        register_analysis_recipe(
            ANALYSIS_TYPE,
            lambda *_args: None,
            replace=True,
            evaluation_states_structure=fail_provider,
        )
        with pytest.raises(AnalysisEvaluationStatesResolutionError) as excinfo:
            resolve_analysis_inputs(
                _analysis_spec(manifest.id, policy="require_durable", root=tmp_path),
                root=tmp_path,
            )

        assert excinfo.value.diagnostic.code == "custody_unavailable"
        assert excinfo.value.diagnostic.details["provider_failure"] == "provider exploded"
        assert isinstance(excinfo.value.__cause__, RuntimeError)
    finally:
        unregister_analysis_recipe(ANALYSIS_TYPE)
        unregister_evaluation_recipe(EVALUATION_TYPE)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ({"schema_version": "feedbax.manifest.evaluation_states_container.v0"}, "schema_mismatch"),
        ({"manifest_id": "feedbax-evaluation-run:wrong"}, "provenance_mismatch"),
    ],
)
def test_require_durable_classifies_schema_and_provenance_failures(
    tmp_path: Path,
    mutation: dict[str, object],
    expected_code: str,
) -> None:
    _register_evaluation()
    try:
        manifest, _ = _evaluation(tmp_path, durable=True)
        _write_mutated_artifact_metadata(tmp_path, manifest, **mutation)

        with pytest.raises(AnalysisEvaluationStatesResolutionError) as excinfo:
            resolve_analysis_inputs(
                _analysis_spec(manifest.id, policy="require_durable", root=tmp_path),
                root=tmp_path,
            )

        assert excinfo.value.diagnostic.code == expected_code
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_require_durable_classifies_missing_custody_bytes(tmp_path: Path) -> None:
    _register_evaluation()
    try:
        manifest, _ = _evaluation(tmp_path, durable=True)
        artifact = manifest.artifacts[0]
        missing = artifact.model_copy(
            update={
                "uri": str(tmp_path / "missing.states.npz"),
                "metadata": {key: value for key, value in artifact.metadata.items() if key != "relative_path"},
            }
        )
        write_manifest(manifest.model_copy(update={"artifacts": [missing]}), root=tmp_path)

        with pytest.raises(AnalysisEvaluationStatesResolutionError) as excinfo:
            resolve_analysis_inputs(
                _analysis_spec(manifest.id, policy="require_durable", root=tmp_path),
                root=tmp_path,
            )

        assert excinfo.value.diagnostic.code == "custody_unavailable"
        assert isinstance(excinfo.value.__cause__, EvaluationStatesCustodyUnavailable)
        assert isinstance(excinfo.value.__cause__.__cause__, FileNotFoundError)
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_recompute_preserves_legacy_durable_loader_metadata_tolerance(tmp_path: Path) -> None:
    _register_evaluation()
    try:
        manifest, _ = _evaluation(tmp_path, durable=True)
        _write_mutated_artifact_metadata(
            tmp_path,
            manifest,
            manifest_id="feedbax-evaluation-run:not-the-supplier",
        )
        evaluation_states_cache_path(manifest.id, root=tmp_path).unlink()

        resolved = resolve_analysis_inputs(
            _analysis_spec(manifest.id, policy="recompute"),
            root=tmp_path,
        )[0]
        assert resolved.evaluation_state_source is not None
        assert resolved.evaluation_state_source.source_kind == "durable"

        with pytest.raises(AnalysisEvaluationStatesResolutionError) as excinfo:
            resolve_analysis_inputs(
                _analysis_spec(manifest.id, policy="require_durable", root=tmp_path),
                root=tmp_path,
            )
        assert excinfo.value.diagnostic.code == "provenance_mismatch"
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_require_durable_classifies_embedded_container_schema_mismatch(tmp_path: Path) -> None:
    _register_evaluation()
    try:
        manifest, _ = _evaluation(tmp_path, durable=True)
        mutated_artifact = _artifact_with_mutated_container_storage(
            tmp_path,
            manifest.artifacts[0],
        )
        write_manifest(
            manifest.model_copy(update={"artifacts": [mutated_artifact]}),
            root=tmp_path,
        )

        with pytest.raises(AnalysisEvaluationStatesResolutionError) as excinfo:
            resolve_analysis_inputs(
                _analysis_spec(manifest.id, policy="require_durable", root=tmp_path),
                root=tmp_path,
            )

        assert excinfo.value.diagnostic.code == "schema_mismatch"
        assert mutated_artifact.artifact_id != manifest.artifacts[0].artifact_id
        assert mutated_artifact.size_bytes is not None
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_all_recompute_supplier_paths_are_truthfully_stamped(tmp_path: Path) -> None:
    _register_evaluation()
    try:
        cached, _ = _evaluation(tmp_path, value=4)
        cached_input = resolve_analysis_inputs(
            _analysis_spec(cached.id, policy="recompute"), root=tmp_path
        )[0]
        assert cached_input.evaluation_state_source is not None
        assert cached_input.evaluation_state_source.source_kind == "evaluation_cache"
        assert cached_input.evaluation_state_source.cache_schema_version == (
            "feedbax.analysis.evaluation-states-cache.v1"
        )
        assert cached_input.evaluation_state_source.cache_key == cached.id

        durable, _ = _evaluation(tmp_path, value=5, durable=True)
        evaluation_states_cache_path(durable.id, root=tmp_path).unlink()
        durable_input = resolve_analysis_inputs(
            _analysis_spec(durable.id, policy="recompute"), root=tmp_path
        )[0]
        assert durable_input.evaluation_state_source is not None
        assert durable_input.evaluation_state_source.source_kind == "durable"
        assert durable_input.evaluation_state_source.artifact_id == durable.artifacts[0].artifact_id
        durable_authority = durable_input.evaluation_state_source.evaluation_manifest_authority
        assert durable_authority is not None
        assert is_authenticated_manifest_ref(durable_authority)

        rederived, _ = _evaluation(tmp_path, value=6)
        evaluation_states_cache_path(rederived.id, root=tmp_path).unlink()
        recomputed_input = resolve_analysis_inputs(
            _analysis_spec(rederived.id, policy="recompute"), root=tmp_path
        )[0]
        assert recomputed_input.evaluation_state_source is not None
        assert recomputed_input.evaluation_state_source.source_kind == "analysis_time_recompute"
        assert recomputed_input.evaluation_state_source.resulting_evaluation_manifest_id == rederived.id
        resulting_authority = (
            recomputed_input.evaluation_state_source.resulting_evaluation_manifest_authority
        )
        assert resulting_authority is not None
        assert is_authenticated_manifest_ref(resulting_authority)
        assert resulting_authority.metadata["ref_schema_version"] == (
            AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION
        )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_completed_analysis_manifests_round_trip_all_supplier_kinds(tmp_path: Path) -> None:
    _register_evaluation()
    _register_analysis()
    try:
        cached, _ = _evaluation(tmp_path, value=14)
        durable, _ = _evaluation(tmp_path, value=15, durable=True)
        recomputed, _ = _evaluation(tmp_path, value=16)
        evaluation_states_cache_path(recomputed.id, root=tmp_path).unlink()

        cases = (
            (cached, "recompute", "evaluation_cache"),
            (durable, "require_durable", "durable"),
            (recomputed, "recompute", "analysis_time_recompute"),
        )
        for evaluation, policy, expected_kind in cases:
            manifest, path = execute_analysis_run_spec(
                _analysis_spec(evaluation.id, policy=policy, root=tmp_path),
                root=tmp_path,
                fig_dump_formats=("json",),
            )
            assert manifest.evaluation_state_sources[0].source_kind == expected_kind
            assert load_manifest(path).evaluation_state_sources == manifest.evaluation_state_sources
    finally:
        unregister_analysis_recipe(ANALYSIS_TYPE)
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_source_evidence_round_trips_and_historical_manifest_has_none(tmp_path: Path) -> None:
    _register_evaluation()
    try:
        evaluation, _ = _evaluation(tmp_path, durable=True)
        evaluation_states_cache_path(evaluation.id, root=tmp_path).unlink()
        spec = _analysis_spec(evaluation.id, policy="require_durable", root=tmp_path)
        resolved = resolve_analysis_inputs(spec, root=tmp_path)[0]
        assert resolved.evaluation_state_source is not None

        context = AnalysisRunContext(spec=spec, root=tmp_path)
        context.record_evaluation_state_sources([resolved.evaluation_state_source])
        manifest, path = context.finalize()
        loaded = load_manifest(path)
        assert loaded.schema_version == ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION
        assert loaded.evaluation_state_sources == manifest.evaluation_state_sources

        historical = manifest.model_dump(mode="json", exclude_none=True)
        historical["schema_version"] = ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION_V1
        historical.pop("evaluation_state_sources")
        historical.pop("evaluation_state_resolution_diagnostics")
        registry_migration = default_spec_registry.migrate(
            "AnalysisRunManifest",
            historical,
            source_version=ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION_V1,
        )
        migrated = load_manifest_bytes(json.dumps(historical).encode())
        assert migrated.schema_version == ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION
        assert migrated.evaluation_state_sources == []
        assert migrated.evaluation_state_resolution_diagnostics == []
        assert registry_migration.payload["evaluation_state_sources"] == []
        assert registry_migration.payload["evaluation_state_resolution_diagnostics"] == []
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_authenticated_manifest_authority_round_trips_in_source_evidence(
    tmp_path: Path,
) -> None:
    _register_evaluation()
    _register_analysis()
    try:
        evaluation, evaluation_path = _evaluation(tmp_path, durable=True)
        authority = authenticated_manifest_ref(
            evaluation,
            evaluation_path,
            "evaluation_run",
        )
        manifest, path = execute_analysis_run_spec(
            _analysis_spec(
                evaluation.id,
                policy="require_durable",
                ref=authority,
            ),
            root=tmp_path,
            fig_dump_formats=("json",),
        )

        source = load_manifest(path).evaluation_state_sources[0]
        assert source == manifest.evaluation_state_sources[0]
        assert source.evaluation_manifest_authority == authority
        assert source.evaluation_manifest_authority.uri is None
        assert (
            source.evaluation_manifest_authority.metadata["manifest_sha256"]
            == authority.metadata["manifest_sha256"]
        )
        assert (
            source.evaluation_manifest_authority.metadata["size_bytes"]
            == authority.metadata["size_bytes"]
        )
        assert source.artifact_size_bytes == evaluation.artifacts[0].size_bytes
    finally:
        unregister_analysis_recipe(ANALYSIS_TYPE)
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_require_durable_rejects_declared_artifact_size_mismatch(tmp_path: Path) -> None:
    _register_evaluation()
    try:
        evaluation, _ = _evaluation(tmp_path, durable=True)
        artifact = evaluation.artifacts[0]
        assert artifact.size_bytes is not None
        mismatched = artifact.model_copy(update={"size_bytes": artifact.size_bytes + 1})
        write_manifest(evaluation.model_copy(update={"artifacts": [mismatched]}), root=tmp_path)

        with pytest.raises(AnalysisEvaluationStatesResolutionError) as excinfo:
            resolve_analysis_inputs(
                _analysis_spec(evaluation.id, policy="require_durable", root=tmp_path),
                root=tmp_path,
            )

        assert excinfo.value.diagnostic.code == "custody_unavailable"
        assert isinstance(excinfo.value.__cause__, EvaluationStatesSizeMismatch)
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_recompute_preserves_legacy_declared_size_tolerance(tmp_path: Path) -> None:
    _register_evaluation()
    try:
        evaluation, _ = _evaluation(tmp_path, durable=True)
        artifact = evaluation.artifacts[0]
        assert artifact.size_bytes is not None
        mismatched = artifact.model_copy(update={"size_bytes": artifact.size_bytes + 1})
        write_manifest(evaluation.model_copy(update={"artifacts": [mismatched]}), root=tmp_path)
        evaluation_states_cache_path(evaluation.id, root=tmp_path).unlink()

        resolved = resolve_analysis_inputs(
            _analysis_spec(evaluation.id, policy="recompute"),
            root=tmp_path,
        )[0]

        assert resolved.evaluation_state_source is not None
        assert resolved.evaluation_state_source.source_kind == "durable"
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)


@pytest.mark.parametrize("authority_failure", ["missing", "hash", "size", "unsafe_uri"])
def test_authenticated_authority_failures_are_guarded_and_stamped(
    tmp_path: Path,
    authority_failure: str,
) -> None:
    _register_evaluation()

    def unreachable_recipe(*_args):
        raise AssertionError("authority failure must precede recipe execution")

    register_analysis_recipe(ANALYSIS_TYPE, unreachable_recipe, replace=True)
    try:
        evaluation, evaluation_path = _evaluation(tmp_path, durable=True)
        authority = authenticated_manifest_ref(evaluation, evaluation_path, "evaluation_run")
        if authority_failure == "missing":
            evaluation_path.unlink()
        elif authority_failure == "hash":
            authority = authority.model_copy(
                update={
                    "metadata": {
                        **authority.metadata,
                        "manifest_sha256": "0" * 64,
                    }
                }
            )
        elif authority_failure == "size":
            authority = authority.model_copy(
                update={
                    "metadata": {
                        **authority.metadata,
                        "size_bytes": authority.metadata["size_bytes"] + 1,
                    }
                }
            )
        else:
            authority = authority.model_copy(update={"uri": "../unsafe.json"})

        with (
            patch(
                "feedbax.analysis.specs.execute_evaluation_run_spec",
                side_effect=AssertionError("require_durable must not recompute"),
            ) as execute,
            pytest.raises(AnalysisRecipeExecutionError) as excinfo,
        ):
            execute_analysis_run_spec(
                _analysis_spec(
                    evaluation.id,
                    policy="require_durable",
                    ref=authority,
                ),
                root=tmp_path,
            )

        failed = load_manifest(excinfo.value.path)
        diagnostic = failed.evaluation_state_resolution_diagnostics[0]
        assert diagnostic.code == "custody_unavailable"
        assert diagnostic.details["authority_failure_type"] in {
            "FileNotFoundError",
            "ValueError",
        }
        execute.assert_not_called()
    finally:
        unregister_analysis_recipe(ANALYSIS_TYPE)
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_id_only_require_durable_ref_is_guarded_and_stamped(tmp_path: Path) -> None:
    _register_evaluation()
    register_analysis_recipe(
        ANALYSIS_TYPE,
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("ID-only authority must fail before recipe execution")
        ),
        replace=True,
    )
    try:
        evaluation, _ = _evaluation(tmp_path, durable=True)
        with pytest.raises(AnalysisRecipeExecutionError) as excinfo:
            execute_analysis_run_spec(
                _analysis_spec(evaluation.id, policy="require_durable"),
                root=tmp_path,
            )
        failed = load_manifest(excinfo.value.path)
        assert failed.evaluation_state_resolution_diagnostics[0].code == "custody_unavailable"
    finally:
        unregister_analysis_recipe(ANALYSIS_TYPE)
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_failed_manifest_carries_resolution_diagnostic_without_recompute(tmp_path: Path) -> None:
    _register_evaluation()

    def unreachable_recipe(*_args):
        raise AssertionError("resolution failure must precede recipe execution")

    register_analysis_recipe(ANALYSIS_TYPE, unreachable_recipe, replace=True)
    try:
        evaluation, _ = _evaluation(tmp_path)
        with (
            patch(
                "feedbax.analysis.specs.execute_evaluation_run_spec",
                side_effect=AssertionError("require_durable must not recompute"),
            ) as execute,
            pytest.raises(AnalysisRecipeExecutionError) as excinfo,
        ):
            execute_analysis_run_spec(
                _analysis_spec(evaluation.id, policy="require_durable", root=tmp_path),
                root=tmp_path,
            )

        failed = load_manifest(excinfo.value.path)
        assert failed.evaluation_state_resolution_diagnostics[0].code == "missing_durable_states"
        execute.assert_not_called()
    finally:
        unregister_analysis_recipe(ANALYSIS_TYPE)
        unregister_evaluation_recipe(EVALUATION_TYPE)


def test_analysis_run_spec_v1_migrates_and_unknown_old_version_rejects() -> None:
    v1 = {
        "schema_id": "feedbax.spec.analysis_run",
        "schema_version": ANALYSIS_RUN_SPEC_SCHEMA_VERSION_V1,
        "analysis_type": ANALYSIS_TYPE,
        "inputs": [],
        "input_requirements": [],
        "params": {},
    }

    migrated = coerce_analysis_run_spec(v1)
    assert migrated.schema_version == ANALYSIS_RUN_SPEC_SCHEMA_VERSION
    assert migrated.evaluation_states_policy == "recompute"

    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            "AnalysisRunSpec",
            {**v1, "schema_version": "feedbax.spec.analysis_run.v0"},
        )


def test_legacy_state_source_version_is_explicitly_rejected() -> None:
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            "AnalysisEvaluationStateSource",
            {
                "schema_id": "feedbax.manifest.analysis_evaluation_state_source",
                "schema_version": ANALYSIS_EVALUATION_STATE_SOURCE_SCHEMA_VERSION_V1,
                "source_kind": "analysis_time_recompute",
                "requested_evaluation_manifest_id": "evaluation",
                "resulting_evaluation_manifest_id": "evaluation",
            },
            source_version=ANALYSIS_EVALUATION_STATE_SOURCE_SCHEMA_VERSION_V1,
        )
