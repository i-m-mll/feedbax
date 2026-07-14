from __future__ import annotations

import json
from pathlib import Path

import pytest

from feedbax.analysis import (
    AnalysisBundleSpec,
    BundlePerInputPrerequisiteBinding,
    BundleStageSpec,
    EvaluationParamsBase,
    StagedExactParentEntry,
    StagedExactParents,
    STAGED_EXACT_PARENTS_SCHEMA_ID,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION,
    authenticated_manifest_ref,
    execute_staged_analysis_bundle,
    register_evaluation_recipe,
    resolve_manifest_input,
    unregister_evaluation_recipe,
    StagedArtifactProviderRootBinding,
)
from feedbax.analysis.evaluation import EvaluationRecipeResult, execute_evaluation_run_spec
from feedbax.analysis.reports import BUNDLE_SUMMARY_REPORT_TYPE
from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    TrainingRunManifest,
    sha256_bytes,
    evaluation_run_manifest_id,
    spec_payload,
    write_manifest,
    safe_manifest_key,
)
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.staged_execution import (
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedExecutionDescriptor,
)
from feedbax.persistence.artifact_custody import open_immutable_artifact_blob_provider
from pydantic import ConfigDict
from feedbax.contracts.selection import ManifestPredicate


pytestmark = [pytest.mark.feedbax_contract, pytest.mark.analysis_recipe_contract]
EVALUATION_TYPE = "feedbax.test.per_input_prerequisite"


def _exact(entries: list[StagedExactParentEntry]) -> StagedExactParents:
    return StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=entries,
    )


def _training_parent(root: Path, index: int) -> StagedExactParentEntry:
    run_id = f"feedbax-training-run:selective-{index}"
    manifest = TrainingRunManifest(
        id=run_id,
        status="completed",
        run_set_id="selective",
        metadata={
            "row_id": f"row-{index}",
            "planned_run_id": run_id,
            "method": "selective",
        },
    )
    raw = manifest.model_dump_json(indent=2).encode()
    relative = Path("inputs") / f"row-{index}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    digest = sha256_bytes(raw)
    return StagedExactParentEntry(
        parent=ParentRef(
            kind="TrainingRunManifest",
            id=run_id,
            role="training_run",
            uri=f"artifact://sha256/{digest}",
            metadata={
                "manifest_sha256": digest,
                "size_bytes": len(raw),
                "run_set_id": "selective",
                "row_id": f"row-{index}",
                "planned_run_id": run_id,
                "manifest_status": "completed",
                "registration_status": "completed",
                "conformance_overall": "pass",
                "certificate_sha256": "c" * 64,
            },
        ),
        execution_uri=relative.as_posix(),
    )


def _prerequisite(root: Path, index: int, *, status: str = "completed") -> ParentRef:
    manifest = EvaluationRunManifest(
        id=f"feedbax-evaluation-run:prerequisite-{index}",
        status=status,
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            EvaluationRunSpec(evaluation_type="feedbax.test.source").model_dump(mode="json"),
        ),
    )
    path = write_manifest(manifest, root=root)
    return authenticated_manifest_ref(manifest, path, "evaluation_run")


def _bundle(entries: list[StagedExactParentEntry], refs: list[ParentRef]) -> AnalysisBundleSpec:
    return AnalysisBundleSpec(
        name="selective-prerequisites",
        predicate=ManifestPredicate(
            manifest_kind="TrainingRunManifest",
            run_ids=[entry.parent.id for entry in entries],
            metadata_equals={"method": "selective"},
        ),
        stages=[
            BundleStageSpec(
                name="evaluate",
                kind="evaluation",
                mode="per-run",
                evaluation_type=EVALUATION_TYPE,
                prerequisite_bindings=[
                    BundlePerInputPrerequisiteBinding(
                        input_id=entries[index].parent.id,
                        bind_as="trained_baseline",
                        parent=refs[index],
                    )
                    for index in range(2)
                ],
            ),
            BundleStageSpec(
                name="group",
                kind="report",
                depends_on=["evaluate"],
                report_type=BUNDLE_SUMMARY_REPORT_TYPE,
            ),
        ],
    )


def test_two_of_six_prerequisites_are_selective_and_preserve_topology(tmp_path: Path) -> None:
    entries = [_training_parent(tmp_path, index) for index in range(6)]
    refs = [_prerequisite(tmp_path, index) for index in range(2)]
    calls: list[EvaluationRunSpec] = []

    def recipe(run_spec, root, _states_path, execution_context):
        calls.append(run_spec)
        parsed = EvaluationParamsBase.model_validate(run_spec.params)
        for prerequisite in (parsed.staged_prerequisites or {}).values():
            resolved = execution_context.resolve_manifest_input(prerequisite.parent)
            assert resolved.manifest.status == "completed"
        return EvaluationRecipeResult(summary_metrics={"ok": 1})

    register_evaluation_recipe(EVALUATION_TYPE, recipe, replace=True)
    try:
        execution = execute_staged_analysis_bundle(
            _bundle(entries, refs),
            root=tmp_path,
            exact_parents=_exact(entries),
        )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)

    assert len(calls) == 6
    assert all(len(call.inputs) == 1 for call in calls)
    assert ["staged_prerequisites" in call.params for call in calls] == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    evaluation_refs = execution.stages[0].manifest_refs
    assert len(evaluation_refs) == 6
    assert execution.stages[1].inputs == evaluation_refs
    assert not {ref.id for ref in refs}.intersection(ref.id for ref in execution.stages[1].inputs)


def test_provider_backed_prerequisite_round_trips_through_provider_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [_training_parent(tmp_path, index) for index in range(2)]
    refs = [_prerequisite(tmp_path, index) for index in range(2)]
    provider_root = tmp_path / "provider"
    provider_root.mkdir()
    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(), explicit_root=provider_root
    )
    for ref in refs:
        source = resolve_manifest_input(ref, tmp_path)
        provider.store_bytes(
            source.raw_bytes,
            role="manifest",
            logical_name=ref.id,
        )
    bundle = _bundle(entries, refs)
    for binding in bundle.stages[0].prerequisite_bindings:
        binding.artifact_provider = "external"
    seen: list[str] = []
    contexts = []

    def recipe(run_spec, _root, _states_path, execution_context):
        contexts.append(execution_context)
        parsed = EvaluationParamsBase.model_validate(run_spec.params)
        for prerequisite in (parsed.staged_prerequisites or {}).values():
            seen.append(execution_context.resolve_manifest_input(prerequisite.parent).manifest.id)
        return EvaluationRecipeResult()

    register_evaluation_recipe(EVALUATION_TYPE, recipe, replace=True)
    try:
        execute_staged_analysis_bundle(
            bundle,
            root=tmp_path,
            exact_parents=_exact(entries),
            execution_descriptor=StagedExecutionDescriptor(
                schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
                schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
                artifact_providers={"external": ImmutableArtifactBlobProviderSpec()},
                checkpoint_custody={},
            ),
            artifact_provider_bindings=[
                StagedArtifactProviderRootBinding("external", provider_root)
            ],
        )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)
    assert seen == [ref.id for ref in refs]
    digest = refs[0].metadata["manifest_sha256"]
    blob_path = provider_root / "artifacts" / "sha256" / digest[:2] / digest
    original_bytes = blob_path.read_bytes()
    blob_path.write_bytes(b"x" * len(original_bytes))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        contexts[0].resolve_manifest_input(refs[0])
    blob_path.write_bytes(original_bytes)
    original_get_bytes = type(provider).get_bytes
    original_root = provider_root.with_name("provider-original")

    def replacing_get_bytes(provider_instance, artifact, *, size_bytes=None):
        raw_bytes = original_get_bytes(
            provider_instance,
            artifact,
            size_bytes=size_bytes,
        )
        provider_root.rename(original_root)
        provider_root.mkdir()
        return raw_bytes

    monkeypatch.setattr(type(provider), "get_bytes", replacing_get_bytes)
    with pytest.raises(ValueError, match="replaced after binding"):
        contexts[0].resolve_manifest_input(refs[0])


def test_public_params_base_composes_required_recipe_fields() -> None:
    class RecipeParams(EvaluationParamsBase):
        model_config = ConfigDict(extra="forbid")
        metric: str

    parsed = RecipeParams.model_validate({"metric": "loss"})
    assert parsed.metric == "loss"
    assert parsed.staged_prerequisites is None


def test_direct_execution_accepts_serialized_params_base_without_prerequisites(
    tmp_path: Path,
) -> None:
    parent = _training_parent(tmp_path, 0).parent
    params = EvaluationParamsBase().model_dump(mode="json")
    spec = EvaluationRunSpec(
        evaluation_type=EVALUATION_TYPE,
        inputs=[parent],
        params=params,
    )
    expected_id = evaluation_run_manifest_id(spec)

    def recipe(run_spec, _root, _states_path, _execution_context):
        assert run_spec.params == {"staged_prerequisites": None}
        return EvaluationRecipeResult()

    register_evaluation_recipe(EVALUATION_TYPE, recipe, replace=True)
    try:
        manifest, _path = execute_evaluation_run_spec(spec, root=tmp_path)
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)

    assert params == {"staged_prerequisites": None}
    assert spec.params == params
    assert manifest.id == expected_id
    assert manifest.provenance.parents == [parent]
    assert manifest.evaluation_spec.inline["params"] == params


@pytest.mark.parametrize("malformed", [[], ""])
def test_direct_execution_rejects_falsey_non_mapping_prerequisites(
    tmp_path: Path,
    malformed: object,
) -> None:
    spec = EvaluationRunSpec(
        evaluation_type=EVALUATION_TYPE,
        params={"staged_prerequisites": malformed},
    )
    calls = 0

    def recipe(_run_spec, _root, _states_path, _execution_context):
        nonlocal calls
        calls += 1
        return EvaluationRecipeResult()

    register_evaluation_recipe(EVALUATION_TYPE, recipe, replace=True)
    try:
        with pytest.raises(TypeError, match="staged_prerequisites must be a mapping or null"):
            execute_evaluation_run_spec(spec, root=tmp_path)
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)
    assert calls == 0


def test_cli_dry_run_round_trips_selective_prerequisites_and_expands_24_eval_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    entries = [_training_parent(tmp_path, index) for index in range(6)]
    refs = [_prerequisite(tmp_path, index) for index in range(2)]
    stages = [
        BundleStageSpec(
            name=f"evaluate-{stage_index}",
            kind="evaluation",
            mode="per-run",
            evaluation_type=EVALUATION_TYPE,
            prerequisite_bindings=[
                BundlePerInputPrerequisiteBinding(
                    input_id=entries[index].parent.id,
                    bind_as="augmented_reference",
                    parent=refs[index],
                )
                for index in range(2)
            ],
        )
        for stage_index in range(4)
    ]
    bundle = AnalysisBundleSpec(
        name="four-by-six-selective-prerequisites",
        predicate=ManifestPredicate(manifest_kind="TrainingRunManifest"),
        stages=stages,
    )
    round_tripped = AnalysisBundleSpec.model_validate_json(bundle.model_dump_json())
    exact_parents_path = tmp_path / "exact-parents.json"
    exact_parents_path.write_text(_exact(entries).model_dump_json(indent=2), encoding="utf-8")

    from feedbax.bin import analysis as analysis_cli

    monkeypatch.setattr(analysis_cli, "load_analysis_bundle", lambda *_args, **_kwargs: round_tripped)
    analysis_cli.main(
        [
            "--bundle",
            "test/selective",
            "--manifest-root",
            str(tmp_path),
            "--exact-parents",
            str(exact_parents_path),
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["matched_run_ids"] == [entry.parent.id for entry in entries]
    assert len(payload["stages"]) == 4
    assert all(stage["status"] == "would_run" for stage in payload["stages"])
    assert sum(len(stage["inputs"]) for stage in payload["stages"]) == 24
    assert [binding.input_id for binding in round_tripped.stages[0].prerequisite_bindings] == [
        entries[0].parent.id,
        entries[1].parent.id,
    ]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"ref_schema_id": "unsupported"}, "schema_id"),
        ({"ref_schema_version": "unsupported"}, "schema_version"),
        ({"drop": "size_bytes"}, "incomplete"),
        ({"kind": "TrainingRunManifest"}, "kind"),
        ({"role": "analysis_run"}, "role"),
        ({"id": "wrong"}, "missing|Missing|id mismatch"),
    ],
)
def test_prerequisite_preflight_rejects_profile_and_identity_drift(
    tmp_path: Path,
    change: dict[str, str],
    message: str,
) -> None:
    entries = [_training_parent(tmp_path, index) for index in range(2)]
    refs = [_prerequisite(tmp_path, index) for index in range(2)]
    ref = refs[0]
    if "drop" in change:
        ref.metadata.pop(change["drop"])
    elif "kind" in change:
        ref.kind = change["kind"]
    elif "role" in change:
        ref.role = change["role"]
    elif "id" in change:
        ref.id = change["id"]
    else:
        ref.metadata.update(change)
    with pytest.raises(ValueError, match=message):
        execute_staged_analysis_bundle(_bundle(entries, refs), root=tmp_path, exact_parents=_exact(entries))


def test_reserved_params_collision_fails_before_recipe(tmp_path: Path) -> None:
    entries = [_training_parent(tmp_path, index) for index in range(2)]
    refs = [_prerequisite(tmp_path, index) for index in range(2)]
    bundle = _bundle(entries, refs)
    bundle.stages[0].local_params = {"staged_prerequisites": {}}
    with pytest.raises(ValueError, match="reserved staged_prerequisites"):
        execute_staged_analysis_bundle(bundle, root=tmp_path, exact_parents=_exact(entries))


def test_absent_bindings_preserve_legacy_evaluation_identity(tmp_path: Path) -> None:
    entry = _training_parent(tmp_path, 0)
    stage = BundleStageSpec(
        name="legacy",
        kind="evaluation",
        mode="per-run",
        evaluation_type=EVALUATION_TYPE,
        local_params={"metric": "loss"},
    )
    bundle = AnalysisBundleSpec(
        name="legacy",
        predicate=ManifestPredicate(manifest_kind="TrainingRunManifest"),
        stages=[stage],
    )
    expected = EvaluationRunSpec(
        evaluation_type=EVALUATION_TYPE,
        inputs=[entry.parent],
        params={"metric": "loss"},
    )
    assert "staged_prerequisites" not in stage.local_params
    assert bundle.model_dump(mode="json")["stages"][0]["prerequisite_bindings"] == []
    assert expected.params == {"metric": "loss"}
    calls: list[EvaluationRunSpec] = []

    def recipe(run_spec, _root, _states_path, _execution_context):
        calls.append(run_spec)
        return EvaluationRecipeResult()

    register_evaluation_recipe(EVALUATION_TYPE, recipe, replace=True)
    try:
        execute_staged_analysis_bundle(bundle, root=tmp_path, exact_parents=_exact([entry]))
        execute_staged_analysis_bundle(bundle, root=tmp_path, exact_parents=_exact([entry]))
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)
    assert calls == [expected]


def test_prerequisite_preflight_rejects_unknown_input_before_recipe(tmp_path: Path) -> None:
    entries = [_training_parent(tmp_path, index) for index in range(2)]
    ref = _prerequisite(tmp_path, 0)
    bundle = _bundle(entries, [ref, ref])
    bundle.stages[0].prerequisite_bindings[0].input_id = "unknown"
    calls: list[EvaluationRunSpec] = []

    def recipe(run_spec, _root, _states_path, _execution_context):
        calls.append(run_spec)
        return EvaluationRecipeResult()

    register_evaluation_recipe(EVALUATION_TYPE, recipe, replace=True)
    try:
        with pytest.raises(ValueError, match="not a selected bundle input"):
            execute_staged_analysis_bundle(
                bundle,
                root=tmp_path,
                    exact_parents=_exact(entries),
            )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)
    assert calls == []


@pytest.mark.parametrize("mutation, message", [
    ("hash", "SHA-256 mismatch"),
    ("size", "size mismatch"),
    ("status", "must be completed"),
])
def test_prerequisite_preflight_rejects_tamper(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    entries = [_training_parent(tmp_path, 0), _training_parent(tmp_path, 1)]
    ref = _prerequisite(tmp_path, 0, status="failed" if mutation == "status" else "completed")
    if mutation == "hash":
        ref.metadata["manifest_sha256"] = "0" * 64
    elif mutation == "size":
        ref.metadata["size_bytes"] += 1
    bundle = _bundle(entries, [ref, _prerequisite(tmp_path, 1)])
    with pytest.raises(ValueError, match=message):
        execute_staged_analysis_bundle(
            bundle,
            root=tmp_path,
            exact_parents=_exact(entries),
        )


def test_prerequisite_contract_rejects_duplicate_and_unsupported_stage(tmp_path: Path) -> None:
    entries = [_training_parent(tmp_path, index) for index in range(2)]
    ref = _prerequisite(tmp_path, 0)
    binding = BundlePerInputPrerequisiteBinding(
        input_id=entries[0].parent.id,
        bind_as="baseline",
        parent=ref,
    )
    with pytest.raises(ValueError, match="duplicate prerequisite input_id"):
        BundleStageSpec(
            name="evaluate",
            kind="evaluation",
            mode="per-run",
            evaluation_type=EVALUATION_TYPE,
            prerequisite_bindings=[binding, binding],
        )
    with pytest.raises(ValueError, match="root per-run evaluation"):
        BundleStageSpec(
            name="grouped",
            kind="evaluation",
            mode="grouped",
            evaluation_type=EVALUATION_TYPE,
            prerequisite_bindings=[binding],
        )


def test_prerequisite_preflight_requires_declared_provider_binding(tmp_path: Path) -> None:
    entries = [_training_parent(tmp_path, index) for index in range(2)]
    refs = [_prerequisite(tmp_path, index) for index in range(2)]
    bundle = _bundle(entries, refs)
    bundle.stages[0].prerequisite_bindings[0].artifact_provider = "missing"
    with pytest.raises(ValueError, match="artifact provider binding is unavailable"):
        execute_staged_analysis_bundle(bundle, root=tmp_path, exact_parents=_exact(entries))


def test_completed_cache_rejects_prerequisite_provenance_drift(tmp_path: Path) -> None:
    entries = [_training_parent(tmp_path, index) for index in range(2)]
    refs = [_prerequisite(tmp_path, index) for index in range(2)]

    def recipe(_run_spec, _root, _states_path, _execution_context):
        return EvaluationRecipeResult()

    register_evaluation_recipe(EVALUATION_TYPE, recipe, replace=True)
    try:
        execution = execute_staged_analysis_bundle(
            _bundle(entries, refs), root=tmp_path, exact_parents=_exact(entries)
        )
        first = execution.stages[0].manifest_refs[0]
        path = (
            tmp_path
            / "manifests"
            / "evaluation_runs"
            / f"{safe_manifest_key(first.id)}.json"
        )
        manifest = EvaluationRunManifest.model_validate_json(path.read_text())
        manifest.provenance.parents = list(manifest.input_training_runs)
        path.write_text(manifest.model_dump_json(indent=2))
        with pytest.raises(ValueError, match="provenance parents"):
            execute_staged_analysis_bundle(
                _bundle(entries, refs), root=tmp_path, exact_parents=_exact(entries)
            )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)
