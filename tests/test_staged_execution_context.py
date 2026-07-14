from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from feedbax.analysis.bundles import (
    AnalysisBundleSpec,
    BundleStageSpec,
    dry_run_staged_analysis_bundle,
    execute_staged_analysis_bundle,
)
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    execute_evaluation_run_spec,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.analysis.evaluation_inputs import (
    EvaluationInputReferenceError,
    resolve_evaluation_inputs,
)
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedArtifactProviderRootBinding,
    StagedCheckpointCustodyRootBinding,
    StagedExecutionContext,
    StagedExecutionContextError,
    StagedParentExecutionLocation,
    resolve_staged_execution_context,
    with_staged_parent_execution_locations,
)
from feedbax.bin.analysis import main as analysis_main
from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.manifest import (
    EvaluationRunSpec,
    ParentRef,
    TrainingRunManifest,
    sha256_bytes,
    write_manifest,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.selection import ManifestPredicate
from feedbax.contracts.staged_execution import (
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedCheckpointCustodySpec,
    StagedExecutionDescriptor,
)
from feedbax.persistence.artifact_custody import open_immutable_artifact_blob_provider
from feedbax.training.checkpoint_custody import write_checkpoint_transaction
from tests.test_checkpoint_custody import _coordinate, _minimax_slots, _run_spec
from tests.test_staged_exact_parents import _exact_document, _write_exact_parent


pytestmark = [pytest.mark.feedbax_contract, pytest.mark.analysis_recipe_contract]

_EVALUATION_TYPE = "feedbax.test.staged_execution_context"


def _descriptor() -> StagedExecutionDescriptor:
    return StagedExecutionDescriptor(
        schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        artifact_providers={
            "primary": ImmutableArtifactBlobProviderSpec(),
            "evidence.backup": ImmutableArtifactBlobProviderSpec(),
        },
        checkpoint_custody={
            "training-checkpoints": StagedCheckpointCustodySpec(
                backend="feedbax-checkpoint-transaction-tree"
            )
        },
    )


def _bindings(tmp_path: Path):
    roots = {
        "primary": tmp_path / "provider-primary",
        "evidence.backup": tmp_path / "provider-backup",
        "training-checkpoints": tmp_path / "checkpoints",
    }
    for root in roots.values():
        root.mkdir(parents=True)
    return (
        roots,
        (
            StagedArtifactProviderRootBinding("primary", roots["primary"]),
            StagedArtifactProviderRootBinding("evidence.backup", roots["evidence.backup"]),
        ),
        (
            StagedCheckpointCustodyRootBinding(
                "training-checkpoints", roots["training-checkpoints"]
            ),
        ),
    )


def _bundle() -> AnalysisBundleSpec:
    return AnalysisBundleSpec(
        name="staged_execution_context",
        predicate=ManifestPredicate(
            manifest_kind="TrainingRunManifest",
            metadata_equals={"method": "minimax"},
        ),
        stages=[
            BundleStageSpec(
                name="evaluate",
                kind="evaluation",
                mode="per-run",
                evaluation_type=_EVALUATION_TYPE,
            )
        ],
    )


@pytest.mark.parametrize(
    "root_identities",
    [{"other": (1, 2)}, {"primary": (1, 2), "extra": (3, 4)}],
)
def test_artifact_provider_root_identity_keys_must_match_opened_providers(
    tmp_path: Path,
    root_identities: dict[str, tuple[int, int]],
) -> None:
    provider_root = tmp_path / "provider"
    provider_root.mkdir()
    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(), explicit_root=provider_root
    )
    with pytest.raises(
        StagedExecutionContextError,
        match="identities must exactly match opened providers",
    ):
        StagedExecutionContext(
            descriptor=None,
            opened_artifact_providers={"primary": provider},
            checkpoint_custody_roots={},
            parent_execution_locations=(),
            _artifact_provider_root_identities=root_identities,
        )


def _write_training(root: Path, run_id: str = "feedbax-training-run:context") -> None:
    write_manifest(
        TrainingRunManifest(
            id=run_id,
            status="completed",
            metadata={"method": "minimax"},
        ),
        root=root,
    )


def test_resolver_binds_two_providers_and_checkpoint_without_portable_roots(
    tmp_path: Path,
) -> None:
    roots, artifact_bindings, checkpoint_bindings = _bindings(tmp_path)
    context = resolve_staged_execution_context(
        _descriptor(),
        artifact_provider_bindings=artifact_bindings,
        checkpoint_custody_bindings=checkpoint_bindings,
    )

    assert context.artifact_provider("primary").root == roots["primary"]
    assert context.artifact_provider("evidence.backup").root == roots["evidence.backup"]
    assert context.checkpoint_custody_root("training-checkpoints") == roots["training-checkpoints"]
    portable = context.descriptor.model_dump_json() if context.descriptor else ""
    assert str(tmp_path) not in portable

    transient = tmp_path / "transient-source.bin"
    transient.write_bytes(b"durable-provider-bytes")
    artifact = context.artifact_provider("primary").store_bytes(
        transient.read_bytes(),
        role="evidence",
        logical_name="evidence.bin",
    )
    transient.unlink()
    assert context.artifact_provider("primary").get_bytes(artifact) == b"durable-provider-bytes"


def test_descriptor_is_discoverable_with_explicit_reject_migration_policy() -> None:
    from feedbax.integrations.provider import provider_manifest

    family = default_spec_registry.resolve("StagedExecutionDescriptor")
    nested = default_spec_registry.resolve("StagedCheckpointCustodySpec")
    manifest = provider_manifest()

    assert family.identity == STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID
    assert family.current_version == STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION
    assert family.policy is not None and family.policy.stance == "reject"
    assert nested.policy is not None and nested.policy.covers == family.kind
    assert manifest.schemas["StagedExecutionDescriptor"] == (
        StagedExecutionDescriptor.model_json_schema()
    )
    assert manifest.capabilities["resolve_staged_execution_context"].input_schema == (
        "StagedExecutionDescriptor"
    )
    with pytest.raises(UnsupportedSpecVersion, match="current_version"):
        default_spec_registry.migrate(
            family.kind,
            {"schema_version": "feedbax.spec.staged_execution.v0"},
        )


@pytest.mark.parametrize(
    ("descriptor_update", "artifact_names", "checkpoint_names", "message"),
    [
        ({"schema_version": "feedbax.spec.staged_execution.v0"}, (), (), "schema_version"),
        ({"schema_version": "feedbax.spec.staged_execution.v2"}, (), (), "schema_version"),
        ({"schema_id": "example.spec.staged_execution"}, (), (), "schema_id"),
        ({"checkpoint_custody": {"training-checkpoints": {"backend": "wrong"}}}, (), (), "invalid"),
        ({}, ("primary",), ("training-checkpoints",), "missing"),
        ({}, ("primary", "evidence.backup", "extra"), ("training-checkpoints",), "extra"),
        ({}, ("primary", "primary"), ("training-checkpoints",), "duplicate"),
    ],
)
def test_descriptor_and_binding_name_sets_fail_closed(
    tmp_path: Path,
    descriptor_update: dict[str, object],
    artifact_names: tuple[str, ...],
    checkpoint_names: tuple[str, ...],
    message: str,
) -> None:
    roots, _artifact_bindings, _checkpoint_bindings = _bindings(tmp_path)
    payload = _descriptor().model_dump(mode="json")
    payload.update(descriptor_update)
    artifacts = tuple(
        StagedArtifactProviderRootBinding(name, roots.get(name, tmp_path / name))
        for name in artifact_names
    )
    checkpoints = tuple(
        StagedCheckpointCustodyRootBinding(name, roots[name]) for name in checkpoint_names
    )
    with pytest.raises((StagedExecutionContextError, ValueError), match=message):
        resolve_staged_execution_context(
            payload,
            artifact_provider_bindings=artifacts,
            checkpoint_custody_bindings=checkpoints,
        )


def test_constructed_or_mutated_descriptor_instances_are_revalidated_before_provider_open(
    tmp_path: Path,
    monkeypatch,
) -> None:
    opened = []
    monkeypatch.setattr(
        "feedbax.analysis.execution_context.open_immutable_artifact_blob_provider",
        lambda *_args, **_kwargs: opened.append(True),
    )
    mutated = _descriptor()
    object.__setattr__(mutated, "schema_version", "feedbax.spec.staged_execution.v0")
    bad_id = StagedExecutionDescriptor.model_construct(
        schema_id="example.spec.staged_execution",
        schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        artifact_providers={},
        checkpoint_custody={},
    )
    bad_name = StagedExecutionDescriptor.model_construct(
        schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        artifact_providers={"Bad Name": ImmutableArtifactBlobProviderSpec()},
        checkpoint_custody={},
    )
    bad_backend = StagedExecutionDescriptor.model_construct(
        schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        artifact_providers={},
        checkpoint_custody={
            "training-checkpoints": StagedCheckpointCustodySpec.model_construct(backend="wrong")
        },
    )

    for descriptor in (mutated, bad_id, bad_name, bad_backend):
        with pytest.raises(StagedExecutionContextError):
            resolve_staged_execution_context(descriptor)

    assert opened == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("case", ["relative", "missing", "file", "symlink", "escape"])
def test_runtime_roots_reject_unsafe_locations(tmp_path: Path, case: str) -> None:
    roots, artifact_bindings, checkpoint_bindings = _bindings(tmp_path)
    bad_root: Path | str
    if case == "relative":
        bad_root = "relative/provider"
    elif case == "missing":
        bad_root = tmp_path / "missing"
    elif case == "file":
        bad_root = tmp_path / "provider-file"
        Path(bad_root).write_text("not a directory", encoding="utf-8")
    elif case == "symlink":
        bad_root = tmp_path / "provider-link"
        Path(bad_root).symlink_to(roots["primary"], target_is_directory=True)
    else:
        bad_root = tmp_path / "provider-primary" / ".."

    bindings = (
        StagedArtifactProviderRootBinding("primary", bad_root),
        artifact_bindings[1],
    )
    with pytest.raises(StagedExecutionContextError, match="root"):
        resolve_staged_execution_context(
            _descriptor(),
            artifact_provider_bindings=bindings,
            checkpoint_custody_bindings=checkpoint_bindings,
        )


def test_checkpoint_binding_mismatch_and_malicious_uri_fail_before_resolution(
    tmp_path: Path,
) -> None:
    _roots, artifact_bindings, checkpoint_bindings = _bindings(tmp_path)
    context = resolve_staged_execution_context(
        _descriptor(),
        artifact_provider_bindings=artifact_bindings,
        checkpoint_custody_bindings=checkpoint_bindings,
    )
    ref = ParentRef(
        kind="TrainingCheckpointTransactionManifest",
        id="transaction",
        role="training_checkpoint_custody",
        uri="../outside.json",
        metadata={
            "checkpoint_custody_binding": "training-checkpoints",
            "manifest_sha256": "a" * 64,
        },
    )

    with pytest.raises(StagedExecutionContextError, match="disagrees"):
        context.resolve_checkpoint_custody_ref(ref, binding_name="other")
    with pytest.raises(StagedExecutionContextError, match="escapes"):
        context.resolve_checkpoint_custody_ref(ref)


def test_real_checkpoint_resolution_uses_pinned_authority_and_preserves_reference(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoint-authority"
    run_spec = _run_spec(minimax=True)
    result = write_checkpoint_transaction(
        checkpoint_root,
        run_spec=run_spec,
        phase_program=run_spec.worker_execution.method_contract.phase_program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_minimax_slots(),
    )
    relative_uri = result.manifest_path.relative_to(checkpoint_root).as_posix()
    parent = ParentRef(
        kind="TrainingCheckpointTransactionManifest",
        id=result.manifest.transaction_id,
        role="training_checkpoint_custody",
        uri=relative_uri,
        metadata={
            "checkpoint_custody_binding": "training-checkpoints",
            "manifest_sha256": sha256_bytes(result.manifest_path.read_bytes()),
        },
    )
    descriptor = StagedExecutionDescriptor(
        schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        artifact_providers={},
        checkpoint_custody={
            "training-checkpoints": StagedCheckpointCustodySpec(
                backend="feedbax-checkpoint-transaction-tree"
            )
        },
    )
    context = resolve_staged_execution_context(
        descriptor,
        checkpoint_custody_bindings=[
            StagedCheckpointCustodyRootBinding("training-checkpoints", checkpoint_root)
        ],
    )

    explicit = context.resolve_checkpoint_custody_ref(
        parent,
        binding_name="training-checkpoints",
        slot_names=["controller"],
    )
    from_metadata = context.resolve_checkpoint_custody_ref(parent, slot_names=["rng"])
    assert explicit.parent_ref == parent
    assert explicit.manifest.transaction_id == result.manifest.transaction_id
    assert set(explicit.slots) == {"controller"}
    assert from_metadata.parent_ref == parent
    assert set(from_metadata.slots) == {"rng"}

    with pytest.raises(StagedExecutionContextError, match="disagrees"):
        context.resolve_checkpoint_custody_ref(parent, binding_name="other")
    with pytest.raises(StagedExecutionContextError, match="escapes"):
        context.resolve_checkpoint_custody_ref(parent.model_copy(update={"uri": "../outside.json"}))

    replacement = tmp_path / "replacement-authority"
    shutil.copytree(checkpoint_root, replacement)
    retained = tmp_path / "retained-authority"
    checkpoint_root.rename(retained)
    checkpoint_root.symlink_to(replacement, target_is_directory=True)
    with pytest.raises(StagedExecutionContextError, match="unavailable|replaced"):
        context.resolve_checkpoint_custody_ref(parent, slot_names=["controller"])


def test_exact_immutable_parent_uses_complete_ref_location_and_preserves_ref(
    tmp_path: Path,
) -> None:
    manifest = TrainingRunManifest(
        id="feedbax-training-run:exact-context",
        status="completed",
    )
    raw = manifest.model_dump_json(indent=2).encode()
    relative = Path("exact") / "training.json"
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_bytes(raw)
    digest = sha256_bytes(raw)
    parent = ParentRef(
        kind="TrainingRunManifest",
        id=manifest.id,
        role="training_run",
        uri=f"artifact://sha256/{digest}",
        metadata={"manifest_sha256": digest, "size_bytes": len(raw), "extra": "preserved"},
    )
    spec = EvaluationRunSpec(evaluation_type=_EVALUATION_TYPE, inputs=[parent])

    with pytest.raises(EvaluationInputReferenceError, match="matching complete-ParentRef"):
        resolve_evaluation_inputs(spec, manifest_root=tmp_path)

    context = with_staged_parent_execution_locations(
        EMPTY_STAGED_EXECUTION_CONTEXT,
        [
            StagedParentExecutionLocation(
                parent=parent,
                root=tmp_path,
                execution_uri=relative.as_posix(),
            )
        ],
    )
    resolved = resolve_evaluation_inputs(
        spec,
        manifest_root=tmp_path,
        execution_context=context,
    )
    assert resolved[0].ref == parent
    assert resolved[0].ref.metadata["extra"] == "preserved"
    with pytest.raises(EvaluationInputReferenceError, match="matching complete-ParentRef"):
        resolve_evaluation_inputs(
            spec.model_copy(
                update={
                    "inputs": [
                        parent.model_copy(
                            update={"metadata": {**parent.metadata, "extra": "narrowed"}}
                        )
                    ]
                }
            ),
            manifest_root=tmp_path,
            execution_context=context,
        )


def test_staged_python_and_dry_run_receive_validated_context_without_extra_effects(
    tmp_path: Path,
) -> None:
    roots, artifact_bindings, checkpoint_bindings = _bindings(tmp_path)
    _write_training(tmp_path)
    artifact = (
        resolve_staged_execution_context(
            _descriptor(),
            artifact_provider_bindings=artifact_bindings,
            checkpoint_custody_bindings=checkpoint_bindings,
        )
        .artifact_provider("primary")
        .store_bytes(
            b"recipe-evidence",
            role="evidence",
            logical_name="recipe.bin",
        )
    )
    calls = []

    def recipe(run_spec, _root, _states_path, execution_context):
        calls.append(execution_context)
        assert execution_context.artifact_provider("primary").get_bytes(artifact) == (
            b"recipe-evidence"
        )
        assert (
            execution_context.checkpoint_custody_root("training-checkpoints")
            == roots["training-checkpoints"]
        )
        return EvaluationRecipeResult(summary_metrics={"inputs": len(run_spec.inputs)})

    register_evaluation_recipe(_EVALUATION_TYPE, recipe, replace=True)
    try:
        dry_root = tmp_path / "dry"
        dry_root.mkdir()
        _write_training(dry_root, "feedbax-training-run:dry")
        dry_result = dry_run_staged_analysis_bundle(
            _bundle(),
            root=dry_root,
            execution_descriptor=_descriptor(),
            artifact_provider_bindings=artifact_bindings,
            checkpoint_custody_bindings=checkpoint_bindings,
        )
        assert dry_result.stages[0].status == "would_run"
        assert calls == []
        assert not (dry_root / "manifests" / "evaluation_runs").exists()

        execution = execute_staged_analysis_bundle(
            _bundle(),
            root=tmp_path,
            execution_descriptor=_descriptor(),
            artifact_provider_bindings=artifact_bindings,
            checkpoint_custody_bindings=checkpoint_bindings,
        )
        assert execution.stages[0].manifest_refs
        assert len(calls) == 1
    finally:
        unregister_evaluation_recipe(_EVALUATION_TYPE)


def test_invalid_context_precedes_recipe_cache_and_output_effects(tmp_path: Path) -> None:
    _write_training(tmp_path)
    calls = []

    def recipe(_run_spec, _root, _states_path, _execution_context):
        calls.append(True)
        return EvaluationRecipeResult()

    register_evaluation_recipe(_EVALUATION_TYPE, recipe, replace=True)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*.json")}
    try:
        with pytest.raises(StagedExecutionContextError, match="exactly match"):
            execute_staged_analysis_bundle(
                _bundle(),
                root=tmp_path,
                execution_descriptor=_descriptor(),
            )
    finally:
        unregister_evaluation_recipe(_EVALUATION_TYPE)
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*.json")}
    assert calls == []
    assert after == before
    assert not (tmp_path / "manifests" / "evaluation_runs").exists()


def test_provider_free_direct_execution_receives_empty_singleton(tmp_path: Path) -> None:
    seen = []

    def recipe(_run_spec, _root, _states_path, execution_context):
        seen.append(execution_context)
        return EvaluationRecipeResult()

    register_evaluation_recipe(_EVALUATION_TYPE, recipe, replace=True)
    try:
        execute_evaluation_run_spec(
            EvaluationRunSpec(evaluation_type=_EVALUATION_TYPE),
            root=tmp_path,
        )
    finally:
        unregister_evaluation_recipe(_EVALUATION_TYPE)
    assert seen == [EMPTY_STAGED_EXECUTION_CONTEXT]


def test_cli_threads_named_bindings_into_staged_recipe(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    roots, _artifact_bindings, _checkpoint_bindings = _bindings(tmp_path)
    manifest_root = tmp_path / "cli-manifests"
    manifest_root.mkdir()
    _write_training(manifest_root, "feedbax-training-run:cli")
    descriptor_path = tmp_path / "execution-descriptor.json"
    descriptor_path.write_text(_descriptor().model_dump_json(indent=2), encoding="utf-8")
    seen = []

    def recipe(_run_spec, _root, _states_path, execution_context):
        seen.append(execution_context)
        return EvaluationRecipeResult()

    register_evaluation_recipe(_EVALUATION_TYPE, recipe, replace=True)
    monkeypatch.setattr(
        "feedbax.bin.analysis.load_analysis_bundle", lambda *_args, **_kwargs: _bundle()
    )
    try:
        analysis_main(
            [
                "--bundle",
                "test/staged-context",
                "--manifest-root",
                str(manifest_root),
                "--fig-dump-dir",
                str(tmp_path / "figures"),
                "--execution-descriptor",
                str(descriptor_path),
                "--artifact-provider",
                f"primary={roots['primary']}",
                "--artifact-provider",
                f"evidence.backup={roots['evidence.backup']}",
                "--checkpoint-custody",
                f"training-checkpoints={roots['training-checkpoints']}",
            ]
        )
    finally:
        unregister_evaluation_recipe(_EVALUATION_TYPE)

    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"][0]["manifest_refs"]
    assert len(seen) == 1
    assert seen[0].descriptor == _descriptor()


def test_cli_dry_run_preflights_context_and_exact_parents_without_effects(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    roots, _artifact_bindings, _checkpoint_bindings = _bindings(tmp_path)
    manifest_root = tmp_path / "cli-dry-manifests"
    manifest_root.mkdir()
    entry = _write_exact_parent(
        manifest_root,
        run_id="feedbax-training-run:cli-dry",
        row_id="cli-dry",
    )
    exact_path = tmp_path / "exact-parents.json"
    exact_path.write_text(_exact_document(entry).model_dump_json(indent=2), encoding="utf-8")
    descriptor_path = tmp_path / "execution-descriptor.json"
    descriptor_path.write_text(_descriptor().model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.setattr(
        "feedbax.bin.analysis.load_analysis_bundle", lambda *_args, **_kwargs: _bundle()
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    analysis_main(
        [
            "--bundle",
            "test/staged-context",
            "--dry-run",
            "--manifest-root",
            str(manifest_root),
            "--exact-parents",
            str(exact_path),
            "--execution-descriptor",
            str(descriptor_path),
            "--artifact-provider",
            f"primary={roots['primary']}",
            "--artifact-provider",
            f"evidence.backup={roots['evidence.backup']}",
            "--checkpoint-custody",
            f"training-checkpoints={roots['training-checkpoints']}",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert payload["stages"][0]["status"] == "would_run"
    assert payload["match_preview"]["parent_refs"] == [entry.parent.model_dump(mode="json")]
    assert after == before
    assert not (manifest_root / "manifests" / "evaluation_runs").exists()


def test_cli_dry_run_invalid_context_has_zero_effects(tmp_path: Path, monkeypatch) -> None:
    roots, _artifact_bindings, _checkpoint_bindings = _bindings(tmp_path)
    manifest_root = tmp_path / "cli-invalid-manifests"
    manifest_root.mkdir()
    _write_training(manifest_root, "feedbax-training-run:cli-invalid")
    descriptor_path = tmp_path / "invalid-execution-descriptor.json"
    payload = _descriptor().model_dump(mode="json")
    payload["schema_version"] = "feedbax.spec.staged_execution.v0"
    descriptor_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "feedbax.bin.analysis.load_analysis_bundle", lambda *_args, **_kwargs: _bundle()
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    with pytest.raises(ValueError, match="schema_version"):
        analysis_main(
            [
                "--bundle",
                "test/staged-context",
                "--dry-run",
                "--manifest-root",
                str(manifest_root),
                "--execution-descriptor",
                str(descriptor_path),
                "--artifact-provider",
                f"primary={roots['primary']}",
                "--artifact-provider",
                f"evidence.backup={roots['evidence.backup']}",
                "--checkpoint-custody",
                f"training-checkpoints={roots['training-checkpoints']}",
            ]
        )

    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not (manifest_root / "manifests" / "evaluation_runs").exists()
