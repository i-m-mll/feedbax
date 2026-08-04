from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

import feedbax.analysis.execution_context as execution_context_module
from feedbax.analysis.bundles import (
    AnalysisBundleSpec,
    BundleStageSpec,
    dry_run_staged_analysis_bundle,
    execute_staged_analysis_bundle,
)
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    execute_evaluation_run_spec,
)
from feedbax.plugins.bootstrap import BootstrapState
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedArtifactProviderRootBinding,
    StagedCheckpointCustodyRootBinding,
    StagedExecutionContext,
    StagedExecutionContextError,
    StagedManifestRootBinding,
    StagedParentExecutionLocation,
    resolve_staged_execution_context,
    with_staged_manifest_provider_inputs,
    with_staged_parent_execution_locations,
    with_staged_repo_root,
)
from feedbax.analysis.evaluation_inputs import (
    EvaluationInputReferenceError,
    resolve_evaluation_inputs,
)
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
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
from feedbax.contracts.spec_storage import training_run_execution_hash
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


def _store_provider_manifest(root: Path, *, manifest_id: str = "feedbax-training-run:bound"):
    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(),
        explicit_root=root,
    )
    manifest = TrainingRunManifest(id=manifest_id, status="completed")
    raw = manifest.model_dump_json(indent=2).encode()
    artifact = provider.store_bytes(
        raw,
        role="training_manifest",
        logical_name="training.json",
    )
    parent = ParentRef(
        kind="TrainingRunManifest",
        id=manifest.id,
        role="training_run",
        metadata={
            "ref_schema_id": "feedbax.ref.authenticated_manifest",
            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
            "manifest_sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
        },
    )
    return provider, manifest, parent


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


def test_repo_root_binding_is_runtime_only_and_rejects_conflicting_authority(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    other_root = tmp_path / "other"
    repo_root.mkdir()
    other_root.mkdir()

    context = with_staged_repo_root(EMPTY_STAGED_EXECUTION_CONTEXT, repo_root)

    assert context.repo_root == repo_root.resolve()
    assert context.descriptor is None
    assert with_staged_repo_root(context, repo_root) is context
    with pytest.raises(StagedExecutionContextError, match="repo_root disagrees"):
        with_staged_repo_root(context, other_root)


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


def test_manifest_provider_lookup_rejects_unknown_id_before_recipe_effects(
    tmp_path: Path,
) -> None:
    roots, artifact_bindings, checkpoint_bindings = _bindings(tmp_path)
    _provider, _manifest, parent = _store_provider_manifest(roots["primary"])
    context = resolve_staged_execution_context(
        _descriptor(),
        artifact_provider_bindings=artifact_bindings,
        checkpoint_custody_bindings=checkpoint_bindings,
    )
    unknown = parent.model_copy(update={"id": "feedbax-training-run:unknown"})

    with pytest.raises(StagedExecutionContextError, match="kind or id"):
        with_staged_manifest_provider_inputs(context, [unknown])


def test_manifest_provider_lookup_rejects_duplicate_exact_authority(
    tmp_path: Path,
) -> None:
    roots, artifact_bindings, checkpoint_bindings = _bindings(tmp_path)
    _provider, manifest, parent = _store_provider_manifest(roots["primary"])
    second = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(),
        explicit_root=roots["evidence.backup"],
    )
    second.store_bytes(
        manifest.model_dump_json(indent=2).encode(),
        role="training_manifest",
        logical_name="training.json",
    )
    context = resolve_staged_execution_context(
        _descriptor(),
        artifact_provider_bindings=artifact_bindings,
        checkpoint_custody_bindings=checkpoint_bindings,
    )

    with pytest.raises(StagedExecutionContextError, match="duplicated across"):
        with_staged_manifest_provider_inputs(context, [parent])


@pytest.mark.parametrize("mismatch", ["size", "sha256"])
def test_manifest_provider_lookup_rejects_authenticated_byte_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    roots, artifact_bindings, checkpoint_bindings = _bindings(tmp_path)
    provider, _manifest, parent = _store_provider_manifest(roots["primary"])
    if mismatch == "size":
        parent = parent.model_copy(
            update={
                "metadata": {
                    **parent.metadata,
                    "size_bytes": int(parent.metadata["size_bytes"]) + 1,
                }
            }
        )
    else:
        artifact_id = f"artifact://sha256/{parent.metadata['manifest_sha256']}"
        path = roots["primary"] / provider.canonical_relative_path(
            artifact_id,
            size_bytes=int(parent.metadata["size_bytes"]),
        )
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    context = resolve_staged_execution_context(
        _descriptor(),
        artifact_provider_bindings=artifact_bindings,
        checkpoint_custody_bindings=checkpoint_bindings,
    )

    with pytest.raises(ValueError, match=mismatch):
        with_staged_manifest_provider_inputs(context, [parent])


def _retained_manifest_root(
    root: Path,
    *,
    manifest_id: str = "feedbax-training-run:retained",
) -> tuple[TrainingRunManifest, ParentRef, Path]:
    manifest = TrainingRunManifest(id=manifest_id, status="completed")
    raw = manifest.model_dump_json(indent=2).encode()
    path = root / "manifests" / "training_runs" / f"{manifest_id.replace(':', '_')}.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    parent = ParentRef(
        kind="TrainingRunManifest",
        id=manifest.id,
        role="training_run",
        metadata={
            "ref_schema_id": "feedbax.ref.authenticated_manifest",
            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
            "manifest_sha256": sha256_bytes(raw),
            "size_bytes": len(raw),
        },
    )
    return manifest, parent, path


def _retained_manifest_context(
    roots: list[Path],
) -> StagedExecutionContext:
    descriptor = StagedExecutionDescriptor(
        schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        artifact_providers={},
        checkpoint_custody={},
    )
    return resolve_staged_execution_context(
        descriptor,
        manifest_root_bindings=[
            StagedManifestRootBinding(f"retained-{index}", root) for index, root in enumerate(roots)
        ],
    )


def test_manifest_root_binding_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(StagedExecutionContextError, match="root is unavailable"):
        _retained_manifest_context([tmp_path / "missing"])


def test_manifest_root_lookup_resolves_an_authenticated_training_parent(tmp_path: Path) -> None:
    root = tmp_path / "retained"
    manifest, parent, path = _retained_manifest_root(root)

    bound = with_staged_manifest_provider_inputs(_retained_manifest_context([root]), [parent])

    location = bound.parent_execution_location(parent)
    assert location.root == root
    assert location.artifact_provider is None
    assert location.execution_uri == str(path.relative_to(root))

    resolved = bound.resolve_manifest_input(parent)
    assert isinstance(resolved.manifest, TrainingRunManifest)
    assert resolved.manifest.kind == "TrainingRunManifest"
    assert resolved.manifest.id == manifest.id
    assert resolved.path == path
    assert resolved.raw_bytes == path.read_bytes()


def test_manifest_root_lookup_rejects_unknown_id(tmp_path: Path) -> None:
    root = tmp_path / "retained"
    root.mkdir()
    with pytest.raises(StagedExecutionContextError, match="unavailable"):
        with_staged_manifest_provider_inputs(
            _retained_manifest_context([root]),
            [
                ParentRef(
                    kind="TrainingRunManifest",
                    id="feedbax-training-run:unknown",
                    role="training_run",
                    metadata={
                        "ref_schema_id": "feedbax.ref.authenticated_manifest",
                        "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
                        "manifest_sha256": "a" * 64,
                        "size_bytes": 1,
                    },
                )
            ],
        )


def test_manifest_root_lookup_rejects_duplicate_id(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _manifest, parent, _path = _retained_manifest_root(first)
    _retained_manifest_root(second)

    with pytest.raises(StagedExecutionContextError, match="duplicated across authorities"):
        with_staged_manifest_provider_inputs(
            _retained_manifest_context([first, second]),
            [parent],
        )


@pytest.mark.parametrize("mismatch", ["size", "sha256"])
def test_manifest_root_lookup_rejects_authenticated_byte_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    root = tmp_path / "retained"
    _manifest, parent, path = _retained_manifest_root(root)
    if mismatch == "size":
        parent = parent.model_copy(
            update={
                "metadata": {
                    **parent.metadata,
                    "size_bytes": int(parent.metadata["size_bytes"]) + 1,
                }
            }
        )
    else:
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])

    with pytest.raises(StagedExecutionContextError, match=mismatch):
        with_staged_manifest_provider_inputs(
            _retained_manifest_context([root]),
            [parent],
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


@pytest.mark.parametrize(
    "uri",
    [
        "transactions//manifest.json",
        "transactions/./manifest.json",
        "transactions/%2e/manifest.json",
        "transactions/%2e%2e/manifest.json",
        "transactions/%2fmanifest.json",
        "transactions/%00/manifest.json",
        "transactions/%5cmanifest.json",
    ],
)
def test_checkpoint_binding_rejects_malformed_relative_uri_before_resolution(
    tmp_path: Path,
    uri: str,
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
        uri=uri,
        metadata={
            "checkpoint_custody_binding": "training-checkpoints",
            "manifest_sha256": "a" * 64,
        },
    )

    with pytest.raises(StagedExecutionContextError, match="ParentRef uri|escapes"):
        context.resolve_checkpoint_custody_ref(ref)


def test_real_checkpoint_resolution_uses_pinned_authority_and_preserves_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    original_resolve = execution_context_module.resolve_bound_checkpoint_custody_ref
    resolutions = 0

    def counted_resolve(*args, **kwargs):
        nonlocal resolutions
        resolutions += 1
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(
        execution_context_module,
        "resolve_bound_checkpoint_custody_ref",
        counted_resolve,
    )

    explicit = context.resolve_checkpoint_custody_ref(
        parent,
        binding_name="training-checkpoints",
        slot_names=["controller"],
    )
    repeated = context.resolve_checkpoint_custody_ref(
        parent,
        binding_name="training-checkpoints",
        slot_names=["controller"],
    )
    from_metadata = context.resolve_checkpoint_custody_ref(parent, slot_names=["rng"])
    assert repeated is explicit
    assert resolutions == 2
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


def test_cached_checkpoint_snapshot_recursively_rejects_mutation(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoint-mutability"
    run_spec = _run_spec(minimax=True)
    slots = _minimax_slots()
    slots["controller"] = {
        "array": np.asarray([1, 2], dtype=np.int32),
        "metadata": {"tag": "original", "labels": ["verified"]},
    }
    result = write_checkpoint_transaction(
        checkpoint_root,
        run_spec=run_spec,
        phase_program=run_spec.worker_execution.method_contract.phase_program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=slots,
    )
    parent = ParentRef(
        kind="TrainingCheckpointTransactionManifest",
        id=result.manifest.transaction_id,
        role="training_checkpoint_custody",
        uri=result.manifest_path.relative_to(checkpoint_root).as_posix(),
        metadata={"manifest_sha256": sha256_bytes(result.manifest_path.read_bytes())},
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

    first = context.resolve_checkpoint_custody_ref(
        parent,
        binding_name="training-checkpoints",
        slot_names=["controller"],
    )
    controller = first.slots["controller"]
    with pytest.raises(ValueError, match="read-only"):
        controller["array"][0] = -1
    with pytest.raises(TypeError, match="immutable"):
        controller["metadata"]["tag"] = "mutated"
    with pytest.raises(TypeError, match="immutable"):
        controller["metadata"]["labels"].append("mutated")
    with pytest.raises(TypeError, match="immutable"):
        first.slots["controller"] = {"array": np.asarray([-1])}

    second = context.resolve_checkpoint_custody_ref(
        parent,
        binding_name="training-checkpoints",
        slot_names=["controller"],
    )
    assert second is first
    np.testing.assert_array_equal(second.slots["controller"]["array"], [1, 2])
    assert second.slots["controller"]["metadata"]["tag"] == "original"


class _StatProxy:
    def __init__(self, wrapped, **overrides):
        self._wrapped = wrapped
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._wrapped, name)


def test_retained_local_reader_rejects_ctime_only_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest.json"
    data = b'{"probe":"ctime"}'
    path.write_bytes(data)
    original_fstat = os.fstat
    regular_calls = 0

    def changed_fstat(fd):
        nonlocal regular_calls
        result = original_fstat(fd)
        if (result.st_mode & 0o170000) == 0o100000:
            regular_calls += 1
            if regular_calls == 2:
                return _StatProxy(result, st_ctime_ns=result.st_ctime_ns + 1)
        return result

    monkeypatch.setattr(os, "fstat", changed_fstat)

    with pytest.raises(StagedExecutionContextError, match="identity changed during read"):
        execution_context_module._read_retained_local_file(
            tmp_path,
            path.name,
            expected_root_identity=execution_context_module._directory_identity(
                tmp_path, kind="test"
            ),
            expected_size=len(data),
            expected_sha256=sha256_bytes(data),
            kind="authenticated manifest",
            require_single_link=True,
        )


def test_retained_local_reader_rejects_final_hard_link_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest.json"
    data = b'{"probe":"links"}'
    path.write_bytes(data)
    original_fstat = os.fstat
    regular_calls = 0

    def changed_fstat(fd):
        nonlocal regular_calls
        result = original_fstat(fd)
        if (result.st_mode & 0o170000) == 0o100000:
            regular_calls += 1
            if regular_calls == 2:
                return _StatProxy(result, st_nlink=result.st_nlink + 1)
        return result

    monkeypatch.setattr(os, "fstat", changed_fstat)

    with pytest.raises(StagedExecutionContextError, match="hard-link count changed"):
        execution_context_module._read_retained_local_file(
            tmp_path,
            path.name,
            expected_root_identity=execution_context_module._directory_identity(
                tmp_path, kind="test"
            ),
            expected_size=len(data),
            expected_sha256=sha256_bytes(data),
            kind="authenticated manifest",
            require_single_link=True,
        )


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


def test_exact_immutable_completed_parent_preserves_valid_execution_hash(
    tmp_path: Path,
) -> None:
    resolved_root = "a" * 64
    execution_hash = training_run_execution_hash(resolved_root, [])
    manifest = TrainingRunManifest(
        id="feedbax-training-run:exact-completed-context",
        status="completed",
        resolved_semantics_root_hash=resolved_root,
        execution_hash=execution_hash,
    )
    raw = manifest.model_dump_json(indent=2).encode()
    relative = Path("exact") / "completed-training.json"
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_bytes(raw)
    parent = authenticated_manifest_ref(manifest, path, "training_run")
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

    resolved = context.resolve_manifest_input(parent)

    assert resolved.manifest.execution_hash == execution_hash
    with pytest.raises(ValueError, match="frozen"):
        resolved.manifest.execution_hash = "b" * 64


def test_staged_python_and_dry_run_receive_validated_context_without_extra_effects(
    tmp_path: Path,
    application_registry_bundle,
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

    application_registry_bundle.evaluation_recipes.register(_EVALUATION_TYPE, recipe)
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
        registries=application_registry_bundle,
    )
    assert execution.stages[0].manifest_refs
    assert len(calls) == 1


def test_invalid_context_precedes_recipe_cache_and_output_effects(
    tmp_path: Path, application_registry_bundle
) -> None:
    _write_training(tmp_path)
    calls = []

    def recipe(_run_spec, _root, _states_path, _execution_context):
        calls.append(True)
        return EvaluationRecipeResult()

    application_registry_bundle.evaluation_recipes.register(_EVALUATION_TYPE, recipe)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*.json")}
    with pytest.raises(StagedExecutionContextError, match="exactly match"):
        execute_staged_analysis_bundle(
            _bundle(),
            root=tmp_path,
            execution_descriptor=_descriptor(),
            registries=application_registry_bundle,
        )
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*.json")}
    assert calls == []
    assert after == before
    assert not (tmp_path / "manifests" / "evaluation_runs").exists()


def test_provider_free_direct_execution_receives_empty_singleton(
    tmp_path: Path, evaluation_registry
) -> None:
    seen = []

    def recipe(_run_spec, _root, _states_path, execution_context):
        seen.append(execution_context)
        return EvaluationRecipeResult()

    evaluation_registry.register(_EVALUATION_TYPE, recipe)
    execute_evaluation_run_spec(
        EvaluationRunSpec(evaluation_type=_EVALUATION_TYPE),
        registry=evaluation_registry,
        root=tmp_path,
    )
    assert seen == [EMPTY_STAGED_EXECUTION_CONTEXT]


def test_cli_threads_named_bindings_into_staged_recipe(
    tmp_path: Path,
    monkeypatch,
    capsys,
    application_registry_bundle,
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

    application_registry_bundle.evaluation_recipes.register(_EVALUATION_TYPE, recipe)
    monkeypatch.setattr(
        "feedbax.bin.analysis.load_analysis_bundle", lambda *_args, **_kwargs: _bundle()
    )

    async def compose_application(**_kwargs):
        return BootstrapState(application_registry_bundle, ())

    monkeypatch.setattr("feedbax.bin.analysis.compose_application", compose_application)
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
