from pathlib import Path
import os
import shutil

import numpy as np
import pytest

import feedbax.analysis.execution_context as execution_context_module

from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedExecutionContext,
    StagedExecutionContextError,
    StagedParentExecutionLocation,
    with_staged_parent_execution_locations,
)
from feedbax.analysis.manifest_inputs import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
)
from feedbax.contracts.evaluation_states import (
    evaluation_states_container_bytes,
    store_evaluation_states_artifact,
)
from feedbax.contracts.manifest import (
    EvaluationRunManifest,
    ParentRef,
    SpecPayload,
    sha256_bytes,
    write_manifest,
)
from feedbax.persistence.artifact_custody import (
    ImmutableArtifactBlobProviderSpec,
    open_immutable_artifact_blob_provider,
)


def _manifest(artifact, *, status="completed") -> EvaluationRunManifest:
    return EvaluationRunManifest(
        id="feedbax-evaluation-run:authority-test",
        status=status,
        evaluation_spec=SpecPayload(
            kind="EvaluationRunSpec",
            schema_id="feedbax.spec.evaluation_run",
            schema_version="feedbax.spec.evaluation_run.v1",
            inline={
                "schema_version": "feedbax.spec.evaluation_run.v1",
                "evaluation_type": "test",
                "training_run_ids": [],
                "inputs": [],
                "params": {},
            },
        ),
        artifacts=[artifact],
    )


def _parent(manifest: EvaluationRunManifest, raw: bytes) -> ParentRef:
    return ParentRef(
        kind="EvaluationRunManifest",
        id=manifest.id,
        role="evaluation_run",
        metadata={
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": sha256_bytes(raw),
            "size_bytes": len(raw),
        },
    )


def _local_context(
    manifest: EvaluationRunManifest, root: Path, *, normalize_locators: bool = True
) -> tuple[StagedExecutionContext, ParentRef]:
    manifest = manifest.model_copy(deep=True)
    if normalize_locators:
        manifest.artifacts = [
            artifact.model_copy(update={"uri": artifact.metadata["relative_path"]})
            if isinstance(artifact.metadata.get("relative_path"), str)
            else artifact
            for artifact in manifest.artifacts
        ]
    path = write_manifest(manifest, root=root, index=False)
    parent = _parent(manifest, path.read_bytes())
    return (
        with_staged_parent_execution_locations(
            EMPTY_STAGED_EXECUTION_CONTEXT,
            [
                StagedParentExecutionLocation(
                    parent=parent,
                    root=root,
                    execution_uri=path.relative_to(root).as_posix(),
                )
            ],
        ),
        parent,
    )


def test_load_evaluation_states_uses_retained_local_authority(tmp_path: Path) -> None:
    expected = {"trajectory": np.arange(4, dtype=np.float32)}
    artifact = store_evaluation_states_artifact(expected, root=tmp_path, manifest_id="eval")
    artifact = artifact.model_copy(update={"uri": artifact.metadata["relative_path"]})
    manifest = _manifest(artifact)
    path = write_manifest(manifest, root=tmp_path, index=False)
    raw = path.read_bytes()
    parent = _parent(manifest, raw)
    context = with_staged_parent_execution_locations(
        EMPTY_STAGED_EXECUTION_CONTEXT,
        [
            StagedParentExecutionLocation(
                parent=parent,
                root=tmp_path,
                execution_uri=path.relative_to(tmp_path).as_posix(),
            )
        ],
    )

    states = context.load_evaluation_states(parent)
    np.testing.assert_array_equal(states["trajectory"], expected["trajectory"])


def test_load_evaluation_states_uses_bound_provider_after_source_deletion(
    tmp_path: Path,
) -> None:
    provider_root = tmp_path / "provider"
    provider_root.mkdir()
    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(), explicit_root=provider_root
    )
    source_cache_root = tmp_path / "source-cache"
    source_cache_root.mkdir()
    source_artifact = store_evaluation_states_artifact(
        {"value": np.asarray([3, 5])},
        root=source_cache_root,
        manifest_id="provider-source",
    )
    states_bytes = (source_cache_root / source_artifact.metadata["relative_path"]).read_bytes()
    artifact = provider.store_bytes(
        states_bytes,
        role="evaluation_states",
        logical_name="states.npz",
    )
    manifest = _manifest(artifact)
    source_manifest_path = write_manifest(manifest, root=source_cache_root, index=False)
    raw = source_manifest_path.read_bytes()
    manifest_artifact = provider.store_bytes(
        raw,
        role="evaluation_run",
        logical_name="evaluation.json",
    )
    parent = _parent(manifest, raw)
    context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"external": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=(
            StagedParentExecutionLocation(
                parent=parent,
                root=provider_root,
                execution_uri=provider.canonical_relative_path(manifest_artifact).as_posix(),
                artifact_provider="external",
            ),
        ),
    )

    shutil.rmtree(source_cache_root)
    assert not source_cache_root.exists()

    states = context.load_evaluation_states(parent)
    np.testing.assert_array_equal(states["value"], np.asarray([3, 5]))

    artifact_path = provider_root / provider.canonical_relative_path(artifact)
    original = artifact_path.read_bytes()
    artifact_path.write_bytes(b"x" * len(original))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        context.load_evaluation_states(parent)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"kind": "TrainingRunManifest"}, "EvaluationRunManifest"),
        ({"role": "training_run"}, "evaluation_run"),
    ],
)
def test_load_evaluation_states_rejects_wrong_parent_profile(
    update: dict[str, str], message: str
) -> None:
    parent = ParentRef(kind="EvaluationRunManifest", id="eval", role="evaluation_run")
    with pytest.raises(StagedExecutionContextError, match=message):
        EMPTY_STAGED_EXECUTION_CONTEXT.load_evaluation_states(parent.model_copy(update=update))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("status", "completed"),
        ("missing", "exactly one"),
        ("wrong_role", "exactly one"),
        ("multiple", "exactly one"),
        ("hash", "sha256"),
        ("size", "size"),
        ("artifact_id", "artifact_id"),
    ],
)
def test_load_evaluation_states_rejects_invalid_manifest_and_artifact_contract(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1, 2])}, root=tmp_path, manifest_id="invalid"
    )
    status = "failed" if mutation == "status" else "completed"
    artifacts = [artifact]
    if mutation == "missing":
        artifacts = []
    elif mutation == "wrong_role":
        artifacts = [artifact.model_copy(update={"role": "other"})]
    elif mutation == "multiple":
        artifacts = [artifact, artifact.model_copy()]
    elif mutation == "hash":
        artifacts = [artifact.model_copy(update={"sha256": "0" * 64})]
    elif mutation == "size":
        artifacts = [artifact.model_copy(update={"size_bytes": artifact.size_bytes + 1})]
    elif mutation == "artifact_id":
        artifacts = [artifact.model_copy(update={"artifact_id": "artifact://sha256/" + "0" * 64})]
    manifest = _manifest(artifacts[0] if len(artifacts) == 1 else artifact, status=status)
    manifest.artifacts = artifacts
    context, parent = _local_context(manifest, tmp_path)

    with pytest.raises((StagedExecutionContextError, ValueError), match=message):
        context.load_evaluation_states(parent)


def test_load_evaluation_states_rejects_local_byte_tamper(tmp_path: Path) -> None:
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1, 2])}, root=tmp_path, manifest_id="tamper"
    )
    context, parent = _local_context(_manifest(artifact), tmp_path)
    relative = artifact.metadata.get("relative_path") or artifact.uri
    path = tmp_path / str(relative)
    original = path.read_bytes()
    path.write_bytes(b"x" * len(original))

    with pytest.raises(ValueError, match="SHA-256 mismatch|sha256"):
        context.load_evaluation_states(parent)


@pytest.mark.parametrize("locator", ["../escape.npz", "/tmp/absolute.npz"])
def test_load_evaluation_states_rejects_noncanonical_local_locator(
    tmp_path: Path,
    locator: str,
) -> None:
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1])}, root=tmp_path, manifest_id="locator"
    ).model_copy(update={"uri": locator})
    context, parent = _local_context(
        _manifest(artifact), tmp_path, normalize_locators=False
    )

    with pytest.raises(StagedExecutionContextError, match="must equal canonical path"):
        context.load_evaluation_states(parent)


def test_load_evaluation_states_rejects_local_path_escape(tmp_path: Path) -> None:
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1])}, root=tmp_path, manifest_id="escape"
    ).model_copy(
        update={
            "uri": "../escape.npz",
            "metadata": {"relative_path": "../escape.npz"},
        }
    )
    context, parent = _local_context(
        _manifest(artifact), tmp_path, normalize_locators=False
    )

    with pytest.raises(StagedExecutionContextError, match="escapes its explicit root"):
        context.load_evaluation_states(parent)


def test_load_evaluation_states_rejects_conflicting_local_locators(tmp_path: Path) -> None:
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1])}, root=tmp_path, manifest_id="conflict"
    ).model_copy(update={"uri": "other/states.npz"})
    context, parent = _local_context(
        _manifest(artifact), tmp_path, normalize_locators=False
    )

    with pytest.raises(StagedExecutionContextError, match="must equal canonical path"):
        context.load_evaluation_states(parent)


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_load_evaluation_states_rejects_local_file_alias(
    tmp_path: Path,
    alias: str,
) -> None:
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1])}, root=tmp_path, manifest_id="alias"
    )
    context, parent = _local_context(_manifest(artifact), tmp_path)
    relative = artifact.metadata["relative_path"]
    path = tmp_path / relative
    if alias == "symlink":
        original = tmp_path / "original.npz"
        path.rename(original)
        path.symlink_to(original)
        message = "symlink"
    else:
        (tmp_path / "alias.npz").hardlink_to(path)
        message = "hard-link"

    with pytest.raises(ValueError, match=message):
        context.load_evaluation_states(parent)


def test_load_evaluation_states_rejects_local_parent_symlink(tmp_path: Path) -> None:
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1])}, root=tmp_path, manifest_id="parent-symlink"
    )
    context, parent = _local_context(_manifest(artifact), tmp_path)
    relative = Path(artifact.metadata["relative_path"])
    parent_path = tmp_path / relative.parent
    moved = tmp_path / "moved-artifact-parent"
    parent_path.rename(moved)
    parent_path.symlink_to(moved, target_is_directory=True)

    with pytest.raises(StagedExecutionContextError, match="symlink|unsafe component"):
        context.load_evaluation_states(parent)


def test_load_evaluation_states_rejects_symlinked_authority_parent(
    tmp_path: Path,
) -> None:
    authority_parent = tmp_path / "authority-parent"
    root = authority_parent / "root"
    root.mkdir(parents=True)
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1])}, root=root, manifest_id="unauthorized-parent"
    )
    context, parent = _local_context(_manifest(artifact), root)

    detached_parent = tmp_path / "authority-parent-detached"
    authority_parent.rename(detached_parent)
    unauthorized_parent = tmp_path / "unauthorized-parent"
    shutil.copytree(detached_parent, unauthorized_parent)
    authority_parent.symlink_to(unauthorized_parent, target_is_directory=True)

    with pytest.raises(StagedExecutionContextError, match="unsafe|replaced"):
        context.load_evaluation_states(parent)


def test_load_evaluation_states_rejects_local_root_replacement_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1])}, root=root, manifest_id="root-replacement"
    )
    context, parent = _local_context(_manifest(artifact), root)
    original_read = os.read
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        if not replaced:
            replaced = True
            root.rename(tmp_path / "authority-original")
            root.mkdir()
        return original_read(descriptor, size)

    monkeypatch.setattr(execution_context_module.os, "read", replacing_read)
    with pytest.raises(StagedExecutionContextError, match="root authority.*replaced"):
        context.load_evaluation_states(parent)


def test_load_evaluation_states_rejects_intermediate_directory_replacement_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1])}, root=root, manifest_id="directory-replacement"
    )
    context, parent = _local_context(_manifest(artifact), root)
    original_read = os.read
    read_count = 0

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal read_count
        read_count += 1
        if read_count == 3:
            directory = root / "artifacts" / "sha256"
            directory.rename(root / "artifacts" / "sha256-detached")
            directory.mkdir()
        return original_read(descriptor, size)

    monkeypatch.setattr(execution_context_module.os, "read", replacing_read)
    with pytest.raises(StagedExecutionContextError, match="directory identity changed"):
        context.load_evaluation_states(parent)


def test_load_evaluation_states_rejects_intermediate_symlink_swap_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1])}, root=root, manifest_id="symlink-swap"
    )
    context, parent = _local_context(_manifest(artifact), root)
    original_read = os.read
    read_count = 0

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal read_count
        read_count += 1
        if read_count == 3:
            directory = root / "artifacts" / "sha256"
            detached = root / "artifacts" / "sha256-detached"
            directory.rename(detached)
            directory.symlink_to(detached, target_is_directory=True)
        return original_read(descriptor, size)

    monkeypatch.setattr(execution_context_module.os, "read", replacing_read)
    with pytest.raises(StagedExecutionContextError, match="directory identity changed"):
        context.load_evaluation_states(parent)


def test_load_evaluation_states_rejects_wrong_existing_provider_binding(
    tmp_path: Path,
) -> None:
    expected_root = tmp_path / "expected-provider"
    wrong_root = tmp_path / "wrong-provider"
    expected_root.mkdir()
    wrong_root.mkdir()
    expected = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(), explicit_root=expected_root
    )
    wrong = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(), explicit_root=wrong_root
    )
    states_bytes, _ = evaluation_states_container_bytes({"value": np.asarray([3])})
    artifact = expected.store_bytes(
        states_bytes, role="evaluation_states", logical_name="states.npz"
    )
    manifest = _manifest(artifact)
    raw = (manifest.model_dump_json(indent=2, exclude_none=True) + "\n").encode()
    manifest_artifact = expected.store_bytes(
        raw, role="evaluation_run", logical_name="evaluation.json"
    )
    parent = _parent(manifest, raw)
    context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"expected": expected, "wrong": wrong},
        checkpoint_custody_roots={},
        parent_execution_locations=(
            StagedParentExecutionLocation(
                parent=parent,
                root=wrong_root,
                execution_uri=expected.canonical_relative_path(manifest_artifact).as_posix(),
                artifact_provider="wrong",
            ),
        ),
    )

    with pytest.raises(FileNotFoundError, match="missing"):
        context.load_evaluation_states(parent)


def test_load_evaluation_states_rejects_provider_manifest_id_mismatch(tmp_path: Path) -> None:
    provider_root = tmp_path / "provider"
    provider_root.mkdir()
    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(), explicit_root=provider_root
    )
    states_bytes, _ = evaluation_states_container_bytes({"value": np.asarray([3])})
    artifact = provider.store_bytes(
        states_bytes, role="evaluation_states", logical_name="states.npz"
    )
    manifest = _manifest(artifact)
    raw = (manifest.model_dump_json(indent=2, exclude_none=True) + "\n").encode()
    manifest_artifact = provider.store_bytes(
        raw, role="evaluation_run", logical_name="evaluation.json"
    )
    parent = _parent(manifest, raw).model_copy(update={"id": "different-evaluation"})
    context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"external": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=(
            StagedParentExecutionLocation(
                parent=parent,
                root=provider_root,
                execution_uri=provider.canonical_relative_path(manifest_artifact).as_posix(),
                artifact_provider="external",
            ),
        ),
    )

    with pytest.raises(StagedExecutionContextError, match="kind or id disagrees"):
        context.load_evaluation_states(parent)


@pytest.mark.parametrize(("mutation", "message"), [("size", "size"), ("id", "artifact_id")])
def test_load_evaluation_states_rejects_provider_artifact_reference_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    provider_root = tmp_path / "provider"
    provider_root.mkdir()
    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(), explicit_root=provider_root
    )
    states_bytes, _ = evaluation_states_container_bytes({"value": np.asarray([3])})
    artifact = provider.store_bytes(
        states_bytes, role="evaluation_states", logical_name="states.npz"
    )
    if mutation == "size":
        artifact = artifact.model_copy(update={"size_bytes": artifact.size_bytes + 1})
    else:
        artifact = artifact.model_copy(
            update={"artifact_id": "artifact://sha256/" + "0" * 64}
        )
    manifest = _manifest(artifact)
    raw = (manifest.model_dump_json(indent=2, exclude_none=True) + "\n").encode()
    manifest_artifact = provider.store_bytes(
        raw, role="evaluation_run", logical_name="evaluation.json"
    )
    parent = _parent(manifest, raw)
    context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"external": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=(
            StagedParentExecutionLocation(
                parent=parent,
                root=provider_root,
                execution_uri=provider.canonical_relative_path(manifest_artifact).as_posix(),
                artifact_provider="external",
            ),
        ),
    )

    with pytest.raises(ValueError, match=message):
        context.load_evaluation_states(parent)


def test_load_evaluation_states_rejects_missing_provider_and_replaced_root(
    tmp_path: Path,
) -> None:
    provider_root = tmp_path / "provider"
    provider_root.mkdir()
    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(), explicit_root=provider_root
    )
    states_bytes, _ = evaluation_states_container_bytes({"value": np.asarray([3])})
    artifact = provider.store_bytes(
        states_bytes, role="evaluation_states", logical_name="states.npz"
    )
    manifest = _manifest(artifact)
    raw = (manifest.model_dump_json(indent=2, exclude_none=True) + "\n").encode()
    manifest_artifact = provider.store_bytes(
        raw, role="evaluation_run", logical_name="evaluation.json"
    )
    parent = _parent(manifest, raw)
    location = StagedParentExecutionLocation(
        parent=parent,
        root=provider_root,
        execution_uri=provider.canonical_relative_path(manifest_artifact).as_posix(),
        artifact_provider="external",
    )
    missing = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={},
        checkpoint_custody_roots={},
        parent_execution_locations=(location,),
    )
    with pytest.raises(StagedExecutionContextError, match="binding is unavailable"):
        missing.load_evaluation_states(parent)

    bound = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={"external": provider},
        checkpoint_custody_roots={},
        parent_execution_locations=(location,),
    )
    original_root = tmp_path / "provider-original"
    provider_root.rename(original_root)
    provider_root.mkdir()
    with pytest.raises(StagedExecutionContextError, match="replaced after binding"):
        bound.load_evaluation_states(parent)
