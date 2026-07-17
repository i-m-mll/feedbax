import copy
from pathlib import Path
import hashlib
import os
import shutil

import jax.tree as jt
import jax.tree_util as jtu
import numpy as np
import pytest

import feedbax.analysis.channel_evidence as channel_evidence_module
import feedbax.analysis.execution_context as execution_context_module

from feedbax.analysis import (
    StagedLocatorAbsoluteError,
    StagedLocatorMismatchError,
    StagedLocatorMissingError,
    StagedLocatorTraversalError,
    resolve_authenticated_evaluation_channels,
)
from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedArtifactProviderRootBinding,
    StagedExecutionContext,
    StagedExecutionContextError,
    StagedParentExecutionLocation,
    resolve_staged_execution_context,
    with_staged_parent_execution_locations,
)
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    execute_evaluation_run_spec,
    register_evaluation_recipe,
    resolve_staged_evaluation_prerequisite,
    unregister_evaluation_recipe,
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
    EvaluationRunSpec,
    ParentRef,
    SpecPayload,
    StagedEvaluationPrerequisite,
    sha256_bytes,
    write_manifest,
)
from feedbax.contracts.staged_execution import (
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedExecutionDescriptor,
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
    artifact = store_evaluation_states_artifact(
        expected,
        root=tmp_path,
        manifest_id="feedbax-evaluation-run:authority-test",
    )
    context, parent = _local_context(
        _manifest(artifact), tmp_path, normalize_locators=False
    )

    states = context.load_evaluation_states(parent)
    np.testing.assert_array_equal(states["trajectory"], expected["trajectory"])


def test_identical_authenticated_states_are_decoded_once_and_reused_immutably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"trajectory": np.arange(4, dtype=np.float32)}
    artifact = store_evaluation_states_artifact(
        expected,
        root=tmp_path,
        manifest_id="feedbax-evaluation-run:authority-test",
    )
    context, parent = _local_context(_manifest(artifact), tmp_path)
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

    first_manifest = context.resolve_manifest_input(parent)
    first = context.load_evaluation_states(parent)
    second = context.load_evaluation_states(parent)

    assert context.resolve_manifest_input(parent) is first_manifest
    assert second is first
    assert loads == 1
    assert not first["trajectory"].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first["trajectory"][0] = -1
    with pytest.raises(TypeError, match="immutable"):
        first["trajectory"] = np.asarray([777], dtype=np.float32)


def test_authenticated_manifest_snapshot_rejects_authority_substitution(
    tmp_path: Path,
) -> None:
    manifest_id = "feedbax-evaluation-run:authority-test"
    first_artifact = store_evaluation_states_artifact(
        {"trajectory": np.asarray([1, 2], dtype=np.int32)},
        root=tmp_path,
        manifest_id=manifest_id,
    )
    second_artifact = store_evaluation_states_artifact(
        {"trajectory": np.asarray([90, 91], dtype=np.int32)},
        root=tmp_path,
        manifest_id=manifest_id,
    )
    context, parent = _local_context(_manifest(first_artifact), tmp_path)
    resolved = context.resolve_manifest_input(parent)
    replacement = second_artifact.model_copy(
        update={"uri": second_artifact.metadata["relative_path"]}
    )

    with pytest.raises(TypeError, match="immutable"):
        resolved.manifest.artifacts[0] = replacement
    with pytest.raises((TypeError, ValueError), match="frozen|immutable"):
        resolved.manifest.artifacts[0].metadata["relative_path"] = (
            second_artifact.metadata["relative_path"]
        )

    states = context.load_evaluation_states(parent)
    np.testing.assert_array_equal(states["trajectory"], np.asarray([1, 2]))
    assert copy.deepcopy(resolved.manifest) == resolved.manifest


class _MutableAux:
    def __init__(self, tag: str):
        self.tag = tag

    def __hash__(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _MutableAux) and self.tag == other.tag

    def __repr__(self) -> str:
        return f"_MutableAux({self.tag!r})"


@jtu.register_pytree_node_class
class _MutableStructureNode:
    def __init__(self, value, tag: str):
        self.value = value
        self.tag = tag

    def tree_flatten(self):
        return (self.value,), _MutableAux(self.tag)

    @classmethod
    def tree_unflatten(cls, aux: _MutableAux, children):
        return cls(children[0], aux.tag)


def test_mutated_custom_pytree_aux_misses_and_rechecks_structure_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = store_evaluation_states_artifact(
        _MutableStructureNode(np.asarray([8], dtype=np.int32), "original"),
        root=tmp_path,
        manifest_id="feedbax-evaluation-run:authority-test",
    )
    context, parent = _local_context(_manifest(artifact), tmp_path)
    structure = jt.structure(_MutableStructureNode(0, "original"))
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

    first = context.load_evaluation_states(parent, structure=structure)
    structure.node_data()[1].tag = "mutated"
    with pytest.raises(ValueError, match="structure fingerprint"):
        context.load_evaluation_states(parent, structure=structure)

    assert first.tag == "original"
    assert loads == 2


def test_requested_structure_is_part_of_authenticated_states_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"trajectory": np.arange(4, dtype=np.float32)}
    artifact = store_evaluation_states_artifact(
        expected,
        root=tmp_path,
        manifest_id="feedbax-evaluation-run:authority-test",
    )
    context, parent = _local_context(_manifest(artifact), tmp_path)
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

    without_structure = context.load_evaluation_states(parent)
    with_structure = context.load_evaluation_states(
        parent,
        structure=jt.structure(expected),
    )

    assert loads == 2
    assert with_structure is not without_structure
    np.testing.assert_array_equal(with_structure["trajectory"], expected["trajectory"])


def test_parent_digest_and_execution_location_differences_miss_the_memo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    locations = []
    parents = []
    artifact_digests = []
    for index, root in enumerate(roots):
        manifest_id = f"feedbax-evaluation-run:authority-{index}"
        artifact = store_evaluation_states_artifact(
            {"trajectory": np.arange(4, dtype=np.float32) + index},
            root=root,
            manifest_id=manifest_id,
        )
        manifest = _manifest(artifact).model_copy(update={"id": manifest_id})
        artifact_digests.append(artifact.sha256)
        path = write_manifest(manifest, root=root, index=False)
        parent = _parent(manifest, path.read_bytes())
        parents.append(parent)
        locations.append(
            StagedParentExecutionLocation(
                parent=parent,
                root=root,
                execution_uri=path.relative_to(root).as_posix(),
            )
        )
    context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={},
        checkpoint_custody_roots={},
        parent_execution_locations=tuple(locations),
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

    resolved = [context.load_evaluation_states(parent) for parent in parents]

    assert loads == 2
    assert resolved[0] is not resolved[1]
    assert parents[0].metadata["manifest_sha256"] != parents[1].metadata["manifest_sha256"]
    assert artifact_digests[0] != artifact_digests[1]
    assert locations[0].root != locations[1].root


def test_execution_location_difference_misses_with_identical_authenticated_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    artifact = store_evaluation_states_artifact(
        {"trajectory": np.arange(4, dtype=np.float32)},
        root=first_root,
        manifest_id="feedbax-evaluation-run:authority-test",
    )
    manifest = _manifest(artifact)
    first_path = write_manifest(manifest, root=first_root, index=False)
    second_root = tmp_path / "second"
    shutil.copytree(first_root, second_root)
    second_path = second_root / first_path.relative_to(first_root)
    first_parent = _parent(manifest, first_path.read_bytes())
    second_parent = first_parent.model_copy(
        update={"metadata": {**first_parent.metadata, "location_variant": "second"}}
    )
    context = StagedExecutionContext(
        descriptor=None,
        opened_artifact_providers={},
        checkpoint_custody_roots={},
        parent_execution_locations=(
            StagedParentExecutionLocation(
                parent=first_parent,
                root=first_root,
                execution_uri=first_path.relative_to(first_root).as_posix(),
            ),
            StagedParentExecutionLocation(
                parent=second_parent,
                root=second_root,
                execution_uri=second_path.relative_to(second_root).as_posix(),
            ),
        ),
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

    first = context.load_evaluation_states(first_parent)
    second = context.load_evaluation_states(second_parent)

    assert first_parent.metadata["manifest_sha256"] == second_parent.metadata["manifest_sha256"]
    assert first is not second
    assert loads == 2


def test_changed_retained_file_identity_invalidates_and_reverifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"trajectory": np.arange(4, dtype=np.float32)}
    artifact = store_evaluation_states_artifact(
        expected,
        root=tmp_path,
        manifest_id="feedbax-evaluation-run:authority-test",
    )
    context, parent = _local_context(_manifest(artifact), tmp_path)
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
    first = context.load_evaluation_states(parent)
    path = tmp_path / artifact.metadata["relative_path"]
    replacement = tmp_path / "replacement.npz"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)

    second = context.load_evaluation_states(parent)

    assert second is not first
    assert loads == 2
    np.testing.assert_array_equal(second["trajectory"], expected["trajectory"])


def test_authenticated_channel_evidence_is_cached_with_its_state_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = np.arange(6, dtype=np.float64).reshape(2, 3)
    states = {"channels": {"noise": channel}, "sample_index": np.arange(2)}
    artifact = store_evaluation_states_artifact(
        states,
        root=tmp_path,
        manifest_id="feedbax-evaluation-run:authority-test",
    )
    manifest = _manifest(artifact).model_copy(
        update={
            "metadata": {
                "channels": [
                    {
                        "name": "noise",
                        "index": 0,
                        "shape": list(channel.shape),
                        "dtype": channel.dtype.str,
                        "byte_order": channel.dtype.str[0],
                        "c_contiguous": True,
                        "sha256": hashlib.sha256(
                            channel.tobytes(order="C")
                        ).hexdigest(),
                    }
                ]
            }
        }
    )
    context, parent = _local_context(manifest, tmp_path)
    prerequisite = StagedEvaluationPrerequisite(parent=parent)
    original_authenticate = channel_evidence_module._authenticate_channel
    authentications = 0

    def counted_authenticate(*args, **kwargs):
        nonlocal authentications
        authentications += 1
        return original_authenticate(*args, **kwargs)

    monkeypatch.setattr(
        channel_evidence_module,
        "_authenticate_channel",
        counted_authenticate,
    )

    first = resolve_authenticated_evaluation_channels(
        prerequisite,
        execution_context=context,
    )
    second = resolve_authenticated_evaluation_channels(
        prerequisite,
        execution_context=context,
    )

    assert second is first
    assert authentications == 1
    assert not first.channels["noise"].flags.writeable


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
        manifest_id="feedbax-evaluation-run:authority-test",
    )
    states_bytes = (source_cache_root / source_artifact.metadata["relative_path"]).read_bytes()
    provider.store_bytes(
        states_bytes,
        role="evaluation_states",
        logical_name="states.npz",
        media_type=source_artifact.media_type,
    )
    artifact = source_artifact.model_copy(
        update={"uri": str(source_cache_root / source_artifact.metadata["relative_path"])}
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

    artifact_path = provider_root / provider.canonical_relative_path(
        artifact.artifact_id,
        size_bytes=artifact.size_bytes,
    )
    original = artifact_path.read_bytes()
    artifact_path.write_bytes(b"x" * len(original))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        context.load_evaluation_states(parent)


def test_public_producer_round_trips_through_authored_provider_binding(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    provider_root = tmp_path / "provider"
    source_root.mkdir()
    provider_root.mkdir()
    expected = {"trajectory": np.arange(4, dtype=np.float32)}
    evaluation_type = "feedbax.test.staged_locator_round_trip"

    def recipe(*_args) -> EvaluationRecipeResult:
        return EvaluationRecipeResult(states=expected)

    register_evaluation_recipe(evaluation_type, recipe, replace=True)
    try:
        manifest, manifest_path = execute_evaluation_run_spec(
            EvaluationRunSpec(
                evaluation_type=evaluation_type,
                params={"states_custody": "durable"},
            ),
            root=source_root,
        )
    finally:
        unregister_evaluation_recipe(evaluation_type)

    artifact = next(item for item in manifest.artifacts if item.role == "evaluation_states")
    relative_path = artifact.metadata["relative_path"]
    states_bytes = (source_root / relative_path).read_bytes()
    assert artifact.uri is None
    assert artifact.artifact_id == f"artifact://sha256/{artifact.sha256}"
    assert artifact.size_bytes == len(states_bytes)

    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(), explicit_root=provider_root
    )
    provider.store_bytes(states_bytes, role="evaluation_states", logical_name="states.npz")
    raw = manifest_path.read_bytes()
    manifest_blob = provider.store_bytes(raw, role="evaluation_run", logical_name="run.json")
    parent = _parent(manifest, raw)
    context = resolve_staged_execution_context(
        StagedExecutionDescriptor(
            schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
            schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
            artifact_providers={"external": ImmutableArtifactBlobProviderSpec()},
            checkpoint_custody={},
        ),
        artifact_provider_bindings=[
            StagedArtifactProviderRootBinding("external", provider_root)
        ],
    )
    context = with_staged_parent_execution_locations(
        context,
        [
            StagedParentExecutionLocation(
                parent=parent,
                root=provider_root,
                execution_uri=provider.canonical_relative_path(manifest_blob).as_posix(),
                artifact_provider="external",
            )
        ],
    )
    loaded = resolve_staged_evaluation_prerequisite(
        StagedEvaluationPrerequisite(parent=parent, artifact_provider="external"),
        execution_context=context,
    )
    np.testing.assert_array_equal(loaded["trajectory"], expected["trajectory"])
    assert artifact.metadata["schema_version"].endswith(".v2")
    assert artifact.metadata["storage_backend"] == "npz.v2"


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


def test_load_evaluation_states_rejects_absolute_local_locator(tmp_path: Path) -> None:
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1])}, root=tmp_path, manifest_id="locator"
    ).model_copy(update={"uri": "/tmp/absolute.npz"})
    context, parent = _local_context(
        _manifest(artifact), tmp_path, normalize_locators=False
    )

    with pytest.raises(StagedLocatorAbsoluteError, match="machine-local absolute") as excinfo:
        context.load_evaluation_states(parent)
    assert isinstance(excinfo.value, StagedExecutionContextError)


def test_load_evaluation_states_rejects_missing_local_locator(tmp_path: Path) -> None:
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1])}, root=tmp_path, manifest_id="missing-locator"
    ).model_copy(update={"uri": None, "metadata": {}})
    context, parent = _local_context(_manifest(artifact), tmp_path)

    with pytest.raises(StagedLocatorMissingError, match="canonical relative locator") as excinfo:
        context.load_evaluation_states(parent)
    assert isinstance(excinfo.value, StagedExecutionContextError)


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

    with pytest.raises(StagedLocatorTraversalError, match="escapes its explicit root") as excinfo:
        context.load_evaluation_states(parent)
    assert isinstance(excinfo.value, StagedExecutionContextError)


def test_load_evaluation_states_rejects_serialized_raw_nul_locator(tmp_path: Path) -> None:
    locator = "bad\x00name.npz"
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1])}, root=tmp_path, manifest_id="raw-nul"
    ).model_copy(
        update={"uri": locator, "metadata": {"relative_path": locator}}
    )
    context, parent = _local_context(
        _manifest(artifact), tmp_path, normalize_locators=False
    )

    with pytest.raises(StagedLocatorTraversalError) as excinfo:
        context.load_evaluation_states(parent)
    assert isinstance(excinfo.value, StagedExecutionContextError)


def test_load_evaluation_states_rejects_conflicting_local_locators(tmp_path: Path) -> None:
    artifact = store_evaluation_states_artifact(
        {"value": np.asarray([1])}, root=tmp_path, manifest_id="conflict"
    ).model_copy(update={"uri": "other/states.npz"})
    context, parent = _local_context(
        _manifest(artifact), tmp_path, normalize_locators=False
    )

    with pytest.raises(StagedLocatorMismatchError, match="must equal canonical path") as excinfo:
        context.load_evaluation_states(parent)
    assert isinstance(excinfo.value, StagedExecutionContextError)


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
