"""Driver-neutral materialization of authenticated orchestration inputs."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from feedbax.contracts.artifact_custody import (
    ContentBinding,
    ContentLeaseRecord,
    ContentObjectRef,
    ContentRetentionTarget,
    ContentTree,
    ContentTreeRef,
    ImmutableArtifactBlobProviderSpec,
)
from feedbax.contracts.base import ArtifactRef
from feedbax.contracts.canonical_json import canonical_json_v1_bytes
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.contracts.staged_execution import (
    STAGED_CHECKPOINT_CUSTODY_BACKEND,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedCheckpointCustodySpec,
    StagedExecutionDescriptor,
    validate_staged_binding_name,
)
from feedbax.orchestration.bundle import ResolvedAssemblyInput, RunBundle
from feedbax.orchestration.staged_root_custody import (
    StagedExecutionRootBindings,
    StagedRootCustody,
    StagedRootCustodyError,
    StagedRootSnapshotBinding,
    materialize_staged_root_snapshot,
    verify_staged_root_snapshot,
    verify_staged_root_snapshot_binding,
)
from feedbax.persistence.artifact_custody import (
    ImmutableArtifactBlobProvider,
    open_immutable_artifact_blob_provider,
)
from feedbax.training.checkpoint_custody import (
    CHECKPOINT_SET_NAME,
    materialize_checkpoint_custody_archive,
    publish_directory_no_replace,
)


class InputMaterializationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class InputProviderRootBinding:
    name: str
    root: Path | str


@dataclass(frozen=True, slots=True)
class StagedInputFile:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class StagedInput:
    target_role: str
    destination: Path
    files: tuple[StagedInputFile, ...]


@dataclass(frozen=True, slots=True)
class ContentInputProjection:
    """Ephemeral exact closure of typed bindings over canonical content."""

    bindings: tuple[ContentBinding, ...]
    trees: tuple[tuple[ContentTreeRef, ContentTree], ...]
    objects: tuple[ContentObjectRef, ...]

    @property
    def unique_bytes(self) -> int:
        return sum(item.size_bytes for item in self.objects)


def acquire_content_projection_lease(
    projection: ContentInputProjection,
    provider: ImmutableArtifactBlobProvider,
    *,
    lease_id: str,
    owner: str,
    expires_at: datetime,
) -> None:
    """Bind a nonterminal attempt to its exact tree roots."""
    if not projection.bindings:
        return
    provider.acquire_lease(
        ContentLeaseRecord(
            lease_id=lease_id,
            owner=owner,
            expires_at=expires_at,
            targets=tuple(
                ContentRetentionTarget(ref_kind="tree", tree=ref) for ref, _tree in projection.trees
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class StagedRootReclamation:
    """Result of reclaiming one run-local staged-root materialization."""

    status: Literal["not-applicable", "reclaimed", "already-reclaimed"]
    custody_refs: tuple[str, ...]
    reclaimed_bytes: int


_STAGED_ROOT_RECLAMATION_SCHEMA_VERSION = "feedbax.orchestration.staged_root_reclamation.v1"


def preflight_resolved_inputs(bundle: RunBundle) -> tuple[list[str], list[dict[str, str]]]:
    failures: list[str] = []
    observed: list[dict[str, str]] = []
    roles: set[str] = set()
    for index, resolved in enumerate(bundle.resolved_inputs):
        source = resolved.custody
        try:
            validate_staged_binding_name(source.target_role)
        except ValueError as exc:
            failures.append(f"resolved_inputs[{index}].target_role: {exc}")
        if source.target_role in roles:
            failures.append(f"duplicate input target role: {source.target_role!r}")
        roles.add(source.target_role)
        if resolved.identity.digest.value != source.artifact.sha256:
            failures.append(f"resolved_inputs[{index}] identity/custody digest mismatch")
        parent = source.materializer.expected_parent_ref
        if not isinstance(parent.metadata.get("manifest_sha256"), str):
            failures.append(f"resolved_inputs[{index}] ParentRef lacks manifest digest")
        observed.append(
            {
                "role": source.target_role,
                "provider_binding": source.provider_binding,
                "artifact_sha256": source.artifact.sha256,
                "transaction_id": parent.id,
            }
        )
    return failures, observed


def preflight_input_provider_bindings(
    bundle: RunBundle, bindings: Sequence[InputProviderRootBinding]
) -> tuple[list[str], list[dict[str, str]]]:
    failures, observed = preflight_resolved_inputs(bundle)
    observed = observed[: len(bundle.resolved_inputs)]
    try:
        providers = _bound_providers(bundle.resolved_inputs, bindings)
        for item in bundle.resolved_inputs:
            providers[item.custody.provider_binding].get_bytes(_artifact_ref(item))
    except Exception as exc:
        failures.append(str(exc))
    return failures, observed


def preflight_staged_root_bindings(
    bundle: RunBundle,
    bindings: Sequence[StagedRootSnapshotBinding],
) -> tuple[list[str], list[dict[str, str]]]:
    """Reject legacy runtime bindings at the successor-schema boundary."""
    del bundle
    if bindings:
        return (["legacy staged roots require explicit offline content-store migration"], [])
    return [], []


def preflight_bundle_input_bindings(
    bundle: RunBundle,
    *,
    provider_bindings: Sequence[InputProviderRootBinding],
    staged_root_bindings: Sequence[StagedRootSnapshotBinding],
    content_store_root: Path | str | None = None,
) -> tuple[list[str], object]:
    """Preflight checkpoint archives and staged roots through one input boundary."""
    provider_failures, provider_observed = preflight_input_provider_bindings(
        bundle, provider_bindings
    )
    staged_failures, _staged_observed = preflight_staged_root_bindings(bundle, staged_root_bindings)
    content_observed: list[dict[str, object]] = []
    content_failures: list[str] = []
    if bundle.content_bindings:
        if content_store_root is None:
            content_failures.append("content bindings require one explicit canonical store root")
        else:
            try:
                projection = compile_content_input_projection(
                    bundle.content_bindings,
                    ImmutableArtifactBlobProvider(content_store_root),
                )
                content_observed = [
                    {
                        "binding_name": binding.binding_name,
                        "root_kind": binding.root_kind,
                        "tree_id": binding.tree.tree_id,
                    }
                    for binding in projection.bindings
                ]
            except Exception as exc:
                content_failures.append(str(exc))
    observed: object = provider_observed
    if bundle.content_bindings:
        observed = {
            "resolved_inputs": provider_observed,
            "content_bindings": content_observed,
        }
    return [*provider_failures, *staged_failures, *content_failures], observed


def compile_content_input_projection(
    bindings: Sequence[ContentBinding],
    provider: ImmutableArtifactBlobProvider,
) -> ContentInputProjection:
    """Resolve and authenticate the unique tree/object closure without writing."""
    ordered = tuple(sorted(bindings, key=lambda item: (item.root_kind, item.binding_name)))
    if tuple(bindings) != ordered:
        raise InputMaterializationError(
            "content bindings must be in canonical (root_kind, binding_name) order"
        )
    keys = [(item.root_kind, item.binding_name) for item in ordered]
    if len(keys) != len(set(keys)):
        raise InputMaterializationError("content bindings contain duplicate typed roles")

    trees: dict[str, tuple[ContentTreeRef, ContentTree]] = {}
    objects: dict[str, ContentObjectRef] = {}
    for binding in ordered:
        validate_staged_binding_name(binding.root_kind)
        validate_staged_binding_name(binding.binding_name)
        if binding.root_kind not in {
            "artifact-provider",
            "checkpoint-custody",
            "manifest-store",
        }:
            raise InputMaterializationError(
                f"unsupported content binding root_kind: {binding.root_kind!r}"
            )
        current = trees.get(binding.tree.tree_id)
        tree = provider.get_tree(binding.tree) if current is None else current[1]
        if current is None:
            trees[binding.tree.tree_id] = (binding.tree, tree)
        objects[binding.tree.descriptor.sha256] = binding.tree.descriptor
        for member in tree.files:
            previous = objects.get(member.object.sha256)
            if previous is not None and previous != member.object:
                raise InputMaterializationError(
                    f"content object metadata conflicts for digest {member.object.sha256}"
                )
            if previous is None:
                provider.get_object(member.object)
                objects[member.object.sha256] = member.object
    return ContentInputProjection(
        bindings=ordered,
        trees=tuple(trees[key] for key in sorted(trees)),
        objects=tuple(objects[key] for key in sorted(objects)),
    )


def materialize_content_input_projection(
    projection: ContentInputProjection,
    provider: ImmutableArtifactBlobProvider,
    *,
    destination_root: Path | str,
    object_root: Path | str | None = None,
) -> tuple[StagedInput, ...]:
    """Materialize each unique tree once and point every typed role at that view."""
    root = Path(destination_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    objects_root = (
        Path(object_root).expanduser().resolve()
        if object_root is not None
        else root.parent / "objects" / "sha256"
    )
    local_objects: dict[str, Path] = {}
    for object_ref in projection.objects:
        destination = objects_root / object_ref.sha256[:2] / object_ref.sha256
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = provider.get_object(object_ref)
        if os.path.lexists(destination):
            if _read_regular_no_follow(destination) != data:
                raise InputMaterializationError(
                    f"conflicting local content object: {object_ref.sha256}"
                )
        else:
            handle, temporary_name = tempfile.mkstemp(
                prefix=".content-object-", dir=destination.parent
            )
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary_name, destination, follow_symlinks=False)
                except FileExistsError:
                    pass
            finally:
                Path(temporary_name).unlink(missing_ok=True)
            if _read_regular_no_follow(destination) != provider.get_object(object_ref):
                raise InputMaterializationError(
                    f"published local content object differs: {object_ref.sha256}"
                )
        local_objects[object_ref.sha256] = destination
    descriptors_root = root.parent / "descriptors"
    descriptors_root.mkdir(parents=True, exist_ok=True)
    for tree_ref, _tree in projection.trees:
        descriptor = descriptors_root / tree_ref.descriptor.sha256
        if not os.path.lexists(descriptor):
            try:
                os.link(
                    local_objects[tree_ref.descriptor.sha256],
                    descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                pass
        if _read_regular_no_follow(descriptor) != provider.get_object(tree_ref.descriptor):
            raise InputMaterializationError(
                f"materialized content-tree descriptor differs: {tree_ref.tree_id}"
            )
    materialized: dict[str, tuple[Path, tuple[StagedInputFile, ...]]] = {}
    for tree_ref, tree in projection.trees:
        destination = root / tree_ref.descriptor.sha256
        if os.path.lexists(destination):
            files = _verify_content_tree_view(provider, tree, destination, root)
        else:
            build = Path(tempfile.mkdtemp(prefix=".content-tree-", dir=root))
            try:
                for directory in tree.directories:
                    (build / directory).mkdir(parents=True, exist_ok=True)
                for member in tree.files:
                    target = build / member.relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(local_objects[member.object.sha256], target, follow_symlinks=False)
                files = _verify_content_tree_view(provider, tree, build, root)
                try:
                    parent_descriptor = os.open(
                        root,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
                    )
                    try:
                        publish_directory_no_replace(
                            parent_descriptor,
                            build.name,
                            destination.name,
                            expected_identity=os.stat(
                                build.name,
                                dir_fd=parent_descriptor,
                                follow_symlinks=False,
                            ),
                        )
                    finally:
                        os.close(parent_descriptor)
                except FileExistsError:
                    _remove_materialization_tree(build)
                    files = _verify_content_tree_view(provider, tree, destination, root)
            except Exception:
                if build.exists():
                    _remove_materialization_tree(build)
                raise
        materialized[tree_ref.tree_id] = (destination, files)
    return tuple(
        StagedInput(
            target_role=f"content/{binding.root_kind}/{binding.binding_name}",
            destination=materialized[binding.tree.tree_id][0],
            files=materialized[binding.tree.tree_id][1],
        )
        for binding in projection.bindings
    )


def _read_regular_no_follow(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise InputMaterializationError(f"content projection member is not regular: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = path.lstat()
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ) or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino):
            raise InputMaterializationError(
                f"content projection member changed during read: {path}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _verify_content_tree_view(
    provider: ImmutableArtifactBlobProvider | None,
    tree: ContentTree,
    destination: Path,
    root: Path,
) -> tuple[StagedInputFile, ...]:
    mode = destination.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise InputMaterializationError("materialized content tree is not a regular directory")
    expected = {item.relative_path: item.object for item in tree.files}
    observed: dict[str, Path] = {}
    observed_directories: set[str] = set()
    for path in sorted(destination.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or (not stat.S_ISDIR(mode) and not stat.S_ISREG(mode)):
            raise InputMaterializationError(f"unsupported content projection member: {path}")
        if stat.S_ISREG(mode):
            observed[path.relative_to(destination).as_posix()] = path
        elif stat.S_ISDIR(mode):
            observed_directories.add(path.relative_to(destination).as_posix())
    if set(observed) != set(expected) or observed_directories != set(tree.directories):
        raise InputMaterializationError("materialized content tree member set differs")
    records: list[StagedInputFile] = []
    for relative_path, object_ref in expected.items():
        data = observed[relative_path].read_bytes()
        if (
            len(data) != object_ref.size_bytes
            or hashlib.sha256(data).hexdigest() != object_ref.sha256
            or (provider is not None and data != provider.get_object(object_ref))
        ):
            raise InputMaterializationError(f"materialized content object differs: {relative_path}")
        records.append(
            StagedInputFile(
                relative_path=observed[relative_path].relative_to(root).as_posix(),
                sha256=object_ref.sha256,
                size_bytes=object_ref.size_bytes,
            )
        )
    return tuple(records)


def materialize_bundle_inputs(
    bundle: RunBundle,
    *,
    destination_root: Path | str,
    provider_bindings: Sequence[InputProviderRootBinding] = (),
    staged_root_bindings: Sequence[StagedRootSnapshotBinding] = (),
    content_store_root: Path | str | None = None,
) -> tuple[StagedInput, ...]:
    failures, _ = preflight_bundle_input_bindings(
        bundle,
        provider_bindings=provider_bindings,
        staged_root_bindings=staged_root_bindings,
        content_store_root=content_store_root,
    )
    if failures:
        raise InputMaterializationError("; ".join(failures))
    providers = _bound_providers(bundle.resolved_inputs, provider_bindings)
    root = Path(destination_root).expanduser().resolve()
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    staged: list[StagedInput] = []
    for resolved in bundle.resolved_inputs:
        source = resolved.custody
        destination = root / "inputs" / source.target_role
        provider = providers[source.provider_binding]
        try:
            if os.path.lexists(destination):
                raise InputMaterializationError(f"input destination already exists: {destination}")
            authority = source.materializer
            result = materialize_checkpoint_custody_archive(
                provider,
                _artifact_ref(resolved),
                destination,
                expected_parent_ref=authority.expected_parent_ref,
                expected_transaction_root_sha256=authority.expected_transaction_root_sha256,
            )
            expected = _expected_files(
                authority.expected_parent_ref.uri, result.resolved_transaction.manifest.slots
            )
            files = _file_manifest(root, destination, expected)
        except InputMaterializationError:
            raise
        except Exception as exc:
            raise InputMaterializationError(
                f"input {source.target_role!r} materialization failed: {exc}"
            ) from exc
        staged.append(StagedInput(source.target_role, destination, files))
    if staged_root_bindings:
        raise InputMaterializationError(
            "legacy staged roots require explicit offline content-store migration"
        )
    return tuple(staged)


def staged_execution_bindings_for_bundle(
    bundle: RunBundle,
    *,
    inputs_root: Path | str,
) -> StagedExecutionRootBindings:
    """Authenticate materialized roots and project exact execution-context arguments."""
    root = _canonical_materialization_root(inputs_root)
    return _content_execution_bindings(bundle.content_bindings, root)


def _content_execution_bindings(
    bindings: Sequence[ContentBinding],
    inputs_root: Path,
) -> StagedExecutionRootBindings:
    from feedbax.analysis.execution_context import (
        StagedArtifactProviderRootBinding,
        StagedCheckpointCustodyRootBinding,
        StagedManifestRootBinding,
    )

    artifact_specs: dict[str, ImmutableArtifactBlobProviderSpec] = {}
    checkpoint_specs: dict[str, StagedCheckpointCustodySpec] = {}
    artifact_bindings: list[StagedArtifactProviderRootBinding] = []
    manifest_bindings: list[StagedManifestRootBinding] = []
    checkpoint_bindings: list[StagedCheckpointCustodyRootBinding] = []
    resolved_views: dict[str, Path] = {}
    for binding in bindings:
        view = resolved_views.get(binding.tree.tree_id)
        if view is None:
            digest = binding.tree.descriptor.sha256
            descriptor_path = inputs_root / "content" / "descriptors" / digest
            raw = descriptor_path.read_bytes()
            if (
                len(raw) != binding.tree.descriptor.size_bytes
                or hashlib.sha256(raw).hexdigest() != digest
            ):
                raise InputMaterializationError(
                    f"content-tree descriptor identity mismatch: {binding.tree.tree_id}"
                )
            tree = ContentTree.model_validate(
                strict_json_loads(raw.decode("utf-8"), ref=binding.tree.tree_id)
            )
            if raw != canonical_json_v1_bytes(tree.model_dump(mode="json")):
                raise InputMaterializationError(
                    f"content-tree descriptor is not canonical: {binding.tree.tree_id}"
                )
            view = inputs_root / "content" / "trees" / digest
            _verify_content_tree_view(None, tree, view, inputs_root)
            resolved_views[binding.tree.tree_id] = view
        if binding.root_kind == "artifact-provider":
            artifact_specs[binding.binding_name] = ImmutableArtifactBlobProviderSpec()
            artifact_bindings.append(StagedArtifactProviderRootBinding(binding.binding_name, view))
        elif binding.root_kind == "manifest-store":
            manifest_bindings.append(StagedManifestRootBinding(binding.binding_name, view))
        elif binding.root_kind == "checkpoint-custody":
            checkpoint_specs[binding.binding_name] = StagedCheckpointCustodySpec(
                backend=STAGED_CHECKPOINT_CUSTODY_BACKEND
            )
            checkpoint_bindings.append(
                StagedCheckpointCustodyRootBinding(binding.binding_name, view)
            )
        else:
            raise InputMaterializationError(
                f"unsupported content binding root_kind: {binding.root_kind!r}"
            )
    return StagedExecutionRootBindings(
        descriptor=StagedExecutionDescriptor(
            schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
            schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
            artifact_providers=artifact_specs,
            checkpoint_custody=checkpoint_specs,
        ),
        artifact_provider_bindings=tuple(artifact_bindings),
        manifest_root_bindings=tuple(manifest_bindings),
        checkpoint_custody_bindings=tuple(checkpoint_bindings),
    )


def reclaim_materialized_staged_roots(
    bundle: RunBundle,
    *,
    inputs_root: Path | str,
) -> StagedRootReclamation:
    """Verify and reclaim a terminal run's run-local staged-root copies.

    The caller owns the liveness decision. Reclamation is restart-safe: the
    authenticated group is atomically renamed before removal, and a later call
    completes an interrupted removal from the isolated name.
    """
    custody_refs = tuple(custody.custody_ref for custody in bundle.staged_roots)
    if not custody_refs:
        return StagedRootReclamation("not-applicable", (), 0)

    root = _canonical_materialization_root(inputs_root)
    root_identity = _directory_identity(root)
    materialized = root / "staged-roots"
    isolated = root / ".staged-roots-reclaiming"
    marker = root / ".staged-roots-reclaiming.json"
    receipt = root / ".staged-roots-reclaimed.json"
    materialized_exists = os.path.lexists(materialized)
    isolated_exists = os.path.lexists(isolated)
    marker_exists = os.path.lexists(marker)
    receipt_exists = os.path.lexists(receipt)
    if receipt_exists:
        if materialized_exists or isolated_exists or marker_exists:
            raise StagedRootCustodyError(
                "staged-root reclamation receipt conflicts with live reclamation state"
            )
        _verify_staged_root_reclamation_marker(bundle, receipt)
        return StagedRootReclamation("already-reclaimed", custody_refs, 0)
    if materialized_exists and isolated_exists:
        raise StagedRootCustodyError(
            "staged-root materialization and reclamation isolate both exist"
        )
    if materialized_exists and marker_exists:
        raise StagedRootCustodyError(
            "staged-root reclamation marker exists beside live materialization"
        )
    if not materialized_exists and not isolated_exists:
        if marker_exists:
            _verify_staged_root_reclamation_marker(bundle, marker)
            _publish_staged_root_reclamation_receipt(marker, receipt)
            return StagedRootReclamation("already-reclaimed", custody_refs, 0)
        raise StagedRootCustodyError(
            "staged-root materialization is missing without a reclamation receipt"
        )

    target = isolated if isolated_exists else materialized
    reclaimed_bytes = sum(
        record.size_bytes for custody in bundle.staged_roots for record in custody.files
    )
    if marker_exists:
        if target != isolated:
            raise StagedRootCustodyError(
                "staged-root reclamation marker lacks its isolated materialization"
            )
        isolated_identity = _verify_staged_root_reclamation_marker(
            bundle,
            marker,
            isolated=isolated,
        )
    else:
        isolated_identity = _verify_materialized_staged_root_group(
            bundle.staged_roots,
            target,
        )
    if target == materialized and not marker_exists:
        parent_descriptor = os.open(
            root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
        )
        try:
            if _stat_identity(os.fstat(parent_descriptor)) != root_identity:
                raise StagedRootCustodyError("staged-root materialization parent was replaced")
            source_identity = os.stat(
                materialized.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if _stat_identity(source_identity) != isolated_identity:
                raise StagedRootCustodyError(
                    "staged-root materialization changed after verification"
                )
            publish_directory_no_replace(
                parent_descriptor,
                materialized.name,
                isolated.name,
                expected_identity=source_identity,
            )
            renamed_identity = os.stat(
                isolated.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if _stat_identity(source_identity) != _stat_identity(renamed_identity):
                raise StagedRootCustodyError(
                    "staged-root reclamation isolate changed during atomic rename"
                )
        finally:
            os.close(parent_descriptor)
        verified_identity = _verify_materialized_staged_root_group(
            bundle.staged_roots,
            isolated,
        )
        if verified_identity != _stat_identity(renamed_identity):
            raise StagedRootCustodyError(
                "staged-root reclamation isolate changed after atomic rename"
            )
        isolated_identity = verified_identity
    if not marker_exists:
        _require_directory_identity(root, root_identity)
        _require_directory_identity(isolated, isolated_identity)
        _write_staged_root_reclamation_marker(
            bundle,
            marker,
            isolated_identity=isolated_identity,
        )

    _require_directory_identity(root, root_identity)
    _require_directory_identity(isolated, isolated_identity)
    _remove_materialization_tree(
        isolated,
        expected_identity=isolated_identity,
    )
    _publish_staged_root_reclamation_receipt(marker, receipt)
    return StagedRootReclamation("reclaimed", custody_refs, reclaimed_bytes)


def _materialize_staged_roots(
    custodies: Sequence[StagedRootCustody],
    *,
    root: Path,
    bindings: Sequence[StagedRootSnapshotBinding],
) -> tuple[StagedInput, ...]:
    if not custodies:
        return ()
    roots = {(binding.kind, binding.name): binding for binding in bindings}
    inputs_root = root / "inputs"
    final_root = inputs_root / "staged-roots"
    if os.path.lexists(final_root):
        raise StagedRootCustodyError(f"staged-root destination already exists: {final_root}")
    build_root = Path(tempfile.mkdtemp(prefix=".staged-roots-", dir=inputs_root))
    staged: list[StagedInput] = []
    try:
        for custody in custodies:
            key = (custody.root_kind, custody.binding_name)
            destination = build_root / custody.root_kind / custody.binding_name
            result = materialize_staged_root_snapshot(
                custody,
                verify_staged_root_snapshot_binding(roots[key]),
                destination,
            )
            files = tuple(
                StagedInputFile(
                    relative_path=(
                        Path("inputs")
                        / "staged-roots"
                        / custody.root_kind
                        / custody.binding_name
                        / record.relative_path
                    ).as_posix(),
                    sha256=record.sha256,
                    size_bytes=record.size_bytes,
                )
                for record in custody.files
            )
            staged.append(
                StagedInput(
                    target_role=(f"staged-roots/{custody.root_kind}/{custody.binding_name}"),
                    destination=(
                        inputs_root / "staged-roots" / custody.root_kind / custody.binding_name
                    ),
                    files=files,
                )
            )
            verify_staged_root_snapshot(custody, result.root)
            verify_staged_root_snapshot_binding(roots[key])
        parent_descriptor = os.open(
            inputs_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
        )
        try:
            publish_directory_no_replace(
                parent_descriptor,
                build_root.name,
                final_root.name,
                expected_identity=os.stat(
                    build_root.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                ),
            )
        finally:
            os.close(parent_descriptor)
    except Exception:
        if build_root.exists():
            _remove_materialization_tree(build_root)
        raise
    return tuple(staged)


def _verify_materialized_staged_root_group(
    custodies: Sequence[StagedRootCustody],
    root: Path,
) -> tuple[int, int]:
    mode = root.lstat().st_mode
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise StagedRootCustodyError(
            f"staged-root materialization is not a regular directory: {root}"
        )
    root_identity = _directory_identity(root)
    expected = {(custody.root_kind, custody.binding_name): custody for custody in custodies}
    expected_kinds = {kind for kind, _name in expected}
    observed: set[tuple[str, str]] = set()
    for kind_path in root.iterdir():
        kind_mode = kind_path.lstat().st_mode
        if not stat.S_ISDIR(kind_mode) or stat.S_ISLNK(kind_mode):
            raise StagedRootCustodyError(
                f"unsupported staged-root materialization member: {kind_path}"
            )
        if kind_path.name not in expected_kinds:
            raise StagedRootCustodyError(
                f"unexpected staged-root materialization kind: {kind_path.name!r}"
            )
        for binding_path in kind_path.iterdir():
            binding_mode = binding_path.lstat().st_mode
            if not stat.S_ISDIR(binding_mode) or stat.S_ISLNK(binding_mode):
                raise StagedRootCustodyError(
                    f"unsupported staged-root materialization member: {binding_path}"
                )
            key = (kind_path.name, binding_path.name)
            custody = expected.get(key)
            if custody is None:
                raise StagedRootCustodyError(
                    f"unexpected staged-root materialization binding: {key!r}"
                )
            verify_staged_root_snapshot(custody, binding_path)
            observed.add(key)
    missing = sorted(set(expected) - observed)
    if missing:
        raise StagedRootCustodyError(
            f"staged-root materialization lacks custody bindings: {missing!r}"
        )
    _require_directory_identity(root, root_identity)
    return root_identity


def _canonical_materialization_root(root: Path | str) -> Path:
    expanded = Path(root).expanduser()
    absolute = Path(os.path.abspath(expanded))
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as exc:
        raise StagedRootCustodyError(
            f"staged-root materialization parent is unavailable: {expanded}"
        ) from exc
    if resolved != absolute:
        raise StagedRootCustodyError(
            f"staged-root materialization parent must not traverse symlinks: {expanded}"
        )
    mode = absolute.lstat().st_mode
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise StagedRootCustodyError(
            f"staged-root materialization parent is not a regular directory: {absolute}"
        )
    return absolute


def _staged_root_reclamation_marker_payload(
    bundle: RunBundle,
    *,
    isolated_identity: tuple[int, int],
) -> dict[str, object]:
    return {
        "schema_version": _STAGED_ROOT_RECLAMATION_SCHEMA_VERSION,
        "run_set_id": bundle.run_set_id,
        "custodies": [
            custody.model_dump(mode="json", exclude_none=True) for custody in bundle.staged_roots
        ],
        "isolated_directory_identity": list(isolated_identity),
    }


def _write_staged_root_reclamation_marker(
    bundle: RunBundle,
    marker: Path,
    *,
    isolated_identity: tuple[int, int],
) -> None:
    payload = _staged_root_reclamation_marker_payload(
        bundle,
        isolated_identity=isolated_identity,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{marker.name}.",
        suffix=".tmp",
        dir=marker.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
        _fsync_directory(marker.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_staged_root_reclamation_marker(
    bundle: RunBundle,
    marker: Path,
    *,
    isolated: Path | None = None,
) -> tuple[int, int]:
    marker_mode = marker.lstat().st_mode
    if not stat.S_ISREG(marker_mode) or stat.S_ISLNK(marker_mode):
        raise StagedRootCustodyError(
            f"staged-root reclamation marker is not a regular file: {marker}"
        )
    try:
        payload = strict_json_loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StagedRootCustodyError("staged-root reclamation marker is invalid") from exc
    expected_custodies = [
        custody.model_dump(mode="json", exclude_none=True) for custody in bundle.staged_roots
    ]
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _STAGED_ROOT_RECLAMATION_SCHEMA_VERSION
        or payload.get("run_set_id") != bundle.run_set_id
        or payload.get("custodies") != expected_custodies
    ):
        raise StagedRootCustodyError("staged-root reclamation marker differs from bundle custody")
    directory_identity = payload.get("isolated_directory_identity")
    if (
        not isinstance(directory_identity, list)
        or len(directory_identity) != 2
        or any(type(value) is not int for value in directory_identity)
    ):
        raise StagedRootCustodyError(
            "staged-root reclamation marker has invalid directory identity"
        )
    if isolated is not None:
        _require_directory_identity(
            isolated,
            (directory_identity[0], directory_identity[1]),
        )
    return directory_identity[0], directory_identity[1]


def _publish_staged_root_reclamation_receipt(
    marker: Path,
    receipt: Path,
) -> None:
    os.replace(marker, receipt)
    _fsync_directory(receipt.parent)


def _stat_identity(observed: os.stat_result) -> tuple[int, int]:
    return observed.st_dev, observed.st_ino


def _directory_identity(root: Path) -> tuple[int, int]:
    observed = root.lstat()
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise StagedRootCustodyError(
            f"staged-root reclamation path is not a regular directory: {root}"
        )
    return _stat_identity(observed)


def _require_directory_identity(root: Path, expected: tuple[int, int]) -> None:
    if _directory_identity(root) != expected:
        raise StagedRootCustodyError(f"staged-root reclamation directory was replaced: {root}")


def _fsync_directory(root: Path) -> None:
    descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_materialization_tree(
    root: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    parent_descriptor = os.open(
        root.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
    )
    directory_descriptor: int | None = None
    try:
        before = os.stat(
            root.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        identity = _stat_identity(before)
        if not stat.S_ISDIR(before.st_mode) or (
            expected_identity is not None and identity != expected_identity
        ):
            raise StagedRootCustodyError(f"staged-root removal target identity changed: {root}")
        directory_descriptor = os.open(
            root.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        if _stat_identity(os.fstat(directory_descriptor)) != identity:
            raise StagedRootCustodyError(f"staged-root removal target changed before open: {root}")
        _remove_materialization_directory_contents(directory_descriptor)
        current = os.stat(
            root.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _stat_identity(current) != identity:
            raise StagedRootCustodyError(f"staged-root removal target was replaced: {root}")
        os.rmdir(root.name, dir_fd=parent_descriptor)
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        os.close(parent_descriptor)


def _remove_materialization_directory_contents(directory_descriptor: int) -> None:
    os.fchmod(directory_descriptor, 0o700)
    for entry in sorted(
        os.scandir(directory_descriptor),
        key=lambda item: os.fsencode(item.name),
    ):
        before = entry.stat(follow_symlinks=False)
        identity = _stat_identity(before)
        if stat.S_ISDIR(before.st_mode):
            child_descriptor = os.open(
                entry.name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            try:
                if _stat_identity(os.fstat(child_descriptor)) != identity:
                    raise StagedRootCustodyError(
                        "staged-root removal directory changed before open"
                    )
                _remove_materialization_directory_contents(child_descriptor)
                current = os.stat(
                    entry.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if _stat_identity(current) != identity:
                    raise StagedRootCustodyError("staged-root removal directory was replaced")
                os.rmdir(entry.name, dir_fd=directory_descriptor)
            finally:
                os.close(child_descriptor)
        elif stat.S_ISREG(before.st_mode):
            descriptor = os.open(
                entry.name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            try:
                if _stat_identity(os.fstat(descriptor)) != identity:
                    raise StagedRootCustodyError("staged-root removal file changed before open")
            finally:
                os.close(descriptor)
            current = os.stat(
                entry.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if _stat_identity(current) != identity:
                raise StagedRootCustodyError("staged-root removal file was replaced")
            os.unlink(entry.name, dir_fd=directory_descriptor)
        else:
            raise StagedRootCustodyError(f"unsupported staged-root removal member: {entry.name!r}")


def _bound_providers(
    inputs: Sequence[ResolvedAssemblyInput], bindings: Sequence[InputProviderRootBinding]
) -> Mapping[str, ImmutableArtifactBlobProvider]:
    roots: dict[str, Path] = {}
    for binding in bindings:
        validate_staged_binding_name(binding.name)
        if binding.name in roots:
            raise InputMaterializationError(f"duplicate provider binding: {binding.name!r}")
        roots[binding.name] = Path(binding.root)
    specs = {item.custody.provider_binding: item.custody.provider for item in inputs}
    missing, extra = sorted(set(specs) - set(roots)), sorted(set(roots) - set(specs))
    if missing or extra:
        raise InputMaterializationError(
            f"input provider bindings differ; missing={missing!r} unexpected={extra!r}"
        )
    return {
        name: open_immutable_artifact_blob_provider(spec, explicit_root=roots[name])
        for name, spec in specs.items()
    }


def _artifact_ref(resolved: ResolvedAssemblyInput) -> ArtifactRef:
    source, artifact = resolved.custody, resolved.custody.artifact
    return ArtifactRef(
        role="training_checkpoint_custody_archive",
        logical_name=f"{source.materializer.expected_parent_ref.id}.checkpoint-custody.tar.gz",
        artifact_id=artifact.artifact_id,
        sha256=artifact.sha256,
        media_type=artifact.media_type,
        size_bytes=artifact.size_bytes,
        storage_backend=artifact.storage_backend,
        uri=artifact.artifact_id,
    )


def _expected_files(uri: str | None, slots: Sequence[object]) -> set[str]:
    assert uri is not None
    parent = PurePosixPath(uri).parent
    return {
        "latest.json",
        uri,
        str(parent / CHECKPOINT_SET_NAME),
        *(str(parent / slot.relative_path) for slot in slots),
    }


def _file_manifest(
    root: Path, destination: Path, expected: set[str]
) -> tuple[StagedInputFile, ...]:
    records, actual = [], set()
    for path in sorted(destination.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or (not stat.S_ISDIR(mode) and not stat.S_ISREG(mode)):
            raise InputMaterializationError(f"unsupported materialized object: {path}")
        if path.is_file():
            data = path.read_bytes()
            actual.add(path.relative_to(destination).as_posix())
            records.append(
                StagedInputFile(
                    path.relative_to(root).as_posix(), hashlib.sha256(data).hexdigest(), len(data)
                )
            )
    if actual != expected:
        raise InputMaterializationError(
            f"materialized file set differs; missing={sorted(expected - actual)!r} "
            f"unexpected={sorted(actual - expected)!r}"
        )
    return tuple(records)
