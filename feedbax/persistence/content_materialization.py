"""Authenticate Store content and publish independent local tree views."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from feedbax._secure_fs import rename_no_replace
from feedbax.contracts.artifact_custody import (
    ArtifactBlobCustodyError,
    ContentBinding,
    ContentLeaseRecord,
    ContentObjectRef,
    ContentRetentionTarget,
    ContentTree,
    ContentTreeRef,
)
from feedbax.contracts.staged_execution import validate_staged_binding_name
from feedbax.persistence.artifact_custody import (
    ImmutableArtifactBlobProvider,
    copy_content_file,
    verify_content_file,
)


class InputMaterializationError(ValueError):
    pass


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

    tree_claims = _content_tree_claims(ordered)
    objects = {ref.descriptor.sha256: ref.descriptor for ref in tree_claims.values()}
    descriptor_digests = set(objects)
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
    trees = tuple((ref, provider.get_tree(ref)) for ref in tree_claims.values())
    for _, tree in trees:
        for member in tree.files:
            _register_object_claim(objects, member.object)
    for digest in sorted(objects.keys() - descriptor_digests):
        provider.verify_object(objects[digest])
    return ContentInputProjection(
        bindings=ordered,
        trees=trees,
        objects=tuple(objects[key] for key in sorted(objects)),
    )


def _register_object_claim(claims: dict[str, ContentObjectRef], ref: ContentObjectRef) -> None:
    if claims.setdefault(ref.sha256, ref) != ref:
        raise InputMaterializationError(
            f"content object metadata conflicts for digest {ref.sha256}"
        )


def _content_tree_claims(bindings: Sequence[ContentBinding]) -> dict[str, ContentTreeRef]:
    trees: dict[str, ContentTreeRef] = {}
    for binding in bindings:
        ref = binding.tree
        if trees.setdefault(ref.tree_id, ref) != ref:
            raise InputMaterializationError(f"content tree metadata conflicts for {ref.tree_id}")
    return dict(sorted(trees.items()))


def publish_content_file(
    source: Path,
    destination: Path,
    object_ref: ContentObjectRef,
    *,
    verify_source_on_reuse: bool = False,
) -> None:
    """Publish one independent file or authenticate the existing winner."""
    if os.path.lexists(destination):
        if verify_source_on_reuse:
            verify_content_file(source, object_ref)
        verify_content_file(destination, object_ref)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".content-object-", dir=destination.parent) as staging:
        staged = Path(staging) / "object"
        copy_content_file(source, staged, object_ref)
        parent_descriptor = os.open(
            destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            rename_no_replace(
                staged.relative_to(destination.parent).as_posix(),
                destination.name,
                source_dir_fd=parent_descriptor,
                destination_dir_fd=parent_descriptor,
                error_factory=InputMaterializationError,
                context="content file publication",
            )
        except FileExistsError:
            verify_content_file(destination, object_ref)
        finally:
            os.close(parent_descriptor)


def materialize_content_input_projection(
    projection: ContentInputProjection,
    provider: ImmutableArtifactBlobProvider,
    *,
    destination_root: Path | str,
) -> tuple[StagedInput, ...]:
    """Publish independent execution files from canonical content objects."""
    root = Path(destination_root).expanduser().resolve()
    materialized = materialize_content_tree_views(
        projection.trees, destination_root=root, source_for=provider.object_path
    )
    return tuple(
        StagedInput(
            target_role=f"content/{binding.root_kind}/{binding.binding_name}",
            destination=materialized[binding.tree.tree_id][0],
            files=materialized[binding.tree.tree_id][1],
        )
        for binding in projection.bindings
    )


def materialize_content_tree_views(
    trees: Sequence[tuple[ContentTreeRef, ContentTree]],
    *,
    destination_root: Path,
    source_for: Callable[[ContentObjectRef], Path],
    tree_root_mode: int = 0o700,
) -> dict[str, tuple[Path, tuple[StagedInputFile, ...]]]:
    """Publish each unique tree with independent writable files and descriptors."""
    destination_root.mkdir(parents=True, exist_ok=True)
    descriptors_root = destination_root.parent / "descriptors"
    materialized: dict[str, tuple[Path, tuple[StagedInputFile, ...]]] = {}
    for tree_ref, tree in trees:
        descriptor = tree_ref.descriptor
        publish_content_file(
            source_for(descriptor),
            descriptors_root / descriptor.sha256,
            descriptor,
        )
        destination = destination_root / descriptor.sha256
        if os.path.lexists(destination):
            _verify_content_tree_view(tree, destination)
        else:
            build = Path(tempfile.mkdtemp(prefix=".content-tree-", dir=destination_root))
            try:
                for directory in tree.directories:
                    (build / directory).mkdir()
                for member in tree.files:
                    copy_content_file(
                        source_for(member.object),
                        build / member.relative_path,
                        member.object,
                    )
                if tree_root_mode != 0o700:
                    build.chmod(tree_root_mode)
                expected = build.lstat()
                parent_descriptor = os.open(
                    destination_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                try:
                    current = os.stat(build.name, dir_fd=parent_descriptor, follow_symlinks=False)
                    if not stat.S_ISDIR(current.st_mode) or _stat_identity(current) != _stat_identity(expected):
                        raise InputMaterializationError("content projection staging directory was replaced")
                    rename_no_replace(
                        build.name, destination.name,
                        source_dir_fd=parent_descriptor,
                        destination_dir_fd=parent_descriptor,
                        error_factory=InputMaterializationError,
                        context="content tree publication",
                    )
                finally:
                    os.close(parent_descriptor)
            except FileExistsError:
                _verify_content_tree_view(tree, destination)
            finally:
                if build.exists():
                    _remove_materialization_tree(build)
        files = tuple(
            StagedInputFile(
                relative_path=(destination / member.relative_path)
                .relative_to(destination_root)
                .as_posix(),
                sha256=member.object.sha256,
                size_bytes=member.object.size_bytes,
            )
            for member in tree.files
        )
        materialized[tree_ref.tree_id] = destination, files
    return materialized


def _verify_content_tree_view(tree: ContentTree, destination: Path) -> None:
    if not stat.S_ISDIR(destination.lstat().st_mode):
        raise InputMaterializationError("materialized content tree is not a regular directory")
    expected = {item.relative_path: item.object for item in tree.files}
    observed: dict[str, Path] = {}
    observed_directories: set[str] = set()
    for path in destination.rglob("*"):
        mode = path.lstat().st_mode
        relative_path = path.relative_to(destination).as_posix()
        if stat.S_ISREG(mode):
            observed[relative_path] = path
        elif stat.S_ISDIR(mode):
            observed_directories.add(relative_path)
        else:
            raise InputMaterializationError(f"unsupported content projection member: {path}")
    if set(observed) != set(expected) or observed_directories != set(tree.directories):
        raise InputMaterializationError("materialized content tree member set differs")
    for relative_path, object_ref in expected.items():
        try:
            verify_content_file(observed[relative_path], object_ref)
        except ArtifactBlobCustodyError as exc:
            raise InputMaterializationError(
                f"materialized content object differs: {relative_path}: {exc}"
            ) from exc


def _stat_identity(observed: os.stat_result) -> tuple[int, int]:
    return observed.st_dev, observed.st_ino


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
            raise InputMaterializationError(f"content projection removal target identity changed: {root}")
        directory_descriptor = os.open(
            root.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        if _stat_identity(os.fstat(directory_descriptor)) != identity:
            raise InputMaterializationError(f"content projection removal target changed before open: {root}")
        _remove_materialization_directory_contents(directory_descriptor)
        current = os.stat(
            root.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _stat_identity(current) != identity:
            raise InputMaterializationError(f"content projection removal target was replaced: {root}")
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
                    raise InputMaterializationError(
                        "content projection removal directory changed before open"
                    )
                _remove_materialization_directory_contents(child_descriptor)
                current = os.stat(
                    entry.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if _stat_identity(current) != identity:
                    raise InputMaterializationError("content projection removal directory was replaced")
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
                    raise InputMaterializationError("content projection removal file changed before open")
            finally:
                os.close(descriptor)
            current = os.stat(
                entry.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if _stat_identity(current) != identity:
                raise InputMaterializationError("content projection removal file was replaced")
            os.unlink(entry.name, dir_fd=directory_descriptor)
        else:
            raise InputMaterializationError(f"unsupported content projection removal member: {entry.name!r}")
