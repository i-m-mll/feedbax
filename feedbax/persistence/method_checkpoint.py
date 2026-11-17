"""Authenticate and transfer portable Train checkpoints without numerical execution."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from feedbax._secure_fs import (
    SecurePathRigor,
    open_directory_chain,
    open_existing_file,
    validate_opened_path,
)
from feedbax.contracts.artifact_custody import (
    CONTENT_OBJECT_REF_PREFIX,
    ArtifactBlobProvider,
    ContentBinding,
    ContentObjectRef,
)
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.method_checkpoint import (
    CheckpointLatestPointer,
    MethodCheckpointManifest,
    method_checkpoint_extra_path,
)
from feedbax.contracts.strict_json import strict_model_validate_json
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.content_materialization import (
    acquire_content_projection_lease,
    compile_content_input_projection,
    materialize_content_input_projection,
)
from feedbax.persistence.model_tree import (
    MODEL_CHECKPOINT_STORE,
    checkpoint_model_object_path,
    read_model_tree,
)
from feedbax.training.checkpoint_errors import CheckpointCustodyError


LATEST_NAME = "latest.json"


def _content_kind(include_extra: bool) -> str:
    return "checkpoint-custody" if include_extra else "checkpoint-model-custody"


def _checkpoint_ref(raw: bytes) -> ContentObjectRef:
    digest = sha256_bytes(raw)
    return ContentObjectRef(
        object_id=CONTENT_OBJECT_REF_PREFIX + digest, sha256=digest, size_bytes=len(raw),
    )


def method_checkpoint_pointer(manifest: MethodCheckpointManifest, raw: bytes) -> CheckpointLatestPointer:
    return CheckpointLatestPointer(
        run_id=manifest.run_id,
        transaction_id=manifest.transaction_id,
        manifest_sha256=sha256_bytes(raw),
        transaction_root_sha256=manifest.state_sha256,
        completed_updates=manifest.clock.completed_updates,
        final=manifest.final,
    )


def read_checkpoint_file(path: Path, expected: ContentObjectRef | None = None) -> bytes:
    with ExitStack() as stack:
        chain = open_directory_chain(
            path.parent, create=False, error_factory=CheckpointCustodyError, context="checkpoint directory",
        )
        for _, fd, _ in chain:
            stack.callback(os.close, fd)
        parent_fd = chain[-1][1]
        fd, info = open_existing_file(
            path.name,
            dir_fd=parent_fd,
            rigor=SecurePathRigor.REGULAR_FILE_IDENTITY,
            error_factory=CheckpointCustodyError,
            context="checkpoint file",
        )
        stream = stack.enter_context(os.fdopen(fd, "rb"))
        if expected is not None and info.st_size != expected.size_bytes:
            raise CheckpointCustodyError(f"{path}: checkpoint file size changed")
        raw = stream.read()
        after = validate_opened_path(
            path.name, fd, dir_fd=parent_fd, rigor=SecurePathRigor.REGULAR_FILE_IDENTITY,
            expected_identity=(info.st_dev, info.st_ino),
            error_factory=CheckpointCustodyError, context="checkpoint file",
        )
        if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise CheckpointCustodyError(f"{path}: checkpoint file changed while reading")
        if expected is not None and _checkpoint_ref(raw) != expected:
            raise CheckpointCustodyError(f"{path}: checkpoint file content changed")
        return raw


def parse_method_checkpoint_manifest(raw: bytes, pointer: CheckpointLatestPointer) -> MethodCheckpointManifest:
    manifest = strict_model_validate_json(MethodCheckpointManifest, raw)
    if raw != canonical_json_v2_bytes(manifest.model_dump(mode="json")):
        raise CheckpointCustodyError("Method checkpoint manifest is not canonical")
    if pointer != method_checkpoint_pointer(manifest, raw):
        raise CheckpointCustodyError("Method checkpoint manifest differs from its exact published pointer")
    return manifest


def _local_manifest(
    root: Path, pointer: CheckpointLatestPointer,
) -> tuple[Path, MethodCheckpointManifest, ContentObjectRef]:
    relative = Path(pointer.manifest_relative_path)
    raw = read_checkpoint_file(root / relative)
    return relative, parse_method_checkpoint_manifest(raw, pointer), _checkpoint_ref(raw)


def read_method_checkpoint_manifest(root: Path, pointer: CheckpointLatestPointer) -> MethodCheckpointManifest:
    return _local_manifest(root, pointer)[1]


def _member_files(
    relative: Path, manifest: MethodCheckpointManifest, manifest_ref: ContentObjectRef,
    model_store: ArtifactBlobProvider, *, include_extra: bool,
) -> dict[Path, ContentObjectRef]:
    model = read_model_tree(model_store, manifest.model.tree, manifest.model.encoding)
    refs = (manifest.model.tree.descriptor, *model.members.values())
    files = {
        relative: manifest_ref,
        **{relative.parent / checkpoint_model_object_path(ref): ref for ref in refs},
    }
    if include_extra:
        files[relative.parent / method_checkpoint_extra_path(manifest.extra)] = manifest.extra.object
    return files


def method_checkpoint_files(
    root: Path, pointer: CheckpointLatestPointer, *, include_extra: bool,
) -> dict[Path, ContentObjectRef]:
    relative, manifest, manifest_ref = _local_manifest(root, pointer)
    return _member_files(
        relative, manifest, manifest_ref,
        ImmutableArtifactBlobProvider(root / relative.parent / MODEL_CHECKPOINT_STORE),
        include_extra=include_extra,
    )


def bind_method_checkpoint_content(
    root: Path, pointer: CheckpointLatestPointer, provider: ArtifactBlobProvider, *,
    include_extra: bool = True,
) -> ContentBinding:
    files = method_checkpoint_files(root, pointer, include_extra=include_extra)
    members = {
        relative.as_posix(): provider.store_object(read_checkpoint_file(root / relative, ref), expected=ref)
        for relative, ref in files.items()
    }
    members[LATEST_NAME] = provider.store_object(canonical_json_v2_bytes(pointer.model_dump(mode="json")))
    kind = _content_kind(include_extra)
    tree, _ = provider.publish_rooted_tree(
        members, root_id=f"{kind}:{pointer.transaction_id}", root_kind="logical-ref",
    )
    return ContentBinding(binding_name=pointer.transaction_id, root_kind=kind, tree=tree)


def verify_method_checkpoint_content(
    store: ImmutableArtifactBlobProvider, pointer: CheckpointLatestPointer, binding: ContentBinding, *,
    include_extra: bool = True,
) -> MethodCheckpointManifest:
    if binding.binding_name != pointer.transaction_id or binding.root_kind != _content_kind(include_extra):
        raise CheckpointCustodyError("Checkpoint binding has the wrong occurrence or state projection")
    members = {file.relative_path: file.object for file in store.get_tree(binding.tree).files}
    latest = members.get(LATEST_NAME)
    if latest is None or store.get_object(latest) != canonical_json_v2_bytes(pointer.model_dump(mode="json")):
        raise CheckpointCustodyError("Checkpoint binding differs from its exact published pointer")
    relative = Path(pointer.manifest_relative_path)
    manifest_ref = members.get(relative.as_posix())
    if manifest_ref is None:
        raise CheckpointCustodyError("Checkpoint binding has no transaction manifest")
    manifest = parse_method_checkpoint_manifest(store.get_object(manifest_ref), pointer)
    expected = _member_files(relative, manifest, manifest_ref, store, include_extra=include_extra)
    if members.keys() != {LATEST_NAME, *(path.as_posix() for path in expected)}:
        raise CheckpointCustodyError("Checkpoint binding has an incomplete or unexpected member closure")
    for path, ref in expected.items():
        if members[path.as_posix()] != ref:
            raise CheckpointCustodyError(f"{path}: checkpoint binding names different content")
        store.verify_object(ref)
    return manifest


@contextmanager
def materialize_checkpoint_content(
    binding: ContentBinding, provider: ArtifactBlobProvider, *,
    owner: str, include_extra: bool = True,
) -> Iterator[Path]:
    if binding.root_kind != _content_kind(include_extra):
        raise CheckpointCustodyError("Checkpoint binding has the wrong state projection")
    transport = binding.model_copy(update={"root_kind": "checkpoint-custody"})
    projection = compile_content_input_projection((transport,), provider)
    lease_id = f"{owner}:{uuid4().hex}"
    acquire_content_projection_lease(
        projection, provider, lease_id=lease_id, owner=owner,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    try:
        with TemporaryDirectory(prefix="feedbax-checkpoint-content-") as temporary:
            (view,) = materialize_content_input_projection(
                projection, provider, destination_root=Path(temporary) / "trees",
            )
            yield view.destination
    finally:
        provider.release_lease(lease_id)
