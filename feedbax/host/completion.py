"""Exact file evidence for an authenticated local completion."""

import hashlib
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from feedbax._secure_fs import (
    SecurePathRigor,
    open_directory_chain,
    open_existing_directory,
    open_existing_file,
)
from feedbax.contracts.artifact_custody import CONTENT_OBJECT_REF_PREFIX, ContentObjectRef
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider, content_object_relative_path
from feedbax.persistence.store_damage import require_sound


class CompletionFileError(ValueError):
    """A pinned completion file is absent, malformed or changed."""


def train_source_slots(launch: dict) -> frozenset[str] | None:
    if launch.get("warm_entrypoint") != "feedbax.training_host_worker:main":
        return None
    payload = launch["payload"]
    if not (payload.get("canonical_request") or {}).get("checkpoint"):
        return None
    value = payload["spec"]["method_payload"]["payload"].get("checkpoint_initialization")
    if not isinstance(value, dict):
        return None
    from feedbax.contracts.studio_training import StudioTrainContinuation

    try:
        return StudioTrainContinuation.model_validate(value).source_slots
    except ValueError as error:
        raise CompletionFileError("Invalid Train checkpoint carryover policy") from error


def _checkpoint_slot_paths(binding: dict) -> dict[str, Path]:
    from feedbax.contracts.strict_json import strict_json_loads

    root = Path(binding["manifest_root"])
    checkpoint = binding["checkpoint"]
    pointer = checkpoint["pointer"]
    manifest = Path(checkpoint["checkpoint_root"]) / pointer["manifest_relative_path"]
    with completion_file(root, manifest.relative_to(root).as_posix()) as (stream, _):
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != pointer["manifest_sha256"]:
        raise CompletionFileError("Train checkpoint manifest changed before source verification")
    return {
        slot["slot"]: (manifest.parent / slot["relative_path"]).relative_to(root)
        for slot in strict_json_loads(data)["slots"]
    }


@contextmanager
def completion_file(root: Path, relative: str) -> Iterator[tuple[BinaryIO, int]]:
    if not isinstance(relative, str) or any(
        part in {"", ".", ".."} or "\\" in part or "\x00" in part for part in relative.split("/")
    ):
        raise CompletionFileError(f"Invalid relative completion path: {relative!r}")
    parts = relative.split("/")
    with ExitStack() as stack:
        root_chain = open_directory_chain(
            root, create=False, error_factory=CompletionFileError, context="completion root"
        )
        for _, fd, _ in root_chain:
            stack.callback(os.close, fd)
        fd = root_chain[-1][1]
        for component in parts[:-1]:
            fd = open_existing_directory(
                component,
                error_factory=CompletionFileError,
                context="completion directory",
                dir_fd=fd,
            )
            stack.callback(os.close, fd)
        member, info = open_existing_file(
            parts[-1],
            rigor=SecurePathRigor.REGULAR_FILE_IDENTITY,
            error_factory=CompletionFileError,
            context="completion file",
            dir_fd=fd,
        )
        stream = stack.enter_context(os.fdopen(member, "rb"))
        yield stream, info.st_size
        after = os.fstat(stream.fileno())
        named = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
        if (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino) or (
            after.st_size,
            after.st_mtime_ns,
        ) != (info.st_size, info.st_mtime_ns):
            raise CompletionFileError(f"Completion file changed while open: {relative}")


def verify_completion_files(
    binding: dict, *, train_slots: frozenset[str] | None = None, checkpoint_model: bool = False,
) -> None:
    """Rehash the producer-attested immutable byte closure before delivery."""
    try:
        root = Path(binding["manifest_root"])
        files = binding["files"]
        if not root.is_absolute() or not isinstance(files, list) or not files:
            raise CompletionFileError("Completion requires an absolute root and file evidence.")
        store = ImmutableArtifactBlobProvider(root)
        slots = _checkpoint_slot_paths(binding) if checkpoint_model or train_slots is not None else {}
        selected = train_slots if train_slots is not None else frozenset({"model"})
        ignored = {
            path.as_posix() for name, path in slots.items()
            if train_slots is not None and name not in selected
        }
        retained = {path.as_posix() for name, path in slots.items() if name in selected}
        model_path = slots.get("model")
        model_root = (
            model_path.parents[3]
            if model_path is not None and model_path.parts[-4:-2] == ("artifacts", "sha256")
            else None
        )
        paths = set()
        for item in files:
            relative = item["path"]
            if relative in paths or type(item["size_bytes"]) is not int:
                raise CompletionFileError("Invalid or duplicate completion file evidence.")
            paths.add(relative)
            if relative in ignored:
                continue
            digest = item["sha256"]
            canonical = content_object_relative_path(digest).as_posix()
            model_member = model_root is not None and relative == (model_root / canonical).as_posix()
            if relative == canonical or model_member or relative in retained:
                ref = ContentObjectRef(
                    object_id=CONTENT_OBJECT_REF_PREFIX + digest,
                    sha256=digest,
                    size_bytes=item["size_bytes"],
                )
                if relative in retained or (model_member and "model" in selected):
                    require_sound(store, ref)
                if relative == canonical or model_member:
                    custody = ImmutableArtifactBlobProvider(root / model_root) if model_member else store
                    require_sound(custody, ref)
                    custody.verify_object(ref, verifier="Host input")
                    continue
            with completion_file(root, relative) as (stream, size):
                if (
                    size != item["size_bytes"]
                    or hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]
                ):
                    raise CompletionFileError(f"Completion file changed: {relative}")
    except CompletionFileError:
        raise
    except (KeyError, TypeError, OSError, ValueError) as error:
        raise CompletionFileError(f"Cannot verify completion files: {error}") from error
