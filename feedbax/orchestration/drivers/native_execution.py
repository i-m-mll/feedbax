"""Native executor context injection shared by orchestration drivers."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from feedbax.orchestration.bundle import ResolvedAssemblyInput, RunBundle, RunRowSpec
from feedbax.orchestration.native_execution import is_native_training_command
from feedbax.contracts.worker import NATIVE_TRAINING_COLLECTION_OUTPUTS
from feedbax.training.diagnostics import NativeExecutionProducerContext


class NativeExecutionContextError(ValueError):
    """Raised when a native row cannot receive one canonical producer context."""


SECURE_CHECKPOINT_SEED_SCRIPT = r"""
import ctypes
import errno
import hashlib
import json
import os
import pathlib
import runpy
import stat
import sys

strict_json_loads = runpy.run_path("feedbax/contracts/strict_json.py")["strict_json_loads"]

source, attempt, target, authority_json = sys.argv[1:]
authority = strict_json_loads(authority_json)
required_options = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
missing_options = [name for name in required_options if not hasattr(os, name)]
required_dir_fd = (os.open, os.mkdir, os.stat)
missing_dir_fd = [function.__name__ for function in required_dir_fd if function not in os.supports_dir_fd]
if os.listdir not in os.supports_fd or missing_options or missing_dir_fd:
    raise RuntimeError(
        "secure checkpoint clone capabilities are unavailable; "
        f"missing_options={missing_options!r} missing_dir_fd={missing_dir_fd!r} "
        f"listdir_fd={os.listdir in os.supports_fd}"
    )
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW

def identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )

def object_identity(value):
    return (value.st_dev, value.st_ino, value.st_mode)

def require_identity(expected, descriptor, parent_descriptor, name):
    descriptor_identity = identity(os.fstat(descriptor))
    path_identity = identity(os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False))
    if descriptor_identity != expected or path_identity != expected:
        raise RuntimeError(f"source entry changed during checkpoint clone: {name}")

def clone_directory(source_descriptor, destination_descriptor):
    directory_identity = identity(os.fstat(source_descriptor))
    names = sorted(os.listdir(source_descriptor))
    for name in names:
        if name in {"", ".", ".."} or "/" in name or "\\" in name:
            raise RuntimeError(f"unsafe checkpoint entry name: {name!r}")
        before = os.stat(name, dir_fd=source_descriptor, follow_symlinks=False)
        expected = identity(before)
        if stat.S_ISDIR(before.st_mode):
            child_source = os.open(name, flags, dir_fd=source_descriptor)
            try:
                require_identity(expected, child_source, source_descriptor, name)
                os.mkdir(name, 0o700, dir_fd=destination_descriptor)
                child_destination = os.open(name, flags, dir_fd=destination_descriptor)
                try:
                    clone_directory(child_source, child_destination)
                finally:
                    os.close(child_destination)
                require_identity(expected, child_source, source_descriptor, name)
            finally:
                os.close(child_source)
        elif stat.S_ISREG(before.st_mode):
            child_source = os.open(
                name,
                os.O_RDONLY
                | os.O_NOFOLLOW
                | os.O_NONBLOCK,
                dir_fd=source_descriptor,
            )
            try:
                require_identity(expected, child_source, source_descriptor, name)
                child_destination = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=destination_descriptor,
                )
                try:
                    while chunk := os.read(child_source, 1024 * 1024):
                        view = memoryview(chunk)
                        while view:
                            written = os.write(child_destination, view)
                            view = view[written:]
                    os.fsync(child_destination)
                finally:
                    os.close(child_destination)
                require_identity(expected, child_source, source_descriptor, name)
            finally:
                os.close(child_source)
        else:
            raise RuntimeError(f"unsafe checkpoint source entry: {name!r}")
    if sorted(os.listdir(source_descriptor)) != names or identity(os.fstat(source_descriptor)) != directory_identity:
        raise RuntimeError("source directory changed during checkpoint clone")

def safe_parts(value, context):
    path = pathlib.PurePosixPath(value)
    parts = path.parts
    if path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise RuntimeError(f"unsafe {context}: {value!r}")
    return parts

governed_identities = {}
directory_listings = {}

def bind_identity(path, value):
    observed = identity(value)
    prior = governed_identities.setdefault(path, observed)
    if prior != observed:
        raise RuntimeError(f"checkpoint governed pathname changed: {path}")

def read_regular(parent_descriptor, parts):
    descriptor = parent_descriptor
    opened = []
    prefix = []
    try:
        for part in parts[:-1]:
            descriptor = os.open(part, flags, dir_fd=descriptor)
            opened.append(descriptor)
            prefix.append(part)
            opened_info = os.fstat(descriptor)
            path_info = os.stat(part, dir_fd=opened[-2] if len(opened) > 1 else parent_descriptor,
                                follow_symlinks=False)
            if identity(opened_info) != identity(path_info):
                raise RuntimeError(f"checkpoint governed directory changed: {'/'.join(prefix)}")
            bind_identity("/".join(prefix), opened_info)
        leaf = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=descriptor,
        )
        try:
            info = os.fstat(leaf)
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"checkpoint governed member is not regular: {'/'.join(parts)}")
            chunks = []
            while chunk := os.read(leaf, 1024 * 1024):
                chunks.append(chunk)
            if identity(os.fstat(leaf)) != identity(info):
                raise RuntimeError(f"checkpoint governed member changed: {'/'.join(parts)}")
            bind_identity("/".join(parts), info)
            return b"".join(chunks), info.st_size
        finally:
            os.close(leaf)
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)

def authenticate_checkpoint(root_descriptor):
    parent_ref = authority["expected_parent_ref"]
    expected_root = authority["expected_transaction_root_sha256"]
    manifest_parts = safe_parts(parent_ref["uri"], "checkpoint manifest path")
    latest_bytes, _ = read_regular(root_descriptor, ("latest.json",))
    manifest_bytes, _ = read_regular(root_descriptor, manifest_parts)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    expected_manifest_sha = parent_ref["metadata"]["manifest_sha256"]
    if manifest_sha != expected_manifest_sha:
        raise RuntimeError("checkpoint manifest digest differs from custody authority")
    latest = strict_json_loads(latest_bytes)
    expected_latest = {
        "transaction_id": parent_ref["id"],
        "manifest_relative_path": parent_ref["uri"],
        "manifest_sha256": expected_manifest_sha,
        "transaction_root_sha256": expected_root,
    }
    if any(latest.get(key) != value for key, value in expected_latest.items()):
        raise RuntimeError("checkpoint latest pointer differs from custody authority")
    manifest = strict_json_loads(manifest_bytes)
    if (
        manifest.get("kind") != "TrainingCheckpointTransactionManifest"
        or manifest.get("transaction_id") != parent_ref["id"]
        or manifest.get("content_integrity_digest", {}).get("transaction_root_sha256")
        != expected_root
    ):
        raise RuntimeError("checkpoint transaction manifest differs from custody authority")
    transaction_parts = manifest_parts[:-1]
    checkpoint_set_parts = (*transaction_parts, "checkpoint-set.json")
    checkpoint_set_bytes, _ = read_regular(
        root_descriptor, checkpoint_set_parts
    )
    checkpoint_set = strict_json_loads(checkpoint_set_bytes)
    checkpoint_transaction = checkpoint_set.get("transaction", {})
    checkpoint_transaction_bytes = checkpoint_transaction.get("bytes", {})
    if (
        checkpoint_set.get("schema_id") != "feedbax.checkpoint_set"
        or checkpoint_set.get("schema_version") != "feedbax.checkpoint_set.v1"
        or checkpoint_transaction.get("domain") != "checkpoint_transaction"
        or checkpoint_transaction.get("identity") != parent_ref["id"]
        or checkpoint_transaction_bytes.get("digest") != manifest_sha
        or checkpoint_transaction_bytes.get("size_bytes") != len(manifest_bytes)
    ):
        raise RuntimeError(
            "checkpoint set does not identify the authenticated transaction manifest"
        )
    expected_files = {
        "latest.json",
        "/".join(manifest_parts),
        "/".join(checkpoint_set_parts),
    }
    for slot in manifest.get("slots", []):
        relative_parts = safe_parts(slot["relative_path"], "checkpoint slot path")
        slot_parts = (*transaction_parts, *relative_parts)
        slot_bytes, slot_size = read_regular(root_descriptor, slot_parts)
        if slot_size != slot["size_bytes"] or hashlib.sha256(slot_bytes).hexdigest() != slot["sha256"]:
            raise RuntimeError(f"checkpoint slot digest or size differs: {slot.get('slot')!r}")
        expected_files.add("/".join(slot_parts))
    actual_files = set()
    expected_directories = {""}
    for path in expected_files:
        parts = path.split("/")
        expected_directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
    def walk(descriptor, prefix=""):
        names = sorted(os.listdir(descriptor))
        directory_listings[prefix] = names
        for name in names:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(info.st_mode):
                if relative not in expected_directories:
                    raise RuntimeError(f"unexpected checkpoint directory: {relative}")
                child = os.open(name, flags, dir_fd=descriptor)
                try:
                    if identity(os.fstat(child)) != identity(info):
                        raise RuntimeError(f"checkpoint governed directory changed: {relative}")
                    bind_identity(relative, info)
                    walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                bind_identity(relative, info)
                actual_files.add(relative)
            else:
                raise RuntimeError(f"unsafe checkpoint entry: {relative}")
    walk(root_descriptor)
    if actual_files != expected_files:
        raise RuntimeError(
            "checkpoint governed file set differs; "
            f"missing={sorted(expected_files - actual_files)!r} "
            f"unexpected={sorted(actual_files - expected_files)!r}"
        )

def revalidate_checkpoint(root_descriptor):
    def walk(descriptor, prefix=""):
        names = sorted(os.listdir(descriptor))
        if names != directory_listings[prefix]:
            raise RuntimeError(f"checkpoint governed directory listing changed: {prefix or '.'}")
        for name in names:
            relative = f"{prefix}/{name}" if prefix else name
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if identity(info) != governed_identities[relative]:
                raise RuntimeError(f"checkpoint governed pathname changed: {relative}")
            if stat.S_ISDIR(info.st_mode):
                child = os.open(name, flags, dir_fd=descriptor)
                try:
                    if identity(os.fstat(child)) != governed_identities[relative]:
                        raise RuntimeError(f"checkpoint governed directory changed: {relative}")
                    walk(child, relative)
                finally:
                    os.close(child)
    walk(root_descriptor)

def publish_no_replace(parent_descriptor, attempt_name, target_name):
    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, "renameat2"):
        function, flag = libc.renameat2, 1
    elif hasattr(libc, "renameatx_np"):
        function, flag = libc.renameatx_np, 0x00000004
    else:
        raise RuntimeError("atomic no-replace directory publication is unavailable")
    function.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    if function(
        parent_descriptor, os.fsencode(attempt_name),
        parent_descriptor, os.fsencode(target_name), flag,
    ) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"checkpoint publication target already exists: {target}")
        raise OSError(error, os.strerror(error), target)

source_descriptor = os.open(source, flags)
attempt_parent = os.path.dirname(attempt) or "."
target_parent = os.path.dirname(target) or "."
if os.path.realpath(attempt_parent) != os.path.realpath(target_parent):
    raise RuntimeError("checkpoint attempt and target must share one parent directory")
attempt_name = os.path.basename(attempt)
target_name = os.path.basename(target)
parent_descriptor = os.open(attempt_parent, flags)
try:
    try:
        os.stat(target_name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"checkpoint publication target already exists: {target}")
    os.mkdir(attempt_name, 0o700, dir_fd=parent_descriptor)
    attempt_identity = object_identity(
        os.stat(attempt_name, dir_fd=parent_descriptor, follow_symlinks=False)
    )
    destination_descriptor = os.open(attempt_name, flags, dir_fd=parent_descriptor)
    try:
        clone_directory(source_descriptor, destination_descriptor)
        authenticate_checkpoint(destination_descriptor)
        os.fsync(destination_descriptor)
        revalidate_checkpoint(destination_descriptor)
        if (
            object_identity(os.fstat(destination_descriptor)) != attempt_identity
            or object_identity(
                os.stat(attempt_name, dir_fd=parent_descriptor, follow_symlinks=False)
            ) != attempt_identity
        ):
            raise RuntimeError("authenticated checkpoint attempt changed before publication")
        publish_no_replace(parent_descriptor, attempt_name, target_name)
    finally:
        os.close(destination_descriptor)
finally:
    os.close(parent_descriptor)
    os.close(source_descriptor)
"""


def seed_authenticated_checkpoint(
    source: Path | str,
    attempt: Path | str,
    target: Path | str,
    resolved: ResolvedAssemblyInput,
) -> None:
    """Clone, authenticate, and atomically publish one checkpoint custody tree."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            SECURE_CHECKPOINT_SEED_SCRIPT,
            str(source),
            str(attempt),
            str(target),
            native_resume_checkpoint_authority_json(resolved),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit={result.returncode}"
        raise NativeExecutionContextError(f"authenticated checkpoint seed failed: {detail}")


def uses_registered_native_execution(row: RunRowSpec) -> bool:
    """Return whether orchestration owns native payload and output routing for a row."""

    return row.launch.payload_routing.get(
        "kind"
    ) == "registered-execution-payload" and is_native_training_command(row.launch.command)


def missing_native_training_collection_outputs(row: RunRowSpec) -> list[str]:
    """Return required row-local native outputs absent from the collection contract."""

    if not uses_registered_native_execution(row):
        return []
    declared = set(row.launch.collect)
    return sorted(set(NATIVE_TRAINING_COLLECTION_OUTPUTS) - declared)


def native_resume_checkpoint_role(bundle: RunBundle, row: RunRowSpec) -> str | None:
    """Resolve the one authenticated checkpoint input required by a native resume."""

    source = native_resume_checkpoint_source(bundle, row)
    return None if source is None else source.custody.target_role


def native_resume_checkpoint_source(
    bundle: RunBundle, row: RunRowSpec
) -> ResolvedAssemblyInput | None:
    """Resolve the exact authenticated custody source for a native resume."""

    command = [str(part) for part in row.launch.command]
    if (
        row.launch.payload_routing.get("kind") != "registered-execution-payload"
        or not is_native_training_command(command)
        or "--resume" not in command
    ):
        return None

    checkpoint_identities = [
        identity
        for identity in row.execution.immutable_inputs
        if identity.kind == "checkpoint-custody-archive"
    ]
    if len(checkpoint_identities) != 1:
        raise NativeExecutionContextError(
            f"native resume row {row.row_id!r} requires exactly one immutable "
            "checkpoint-custody-archive input; "
            f"observed {len(checkpoint_identities)}"
        )
    expected = checkpoint_identities[0]
    resolved = [item for item in bundle.resolved_inputs if item.identity == expected]
    if len(resolved) != 1:
        raise NativeExecutionContextError(
            f"native resume row {row.row_id!r} requires exactly one resolved custody "
            f"source for checkpoint input {expected.identifier!r}; observed {len(resolved)}"
        )
    source = resolved[0]
    if source.custody.materializer.kind != "checkpoint-custody-archive":
        raise NativeExecutionContextError(
            f"native resume row {row.row_id!r} checkpoint source has unsupported "
            f"materializer kind {source.custody.materializer.kind!r}"
        )
    return source


def native_resume_checkpoint_authority_json(resolved: ResolvedAssemblyInput) -> str:
    """Serialize only the out-of-band authority needed by the seed protocol."""

    materializer = resolved.custody.materializer
    return json.dumps(
        {
            "expected_parent_ref": materializer.expected_parent_ref.model_dump(
                mode="json", exclude_none=True
            ),
            "expected_transaction_root_sha256": (materializer.expected_transaction_root_sha256),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def bind_native_execution_command(
    command: Sequence[str],
    *,
    row: RunRowSpec,
    payload_path: Path | str,
    collection_root: Path | str,
    update_budget: int | None = None,
) -> tuple[list[str], RunRowSpec]:
    """Bind one registered native row to its staged input and output roots."""

    normalized = [str(part) for part in command]
    if row.launch.payload_routing.get("kind") != "registered-execution-payload":
        return normalized, row
    if not is_native_training_command(normalized):
        return normalized, row
    command_index = normalized.index("execute-training-run-spec")
    if command_index + 1 < len(normalized) and not normalized[command_index + 1].startswith("-"):
        raise NativeExecutionContextError(
            "registered execution payload routing owns the native spec argument"
        )

    output_options = ("--manifest-root", "--checkpoint-root", "--run-id")
    conflicting = sorted(
        part
        for part in normalized
        if any(part == option or part.startswith(f"{option}=") for option in output_options)
    )
    if conflicting:
        raise NativeExecutionContextError(
            "registered native row output bindings are orchestration-owned; remove "
            f"caller-supplied options {conflicting!r}"
        )
    if update_budget is not None and any(
        part == "--update-budget" or part.startswith("--update-budget=") for part in normalized
    ):
        raise NativeExecutionContextError(
            "native update budget is orchestration-owned; remove caller-supplied --update-budget"
        )

    staged_payload = str(payload_path)
    row_root = Path(collection_root)
    normalized.insert(command_index + 1, staged_payload)
    provenance = row.execution.row_provenance
    if provenance is None:
        raise NativeExecutionContextError(
            f"native execution row {row.row_id!r} lacks TrainingRowProvenance"
        )
    normalized.extend(
        [
            "--manifest-root",
            str(row_root / "manifests"),
            "--checkpoint-root",
            str(row_root / "checkpoints"),
            "--run-id",
            provenance.planned_run_id,
        ]
    )
    if update_budget is not None:
        if (
            isinstance(update_budget, bool)
            or not isinstance(update_budget, int)
            or update_budget <= 0
        ):
            raise NativeExecutionContextError(
                "update_budget must be a positive non-boolean integer"
            )
        normalized.extend(["--update-budget", str(update_budget)])
    bound_row = row.model_copy(
        update={
            "execution": row.execution.model_copy(
                update={"payload": row.execution.payload.model_copy(update={"uri": staged_payload})}
            )
        }
    )
    return normalized, bound_row


def inject_native_execution_context(
    command: Sequence[str],
    *,
    row: RunRowSpec,
    environment_fingerprint: str,
    collection_root: Path | str,
) -> list[str]:
    """Append one canonical inline producer context to a native row command.

    Non-native commands are returned unchanged. Native commands must use the
    row's assembly envelope and authored-to-execution provenance; pre-supplied
    context options are rejected so orchestration never launches with a caller
    wrapper that can drift from the canonical row.
    """

    normalized = [str(part) for part in command]
    if not is_native_training_command(normalized):
        return normalized
    if row.execution.row_provenance is None:
        raise NativeExecutionContextError(
            f"native execution row {row.row_id!r} lacks TrainingRowProvenance"
        )
    if not environment_fingerprint:
        raise NativeExecutionContextError(
            f"native execution row {row.row_id!r} lacks a realized environment fingerprint"
        )
    context_options = ("--execution-context", "--execution-context-json")
    conflicting = sorted(
        part
        for part in normalized
        if any(part == option or part.startswith(f"{option}=") for option in context_options)
    )
    if conflicting:
        raise NativeExecutionContextError(
            "native execution context is orchestration-owned; remove caller-supplied "
            f"options {conflicting!r}"
        )
    context = NativeExecutionProducerContext(
        execution=row.execution,
        environment_fingerprint=environment_fingerprint,
        collection_root=str(collection_root),
    )
    payload = json.dumps(
        context.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    return [*normalized, "--execution-context-json", payload]
