"""Fail-closed materialization of preserved collection outputs for stage retry."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from feedbax._secure_fs import (
    SecurePathRigor,
    open_directory_chain,
    open_existing_directory,
    open_existing_file,
    take_directory_chain_leaf,
    validate_opened_path,
)
from feedbax.orchestration.bundle import RunBundle, RunRowSpec
from feedbax.orchestration.state import RunSetState


class CollectionRecoveryError(ValueError):
    """Raised when preserved collection output cannot be recovered safely."""


@dataclass(frozen=True, slots=True)
class CollectionRecoveryBinding:
    """Explicit run-row binding to a preserved collected-output root."""

    row_id: str
    root: Path | str


@dataclass(frozen=True, slots=True)
class RecoveredCollection:
    """Fresh materialization and its non-secret integrity evidence."""

    outputs: Mapping[str, str]
    evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _CopiedFile:
    relative_path: str
    sha256: str
    size_bytes: int


def recover_collected_outputs(
    bundle: RunBundle,
    row: RunRowSpec,
    state: RunSetState,
    *,
    bindings: Sequence[CollectionRecoveryBinding],
) -> RecoveredCollection | None:
    """Copy an exact preserved output set into a fresh retry-owned tree.

    Recovery is intentionally available only after a completed RunPod teardown
    proves both exact pod absence and empty final account inventory. Both source
    and destination are traversed relative to pinned, no-follow descriptors.
    The original bytes are never moved, renamed, or modified.
    """

    selected = _select_binding(bundle, row, state, bindings)
    if selected is None:
        return None
    _require_completed_teardown(state)

    source_root = Path(selected.root)
    expected_source = bundle.run_set_dir / "collected" / row.row_id
    if not source_root.is_absolute():
        raise CollectionRecoveryError("collection recovery root must be absolute")
    if os.path.abspath(source_root) != os.path.abspath(expected_source):
        raise CollectionRecoveryError(
            "collection recovery root must be the preserved run-owned row collection root"
        )

    logical_names = _logical_output_names(row.launch.collect)
    descriptors: list[int] = []
    named_descriptors: list[tuple[int, str, int, str]] = []
    try:
        run_fd = _open_directory(bundle.run_set_dir, context="run-set root")
        descriptors.append(run_fd)
        run_identity = _binding_identity(os.fstat(run_fd))

        source_parent_fd = _open_directory(
            "collected",
            dir_fd=run_fd,
            context="run-set collected root",
        )
        descriptors.append(source_parent_fd)
        named_descriptors.append((run_fd, "collected", source_parent_fd, "source parent"))
        source_fd = _open_directory(
            row.row_id,
            dir_fd=source_parent_fd,
            context=f"collection recovery root for row {row.row_id!r}",
        )
        descriptors.append(source_fd)
        named_descriptors.append((source_parent_fd, row.row_id, source_fd, "preserved row root"))
        source_identity = _identity(os.fstat(source_fd))
        source_entries = sorted(os.listdir(source_fd))
        expected_entries = sorted(logical_names)
        if source_entries != expected_entries:
            raise CollectionRecoveryError(
                "preserved collection output map differs from the declaration; "
                f"missing={sorted(set(expected_entries) - set(source_entries))!r} "
                f"unexpected={sorted(set(source_entries) - set(expected_entries))!r}"
            )

        attempt_name = f"collect-recovery-{state.stage('COLLECT').attempts}"
        attempts_fd = _open_or_create_directory(
            run_fd,
            ".stage-attempts",
            context="stage-attempts root",
        )
        descriptors.append(attempts_fd)
        named_descriptors.append((run_fd, ".stage-attempts", attempts_fd, "attempts root"))
        attempt_fd = _create_directory(
            attempts_fd,
            attempt_name,
            context="collection recovery attempt root",
        )
        descriptors.append(attempt_fd)
        named_descriptors.append((attempts_fd, attempt_name, attempt_fd, "attempt root"))
        destination_collected_fd = _create_directory(
            attempt_fd,
            "collected",
            context="recovery collected root",
        )
        descriptors.append(destination_collected_fd)
        named_descriptors.append(
            (attempt_fd, "collected", destination_collected_fd, "recovery collected root")
        )
        destination_row_fd = _create_directory(
            destination_collected_fd,
            row.row_id,
            context="recovery row root",
        )
        descriptors.append(destination_row_fd)
        named_descriptors.append(
            (destination_collected_fd, row.row_id, destination_row_fd, "recovery row root")
        )
        destination_root = (
            bundle.run_set_dir / ".stage-attempts" / attempt_name / "collected" / row.row_id
        )

        output_paths: dict[str, str] = {}
        file_evidence: list[dict[str, object]] = []
        for logical_name in logical_names:
            source_parts, nested = _select_source_member(source_fd, logical_name)
            records = _copy_member_no_follow(
                source_fd,
                source_parts,
                destination_row_fd,
                logical_name,
            )
            output_paths[logical_name] = str(destination_root / logical_name)
            file_evidence.extend(
                {
                    "logical_name": logical_name,
                    "relative_path": record.relative_path,
                    "sha256": record.sha256,
                    "size_bytes": record.size_bytes,
                }
                for record in records
            )
            if nested:
                file_evidence.append(
                    {
                        "logical_name": logical_name,
                        "relative_path": ".",
                        "layout_normalized": "single-repeated-basename",
                    }
                )

        if sorted(os.listdir(source_fd)) != source_entries:
            raise CollectionRecoveryError("preserved collection root changed while copying")
        if _identity(os.fstat(source_fd)) != source_identity:
            raise CollectionRecoveryError(
                "preserved collection root identity changed while copying"
            )
        _validate_named_descriptors(named_descriptors)
        try:
            final_run_identity = _binding_identity(
                os.stat(bundle.run_set_dir, follow_symlinks=False)
            )
        except OSError as exc:
            raise CollectionRecoveryError("run-set root was replaced while copying") from exc
        if final_run_identity != run_identity:
            raise CollectionRecoveryError("run-set root was replaced while copying")
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)

    return RecoveredCollection(
        outputs=output_paths,
        evidence={
            "mode": "preserved-local-output-recovery",
            "row_id": row.row_id,
            "source_root": str(source_root),
            "destination_root": str(destination_root),
            "declared_outputs": list(logical_names),
            "files": file_evidence,
            "original_evidence_untouched": True,
            "provider_calls": 0,
        },
    )


def _select_binding(
    bundle: RunBundle,
    row: RunRowSpec,
    state: RunSetState,
    bindings: Sequence[CollectionRecoveryBinding],
) -> CollectionRecoveryBinding | None:
    if not bindings:
        return None
    known_rows = {item.row_id for item in bundle.rows}
    unexpected_rows = sorted({binding.row_id for binding in bindings} - known_rows)
    if unexpected_rows:
        raise CollectionRecoveryError(
            f"collection recovery bindings name unknown rows: {unexpected_rows!r}"
        )
    selected = [binding for binding in bindings if binding.row_id == row.row_id]
    if not selected:
        raise CollectionRecoveryError(
            f"collection recovery binding is missing for row {row.row_id!r}"
        )
    if len(selected) != 1:
        raise CollectionRecoveryError(
            f"collection recovery binding is ambiguous for row {row.row_id!r}"
        )
    collect_stage = state.stage("COLLECT")
    if collect_stage.status != "running" or collect_stage.attempts < 2:
        raise CollectionRecoveryError(
            "collection recovery is only valid while retrying a failed COLLECT stage"
        )
    if state.rows[row.row_id].status != "completed":
        raise CollectionRecoveryError(f"collection recovery requires completed row {row.row_id!r}")
    return selected[0]


def _require_completed_teardown(state: RunSetState) -> None:
    teardown = state.stage("TEARDOWN")
    outputs = teardown.outputs
    provision = state.provision_record
    provision_stage = state.stage("PROVISION")
    pod_id = outputs.get("pod_id")
    pod_absence = outputs.get("pod_absence")
    final_inventory = outputs.get("final_pod_inventory")
    valid_pod_id = isinstance(pod_id, str) and bool(pod_id)
    if (
        teardown.status != "completed"
        or provision_stage.status != "completed"
        or not isinstance(provision, Mapping)
        or dict(provision_stage.outputs) != dict(provision)
        or outputs.get("driver") != "runpod"
        or outputs.get("teardown") not in {"removed", "stopped-then-removed"}
        or not valid_pod_id
        or provision.get("pod_id") != pod_id
        or not isinstance(pod_absence, Mapping)
        or pod_absence.get("verified") is not True
        or pod_absence.get("pod_id") != pod_id
        or pod_absence.get("terminal_observation") != "not-found"
    ):
        raise CollectionRecoveryError(
            "collection recovery requires completed RunPod removal with exact pod absence"
        )
    if (
        not isinstance(final_inventory, Mapping)
        or final_inventory.get("verified") is not True
        or final_inventory.get("outcome") != "empty"
        or final_inventory.get("pod_count") != 0
        or final_inventory.get("pod_ids") != []
        or final_inventory.get("scope") != "provider-account"
        or final_inventory.get("observation_basis") != "runpodctl pod list --output json"
    ):
        raise CollectionRecoveryError(
            "collection recovery requires verified empty provider-account inventory"
        )


def _logical_output_names(sources: Sequence[str]) -> tuple[str, ...]:
    names = tuple(Path(source).name for source in sources)
    if any(not name or name in {".", ".."} for name in names):
        raise CollectionRecoveryError("declared collection output has no safe logical name")
    if len(set(names)) != len(names):
        raise CollectionRecoveryError("declared collection output names are ambiguous")
    return names


def _select_source_member(root_fd: int, logical_name: str) -> tuple[tuple[str, ...], bool]:
    source_stat = _stat_member(root_fd, logical_name, context=logical_name)
    if stat.S_ISREG(source_stat.st_mode):
        return (logical_name,), False
    if not stat.S_ISDIR(source_stat.st_mode):
        raise CollectionRecoveryError(
            f"preserved output {logical_name!r} is not a regular file or directory"
        )
    directory_fd = _open_directory(logical_name, dir_fd=root_fd, context=logical_name)
    try:
        entries = set(os.listdir(directory_fd))
        if entries == {logical_name}:
            nested_stat = _stat_member(
                directory_fd,
                logical_name,
                context=f"{logical_name} nested basename",
            )
            if not stat.S_ISDIR(nested_stat.st_mode):
                raise CollectionRecoveryError(
                    f"preserved output {logical_name!r} has an ambiguous repeated basename"
                )
            return (logical_name, logical_name), True
        if logical_name in entries:
            raise CollectionRecoveryError(
                f"preserved output {logical_name!r} has an ambiguous nested basename"
            )
        return (logical_name,), False
    finally:
        os.close(directory_fd)


def _copy_member_no_follow(
    source_root_fd: int,
    source_parts: tuple[str, ...],
    destination_root_fd: int,
    destination_name: str,
) -> tuple[_CopiedFile, ...]:
    source_parent_fd = os.dup(source_root_fd)
    try:
        for part in source_parts[:-1]:
            next_fd = _open_directory(part, dir_fd=source_parent_fd, context="source member")
            os.close(source_parent_fd)
            source_parent_fd = next_fd
        return _copy_entry(
            source_parent_fd,
            source_parts[-1],
            destination_root_fd,
            destination_name,
            relative_prefix="",
        )
    finally:
        os.close(source_parent_fd)


def _copy_entry(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
    *,
    relative_prefix: str,
) -> tuple[_CopiedFile, ...]:
    before = _stat_member(source_parent_fd, source_name, context=source_name)
    if stat.S_ISREG(before.st_mode):
        return (
            _copy_regular_file(
                source_parent_fd,
                source_name,
                destination_parent_fd,
                destination_name,
                before=before,
                relative_path=relative_prefix or destination_name,
            ),
        )
    if not stat.S_ISDIR(before.st_mode):
        raise CollectionRecoveryError(f"unsupported preserved collection object: {source_name!r}")

    destination_fd = _create_directory(
        destination_parent_fd,
        destination_name,
        context=f"recovered directory {destination_name!r}",
    )
    source_fd = _open_directory(
        source_name,
        dir_fd=source_parent_fd,
        context=source_name,
    )
    try:
        entries = sorted(os.listdir(source_fd))
        records: list[_CopiedFile] = []
        for child in entries:
            child_relative = f"{relative_prefix}/{child}" if relative_prefix else child
            records.extend(
                _copy_entry(
                    source_fd,
                    child,
                    destination_fd,
                    child,
                    relative_prefix=child_relative,
                )
            )
        if sorted(os.listdir(source_fd)) != entries or _identity(os.fstat(source_fd)) != _identity(
            before
        ):
            raise CollectionRecoveryError(
                f"preserved collection directory changed while copying: {source_name!r}"
            )
        _validate_named_descriptor(
            destination_parent_fd,
            destination_name,
            destination_fd,
            context="recovered directory",
        )
        return tuple(records)
    finally:
        os.close(source_fd)
        os.close(destination_fd)


def _copy_regular_file(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
    *,
    before: os.stat_result,
    relative_path: str,
) -> _CopiedFile:
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    source_fd: int | None = None
    destination_fd: int | None = None
    digest = hashlib.sha256()
    size = 0
    try:
        source_fd, opened = open_existing_file(
            source_name,
            rigor=SecurePathRigor.REGULAR_FILE_IDENTITY,
            error_factory=CollectionRecoveryError,
            context=f"preserved collection file {source_name!r}",
            dir_fd=source_parent_fd,
        )
        if _identity(before) != _identity(opened):
            raise CollectionRecoveryError(
                f"preserved collection file changed before copying: {source_name!r}"
            )
        destination_fd = os.open(
            destination_name,
            destination_flags,
            0o600,
            dir_fd=destination_parent_fd,
        )
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        if _identity(opened) != _identity(after) or size != after.st_size:
            raise CollectionRecoveryError(
                f"preserved collection file changed while copying: {source_name!r}"
            )
        _validate_named_descriptor(
            destination_parent_fd,
            destination_name,
            destination_fd,
            context="recovered file",
            rigor=SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
        )
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)
    return _CopiedFile(relative_path=relative_path, sha256=digest.hexdigest(), size_bytes=size)


def _open_or_create_directory(parent_fd: int, name: str, *, context: str) -> int:
    try:
        return _open_directory(name, dir_fd=parent_fd, context=context)
    except CollectionRecoveryError:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            return _open_directory(name, dir_fd=parent_fd, context=context)
        except OSError as exc:
            raise CollectionRecoveryError(f"{context} is unsafe or unavailable") from exc
        return _open_directory(name, dir_fd=parent_fd, context=context)


def _create_directory(parent_fd: int, name: str, *, context: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError as exc:
        raise CollectionRecoveryError(f"{context} already exists") from exc
    except OSError as exc:
        raise CollectionRecoveryError(f"{context} is unsafe or unavailable") from exc
    return _open_directory(name, dir_fd=parent_fd, context=context)


def _open_directory(
    path: str | Path,
    *,
    dir_fd: int | None = None,
    context: str,
) -> int:
    try:
        if dir_fd is not None:
            return open_existing_directory(
                path,
                error_factory=CollectionRecoveryError,
                context=context,
                dir_fd=dir_fd,
            )
        records = open_directory_chain(
            Path(path),
            create=False,
            error_factory=CollectionRecoveryError,
            context=context,
        )
        return take_directory_chain_leaf(records)
    except OSError as exc:
        raise CollectionRecoveryError(f"{context} is unsafe or unavailable") from exc


def _stat_member(parent_fd: int, name: str, *, context: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise CollectionRecoveryError(f"{context} is unsafe or unavailable") from exc


def _validate_named_descriptors(
    bindings: Sequence[tuple[int, str, int, str]],
) -> None:
    for parent_fd, name, descriptor, context in bindings:
        _validate_named_descriptor(parent_fd, name, descriptor, context=context)


def _validate_named_descriptor(
    parent_fd: int,
    name: str,
    descriptor: int,
    *,
    context: str,
    rigor: SecurePathRigor = SecurePathRigor.DIRECTORY_IDENTITY,
) -> None:
    validate_opened_path(
        name,
        descriptor,
        rigor=rigor,
        error_factory=CollectionRecoveryError,
        context=context,
        dir_fd=parent_fd,
    )


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _binding_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode)
