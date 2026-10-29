"""Shared no-follow policy for run-owned collection publication."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from collections.abc import Sequence
from uuid import uuid4

from feedbax._secure_fs import (
    SecurePathRigor,
    close_directory_chain,
    open_directory_chain,
    open_existing_directory,
    open_existing_path,
    rename_no_replace,
    rename_exchange,
    validate_opened_path,
    recheck_directory_chain,
)
from feedbax.orchestration.bundle import RunBundle, RunRowSpec
from feedbax.orchestration.collection_recovery import (
    CollectionRecoveryError,
    copy_collection_member,
)


class CollectionDestinationError(ValueError):
    """A run-owned output destination is unsafe or changed during publication."""


def check_collection_destinations(
    bundle: RunBundle,
    *,
    rows: Sequence[RunRowSpec] | None = None,
) -> None:
    """Check destination bindings, independently of the size of retained output trees."""
    try:
        records = open_directory_chain(
            bundle.run_set_dir,
            create=False,
            error_factory=CollectionDestinationError,
            context="collection destination",
        )
    except FileNotFoundError:
        return
    try:
        root_fd = records[-1][1]
        for name in ("collected", "events", "failure-logs"):
            _check_destination_binding(root_fd, Path(name), directory=True)
        for row in bundle.rows if rows is None else rows:
            collected = Path("collected") / row.row_id
            _check_destination_binding(root_fd, collected, directory=True)
            event_name = f"{row.row_id}.events.jsonl"
            names = {Path(source).name for source in row.launch.collect} or {
                row.row_id,
                event_name,
                "run-config.json",
                "bundle.json",
            }
            for name in names:
                if name:
                    _check_destination_binding(root_fd, collected / name, directory=False)
            _check_destination_binding(root_fd, Path("events") / event_name, directory=False)
        recheck_directory_chain(
            records, error_factory=CollectionDestinationError, context="collection destination"
        )
    finally:
        close_directory_chain(records)


def _check_destination_binding(root_fd: int, path: Path, *, directory: bool) -> None:
    bindings: list[tuple[int, str, int]] = []
    parent_fd = root_fd
    try:
        for part in path.parts if directory else path.parent.parts:
            try:
                child_fd = open_existing_directory(
                    part,
                    dir_fd=parent_fd,
                    error_factory=CollectionDestinationError,
                    context="collection destination",
                )
            except FileNotFoundError:
                return
            bindings.append((parent_fd, part, child_fd))
            parent_fd = child_fd
        if not directory:
            try:
                value = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            _require_output_type(value)
    finally:
        try:
            for parent_fd, name, child_fd in bindings:
                validate_opened_path(
                    name,
                    child_fd,
                    dir_fd=parent_fd,
                    rigor=SecurePathRigor.DIRECTORY_IDENTITY,
                    error_factory=CollectionDestinationError,
                    context="collection destination",
                )
        finally:
            for _, _, child_fd in reversed(bindings):
                os.close(child_fd)


def _require_output_type(value: os.stat_result) -> None:
    if not stat.S_ISDIR(value.st_mode) and not (
        stat.S_ISREG(value.st_mode) and value.st_nlink == 1
    ):
        raise CollectionDestinationError("collection destination member is unsafe")


def _check_members(directory_fd: int) -> None:
    for name in os.listdir(directory_fd):
        value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(value.st_mode):
            child_fd = open_existing_directory(
                name,
                dir_fd=directory_fd,
                error_factory=CollectionDestinationError,
                context="collection destination member",
            )
            try:
                _check_members(child_fd)
            finally:
                os.close(child_fd)
        elif not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise CollectionDestinationError(
                f"collection destination contains an unsafe member: {name!r}"
            )


def publish_collection_output(
    bundle: RunBundle,
    source: Path,
    destination: Path,
    *,
    consume_source: bool = False,
) -> None:
    """Prepare a replacement privately, then atomically commit it to a pinned destination."""
    relative = destination.relative_to(bundle.run_set_dir)
    if (
        not relative.parts
        or relative.parts[0] not in {"collected", "events", "failure-logs"}
        or any(part in {".", ".."} for part in relative.parts)
    ):
        raise CollectionDestinationError(f"invalid collection destination: {destination}")
    destination_records = open_directory_chain(
        destination.parent,
        create=True,
        error_factory=CollectionDestinationError,
        context="collection destination",
    )
    source_records = []
    staging_fd = None
    staging_name = f".collection-{uuid4().hex}"
    try:
        destination_fd = destination_records[-1][1]
        source_records = open_directory_chain(
            source.parent.resolve(),
            create=False,
            error_factory=CollectionDestinationError,
            context="collection source",
        )
        source_parent_fd = source_records[-1][1]
        source_stat = os.stat(source.name, dir_fd=source_parent_fd, follow_symlinks=False)
        os.mkdir(staging_name, mode=0o700, dir_fd=destination_fd)
        staging_fd = open_existing_directory(
            staging_name,
            dir_fd=destination_fd,
            error_factory=CollectionDestinationError,
            context="collection staging",
        )
        if consume_source and source_stat.st_dev == os.fstat(destination_fd).st_dev:
            _move_staged_output(source_parent_fd, source.name, staging_fd, "output")
        else:
            try:
                copy_collection_member(source_parent_fd, (source.name,), staging_fd, "output")
            except CollectionRecoveryError as exc:
                raise CollectionDestinationError(str(exc)) from exc
        recheck_directory_chain(
            destination_records,
            error_factory=CollectionDestinationError,
            context="collection destination",
        )
        try:
            existing = os.stat(destination.name, dir_fd=destination_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            _require_output_type(existing)
            rename_exchange(
                "output",
                destination.name,
                source_dir_fd=staging_fd,
                destination_dir_fd=destination_fd,
                error_factory=CollectionDestinationError,
                context="collection publication",
            )
        else:
            rename_no_replace(
                "output",
                destination.name,
                source_dir_fd=staging_fd,
                destination_dir_fd=destination_fd,
                error_factory=CollectionDestinationError,
                context="collection publication",
            )
        try:
            recheck_directory_chain(
                destination_records,
                error_factory=CollectionDestinationError,
                context="collection destination",
            )
        except BaseException:
            if existing is not None:
                rename_exchange(
                    "output",
                    destination.name,
                    source_dir_fd=staging_fd,
                    destination_dir_fd=destination_fd,
                    error_factory=CollectionDestinationError,
                    context="collection rollback",
                )
            else:
                rename_no_replace(
                    destination.name,
                    "output",
                    source_dir_fd=destination_fd,
                    destination_dir_fd=staging_fd,
                    error_factory=CollectionDestinationError,
                    context="collection rollback",
                )
            raise
    finally:
        try:
            if staging_fd is not None:
                os.close(staging_fd)
                shutil.rmtree(staging_name, dir_fd=destination_records[-1][1])
        finally:
            close_directory_chain(source_records)
            close_directory_chain(destination_records)


def _move_staged_output(source_fd: int, source: str, destination_fd: int, destination: str) -> None:
    value = os.stat(source, dir_fd=source_fd, follow_symlinks=False)
    rigor = (
        SecurePathRigor.DIRECTORY_IDENTITY
        if stat.S_ISDIR(value.st_mode)
        else SecurePathRigor.SINGLE_LINK_FILE_IDENTITY
    )
    descriptor, _ = open_existing_path(
        source,
        dir_fd=source_fd,
        rigor=rigor,
        error_factory=CollectionDestinationError,
        context="staged collection output",
    )
    try:
        if rigor is SecurePathRigor.DIRECTORY_IDENTITY:
            _check_members(descriptor)
        rename_no_replace(
            source,
            destination,
            source_dir_fd=source_fd,
            destination_dir_fd=destination_fd,
            error_factory=CollectionDestinationError,
            context="collection publication",
        )
        validate_opened_path(
            destination,
            descriptor,
            dir_fd=destination_fd,
            rigor=rigor,
            error_factory=CollectionDestinationError,
            context="published collection output",
        )
    finally:
        os.close(descriptor)
