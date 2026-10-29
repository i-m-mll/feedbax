"""Shared no-follow policy for run-owned collection publication."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from feedbax._secure_fs import (
    SecurePathRigor,
    close_directory_chain,
    open_directory_chain,
    open_existing_directory,
    open_existing_path,
    rename_no_replace,
    validate_opened_path,
    recheck_directory_chain,
)
from feedbax.orchestration.bundle import RunBundle
from feedbax.orchestration.collection_recovery import (
    CollectionRecoveryError,
    copy_collection_member,
)


class CollectionDestinationError(ValueError):
    """A run-owned output destination is unsafe or changed during publication."""


def check_collection_destinations(bundle: RunBundle) -> None:
    """Inspect existing output components without creating directories or following links."""
    paths = [bundle.run_set_dir / name for name in ("collected", "events", "failure-logs")]
    paths.extend(bundle.run_set_dir / "collected" / row.row_id for row in bundle.rows)
    for path in paths:
        try:
            records = open_directory_chain(
                path,
                create=False,
                error_factory=CollectionDestinationError,
                context="collection destination",
            )
        except FileNotFoundError:
            continue
        try:
            _check_members(records[-1][1])
            recheck_directory_chain(
                records, error_factory=CollectionDestinationError, context="collection destination"
            )
        finally:
            close_directory_chain(records)


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
    """Copy into pinned run-owned directories, never through a destination alias."""
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
    try:
        destination_fd = destination_records[-1][1]
        try:
            existing = os.stat(destination.name, dir_fd=destination_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISDIR(existing.st_mode):
                existing_fd = open_existing_directory(
                    destination.name,
                    dir_fd=destination_fd,
                    error_factory=CollectionDestinationError,
                    context="collection destination member",
                )
                try:
                    _check_members(existing_fd)
                finally:
                    os.close(existing_fd)
                shutil.rmtree(destination.name, dir_fd=destination_fd)
            elif stat.S_ISREG(existing.st_mode) and existing.st_nlink == 1:
                os.unlink(destination.name, dir_fd=destination_fd)
            else:
                raise CollectionDestinationError("collection destination member is unsafe")
        source_records = open_directory_chain(
            source.parent.resolve(),
            create=False,
            error_factory=CollectionDestinationError,
            context="collection source",
        )
        source_parent_fd = source_records[-1][1]
        source_stat = os.stat(source.name, dir_fd=source_parent_fd, follow_symlinks=False)
        if consume_source and source_stat.st_dev == os.fstat(destination_fd).st_dev:
            _move_staged_output(source_parent_fd, source.name, destination_fd, destination.name)
        else:
            try:
                copy_collection_member(
                    source_parent_fd, (source.name,), destination_fd, destination.name
                )
            except CollectionRecoveryError as exc:
                raise CollectionDestinationError(str(exc)) from exc
        recheck_directory_chain(
            destination_records,
            error_factory=CollectionDestinationError,
            context="collection destination",
        )
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
