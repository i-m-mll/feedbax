"""Shared no-follow policy for run-owned collection publication."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from feedbax._secure_fs import (
    close_directory_chain,
    open_directory_chain,
    open_existing_directory,
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


def publish_collection_output(bundle: RunBundle, source: Path, destination: Path) -> None:
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
        try:
            copy_collection_member(
                source_records[-1][1], (source.name,), destination_fd, destination.name
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
