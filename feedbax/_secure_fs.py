"""Strict descriptor-relative filesystem authority.

This import-leaf module owns the security invariants shared by custody,
analysis, training, and orchestration.  Callers keep their domain-specific
error types and operation rules, but cannot weaken path traversal silently.
"""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import sys
from collections.abc import Callable, Iterable
from enum import Enum
from pathlib import Path
from typing import TypeAlias


ErrorFactory: TypeAlias = Callable[[str], Exception]
DirectoryRecord: TypeAlias = tuple[Path, int, os.stat_result]
_DIR_FD_SUPPORT = frozenset(operation.__name__ for operation in os.supports_dir_fd)
_FOLLOW_SYMLINK_SUPPORT = frozenset(operation.__name__ for operation in os.supports_follow_symlinks)


def rename_no_replace(
    source_name: str,
    destination_name: str,
    *,
    source_dir_fd: int,
    destination_dir_fd: int,
    error_factory: ErrorFactory,
    context: str,
) -> None:
    """Atomically rename one descriptor-relative entry without replacement."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if hasattr(libc, "renameatx_np"):
        result = libc.renameatx_np(
            source_dir_fd,
            source_bytes,
            destination_dir_fd,
            destination_bytes,
            0x00000004,
        )
    elif hasattr(libc, "renameat2"):
        result = libc.renameat2(
            source_dir_fd,
            source_bytes,
            destination_dir_fd,
            destination_bytes,
            1,
        )
    else:
        raise error_factory(f"{context} requires atomic no-replace rename support")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(f"{context} destination already exists: {destination_name}")
    raise error_factory(f"{context} failed: {os.strerror(error)}")


class SecurePathRigor(Enum):
    """The object guarantee required from a descriptor-relative open."""

    DIRECTORY_IDENTITY = "directory-identity"
    REGULAR_FILE_IDENTITY = "regular-file-identity"
    SINGLE_LINK_FILE_IDENTITY = "single-link-file-identity"


def canonicalize_trusted_system_aliases(path: Path | str) -> Path:
    """Canonicalize only Darwin's fixed first-level ``/tmp`` and ``/var`` aliases."""
    absolute_path = Path(os.path.abspath(path))
    if sys.platform != "darwin" or len(absolute_path.parts) < 2:
        return absolute_path
    alias_name = absolute_path.parts[1]
    expected = {
        "tmp": (Path("/private/tmp"), {"private/tmp", "/private/tmp"}),
        "var": (Path("/private/var"), {"private/var", "/private/var"}),
    }.get(alias_name)
    if expected is None:
        return absolute_path
    canonical_prefix, allowed_targets = expected
    alias_path = Path(absolute_path.anchor) / alias_name
    try:
        alias_stat = alias_path.lstat()
        alias_target = os.readlink(alias_path)
    except OSError:
        return absolute_path
    if not stat.S_ISLNK(alias_stat.st_mode) or alias_target not in allowed_targets:
        return absolute_path
    return canonical_prefix.joinpath(*absolute_path.parts[2:])


def require_secure_path_capabilities(
    rigor: SecurePathRigor,
    *,
    error_factory: ErrorFactory,
    extra_dir_fd_operations: Iterable[Callable[..., object]] = (),
    require_link_no_follow: bool = False,
) -> None:
    """Fail closed unless the runtime can enforce ``rigor``."""
    required_constants = {"O_NOFOLLOW"}
    if rigor is SecurePathRigor.DIRECTORY_IDENTITY:
        required_constants.add("O_DIRECTORY")
    else:
        required_constants.add("O_NONBLOCK")
    missing = [name for name in required_constants if not getattr(os, name, 0)]

    required_operation_names = ("open", "stat", *(op.__name__ for op in extra_dir_fd_operations))
    missing.extend(name for name in required_operation_names if name not in _DIR_FD_SUPPORT)
    if "stat" not in _FOLLOW_SYMLINK_SUPPORT:
        missing.append("stat(follow_symlinks=False)")
    if require_link_no_follow and "link" not in _FOLLOW_SYMLINK_SUPPORT:
        missing.append("link(follow_symlinks=False)")
    if missing:
        raise error_factory(
            "secure filesystem operations require descriptor-relative no-follow support; "
            "unavailable: " + ", ".join(sorted(set(missing)))
        )


def directory_open_flags(*, error_factory: ErrorFactory) -> int:
    require_secure_path_capabilities(
        SecurePathRigor.DIRECTORY_IDENTITY,
        error_factory=error_factory,
    )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def file_open_flags(
    *,
    rigor: SecurePathRigor,
    error_factory: ErrorFactory,
    writable: bool = False,
) -> int:
    if rigor is SecurePathRigor.DIRECTORY_IDENTITY:
        raise ValueError("file_open_flags requires a regular-file rigor")
    require_secure_path_capabilities(rigor, error_factory=error_factory)
    access = os.O_RDWR if writable else os.O_RDONLY
    return access | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def open_existing_path(
    path: Path | str,
    *,
    rigor: SecurePathRigor,
    error_factory: ErrorFactory,
    context: str,
    dir_fd: int | None = None,
    wrap_os_errors: bool = True,
) -> tuple[int, os.stat_result]:
    """Open and pin one existing path, with a post-open named recheck."""
    flags = (
        directory_open_flags(error_factory=error_factory)
        if rigor is SecurePathRigor.DIRECTORY_IDENTITY
        else file_open_flags(rigor=rigor, error_factory=error_factory)
    )
    descriptor: int | None = None
    try:
        before = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
        descriptor = os.open(path, flags, dir_fd=dir_fd)
        opened = validate_opened_path(
            path,
            descriptor,
            rigor=rigor,
            error_factory=error_factory,
            context=context,
            dir_fd=dir_fd,
            expected_identity=_identity(before),
            wrap_os_errors=wrap_os_errors,
        )
        expected_type = (
            stat.S_ISDIR if rigor is SecurePathRigor.DIRECTORY_IDENTITY else stat.S_ISREG
        )
        if not expected_type(before.st_mode):
            expected_name = (
                "directory" if rigor is SecurePathRigor.DIRECTORY_IDENTITY else "regular file"
            )
            raise error_factory(f"{context} is not a {expected_name}")
        if rigor is SecurePathRigor.SINGLE_LINK_FILE_IDENTITY and before.st_nlink != 1:
            raise error_factory(
                f"{context} must have exactly one hard link; mutable hard-link aliases are unsafe"
            )
        result = descriptor
        descriptor = None
        return result, opened
    except FileNotFoundError:
        raise
    except Exception as exc:
        if isinstance(exc, OSError):
            if not wrap_os_errors:
                raise
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise error_factory(
                    f"{context} contains a symlink or non-directory component and is unsafe"
                ) from exc
            raise error_factory(f"{context} has an unsafe component or is unavailable") from exc
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def validate_opened_path(
    path: Path | str,
    descriptor: int,
    *,
    rigor: SecurePathRigor,
    error_factory: ErrorFactory,
    context: str,
    dir_fd: int | None = None,
    expected_identity: tuple[int, int] | None = None,
    wrap_os_errors: bool = True,
) -> os.stat_result:
    """Validate one descriptor against its named path after it has been opened."""
    require_secure_path_capabilities(rigor, error_factory=error_factory)
    try:
        opened = os.fstat(descriptor)
        named = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
    except OSError as exc:
        if not wrap_os_errors:
            raise
        raise error_factory(f"{context} is unsafe or unavailable") from exc
    expected_type = stat.S_ISDIR if rigor is SecurePathRigor.DIRECTORY_IDENTITY else stat.S_ISREG
    if not expected_type(opened.st_mode) or not expected_type(named.st_mode):
        expected_name = (
            "directory" if rigor is SecurePathRigor.DIRECTORY_IDENTITY else "regular file"
        )
        raise error_factory(f"{context} is not a {expected_name}")
    opened_identity = _identity(opened)
    if _identity(named) != opened_identity or (
        expected_identity is not None and expected_identity != opened_identity
    ):
        raise error_factory(f"{context} identity changed while opening")
    if rigor is SecurePathRigor.SINGLE_LINK_FILE_IDENTITY and (
        opened.st_nlink != 1 or named.st_nlink != 1
    ):
        raise error_factory(
            f"{context} must have exactly one hard link; mutable hard-link aliases are unsafe"
        )
    return opened


def open_existing_directory(
    path: Path | str,
    *,
    error_factory: ErrorFactory,
    context: str,
    dir_fd: int | None = None,
    wrap_os_errors: bool = True,
) -> int:
    descriptor, _ = open_existing_path(
        path,
        rigor=SecurePathRigor.DIRECTORY_IDENTITY,
        error_factory=error_factory,
        context=context,
        dir_fd=dir_fd,
        wrap_os_errors=wrap_os_errors,
    )
    return descriptor


def open_existing_file(
    path: Path | str,
    *,
    rigor: SecurePathRigor,
    error_factory: ErrorFactory,
    context: str,
    dir_fd: int | None = None,
    wrap_os_errors: bool = True,
) -> tuple[int, os.stat_result]:
    if rigor is SecurePathRigor.DIRECTORY_IDENTITY:
        raise ValueError("open_existing_file requires a regular-file rigor")
    return open_existing_path(
        path,
        rigor=rigor,
        error_factory=error_factory,
        context=context,
        dir_fd=dir_fd,
        wrap_os_errors=wrap_os_errors,
    )


def open_directory_chain(
    directory: Path | str,
    *,
    create: bool,
    error_factory: ErrorFactory,
    context: str,
    create_mode: int = 0o777,
) -> list[DirectoryRecord]:
    """Open every component of an absolute path without following links."""
    absolute_directory = canonicalize_trusted_system_aliases(directory)
    anchor = Path(absolute_directory.anchor)
    if not anchor.anchor:
        raise error_factory(f"{context} must resolve to an absolute path: {directory}")
    require_secure_path_capabilities(
        SecurePathRigor.DIRECTORY_IDENTITY,
        error_factory=error_factory,
        extra_dir_fd_operations=((os.mkdir,) if create else ()),
    )
    records: list[DirectoryRecord] = []
    try:
        descriptor, anchor_stat = open_existing_path(
            anchor,
            rigor=SecurePathRigor.DIRECTORY_IDENTITY,
            error_factory=error_factory,
            context=f"{context} anchor",
        )
        records.append((anchor, descriptor, anchor_stat))
        current_path = anchor
        for component in absolute_directory.parts[1:]:
            current_path /= component
            try:
                next_descriptor, next_stat = open_existing_path(
                    component,
                    rigor=SecurePathRigor.DIRECTORY_IDENTITY,
                    error_factory=error_factory,
                    context=f"{context} component {current_path}",
                    dir_fd=records[-1][1],
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=create_mode, dir_fd=records[-1][1])
                except FileExistsError:
                    pass
                next_descriptor, next_stat = open_existing_path(
                    component,
                    rigor=SecurePathRigor.DIRECTORY_IDENTITY,
                    error_factory=error_factory,
                    context=f"{context} component {current_path}",
                    dir_fd=records[-1][1],
                )
            records.append((current_path, next_descriptor, next_stat))
        return records
    except OSError as exc:
        close_directory_chain(records, raise_errors=False)
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise error_factory(f"{context} traverses a symlink or non-directory") from exc
        raise
    except Exception:
        close_directory_chain(records, raise_errors=False)
        raise


def recheck_directory_chain(
    records: list[DirectoryRecord],
    *,
    error_factory: ErrorFactory,
    context: str,
) -> None:
    """Prove an open chain still names the same external directories."""
    for path, descriptor, initial_stat in records:
        try:
            descriptor_stat = os.fstat(descriptor)
            path_stat = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise error_factory(f"{context} disappeared during the operation: {path}") from exc
        expected_identity = _identity(initial_stat)
        if (
            not stat.S_ISDIR(descriptor_stat.st_mode)
            or not stat.S_ISDIR(path_stat.st_mode)
            or _identity(descriptor_stat) != expected_identity
            or _identity(path_stat) != expected_identity
        ):
            raise error_factory(f"{context} identity changed during the operation: {path}")


def close_descriptors(descriptors: Iterable[int], *, raise_errors: bool = True) -> None:
    """Close every descriptor even when an earlier close fails."""
    first_error: OSError | None = None
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError as exc:
            if first_error is None:
                first_error = exc
    if raise_errors and first_error is not None:
        raise first_error


def close_directory_chain(
    records: list[DirectoryRecord],
    *,
    raise_errors: bool = True,
) -> None:
    close_descriptors(
        (descriptor for _, descriptor, _ in reversed(records)),
        raise_errors=raise_errors,
    )


def take_directory_chain_leaf(records: list[DirectoryRecord]) -> int:
    """Transfer the leaf descriptor while reliably closing its ancestors."""
    if not records:
        raise ValueError("a directory chain must contain at least its anchor")
    leaf_descriptor = records[-1][1]
    try:
        close_descriptors(descriptor for _, descriptor, _ in reversed(records[:-1]))
    except Exception:
        close_descriptors((leaf_descriptor,), raise_errors=False)
        raise
    records.clear()
    return leaf_descriptor
