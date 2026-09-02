"""Content-addressed snapshots of governed Git working-tree bytes."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from subprocess import CalledProcessError, PIPE, run as subprocess_run
from typing import Any, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel


REPO_SNAPSHOT_MANIFEST_SCHEMA_ID = "feedbax.orchestration.repo_snapshot_manifest"
REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION = (
    "feedbax.orchestration.repo_snapshot_manifest.v1"
)

REPO_SNAPSHOT_CACHE_DIR_ENV = "FEEDBAX_REPO_SNAPSHOT_CACHE_DIR"

_PUBLISH_ATTEMPTS = 8


class RepoSnapshotError(RuntimeError):
    """Raised when governed repository bytes cannot be sealed safely."""


class RepoSnapshotCacheFault(RepoSnapshotError):
    """Raised when sealed cache bytes disagree with the content address holding them.

    A content-addressed cache path asserts the digest of the bytes stored under it, so
    disagreement there means the cache entry is damaged — truncated, emptied by an
    operating-system temporary-file reaper, or otherwise mutated. It is never evidence
    that the caller's own content differs, which is what `RepoSnapshotError` reports for
    a tree verified against a recorded manifest authority.
    """


class RepoSnapshotRecord(StrictModel):
    """Durable identity of one sealed tracked-working-tree snapshot."""

    commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    dirty: bool
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_count: int = Field(ge=0)


class RepoSnapshotManifest(StrictModel):
    """Versioned transfer authority for all configured local repositories."""

    schema_id: Literal[REPO_SNAPSHOT_MANIFEST_SCHEMA_ID] = REPO_SNAPSHOT_MANIFEST_SCHEMA_ID
    schema_version: Literal[REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION] = (
        REPO_SNAPSHOT_MANIFEST_SCHEMA_VERSION
    )
    repos: dict[str, RepoSnapshotRecord] = Field(default_factory=dict)


@dataclass(frozen=True)
class SealedRepoSnapshot:
    """Local immutable bytes plus their durable transfer record."""

    name: str
    source_root: Path
    staging_root: Path
    record: RepoSnapshotRecord


@dataclass(frozen=True)
class SealedRepoSnapshots:
    """One producer result shared by orchestration and execution planning."""

    manifest: RepoSnapshotManifest
    snapshots: Mapping[str, SealedRepoSnapshot]


@dataclass(frozen=True)
class _TrackedEntry:
    mode: bytes
    path_bytes: bytes

    @property
    def path(self) -> Path:
        return Path(os.fsdecode(self.path_bytes))


def default_repo_snapshot_cache_dir() -> Path:
    """Return the per-checkout parent directory for sealed repo-snapshot bytes.

    The sealed cache is a durable content-addressed store, so it must not live in the
    machine-global temporary directory: every checkout, worktree, test worker, and
    unrelated process on the host would share one tree, and the operating system's
    periodic temporary-file reaper is entitled to delete files out of it while leaving
    the read-only entry directories behind.

    Resolution follows the persistent JAX compilation cache: an explicit
    `FEEDBAX_REPO_SNAPSHOT_CACHE_DIR` override first, then the Git common directory of
    the running Feedbax checkout, which keeps sibling checkouts apart while letting
    worktrees of one checkout share. When Feedbax is not installed from a Git checkout
    there is no common directory, so the fall-back is a per-installation namespace under
    the user cache directory.
    """
    override = os.environ.get(REPO_SNAPSHOT_CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    package_root = Path(__file__).resolve().parents[2]
    common_dir = git_common_dir(package_root)
    if common_dir is not None:
        return common_dir / "feedbax_repo_snapshots"
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    namespace = hashlib.sha256(os.fsencode(package_root)).hexdigest()[:16]
    return cache_home / "feedbax" / "repo-snapshots" / namespace


def seal_repo_snapshots(
    repos: Mapping[str, Path | str],
    *,
    snapshot_parent: Path | str,
) -> SealedRepoSnapshots:
    """Seal exactly the tracked working-tree bytes for configured Git roots."""
    snapshots = {
        name: seal_repo_snapshot(name, root, snapshot_parent=snapshot_parent)
        for name, root in sorted(repos.items())
    }
    return SealedRepoSnapshots(
        manifest=RepoSnapshotManifest(
            repos={name: snapshot.record for name, snapshot in snapshots.items()}
        ),
        snapshots=snapshots,
    )


def seal_repo_snapshot(
    name: str,
    root: Path | str,
    *,
    snapshot_parent: Path | str,
) -> SealedRepoSnapshot:
    """Seal one Git top-level's tracked working-tree bytes without dereferencing links."""
    source_root = Path(root).expanduser().resolve()
    if _git(source_root, "rev-parse", "--show-cdup") != b"\n":
        raise RepoSnapshotError(
            f"configured repo root must equal the Git top level: {source_root}"
        )

    commit = os.fsdecode(_git(source_root, "rev-parse", "HEAD").strip())
    dirty = bool(_git(source_root, "status", "--porcelain=v1", "-z"))
    entries = _tracked_entries(source_root)
    if any(entry.mode == b"160000" for entry in entries):
        raise RepoSnapshotError(
            f"configured repo {name!r} contains a gitlink/submodule; govern it separately"
        )

    parent = Path(snapshot_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(tempfile.mkdtemp(prefix=".repo-snapshot-", dir=parent))
    try:
        for entry in entries:
            relative = _validated_relative_path(entry.path)
            source = source_root / relative
            if not source.is_symlink() and not source.exists():
                continue
            destination = build_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                target = os.readlink(source)
                os.symlink(target, destination)
            elif source.is_file():
                if entry.mode not in {b"100644", b"100755", b"120000"}:
                    raise RepoSnapshotError(f"tracked regular file has unsafe type: {relative}")
                with source.open("rb") as source_handle, destination.open("wb") as output:
                    while chunk := source_handle.read(1024 * 1024):
                        output.write(chunk)
                destination.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)
            else:
                raise RepoSnapshotError(
                    f"unsupported tracked Git mode {os.fsdecode(entry.mode)!r}: {relative}"
                )
        content_sha256, file_count = _snapshot_tree_identity(build_root)
        name_key = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        staging_root = parent / name_key / content_sha256
        staging_root.parent.mkdir(parents=True, exist_ok=True)
        _publish_sealed_snapshot(build_root, staging_root, content_sha256, file_count)
        record = RepoSnapshotRecord(
            commit=commit,
            dirty=dirty,
            content_sha256=content_sha256,
            file_count=file_count,
        )
        return SealedRepoSnapshot(
            name=name,
            source_root=source_root,
            staging_root=staging_root,
            record=record,
        )
    except Exception:
        if build_root.exists():
            _remove_tree(build_root)
        raise


def restore_repo_snapshots(
    repos: Mapping[str, Path | str],
    manifest: RepoSnapshotManifest,
    *,
    snapshot_parent: Path | str,
) -> SealedRepoSnapshots:
    """Restore and verify sealed local snapshot authority from durable state."""
    if set(repos) != set(manifest.repos):
        raise RepoSnapshotError("persisted repo snapshot names do not match configured repos")
    parent = Path(snapshot_parent).expanduser().resolve()
    snapshots: dict[str, SealedRepoSnapshot] = {}
    for name, root in sorted(repos.items()):
        record = manifest.repos[name]
        name_key = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        staging_root = parent / name_key / record.content_sha256
        if not staging_root.is_dir():
            raise RepoSnapshotCacheFault(f"sealed repo snapshot is unavailable: {staging_root}")
        if not _cache_entry_matches(staging_root, record.content_sha256, record.file_count):
            raise RepoSnapshotCacheFault(
                "sealed snapshot cache entry is damaged: its bytes no longer match the "
                f"content address holding them: {staging_root}"
            )
        _seal_tree_modes(staging_root)
        snapshots[name] = SealedRepoSnapshot(
            name=name,
            source_root=Path(root).expanduser().resolve(),
            staging_root=staging_root,
            record=record,
        )
    return SealedRepoSnapshots(manifest=manifest, snapshots=snapshots)


def snapshot_manifest_digest(manifest: RepoSnapshotManifest) -> str:
    """Return the stable identity used to bind snapshot bytes into reuse keys."""
    payload = manifest.model_dump_json(exclude_none=True, by_alias=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_repo_snapshot(
    root: Path | str,
    *,
    content_sha256: str,
    file_count: int,
) -> None:
    """Fail closed unless a transferred tree still matches its recorded sealed identity.

    This checks arbitrary bytes — typically a transfer destination — against a recorded
    manifest authority, so a mismatch is a genuine content mismatch and never a local
    cache fault. Sealed cache entries are checked against the content address holding
    them instead, and report `RepoSnapshotCacheFault`.
    """
    observed_sha256, observed_count = _snapshot_tree_identity(Path(root))
    if observed_sha256 != content_sha256 or observed_count != file_count:
        raise RepoSnapshotError(f"sealed repo snapshot digest mismatch: {root}")


def git_common_dir(root: Path) -> Path | None:
    """Return the Git common directory of a checkout, or None when there is no checkout."""
    try:
        result = subprocess_run(
            [
                "git",
                "-C",
                os.fspath(root),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            check=False,
            stdout=PIPE,
            stderr=PIPE,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    common_dir = os.fsdecode(result.stdout).strip()
    return Path(common_dir).resolve() if common_dir else None


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess_run(
            ["git", "-C", os.fspath(root), *args],
            check=True,
            stdout=PIPE,
            stderr=PIPE,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout
    except CalledProcessError as exc:
        detail = os.fsdecode(exc.stderr).strip() or f"exit={exc.returncode}"
        raise RepoSnapshotError(f"Git snapshot query failed for {root}: {detail}") from exc


def _tracked_entries(root: Path) -> list[_TrackedEntry]:
    records = _git(root, "ls-files", "--stage", "-z").split(b"\0")
    entries: list[_TrackedEntry] = []
    for record in records:
        if not record:
            continue
        metadata, separator, path_bytes = record.partition(b"\t")
        parts = metadata.split(b" ")
        if not separator or len(parts) != 3:
            raise RepoSnapshotError("git ls-files returned an invalid tracked-path record")
        mode, _object_id, stage = parts
        if stage != b"0":
            raise RepoSnapshotError(
                f"unmerged tracked path cannot be sealed: {os.fsdecode(path_bytes)!r}"
            )
        entries.append(_TrackedEntry(mode=mode, path_bytes=path_bytes))
    return entries


def _validated_relative_path(path: Path) -> Path:
    pure = PurePosixPath(path.as_posix())
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise RepoSnapshotError(f"unsafe tracked path: {path!s}")
    return Path(*pure.parts)


def _hash_field(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _publish_sealed_snapshot(
    build_root: Path,
    staging_root: Path,
    content_sha256: str,
    file_count: int,
) -> None:
    """Publish a verified build tree atomically at its content-addressed path.

    The tree is sealed read-only before it is published, so the rename makes an already
    final tree visible in one step and a concurrent reader can never observe a tree
    mid-seal. Sealing the top-level directory is the one exception: a directory whose own
    mode lacks write permission cannot be renamed, so it is sealed immediately after
    publication. That trailing chmod changes neither the tree digest, which encodes only
    the executable bit, nor its readability.

    A published entry whose bytes disagree with the content address holding them is a
    damaged cache entry rather than a content mismatch, so it is quarantined and replaced
    with the freshly verified tree instead of failing the run.
    """
    _seal_tree_modes(build_root, include_root=False)
    build_root.chmod(0o755)
    for _attempt in range(_PUBLISH_ATTEMPTS):
        try:
            os.replace(build_root, staging_root)
        except OSError:
            if not staging_root.exists():
                raise
        else:
            staging_root.chmod(0o555)
            return
        if _cache_entry_matches(staging_root, content_sha256, file_count):
            try:
                _seal_tree_modes(staging_root)
            except FileNotFoundError:
                # A concurrent repair replaced the entry between the check and the seal.
                continue
            _remove_tree(build_root)
            return
        _quarantine_cache_entry(staging_root)
    raise RepoSnapshotCacheFault(
        f"sealed snapshot cache entry could not be repaired after "
        f"{_PUBLISH_ATTEMPTS} attempts: {staging_root}"
    )


def _cache_entry_matches(root: Path, content_sha256: str, file_count: int) -> bool:
    """Report whether a cache entry still holds the bytes its path claims."""
    try:
        observed_sha256, observed_count = _snapshot_tree_identity(root)
    except (OSError, RepoSnapshotError):
        return False
    return observed_sha256 == content_sha256 and observed_count == file_count


def _quarantine_cache_entry(staging_root: Path) -> None:
    """Move a damaged cache entry aside so a verified tree can take its place."""
    quarantine = Path(
        tempfile.mkdtemp(prefix=f".damaged-{staging_root.name}-", dir=staging_root.parent)
    )
    try:
        staging_root.chmod(0o755)
        os.replace(staging_root, quarantine)
    except FileNotFoundError:
        # A concurrent sealer already repaired the entry; the caller retries.
        quarantine.rmdir()
        return
    except OSError as exc:
        quarantine.rmdir()
        raise RepoSnapshotCacheFault(
            f"damaged sealed snapshot cache entry cannot be quarantined: {staging_root}"
        ) from exc
    try:
        _remove_tree(quarantine)
    except OSError:
        # The entry is already out of the content-addressed namespace and inert; leaving
        # the quarantined bytes behind must not fail the run that repaired the cache.
        pass


def _seal_tree_modes(root: Path, *, include_root: bool = True) -> None:
    """Remove write permission from the content-addressed authority tree."""
    root_key = os.path.normpath(os.fspath(root))
    for directory, _subdirs, files in os.walk(root):
        directory_path = Path(directory)
        for filename in files:
            path = directory_path / filename
            if not path.is_symlink():
                path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
        if include_root or os.path.normpath(directory) != root_key:
            directory_path.chmod(0o555)


def _remove_tree(root: Path) -> None:
    for directory, subdirs, files in os.walk(root):
        directory_path = Path(directory)
        directory_path.chmod(0o755)
        for name in [*subdirs, *files]:
            path = directory_path / name
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() else 0o644)
    shutil.rmtree(root)


def _snapshot_tree_identity(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    entries = _snapshot_entries(root)
    for relative_bytes, path in entries:
        _hash_field(digest, relative_bytes)
        if path.is_symlink():
            _hash_field(digest, b"120000")
            _hash_field(digest, os.fsencode(os.readlink(path)))
            continue
        mode = b"100755" if path.stat().st_mode & 0o111 else b"100644"
        _hash_field(digest, mode)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(len(chunk).to_bytes(8, "big"))
                digest.update(chunk)
        digest.update((0).to_bytes(8, "big"))
    return digest.hexdigest(), len(entries)


def _snapshot_entries(root: Path) -> list[tuple[bytes, Path]]:
    entries: list[tuple[bytes, Path]] = []

    def visit(directory: Path) -> None:
        children = sorted(os.scandir(directory), key=lambda entry: os.fsencode(entry.name))
        for child in children:
            path = Path(child.path)
            relative_bytes = os.fsencode(path.relative_to(root).as_posix())
            if child.is_symlink():
                entries.append((relative_bytes, path))
            elif child.is_dir(follow_symlinks=False):
                visit(path)
            elif child.is_file(follow_symlinks=False):
                entries.append((relative_bytes, path))
            else:
                raise RepoSnapshotError(f"sealed snapshot contains unsafe file type: {path}")

    visit(root)
    entries.sort(key=lambda item: item[0])
    return entries


def _main() -> None:
    parser = argparse.ArgumentParser(description="Verify a sealed Feedbax repo snapshot")
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--root", required=True)
    parser.add_argument("--content-sha256", required=True)
    parser.add_argument("--file-count", required=True, type=int)
    args = parser.parse_args()
    verify_repo_snapshot(
        args.root,
        content_sha256=args.content_sha256,
        file_count=args.file_count,
    )


if __name__ == "__main__":
    _main()
