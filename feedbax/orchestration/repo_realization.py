"""Content-addressed plans for realizing sealed local repositories."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.manifest import StrictModel
from feedbax.orchestration.repo_snapshot import (
    RepoSnapshotError,
    RepoSnapshotManifest,
    RepoSnapshotRecord,
    SealedRepoSnapshot,
    seal_repo_snapshots,
)


REPO_REALIZATION_PLAN_SCHEMA_ID = "feedbax.orchestration.repo_realization_plan"
REPO_REALIZATION_PLAN_SCHEMA_VERSION = (
    "feedbax.orchestration.repo_realization_plan.v1"
)


class RepoRealizationError(RepoSnapshotError):
    """Raised when sealed repository bytes cannot be realized safely."""


class RepoRealizationEntry(StrictModel):
    """The roots and sealed identity of one repository in a realization plan."""

    # These roots aid diagnostics and restore lookup, but are not portable identity.
    local_root: str
    staging_root: str
    remote_root: str
    snapshot: RepoSnapshotRecord
    sealed_lock_digests: dict[str, str] = Field(default_factory=dict)

    @field_validator("sealed_lock_digests")
    @classmethod
    def _sort_lock_digests(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for relative_path, digest in value.items():
            key = _canonical_relative_posix_path(relative_path, allow_dot=False)
            if key in normalized:
                raise ValueError(f"duplicate canonical lock-relative path: {key!r}")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"invalid sealed lock digest for {key!r}")
            normalized[key] = digest
        return dict(sorted(normalized.items()))

    @field_validator("remote_root")
    @classmethod
    def _normalize_remote_root(cls, value: str) -> str:
        return _canonical_remote_root(value)


class EditableSourceResolution(StrictModel):
    """One fully keyed editable source resolved before provider acquisition."""

    consumer_repo: str
    lock_relative_path: str
    source_form: str
    spelling: str
    target_repo: str
    target_subpath: str

    @field_validator("lock_relative_path")
    @classmethod
    def _validate_lock_relative_path(cls, value: str) -> str:
        return _canonical_relative_posix_path(value, allow_dot=False)

    @field_validator("spelling")
    @classmethod
    def _validate_spelling(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or not value or value != path.as_posix():
            raise ValueError(f"editable source spelling must be canonical and relative: {value!r}")
        return value

    @field_validator("target_subpath")
    @classmethod
    def _validate_target_subpath(cls, value: str) -> str:
        return _canonical_relative_posix_path(value, allow_dot=True)

    def sort_key(self) -> tuple[str, str, str, str]:
        """Return the complete consumer-side identity key."""
        return (
            self.consumer_repo,
            self.lock_relative_path,
            self.source_form,
            self.spelling,
        )


class RepoRealizationPlan(StrictModel):
    """Portable, content-addressed authority for a multi-repository transfer."""

    schema_id: Literal[REPO_REALIZATION_PLAN_SCHEMA_ID] = REPO_REALIZATION_PLAN_SCHEMA_ID
    schema_version: Literal[REPO_REALIZATION_PLAN_SCHEMA_VERSION] = (
        REPO_REALIZATION_PLAN_SCHEMA_VERSION
    )
    primary_repo: str
    repos: dict[str, RepoRealizationEntry]
    editable_source_resolutions: list[EditableSourceResolution] = Field(default_factory=list)
    snapshot_manifest: RepoSnapshotManifest
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("repos")
    @classmethod
    def _sort_repos(cls, value: dict[str, RepoRealizationEntry]) -> dict[str, RepoRealizationEntry]:
        return dict(sorted(value.items()))

    @field_validator("editable_source_resolutions")
    @classmethod
    def _sort_resolutions(
        cls, value: list[EditableSourceResolution]
    ) -> list[EditableSourceResolution]:
        ordered = sorted(value, key=EditableSourceResolution.sort_key)
        keys = [item.sort_key() for item in ordered]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate fully keyed editable source resolution")
        return ordered

    @model_validator(mode="after")
    def _validate_authority(self) -> RepoRealizationPlan:
        repo_keys = set(self.repos)
        if self.primary_repo not in repo_keys:
            raise ValueError(
                f"primary_repo {self.primary_repo!r} is not a planned repo key: "
                f"{sorted(repo_keys)!r}"
            )
        if repo_keys != set(self.snapshot_manifest.repos):
            raise ValueError("plan repo keys do not match snapshot manifest repo keys")
        for name, entry in self.repos.items():
            if entry.snapshot != self.snapshot_manifest.repos[name]:
                raise ValueError(f"plan snapshot differs from manifest for repo {name!r}")
        validate_non_overlapping_remote_roots(
            {name: entry.remote_root for name, entry in self.repos.items()}
        )
        for resolution in self.editable_source_resolutions:
            if resolution.consumer_repo not in repo_keys:
                raise ValueError(
                    f"editable resolution has unknown consumer repo "
                    f"{resolution.consumer_repo!r}"
                )
            if resolution.target_repo not in repo_keys:
                raise ValueError(
                    f"editable resolution has unknown target repo {resolution.target_repo!r}"
                )
            if resolution.lock_relative_path not in self.repos[
                resolution.consumer_repo
            ].sealed_lock_digests:
                raise ValueError(
                    "editable resolution names an unsealed consumer lock: "
                    f"{resolution.consumer_repo}:{resolution.lock_relative_path}"
                )
        expected = repo_realization_plan_digest(self)
        if self.plan_digest != expected:
            raise ValueError(
                f"repo realization plan digest mismatch: expected {expected}, "
                f"observed {self.plan_digest}"
            )
        return self

    @classmethod
    def create(
        cls,
        *,
        primary_repo: str,
        repos: Mapping[str, RepoRealizationEntry],
        editable_source_resolutions: Sequence[EditableSourceResolution],
        snapshot_manifest: RepoSnapshotManifest,
    ) -> RepoRealizationPlan:
        """Create a plan with its canonical digest populated."""
        payload: dict[str, Any] = {
            "primary_repo": primary_repo,
            "repos": dict(repos),
            "editable_source_resolutions": list(editable_source_resolutions),
            "snapshot_manifest": snapshot_manifest,
        }
        unchecked = cls.model_construct(
            schema_id=REPO_REALIZATION_PLAN_SCHEMA_ID,
            schema_version=REPO_REALIZATION_PLAN_SCHEMA_VERSION,
            plan_digest="0" * 64,
            **payload,
        )
        return cls(**payload, plan_digest=repo_realization_plan_digest(unchecked))


@dataclass(frozen=True)
class SealedLocalRepoRealization:
    """Shared component result for one sealed local repository."""

    snapshot: SealedRepoSnapshot
    sealed_lock_digests: Mapping[str, str]


def seal_local_repo_realizations(
    repos: Mapping[str, Path | str],
    *,
    lock_relative_paths: Mapping[str, Sequence[str]],
    snapshot_parent: Path | str,
    expected_lock_digests: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[RepoSnapshotManifest, Mapping[str, SealedLocalRepoRealization]]:
    """Seal repos once, then digest declared locks exclusively from sealed bytes."""
    unknown = set(lock_relative_paths) - set(repos)
    if unknown:
        raise RepoRealizationError(f"lock paths name unknown repos: {sorted(unknown)!r}")
    expected = expected_lock_digests or {}
    unknown_expected = set(expected) - set(repos)
    if unknown_expected:
        raise RepoRealizationError(
            f"expected lock digests name unknown repos: {sorted(unknown_expected)!r}"
        )
    sealed = seal_repo_snapshots(repos, snapshot_parent=snapshot_parent)
    realized: dict[str, SealedLocalRepoRealization] = {}
    for name, snapshot in sorted(sealed.snapshots.items()):
        digests: dict[str, str] = {}
        for relative_path in sorted(lock_relative_paths.get(name, ())):
            canonical = _canonical_relative_posix_path(relative_path, allow_dot=False)
            if canonical in digests:
                raise RepoRealizationError(
                    f"duplicate canonical lock-relative path for repo {name!r}: {canonical!r}"
                )
            digest = sealed_lock_digest(snapshot.staging_root, canonical)
            declared = expected.get(name, {}).get(canonical)
            if declared is not None and declared != digest:
                raise RepoRealizationError(
                    f"declared lockfile hash mismatch for {name}:{canonical}: "
                    f"expected {declared}, observed {digest}"
                )
            digests[canonical] = digest
        extra_expected = set(expected.get(name, {})) - set(digests)
        if extra_expected:
            raise RepoRealizationError(
                f"expected lock digests lack declared lock paths for repo {name!r}: "
                f"{sorted(extra_expected)!r}"
            )
        realized[name] = SealedLocalRepoRealization(
            snapshot=snapshot,
            sealed_lock_digests=dict(sorted(digests.items())),
        )
    return sealed.manifest, realized


def sealed_lock_digest(staging_root: Path | str, lock_relative_path: str) -> str:
    """Hash a regular, non-symlink lock inside a sealed staging root."""
    return hashlib.sha256(
        read_sealed_lock_bytes(staging_root, lock_relative_path)
    ).hexdigest()


def read_sealed_lock_bytes(
    staging_root: Path | str, lock_relative_path: str
) -> bytes:
    """Read a regular lock from a sealed root without following symlinks."""
    root = Path(staging_root)
    relative = _canonical_relative_posix_path(lock_relative_path, allow_dot=False)
    path = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise RepoRealizationError(f"cannot inspect sealed lock {path}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RepoRealizationError(f"sealed lock must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise RepoRealizationError(f"sealed lock must be a regular file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RepoRealizationError(
            f"cannot open sealed lock without following links {path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RepoRealizationError(f"sealed lock must be a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def validate_non_overlapping_remote_roots(
    remote_roots: Mapping[str, str],
) -> dict[str, str]:
    """Normalize absolute POSIX roots and reject any overlapping pair."""
    normalized = {
        name: _canonical_remote_root(root) for name, root in sorted(remote_roots.items())
    }
    items = list(normalized.items())
    for index, (left_name, left_raw) in enumerate(items):
        left = PurePosixPath(left_raw)
        for right_name, right_raw in items[index + 1 :]:
            right = PurePosixPath(right_raw)
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise RepoRealizationError(
                    "overlapping remote repo roots are unsafe for wholesale --delete: "
                    f"{left_name}={left_raw!r}, {right_name}={right_raw!r}"
                )
    return normalized


def repo_realization_plan_projection(plan: RepoRealizationPlan) -> dict[str, Any]:
    """Return the single canonical, location-independent plan identity projection."""
    payload = plan.model_dump(mode="json", exclude={"plan_digest"})
    for entry in payload["repos"].values():
        entry.pop("local_root", None)
        entry.pop("staging_root", None)
        entry["sealed_lock_digests"] = dict(sorted(entry["sealed_lock_digests"].items()))
    payload["repos"] = dict(sorted(payload["repos"].items()))
    payload["snapshot_manifest"]["repos"] = dict(
        sorted(payload["snapshot_manifest"]["repos"].items())
    )
    payload["editable_source_resolutions"] = [
        resolution.model_dump(mode="json")
        for resolution in sorted(
            plan.editable_source_resolutions,
            key=EditableSourceResolution.sort_key,
        )
    ]
    return payload


def repo_realization_plan_digest(plan: RepoRealizationPlan) -> str:
    """Return the SHA-256 digest of the canonical plan identity projection."""
    encoded = json.dumps(
        repo_realization_plan_projection(plan),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_remote_root(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute():
        raise ValueError(f"remote repo root must be absolute: {value!r}")
    return posixpath.normpath(path.as_posix())


def _canonical_relative_posix_path(value: str, *, allow_dot: bool) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not value or value != path.as_posix():
        raise ValueError(f"path must be canonical and relative: {value!r}")
    if any(part == ".." for part in path.parts):
        raise ValueError(f"path may not escape its repository: {value!r}")
    if value == "." and not allow_dot:
        raise ValueError("path must name a file below the repository root")
    return value
