"""Local observations of verified failures at canonical object addresses."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
import os
from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ArtifactBlobIntegrityError, ContentObjectRef
from feedbax.contracts.base import StrictModel
from feedbax.contracts.request_refs import Sha256
from feedbax.persistence._store_metadata import sync_directory, write_store_metadata

if TYPE_CHECKING:
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


_explicit_checks: ContextVar[frozenset[Path]] = ContextVar("explicit_store_checks", default=frozenset())
_quiet_reads: ContextVar[bool] = ContextVar("inspect_store_without_recording_damage", default=False)


class ObjectVerificationError(ArtifactBlobIntegrityError):
    """Stable canonical bytes disagree with the expected size or digest."""

    def __init__(
        self, detail: str, *, found_size: int, found_sha256: str | None = None,
    ) -> None:
        self.found_size = found_size
        self.found_sha256 = found_sha256
        super().__init__(detail)


class KnownDamage(ArtifactBlobIntegrityError):
    """An object needed by a reader has an outstanding verification failure."""

    def __init__(self, observation: DamageObservation) -> None:
        self.observation = observation
        super().__init__(observation.reason)


@contextmanager
def explicit_check(store: ImmutableArtifactBlobProvider) -> Iterator[None]:
    token = _explicit_checks.set(_explicit_checks.get() | {store.root})
    try:
        yield
    finally:
        _explicit_checks.reset(token)


@contextmanager
def inspect_without_recording() -> Iterator[None]:
    token = _quiet_reads.set(True)
    try:
        yield
    finally:
        _quiet_reads.reset(token)


class DamageObservation(StrictModel):
    expected: ContentObjectRef
    found: str
    verifier: str
    observed_at: datetime
    descriptor_sha256: Sha256 | None = None
    occurrence: tuple[str, ...] | None
    found_size: int | None = None
    found_sha256: Sha256 | None = None

    @property
    def reason(self) -> str:
        return f"Damaged {self.expected.object_id}: {self.found}"


def _path(store: ImmutableArtifactBlobProvider, digest: str) -> Path:
    from feedbax.persistence.artifact_custody import content_object_relative_path

    _, _, prefix, name = content_object_relative_path(digest).parts
    return store.root / "integrity" / prefix / name


def read_damage(store: ImmutableArtifactBlobProvider, digest: str) -> DamageObservation | None:
    from feedbax.persistence.artifact_custody import _read_store_document

    path = _path(store, digest)
    if not os.path.lexists(store.root / "integrity"):
        return None
    record = _read_store_document(path)
    if record is None:
        return None
    observation = DamageObservation.model_validate(record[1])
    if observation.expected.sha256 != digest:
        raise ArtifactBlobIntegrityError(f"Damage observation names a different object: {path}")
    return observation


def record_damage(
    store: ImmutableArtifactBlobProvider, ref: ContentObjectRef, error: Exception,
    *, verifier: str, descriptor_sha256: str | None = None,
) -> None:
    """Record a failure while the verifier holds the Store lock."""
    from feedbax.persistence.artifact_custody import _encode_store_document

    previous = read_damage(store, ref.sha256)
    if descriptor_sha256 is None and previous is not None:
        descriptor_sha256 = previous.descriptor_sha256
    current = occurrence(store, ref)
    observation = DamageObservation(
        expected=ref, found=str(error), verifier=verifier, observed_at=datetime.now(UTC),
        descriptor_sha256=descriptor_sha256,
        occurrence=tuple(map(str, current)) if current is not None else None,
        found_size=error.found_size if isinstance(error, ObjectVerificationError) else None,
        found_sha256=error.found_sha256 if isinstance(error, ObjectVerificationError) else None,
    )
    write_store_metadata(
        store.root, _path(store, ref.sha256),
        _encode_store_document(observation.model_dump(mode="json")),
        error_factory=ArtifactBlobIntegrityError,
    )


def clear_damage(store: ImmutableArtifactBlobProvider, digest: str) -> None:
    """Remove a settled observation while holding the Store lock."""
    path = _path(store, digest)
    if read_damage(store, digest) is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return
    sync_directory(path.parent)


def require_sound(store: ImmutableArtifactBlobProvider, ref: ContentObjectRef) -> None:
    if store.root in _explicit_checks.get():
        return
    if observation := read_damage(store, ref.sha256):
        raise KnownDamage(observation)


def occurrence(store: ImmutableArtifactBlobProvider, ref: ContentObjectRef) -> tuple[int, ...] | None:
    from feedbax.persistence.artifact_custody import _file_state

    try:
        return _file_state(store.object_path(ref).lstat())
    except FileNotFoundError:
        return None


def observe_failure(
    store: ImmutableArtifactBlobProvider, ref: ContentObjectRef, before: tuple[int, ...] | None,
    error: Exception, *, verifier: str, descriptor_sha256: str | None = None,
) -> None:
    if _quiet_reads.get():
        return
    with store._exclusive_lock(create=False):
        if occurrence(store, ref) == before:
            record_damage(store, ref, error, verifier=verifier, descriptor_sha256=descriptor_sha256)
