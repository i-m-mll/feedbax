"""Fail-closed resolution of declared evaluation manifest inputs."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError

from feedbax.analysis.execution_context import (
    EMPTY_STAGED_EXECUTION_CONTEXT,
    StagedExecutionContext,
    StagedExecutionContextError,
)
from feedbax.analysis.manifest_inputs import is_authenticated_manifest_ref
from feedbax.contracts.manifest import EvaluationRunSpec, ParentRef, TrainingRunManifest


_TRAINING_MANIFEST_KIND = "TrainingRunManifest"
_TRAINING_MANIFEST_ROLE = "training_run"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd


class EvaluationInputResolutionError(ValueError):
    """Base error for evaluation input resolution failures."""


class EvaluationInputCardinalityError(EvaluationInputResolutionError):
    """Raised when an evaluation spec does not declare exactly one input."""


class EvaluationInputReferenceError(EvaluationInputResolutionError):
    """Raised when an input ref is missing, duplicated, or has the wrong contract."""


class EvaluationInputPathError(EvaluationInputResolutionError):
    """Raised when an input path is unsupported or escapes the allowed root."""


class EvaluationInputAmbiguityError(EvaluationInputResolutionError):
    """Raised when an input ID resolves to more than one manifest file."""


class EvaluationInputIntegrityError(EvaluationInputResolutionError):
    """Raised when a declared manifest hash is invalid or does not match."""


class EvaluationInputManifestError(EvaluationInputResolutionError):
    """Raised when resolved bytes are not the exact declared typed manifest."""


@dataclass(frozen=True, slots=True)
class ResolvedEvaluationInput:
    """A verified training-run parent declared by an evaluation spec.

    For context-free and legacy resolution, ``path`` is canonical beneath the
    explicit manifest root and ``reference`` is relative to that root. A complete
    authenticated ref with a matching staged location instead uses that separately
    retained exact authority, which may differ from the row/output root. ``sha256``
    and ``size_bytes`` describe the same bytes used to validate ``manifest``.
    """

    ref: ParentRef
    manifest: TrainingRunManifest
    path: Path
    reference: str
    sha256: str
    size_bytes: int

    @property
    def kind(self) -> str:
        """Return the verified manifest kind."""
        return self.manifest.kind

    @property
    def id(self) -> str:
        """Return the verified manifest ID."""
        return self.manifest.id


@dataclass(frozen=True, slots=True)
class _ManifestCandidate:
    path: Path
    reference: str
    raw_bytes: bytes
    payload: object


def resolve_evaluation_inputs(
    run_spec: EvaluationRunSpec,
    *,
    manifest_root: Path | str,
    require_unique_manifest_id: bool = True,
    execution_context: StagedExecutionContext = EMPTY_STAGED_EXECUTION_CONTEXT,
) -> tuple[ResolvedEvaluationInput, ...]:
    """Resolve the exact training manifest declared by an evaluation run spec.

    Resolution is read-only. Context-free and legacy inputs resolve beneath the
    mandatory ``manifest_root``. A complete authenticated ref with a matching
    staged location resolves through that separately retained exact authority,
    which may differ from the row/output root. Resolution never consults an index,
    an ambient default root, ``training_run_ids``, or a latest pointer.

    Args:
        run_spec: Registered evaluation recipe's validated run specification.
        manifest_root: Explicit filesystem root mandatory for non-staged resolution.
        require_unique_manifest_id: Whether an explicit URI must also be the
            only manifest under ``manifest_root/manifests`` with the declared
            ID. Disable only when a separate frozen authority already binds an
            exact URI and immutable content digest.

    Returns:
        A one-item tuple containing the verified training-run manifest input.

    Raises:
        EvaluationInputResolutionError: If any declaration, path, hash, or
            typed-manifest invariant is not satisfied.
    """
    refs = tuple(run_spec.inputs)
    if not refs:
        raise EvaluationInputCardinalityError(
            "EvaluationRunSpec.inputs must contain exactly one training manifest; got zero"
        )
    if len(refs) != 1:
        keys = [(ref.kind, ref.id, ref.role, ref.uri) for ref in refs]
        if len(set(keys)) != len(keys):
            raise EvaluationInputReferenceError(
                "EvaluationRunSpec.inputs contains a duplicate ParentRef declaration"
            )
        raise EvaluationInputCardinalityError(
            f"EvaluationRunSpec.inputs must contain exactly one training manifest; got {len(refs)}"
        )

    ref = refs[0]
    _validate_training_ref(ref)
    try:
        authenticated = is_authenticated_manifest_ref(ref)
    except ValueError as exc:
        raise EvaluationInputReferenceError(str(exc)) from exc
    if authenticated and execution_context.parent_execution_locations:
        try:
            resolved = execution_context.resolve_manifest_input(ref)
            location = execution_context.parent_execution_location(ref)
        except (StagedExecutionContextError, ValueError) as exc:
            raise EvaluationInputReferenceError(
                "authenticated evaluation input requires its matching complete-ParentRef "
                "staged execution authority"
            ) from exc
        if not isinstance(resolved.manifest, TrainingRunManifest):
            raise EvaluationInputManifestError(
                "Staged evaluation input did not resolve to a TrainingRunManifest"
            )
        digest = hashlib.sha256(resolved.raw_bytes).hexdigest()
        return (
            ResolvedEvaluationInput(
                ref=ref,
                manifest=resolved.manifest,
                path=resolved.path,
                reference=location.execution_uri,
                sha256=digest,
                size_bytes=len(resolved.raw_bytes),
            ),
        )
    effective_ref = ref
    requested_root = _resolve_root(manifest_root)
    if (ref.uri or "").startswith("artifact://sha256/"):
        try:
            location = execution_context.parent_execution_location(ref)
        except StagedExecutionContextError as exc:
            raise EvaluationInputReferenceError(
                "immutable exact evaluation input requires a matching complete-ParentRef "
                "execution location"
            ) from exc
        root = _resolve_root(location.root)
        if root != requested_root:
            raise EvaluationInputPathError(
                "immutable exact evaluation input location root does not match manifest_root"
            )
        effective_ref = ref.model_copy(update={"uri": location.execution_uri})
        require_unique_manifest_id = False
    else:
        root = requested_root
    with _open_root_fd(root) as root_fd:
        candidate = _resolve_candidate(
            effective_ref,
            root,
            root_fd,
            require_unique_manifest_id=require_unique_manifest_id,
        )
    digest = hashlib.sha256(candidate.raw_bytes).hexdigest()
    _validate_declared_hash(ref, digest)
    _validate_declared_size(ref, len(candidate.raw_bytes))
    manifest = _validate_training_manifest(candidate, ref)

    return (
        ResolvedEvaluationInput(
            ref=ref,
            manifest=manifest,
            path=candidate.path,
            reference=candidate.reference,
            sha256=digest,
            size_bytes=len(candidate.raw_bytes),
        ),
    )


def _validate_training_ref(ref: ParentRef) -> None:
    if ref.role != _TRAINING_MANIFEST_ROLE:
        raise EvaluationInputReferenceError(
            f"Evaluation input ParentRef role must be 'training_run'; got {ref.role!r}"
        )
    if ref.kind != _TRAINING_MANIFEST_KIND:
        raise EvaluationInputReferenceError(
            f"Evaluation input ParentRef kind must be 'TrainingRunManifest'; got {ref.kind!r}"
        )
    if not ref.id.strip():
        raise EvaluationInputReferenceError("Evaluation input ParentRef id must not be empty")


def _resolve_root(manifest_root: Path | str) -> Path:
    root = Path(manifest_root).expanduser()
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise EvaluationInputPathError(f"Allowed manifest root does not exist: {root}") from exc
    if not resolved.is_dir():
        raise EvaluationInputPathError(f"Allowed manifest root is not a directory: {resolved}")
    return resolved


def _resolve_candidate(
    ref: ParentRef,
    root: Path,
    root_fd: int,
    *,
    require_unique_manifest_id: bool,
) -> _ManifestCandidate:
    if ref.uri is not None:
        selected_reference = _resolve_ref_uri(ref.uri)
        selected = _read_candidate(selected_reference, root, root_fd)
        payload = selected.payload
        if not isinstance(payload, dict):
            raise EvaluationInputManifestError(
                f"Resolved manifest JSON must be an object: {selected.path}"
            )
        loaded_id = payload.get("id")
        if loaded_id != ref.id:
            raise EvaluationInputManifestError(
                "Resolved manifest id does not match ParentRef: "
                f"declared={ref.id!r}, loaded={loaded_id!r}"
            )
        if not require_unique_manifest_id:
            return selected
        matches = _find_id_candidates(ref.id, root, root_fd, selected=selected)
        _require_unambiguous(ref.id, matches)
        if matches[0].path != selected.path:
            raise EvaluationInputAmbiguityError(
                f"ParentRef URI does not select the unique manifest for id {ref.id!r}"
            )
        return selected

    if not require_unique_manifest_id:
        raise EvaluationInputReferenceError(
            "require_unique_manifest_id=False requires an explicit ParentRef uri"
        )

    matches = _find_id_candidates(ref.id, root, root_fd)
    return _require_unambiguous(ref.id, matches)


def _resolve_ref_uri(uri: str) -> Path:
    if not uri.strip():
        raise EvaluationInputPathError("Evaluation input ParentRef uri must not be empty")
    parsed = urlsplit(uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise EvaluationInputPathError(
            f"Evaluation input ParentRef uri must be a relative local path: {uri!r}"
        )
    decoded = unquote(parsed.path)
    if "\\" in decoded:
        raise EvaluationInputPathError(
            f"Evaluation input ParentRef uri contains an unsupported path separator: {uri!r}"
        )
    relative = Path(decoded)
    if relative.is_absolute() or ".." in relative.parts:
        raise EvaluationInputPathError(
            f"Evaluation input ParentRef uri escapes the allowed manifest root: {uri!r}"
        )
    return relative


@contextmanager
def _open_root_fd(root: Path) -> Iterator[int]:
    if not _OPEN_SUPPORTS_DIR_FD or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise EvaluationInputPathError(
            "Secure descriptor-relative manifest resolution is unavailable on this platform"
        )
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not directory_flag:
        raise EvaluationInputPathError(
            "Secure descriptor-relative manifest resolution requires O_DIRECTORY"
        )
    flags = os.O_RDONLY | directory_flag | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        root_fd = os.open(root, flags)
    except OSError as exc:
        raise EvaluationInputPathError(f"Unable to securely open manifest root: {root}") from exc
    try:
        yield root_fd
    finally:
        os.close(root_fd)


def _read_candidate(relative: Path, root: Path, root_fd: int) -> _ManifestCandidate:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise EvaluationInputPathError(
            f"Evaluation input manifest path escapes the allowed root: {relative}"
        )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    opened_directories: list[int] = []
    parent_fd = root_fd
    try:
        for component in relative.parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            opened_directories.append(next_fd)
            parent_fd = next_fd
        file_fd = os.open(relative.parts[-1], file_flags, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            error = EvaluationInputPathError(
                f"Evaluation input manifest path contains a symlink or non-directory: {relative}"
            )
        elif exc.errno == errno.ENOENT:
            error = EvaluationInputReferenceError(
                f"Evaluation input manifest does not exist: {relative}"
            )
        else:
            error = EvaluationInputReferenceError(
                f"Unable to securely open evaluation input manifest: {relative}"
            )
        raise error from exc
    finally:
        for directory_fd in reversed(opened_directories):
            os.close(directory_fd)

    try:
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise EvaluationInputPathError(
                f"Evaluation input manifest is not a regular file: {relative}"
            )
        with os.fdopen(file_fd, "rb", closefd=True) as file:
            raw_bytes = file.read()
    except Exception:
        try:
            os.close(file_fd)
        except OSError:
            pass
        raise

    path = root.joinpath(*relative.parts)
    try:
        payload: object = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationInputManifestError(
            f"Evaluation input manifest is not valid JSON: {path}"
        ) from exc
    return _ManifestCandidate(
        path=path,
        reference=relative.as_posix(),
        raw_bytes=raw_bytes,
        payload=payload,
    )


def _find_id_candidates(
    manifest_id: str,
    root: Path,
    root_fd: int,
    *,
    selected: _ManifestCandidate | None = None,
) -> list[_ManifestCandidate]:
    """Enumerate every manifest under ``root`` that declares ``manifest_id``.

    This is a *uniqueness census*, not a search that stops at the first hit, so
    every discovered candidate has to be decided. A candidate that cannot be
    read or parsed is undecidable: it may well be a second copy of the very ID
    being resolved, and skipping it makes the surviving copy look unique. So an
    undecidable candidate refuses the census instead of disappearing from it.
    """
    manifests_dir = root / "manifests"
    paths = sorted(manifests_dir.glob("**/*.json")) if manifests_dir.is_dir() else []
    candidates: dict[str, _ManifestCandidate] = {}
    if selected is not None:
        candidates[selected.reference] = selected

    for discovered in paths:
        try:
            relative = discovered.relative_to(root)
        except ValueError:
            continue
        reference = relative.as_posix()
        if reference in candidates:
            continue
        try:
            candidate = _read_candidate(relative, root, root_fd)
        except EvaluationInputResolutionError as exc:
            raise EvaluationInputAmbiguityError(
                f"Manifest {manifest_id!r} cannot be proved unique under the allowed root: "
                f"candidate {reference} is unreadable or malformed ({exc}). A uniqueness census "
                "decides every candidate it discovers; one it cannot decide might be the "
                "duplicate, so it is a refusal and never a skip."
            ) from exc
        if isinstance(candidate.payload, dict) and candidate.payload.get("id") == manifest_id:
            candidates[reference] = candidate
    return [candidates[reference] for reference in sorted(candidates)]


def _require_unambiguous(
    manifest_id: str,
    candidates: list[_ManifestCandidate],
) -> _ManifestCandidate:
    if not candidates:
        raise EvaluationInputReferenceError(f"Manifest {manifest_id!r} was not found")
    if len(candidates) > 1:
        paths = ", ".join(str(candidate.path) for candidate in candidates)
        raise EvaluationInputAmbiguityError(
            f"Manifest {manifest_id!r} is duplicated under the allowed root: {paths}"
        )
    return candidates[0]


def _validate_declared_hash(ref: ParentRef, observed: str) -> None:
    declared = ref.metadata.get("manifest_sha256")
    if declared is None:
        return
    if not isinstance(declared, str) or _SHA256_PATTERN.fullmatch(declared) is None:
        raise EvaluationInputIntegrityError(
            "ParentRef metadata.manifest_sha256 must be exactly 64 lowercase hex characters"
        )
    if declared != observed:
        raise EvaluationInputIntegrityError(
            "Resolved manifest SHA-256 does not match ParentRef metadata.manifest_sha256: "
            f"declared={declared}, observed={observed}"
        )


def _validate_declared_size(ref: ParentRef, observed: int) -> None:
    declared = ref.metadata.get("size_bytes")
    if declared is None:
        return
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
        raise EvaluationInputIntegrityError(
            "ParentRef metadata.size_bytes must be a nonnegative integer"
        )
    if declared != observed:
        raise EvaluationInputIntegrityError(
            "Resolved manifest byte size does not match ParentRef metadata.size_bytes: "
            f"declared={declared}, observed={observed}"
        )


def _validate_training_manifest(
    candidate: _ManifestCandidate,
    ref: ParentRef,
) -> TrainingRunManifest:
    try:
        manifest = TrainingRunManifest.model_validate_json(candidate.raw_bytes)
    except ValidationError as exc:
        raise EvaluationInputManifestError(
            f"Resolved input is not a valid TrainingRunManifest: {candidate.path}"
        ) from exc
    if manifest.kind != ref.kind:
        raise EvaluationInputManifestError(
            "Resolved manifest kind does not match ParentRef: "
            f"declared={ref.kind!r}, loaded={manifest.kind!r}"
        )
    if manifest.id != ref.id:
        raise EvaluationInputManifestError(
            "Resolved manifest id does not match ParentRef: "
            f"declared={ref.id!r}, loaded={manifest.id!r}"
        )
    return manifest
