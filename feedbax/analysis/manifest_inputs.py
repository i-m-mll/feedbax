"""Portable authentication and local resolution for staged manifest inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from feedbax._secure_fs import (
    SecurePathRigor,
    close_descriptors,
    open_directory_chain,
    open_existing_directory,
    open_existing_file,
    validate_opened_path,
)
from feedbax.contracts.manifest import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    AnyManifest,
    ParentRef,
    authenticated_manifest_ref_profile,
    canonical_manifest_relative_path,
    load_manifest_bytes,
)


#: Manifest kinds admitted as authenticated staged inputs. Their durable
#: locations come from the canonical layout, never from a local copy of it.
_STAGED_MANIFEST_KINDS = frozenset(
    {
        "TrainingRunManifest",
        "EvaluationRunManifest",
        "AnalysisRunManifest",
        "FigureManifest",
        "ReportManifest",
    }
)


@dataclass(frozen=True)
class ResolvedManifestInput:
    """Authenticated manifest bytes resolved at one machine-local path."""

    ref: ParentRef
    manifest: AnyManifest
    path: Path
    raw_bytes: bytes


def is_authenticated_manifest_ref(ref: ParentRef) -> bool:
    """Return whether *ref* declares the complete supported authentication profile.

    Partial and unknown profiles raise instead of silently degrading to legacy lookup.
    """

    return authenticated_manifest_ref_profile(ref) is not None


def restated_parent_differences(stated: ParentRef, bound: ParentRef) -> tuple[str, ...]:
    """Return how a document's restated parent disagrees with the bound one.

    A compiled document cannot authenticate a parent, so the only thing its
    restatement can do is agree or disagree about **which artifact** the binding
    names. That is three facts: the manifest kind, the manifest id, and the
    authenticated byte profile the restatement carries if it carries one.

    ``role`` is deliberately not among them, and its absence is the contract
    rather than a tolerance. A ``ParentRef``'s role is the *consumer's* addressing
    string, and for a staged prerequisite the consumer binding in the compile
    lock is what states it — that is where the binding name comes from in the
    first place. A document that also carried a role would either restate the
    lock's, adding nothing, or contradict it, in which case honoring the document
    would let a plan rename a binding the lock owns. Neither is a reason to
    refuse an otherwise identical artifact: the corpus habit of recording an
    artifact's own kind-ish role ("evaluation_run") beside a binding the lock
    names ("paired_trial_bank") is two true statements about different things.

    ``uri`` is excluded for the same reason: where bytes are staged from is the
    executing environment's, not the plan's.
    """
    differences: list[str] = []
    if stated.kind != bound.kind:
        differences.append(
            f"kind: the document restates {stated.kind!r} and the lock binds {bound.kind!r}"
        )
    if stated.id != bound.id:
        differences.append(
            f"id: the document restates {stated.id!r} and the lock binds {bound.id!r}"
        )
    stated_profile, stated_defect = _restated_profile(stated)
    bound_profile, bound_defect = _restated_profile(bound)
    if stated_defect is not None:
        differences.append(
            f"byte profile: the document states an authentication profile this build cannot "
            f"read ({stated_defect}); a half-stated profile is not a weaker claim than none, "
            "it is an unreadable one, and it is refused rather than dropped from the "
            "comparison"
        )
    if bound_defect is not None:
        differences.append(
            f"byte profile: the bound parent states an authentication profile this build "
            f"cannot read ({bound_defect}); the binding side is the authority, so an "
            "unreadable profile there is a refusal and never a comparison that is skipped"
        )
    if stated_defect is None and bound_defect is None:
        if stated_profile is not None and stated_profile != bound_profile:
            differences.append(
                f"byte profile: the document restates {stated_profile} and the lock binds "
                f"{bound_profile}"
            )
    return tuple(differences)


def _restated_profile(ref: ParentRef) -> tuple[tuple[str, int] | None, str | None]:
    """Return one ref's byte profile and, if it is unreadable, why.

    Three outcomes, and they are three different facts. A complete profile is a
    profile. No profile at all is the honest absence of a claim, which a
    *document* is entitled to: it cannot authenticate a parent, so restating
    nothing about bytes adds nothing to refuse over. A *partial* profile is
    neither: something stated half an authentication, and treating that as
    "states nothing" would let a malformed claim silently drop out of the
    comparison it was supposed to be subject to. So it comes back as a defect
    the caller reports.
    """
    try:
        return authenticated_manifest_ref_profile(ref), None
    except ValueError as exc:
        return None, str(exc)


def authenticated_manifest_ref(
    manifest: AnyManifest,
    path: Path | str,
    role: str,
) -> ParentRef:
    """Create a portable ref authenticating the exact final bytes at *path*."""

    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    parsed = load_manifest_bytes(raw)
    if parsed.kind != manifest.kind or parsed.id != manifest.id:
        raise ValueError(
            f"Manifest bytes at {manifest_path} do not match {manifest.kind} {manifest.id!r}"
        )
    return ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role=role,
        uri=None,
        metadata={
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
    )


def authenticated_manifest_ref_from_read(
    path: Path | str,
    *,
    expected_kind: str,
    expected_id: str,
    role: str,
) -> ParentRef:
    """Authenticate one manifest by performing the read that profiles it.

    Prefer this over :func:`authenticated_manifest_ref` everywhere a caller does
    not already hold a proven profile. The difference is the signature, and the
    signature is the point: this takes a *location* and the identity it is
    expected to hold, so it reads as what it is — the authenticating read.

    :func:`authenticated_manifest_ref` takes a manifest *object* plus a path,
    which reads as "authenticate this thing I already have" and invites exactly
    the defect: a caller that already resolved, verified, or admitted a manifest
    hands it over, the path is opened again, and the profile recorded as proof
    describes whatever is there on the second read. When a profile has already
    been established, restate it — do not mint a new one.
    """
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    parsed = load_manifest_bytes(raw)
    if parsed.kind != expected_kind or parsed.id != expected_id:
        raise ValueError(
            f"Manifest bytes at {manifest_path} are {parsed.kind} {parsed.id!r}, not "
            f"{expected_kind} {expected_id!r}"
        )
    return ParentRef(
        kind=parsed.kind,
        id=parsed.id,
        role=role,
        uri=None,
        metadata={
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
    )


def is_staged_manifest_kind(kind: str) -> bool:
    """Return whether *kind* is admitted as an authenticated staged manifest input."""
    return kind in _STAGED_MANIFEST_KINDS


def staged_manifest_kinds() -> frozenset[str]:
    """Return every manifest kind admitted as an authenticated staged input."""
    return _STAGED_MANIFEST_KINDS


def canonical_staged_manifest_locator(kind: str, manifest_id: str) -> str:
    """Return where one admitted staged manifest's canonical bytes live, root-relative.

    The location comes from the canonical layout helper rather than from a local
    copy of it, so a kind admitted here is addressed exactly the way
    :func:`~feedbax.contracts.manifest.write_manifest` writes it.
    """
    if not is_staged_manifest_kind(kind):
        raise ValueError(f"Authenticated staged manifest kind {kind!r} is not supported")
    return canonical_manifest_relative_path(kind, manifest_id)


def _canonical_locator(ref: ParentRef) -> str:
    return canonical_staged_manifest_locator(ref.kind, ref.id)


def _locator_parts(locator: Path | str) -> tuple[str, ...]:
    value = os.fspath(locator)
    if not value or "\\" in value:
        raise ValueError(f"Invalid manifest runtime locator: {value!r}")
    split = urlsplit(value)
    if split.scheme or split.netloc or split.query or split.fragment:
        raise ValueError(f"Manifest runtime locator must be a plain relative path: {value!r}")
    decoded = unquote(value)
    if decoded != value and ("\\" in decoded or "?" in decoded or "#" in decoded):
        raise ValueError(f"Invalid encoded manifest runtime locator: {value!r}")
    path = PurePosixPath(decoded)
    if path.is_absolute() or not path.parts:
        raise ValueError(f"Manifest runtime locator must be non-empty and relative: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Manifest runtime locator escapes its root: {value!r}")
    return tuple(path.parts)


def _read_regular_file(root: Path, parts: tuple[str, ...]) -> bytes:
    descriptors: list[int] = []
    try:
        records = open_directory_chain(
            root,
            create=False,
            error_factory=ValueError,
            context="authenticated manifest root",
        )
        descriptors.extend(descriptor for _, descriptor, _ in records)
        current = descriptors[-1]
        for component in parts[:-1]:
            current = open_existing_directory(
                component,
                error_factory=ValueError,
                context="authenticated manifest directory",
                dir_fd=current,
            )
            descriptors.append(current)
        file_descriptor, file_stat = open_existing_file(
            parts[-1],
            rigor=SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
            error_factory=ValueError,
            context="authenticated manifest",
            dir_fd=current,
        )
        descriptors.append(file_descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final_stat = os.fstat(file_descriptor)
        validate_opened_path(
            parts[-1],
            file_descriptor,
            rigor=SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
            error_factory=ValueError,
            context="authenticated manifest",
            dir_fd=current,
            expected_identity=(file_stat.st_dev, file_stat.st_ino),
        )
        if (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
            final_stat.st_mtime_ns,
            final_stat.st_ctime_ns,
        ) != (
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        ):
            raise ValueError("Authenticated manifest identity changed during read")
        return b"".join(chunks)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Authenticated manifest product is missing: {'/'.join(parts)}"
        ) from exc
    except OSError as exc:
        raise ValueError(f"Authenticated manifest locator is unsafe: {'/'.join(parts)}") from exc
    finally:
        close_descriptors(reversed(descriptors))


def resolve_manifest_input(
    ref: ParentRef,
    manifest_root: Path | str,
    runtime_locator: Path | str | None = None,
) -> ResolvedManifestInput:
    """Resolve and authenticate one staged manifest ref before downstream effects."""

    profile = authenticated_manifest_ref_profile(ref)
    if profile is None:
        raise ValueError(f"Manifest ref {ref.id!r} makes no authenticated claim")
    digest, expected_size = profile
    parts = _locator_parts(
        runtime_locator if runtime_locator is not None else _canonical_locator(ref)
    )
    root = Path(manifest_root)
    raw = _read_regular_file(root, parts)
    if len(raw) != expected_size:
        raise ValueError(f"Authenticated manifest size mismatch for {ref.id!r}")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Authenticated manifest SHA-256 mismatch for {ref.id!r}")
    manifest = load_manifest_bytes(raw)
    if manifest.kind != ref.kind:
        raise ValueError(
            f"Authenticated manifest kind mismatch: expected {ref.kind!r}, got {manifest.kind!r}"
        )
    if manifest.id != ref.id:
        raise ValueError(
            f"Authenticated manifest id mismatch: expected {ref.id!r}, got {manifest.id!r}"
        )
    return ResolvedManifestInput(
        ref=ref,
        manifest=manifest,
        path=root.joinpath(*parts),
        raw_bytes=raw,
    )


def resolve_contained_manifest_input(
    ref: ParentRef,
    manifest_root: Path | str,
    runtime_locator: Path | str,
) -> ResolvedManifestInput:
    """Resolve an explicitly located provider-free manifest within one supplied root."""

    parts = _locator_parts(runtime_locator)
    root = Path(manifest_root)
    raw = _read_regular_file(root, parts)
    manifest = load_manifest_bytes(raw)
    if manifest.kind != ref.kind or manifest.id != ref.id:
        raise ValueError("Contained manifest kind or id disagrees with ParentRef")
    return ResolvedManifestInput(
        ref=ref,
        manifest=manifest,
        path=root.joinpath(*parts),
        raw_bytes=raw,
    )
