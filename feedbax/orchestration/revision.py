"""Resolve and enforce the provenance of the Feedbax package currently imported.

``resolve_feedbax_revision``/``assert_feedbax_revision_pin`` are the original,
already-wired execution-boundary check (Mandible-Issue 3149d58): they resolve the
Git commit of the checkout that supplied ``import feedbax`` and fail closed on a
locked-revision mismatch or an unresolvable source. They are kept exactly as
shipped so existing PREFLIGHT/LAUNCH call sites and their tests are unaffected.

``resolve_feedbax_provenance``/``check_feedbax_provenance`` extend that surface
(Mandible-Issue 7e7dac8) with working-tree cleanliness and an explicit,
never-default override, for callers that want the stricter guarantee described
by that issue: fail closed on mismatch, a dirty supplying tree, or unverifiable
provenance, and skip with a visible warning (never silently) when no pin is
supplied at all.
"""

from __future__ import annotations

import os
import re
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import feedbax


_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "LC_ALL": "C",
    "PATH": os.defpath,
}


class FeedbaxRevisionError(RuntimeError):
    """The loaded Feedbax package cannot satisfy a locked revision pin."""


def _feedbax_package_root() -> Path:
    """Return the resolved directory that supplied the imported Feedbax package."""
    source = getattr(feedbax, "__file__", None)
    if source is None:
        raise FeedbaxRevisionError("the imported Feedbax module has no source path")
    return Path(source).resolve().parent


def resolve_feedbax_revision() -> str:
    """Return the full commit of the checkout that supplied the imported package."""
    package_root = _feedbax_package_root()
    try:
        result = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "--verify", "HEAD^{commit}"],
            capture_output=True,
            check=True,
            env=_GIT_ENVIRONMENT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FeedbaxRevisionError(
            "cannot resolve the revision of the imported Feedbax module source"
        ) from exc
    revision = result.stdout.strip().lower()
    if not _GIT_REVISION_RE.fullmatch(revision):
        raise FeedbaxRevisionError(
            "the imported Feedbax module source did not resolve to a full lowercase Git commit"
        )
    return revision


def assert_feedbax_revision_pin(locked_revision: str) -> str:
    """Fail closed unless the imported Feedbax package matches ``locked_revision``."""
    if not _GIT_REVISION_RE.fullmatch(locked_revision):
        raise FeedbaxRevisionError(
            "locked Feedbax revision pin must be a full lowercase Git commit"
        )
    actual_revision = resolve_feedbax_revision()
    if actual_revision != locked_revision:
        raise FeedbaxRevisionError(
            "Feedbax revision pin mismatch: "
            f"locked={locked_revision} loaded={actual_revision}"
        )
    return actual_revision


@dataclass(frozen=True)
class FeedbaxProvenance:
    """The resolved identity of the Feedbax package currently imported."""

    source_path: Path
    revision: str
    dirty: bool


def _feedbax_tree_is_dirty(package_root: Path) -> bool:
    """Return whether the checkout supplying ``package_root`` has uncommitted changes."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(package_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
                "--",
                ".",
            ],
            capture_output=True,
            check=True,
            env=_GIT_ENVIRONMENT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FeedbaxRevisionError(
            "cannot resolve the working-tree cleanliness of the imported Feedbax "
            "module source"
        ) from exc
    return bool(result.stdout.strip())


def resolve_feedbax_provenance() -> FeedbaxProvenance:
    """Resolve the full identity (revision and dirty state) of the imported package.

    Raises ``FeedbaxRevisionError`` when either the revision or the working-tree
    cleanliness of the supplying checkout cannot be resolved. Unverifiable
    provenance is never silently treated as clean or matching.
    """
    revision = resolve_feedbax_revision()
    package_root = _feedbax_package_root()
    dirty = _feedbax_tree_is_dirty(package_root)
    return FeedbaxProvenance(source_path=package_root, revision=revision, dirty=dirty)


def check_feedbax_provenance(
    locked_revision: str | None,
    *,
    override: bool = False,
) -> FeedbaxProvenance | None:
    """Fail closed unless the imported Feedbax package matches ``locked_revision``.

    Enforces both revision match and a clean supplying working tree.
    ``locked_revision=None`` means the caller's inputs did not supply a pin at
    all; the check is skipped with a visible warning rather than silently
    passing. ``override=True`` is an explicit, narrow escape hatch for an
    operator who has already accepted the risk of a specific launch; it must
    never be the default, and it always emits a visible warning naming what was
    bypassed rather than bypassing silently.
    """
    if locked_revision is None:
        warnings.warn(
            "Feedbax revision pin is absent from the supplied inputs; skipping "
            "provenance verification.",
            stacklevel=2,
        )
        return None
    if not _GIT_REVISION_RE.fullmatch(locked_revision):
        raise FeedbaxRevisionError(
            "locked Feedbax revision pin must be a full lowercase Git commit"
        )
    if override:
        try:
            provenance = resolve_feedbax_provenance()
        except FeedbaxRevisionError as exc:
            warnings.warn(
                "Feedbax provenance override in effect; unverifiable provenance "
                f"ignored: {exc}",
                stacklevel=2,
            )
            return None
        if provenance.dirty or provenance.revision != locked_revision:
            warnings.warn(
                "Feedbax provenance override in effect; ignoring provenance check: "
                f"locked={locked_revision} observed={provenance.revision} "
                f"dirty={provenance.dirty}",
                stacklevel=2,
            )
        return provenance
    provenance = resolve_feedbax_provenance()
    if provenance.dirty:
        raise FeedbaxRevisionError(
            "Feedbax provenance check failed: the checkout supplying the imported "
            f"package has uncommitted changes: {provenance.source_path}"
        )
    if provenance.revision != locked_revision:
        raise FeedbaxRevisionError(
            "Feedbax revision pin mismatch: "
            f"locked={locked_revision} loaded={provenance.revision}"
        )
    return provenance
