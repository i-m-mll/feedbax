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

import importlib.util
import os
import re
import subprocess
import warnings
from collections.abc import Mapping
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


def resolve_repo_revision_at(source_root: Path) -> str:
    """Return the full commit currently on ``HEAD`` of the checkout at ``source_root``."""
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "--verify", "HEAD^{commit}"],
            capture_output=True,
            check=True,
            env=_GIT_ENVIRONMENT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FeedbaxRevisionError(
            f"cannot resolve the Git revision of the checkout at {source_root}"
        ) from exc
    revision = result.stdout.strip().lower()
    if not _GIT_REVISION_RE.fullmatch(revision):
        raise FeedbaxRevisionError(
            f"the checkout at {source_root} did not resolve to a full lowercase Git commit"
        )
    return revision


def _git_toplevel(path: Path) -> Path:
    """Return the resolved top-level directory of the Git checkout containing ``path``."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=True,
            env=_GIT_ENVIRONMENT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FeedbaxRevisionError(
            f"cannot resolve the Git top-level directory of {path}"
        ) from exc
    return Path(result.stdout.strip()).resolve()


def resolve_science_repo_import_revisions() -> dict[str, str]:
    """Resolve the revision of every checkout supplying ``feedbax.plugins`` modules.

    CERTIFY re-derives row payloads by importing the science-repo code published
    through the shared ``feedbax.plugins`` entry-point group. This resolves the
    Git checkout of each such provider — excluding the Feedbax package's own
    checkout — to the full commit currently on its ``HEAD``, so a recertification
    boundary can assert the imported science revision against the run's realized
    repository snapshot before deriving any payload.

    Returns:
        A mapping of resolved checkout path to its ``HEAD`` commit. Providers
        whose module source or containing checkout cannot be resolved are skipped
        rather than guessed; the caller decides how to treat an empty result.
    """
    from feedbax.plugins.discovery import feedbax_plugin_entry_points

    feedbax_root = _git_toplevel(_feedbax_package_root())
    revisions: dict[str, str] = {}
    for entry_point in feedbax_plugin_entry_points():
        module_name = getattr(entry_point, "module", None)
        if not isinstance(module_name, str) or not module_name:
            continue
        top_level = module_name.split(".", 1)[0]
        try:
            spec = importlib.util.find_spec(top_level)
        except (ImportError, ValueError, AttributeError):
            continue
        origin = getattr(spec, "origin", None) if spec is not None else None
        if not origin:
            continue
        try:
            root = _git_toplevel(Path(origin).resolve().parent)
        except FeedbaxRevisionError:
            continue
        if root == feedbax_root:
            continue
        revisions[str(root)] = resolve_repo_revision_at(root)
    return revisions


def assert_science_repo_revision_pin(
    *,
    primary_repo: str,
    realized_revision: str,
    imported_revisions: Mapping[str, str],
) -> None:
    """Fail closed unless every imported science checkout matches the run's snapshot.

    Args:
        primary_repo: Name of the run's primary (science) repository.
        realized_revision: Commit the run's realized repository snapshot pinned
            for the primary science repository.
        imported_revisions: Mapping of each currently imported science checkout
            path to its resolved ``HEAD`` commit.

    Any divergence means CERTIFY would re-derive payloads with code the run never
    realized, so the check raises ``FeedbaxRevisionError`` naming both revisions
    and the offending checkout.
    """
    for source_path, revision in sorted(imported_revisions.items()):
        if revision != realized_revision:
            raise FeedbaxRevisionError(
                "science repo revision pin mismatch for "
                f"{primary_repo!r}: run realized {realized_revision}, but the "
                f"imported science checkout at {source_path} is {revision}"
            )


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
