"""Resolve and enforce the provenance of the Feedbax package currently imported.

``resolve_feedbax_revision``/``assert_feedbax_revision_exact`` are the original,
already-wired execution-boundary check (Mandible-Issue 3149d58): they resolve the
Git commit of the checkout that supplied ``import feedbax`` and fail closed on a
locked-revision mismatch or an unresolvable source. Their behaviour is exactly as
shipped, so the PREFLIGHT/LAUNCH call sites and their tests are unaffected.

``assert_feedbax_revision_pin`` is the *authoring-time* half of that gate
(Mandible-Issue 0c2b295). An authored spec or lock in a downstream science repo
records the revision it was qualified against; that pin is a well-formedness and
ancestry statement, not a live identity statement, because the run's actual
provenance is minted per run set into ``RunBundle.feedbax_revision`` and asserted
exactly at launch. Authoring-time therefore asserts that the pin is well formed
and that the locked commit is an ancestor of the installed revision: it still
fails on a fabricated pin and on a pin from an abandoned branch, but it does not
fail merely because the dependency moved forward.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import feedbax
from feedbax._distribution_provenance import (
    PROVENANCE_FILENAME,
    SCHEMA_VERSION,
    DistributionProvenanceError,
    load_and_verify_provenance,
)


_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
FEEDBAX_DISTRIBUTION_PROVENANCE_SCHEMA_VERSION = SCHEMA_VERSION

_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "LC_ALL": "C",
    "PATH": os.defpath,
}


class FeedbaxRevisionError(RuntimeError):
    """The loaded Feedbax package cannot satisfy a locked revision pin."""


class FeedbaxRevisionAncestryWarning(UserWarning):
    """Ancestry of a locked revision pin could not be determined either way.

    Raised as a warning rather than an error because the authoring-time check is
    advisory by construction: exact, fail-closed enforcement of the revision a
    run actually executed against lives at the launch boundary, against the
    per-run-set ``RunBundle.feedbax_revision``. Callers who want the
    undeterminable case to be fatal can promote it with
    ``warnings.simplefilter("error", FeedbaxRevisionAncestryWarning)`` or
    ``-W error::feedbax.orchestration.revision.FeedbaxRevisionAncestryWarning``.
    """


def _feedbax_package_root() -> Path:
    """Return the resolved directory that supplied the imported Feedbax package."""
    source = getattr(feedbax, "__file__", None)
    if source is None:
        raise FeedbaxRevisionError("the imported Feedbax module has no source path")
    return Path(source).resolve().parent


def _resolve_checkout_revision(package_root: Path) -> str | None:
    """Return a revision only when Git owns this exact package directory."""
    top_level = _run_git(package_root, ["rev-parse", "--show-toplevel"])
    if top_level is None or top_level.returncode != 0:
        return None
    checkout_root = Path(top_level.stdout.strip()).resolve()
    if package_root != checkout_root / "feedbax":
        return None
    tracked = _run_git(
        checkout_root, ["ls-files", "--error-unmatch", "feedbax/__init__.py"]
    )
    if tracked is None or tracked.returncode != 0:
        return None
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


def _load_distribution_revision(package_root: Path) -> str | None:
    """Load and verify the versioned provenance embedded in an installed wheel."""
    provenance_path = package_root / PROVENANCE_FILENAME
    if not provenance_path.exists():
        return None
    try:
        _encoded, revision = load_and_verify_provenance(package_root)
    except DistributionProvenanceError as exc:
        raise FeedbaxRevisionError(f"installed {exc}") from exc
    return revision


def resolve_feedbax_revision() -> str:
    """Return the verified commit identity of the imported checkout or wheel."""
    package_root = _feedbax_package_root()
    checkout_revision = _resolve_checkout_revision(package_root)
    distribution_revision = _load_distribution_revision(package_root)
    if checkout_revision is not None and distribution_revision is not None:
        if checkout_revision != distribution_revision:
            raise FeedbaxRevisionError(
                "conflicting Feedbax revision identities: "
                f"checkout={checkout_revision} distribution={distribution_revision}"
            )
        return checkout_revision
    if checkout_revision is not None:
        return checkout_revision
    if distribution_revision is not None:
        return distribution_revision
    raise FeedbaxRevisionError(
        "cannot resolve a verified revision identity for the imported Feedbax package; "
        "it is neither a Git-owned checkout nor a provenance-bearing wheel"
    )


def assert_feedbax_revision_exact(locked_revision: str) -> str:
    """Fail closed unless the imported Feedbax package matches ``locked_revision``.

    This is the launch-time boundary check. Callers pass the revision Feedbax
    minted for the run set (``RunBundle.feedbax_revision``), so anything other
    than exact identity means the code about to execute is not the code the run
    was assembled from.
    """
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


def _run_git(root: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
    """Run a Git command in ``root``, returning ``None`` if Git cannot be invoked."""
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            check=False,
            env=_GIT_ENVIRONMENT,
            text=True,
        )
    except OSError:
        return None


def _checkout_holds_complete_history(package_root: Path) -> bool:
    """Return whether the checkout at ``package_root`` can hold every reachable object.

    A shallow clone or a partial (promisor) clone may legitimately lack objects
    that nonetheless exist upstream, so a missing object in such a checkout
    proves nothing. In a complete checkout the absence of an object is decisive:
    every ancestor of ``HEAD`` is necessarily present locally.
    """
    shallow = _run_git(package_root, ["rev-parse", "--is-shallow-repository"])
    if shallow is None or shallow.returncode != 0 or shallow.stdout.strip() != "false":
        return False
    promisor = _run_git(package_root, ["config", "--get-regexp", r"^remote\..*\.promisor$"])
    # Exit 1 is Git's "no matching configuration key", i.e. not a partial clone.
    if promisor is None or promisor.returncode not in (0, 1):
        return False
    return not (promisor.returncode == 0 and promisor.stdout.strip())


def _pin_ancestry_state(
    package_root: Path, locked_revision: str, loaded_revision: str
) -> tuple[str, str | None]:
    """Classify ``locked_revision`` against ``loaded_revision`` in ``package_root``.

    Returns:
        ``("ancestor", None)``, ``("not-ancestor", reason)``, or
        ``("undeterminable", reason)``. The three states are kept distinct here
        so that "the checkout cannot answer" never reaches a caller looking like
        "the answer is no".
    """
    if locked_revision == loaded_revision:
        return "ancestor", None
    known = _run_git(package_root, ["cat-file", "-e", f"{locked_revision}^{{commit}}"])
    if known is None:
        return "undeterminable", "Git could not be invoked for the supplying checkout"
    if known.returncode != 0:
        if _checkout_holds_complete_history(package_root):
            return (
                "not-ancestor",
                "the locked commit is unknown to the complete checkout that supplied "
                "the imported package, so it is not one of its ancestors",
            )
        return (
            "undeterminable",
            "the locked commit is absent from a shallow or partial checkout, which "
            "cannot distinguish a nonexistent commit from an unfetched one",
        )
    ancestry = _run_git(
        package_root, ["merge-base", "--is-ancestor", locked_revision, loaded_revision]
    )
    if ancestry is None:
        return "undeterminable", "Git could not be invoked for the supplying checkout"
    if ancestry.returncode == 0:
        return "ancestor", None
    if ancestry.returncode == 1:
        return "not-ancestor", "the locked commit is not reachable from the loaded revision"
    detail = ancestry.stderr.strip() or f"git merge-base exited {ancestry.returncode}"
    return "undeterminable", f"the ancestry query failed: {detail}"


def assert_feedbax_revision_pin(locked_revision: str) -> str:
    """Fail closed unless an authored pin is well formed and an ancestor of the install.

    This is the authoring-time gate for a revision recorded in a spec or lock: it
    states which Feedbax the document was qualified against, not which Feedbax a
    run executed (that is minted per run set and asserted exactly at launch by
    ``assert_feedbax_revision_exact``). A pin therefore stays valid as the
    dependency moves forward, while a fabricated commit or a commit from an
    abandoned branch still fails, because neither is an ancestor of the install.

    Three outcomes are distinguished, and each names itself in its message:

    - **malformed** — the pin is not a full lowercase Git commit. Raises.
    - **not-an-ancestor** — the pin is a real, decidable non-ancestor, including
      a commit unknown to a complete checkout. Raises.
    - **undeterminable** — the installed package has no resolvable Git history
      (a wheel or other non-checkout install), or its checkout is shallow or
      partial and lacks the object. Warns with
      ``FeedbaxRevisionAncestryWarning`` and passes, never silently.

    Failing closed on the undeterminable case would make the check unusable
    wherever Feedbax is not installed from a checkout, and would be a check on
    the installation rather than on the pin; passing it silently would weaken the
    gate invisibly. The warning is the explicit, promotable middle.

    Args:
        locked_revision: The full lowercase Git commit recorded in the document.

    Returns:
        ``locked_revision``, so a caller can assert the pin round-trips.

    Raises:
        FeedbaxRevisionError: The pin is malformed or decidably not an ancestor.
    """
    if not _GIT_REVISION_RE.fullmatch(locked_revision):
        raise FeedbaxRevisionError(
            "locked Feedbax revision pin must be a full lowercase Git commit: "
            f"locked={locked_revision!r}"
        )
    try:
        loaded_revision = resolve_feedbax_revision()
    except FeedbaxRevisionError as exc:
        warnings.warn(
            "Feedbax revision pin ancestry is undeterminable: "
            f"locked={locked_revision} loaded=<unresolved>; {exc}. The imported "
            "package has no resolvable Git history, so the authoring-time ancestry "
            "check cannot run; launch-time enforcement is unaffected.",
            FeedbaxRevisionAncestryWarning,
            stacklevel=2,
        )
        return locked_revision
    state, reason = _pin_ancestry_state(_feedbax_package_root(), locked_revision, loaded_revision)
    if state == "not-ancestor":
        raise FeedbaxRevisionError(
            "Feedbax revision pin is not an ancestor of the loaded revision: "
            f"locked={locked_revision} loaded={loaded_revision}; {reason}"
        )
    if state == "undeterminable":
        warnings.warn(
            "Feedbax revision pin ancestry is undeterminable: "
            f"locked={locked_revision} loaded={loaded_revision}; {reason}. "
            "Launch-time enforcement is unaffected.",
            FeedbaxRevisionAncestryWarning,
            stacklevel=2,
        )
    return locked_revision


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
    package_root = _feedbax_package_root()
    revision = resolve_feedbax_revision()
    dirty = (
        _feedbax_tree_is_dirty(package_root)
        if _resolve_checkout_revision(package_root) is not None
        else False
    )
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
