"""Checks for downstream pins of an editable upstream package checkout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tomllib
import warnings


@dataclass(frozen=True)
class VersionPinReport:
    """Outcome of a downstream-to-upstream version-pin check."""

    package_name: str
    pin_file: Path
    pinned_revision: str
    checkout_root: Path | None
    head_revision: str | None
    remote_ref: str
    skipped: bool = False
    skip_reason: str | None = None
    unpublished_allowed: bool = False


def find_editable_git_root(package_path: Path | str) -> Path | None:
    """Find the Git checkout containing an explicit package path, or return ``None``.

    Returning ``None`` is the documented skip path for wheels and other
    non-editable installs. Callers may pass the resulting report to
    ``pytest.skip(report.skip_reason)``.
    """
    resolved = Path(package_path).resolve()
    package_dir = resolved.parent if resolved.is_file() else resolved
    for candidate in (package_dir, *package_dir.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def check_version_pin(
    *,
    package_name: str,
    pin_file: Path | str,
    package_path: Path | str | None = None,
    checkout_root: Path | str | None = None,
    remote_ref: str = "refs/remotes/origin/develop",
    escape_hatch_env: str | None = None,
    environ: Mapping[str, str] = os.environ,
) -> VersionPinReport:
    """Check HEAD equality and remote reachability for a downstream pin.

    ``pin_file`` must be a TOML file containing a non-empty ``rev`` string. If
    ``checkout_root`` is omitted, ``package_path`` is required and walked
    upward for a Git checkout. A non-editable install returns a skipped report;
    pytest callers should call ``pytest.skip(report.skip_reason)`` when
    ``report.skipped`` is true.
    """
    path = Path(pin_file)
    pinned = _load_pinned_revision(path)
    if checkout_root is None and package_path is None:
        raise ValueError("package_path or checkout_root is required")
    root = (
        Path(checkout_root).resolve()
        if checkout_root is not None
        else find_editable_git_root(Path(package_path))  # type: ignore[arg-type]
    )
    if root is None:
        reason = (
            f"{package_name} is not installed from a Git checkout; "
            "the pinned revision is expected to be selected by installation"
        )
        return VersionPinReport(
            package_name=package_name,
            pin_file=path,
            pinned_revision=pinned,
            checkout_root=None,
            head_revision=None,
            remote_ref=remote_ref,
            skipped=True,
            skip_reason=reason,
        )
    head = _git_stdout(root, ["rev-parse", "HEAD"])
    if head != pinned:
        raise AssertionError(
            f"{path} pins {package_name} at {pinned}, but the editable checkout at "
            f"{root} is at {head}. Bump the pin in the same change wave."
        )
    remote_revision = _remote_revision(root, remote_ref, package_name=package_name)
    reachability = _run_git(
        root, ["merge-base", "--is-ancestor", pinned, remote_ref], check=False
    )
    allow_unpublished = bool(escape_hatch_env and environ.get(escape_hatch_env) == "1")
    if reachability.returncode == 1:
        if not allow_unpublished:
            raise AssertionError(
                f"{path} pins {package_name} at {pinned}, but that commit is not reachable "
                f"from last-fetched {remote_ref} ({remote_revision}) in {root}. Push the "
                "upstream branch or fetch its remote ref."
            )
        warnings.warn(
            f"WARNING: {escape_hatch_env}=1 allows unpublished {package_name} pin {pinned}; "
            f"last-fetched {remote_ref} is {remote_revision}. Do not rely on remote CI.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif reachability.returncode != 0:
        reachability.check_returncode()
    return VersionPinReport(
        package_name=package_name,
        pin_file=path,
        pinned_revision=pinned,
        checkout_root=root,
        head_revision=head,
        remote_ref=remote_ref,
        unpublished_allowed=reachability.returncode == 1,
    )


def _load_pinned_revision(path: Path) -> str:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    revision = data.get("rev")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"{path}: rev must be a non-empty string")
    return revision


def _run_git(
    root: Path, args: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _git_stdout(root: Path, args: Sequence[str]) -> str:
    return _run_git(root, args).stdout.strip()


def _remote_revision(root: Path, remote_ref: str, *, package_name: str) -> str:
    try:
        return _git_stdout(root, ["rev-parse", "--verify", remote_ref])
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            f"{package_name} checkout at {root} has no last-fetched {remote_ref}; "
            "fetch the upstream remote ref before checking publish reachability"
        ) from exc
