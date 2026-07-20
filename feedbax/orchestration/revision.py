"""Resolve and enforce the revision of the Feedbax package currently imported."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import feedbax


_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class FeedbaxRevisionError(RuntimeError):
    """The loaded Feedbax package cannot satisfy a locked revision pin."""


def resolve_feedbax_revision() -> str:
    """Return the full commit of the checkout that supplied the imported package."""
    source = getattr(feedbax, "__file__", None)
    if source is None:
        raise FeedbaxRevisionError("the imported Feedbax module has no source path")
    package_root = Path(source).resolve().parent
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "LC_ALL": "C",
        "PATH": os.defpath,
    }
    try:
        result = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "--verify", "HEAD^{commit}"],
            capture_output=True,
            check=True,
            env=environment,
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
