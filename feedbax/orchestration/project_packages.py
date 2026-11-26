"""Pinned project-package sources for a remote Host session."""

from importlib import metadata
import os
from pathlib import Path, PurePosixPath
import subprocess
from urllib.parse import urlsplit
from urllib.request import url2pathname
from typing import Annotated

from packaging.utils import canonicalize_name
from pydantic import ConfigDict, Field, field_validator, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.orchestration.repo_snapshot import RepoSnapshotRecord, RepoSnapshotSourceFault, seal_repo_snapshot
from feedbax.plugins.declarations import DistributionSource


class ProjectPackagePin(StrictModel):
    """The distribution and install directory inside one clean, sealed Git snapshot."""

    model_config = ConfigDict(frozen=True)

    distribution: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
    version: Annotated[str, Field(min_length=1)]
    project_dir: str
    snapshot: RepoSnapshotRecord

    @field_validator("project_dir")
    @classmethod
    def _relative_directory(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value or "\x00" in value or "\\" in value:
            raise ValueError("The install directory must be relative to the sealed repository.")
        return value

    @model_validator(mode="after")
    def _clean(self) -> "ProjectPackagePin":
        if self.snapshot.dirty:
            raise ValueError("Remote project packages require a clean Git commit.")
        return self



def pin_project_package(source: DistributionSource, snapshot_parent: Path) -> ProjectPackagePin:
    """Seal the local editable install that supplied the planner's declaration."""
    name = canonicalize_name(source.distribution)
    try:
        distribution = metadata.distribution(name)
        installed = strict_json_loads(distribution.read_text("direct_url.json") or "{}", ref=f"{name} direct_url.json")
        url = urlsplit(installed.get("url", ""))
        editable = installed.get("dir_info", {}).get("editable") is True
    except (AttributeError, TypeError, ValueError, metadata.PackageNotFoundError) as error:
        raise RepoSnapshotSourceFault("The installed package has no usable local source metadata.") from error
    if distribution.version != source.version or not editable or url.scheme != "file" or (
        url.netloc not in ("", "localhost") or url.query or url.fragment
    ):
        raise RepoSnapshotSourceFault("Install the project package in editable mode from its local Git checkout.")
    try:
        project = Path(url2pathname(url.path)).resolve(strict=True)
        root = Path(os.fsdecode(subprocess.check_output(
            ["git", "-C", str(project), "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL, timeout=5,
        )).strip()).resolve(strict=True)
        relative = project.relative_to(root)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise RepoSnapshotSourceFault("The installed package source is not in a local Git checkout.") from error

    sealed = seal_repo_snapshot(name, root, snapshot_parent=snapshot_parent, require_clean=True)
    if not (sealed.staging_root / relative).resolve().is_relative_to(sealed.staging_root):
        raise RepoSnapshotSourceFault("The install directory leaves the sealed repository.")
    for path in sealed.staging_root.rglob("*"):
        if path.is_symlink() and (
            Path(os.readlink(path)).is_absolute() or not path.resolve().is_relative_to(sealed.staging_root)
        ):
            raise RepoSnapshotSourceFault("The package contains a link outside its sealed Git checkout.")
    return ProjectPackagePin(
        distribution=name, version=source.version, project_dir=relative.as_posix(), snapshot=sealed.record,
    )
