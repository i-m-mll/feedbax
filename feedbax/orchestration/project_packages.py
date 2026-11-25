"""Pinned project-package sources for a remote Host session."""

from pathlib import PurePosixPath
from typing import Annotated

from pydantic import ConfigDict, Field, field_validator, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.orchestration.repo_snapshot import RepoSnapshotRecord


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
