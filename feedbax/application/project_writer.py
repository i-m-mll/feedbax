"""Write ordinary authored project files with the application's installed declarations."""

from __future__ import annotations

from collections.abc import Collection, Iterable

from pydantic import BaseModel, JsonValue

from feedbax.application.stored_regions import StoredRegionRegistries
from feedbax.contracts import project_io, project_storage
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, StudioProjectFile
from feedbax.contracts.project_storage import StoredPath
from feedbax.contracts.stored_regions import StoredRegionResolver


def dump_text_project(
    project: Project, configuration: ProjectConfiguration, *, registries: StoredRegionResolver | None = None,
    only: Collection[str] | None = None,
) -> dict[str, bytes]:
    return project_io.dump_text_project(
        project, configuration, registries=StoredRegionRegistries() if registries is None else registries, only=only,
    )


def dump_studio_project(project_file: StudioProjectFile, *, registries: StoredRegionResolver | None = None) -> bytes:
    return project_io.dump_studio_project(
        project_file, registries=StoredRegionRegistries() if registries is None else registries,
    )


def write_project_file(
    model: BaseModel, *, exclude: set[str] | None = None, protected: Iterable[StoredPath] = (),
    registries: StoredRegionResolver | None = None, source: str | None = None,
) -> bytes:
    return project_storage.write_project_file(
        model, exclude=exclude, protected=protected, source=source,
        registries=StoredRegionRegistries() if registries is None else registries,
    )


def stored_defaults_manifest() -> dict[str, JsonValue]:
    from feedbax.training.optimizers import OptimizerSpec, ScheduleSpec

    return project_storage.stored_defaults_manifest((
        OptimizerSpec, ScheduleSpec, *StoredRegionRegistries().feedbax_models(),
    ))
