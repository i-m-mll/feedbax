"""Read and serialize the two authored-project representations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from feedbax.contracts.base import StrictModel
from feedbax.contracts.project import Document, Payload, Project, Step
from feedbax.contracts.project_experiment import PROJECT_DECLARATION_FILENAME
from feedbax.contracts.project_inputs import NonemptyText, ProjectName
from feedbax.contracts.project_sweep import Row, Sweep
from feedbax.contracts.strict_json import strict_json_loads

PROJECT_SCHEMA_VERSION = "feedbax.project.v1"


class ProjectConfiguration(StrictModel):
    project: ProjectName
    store: NonemptyText
    settings: Payload = Field(default_factory=dict)


class TextProjectManifest(ProjectConfiguration):
    schema_version: Literal["feedbax.project.v1"]
    sweep: Sweep | None = None
    variants: list[ProjectName] = Field(default_factory=list)


class StudioProjectFile(StrictModel):
    project: Project
    presentation: Payload


@dataclass(frozen=True)
class TextProject:
    project: Project
    configuration: ProjectConfiguration
    root: Path

    @property
    def store_path(self) -> Path:
        return (self.root / self.configuration.store).resolve()


class ProjectReadError(ValueError):
    """An invalid or unreadable project source."""

    def __init__(self, source: str | Path, message: str):
        self.source = str(source)
        super().__init__(f"{self.source}: {message}")


def _parse[Model: StrictModel](model: type[Model], data: bytes, source: str | Path) -> Model:
    try:
        return model.model_validate(strict_json_loads(data, ref=str(source)))
    except (ValueError, UnicodeError) as error:
        raise ProjectReadError(source, str(error)) from error


def _read[Model: StrictModel](model: type[Model], path: Path) -> Model:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ProjectReadError(path, str(error)) from error
    return _parse(model, data, path)


def _files(directory: Path) -> list[Path]:
    try:
        return sorted(path for path in directory.iterdir() if path.suffix == ".json")
    except FileNotFoundError:
        return []
    except OSError as error:
        raise ProjectReadError(directory, str(error)) from error


def load_text_project(root: Path | str) -> TextProject:
    """Read the declared project, with one named document, step or variant per file."""
    root = Path(root).resolve()
    path = root / PROJECT_DECLARATION_FILENAME
    manifest = _read(TextProjectManifest, path)
    variants = manifest.variants
    if len(set(variants)) != len(variants):
        raise ProjectReadError(path, "variant names must be unique")
    if manifest.sweep is None and variants:
        raise ProjectReadError(path, "variant files require a project sweep")
    if manifest.sweep is not None and manifest.sweep.rows:
        raise ProjectReadError(path, "variant rows belong in variants/*.json")
    found_variants = {file.stem for file in _files(root / "variants")}
    if found_variants != set(variants):
        missing = sorted(set(variants) - found_variants)
        undeclared = sorted(found_variants - set(variants))
        raise ProjectReadError(
            path, f"variant files differ from the manifest ({missing=}, {undeclared=})"
        )
    try:
        project = Project(
            documents={file.stem: _read(Document, file) for file in _files(root / "documents")},
            steps={file.stem: _read(Step, file) for file in _files(root / "steps")},
            sweep=(
                manifest.sweep.model_copy(
                    update={
                        "rows": {
                            name: _read(Row, root / "variants" / f"{name}.json")
                            for name in variants
                        }
                    }
                )
                if manifest.sweep is not None
                else None
            ),
        )
    except ValidationError as error:
        raise ProjectReadError(path, str(error)) from error
    configuration = ProjectConfiguration(
        project=manifest.project, store=manifest.store, settings=manifest.settings
    )
    return TextProject(project, configuration, root)


def parse_studio_project(data: bytes, *, source: str = "Studio project") -> StudioProjectFile:
    """Read the Studio file without interpreting its presentation payload."""
    return _parse(StudioProjectFile, data, source)


def _encode(model: StrictModel, *, exclude: set[str] | None = None) -> bytes:
    return (
        json.dumps(
            model.model_dump(
                mode="json",
                by_alias=True,
                exclude_unset=True,
                exclude=exclude,
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def dump_studio_project(project_file: StudioProjectFile) -> bytes:
    """Serialize the semantic model beside its presentation."""
    return _encode(project_file)


def dump_text_project(project: Project, configuration: ProjectConfiguration) -> dict[str, bytes]:
    """Produce the complete set of relative paths and bytes for a text project."""
    rows = project.sweep.rows if project.sweep is not None else {}
    manifest = TextProjectManifest(
        schema_version=PROJECT_SCHEMA_VERSION,
        **configuration.model_dump(exclude_unset=True),
        sweep=project.sweep.model_copy(update={"rows": {}}) if project.sweep is not None else None,
        variants=list(rows),
    )
    return {
        PROJECT_DECLARATION_FILENAME: _encode(manifest),
        **{
            f"documents/{name}.json": _encode(document)
            for name, document in project.documents.items()
        },
        **{f"steps/{name}.json": _encode(step) for name, step in project.steps.items()},
        **{f"variants/{name}.json": _encode(row) for name, row in rows.items() if row is not None},
    }
