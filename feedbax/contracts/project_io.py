"""Read and serialize the two authored-project representations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from os import fdopen
from os.path import lexists
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Literal

from pydantic import Field, ValidationError

from feedbax._secure_fs import (
    SecurePathRigor,
    close_directory_chain,
    open_directory_chain,
    open_existing_file,
    recheck_directory_chain,
)
from feedbax.contracts._authoring_inputs import AuthoringInputs
from feedbax.contracts.base import StrictModel, sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.project import Document, Payload, Project, Step
from feedbax.contracts.project_experiment import PROJECT_DECLARATION_FILENAME
from feedbax.contracts.project_inputs import NonemptyText, ProjectName
from feedbax.contracts.project_sweep import Row, Sweep
from feedbax.contracts.strict_json import strict_json_loads

PROJECT_SCHEMA_VERSION = "feedbax.project.v1"
PROJECT_FILE_FAMILIES = ("documents", "steps", "variants")
STUDIO_PRESENTATION_FILENAME = "presentation.json"
STUDIO_PATCH_JOURNAL_FILENAME = ".studio-patch.json"
STUDIO_WRITE_LOCK_FILENAME = ".studio-write.lock"
type TextFileStamp = tuple[str, int, int, int, int, int]


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
    revision: str
    file_revisions: Mapping[str, str]
    files: Mapping[str, bytes]
    manifest: TextProjectManifest

    @property
    def store_path(self) -> Path:
        return (self.root / self.configuration.store).resolve()


class TextProjectExternalInputs:
    """Read pinned project inputs under the caller's filesystem authority."""

    def __init__(
        self, text: TextProject, *, authority: Path | None = None,
        authority_identity: tuple[int, int] | None = None,
    ) -> None:
        self.root = text.root
        self.store = text.store_path
        self.authority = authority
        self.authority_identity = authority_identity
        self.paths: set[Path] = set()
        self.inputs = AuthoringInputs(read=self._read_target)

    def _read_target(self, target: Path) -> bytes:
        context = "pinned project input"
        chain = open_directory_chain(target.parent, create=False, error_factory=OSError, context=context)
        try:
            if self.authority_identity is not None:
                for path, _, info in chain:
                    if path == self.authority and (info.st_dev, info.st_ino) == self.authority_identity:
                        break
                else:
                    raise OSError("The authorized project folder changed")
            descriptor, _ = open_existing_file(
                target.name, dir_fd=chain[-1][1],
                rigor=SecurePathRigor.REGULAR_FILE_IDENTITY, error_factory=OSError, context=context,
            )
            with fdopen(descriptor, "rb") as stream:
                raw = stream.read()
            recheck_directory_chain(chain, error_factory=OSError, context=context)
            return raw
        finally:
            close_directory_chain(chain)

    def read(self, name: str) -> bytes:
        source = self.root / name
        try:
            target = source.resolve()
            if self.authority is not None and not target.is_relative_to(self.authority):
                raise OSError(f"{name}: the external pin is outside the authorized directory")
            if target.is_relative_to(self.store):
                raise OSError(f"{name}: an external pin cannot read the declared Store")
            self.paths.update((source, target))
            return self.inputs.read_bytes(source, resolved_path=target)
        except (ValueError, RuntimeError) as error:
            raise OSError(str(error)) from error


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


def _files(directory: Path) -> list[Path]:
    try:
        return sorted(path for path in directory.iterdir() if path.suffix == ".json")
    except FileNotFoundError:
        return []
    except OSError as error:
        raise ProjectReadError(directory, str(error)) from error


def text_project_paths(root: Path) -> tuple[Path, ...]:
    """Enumerate the paths owned by the text project codec."""
    return (
        root / PROJECT_DECLARATION_FILENAME,
        *(path for family in PROJECT_FILE_FAMILIES for path in _files(root / family)),
    )


def find_text_project(start: Path) -> Path:
    """Locate the nearest project declaration at or above a directory."""
    start = start.resolve()
    if not start.is_dir():
        raise ProjectReadError(start, "expected a project directory")
    for root in (start, *start.parents):
        if lexists(root / PROJECT_DECLARATION_FILENAME):
            return root
    raise ProjectReadError(start, f"no {PROJECT_DECLARATION_FILENAME} found in this directory or its parents")


def text_file_stamp(path: Path, root: Path) -> TextFileStamp:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ProjectReadError(path, "expected a regular file")
    return (
        path.relative_to(root).as_posix(), info.st_dev, info.st_ino,
        info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )


def text_project_fingerprint(root: Path) -> tuple[TextFileStamp, ...]:
    """Observe the authored files without opening or recovering the project."""
    for family in PROJECT_FILE_FAMILIES:
        directory = root / family
        try:
            info = directory.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(info.st_mode):
            raise ProjectReadError(directory, "expected a directory")
    return tuple(text_file_stamp(path, root) for path in text_project_paths(root))


def read_text_project_snapshot(root: Path) -> TextProject:
    """Load one stable authored snapshot without creating a lock or repairing a journal."""
    journal = root / STUDIO_PATCH_JOURNAL_FILENAME
    try:
        before = text_project_fingerprint(root)
        if lexists(journal):
            raise ProjectReadError(journal, "a Studio edit is pending")
        text = load_text_project(root)
        if before != text_project_fingerprint(root) or lexists(journal):
            raise ProjectReadError(root, "project changed while it was being read")
    except OSError as error:
        raise ProjectReadError(root, str(error)) from error
    return text


def load_text_project(root: Path | str) -> TextProject:
    """Read the declared project, with one named document, step or variant per file."""
    root = Path(root).resolve()
    files = {}
    for file in text_project_paths(root):
        try:
            files[file.relative_to(root).as_posix()] = file.read_bytes()
        except OSError as error:
            raise ProjectReadError(file, str(error)) from error
    return parse_text_project_files(root, files)


def parse_text_project_files(root: Path, files: Mapping[str, bytes]) -> TextProject:
    """Validate the complete set of files enumerated by the text codec."""
    path = root / PROJECT_DECLARATION_FILENAME
    named_paths = [Path(name) for name in sorted(files) if name != PROJECT_DECLARATION_FILENAME]

    def read[Model: StrictModel](model: type[Model], name: str) -> Model:
        return _parse(model, files[name], root / name)

    manifest = read(TextProjectManifest, PROJECT_DECLARATION_FILENAME)
    variants = manifest.variants
    if len(set(variants)) != len(variants):
        raise ProjectReadError(path, "variant names must be unique")
    if manifest.sweep is None and variants:
        raise ProjectReadError(path, "variant files require a project sweep")
    if manifest.sweep is not None and manifest.sweep.rows:
        raise ProjectReadError(path, "variant rows belong in variants/*.json")
    found_variants = {file.stem for file in named_paths if file.parent.name == "variants"}
    if found_variants != set(variants):
        missing = sorted(set(variants) - found_variants)
        undeclared = sorted(found_variants - set(variants))
        raise ProjectReadError(
            path, f"variant files differ from the manifest ({missing=}, {undeclared=})"
        )
    try:
        project = Project(
            documents={file.stem: read(Document, file.as_posix()) for file in named_paths if file.parent.name == "documents"},
            steps={file.stem: read(Step, file.as_posix()) for file in named_paths if file.parent.name == "steps"},
            sweep=(
                manifest.sweep.model_copy(
                    update={
                        "rows": {
                            name: read(Row, f"variants/{name}.json")
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
    digests = {name: sha256_bytes(data) for name, data in files.items()}
    revision = sha256_bytes(canonical_json_v2_bytes([
        "feedbax.project-folder.v1", [[name, digest] for name, digest in sorted(digests.items())],
    ]))
    return TextProject(
        project, configuration, root, revision, MappingProxyType(digests), MappingProxyType(dict(files)), manifest,
    )


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


def project_semantic_root(project: Project) -> str:
    """Hash authored Project content with its ordered sweep maps intact."""
    return sha256_bytes(_encode(project))


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


def rename_text_project_declaration(data: bytes, name: ProjectName) -> bytes:
    """Change the declared name while preserving the other authored fields and omissions."""
    manifest = _parse(TextProjectManifest, data, PROJECT_DECLARATION_FILENAME)
    return _encode(manifest.model_copy(update={"project": name}))


_SWEEP_DECLARATION_FIELDS = {"sweep", "variants"}
_PROJECT_CONFIGURATION_FIELDS = set(TextProjectManifest.model_fields) - _SWEEP_DECLARATION_FIELDS


def authored_declaration_revision(manifest: TextProjectManifest) -> str:
    return sha256_bytes(_encode(manifest, exclude=_PROJECT_CONFIGURATION_FIELDS))


def replay_text_project_declaration(current: TextProjectManifest, data: bytes) -> bytes:
    """Restore the authored sweep declaration while retaining the current configuration."""
    target = _parse(TextProjectManifest, data, PROJECT_DECLARATION_FILENAME)
    current_configuration = _encode(current, exclude=_SWEEP_DECLARATION_FIELDS)
    if current_configuration == _encode(target, exclude=_SWEEP_DECLARATION_FIELDS):
        return data
    return _encode(TextProjectManifest.model_validate({
        **current.model_dump(exclude_unset=True, include=_PROJECT_CONFIGURATION_FIELDS),
        **target.model_dump(exclude_unset=True, include=_SWEEP_DECLARATION_FIELDS),
    }))
