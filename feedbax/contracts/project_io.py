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

from pydantic import Field, TypeAdapter, ValidationError

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
from feedbax.contracts.project_inputs import NonemptyText, ProjectName
from feedbax.contracts.project_sweep import Row, Sweep
from feedbax.contracts.strict_json import strict_json_loads

PROJECT_SCHEMA_VERSION = "feedbax.project.v1"
PROJECT_DECLARATION_SUFFIX = ".feedbax"
_PROJECT_NAME = TypeAdapter(ProjectName)
PROJECT_FILE_FAMILIES = ("documents", "steps", "variants")
STUDIO_PRESENTATION_FILENAME = "presentation.json"
STUDIO_PATCH_JOURNAL_FILENAME = ".studio-patch.json"
STUDIO_WRITE_LOCK_FILENAME = ".studio-write.lock"
type TextFileStamp = tuple[str, int, int, int, int, int]


class ProjectConfiguration(StrictModel):
    project: ProjectName
    store: NonemptyText
    settings: Payload = Field(default_factory=dict)


class TextProjectManifest(StrictModel):
    schema_version: Literal["feedbax.project.v1"]
    store: NonemptyText
    settings: Payload = Field(default_factory=dict)
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
    def declaration(self) -> str:
        return project_declaration_name(self.configuration.project)

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


def _directory_entries(directory: Path) -> list[Path]:
    try:
        return sorted(directory.iterdir())
    except FileNotFoundError:
        return []
    except OSError as error:
        raise ProjectReadError(directory, str(error)) from error


def project_declaration_name(name: ProjectName) -> str:
    return _PROJECT_NAME.validate_python(name) + PROJECT_DECLARATION_SUFFIX


def _one_declaration(root: Path, candidates: list[str]) -> str:
    if not candidates:
        raise ProjectReadError(root, f"no *{PROJECT_DECLARATION_SUFFIX} project declaration found")
    if len(candidates) != 1:
        raise ProjectReadError(root, f"multiple project declarations: {', '.join(candidates)}")
    name, = candidates
    try:
        _PROJECT_NAME.validate_python(name.removesuffix(PROJECT_DECLARATION_SUFFIX))
    except ValidationError as error:
        raise ProjectReadError(root / name, "invalid project name in declaration filename") from error
    return name


def _declaration_candidates(root: Path) -> list[str]:
    return [path.name for path in _directory_entries(root) if path.name.endswith(PROJECT_DECLARATION_SUFFIX)]


def _regular_declaration(root: Path, candidates: list[str]) -> Path:
    name = _one_declaration(root, candidates)
    path = root / name
    text_file_stamp(path, root)
    return path


def text_project_declaration(root: Path) -> Path:
    return _regular_declaration(root, _declaration_candidates(root))


def text_project_paths(root: Path) -> tuple[Path, ...]:
    """Enumerate the paths owned by the text project codec."""
    return (
        text_project_declaration(root),
        *(path for family in PROJECT_FILE_FAMILIES for path in _directory_entries(root / family)
          if path.suffix == ".json"),
    )


def find_text_project(start: Path) -> Path:
    """Locate a named declaration or the nearest project at or above a directory."""
    return locate_text_project(start)[0]


def locate_text_project(start: Path) -> tuple[Path, str | None]:
    """Find the project folder and retain the name of an explicitly selected file."""
    start = start.absolute()
    if start.name.endswith(PROJECT_DECLARATION_SUFFIX) and (start.is_symlink() or not start.is_dir()):
        root = start.parent.resolve()
        require_text_declaration(root, start.name)
        return root, start.name
    start = start.resolve()
    if not start.is_dir():
        raise ProjectReadError(start, "expected a project directory")
    for root in (start, *start.parents):
        if candidates := _declaration_candidates(root):
            _regular_declaration(root, candidates)
            return root, None
    raise ProjectReadError(start, f"no *{PROJECT_DECLARATION_SUFFIX} found in this directory or its parents")


def require_text_declaration(root: Path, name: str) -> None:
    if text_project_declaration(root).name != name:
        raise ProjectReadError(root / name, "not the folder's project declaration")


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


def read_text_project_snapshot(root: Path, *, declaration: str | None = None) -> TextProject:
    """Load one stable authored snapshot without creating a lock or repairing a journal."""
    journal = root / STUDIO_PATCH_JOURNAL_FILENAME
    try:
        before = text_project_fingerprint(root)
        if lexists(journal):
            raise ProjectReadError(journal, "a Studio edit is pending")
        text = load_text_project(root)
        if declaration is not None and text.declaration != declaration:
            raise ProjectReadError(root / declaration, "not the folder's project declaration")
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
    declaration = _one_declaration(root, [
        name for name in sorted(files) if "/" not in name and name.endswith(PROJECT_DECLARATION_SUFFIX)
    ])
    path = root / declaration
    named_paths = [Path(name) for name in sorted(files) if name != declaration]

    def read[Model: StrictModel](model: type[Model], name: str) -> Model:
        return _parse(model, files[name], root / name)

    manifest = read(TextProjectManifest, declaration)
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
        project=path.stem, store=manifest.store, settings=manifest.settings
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
        **configuration.model_dump(exclude_unset=True, exclude={"project"}),
        sweep=project.sweep.model_copy(update={"rows": {}}) if project.sweep is not None else None,
        variants=list(rows),
    )
    return {
        project_declaration_name(configuration.project): _encode(manifest),
        **{
            f"documents/{name}.json": _encode(document)
            for name, document in project.documents.items()
        },
        **{f"steps/{name}.json": _encode(step) for name, step in project.steps.items()},
        **{f"variants/{name}.json": _encode(row) for name, row in rows.items() if row is not None},
    }


_SWEEP_DECLARATION_FIELDS = {"sweep", "variants"}
_PROJECT_CONFIGURATION_FIELDS = set(TextProjectManifest.model_fields) - _SWEEP_DECLARATION_FIELDS


def authored_declaration_revision(manifest: TextProjectManifest) -> str:
    return sha256_bytes(_encode(manifest, exclude=_PROJECT_CONFIGURATION_FIELDS))


def replay_text_project_declaration(current: TextProjectManifest, data: bytes, *, declaration: str) -> bytes:
    """Restore the authored sweep declaration while retaining the current configuration."""
    target = _parse(TextProjectManifest, data, declaration)
    current_configuration = _encode(current, exclude=_SWEEP_DECLARATION_FIELDS)
    if current_configuration == _encode(target, exclude=_SWEEP_DECLARATION_FIELDS):
        return data
    return _encode(TextProjectManifest.model_validate({
        **current.model_dump(exclude_unset=True, include=_PROJECT_CONFIGURATION_FIELDS),
        **target.model_dump(exclude_unset=True, include=_SWEEP_DECLARATION_FIELDS),
    }))
