"""Read and write the text-project files addressed by Studio."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from feedbax._atomic import sync_parent_directories, write_bytes_atomic
from feedbax._secure_fs import directory_open_flags, rename_no_replace
from feedbax.contracts.base import StrictModel, sha256_bytes
from feedbax.contracts.matrix_core import apply_value_patch_in_place
from feedbax.contracts.project import Project, Sha256, json_pointer_parts
from feedbax.contracts.project_experiment import PROJECT_DECLARATION_FILENAME
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project_io import (
    ProjectConfiguration,
    ProjectReadError,
    STUDIO_PATCH_JOURNAL_FILENAME,
    STUDIO_PRESENTATION_FILENAME,
    STUDIO_WRITE_LOCK_FILENAME,
    TextFileStamp,
    TextProject,
    dump_text_project,
    read_text_project_snapshot,
    project_semantic_root,
    rename_text_project_declaration,
    text_file_stamp,
    text_project_fingerprint,
)
from feedbax.contracts.project_plan import ProjectPlan, ProjectPlanMemo, RunKey, compile_project
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.contracts.studio_project import (
    AuthoredFileAddress,
    CreateNamedDocument,
    PresentationDocument,
    SetProjectValue,
    StudioAuthoredFileRevision,
    StudioPresentationWrite,
    StudioProjectDocument,
    StudioProjectCreateFile,
    StudioProjectPatch,
    StudioProjectPatchBatch,
    StudioProjectRename,
    StudioProjectRevision,
    StudioProjectSnapshot,
)


_FAMILIES = {"document": "documents", "step": "steps", "variant": "variants"}
_KINDS = {family: kind for kind, family in _FAMILIES.items()}
_FOLDER_ID = TypeAdapter(ProjectName)
type _Resource = Literal["project", "presentation"]


class ProjectFolderNotFound(FileNotFoundError):
    """The addressed project does not exist in the Studio project directory."""


class ProjectFolderInvalid(ValueError):
    """The folder or the requested edit is invalid."""


class ProjectFileExists(FileExistsError):
    """The requested authored name already has a file in its family."""


class ProjectRevisionConflict(RuntimeError):
    """The addressed resource changed before the edit could be published."""

    def __init__(self, resource: _Resource, current: str) -> None:
        self.resource = resource
        self.current = current
        super().__init__(f"{resource.capitalize()} changed before this edit.")


@dataclass(frozen=True)
class SettledProjectPlan:
    """The compiler portion of Plan, bound to one saved project revision."""

    project_id: str
    revision: str
    semantic_root: str
    configuration: ProjectConfiguration
    store_path: Path
    plan: ProjectPlan

    @classmethod
    def from_text(cls, project_id: str, text: TextProject, plan: ProjectPlan) -> SettledProjectPlan:
        return cls(
            project_id, text.revision, project_semantic_root(text.project),
            text.configuration, text.store_path, plan,
        )


def _regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ProjectFolderInvalid(f"{path.name} must be a regular file")
    return info


class _PendingFile(StrictModel):
    address: AuthoredFileAddress
    before: Sha256
    data: str

    @property
    def relative(self) -> str:
        return f"{_FAMILIES[self.address.kind]}/{self.address.id}.json"


_PENDING_FILES = TypeAdapter(list[_PendingFile])


def _finish_patch(root: Path) -> None:
    journal = root / STUDIO_PATCH_JOURNAL_FILENAME
    try:
        _regular(journal)
    except FileNotFoundError:
        return
    try:
        files = _PENDING_FILES.validate_json(journal.read_bytes())
    except ValidationError as error:
        raise ProjectFolderInvalid("Studio edit journal is invalid") from error
    replacements = []
    for file in files:
        path = root / file.relative
        if not stat.S_ISDIR(path.parent.lstat().st_mode):
            raise ProjectFolderInvalid(f"{path.parent.name} must be a directory")
        mode = stat.S_IMODE(_regular(path).st_mode)
        data = file.data.encode()
        current = sha256_bytes(path.read_bytes())
        if current == sha256_bytes(data):
            continue
        if current != file.before:
            raise ProjectFolderInvalid(f"{file.relative} changed during a Studio edit")
        replacements.append((file, path, mode, data))

    for file, path, mode, data in replacements:
        def guard() -> None:
            _regular(path)
            if sha256_bytes(path.read_bytes()) != file.before:
                raise ProjectFolderInvalid(f"{file.relative} changed during a Studio edit")

        write_bytes_atomic(path, data, mode=mode, before_publish=guard)
    journal.unlink()
    sync_parent_directories(journal)


@contextmanager
def _folder_lock(root: Path, *, write: bool = True) -> Iterator[None]:
    lock = root / STUDIO_WRITE_LOCK_FILENAME
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
        opened = os.fstat(descriptor)
        current = _regular(lock)
        if opened.st_nlink != 1 or (opened.st_dev, opened.st_ino) != (
            current.st_dev,
            current.st_ino,
        ):
            raise ProjectFolderInvalid("Project write lock changed")
        if os.path.lexists(root / STUDIO_PATCH_JOURNAL_FILENAME):
            if not write:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            _finish_patch(root)
        yield
    finally:
        os.close(descriptor)


def _check_revision(resource: _Resource, expected: str, current: str) -> None:
    if expected != current:
        raise ProjectRevisionConflict(resource, current)


def _read_presentation(root: Path) -> tuple[bytes | None, str]:
    path = root / STUDIO_PRESENTATION_FILENAME
    try:
        _regular(path)
        data = path.read_bytes()
    except FileNotFoundError:
        return None, sha256_bytes(b"feedbax.presentation.v1\0")
    return data, sha256_bytes(b"feedbax.presentation.v1\1" + data)


def _encode_presentation(presentation: PresentationDocument) -> bytes:
    return (presentation.model_dump_json(indent=2, exclude_unset=True) + "\n").encode()


class ProjectFolderService:
    """Own revision guards for project folders under the app-managed root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._plan_memo = lru_cache(maxsize=8)(lambda _project_id: ProjectPlanMemo())

    def _folder(self, project_id: str) -> Path:
        try:
            _FOLDER_ID.validate_python(project_id)
        except ValidationError as error:
            raise ProjectFolderInvalid("Invalid project name") from error
        folder = self.root / project_id
        try:
            info = folder.lstat()
        except FileNotFoundError as error:
            raise ProjectFolderNotFound(project_id) from error
        if not stat.S_ISDIR(info.st_mode):
            raise ProjectFolderInvalid("Project must be a directory")
        return folder

    def fingerprint(self, project_id: str) -> tuple[TextFileStamp, ...]:
        root = self._folder(project_id)
        with _folder_lock(root, write=False):
            try:
                authored = text_project_fingerprint(root)
                try:
                    return (*authored, text_file_stamp(root / STUDIO_PRESENTATION_FILENAME, root))
                except FileNotFoundError:
                    return authored
            except (OSError, ProjectReadError) as error:
                raise ProjectFolderInvalid(str(error)) from error

    def _load(self, root: Path) -> TextProject:
        try:
            return read_text_project_snapshot(root)
        except ProjectReadError as error:
            raise ProjectFolderInvalid(str(error)) from error

    def _snapshot(self, project_id: str, text: TextProject) -> StudioProjectSnapshot:
        data, presentation_revision = _read_presentation(text.root)
        presentation_error = None
        try:
            presentation = (
                PresentationDocument.model_validate(strict_json_loads(data, ref=STUDIO_PRESENTATION_FILENAME))
                if data is not None
                else PresentationDocument()
            )
        except (ValueError, UnicodeError):
            presentation = None
            presentation_error = "unreadable_presentation"
        return StudioProjectSnapshot(
            project_id=project_id,
            revision=text.revision,
            project=text.project,
            semantic_root=project_semantic_root(text.project),
            presentation=presentation,
            presentation_revision=presentation_revision,
            presentation_error=presentation_error,
            file_revisions=[
                StudioAuthoredFileRevision(
                    address=AuthoredFileAddress(kind=_KINDS[path.parent.name], id=path.stem),
                    revision=digest,
                )
                for relative, digest in text.file_revisions.items()
                if (path := Path(relative)).parent.name in _KINDS
            ],
        )

    def _document(self, project_id: str, text: TextProject) -> StudioProjectDocument:
        return StudioProjectDocument(
            configuration=text.configuration,
            snapshot=self._snapshot(project_id, text),
        )

    def open(self, project_id: str) -> StudioProjectDocument:
        return self._document(project_id, self._read_text(project_id))

    def _read_text(self, project_id: str) -> TextProject:
        root = self._folder(project_id)
        with _folder_lock(root, write=False):
            return self._load(root)

    def read(self, project_id: str, *, revision: str | None = None) -> StudioProjectSnapshot:
        text = self._read_text(project_id)
        if revision is not None:
            _check_revision("project", revision, text.revision)
        return self._snapshot(project_id, text)

    def read_plan(
        self, project_id: str, *, revision: str | None, source_runs: Iterable[RunKey] = ()
    ) -> SettledProjectPlan:
        """Compile the acknowledged folder snapshot without opening the result store."""
        text = self._read_text(project_id)
        if revision is not None:
            _check_revision("project", revision, text.revision)
        return self._compile(project_id, text, source_runs)

    def _compile(
        self, project_id: str, text: TextProject, source_runs: Iterable[RunKey] = ()
    ) -> SettledProjectPlan:
        return SettledProjectPlan.from_text(
            project_id, text,
            compile_project(
                text.project, source_runs=source_runs, memo=self._plan_memo(project_id)
            ),
        )

    @contextmanager
    def admission_plan(self, project_id: str) -> Iterator[SettledProjectPlan]:
        """Hold the managed folder's writer lock through comparison and Host admission."""
        root = self._folder(project_id)
        with _folder_lock(root):
            yield self._compile(project_id, self._load(root))

    def new(self) -> StudioProjectDocument:
        project_id = f"untitled-{uuid4().hex}"
        configuration = ProjectConfiguration(project=project_id, store="results")
        files = dump_text_project(Project(), configuration)
        files[STUDIO_PRESENTATION_FILENAME] = _encode_presentation(PresentationDocument())
        self.root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".studio-project-", dir=self.root) as staged:
            staging = Path(staged)
            for relative, data in files.items():
                write_bytes_atomic(staging / relative, data, replace=False)
            parent = os.open(self.root, directory_open_flags(error_factory=OSError))
            try:
                rename_no_replace(
                    staging.name,
                    project_id,
                    source_dir_fd=parent,
                    destination_dir_fd=parent,
                    error_factory=OSError,
                    context="new Studio project",
                )
            finally:
                os.close(parent)
            sync_parent_directories(self.root / project_id)
        return self.open(project_id)

    def rename(self, project_id: str, body: StudioProjectRename) -> StudioProjectDocument:
        root = self._folder(project_id)
        with _folder_lock(root):
            text = self._load(root)
            _check_revision("project", body.revision, text.revision)
            if body.name != text.configuration.project:
                declaration = root / PROJECT_DECLARATION_FILENAME
                mode = stat.S_IMODE(_regular(declaration).st_mode)
                try:
                    replacement = rename_text_project_declaration(declaration.read_bytes(), body.name)
                except ProjectReadError as error:
                    raise ProjectFolderInvalid(str(error)) from error
                write_bytes_atomic(
                    declaration,
                    replacement,
                    mode=mode,
                    before_publish=lambda: _check_revision(
                        "project", text.revision, self._load(root).revision
                    ),
                )
                text = self._load(root)
            return self._document(project_id, text)

    def create_file(self, project_id: str, body: StudioProjectCreateFile) -> StudioProjectDocument:
        root = self._folder(project_id)
        file = body.file
        family = _FAMILIES[file.kind]
        relative = f"{family}/{file.id}.json"
        with _folder_lock(root):
            text = self._load(root)
            _check_revision("project", body.revision, text.revision)
            entries = text.project.documents if isinstance(file, CreateNamedDocument) else text.project.steps
            if file.id in entries:
                raise ProjectFileExists(relative)
            entry = file.document if isinstance(file, CreateNamedDocument) else file.step
            project = Project.model_validate({
                **text.project.model_dump(mode="json", by_alias=True, exclude_unset=True),
                family: {**entries, file.id: entry},
            })
            data = dump_text_project(project, text.configuration)[relative]
            try:
                write_bytes_atomic(
                    root / relative,
                    data,
                    replace=False,
                    before_publish=lambda: _check_revision(
                        "project", text.revision, self._load(root).revision
                    ),
                )
            except FileExistsError as error:
                raise ProjectFileExists(relative) from error
            return self._document(project_id, self._load(root))

    def revision(self, project_id: str) -> StudioProjectRevision:
        text = self._read_text(project_id)
        return StudioProjectRevision(
            project_id=project_id,
            revision=text.revision,
            presentation_revision=_read_presentation(text.root)[1],
        )

    def patch(self, project_id: str, body: StudioProjectPatch | StudioProjectPatchBatch) -> StudioProjectSnapshot:
        root = self._folder(project_id)
        with _folder_lock(root):
            text = self._load(root)
            _check_revision("project", body.revision, text.revision)
            files = body.files if isinstance(body, StudioProjectPatchBatch) else [body]
            original = dump_text_project(text.project, text.configuration)
            payload = text.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
            addresses = {}
            try:
                for file in files:
                    family = _FAMILIES[file.address.kind]
                    relative = f"{family}/{file.address.id}.json"
                    if relative not in original:
                        raise ProjectFolderNotFound(file.address.id)
                    if relative in addresses:
                        raise ProjectFolderInvalid(f"{relative} is addressed twice")
                    addresses[relative] = file.address
                    target = payload["sweep"]["rows"] if family == "variants" else payload[family]
                    entry = target[file.address.id]
                    for patch in file.patches:
                        apply_value_patch_in_place(
                            entry,
                            json_pointer_parts(patch.path),
                            patch.op,
                            patch.value if isinstance(patch, SetProjectValue) else None,
                            path=patch.path,
                        )
                project = Project.model_validate(payload)
                encoded = dump_text_project(project, text.configuration)
            except (ValueError, TypeError, RecursionError) as error:
                raise ProjectFolderInvalid(str(error)) from error
            changed = [relative for relative in addresses if encoded[relative] != original[relative]]

            def guard() -> None:
                _check_revision("project", text.revision, self._load(root).revision)

            if len(changed) == 1:
                relative = changed[0]
                path = root / relative
                write_bytes_atomic(
                    path,
                    encoded[relative],
                    mode=stat.S_IMODE(_regular(path).st_mode),
                    before_publish=guard,
                )
            elif changed:
                pending = [
                    _PendingFile(
                        address=addresses[relative], before=text.file_revisions[relative],
                        data=encoded[relative].decode(),
                    )
                    for relative in changed
                ]
                # Publishing the journal commits the edit. Readers recover it before opening the folder.
                write_bytes_atomic(
                    root / STUDIO_PATCH_JOURNAL_FILENAME, _PENDING_FILES.dump_json(pending), replace=False,
                    before_publish=guard,
                )
                _finish_patch(root)
            if changed:
                text = self._load(root)
            return self._snapshot(project_id, text)

    def write_presentation(
        self, project_id: str, body: StudioPresentationWrite
    ) -> StudioProjectSnapshot:
        root = self._folder(project_id)
        with _folder_lock(root):
            text = self._load(root)
            data, revision = _read_presentation(root)
            _check_revision("presentation", body.revision, revision)
            replacement = _encode_presentation(body.presentation)
            if replacement != data:
                path = root / STUDIO_PRESENTATION_FILENAME
                mode = stat.S_IMODE(_regular(path).st_mode) if data is not None else 0o600
                write_bytes_atomic(
                    path,
                    replacement,
                    mode=mode,
                    before_publish=lambda: _check_revision(
                        "presentation", revision, _read_presentation(root)[1]
                    ),
                )
                text = self._load(root)
            return self._snapshot(project_id, text)
