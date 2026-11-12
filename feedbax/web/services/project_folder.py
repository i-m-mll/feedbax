"""Read and write the text-project files addressed by Studio."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from feedbax._atomic import write_bytes_atomic
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.matrix_core import apply_value_patch_in_place
from feedbax.contracts.project import Project, json_pointer_parts
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project_io import (
    ProjectReadError,
    TextProject,
    dump_text_project,
    load_text_project,
    project_semantic_root,
    text_project_paths,
)
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.contracts.studio_project import (
    AuthoredFileAddress,
    PresentationDocument,
    SetProjectValue,
    StudioAuthoredFileRevision,
    StudioPresentationWrite,
    StudioProjectPatch,
    StudioProjectRevision,
    StudioProjectSnapshot,
)


_PRESENTATION = "presentation.json"
_FAMILIES = {"document": "documents", "step": "steps", "variant": "variants"}
_KINDS = {family: kind for kind, family in _FAMILIES.items()}
_FOLDER_ID = TypeAdapter(ProjectName)
type _FileStamp = tuple[str, int, int, int, int, int]
type _Resource = Literal["project", "presentation"]


class ProjectFolderNotFound(FileNotFoundError):
    """The addressed project does not exist in the Studio project directory."""


class ProjectFolderInvalid(ValueError):
    """The folder or the requested edit is invalid."""


class ProjectRevisionConflict(RuntimeError):
    """The addressed resource changed before the edit could be published."""

    def __init__(self, resource: _Resource, current: str) -> None:
        self.resource = resource
        self.current = current
        super().__init__(f"{resource.capitalize()} changed before this edit.")


def _regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ProjectFolderInvalid(f"{path.name} must be a regular file")
    return info


def _file_stamp(path: Path, root: Path) -> _FileStamp:
    info = _regular(path)
    return (
        path.relative_to(root).as_posix(),
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@contextmanager
def _writer_lock(root: Path) -> Iterator[None]:
    lock = root / ".studio-write.lock"
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened = os.fstat(descriptor)
        current = _regular(lock)
        if opened.st_nlink != 1 or (opened.st_dev, opened.st_ino) != (
            current.st_dev,
            current.st_ino,
        ):
            raise ProjectFolderInvalid("Project write lock changed")
        yield
    finally:
        os.close(descriptor)


def _check_revision(resource: _Resource, expected: str, current: str) -> None:
    if expected != current:
        raise ProjectRevisionConflict(resource, current)


def _read_presentation(root: Path) -> tuple[bytes | None, str]:
    path = root / _PRESENTATION
    try:
        _regular(path)
        data = path.read_bytes()
    except FileNotFoundError:
        return None, sha256_bytes(b"feedbax.presentation.v1\0")
    return data, sha256_bytes(b"feedbax.presentation.v1\1" + data)


class ProjectFolderService:
    """Own revision guards for project folders under the app-managed root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

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

    def _authored_fingerprint(self, root: Path) -> tuple[_FileStamp, ...]:
        for family in _FAMILIES.values():
            try:
                info = (root / family).lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(info.st_mode):
                raise ProjectFolderInvalid(f"{family} must be a directory")
        try:
            return tuple(_file_stamp(path, root) for path in text_project_paths(root))
        except (OSError, ProjectReadError) as error:
            raise ProjectFolderInvalid("Project files are unreadable") from error

    def fingerprint(self, project_id: str) -> tuple[_FileStamp, ...]:
        root = self._folder(project_id)
        authored = self._authored_fingerprint(root)
        try:
            return (*authored, _file_stamp(root / _PRESENTATION, root))
        except FileNotFoundError:
            return authored

    def _load(self, root: Path) -> TextProject:
        before = self._authored_fingerprint(root)
        try:
            text = load_text_project(root)
        except ProjectReadError as error:
            raise ProjectFolderInvalid(str(error)) from error
        if before != self._authored_fingerprint(root):
            raise ProjectFolderInvalid("Project changed while it was being read")
        return text

    def _snapshot(self, project_id: str, text: TextProject) -> StudioProjectSnapshot:
        data, presentation_revision = _read_presentation(text.root)
        presentation_error = None
        try:
            presentation = (
                PresentationDocument.model_validate(strict_json_loads(data, ref=_PRESENTATION))
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

    def read(self, project_id: str) -> StudioProjectSnapshot:
        return self._snapshot(project_id, self._load(self._folder(project_id)))

    def revision(self, project_id: str) -> StudioProjectRevision:
        root = self._folder(project_id)
        return StudioProjectRevision(
            project_id=project_id,
            revision=self._load(root).revision,
            presentation_revision=_read_presentation(root)[1],
        )

    def patch(self, project_id: str, body: StudioProjectPatch) -> StudioProjectSnapshot:
        root = self._folder(project_id)
        with _writer_lock(root):
            text = self._load(root)
            _check_revision("project", body.revision, text.revision)
            family = _FAMILIES[body.address.kind]
            relative = f"{family}/{body.address.id}.json"
            original = dump_text_project(text.project, text.configuration)
            if relative not in original:
                raise ProjectFolderNotFound(body.address.id)
            payload = text.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
            target = payload["sweep"]["rows"] if family == "variants" else payload[family]
            entry = target[body.address.id]
            try:
                for patch in body.patches:
                    apply_value_patch_in_place(
                        entry,
                        json_pointer_parts(patch.path),
                        patch.op,
                        patch.value if isinstance(patch, SetProjectValue) else None,
                        path=patch.path,
                    )
                project = Project.model_validate(payload)
                replacement = dump_text_project(project, text.configuration)[relative]
            except (ValueError, TypeError, RecursionError) as error:
                raise ProjectFolderInvalid(str(error)) from error
            if replacement != original[relative]:
                path = root / relative
                write_bytes_atomic(
                    path,
                    replacement,
                    mode=stat.S_IMODE(_regular(path).st_mode),
                    before_publish=lambda: _check_revision(
                        "project", text.revision, self._load(root).revision
                    ),
                )
                text = self._load(root)
            return self._snapshot(project_id, text)

    def write_presentation(
        self, project_id: str, body: StudioPresentationWrite
    ) -> StudioProjectSnapshot:
        root = self._folder(project_id)
        with _writer_lock(root):
            text = self._load(root)
            data, revision = _read_presentation(root)
            _check_revision("presentation", body.revision, revision)
            replacement = (
                body.presentation.model_dump_json(indent=2, exclude_unset=True) + "\n"
            ).encode()
            if replacement != data:
                path = root / _PRESENTATION
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
