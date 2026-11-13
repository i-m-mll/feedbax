"""Read and write the text-project files addressed by Studio."""

from __future__ import annotations

from base64 import b64decode, urlsafe_b64encode
from binascii import Error as Base64Error
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import fcntl
from functools import lru_cache
from hmac import compare_digest, digest
import os
from pathlib import Path
from secrets import token_bytes
import stat
from tempfile import TemporaryDirectory
from typing import Literal, TYPE_CHECKING
from uuid import uuid4

from pydantic import JsonValue, TypeAdapter, ValidationError

from feedbax._atomic import sync_parent_directories, write_bytes_atomic
from feedbax._secure_fs import directory_open_flags, rename_no_replace
from feedbax.expression_registry import FunctionRegistry
from feedbax.tasks.registry import PythonTaskRegistry
from feedbax.contracts.base import StrictModel, sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.figure_document import FIGURE_DOCUMENT_SCHEMA, is_figure_document_pin
from feedbax.contracts.matrix_core import apply_value_patch_in_place
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project import Project, ProjectValueEdit, Sha256, json_pointer_parts
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
    TextProjectExternalInputs,
    dump_text_project,
    read_text_project_snapshot,
    project_semantic_root,
    rename_text_project_declaration,
    text_file_stamp,
    text_project_fingerprint,
)
from feedbax.contracts.project_plan import ProjectPlan, ProjectPlanMemo, RunKey, compile_project
from feedbax.contracts.project_resolution import BaseReaders
from feedbax.contracts.project_sweep import GeneratedAxis, NamedAxis, Sweep, model_field_target
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.contracts.studio_project import (
    PRESENTATION_ADDRESS,
    AuthoredFileAddress,
    CreateNamedDocument,
    PresentationDocument,
    StudioProjectFilePatch,
    RemoveProjectValue,
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

if TYPE_CHECKING:
    from feedbax.analysis.operations import OperationRegistry
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.model_edit import (
        ModelConnectionReview, ModelConnectionReviewRequest, ModelEditRequest, ModelEditResponse, ModelUndoRequest,
    )


_FAMILIES = {"document": "documents", "step": "steps", "variant": "variants"}
_KINDS = {family: kind for kind, family in _FAMILIES.items()}
_FOLDER_ID = TypeAdapter(ProjectName)
type _Resource = Literal["project", "presentation"]


class _ModelInverse(StrictModel):
    project: ProjectName
    document: ProjectName
    payload: tuple[ProjectValueEdit, ...]
    layout: tuple[ProjectValueEdit, ...]
    changed: tuple[ModelAddress, ...]
    before: Sha256
    had_entry: bool
    had_documents: bool


_MODEL_LAYOUT_FIELDS = ("positions", "flipped", "collapsed", "waypoints")


def _document_sweep_paths(project: Project, document: str) -> set[tuple[str, ...] | ModelAddress]:
    prefix = f"documents.{document}.payload."

    def paths(sweep: Sweep):
        for axis in sweep.axes.values():
            if isinstance(axis, GeneratedAxis):
                yield axis.path
            if isinstance(axis, NamedAxis):
                for entry in axis.values.values():
                    if entry is not None:
                        yield from entry.patches
        for entry in (*sweep.rows.values(), *sweep.exceptions.values()):
            if entry is not None:
                yield from entry.patches

    targets = set()
    for sweep in (project.sweep, *(step.sweep for step in project.steps.values())):
        if sweep is None:
            continue
        for path in paths(sweep):
            if path.startswith(prefix):
                targets.add(tuple(path.removeprefix(prefix).split(".")))
                continue
            try:
                target = model_field_target(path)
            except ValueError:
                continue  # The Plan reports malformed authored targets.
            if target is not None and target.address.document == document:
                targets.add(target.address)
    return targets


def _model_field_exists(payload, path):
    for key in path:
        if isinstance(payload, list) and key.isdecimal() and int(key) < len(payload):
            payload = payload[int(key)]
        elif isinstance(payload, dict) and key in payload:
            payload = payload[key]
        else:
            return False
    return True


def _loses_model_target(payload, next_payload, targets, state, result, authority, functions):
    from feedbax.studio.model_edit import ModelEditDraft

    if any(_model_field_exists(payload, path) and not _model_field_exists(next_payload, path)
           for path in targets if isinstance(path, tuple)):
        return True
    addresses = [address for address in targets if isinstance(address, ModelAddress)]
    if not addresses:
        return False
    document = addresses[0].document
    before = ModelEditDraft(document, state, authority.port, authority.accepts)
    after = ModelEditDraft(document, result, authority.port, authority.accepts)
    return any(
        before.has_field(address, functions=functions) and not after.has_field(address, functions=functions)
        for address in addresses
    )


def _model_edit_source(
    snapshot: StudioProjectSnapshot, document: ProjectName,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    source = snapshot.project.documents.get(document)
    if source is None or source.schema_id != MODEL_DOCUMENT_SCHEMA_ID:
        raise ProjectFolderNotFound(document)
    if snapshot.presentation is None:
        raise ProjectFolderInvalid("The project presentation document is unreadable")
    entry = snapshot.presentation.documents.get(document)
    present = entry.model_dump(mode="json", exclude_unset=True) if entry is not None else {}
    layout = {key: present[key] for key in _MODEL_LAYOUT_FIELDS if key in present}
    assert isinstance(source.payload, dict)
    return source.payload, layout



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


def _apply_file_edits(entry: dict, file: StudioProjectFilePatch, *, model_operation: bool = False) -> None:
    def check_model() -> None:
        if file.address.kind == "document" and not model_operation and entry.get("schema") == MODEL_DOCUMENT_SCHEMA_ID:
            raise ProjectFolderInvalid("Model documents are edited through addressed operations")

    check_model()
    for patch in file.patches:
        apply_value_patch_in_place(
            entry, json_pointer_parts(patch.path), patch.op,
            patch.value if isinstance(patch, SetProjectValue) else None, path=patch.path,
        )
    check_model()


def _project_file_entry(payload: dict, address: AuthoredFileAddress) -> dict:
    family = _FAMILIES[address.kind]
    return (payload["sweep"]["rows"] if family == "variants" else payload[family])[address.id]


def preview_authored_files(project: Project, files: Sequence[StudioProjectFilePatch]) -> Project:
    payload = project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    for file in files:
        _apply_file_edits(_project_file_entry(payload, file.address), file)
    return Project.model_validate(payload)


@dataclass(frozen=True)
class SettledProjectPlan:
    """The compiler portion of Plan, bound to one saved project revision."""

    project_id: str
    revision: str
    semantic_root: str
    configuration: ProjectConfiguration
    store_path: Path
    plan: ProjectPlan
    readers: BaseReaders

    @classmethod
    def from_text(
        cls, project_id: str, text: TextProject, plan: ProjectPlan, *, readers: BaseReaders = BaseReaders(),
    ) -> SettledProjectPlan:
        return cls(
            project_id, text.revision, project_semantic_root(text.project),
            text.configuration, text.store_path, plan, readers,
        )


def _may_read_figure_pin(project: Project) -> bool:
    figures = {
        name: document for name, document in project.documents.items()
        if document.schema_id == FIGURE_DOCUMENT_SCHEMA
    }
    if any(is_figure_document_pin(document.payload) for document in figures.values()):
        return True
    if any(figures.keys() & step.document_overrides.keys() for step in project.steps.values()):
        return True
    if (sweep := project.sweep) is None:
        return False

    def affects_figures(path: str) -> bool:
        family, *parts = path.split(".")
        return family == "documents" or bool(figures) and family == "steps" and (
            len(parts) == 1 or parts[1:2] == ["document_overrides"]
        )

    patches = [
        *sweep.rows.values(), *sweep.exceptions.values(),
        *(value for axis in sweep.axes.values() if isinstance(axis, NamedAxis) for value in axis.values.values()),
    ]
    return any(
        affects_figures(path) for value in patches if value is not None for path in value.patches
    ) or any(
        isinstance(axis, GeneratedAxis) and affects_figures(axis.path) for axis in sweep.axes.values()
    )

def _regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ProjectFolderInvalid(f"{path.name} must be a regular file")
    return info


class _PendingFile(StrictModel):
    address: AuthoredFileAddress
    before: Sha256 | None
    data: str

    @property
    def relative(self) -> str:
        return _relative_file(self.address)


_PENDING_FILES = TypeAdapter(list[_PendingFile])


def _relative_file(address: AuthoredFileAddress) -> str:
    if address == PRESENTATION_ADDRESS:
        return STUDIO_PRESENTATION_FILENAME
    if address.kind not in _FAMILIES:
        raise ProjectFolderInvalid("Invalid project file address")
    return f"{_FAMILIES[address.kind]}/{address.id}.json"


def _patch_preimage(path: Path, file: _PendingFile) -> tuple[int, str | None]:
    try:
        mode = stat.S_IMODE(_regular(path).st_mode)
        return mode, sha256_bytes(path.read_bytes())
    except FileNotFoundError:
        if file.relative != STUDIO_PRESENTATION_FILENAME or file.before is not None:
            raise
        return 0o600, None


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
        mode, current = _patch_preimage(path, file)
        data = file.data.encode()
        if current == sha256_bytes(data):
            continue
        if current != file.before:
            raise ProjectFolderInvalid(f"{file.relative} changed during a Studio edit")
        replacements.append((file, path, mode, data))

    for file, path, mode, data in replacements:
        def guard() -> None:
            if _patch_preimage(path, file)[1] != file.before:
                raise ProjectFolderInvalid(f"{file.relative} changed during a Studio edit")

        write_bytes_atomic(path, data, mode=mode, replace=file.before is not None, before_publish=guard)
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


def _decode_presentation(data: bytes | None) -> PresentationDocument:
    return (
        PresentationDocument.model_validate(strict_json_loads(data, ref=STUDIO_PRESENTATION_FILENAME))
        if data is not None else PresentationDocument()
    )


def write_new_project_folder(
    destination: Path,
    project: Project,
    configuration: ProjectConfiguration,
    presentation: PresentationDocument,
) -> None:
    """Publish the ordinary authored folder and presentation in one exclusive rename."""
    root = destination.parent.resolve(strict=True)
    files = dump_text_project(project, configuration)
    files[STUDIO_PRESENTATION_FILENAME] = _encode_presentation(presentation)
    with TemporaryDirectory(prefix=".studio-project-", dir=root) as staged:
        staging = Path(staged)
        for relative, data in files.items():
            write_bytes_atomic(staging / relative, data, replace=False)
        parent = os.open(root, directory_open_flags(error_factory=OSError))
        try:
            rename_no_replace(
                staging.name,
                destination.name,
                source_dir_fd=parent,
                destination_dir_fd=parent,
                error_factory=OSError,
                context="new Studio project",
            )
        finally:
            os.close(parent)
        sync_parent_directories(root / destination.name)


class ProjectFolderService:
    """Own revision guards for project folders under the app-managed root."""

    def __init__(
        self, root: Path, *, operations: OperationRegistry | None = None,
        components: ComponentRegistry | None = None,
        functions: FunctionRegistry | None = None,
        python_tasks: PythonTaskRegistry | None = None,
    ) -> None:
        self.root = root.resolve()
        self.operations = operations
        self.components = components
        self.functions = functions
        self.python_tasks = python_tasks
        self._plan_memo = lru_cache(maxsize=8)(lambda _project_id: ProjectPlanMemo())
        self._model_undo_key = token_bytes(32)

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
            presentation = _decode_presentation(data)
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
        root = self._folder(project_id)
        with _folder_lock(root, write=False):
            return self._document(project_id, self._load(root))

    def open_selected_folder(self, folder: Path) -> StudioProjectDocument:
        """Open the managed direct child selected by the native folder picker."""
        try:
            inside = folder.is_absolute() and folder.parent.resolve(strict=True) == self.root
        except (OSError, RuntimeError, ValueError):
            inside = False
        if not inside:
            raise ProjectFolderInvalid("Choose a project folder inside the Studio projects directory.")
        return self.open(folder.name)

    def _read_text(self, project_id: str) -> TextProject:
        root = self._folder(project_id)
        with _folder_lock(root, write=False):
            return self._load(root)

    def read(self, project_id: str, *, revision: str | None = None) -> StudioProjectSnapshot:
        root = self._folder(project_id)
        with _folder_lock(root, write=False):
            text = self._load(root)
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
        readers = BaseReaders(
            figure_external=TextProjectExternalInputs(text, authority=self.root).read,
        ) if _may_read_figure_pin(text.project) else BaseReaders()
        return SettledProjectPlan.from_text(
            project_id, text,
            compile_project(
                text.project, source_runs=source_runs, memo=self._plan_memo(project_id),
                operations=self.operations, components=self.components, functions=self.functions,
                python_tasks=self.python_tasks, readers=readers,
            ),
            readers=readers,
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
        self.root.mkdir(parents=True, exist_ok=True)
        write_new_project_folder(self.root / project_id, Project(), configuration, PresentationDocument())
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
        root = self._folder(project_id)
        with _folder_lock(root, write=False):
            return StudioProjectRevision(
                project_id=project_id,
                revision=self._load(root).revision,
                presentation_revision=_read_presentation(root)[1],
            )

    def patch(self, project_id: str, body: StudioProjectPatch | StudioProjectPatchBatch) -> StudioProjectSnapshot:
        return self._patch(project_id, body, model_operation=False)

    def review_model_connections(
        self, project_id: str, body: ModelConnectionReviewRequest, components: ComponentRegistry,
    ) -> ModelConnectionReview:
        from feedbax.contracts.model_document import ModelDocument
        from feedbax.studio.model_edit import review_model_connections
        from feedbax.studio.model_ports import ModelRegistryPorts

        snapshot = self.read(project_id, revision=body.revision)
        payload, _ = _model_edit_source(snapshot, body.graph.document)
        try:
            authority = ModelRegistryPorts(components)
            library = tuple({
                "kind": "leaf", "type": item.identity.type_id if item.identity else item.name,
                "revision": item.revision, "fields": {}, "port_counts": {},
            } for item in components.list_all() if not item.is_composite)
            return review_model_connections(
                ModelDocument.model_validate(payload), body, layout=authority.layout, next_input=authority.next_input,
                library=library, accepts=authority.accepts,
            )
        except ValueError as error:
            raise ProjectFolderInvalid(str(error)) from error

    def edit_model(
        self, project_id: str, body: ModelEditRequest, components: ComponentRegistry,
    ) -> ModelEditResponse:
        from feedbax.contracts.model_edit import (
            ModelProjectionRequest,
            ModelEditApplied, ModelEditProblem, ModelEditRefused, ModelEditResponse, ModelEditState,
        )
        from feedbax.studio.model_edit import apply_model_edit
        from feedbax.studio.model_ports import ModelRegistryPorts
        from feedbax.web.services.model_projection import selected_task_signal

        snapshot = self.read(project_id, revision=body.revision)
        _check_revision("presentation", body.presentation_revision, snapshot.presentation_revision)
        payload, layout = _model_edit_source(snapshot, body.document)
        state = ModelEditState.model_validate({
            "document": payload, **layout,
        })
        authority = ModelRegistryPorts(components)
        result = apply_model_edit(
            body.document, state, body.command, ports=authority.port, accepts=authority.accepts,
            next_input=authority.next_input, functions=self.functions,
            task_signal=lambda command: selected_task_signal(self, project_id, ModelProjectionRequest(
                revision=body.revision, run=command.run, document=body.document,
            ), command.signal),
        )
        if isinstance(result, ModelEditProblem):
            return ModelEditResponse(result=ModelEditRefused(status="refused", problem=result))
        token = None
        if result.state != state:
            next_payload = result.state.document.model_dump(mode="json") if result.state.document != state.document else payload
            targets = _document_sweep_paths(snapshot.project, body.document) if next_payload is not payload else ()
            if targets and _loses_model_target(payload, next_payload, targets, state, result.state, authority, self.functions):
                return ModelEditResponse(result=ModelEditRefused(
                    status="refused", problem=ModelEditProblem(
                        address=result.changed[0], detail="Convert the affected sweep to a Value before moving or removing its field.",
                    ),
                ))
            next_layout = layout.copy()
            updated_layout = result.state.model_dump(mode="json", include=set(_MODEL_LAYOUT_FIELDS))
            for key in _MODEL_LAYOUT_FIELDS:
                if getattr(result.state, key) == getattr(state, key):
                    continue
                if updated_layout[key]:
                    next_layout[key] = updated_layout[key]
                else:
                    next_layout.pop(key, None)
            token = self._model_inverse(
                snapshot, body.document, (payload, layout), (next_payload, next_layout), result.changed,
            )
            snapshot = self._write_model_change(snapshot, body.document, next_payload, next_layout)
        return ModelEditResponse(result=ModelEditApplied(
            status="applied", snapshot=snapshot, changed=result.changed, clipboard=result.clipboard,
            undo_token=token,
        ))

    def undo_model(self, project_id: str, body: ModelUndoRequest) -> ModelEditResponse:
        from feedbax.contracts.model_edit import ModelEditApplied, ModelEditResponse
        from feedbax.studio.model_undo import apply_delta

        try:
            encoded = b64decode(body.token, altchars=b"-_", validate=True)
            signature, data = encoded[:32], encoded[32:]
            if not compare_digest(signature, digest(self._model_undo_key, data, "sha256")):
                raise ValueError("Invalid model undo signature")
            edit = _ModelInverse.model_validate_json(data)
        except (Base64Error, ValueError) as error:
            raise ProjectFolderInvalid("The model edit history entry is unavailable") from error
        if edit.project != project_id:
            raise ProjectFolderInvalid("The model edit history entry belongs to another project")
        snapshot = self.read(project_id, revision=body.revision)
        _check_revision("presentation", body.presentation_revision, snapshot.presentation_revision)
        current = _model_edit_source(snapshot, edit.document)
        if self._model_edit_digest(current) != edit.before:
            raise ProjectRevisionConflict("project", snapshot.revision)
        payload, layout = deepcopy(current)
        apply_delta(payload, edit.payload)
        apply_delta(layout, edit.layout)
        token = self._model_inverse(snapshot, edit.document, current, (payload, layout), edit.changed)
        snapshot = self._write_model_change(
            snapshot, edit.document, payload, layout,
            had_entry=edit.had_entry, had_documents=edit.had_documents,
        )
        return ModelEditResponse(result=ModelEditApplied(
            status="applied", snapshot=snapshot, changed=edit.changed, undo_token=token,
        ))

    @staticmethod
    def _model_edit_digest(value: tuple[dict, dict]) -> str:
        return sha256_bytes(canonical_json_v2_bytes(list(value)))

    def _model_inverse(
        self, snapshot: StudioProjectSnapshot, document: ProjectName,
        before: tuple[dict, dict], after: tuple[dict, dict],
        changed: tuple[ModelAddress, ...],
    ) -> str:
        from feedbax.studio.model_undo import inverse_delta

        assert snapshot.presentation is not None
        inverse = _ModelInverse(
            project=snapshot.project_id, document=document, before=self._model_edit_digest(after),
            payload=tuple(inverse_delta(before[0], after[0])),
            layout=tuple(inverse_delta(before[1], after[1])),
            changed=changed,
            had_entry=document in snapshot.presentation.documents,
            had_documents="documents" in snapshot.presentation.model_fields_set,
        )
        data = inverse.model_dump_json().encode()
        return urlsafe_b64encode(digest(self._model_undo_key, data, "sha256") + data).decode()

    def _write_model_change(
        self, snapshot: StudioProjectSnapshot, document: ProjectName,
        payload: dict[str, JsonValue], layout: dict[str, JsonValue],
        *, had_entry: bool = True, had_documents: bool = True,
    ) -> StudioProjectSnapshot:
        from feedbax.contracts.project_sources import pointer

        assert snapshot.presentation is not None
        before, old_layout = _model_edit_source(snapshot, document)
        files: list[StudioProjectFilePatch] = []
        if canonical_json_v2_bytes(before) != canonical_json_v2_bytes(payload):
            files.append(StudioProjectFilePatch(
                address=AuthoredFileAddress(kind="document", id=document),
                patches=[SetProjectValue(op="replace", path="/payload", value=payload)],
            ))
        if layout != old_layout:
            entry = snapshot.presentation.documents.get(document)
            current = entry.model_dump(mode="json", exclude_unset=True) if entry is not None else {}
            next_entry = {key: value for key, value in current.items() if key not in _MODEL_LAYOUT_FIELDS} | layout
            path = pointer(("documents", document))
            if not next_entry and not had_entry:
                if not had_documents and snapshot.presentation.documents.keys() == {document}:
                    path = "/documents"
                patch = RemoveProjectValue(op="remove", path=path)
            elif "documents" not in snapshot.presentation.model_fields_set:
                patch = SetProjectValue(op="add", path="/documents", value={document: next_entry})
            else:
                patch = SetProjectValue(op="replace" if entry is not None else "add", path=path, value=next_entry)
            files.append(StudioProjectFilePatch(address=PRESENTATION_ADDRESS, patches=[patch]))
        if not files:
            return snapshot
        return self._patch(snapshot.project_id, StudioProjectPatchBatch(
            revision=snapshot.revision, presentation_revision=snapshot.presentation_revision, files=files,
        ), model_operation=True)

    def _patch(
        self, project_id: str, body: StudioProjectPatch | StudioProjectPatchBatch, *, model_operation: bool,
    ) -> StudioProjectSnapshot:
        root = self._folder(project_id)
        with _folder_lock(root):
            text = self._load(root)
            _check_revision("project", body.revision, text.revision)
            files = body.files if isinstance(body, StudioProjectPatchBatch) else [body]
            original = dump_text_project(text.project, text.configuration)
            payload = text.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
            addresses = {}
            presentation_data = None
            edits_presentation = any(file.address == PRESENTATION_ADDRESS for file in files)
            if edits_presentation:
                if body.presentation_revision is None:
                    raise ProjectFolderInvalid("A presentation edit requires its revision")
                presentation_data, current = _read_presentation(root)
                _check_revision("presentation", body.presentation_revision, current)
            try:
                presentation = (
                    _decode_presentation(presentation_data).model_dump(mode="json", exclude_unset=True)
                    if edits_presentation else None
                )
                for file in files:
                    relative = _relative_file(file.address)
                    if relative in addresses:
                        raise ProjectFolderInvalid(f"{relative} is addressed twice")
                    addresses[relative] = file.address
                    if relative == STUDIO_PRESENTATION_FILENAME:
                        assert presentation is not None
                        entry = presentation
                    else:
                        if relative not in original:
                            raise ProjectFolderNotFound(file.address.id)
                        entry = _project_file_entry(payload, file.address)
                    _apply_file_edits(entry, file, model_operation=model_operation)
                project = Project.model_validate(payload)
                encoded = dump_text_project(project, text.configuration)
                if presentation is not None:
                    original[STUDIO_PRESENTATION_FILENAME] = _encode_presentation(_decode_presentation(presentation_data))
                    encoded[STUDIO_PRESENTATION_FILENAME] = _encode_presentation(PresentationDocument.model_validate(presentation))
            except (ValueError, TypeError, RecursionError) as error:
                raise ProjectFolderInvalid(str(error)) from error
            changed = [relative for relative in addresses if encoded[relative] != original[relative]]

            def guard() -> None:
                _check_revision("project", text.revision, self._load(root).revision)
                if edits_presentation:
                    _check_revision("presentation", body.presentation_revision, _read_presentation(root)[1])

            if len(changed) == 1 and changed[0] != STUDIO_PRESENTATION_FILENAME:
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
                        address=addresses[relative],
                        before=(
                            sha256_bytes(presentation_data) if presentation_data is not None else None
                        ) if relative == STUDIO_PRESENTATION_FILENAME else text.file_revisions[relative],
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
