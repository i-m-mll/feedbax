"""Revision-guarded edits to a named text project and its presentation."""

from __future__ import annotations

from base64 import b64decode, b64encode
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
import fcntl
from functools import lru_cache, partial
import os
from pathlib import Path
import re
import stat
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Literal, Protocol, TYPE_CHECKING
from uuid import uuid4

from pydantic import JsonValue, TypeAdapter, ValidationError

from feedbax._atomic import sync_parent_directories, write_bytes_atomic
from feedbax._secure_fs import (
    DirectoryRecord, SecurePathRigor, close_directory_chain, directory_open_flags, open_directory_chain,
    open_existing_directory, open_existing_file, recheck_directory_chain, rename_no_replace,
)
from feedbax.expression_registry import FunctionRegistry
from feedbax.tasks.registry import PythonTaskRegistry
from feedbax.training.methods import TrainingMethodRegistry
from feedbax.contracts.base import StrictModel, sha256_bytes
from feedbax.contracts.figure_document import FIGURE_DOCUMENT_SCHEMA, is_figure_document_pin
from feedbax.contracts.matrix_core import apply_value_patch_in_place
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project import (
    BasedDocument, Document, Project, RootDocument, Sha256, diff, json_pointer_parts, same_json,
)
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
    authored_declaration_revision,
    dump_text_project,
    parse_text_project_files,
    project_declaration_name,
    read_text_project_declaration,
    read_text_project_snapshot,
    project_semantic_root,
    replay_text_project_declaration,
    require_text_declaration,
    text_file_stamp,
    text_project_fingerprint,
)
from feedbax.contracts.project_plan import ProjectPlan, ProjectPlanMemo, RunKey, compile_project, compiler_authority
from feedbax.contracts.project_resolution import BaseReaders, ProjectDiagnostic, compose_document
from feedbax.contracts.project_sweep import GeneratedAxis, NamedAxis, Sweep, model_field_target
from feedbax.persistence.project_subject import terminal_train_reader
from feedbax.project_members import (
    detach_refusals, duplicate_step, generated_name, references_member, rename_member,
)
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.studio.selection import StoredEdit
from feedbax.contracts.studio_project import (
    PRESENTATION_ADDRESS,
    AuthoredFileAddress,
    CreateNamedDocument,
    MemberFamily,
    PresentationDocument,
    ReplayDeclarationAddress,
    ReplayFileAddress,
    StudioProjectFilePatch,
    RemoveProjectValue,
    SetProjectValue,
    StudioAuthoredDeclaration,
    StudioAuthoredFileRevision,
    StudioMemberCommand,
    StudioMemberCommandResult,
    StudioPresentationWrite,
    StudioProjectDocument,
    StudioProjectCreateFile,
    StudioProjectPatch,
    StudioProjectPatchBatch,
    StudioProjectRename,
    StudioProjectReplay,
    StudioProjectRevision,
    StudioProjectSnapshot,
)

if TYPE_CHECKING:
    from feedbax.analysis.operations import OperationRegistry
    from feedbax.plot.templates import FigureTemplateRegistry
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.model_edit import (
        ModelClipboard, ModelEditProblem, ModelEditRequest,
    )


_FAMILIES = {"document": "documents", "step": "steps", "variant": "variants"}
_KINDS = {family: kind for kind, family in _FAMILIES.items()}
_FOLDER_ID = TypeAdapter(ProjectName)
_CANONICAL_PROJECT_ID = re.compile(r"external-[0-9a-f]{64}")
type _Resource = Literal["project", "presentation"]


_MODEL_LAYOUT_FIELDS = ("positions", "flipped", "collapsed", "waypoints")


def _document_sweep_paths(
    project: Project, document: str, read_plan: Callable[[], ProjectPlan],
) -> list[tuple[str, ...] | ModelAddress]:
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

    targets: list[tuple[str, ...] | ModelAddress] = []
    scoped: dict[str, list[ModelAddress]] = {}
    has_step_targets = False
    for sweep in (project.sweep, *(step.sweep for step in project.steps.values())):
        if sweep is None:
            continue
        for path in paths(sweep):
            if path.startswith(prefix):
                targets.append(tuple(path.removeprefix(prefix).split(".")))
                continue
            try:
                target = model_field_target(path)
            except ValueError:
                continue  # The Plan reports malformed authored targets.
            if target is None:
                continue
            if target.document == document:
                targets.append(target.address)
            elif target.step is not None:
                scoped.setdefault(target.step, []).append(target.address)
            elif target.document is None:
                has_step_targets = True
    if not scoped and not has_step_targets:
        return targets

    plan = read_plan()
    rows = {row.key: row for row in plan.rows}

    def uses_model(row) -> bool:
        ids = row.document_ids
        return document in (ids.get("payload.trial_documents.model"), ids.get("payload.graph"), ids.get("inputs.subject"))

    steps = {(key.variant, key.step) for key, row in rows.items() if uses_model(row)}
    for run in plan.runs:
        subject = run.inputs.get("subject") if run.kind == "evaluate" else None
        if subject is not None and any(key in rows and uses_model(rows[key]) for key in subject.sources):
            steps.add((run.key.variant, run.key.step))
    for variant, name in steps:
        targets.extend(scoped.get(name, ()))
        if sweep := plan.steps[variant, name].sweep:
            for path in paths(sweep):
                try:
                    target = model_field_target(path)
                except ValueError:
                    continue  # The Plan reports malformed authored targets.
                if target is not None and target.step is None and target.document is None:
                    targets.append(target.address)
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
    before = ModelEditDraft(state, authority.port, authority.accepts)
    after = ModelEditDraft(result, authority.port, authority.accepts)
    return any(
        before.has_field(address, functions=functions) and not after.has_field(address, functions=functions)
        for address in addresses
    )


def model_edit_source(
    snapshot: StudioProjectSnapshot, document: ProjectName,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    source = compose_document(snapshot.project.documents.get, document)
    if isinstance(source, ProjectDiagnostic) and source.document == document and source.document_path is None:
        raise ProjectFolderNotFound(document)
    if isinstance(source, ProjectDiagnostic):
        raise ProjectFolderInvalid(source.detail)
    if source.schema_id != MODEL_DOCUMENT_SCHEMA_ID:
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
        _apply_file_edits(_project_file_entry(payload, file.address), file, model_operation=True)
    return Project.model_validate(payload)


def document_file(
    name: str, document: Document, content: dict[str, JsonValue], base: dict[str, JsonValue] | None,
) -> StudioProjectFilePatch | None:
    """The document's file holding ``content``: its payload, or its differences from ``base``; none when unchanged."""
    address = AuthoredFileAddress(kind="document", id=name)
    if isinstance(document, RootDocument):
        if same_json(document.payload, content):
            return None
        return StudioProjectFilePatch(address=address, patches=[SetProjectValue(op="replace", path="/payload", value=content)])
    assert base is not None
    differences = diff(base, content)
    if differences == document.differences:
        return None
    return StudioProjectFilePatch(address=address, patches=[SetProjectValue(
        op="replace" if "differences" in document.model_fields_set else "add", path="/differences",
        value={path: value.model_dump(mode="json") for path, value in differences.items()},
    )])


type ModelStore = Callable[[Project, ModelEditRequest, dict[str, JsonValue], dict[str, JsonValue]], StoredEdit]
"""Store a Model edit's change of the shown document's composed content, from the first payload to the second, after
the request's step files: the documents it creates and the patches of files."""


def _layout_patches(
    presentation: PresentationDocument, layouts: dict[str, dict[str, JsonValue]],
    *, had_entry: bool = True, had_documents: bool = True,
) -> list[SetProjectValue | RemoveProjectValue]:
    """The presentation patches that give each document its Model layout, keeping its other presentation."""
    from feedbax.contracts.project_sources import pointer

    patches: list[SetProjectValue | RemoveProjectValue] = []
    has_documents = "documents" in presentation.model_fields_set
    for document, layout in layouts.items():
        entry = presentation.documents.get(document)
        current = entry.model_dump(mode="json", exclude_unset=True) if entry is not None else {}
        next_entry = {key: value for key, value in current.items() if key not in _MODEL_LAYOUT_FIELDS} | layout
        if next_entry == current:
            continue
        path = pointer(("documents", document))
        if not next_entry and not had_entry:
            if not had_documents and presentation.documents.keys() == {document}:
                path = "/documents"
            patches.append(RemoveProjectValue(op="remove", path=path))
        elif not has_documents:
            patches.append(SetProjectValue(op="add", path="/documents", value={document: next_entry}))
            has_documents = True
        else:
            patches.append(SetProjectValue(op="replace" if entry is not None else "add", path=path, value=next_entry))
    return patches


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
    project_root: Path
    detach_refused: frozenset[str]

    @classmethod
    def from_text(
        cls, location: ProjectFolderLocation, text: TextProject, plan: ProjectPlan, *,
        readers: BaseReaders = BaseReaders(),
        store: Path | None = None,
    ) -> SettledProjectPlan:
        return cls(
            location.project_id, text.revision, project_semantic_root(text.project),
            text.configuration, text.store_path if store is None else store, plan, readers, text.root,
            detach_refusals(plan, text.project.documents),
        )


class SavedProjectPlans(Protocol):
    def read_plan(self, project_id: str, *, revision: str | None) -> SettledProjectPlan: ...

    def admission_plan(self, project_id: str) -> AbstractContextManager[SettledProjectPlan]: ...


@dataclass(frozen=True)
class _SavedCompilation:
    stamp: tuple[object, ...]
    plan: SettledProjectPlan
    observations: dict[str, dict[str, object]]


class SavedPlanMemo(ProjectPlanMemo):
    """Retain a reviewed compilation and the data its external reads observed."""

    def __init__(self) -> None:
        super().__init__()
        self.lock = Lock()
        self.saved: _SavedCompilation | None = None


def _observe_plan_input[T](reader: Callable[[str], T], observations: dict[str, object], name: str) -> T:
    try:
        value = reader(name)
    except (OSError, ValueError) as error:
        observations[name] = (type(error), str(error))
        raise
    observations[name] = sha256_bytes(value) if isinstance(value, bytes) else value
    return value


def _same_plan_inputs(readers: BaseReaders, expected: dict[str, dict[str, object]]) -> bool:
    for kind, observed in expected.items():
        reader = getattr(readers, kind)
        if reader is None:
            return False
        current: dict[str, object] = {}
        for name in observed:
            try:
                _observe_plan_input(reader, current, name)
            except (OSError, ValueError):
                pass  # The failed read is also an observation.
        if observed != current:
            return False
    return True


def compile_saved_project(
    text: TextProject, *, project_id: str, store: Path, readers: BaseReaders,
    operations: OperationRegistry | None = None, components: ComponentRegistry | None = None,
    functions: FunctionRegistry | None = None, python_tasks: PythonTaskRegistry | None = None,
    figure_templates: FigureTemplateRegistry | None = None,
    source_runs: Iterable[RunKey] = (), memo: SavedPlanMemo | None = None,
    training_methods: TrainingMethodRegistry | None = None,
) -> SettledProjectPlan:
    """Compile exact authored bytes and reuse them only with matching input authority."""
    readers = replace(readers, terminal_train=terminal_train_reader(store))
    semantic_root = project_semantic_root(text.project)
    source_runs = frozenset(source_runs)
    stamp = (
        project_id, text.revision, semantic_root, text.root, store, source_runs,
        compiler_authority(components, operations, functions, python_tasks, training_methods, figure_templates),
        readers.figure_external is not None,
    )
    observations: dict[str, dict[str, object]] = {}
    observed_readers = {}
    for kind in ("terminal_train", "figure_external"):
        if (reader := getattr(readers, kind)) is not None:
            observations[kind] = {}
            observed_readers[kind] = partial(_observe_plan_input, reader, observations[kind])
    memo = memo if memo is not None else SavedPlanMemo()
    reusable = readers.run is None and readers.row is None and readers.external is None
    with memo.lock:
        previous = memo.saved
        if (
            reusable and previous is not None and previous.stamp == stamp
            and _same_plan_inputs(readers, previous.observations)
        ):
            return replace(previous.plan, plan=previous.plan.plan.detached(), readers=readers)
        plan = compile_project(
            text.project, source_runs=source_runs, memo=memo,
            operations=operations, components=components, functions=functions, python_tasks=python_tasks,
            figure_templates=figure_templates, training_methods=training_methods,
            readers=replace(readers, **observed_readers),
        )
        settled = SettledProjectPlan(
            project_id, text.revision, semantic_root, text.configuration, store, plan, readers, text.root,
            detach_refusals(plan, text.project.documents),
        )
        if reusable:
            memo.saved = _SavedCompilation(stamp, settled, observations)
        return settled


def _may_read_figure_pin(project: Project) -> bool:
    figures = {
        name: compose_document(project.documents.get, name) for name, document in project.documents.items()
        if document.schema_id == FIGURE_DOCUMENT_SCHEMA
    }
    if any(
        not isinstance(document, ProjectDiagnostic) and is_figure_document_pin(document.payload)
        for document in figures.values()
    ):
        return True
    if (sweep := project.sweep) is None:
        return False

    def affects_figures(path: str) -> bool:
        family, *parts = path.split(".")
        return family == "documents" or bool(figures) and family == "steps" and len(parts) == 1

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


class _PendingDeclarationAddress(StrictModel):
    kind: Literal["declaration"]
    id: ProjectName


class _PendingFile(StrictModel):
    address: AuthoredFileAddress | _PendingDeclarationAddress
    before: Sha256 | None
    data: str | None
    encoding: Literal["utf8", "base64"] = "utf8"

    @property
    def relative(self) -> str:
        if isinstance(self.address, _PendingDeclarationAddress):
            return project_declaration_name(self.address.id)
        return _relative_file(self.address)

    def target(self) -> bytes | None:
        if self.data is None:
            return None
        return b64decode(self.data, validate=True) if self.encoding == "base64" else self.data.encode()


_PENDING_FILES = TypeAdapter(list[_PendingFile])


def _relative_file(address: AuthoredFileAddress | ReplayFileAddress) -> str:
    if address == PRESENTATION_ADDRESS:
        return STUDIO_PRESENTATION_FILENAME
    if address.kind not in _FAMILIES:
        raise ProjectFolderInvalid("Invalid project file address")
    return f"{_FAMILIES[address.kind]}/{address.id}.json"


@contextmanager
def _patch_parent(root_fd: int, relative: Path, *, create: bool = False) -> Iterator[int]:
    if relative.parent == Path("."):
        yield root_fd
        return
    family = relative.parent.name
    if create:
        try:
            os.mkdir(family, dir_fd=root_fd)
        except FileExistsError:
            pass
        else:
            os.fsync(root_fd)
    descriptor = open_existing_directory(
        family, dir_fd=root_fd, error_factory=ProjectFolderInvalid, context=family,
    )
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def _patch_preimage(relative: Path, directory_fd: int, *, exact_name: bool = False) -> tuple[int, str | None]:
    if exact_name and relative.name not in os.listdir(directory_fd):
        return 0o600, None
    try:
        descriptor, info = open_existing_file(
            relative.name, dir_fd=directory_fd, rigor=SecurePathRigor.REGULAR_FILE_IDENTITY,
            error_factory=ProjectFolderInvalid, context=relative.as_posix(),
        )
    except FileNotFoundError:
        return 0o600, None
    with os.fdopen(descriptor, "rb") as stream:
        return stat.S_IMODE(info.st_mode), sha256_bytes(stream.read())


def _finish_patch_at(root: Path, root_fd: int) -> None:
    try:
        descriptor, _ = open_existing_file(
            STUDIO_PATCH_JOURNAL_FILENAME, dir_fd=root_fd, rigor=SecurePathRigor.REGULAR_FILE_IDENTITY,
            error_factory=ProjectFolderInvalid, context="Studio edit journal",
        )
        with os.fdopen(descriptor, "rb") as stream:
            files = _PENDING_FILES.validate_json(stream.read())
        targets = [(file, Path(file.relative), file.target()) for file in files]
    except FileNotFoundError:
        return
    except ValueError as error:
        raise ProjectFolderInvalid("Studio edit journal is invalid") from error
    if not files or len({relative for _, relative, _ in targets}) != len(files):
        raise ProjectFolderInvalid("Studio edit journal is invalid")
    for file, relative, _ in targets:
        if isinstance(file.address, _PendingDeclarationAddress):
            try:
                require_text_declaration(root, relative.name)
            except ProjectReadError as error:
                raise ProjectFolderInvalid(str(error)) from error
    for file, relative, data in targets:
        target = sha256_bytes(data) if data is not None else None
        try:
            with _patch_parent(root_fd, relative) as directory_fd:
                _, current = _patch_preimage(
                    relative, directory_fd, exact_name=file.before is None or data is None,
                )
        except FileNotFoundError:
            current = None
        if current not in (target, file.before):
            raise ProjectFolderInvalid(f"{relative} changed during a Studio edit")

    # Clear old spellings before creating new ones on case-insensitive volumes.
    for file, relative, data in sorted(targets, key=lambda target: target[2] is not None):
        exact_name = file.before is None or data is None
        try:
            with _patch_parent(root_fd, relative, create=file.before is None and data is not None) as directory_fd:
                mode, current = _patch_preimage(relative, directory_fd, exact_name=exact_name)
                target = sha256_bytes(data) if data is not None else None
                if current == target:
                    os.fsync(directory_fd)
                    continue

                def guard() -> None:
                    if _patch_preimage(relative, directory_fd, exact_name=exact_name)[1] != file.before:
                        raise ProjectFolderInvalid(f"{relative} changed during a Studio edit")

                if data is None:
                    guard()
                    os.unlink(relative.name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
                else:
                    write_bytes_atomic(
                        root / relative, data, mode=mode, replace=file.before is not None,
                        before_publish=guard, directory_fd=directory_fd,
                    )
        except FileNotFoundError:
            if data is not None:
                raise

    os.fsync(root_fd)
    os.unlink(STUDIO_PATCH_JOURNAL_FILENAME, dir_fd=root_fd)
    os.fsync(root_fd)


def _publish_patch(folder: _LockedProjectFolder, files: list[_PendingFile], guard: Callable[[], None]) -> None:
    if not files:
        return
    file = files[0]
    if len(files) == 1 and file.before is not None and file.data is not None and file.address != PRESENTATION_ADDRESS:
        relative = Path(file.relative)
        with _patch_parent(folder.descriptor, relative) as directory_fd:
            mode, _ = _patch_preimage(relative, directory_fd)
        data = file.target()
        assert data is not None
        folder.write(folder.path / relative, data, mode=mode, before_publish=guard)
    else:
        # Readers recover the edit after its journal has been published.
        folder.write(
            folder.path / STUDIO_PATCH_JOURNAL_FILENAME, _PENDING_FILES.dump_json(files), replace=False,
            before_publish=guard,
        )
        _finish_patch_at(folder.path, folder.descriptor)


def _diagnostic_counts(diagnostics: Iterable[ProjectDiagnostic]) -> Counter:
    return Counter((item.code, item.step, item.variant, item.row, item.element) for item in diagnostics)


def _free_member_name(project: dict, family: MemberFamily, member: str) -> str:
    """The name of a copy: a document's stem numbered, a step's kind (`train-2`), whichever is free of members and references."""
    if family == "documents":
        stem = re.sub(r"-[0-9]+$", "", member)
        return generated_name(stem, lambda name: name == stem or name in project["documents"]
                              or references_member(project, "documents", name))
    return generated_name(project["steps"][member]["kind"], lambda name: name in project["steps"]
                          or references_member(project, "steps", name))


def _check_file_spellings(root: Path, files: Iterable[str]) -> None:
    lock = root / STUDIO_WRITE_LOCK_FILENAME
    try:
        alternate = lock.with_name(lock.name.upper()).lstat()
    except FileNotFoundError:
        return
    if not os.path.samestat(alternate, lock.lstat()):
        return
    folded = set()
    for name in files:
        normalized = name.casefold()
        if normalized in folded:
            raise ProjectFolderInvalid(f"{name} aliases another authored file on this volume")
        folded.add(normalized)


@dataclass(frozen=True)
class ProjectFolderLocation:
    path: Path
    identity: tuple[int, int]

    @property
    def project_id(self) -> str:
        locator = os.fsencode(self.path) + b"\0" + str(self.identity).encode("ascii")
        return f"external-{sha256_bytes(locator)}"

    @classmethod
    def selected(cls, path: Path) -> ProjectFolderLocation:
        with _project_directory_chain(path) as chain:
            opened, descriptor, info = chain[-1]
            if hasattr(fcntl, "F_GETPATH"):
                # APFS can open a differently cased spelling of the same folder.
                spelling = fcntl.fcntl(descriptor, fcntl.F_GETPATH, bytes(1024))
                opened = Path(os.fsdecode(spelling.split(b"\0", 1)[0]))
            return cls(opened, (info.st_dev, info.st_ino))


@contextmanager
def _project_directory_chain(path: Path) -> Iterator[list[DirectoryRecord]]:
    try:
        chain = open_directory_chain(
            path, create=False, error_factory=ProjectFolderInvalid, context="Project folder",
        )
    except FileNotFoundError as error:
        raise ProjectFolderNotFound("Project folder is missing") from error
    try:
        yield chain
    finally:
        close_directory_chain(chain)


@dataclass(frozen=True)
class _LockedProjectFolder:
    location: ProjectFolderLocation
    descriptor: int
    check: Callable[[], None]

    @property
    def path(self) -> Path:
        return self.location.path

    def write(
        self, path: Path, data: bytes, *, before_publish: Callable[[], None],
        mode: int = 0o600, replace: bool = True,
    ) -> None:
        def guard() -> None:
            self.check()
            before_publish()

        relative = path.relative_to(self.path)
        self.check()
        with _patch_parent(self.descriptor, relative, create=not replace) as directory:
            write_bytes_atomic(path, data, mode=mode, replace=replace, before_publish=guard, directory_fd=directory)


@contextmanager
def _folder_lock(
    location: ProjectFolderLocation, *, write: bool = True, declaration: str | None = None,
    verify_project: bool = False,
) -> Iterator[_LockedProjectFolder]:
    root = location.path
    with _project_directory_chain(root) as chain:
        _, directory, info = chain[-1]
        if (info.st_dev, info.st_ino) != location.identity:
            raise ProjectFolderNotFound("The selected project folder was replaced")

        def check_folder() -> None:
            recheck_directory_chain(chain, error_factory=ProjectFolderInvalid, context="Project folder")

        def check_declaration() -> None:
            if verify_project or declaration is not None:
                try:
                    read_text_project_declaration(root, directory, expected=declaration)
                except (OSError, ProjectReadError) as error:
                    raise ProjectFolderInvalid(str(error)) from error

        check_declaration()
        lock = STUDIO_WRITE_LOCK_FILENAME
        descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            check_folder()
            opened = os.fstat(descriptor)
            current = os.stat(lock, dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or (opened.st_dev, opened.st_ino) != (
                current.st_dev, current.st_ino,
            ):
                raise ProjectFolderInvalid("Project write lock changed")
            try:
                os.stat(STUDIO_PATCH_JOURNAL_FILENAME, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if not write:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    check_folder()
                check_declaration()
                _finish_patch_at(root, directory)
            yield _LockedProjectFolder(location, directory, check_folder)
            check_folder()
        finally:
            os.close(descriptor)


def check_revision(resource: _Resource, expected: str, current: str) -> None:
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


@dataclass(frozen=True)
class ModelEditChange:
    before_snapshot: StudioProjectSnapshot
    before: tuple[dict[str, JsonValue], dict[str, JsonValue]]
    after: tuple[dict[str, JsonValue], dict[str, JsonValue]]


@dataclass(frozen=True)
class AppliedModelEdit:
    """``change`` is the shown document's change when the edit wrote its file and layout alone."""

    snapshot: StudioProjectSnapshot
    changed: tuple[ModelAddress, ...]
    clipboard: ModelClipboard | None
    change: ModelEditChange | None
    created: tuple[str, ...] = ()


class ProjectFolder:
    """Read and edit named projects under a caller-supplied parent directory."""

    def __init__(
        self, root: Path, *, operations: OperationRegistry | None = None,
        components: ComponentRegistry | None = None,
        functions: FunctionRegistry | None = None,
        python_tasks: PythonTaskRegistry | None = None,
        figure_templates: FigureTemplateRegistry | None = None,
        local_folder: ProjectFolderLocation | None = None,
        training_methods: TrainingMethodRegistry | None = None,
    ) -> None:
        self.root = root.resolve()
        self._local_folder = local_folder
        self.operations = operations
        self.figure_templates = figure_templates
        self.components = components
        self.functions = functions
        self.python_tasks = python_tasks
        self.training_methods = training_methods
        self._plan_memo = lru_cache(maxsize=8)(lambda _project_id: SavedPlanMemo())
        self._known_locations: dict[str, ProjectFolderLocation] = {}
        self._known_locations_lock = Lock()

    def _folder(self, project_id: str) -> ProjectFolderLocation:
        if self._local_folder is not None:
            if project_id not in (self._local_folder.path.name, self._local_folder.project_id):
                raise ProjectFolderNotFound(project_id)
            return self._local_folder
        with self._known_locations_lock:
            location = self._known_locations.get(project_id)
        if location is not None:
            return location
        if _CANONICAL_PROJECT_ID.fullmatch(project_id):
            if self.root.is_dir():
                for child in self.root.iterdir():
                    if child.is_symlink() or not child.is_dir():
                        continue
                    try:
                        location = ProjectFolderLocation.selected(child)
                    except (ProjectFolderInvalid, ProjectFolderNotFound):
                        continue
                    if location.project_id == project_id:
                        self._remember_location(location)
                        return location
            raise ProjectFolderNotFound(project_id)
        try:
            _FOLDER_ID.validate_python(project_id)
        except ValidationError as error:
            raise ProjectFolderInvalid("Invalid project name") from error
        folder = self.root / project_id
        try:
            return ProjectFolderLocation.selected(folder)
        except ProjectFolderNotFound as error:
            raise ProjectFolderNotFound(project_id) from error

    def _remember_location(self, location: ProjectFolderLocation) -> None:
        with self._known_locations_lock:
            self._known_locations[location.project_id] = location

    def project_id(self, address: str) -> str:
        return self._folder(address).project_id

    def _locked_folder(self, project_id: str, *, write: bool = True) -> AbstractContextManager[_LockedProjectFolder]:
        return _folder_lock(self._folder(project_id), write=write)

    def fingerprint(self, project_id: str) -> tuple[TextFileStamp, ...]:
        with self._locked_folder(project_id, write=False) as folder:
            root = folder.path
            try:
                authored = text_project_fingerprint(root)
                try:
                    return (*authored, text_file_stamp(root / STUDIO_PRESENTATION_FILENAME, root))
                except FileNotFoundError:
                    return authored
            except (OSError, ProjectReadError) as error:
                raise ProjectFolderInvalid(str(error)) from error

    def _load(self, root: Path, *, declaration: str | None = None) -> TextProject:
        try:
            return read_text_project_snapshot(root, declaration=declaration)
        except ProjectReadError as error:
            raise ProjectFolderInvalid(str(error)) from error

    def _snapshot(self, location: ProjectFolderLocation, text: TextProject) -> StudioProjectSnapshot:
        data, presentation_revision = _read_presentation(text.root)
        presentation_error = None
        try:
            presentation = _decode_presentation(data)
        except (ValueError, UnicodeError):
            presentation = None
            presentation_error = "unreadable_presentation"
        documents = text.project.documents
        composed = {
            name: entry.payload for name, document in documents.items() if isinstance(document, BasedDocument)
            if isinstance(entry := compose_document(documents.get, name), RootDocument)
        }
        return StudioProjectSnapshot(
            project_id=location.project_id,
            composed=composed,
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
                    data=b64encode(text.files[relative]).decode("ascii"),
                )
                for relative, digest in text.file_revisions.items()
                if (path := Path(relative)).parent.name in _KINDS
            ],
            declaration=StudioAuthoredDeclaration(
                revision=text.file_revisions[text.declaration],
                authored_revision=authored_declaration_revision(text.manifest),
                data=b64encode(text.files[text.declaration]).decode("ascii"),
            ),
        )

    def _document(self, location: ProjectFolderLocation, text: TextProject) -> StudioProjectDocument:
        return StudioProjectDocument(
            configuration=text.configuration,
            snapshot=self._snapshot(location, text),
        )

    def open(self, project_id: str) -> StudioProjectDocument:
        return self._open_folder(self._folder(project_id))

    def represented_declaration(self, project_id: str, identity: tuple[int, int]) -> Path:
        location = self._folder(project_id)
        if location.identity != identity:
            raise ProjectFolderNotFound("The saved project folder was replaced")
        with _project_directory_chain(location.path) as chain:
            _, directory, info = chain[-1]
            if (info.st_dev, info.st_ino) != identity:
                raise ProjectFolderNotFound("The saved project folder was replaced")
            try:
                declaration = read_text_project_declaration(location.path, directory)
            except ProjectReadError as error:
                raise ProjectFolderInvalid(str(error)) from error
            recheck_directory_chain(chain, error_factory=ProjectFolderInvalid, context="Project folder")
            return declaration

    def _open_folder(
        self, location: ProjectFolderLocation, *, declaration: str | None = None,
    ) -> StudioProjectDocument:
        with _folder_lock(location, write=False, declaration=declaration, verify_project=True) as folder:
            document = self._document(location, self._load(folder.path, declaration=declaration))
        self._remember_location(location)
        return document

    def read(self, project_id: str, *, revision: str | None = None) -> StudioProjectSnapshot:
        with self._locked_folder(project_id, write=False) as folder:
            text = self._load(folder.path)
            if revision is not None:
                check_revision("project", revision, text.revision)
            return self._snapshot(folder.location, text)

    def read_plan(
        self, project_id: str, *, revision: str | None, source_runs: Iterable[RunKey] = (),
        declaration: str | None = None,
    ) -> SettledProjectPlan:
        """Compile the acknowledged folder snapshot and resolve its Store subjects."""
        location = self._folder(project_id)
        text = self._read_authored_snapshot(location, revision=revision, declaration=declaration)
        return self._compile(location, text, source_runs)

    def read_authored_snapshot(
        self, project_id: str, *, revision: str | None, declaration: str | None = None,
    ) -> TextProject:
        """Freeze the exact authored files under the project folder's read lock."""
        return self._read_authored_snapshot(self._folder(project_id), revision=revision, declaration=declaration)

    def _read_authored_snapshot(
        self, location: ProjectFolderLocation, *, revision: str | None, declaration: str | None,
    ) -> TextProject:
        with _folder_lock(location, write=False, declaration=declaration) as folder:
            text = self._load(folder.path, declaration=declaration)
        if revision is not None:
            check_revision("project", revision, text.revision)
        return text

    def _compile(
        self, location: ProjectFolderLocation, text: TextProject, source_runs: Iterable[RunKey] = ()
    ) -> SettledProjectPlan:
        project_id = location.project_id
        return compile_saved_project(
            text, project_id=project_id, store=text.store_path, readers=self._base_readers(location, text),
            source_runs=source_runs, memo=self._plan_memo(project_id),
            operations=self.operations, components=self.components, functions=self.functions,
            python_tasks=self.python_tasks, figure_templates=self.figure_templates, training_methods=self.training_methods,
        )

    def _base_readers(self, location: ProjectFolderLocation, text: TextProject) -> BaseReaders:
        if not _may_read_figure_pin(text.project):
            return BaseReaders()
        return BaseReaders(figure_external=TextProjectExternalInputs(text, authority=self.root).read)

    @contextmanager
    def admission_plan(self, project_id: str) -> Iterator[SettledProjectPlan]:
        """Hold the project folder's writer lock through comparison and Host admission."""
        with self._locked_folder(project_id) as folder:
            plan = self._compile(folder.location, self._load(folder.path))
            folder.check()
            yield plan

    def new(self) -> StudioProjectDocument:
        return self._new_folder()[0]

    def _new_folder(self) -> tuple[StudioProjectDocument, ProjectFolderLocation]:
        project_id = f"untitled-{uuid4().hex}"
        configuration = ProjectConfiguration(project=project_id, store="results")
        self.root.mkdir(parents=True, exist_ok=True)
        write_new_project_folder(self.root / project_id, Project(), configuration, PresentationDocument())
        location = self._folder(project_id)
        return self._open_folder(location), location

    def rename(self, project_id: str, body: StudioProjectRename) -> StudioProjectDocument:
        with self._locked_folder(project_id) as folder:
            root = folder.path
            text = self._load(root)
            check_revision("project", body.revision, text.revision)
            if body.name != text.configuration.project:
                try:
                    destination = project_declaration_name(body.name)
                    check_revision("project", text.revision, self._load(root).revision)
                    folder.check()
                    rename_no_replace(
                        text.declaration, destination, source_dir_fd=folder.descriptor,
                        destination_dir_fd=folder.descriptor, error_factory=ProjectFolderInvalid,
                        context="Rename project declaration",
                    )
                except FileExistsError as error:
                    raise ProjectFileExists(destination) from error
                os.fsync(folder.descriptor)
                folder.check()
                text = self._load(root)
            return self._document(folder.location, text)

    def create_file(self, project_id: str, body: StudioProjectCreateFile) -> StudioProjectDocument:
        file = body.file
        family = _FAMILIES[file.kind]
        relative = f"{family}/{file.id}.json"
        with self._locked_folder(project_id) as folder:
            root = folder.path
            text = self._load(root)
            check_revision("project", body.revision, text.revision)
            entries = text.project.documents if isinstance(file, CreateNamedDocument) else text.project.steps
            if file.id in entries:
                raise ProjectFileExists(relative)
            entry = file.document if isinstance(file, CreateNamedDocument) else file.step
            project = Project.model_validate({
                **text.project.model_dump(mode="json", by_alias=True, exclude_unset=True),
                family: {**entries, file.id: entry},
            })
            data = dump_text_project(project, text.configuration)[relative]
            def guard() -> None:
                check_revision("project", text.revision, self._load(root).revision)

            try:
                folder.write(root / relative, data, replace=False, before_publish=guard)
            except FileExistsError as error:
                raise ProjectFileExists(relative) from error
            return self._document(folder.location, self._load(root))

    def revision(self, project_id: str) -> StudioProjectRevision:
        with self._locked_folder(project_id, write=False) as folder:
            root = folder.path
            return StudioProjectRevision(
                project_id=project_id,
                revision=self._load(root).revision,
                presentation_revision=_read_presentation(root)[1],
            )

    def patch(self, project_id: str, body: StudioProjectPatch | StudioProjectPatchBatch) -> StudioProjectSnapshot:
        return self._patch(project_id, body, model_operation=False)

    def replay(self, project_id: str, body: StudioProjectReplay) -> StudioProjectSnapshot:
        """Apply a window's guarded replay against one acknowledged folder snapshot."""
        with self._locked_folder(project_id) as folder:
            root = folder.path
            text = self._load(root)
            check_revision("project", body.revision, text.revision)
            proposed = dict(text.files)
            pending = []
            seen = set()
            try:
                for file in body.files:
                    declaration = isinstance(file.address, ReplayDeclarationAddress)
                    relative = text.declaration if declaration else _relative_file(file.address)
                    if relative in seen:
                        raise ProjectFolderInvalid(f"{relative} is addressed twice")
                    seen.add(relative)
                    if text.file_revisions.get(relative) != file.before:
                        raise ProjectRevisionConflict("project", text.revision)
                    if file.after is None:
                        if declaration:
                            raise ProjectFolderInvalid("The project declaration is required")
                        data = None
                        proposed.pop(relative, None)
                    else:
                        data = b64decode(file.after, validate=True)
                        if declaration:
                            data = replay_text_project_declaration(text.manifest, data, declaration=relative)
                        proposed[relative] = data
                    if data != text.files.get(relative):
                        pending.append(_PendingFile(
                            address=_PendingDeclarationAddress(kind="declaration", id=text.configuration.project)
                            if declaration else file.address.model_dump(), before=file.before,
                            data=b64encode(data).decode("ascii") if data is not None else None,
                            encoding="base64",
                        ))
                presentation_data, presentation_revision = _read_presentation(root)
                if body.presentation is not None:
                    check_revision("presentation", body.presentation.before, presentation_revision)
                    data = _encode_presentation(body.presentation.after)
                    if data != presentation_data:
                        pending.append(_PendingFile(
                            address=PRESENTATION_ADDRESS,
                            before=sha256_bytes(presentation_data) if presentation_data is not None else None,
                            data=data.decode(),
                        ))
                if pending:
                    _check_file_spellings(root, proposed)
                    validated = parse_text_project_files(root, proposed)
                    # Validate the encodings the acknowledged snapshot will need before committing.
                    project_semantic_root(validated.project)
                    authored_declaration_revision(validated.manifest)
            except (ValueError, RecursionError) as error:
                if isinstance(error, ProjectFolderInvalid):
                    raise
                raise ProjectFolderInvalid(str(error)) from error

            def guard() -> None:
                check_revision("project", text.revision, self._load(root).revision)
                if body.presentation is not None:
                    check_revision("presentation", presentation_revision, _read_presentation(root)[1])

            if pending:
                _publish_patch(folder, pending, guard)
                text = self._load(root)
            return self._snapshot(folder.location, text)

    def member_command(self, project_id: str, body: StudioMemberCommand) -> StudioMemberCommandResult:
        """Rename, duplicate or detach a document, or rename or duplicate a step, with its references and presentation, in one write."""
        command = body.command
        with self._locked_folder(project_id) as folder:
            root = folder.path
            text = self._load(root)
            check_revision("project", body.revision, text.revision)
            presentation_data, presentation_revision = _read_presentation(root)
            check_revision("presentation", body.presentation_revision, presentation_revision)
            family: MemberFamily = "documents" if command.kind == "detach" else command.family
            member = command.member
            members = getattr(text.project, family)
            if member not in members:
                raise ProjectFolderNotFound(f"{family}/{member}.json")
            source = text.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
            shown_before = _decode_presentation(presentation_data).model_dump(mode="json", exclude_unset=True)
            shown = dict(shown_before)

            def composed(name: str) -> dict:
                document = compose_document(text.project.documents.get, name)
                if isinstance(document, ProjectDiagnostic):
                    raise ProjectFolderInvalid(f"{name} cannot be composed: {document.detail}")
                return document.model_dump(mode="json", by_alias=True)

            if command.kind == "detach":
                name = member
                if not isinstance(members[member], BasedDocument):
                    raise ProjectFolderInvalid(f"{member} has no base.")
                if member in self._compile(folder.location, text).detach_refused:
                    raise ProjectFolderInvalid(
                        f"Detaching {member} would leave an Evaluate step's Model document sharing no document with its subject's.",
                    )
                project = {**source, "documents": {**source["documents"], member: composed(member)}}
            else:
                name = command.name or _free_member_name(source, family, member)
                if name in members:
                    raise ProjectFileExists(f"{family}/{name}.json")
                if references_member(source, family, name):
                    raise ProjectFolderInvalid(f"A reference already names {name}.")
                if command.kind == "rename":
                    project, shown = rename_member(source, shown, family, member, name)
                elif family == "steps":
                    project, shown = duplicate_step(source, shown, member, name)
                else:
                    project = {**source, "documents": {**source["documents"], name: composed(member)}}
                    if member in shown.get("documents", {}):
                        shown["documents"] = {**shown["documents"], name: shown["documents"][member]}
            try:
                validated = Project.model_validate(project)
                presentation = PresentationDocument.model_validate(shown)
            except ValueError as error:
                raise ProjectFolderInvalid(str(error)) from error
            if command.kind == "rename":
                plan = compile_project(
                    validated, readers=self._base_readers(folder.location, text), operations=self.operations,
                    components=self.components, functions=self.functions, python_tasks=self.python_tasks,
                    figure_templates=self.figure_templates, training_methods=self.training_methods,
                )
                introduced = _diagnostic_counts(plan.diagnostics) - _diagnostic_counts(
                    self._compile(folder.location, text).plan.diagnostics,
                )
                if introduced:
                    raise ProjectFolderInvalid(
                        f"Renaming {member} leaves a reference this rename does not rewrite.",
                    )
            original = dump_text_project(text.project, text.configuration)
            encoded = dump_text_project(validated, text.configuration)
            copied, source_file = f"{family}/{name}.json", f"{family}/{member}.json"
            if encoded[copied] == original[source_file]:
                encoded[copied] = text.files[source_file]
            pending = []
            for relative in sorted(original.keys() | encoded.keys()):
                if original.get(relative) == encoded.get(relative):
                    continue
                path = Path(relative)
                data = encoded.get(relative)
                pending.append(_PendingFile(
                    address=_PendingDeclarationAddress(kind="declaration", id=text.configuration.project)
                    if relative == text.declaration
                    else AuthoredFileAddress(kind=_KINDS[path.parent.name], id=path.stem),
                    before=text.file_revisions.get(relative),
                    data=b64encode(data).decode("ascii") if data is not None else None,
                    encoding="base64",
                ))
            if shown != shown_before:
                pending.append(_PendingFile(
                    address=PRESENTATION_ADDRESS,
                    before=sha256_bytes(presentation_data) if presentation_data is not None else None,
                    data=_encode_presentation(presentation).decode(),
                ))
            _check_file_spellings(root, encoded)

            def guard() -> None:
                check_revision("project", text.revision, self._load(root).revision)
                check_revision("presentation", presentation_revision, _read_presentation(root)[1])

            _publish_patch(folder, pending, guard)
            return StudioMemberCommandResult(member=name, snapshot=self._snapshot(folder.location, self._load(root)))

    def apply_model_command(
        self, project_id: str, body: ModelEditRequest, components: ComponentRegistry, store: ModelStore,
    ) -> AppliedModelEdit | ModelEditProblem:
        """Apply a named Model operation to the shown document's composed content and commit what ``store`` makes of
        it with the layout, in one write.

        A document the edit creates gets the layout of its base, the edited layout when its base is the shown one.
        """
        from feedbax.contracts.model_edit import ModelEditProblem, ModelEditState, ModelProjectionRequest
        from feedbax.studio.model_edit import apply_model_edit
        from feedbax.studio.model_ports import ModelRegistryPorts
        from feedbax.project_model_selection import selected_task_signal

        snapshot = self.read(project_id, revision=body.revision)
        check_revision("presentation", body.presentation_revision, snapshot.presentation_revision)
        payload, layout = model_edit_source(snapshot, body.document)
        state = ModelEditState.model_validate({
            "document": payload, **layout,
        })
        authority = ModelRegistryPorts(components)
        result = apply_model_edit(
            state, body.command, ports=authority.port, accepts=authority.accepts,
            next_input=authority.next_input, functions=self.functions,
            task_signal=lambda command: selected_task_signal(self, project_id, ModelProjectionRequest(
                revision=body.revision, run=command.run, document=body.document,
            ), command.signal),
        )
        if isinstance(result, ModelEditProblem):
            return result
        if any(file.address.kind != "step" for file in body.files):
            raise ProjectFolderInvalid("A Model edit writes step files alongside its command.")
        change = None
        created: tuple[str, ...] = ()
        if result.state != state or body.files:
            next_payload = result.state.document.model_dump(mode="json") if result.state.document != state.document else payload
            targets = _document_sweep_paths(
                snapshot.project, body.document, lambda: self.read_plan(project_id, revision=body.revision).plan,
            ) if next_payload is not payload else ()
            if targets and _loses_model_target(payload, next_payload, targets, state, result.state, authority, self.functions):
                return ModelEditProblem(
                    address=result.changed[0], detail="Convert the affected sweep to a Value before moving or removing its field.",
                )
            next_layout = layout.copy()
            updated_layout = result.state.model_dump(mode="json", include=set(_MODEL_LAYOUT_FIELDS))
            for key in _MODEL_LAYOUT_FIELDS:
                if getattr(result.state, key) == getattr(state, key):
                    continue
                if updated_layout[key]:
                    next_layout[key] = updated_layout[key]
                else:
                    next_layout.pop(key, None)
            stored = store(snapshot.project, body, payload, next_payload) if next_payload is not payload or body.files \
                else StoredEdit([])
            assert snapshot.presentation is not None
            layouts: dict[str, dict[str, JsonValue]] = {}
            narrowed = False
            for document in stored.created:
                base = document.document.base.document if isinstance(document.document, BasedDocument) else None
                if base == body.document:
                    narrowed = True
                    layouts[document.id] = next_layout
                elif base is not None and (entry := snapshot.presentation.documents.get(base)) is not None:
                    shown = entry.model_dump(mode="json", exclude_unset=True)
                    layouts[document.id] = {key: shown[key] for key in _MODEL_LAYOUT_FIELDS if key in shown}
            if not narrowed:
                layouts[body.document] = next_layout
            shown_file = AuthoredFileAddress(kind="document", id=body.document)
            alone = not stored.created and all(file.address == shown_file for file in stored.files)
            created = tuple(document.id for document in stored.created)
            files = list(stored.files)
            if patches := _layout_patches(snapshot.presentation, layouts):
                files.append(StudioProjectFilePatch(address=PRESENTATION_ADDRESS, patches=patches))
            before = snapshot
            if files or stored.created:
                snapshot = self._patch(snapshot.project_id, StudioProjectPatchBatch(
                    revision=snapshot.revision, presentation_revision=snapshot.presentation_revision, files=files,
                ), model_operation=True, created=stored.created)
            if alone:
                change = ModelEditChange(before, (payload, layout), model_edit_source(snapshot, body.document))
        return AppliedModelEdit(snapshot, result.changed, result.clipboard, change, created)

    def _write_model_change(
        self, snapshot: StudioProjectSnapshot, document: ProjectName,
        payload: dict[str, JsonValue], layout: dict[str, JsonValue],
        *, had_entry: bool = True, had_documents: bool = True,
    ) -> StudioProjectSnapshot:
        """Give ``document`` the composed content ``payload`` and the Model ``layout``."""
        assert snapshot.presentation is not None
        entry = snapshot.project.documents[document]
        base = None
        if isinstance(entry, BasedDocument):
            base, _ = model_edit_source(snapshot, entry.base.document)
        files = [file] if (file := document_file(document, entry, payload, base)) is not None else []
        if patches := _layout_patches(
            snapshot.presentation, {document: layout}, had_entry=had_entry, had_documents=had_documents,
        ):
            files.append(StudioProjectFilePatch(address=PRESENTATION_ADDRESS, patches=patches))
        if not files:
            return snapshot
        return self._patch(snapshot.project_id, StudioProjectPatchBatch(
            revision=snapshot.revision, presentation_revision=snapshot.presentation_revision, files=files,
        ), model_operation=True)

    def _patch(
        self, project_id: str, body: StudioProjectPatch | StudioProjectPatchBatch, *, model_operation: bool,
        created: Sequence[CreateNamedDocument] = (),
    ) -> StudioProjectSnapshot:
        """Apply ``body``'s file patches, after creating the ``created`` documents, in one write."""
        with self._locked_folder(project_id) as folder:
            root = folder.path
            text = self._load(root)
            check_revision("project", body.revision, text.revision)
            files = body.files if isinstance(body, StudioProjectPatchBatch) else [body]
            original = dump_text_project(text.project, text.configuration)
            payload = text.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
            addresses = {}
            made = set()
            for document in created:
                address = AuthoredFileAddress(kind="document", id=document.id)
                relative = _relative_file(address)
                if document.id in payload.setdefault("documents", {}) or relative in original:
                    raise ProjectFileExists(relative)
                payload["documents"][document.id] = document.document.model_dump(mode="json", by_alias=True, exclude_unset=True)
                addresses[relative] = address
                made.add(relative)
            presentation_data = None
            edits_presentation = any(file.address == PRESENTATION_ADDRESS for file in files)
            if edits_presentation:
                if body.presentation_revision is None:
                    raise ProjectFolderInvalid("A presentation edit requires its revision")
                presentation_data, current = _read_presentation(root)
                check_revision("presentation", body.presentation_revision, current)
            try:
                presentation = (
                    _decode_presentation(presentation_data).model_dump(mode="json", exclude_unset=True)
                    if edits_presentation else None
                )
                for file in files:
                    relative = _relative_file(file.address)
                    if relative in addresses and relative not in made:
                        raise ProjectFolderInvalid(f"{relative} is addressed twice")
                    addresses[relative] = file.address
                    if relative == STUDIO_PRESENTATION_FILENAME:
                        assert presentation is not None
                        entry = presentation
                    else:
                        if relative not in original and relative not in made:
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
            changed = [relative for relative in addresses if encoded[relative] != original.get(relative)]

            def guard() -> None:
                check_revision("project", text.revision, self._load(root).revision)
                if edits_presentation:
                    check_revision("presentation", body.presentation_revision, _read_presentation(root)[1])

            if changed:
                _publish_patch(folder, [
                    _PendingFile(
                        address=addresses[relative],
                        before=(
                            sha256_bytes(presentation_data) if presentation_data is not None else None
                        ) if relative == STUDIO_PRESENTATION_FILENAME else text.file_revisions.get(relative),
                        data=encoded[relative].decode(),
                    )
                    for relative in changed
                ], guard)
                text = self._load(root)
            return self._snapshot(folder.location, text)

    def write_presentation(
        self, project_id: str, body: StudioPresentationWrite
    ) -> StudioProjectSnapshot:
        with self._locked_folder(project_id) as folder:
            root = folder.path
            text = self._load(root)
            data, revision = _read_presentation(root)
            check_revision("presentation", body.revision, revision)
            replacement = _encode_presentation(body.presentation)
            if replacement != data:
                path = root / STUDIO_PRESENTATION_FILENAME
                mode = stat.S_IMODE(_regular(path).st_mode) if data is not None else 0o600
                def guard() -> None:
                    check_revision("presentation", revision, _read_presentation(root)[1])

                folder.write(path, replacement, mode=mode, before_publish=guard)
                text = self._load(root)
            return self._snapshot(folder.location, text)
