"""Revision-guarded edits to a named text project and its presentation."""

from __future__ import annotations

from base64 import b64decode, b64encode
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import fcntl
from functools import lru_cache, partial
import os
from pathlib import Path
import re
from secrets import token_hex
from shutil import rmtree
import stat
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Literal, Protocol, TYPE_CHECKING

from pydantic import JsonValue, TypeAdapter, ValidationError

from feedbax._atomic import sync_parent_directories, write_bytes_atomic
from feedbax._secure_fs import (
    DirectoryRecord, SecurePathRigor, close_directory_chain, directory_open_flags, open_directory_chain,
    open_existing_directory, open_existing_file, recheck_directory_chain, rename_no_replace,
)
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.tasks.registry import PythonTaskRegistry
from feedbax.training.methods import TrainingMethodRegistry
from feedbax.synthesis.methods import SynthesisMethodRegistry
from feedbax.contracts.authored_arrays import AuthoredArrayBudgetError, check_authored_array_budget
from feedbax.contracts.base import StrictModel, sha256_bytes
from feedbax.contracts.figure_document import STORED_FIGURE_DOCUMENT_SCHEMA_ID, is_figure_document_pin
from feedbax.contracts.matrix_core import apply_value_patch_in_place
from feedbax.contracts.jobs import StudioProjectWindowMapProblem
from feedbax.contracts.loaded_model import LOADED_MODEL_SCHEMA_ID
from feedbax.contracts.artifact_custody import ArtifactBlobCustodyError
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.stored_model import (
    STORED_MODEL_DOCUMENT_SCHEMA_ID, StoredModelDocument, authored_model_payload,
)
from feedbax.contracts.project import (
    BasedDocument, Document, LoadedModelDocument, OrderProjectValue, Project, RootDocument, Sha256,
    apply_differences, apply_document_edit, diff, json_pointer_parts, member_name,
)
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project_storage import stored_keyed_collection, write_project_file
from feedbax.contracts.stored_regions import StoredRegionRegistries
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
    text_project_changes,
    text_project_fingerprint,
    text_project_settings_edit,
)
from feedbax.planner.project_plan import (
    ProjectPlan, ProjectPlanMemo, RunKey, _builtin_model_registry, compile_project, compiler_authority, project_variants,
)
from feedbax.contracts.project_resolution import BaseReaders, ProjectDiagnostic, ResolutionCode, compose_document
from feedbax.plugins.declarations import declaration_package
from feedbax.project_environment import PackageAvailability
from feedbax.contracts.project_sweep import GeneratedAxis, NamedAxis, Sweep, model_field_path, model_field_target
from feedbax.contracts.result_address import ANALYSIS_SCHEMA, analysis_elements, payload_elements
from feedbax.persistence.project_subject import terminal_train_reader
from feedbax.project_model_assets import (
    ModelAssetError, portable_model_reader, publish_project_model, write_model_assets,
)
from feedbax.persistence.reachability import DEFAULT_SUPERSEDED_DAYS
from feedbax.project_members import (
    change_member_key, duplicate_document_name, duplicate_step, fresh_element_keys, generated_name,
    allocate_member_name, label_member, merge_cards, new_document_name, new_step_name,
    preserve_dangling_member, preserve_loaded_archive_paths, regroup_refused, split_card,
)
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.examples import ShippedExample
from feedbax.studio.selection import StoredEdit, merged_files
from feedbax.contracts.studio_project import (
    DECLARATION_ADDRESS,
    PRESENTATION_ADDRESS,
    AuthoredFileAddress,
    CreateNamedDocument,
    DocumentPresentation,
    MemberFamily,
    NamedFileCreation,
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
    StudioMemberName,
    StudioMemberNameRequest,
    StudioNewDocumentName,
    StudioNewStepName,
    StudioPresentationWrite,
    StudioProjectDocument,
    StudioProjectList,
    StudioProjectListEntry,
    StudioProjectCreateFile,
    StudioProjectCreateFiles,
    StudioProjectPatch,
    StudioProjectPatchBatch,
    StudioProjectRename,
    StudioProjectReplay,
    StudioProjectMemberNames,
    StudioProjectRevision,
    StudioProjectSnapshot,
)

if TYPE_CHECKING:
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.analysis.operations import OperationRegistry
    from feedbax.plot.templates import FigureTemplateRegistry
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.model_edit import (
        ModelClipboard, ModelEditAsked, ModelEditProblem, ModelEditRequest, ModelGraphProblem,
    )


_FAMILIES = {"document": "documents", "step": "steps", "variant": "variants"}
_KINDS = {family: kind for kind, family in _FAMILIES.items()}
_FOLDER_ID = TypeAdapter(ProjectName)
_CANONICAL_PROJECT_ID = re.compile(r"external-[0-9a-f]{64}")
type _Resource = Literal["project", "presentation"]


_MODEL_LAYOUT_FIELDS = ("positions", "flipped", "collapsed", "waypoints")


def _runtime_model_layout(entry: DocumentPresentation | None) -> dict[str, JsonValue]:
    if entry is None:
        return {}
    stored = entry.model_dump(mode="json", exclude_unset=True)
    return {
        name: {ModelAddress.from_authored_text(key).to_text(): value for key, value in stored[name].items()}
        for name in _MODEL_LAYOUT_FIELDS if name in stored
    }


def _validate_model_presentation(presentation: PresentationDocument, project: Project) -> None:
    models = {name for name, document in project.documents.items() if document.schema_id == STORED_MODEL_DOCUMENT_SCHEMA_ID}

    def check(keys: Iterable[str], path: tuple[str, ...]) -> None:
        for key in keys:
            try:
                ModelAddress.from_authored_text(key)
            except ValueError as error:
                at = pointer((*path, key))
                raise ProjectFolderInvalid(f"{STUDIO_PRESENTATION_FILENAME} at {at}: {error}") from error

    for name in models & presentation.documents.keys():
        entry = presentation.documents[name]
        for section in _MODEL_LAYOUT_FIELDS:
            check(getattr(entry, section), ("documents", name, section))
    for name, entry in presentation.steps.items():
        for model, positions in entry.model_positions.items():
            check(positions, ("steps", name, "model_positions", model))


def _model_sweep_changes(
    project: Project, document: str, before, after, moves: dict[str, ModelAddress],
    *, before_payload: dict[str, JsonValue], after_payload: dict[str, JsonValue], functions, read_plan,
) -> tuple[list[StudioProjectFilePatch], list[StudioProjectWindowMapProblem]]:
    """Track moved fields and report removed targets on their authored sweep patches."""
    from feedbax.contracts.project import base_chain
    from feedbax.contracts.set_document import step_model

    if project.sweep is None and not any(entry.sweep is not None for entry in project.steps.values()):
        return [], []
    plan = read_plan()
    problems = []
    readers = {
        row.key.step for row in plan.rows if document in row.document_uses or document in row.followed_documents
    }
    readers.update(
        name for name in project.steps if (model := step_model(project, name)) is not None
        and document in base_chain(project.documents, model[0])
    )

    def target(path: str, at: tuple[str, ...], step: str | None) -> str:
        try:
            model = model_field_target(path)
        except ValueError:
            return path
        if model is not None:
            if (model.scope is None and step not in readers) or (model.step is not None and model.step not in readers):
                return path
            if model.document is not None and document not in base_chain(project.documents, model.document):
                return path
            old = model.address
            new = old.relocated(moves)
            if not before.has_field(old, functions=functions):
                return path
            if not after.has_field(new, functions=functions):
                mark_missing(old, at, step)
                return path
            if new == old:
                return path
            return model_field_path(new, model.scope)
        parts = path.split(".")
        if len(parts) < 4 or (parts[0], parts[2]) != ("documents", "payload"):
            return path
        if document not in base_chain(project.documents, parts[1]):
            return path
        old = ModelAddress.at_stored_path(parts[3:])
        new = old.relocated(moves)
        if _model_path_present(before_payload, parts[3:]) and not _model_path_present(after_payload, new.stored_path()):
            mark_missing(old, at, step)
            return path
        return ".".join((*parts[:3], *new.stored_path())) if new != old else path

    def mark_missing(address: ModelAddress, at: tuple[str, ...], step: str | None) -> None:
        path = pointer(at)
        problems.append(StudioProjectWindowMapProblem(
            code="invalid_sweep", path=path, detail=f"The Model field {address.label()} no longer exists.",
            step=step, variant=None, row=None, place=None, authored_path=path,
            address=f"steps/{step}.json" if step else None, document_path=pointer(address.stored_path()),
        ))

    def patches_at(item: dict, path: tuple[str, ...], step: str | None) -> list[SetProjectValue]:
        patches = item.get("patches", {})
        changed = {}
        for key, value in patches.items():
            at = (*path, "patches", key)
            moved = target(key, at, step)
            if moved != key and moved in patches:
                authored = pointer(at)
                problems.append(StudioProjectWindowMapProblem(
                    code="invalid_sweep", path=authored, detail="The moved field already has an authored sweep patch.",
                    step=step, variant=None, row=None, place=None, authored_path=authored,
                    address=f"steps/{step}.json" if step else None,
                ))
                moved = key
            changed[moved] = value
        if changed == patches:
            return []
        return [SetProjectValue(op="replace", path=pointer((*path, "patches")), value=changed)]

    def sweep_edits(sweep: Sweep, prefix: tuple[str, ...], step: str | None) -> list[SetProjectValue]:
        changes = []
        for name, axis in sweep.axes.items():
            path = (*prefix, "axes", name)
            if isinstance(axis, GeneratedAxis):
                if (moved := target(axis.path, (*path, "path"), step)) != axis.path:
                    changes.append(SetProjectValue(op="replace", path=pointer((*path, "path")), value=moved))
            elif isinstance(axis, NamedAxis):
                for value_id, value in axis.values.items():
                    if value is not None:
                        changes.extend(patches_at(value.model_dump(mode="json", exclude_unset=True),
                                                  (*path, "values", value_id), step))
        for kind, entries in (("rows", sweep.rows if step is not None else {}), ("exceptions", sweep.exceptions)):
            for name, entry in entries.items():
                if entry is not None:
                    changes.extend(patches_at(entry.model_dump(mode="json", exclude_unset=True), (*prefix, kind, name), step))
        return changes

    files = []
    for step, entry in project.steps.items():
        if entry.sweep is not None and (changes := sweep_edits(entry.sweep, ("sweep",), step)):
            files.append(StudioProjectFilePatch(address=AuthoredFileAddress(kind="step", id=step), patches=changes))
    if (sweep := project.sweep) is not None:
        if changes := sweep_edits(sweep, ("sweep",), None):
            files.append(StudioProjectFilePatch(address=DECLARATION_ADDRESS, patches=changes))
        for name, row in sweep.rows.items():
            if row is not None and (changes := patches_at(row.model_dump(mode="json", exclude_unset=True), (), None)):
                files.append(StudioProjectFilePatch(address=AuthoredFileAddress(kind="variant", id=name), patches=changes))
    return files, problems


def _model_path_present(value, path):
    for part in path:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdecimal() and int(part) < len(value):
            value = value[int(part)]
        else:
            return False
    return True


def model_edit_source(
    snapshot: StudioProjectSnapshot, document: ProjectName,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    source = compose_document(snapshot.project.documents.get, document)
    if isinstance(source, ProjectDiagnostic) and source.document == document and source.document_path is None:
        raise ProjectFolderNotFound(document)
    if isinstance(source, ProjectDiagnostic):
        raise ProjectFolderInvalid(source.detail)
    if source.schema_id != STORED_MODEL_DOCUMENT_SCHEMA_ID:
        raise ProjectFolderNotFound(document)
    if snapshot.presentation is None:
        raise ProjectFolderInvalid("The project presentation document is unreadable")
    assert isinstance(source.payload, dict)
    return source.payload, _runtime_model_layout(snapshot.presentation.documents.get(document))



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
    original_schema = entry.get("schema")
    order_only = all(isinstance(patch, OrderProjectValue) for patch in file.patches)

    def check_model() -> None:
        if file.address.kind == "document" and not model_operation and entry.get("schema") == STORED_MODEL_DOCUMENT_SCHEMA_ID:
            raise ProjectFolderInvalid("Model documents are edited through addressed operations")

    if not order_only:
        check_model()
    for patch in file.patches:
        path = json_pointer_parts(patch.path)
        if isinstance(patch, OrderProjectValue):
            if (
                file.address.kind != "document" or path[:1] != ("payload",)
                or not isinstance(original_schema, str) or entry.get("schema") != original_schema
            ):
                raise ProjectFolderInvalid("An Order edit must name a keyed collection in a document payload")
            apply_document_edit(
                entry, patch, None, keyed_collection=lambda parts: stored_keyed_collection(original_schema, parts[1:]),
            )
            continue
        if file.address == DECLARATION_ADDRESS and not model_operation:
            if path[0] != "settings":
                raise ProjectFolderInvalid("Only project settings can be edited through the declaration address")
            if path == ("settings",) and isinstance(patch, RemoveProjectValue):
                raise ProjectFolderInvalid("Replace project settings with an empty object to clear them")
        apply_value_patch_in_place(
            entry, path, patch.op,
            patch.value if isinstance(patch, SetProjectValue) else None, path=patch.path,
        )
    if not order_only:
        check_model()


def _project_file_entry(payload: dict, address: AuthoredFileAddress) -> dict:
    if address == DECLARATION_ADDRESS:
        return payload
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
    keyed = partial(stored_keyed_collection, document.schema_id)
    if isinstance(document, RootDocument):
        if isinstance(document.payload, dict) and not diff(document.payload, content, keyed_collection=keyed):
            return None
        return StudioProjectFilePatch(address=address, patches=[SetProjectValue(op="replace", path="/payload", value=content)])
    assert base is not None
    differences = diff(base, content, keyed_collection=keyed)
    if differences == document.differences:
        previous = apply_differences(base, document.differences, keyed_collection=keyed)
        if not diff(previous, content, keyed_collection=keyed):
            return None
    return StudioProjectFilePatch(address=address, patches=[SetProjectValue(
        op="replace" if "differences" in document.model_fields_set else "add", path="/differences",
        value={path: value.model_dump(mode="json") for path, value in differences.items()},
    )])


type ModelStore = Callable[
    [Project, ModelEditRequest, dict[str, JsonValue], dict[str, JsonValue]], StoredEdit | ModelEditAsked,
]
"""Store a Model edit's change of the shown document's composed content, from the first payload to the second, after
the request's step files: the documents it creates and the patches of files."""


def _layout_patches(
    presentation: PresentationDocument, layouts: dict[str, dict[str, JsonValue]],
    *, had_entry: bool = True, had_documents: bool = True,
) -> list[SetProjectValue | RemoveProjectValue]:
    """The presentation patches that give each document its Model layout, keeping its other presentation."""

    patches: list[SetProjectValue | RemoveProjectValue] = []
    has_documents = "documents" in presentation.model_fields_set
    for document, layout in layouts.items():
        entry = presentation.documents.get(document)
        current = entry.model_dump(mode="json", exclude_unset=True) if entry is not None else {}
        authored = {
            name: {ModelAddress.from_text(key).to_authored_text(): value for key, value in entries.items()}
            for name, entries in layout.items()
        }
        next_entry = {key: value for key, value in current.items() if key not in _MODEL_LAYOUT_FIELDS} | authored
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
    regroup_refused: frozenset[str]

    @classmethod
    def from_text(
        cls, location: ProjectFolderLocation, text: TextProject, plan: ProjectPlan, *,
        readers: BaseReaders = BaseReaders(),
        store: Path | None = None,
    ) -> SettledProjectPlan:
        return cls(
            location.project_id, text.revision, project_semantic_root(text.project),
            text.configuration, text.store_path if store is None else store, plan, readers, text.root,
            regroup_refused(text.project),
        )


class SavedProjectPlans(Protocol):
    def read_plan(self, project_id: str, *, revision: str | None) -> SettledProjectPlan: ...

    def admission_plan(self, project_id: str) -> AbstractContextManager[SettledProjectPlan]: ...


@dataclass(frozen=True)
class _SavedCompilation:
    plan: SettledProjectPlan
    observations: dict[str, dict[object, object]]


_SAVED_PLAN_CACHE_SIZE = 4


class SavedPlanMemo(ProjectPlanMemo):
    """Retain recent reviewed compilations and their observed inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.lock = Lock()
        self.saved: OrderedDict[tuple[object, ...], _SavedCompilation] = OrderedDict()


def _observe_plan_input[Key, T](reader: Callable[[Key], T], observations: dict[Key, object], name: Key) -> T:
    try:
        value = reader(name)
    except (OSError, ValueError) as error:
        observations[name] = (type(error), str(error))
        raise
    observations[name] = sha256_bytes(value) if isinstance(value, bytes) else value
    return value


def _same_plan_inputs(readers: BaseReaders, expected: dict[str, dict[object, object]]) -> bool:
    for kind, observed in expected.items():
        reader = getattr(readers, kind)
        if reader is None:
            return False
        current: dict[object, object] = {}
        for name in observed:
            try:
                _observe_plan_input(reader, current, name)
            except (OSError, ValueError):
                pass  # The failed read is also an observation.
        if observed != current:
            return False
    return True


def package_problems(plan: ProjectPlan, packages: PackageAvailability, folder: Path) -> ProjectPlan:
    """Report each unavailable package once, as a project Problem that the rows needing it point at."""
    reasons: dict[str, str] = {}
    diagnostics = []
    for diagnostic in plan.diagnostics:
        package = declaration_package(diagnostic.missing_key) if diagnostic.missing_key is not None else None
        if package is not None and (reason := packages.unavailable(package, folder)) is not None:
            reasons[package] = reason
            diagnostic = replace(diagnostic, code=ResolutionCode.MISSING_PACKAGE, detail=reason, package=package)
        diagnostics.append(diagnostic)
    if not reasons:
        return plan
    return replace(plan, diagnostics=(
        *(ProjectDiagnostic(ResolutionCode.PACKAGE_UNAVAILABLE, "", reason, package=package)
          for package, reason in reasons.items()),
        *diagnostics,
    ))


def compile_saved_project(
    text: TextProject, *, project_id: str, store: Path, readers: BaseReaders,
    operations: OperationRegistry | None = None, components: ComponentRegistry | None = None,
    functions: FunctionRegistry | None = None, python_tasks: PythonTaskRegistry | None = None,
    figure_templates: FigureTemplateRegistry | None = None,
    source_runs: Iterable[RunKey] = (), memo: SavedPlanMemo | None = None,
    training_methods: TrainingMethodRegistry | None = None,
    synthesis_methods: SynthesisMethodRegistry | None = None,
    packages: PackageAvailability | None = None,
) -> SettledProjectPlan:
    """Compile exact authored bytes and reuse them only with matching input authority.

    With ``packages``, each declaration of an unavailable package is reported once for the project.
    """
    readers = replace(
        readers, terminal_train=terminal_train_reader(store),
        imported_model=portable_model_reader(text.root, store, components or _builtin_model_registry()),
    )
    semantic_root = project_semantic_root(text.project)
    source_runs = frozenset(source_runs)
    stamp = (
        project_id, text.revision, semantic_root, text.root, store, source_runs,
        compiler_authority(components, operations, functions, python_tasks, training_methods, figure_templates, synthesis_methods),
        readers.external is not None, readers.figure_external is not None,
    )
    observations: dict[str, dict[object, object]] = {}
    observed_readers = {}
    for kind in ("external", "terminal_train", "imported_model", "figure_external"):
        if (reader := getattr(readers, kind)) is not None:
            observations[kind] = {}
            observed_readers[kind] = partial(_observe_plan_input, reader, observations[kind])
    memo = memo if memo is not None else SavedPlanMemo()
    reusable = readers.run is None and readers.row is None
    with memo.lock:
        previous = memo.saved.get(stamp) if reusable else None
        if previous is not None and _same_plan_inputs(readers, previous.observations):
            memo.saved.move_to_end(stamp)
            settled = replace(previous.plan, plan=previous.plan.plan.detached(), readers=readers)
        else:
            plan = compile_project(
                text.project, source_runs=source_runs, memo=memo,
                operations=operations, components=components, functions=functions, python_tasks=python_tasks,
                figure_templates=figure_templates, training_methods=training_methods, synthesis_methods=synthesis_methods,
                readers=replace(readers, **observed_readers),
            )
            settled = SettledProjectPlan(
                project_id, text.revision, semantic_root, text.configuration, store, plan, readers, text.root,
                regroup_refused(text.project),
            )
            if reusable:
                memo.saved[stamp] = _SavedCompilation(settled, observations)
                memo.saved.move_to_end(stamp)
                if len(memo.saved) > _SAVED_PLAN_CACHE_SIZE:
                    memo.saved.popitem(last=False)
    if packages is not None:
        settled = replace(settled, plan=package_problems(settled.plan, packages, text.root))
    return settled


def _may_read_figure_pin(project: Project) -> bool:
    figures = {
        name: compose_document(project.documents.get, name) for name, document in project.documents.items()
        if document.schema_id == STORED_FIGURE_DOCUMENT_SCHEMA_ID
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


class _PendingFile(StrictModel):
    address: AuthoredFileAddress
    before: Sha256 | None
    data: str | None
    encoding: Literal["utf8", "base64"] = "utf8"

    @property
    def relative(self) -> str:
        if self.address.kind == "declaration":
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
        if file.address.kind == "declaration":
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


def _free_member_name(project: dict, family: MemberFamily, member: str) -> str:
    """The name of a copy: a document's stem numbered, or a step's kind, whichever key is unused."""
    if family == "documents":
        stem = re.sub(r"-[0-9]+$", "", member)
        return generated_name(stem, lambda name: name == stem or name in project["documents"])
    return generated_name(project["steps"][member]["kind"], project["steps"].__contains__)


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
    verify_project: bool = False, detach: Callable[[int], None] | None = None,
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
            if detach is not None:
                detach(chain[-2][1])
        finally:
            os.close(descriptor)


def _delete_project_folder(location: ProjectFolderLocation, parent: int) -> None:
    """Detach the verified directory before removing any of its contents."""
    staging = f".studio-delete-{token_hex(16)}"
    os.mkdir(staging, mode=0o700, dir_fd=parent)
    empty = True
    try:
        staged = os.open(staging, directory_open_flags(error_factory=OSError), dir_fd=parent)
        try:
            rename_no_replace(
                location.path.name, "project", source_dir_fd=parent, destination_dir_fd=staged,
                error_factory=OSError, context="Studio project deletion",
            )
            empty = False
            moved = os.stat("project", dir_fd=staged, follow_symlinks=False)
            if not stat.S_ISDIR(moved.st_mode) or (moved.st_dev, moved.st_ino) != location.identity:
                rename_no_replace(
                    "project", location.path.name, source_dir_fd=staged, destination_dir_fd=parent,
                    error_factory=OSError, context="restore replaced Studio project folder",
                )
                empty = True
                raise ProjectFolderNotFound("The selected project folder was replaced")
            rmtree("project", dir_fd=staged)
            empty = True
        finally:
            os.close(staged)
    finally:
        # A failed restore or removal leaves recoverable files inside staging.
        if empty:
            os.rmdir(staging, dir_fd=parent)
    os.fsync(parent)


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
    return write_project_file(presentation, source=STUDIO_PRESENTATION_FILENAME)


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
    *, registries: StoredRegionRegistries | None = None,
) -> None:
    """Publish the ordinary authored folder and presentation in one exclusive rename."""
    root = destination.parent.resolve(strict=True)
    files = dump_text_project(project, configuration, registries=registries)
    _validate_model_presentation(presentation, project)
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
    problems: tuple[ModelGraphProblem, ...] = ()
    sweep_problems: tuple[StudioProjectWindowMapProblem, ...] = ()


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
        synthesis_methods: SynthesisMethodRegistry | None = None,
        load_examples: Callable[[], Mapping[str, ShippedExample]] = dict,
        packages: PackageAvailability | None = None,
    ) -> None:
        self.root = root.resolve()
        self.packages = packages
        self._local_folder = local_folder
        self.operations = operations
        self.figure_templates = figure_templates
        self.components = components
        self.functions = functions
        self.python_tasks = python_tasks
        self.training_methods = training_methods
        self.synthesis_methods = synthesis_methods
        self._stored_regions = StoredRegionRegistries(
            training_methods=training_methods, synthesis_methods=synthesis_methods,
            operations=operations, python_tasks=python_tasks, components=components,
        )
        self._load_examples = load_examples
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
            _validate_model_presentation(presentation, text.project)
        except (ValueError, UnicodeError):
            presentation = None
            presentation_error = "unreadable_presentation"
        documents = text.project.documents
        composed = {}
        problems = []
        for name, document in documents.items():
            if isinstance(document, BasedDocument):
                entry = compose_document(documents.get, name)
                if isinstance(entry, ProjectDiagnostic):
                    problems.append(StudioProjectWindowMapProblem.from_diagnostic(
                        replace(entry, authored_file=f"documents/{entry.document or name}.json"),
                        declaration=text.declaration,
                    ))
                else:
                    composed[name] = entry.payload
        names = StudioProjectMemberNames.from_project(text.project)
        variants = project_variants(
            text.project, components=self.components, functions=self.functions, python_tasks=self.python_tasks,
            training_methods=self.training_methods,
        )
        return StudioProjectSnapshot(
            project_id=location.project_id,
            variant_names={
                key: names if project is text.project else StudioProjectMemberNames.from_project(project)
                for key, project in variants.items()
            },
            composed=composed,
            problems=problems,
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
            settings=text.configuration.settings,
            default_keep_superseded_days=DEFAULT_SUPERSEDED_DAYS,
        )

    def _document(self, location: ProjectFolderLocation, text: TextProject) -> StudioProjectDocument:
        return StudioProjectDocument(
            configuration=text.configuration,
            snapshot=self._snapshot(location, text),
        )

    def open(self, project_id: str) -> StudioProjectDocument:
        return self._open_folder(self._folder(project_id))

    def list_projects(self) -> StudioProjectList:
        """List valid home projects without opening them or recovering pending writes."""
        try:
            children = tuple(self.root.iterdir())
        except FileNotFoundError:
            children = ()
        except OSError as error:
            raise ProjectFolderInvalid(str(error)) from error
        projects = []
        for child in children:
            try:
                if not child.is_dir(follow_symlinks=False):
                    continue
                location = ProjectFolderLocation.selected(child)
                text = self._load(location.path)
                stamps = (text_file_stamp(location.path / name, location.path) for name in text.files)
                modified_ns = max(mtime_ns for _, _, _, _, mtime_ns, _ in stamps)
                if ProjectFolderLocation.selected(child) != location:
                    continue
            except (OSError, ProjectFolderInvalid, ProjectReadError):
                continue
            projects.append(StudioProjectListEntry(
                project_id=location.project_id, name=text.configuration.project,
                folder=str(location.path), step_count=len(text.project.steps),
                modified=datetime.fromtimestamp(modified_ns / 1_000_000_000, tz=timezone.utc),
            ))
        return StudioProjectList(projects=sorted(projects, key=lambda project: (project.name, project.project_id)))

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
            synthesis_methods=self.synthesis_methods, packages=self.packages,
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

    def project_examples(self) -> Mapping[str, ShippedExample]:
        return self._load_examples()

    def new(self, name: str, *, example: str | None = None) -> StudioProjectDocument:
        if re.fullmatch(r"untitled-[0-9a-f]+", name, re.IGNORECASE):
            raise ProjectFolderInvalid("Choose a name for the project.")
        configuration = ProjectConfiguration(project=name, store="results")
        project = Project()
        if example is not None:
            shipped = self.project_examples().get(example)
            if shipped is None:
                raise ProjectFolderNotFound(example)
            project = shipped.project
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / name
        try:
            write_new_project_folder(
                destination, project, configuration, PresentationDocument(), registries=self._stored_regions,
            )
        except FileExistsError as error:
            raise ProjectFileExists(name) from error
        except OSError as error:
            raise ProjectFolderInvalid(str(error)) from error
        return self._open_folder(ProjectFolderLocation.selected(destination))

    def delete(self, project_id: str) -> None:
        if not _CANONICAL_PROJECT_ID.fullmatch(project_id):
            raise ProjectFolderNotFound(project_id)
        location = self._folder(project_id)
        if location.path.parent != self.root:
            raise ProjectFolderNotFound(project_id)
        with _folder_lock(location, verify_project=True, detach=partial(_delete_project_folder, location)) as folder:
            self._load(folder.path)
        with self._known_locations_lock:
            self._known_locations.pop(project_id, None)

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

    def create_file(
        self, project_id: str, body: StudioProjectCreateFile | StudioProjectCreateFiles,
    ) -> StudioProjectDocument:
        files = [body.file] if isinstance(body, StudioProjectCreateFile) else body.files
        with self._locked_folder(project_id) as folder:
            text = self._load(folder.path)
            check_revision("project", body.revision, text.revision)
            return self._create_files(folder, text, files)

    def _create_files(
        self, folder: _LockedProjectFolder, text: TextProject, files: Sequence[NamedFileCreation],
        *, before_create: Callable[[], None] | None = None,
    ) -> StudioProjectDocument:
        root = folder.path
        documents = dict(text.project.documents)
        steps = dict(text.project.steps)
        additions = {}
        for file in files:
            address = AuthoredFileAddress(kind=file.kind, id=file.id)
            relative = _relative_file(address)
            if relative in text.files or relative in additions:
                raise ProjectFileExists(relative)
            additions[relative] = address
            if isinstance(file, CreateNamedDocument):
                documents[file.id] = file.document
            else:
                steps[file.id] = file.step
        project = Project(documents=documents, steps=steps, sweep=text.project.sweep)
        try:
            encoded = dump_text_project(project, text.configuration, registries=self._stored_regions, only=additions)
        except ValueError as error:
            raise ProjectFolderInvalid(str(error)) from error
        _check_file_spellings(root, text.files.keys() | encoded.keys())

        def guard() -> None:
            check_revision("project", text.revision, self._load(root).revision)

        if before_create is not None:
            before_create()
        _publish_patch(folder, [
            _PendingFile(address=address, before=None, data=encoded[relative].decode())
            for relative, address in additions.items()
        ], guard)
        return self._document(folder.location, self._load(root))

    def publish_loaded_model(
        self, project_id: str, *, revision: str, name: str, model: LoadedModelDocument,
        staging: ImmutableArtifactBlobProvider, raw_archive: bytes, label: str | None = None,
    ) -> StudioProjectSnapshot:
        """Commit one checked Load after its exact project assets and Store root are durable."""
        with self._locked_folder(project_id) as folder:
            text = self._load(folder.path)
            check_revision("project", revision, text.revision)
            existing = name in text.project.documents
            if existing:
                document = compose_document(text.project.documents.get, name)
                if isinstance(document, ProjectDiagnostic) or document.schema_id != LOADED_MODEL_SCHEMA_ID:
                    raise ProjectFileExists(f"documents/{name}.json")
                saved = LoadedModelDocument.model_validate(document.payload)
                if saved.lower() != model.lower() or saved.model != model.model:
                    raise ProjectFileExists(f"documents/{name}.json")
                model = saved
            archive = model.archive_path(name)
            source = model.lower()

            def check() -> None:
                folder.check()
                check_revision("project", revision, self._load(folder.path).revision)

            def publish() -> None:
                try:
                    write_model_assets(
                        folder.path, folder.location.identity, archive, raw_archive, source,
                        staging.get_object(source.model), check=check,
                    )
                    publish_project_model(text.store_path, staging, source)
                except FileExistsError as error:
                    raise ProjectFileExists(*error.args) from error
                except (ModelAssetError, ArtifactBlobCustodyError) as error:
                    raise ProjectFolderInvalid(str(error)) from error

            if existing:
                publish()
                check()
                return self._snapshot(folder.location, text)
            file = CreateNamedDocument(kind="document", id=name, document=RootDocument(
                schema=LOADED_MODEL_SCHEMA_ID, payload=model.model_dump(mode="json"),
                label=label or new_document_name(text.project.documents, LOADED_MODEL_SCHEMA_ID).label,
            ))
            return self._create_files(folder, text, [file], before_create=publish).snapshot

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
                            address=AuthoredFileAddress(kind="declaration", id=text.configuration.project)
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
                    if body.presentation is not None:
                        _validate_model_presentation(body.presentation.after, validated.project)
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

    def member_name(self, project_id: str, body: StudioMemberNameRequest) -> StudioMemberName:
        """Allocate an authored name against the same revision as its eventual write."""
        project = self.read_authored_snapshot(project_id, revision=body.revision).project
        member = body.member
        if isinstance(member, StudioNewDocumentName):
            return new_document_name(project.documents, member.schema_id, member.label)
        if isinstance(member, StudioNewStepName):
            return new_step_name(project.steps, member.step)
        if member.document not in project.documents:
            raise ProjectFolderNotFound(f"documents/{member.document}.json")
        entry = compose_document(project.documents.get, member.document)
        if isinstance(entry, ProjectDiagnostic):
            raise ProjectFolderInvalid(entry.detail)
        if entry.schema_id != ANALYSIS_SCHEMA or not isinstance(entry.payload, dict):
            raise ProjectFolderInvalid("Choose an Analysis document.")
        keys = {"states", *entry.payload.get("inputs", {})}
        labels = set()
        for name, document in project.documents.items():
            if document.schema_id != ANALYSIS_SCHEMA:
                continue
            composed = compose_document(project.documents.get, name)
            if isinstance(composed, ProjectDiagnostic):
                continue
            elements = analysis_elements(composed.payload)
            keys.update(elements)
            labels.update(element["label"] for element in elements.values()
                          if isinstance(element, dict) and isinstance(element.get("label"), str))
        fallback = "operation" if member.family == "operations" else "figure"
        return allocate_member_name(member.label, labels, keys.__contains__, fallback)

    def member_command(self, project_id: str, body: StudioMemberCommand) -> StudioMemberCommandResult:
        """Label a member, change its key, duplicate or detach it, or split or merge
        Analyze cards, in one write.

        A key change, a split and a merge rewrite the references and presentation they move. The saved result
        carries any remaining authored Problems.
        """
        command = body.command
        with self._locked_folder(project_id) as folder:
            root = folder.path
            text = self._load(root)
            check_revision("project", body.revision, text.revision)
            presentation_data, presentation_revision = _read_presentation(root)
            check_revision("presentation", body.presentation_revision, presentation_revision)
            member = command.member
            source = text.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
            shown_before = _decode_presentation(presentation_data).model_dump(mode="json", exclude_unset=True)
            shown = dict(shown_before)

            def composed(name: str) -> dict:
                """The document ``name`` as a root document holding its composed payload, under its own label."""
                document = compose_document(text.project.documents.get, name)
                if isinstance(document, ProjectDiagnostic):
                    raise ProjectFolderInvalid(f"{member_name(text.project.documents, name)} cannot be composed: {document.detail}")
                return document.model_dump(mode="json", by_alias=True)

            def element_keys() -> set[str]:
                return {
                    key for name, document in text.project.documents.items() if document.schema_id == ANALYSIS_SCHEMA
                    if isinstance(entry := compose_document(text.project.documents.get, name), RootDocument)
                    for key in payload_elements(entry.payload)
                }

            if command.kind in ("split", "merge"):
                for card in (member, *(command.others if command.kind == "merge" else ())):
                    if card not in text.project.steps:
                        raise ProjectFolderNotFound(f"steps/{card}.json")
            elif command.kind in ("rename", "change_key") and command.family == "elements":
                document = command.document
                assert document is not None
                if document not in text.project.documents:
                    raise ProjectFolderNotFound(f"documents/{document}.json")
                if member not in payload_elements(composed(document)["payload"]):
                    raise ProjectFolderNotFound(f"documents/{document}.json#{member}")
            elif command.kind == "detach" or command.family == "documents":
                if member not in text.project.documents:
                    raise ProjectFolderNotFound(f"documents/{member}.json")
            elif member not in text.project.steps:
                raise ProjectFolderNotFound(f"steps/{member}.json")

            if command.kind == "split":
                card = new_step_name(text.project.steps, "analyze", text.project.steps[member].label)
                name = card.id
                original = source["steps"][member]["base"].get("document")
                document = new_document_name(
                    text.project.documents, ANALYSIS_SCHEMA,
                    text.project.documents[original].label if original in text.project.documents else None, copy=True,
                )
                try:
                    source = preserve_dangling_member(source, "steps", name)
                    source = preserve_dangling_member(source, "documents", document.id)
                    project, shown = split_card(source, shown, member, set(command.elements), document, card)
                except ValueError as error:
                    raise ProjectFolderInvalid(str(error)) from error
            elif command.kind == "merge":
                name = member
                try:
                    project, shown = merge_cards(source, shown, member, command.others)
                except ValueError as error:
                    raise ProjectFolderInvalid(str(error)) from error
            elif command.kind == "detach":
                name = member
                if not isinstance(text.project.documents[member], BasedDocument):
                    raise ProjectFolderInvalid(f"{member_name(text.project.documents, member)} has no base.")
                project = {**source, "documents": {**source["documents"], member: composed(member)}}
            elif command.kind == "rename":
                name = member
                project = label_member(source, command.family, member, command.label, document=command.document)
            elif command.kind == "change_key" and command.family == "elements":
                name = command.key
                if name in element_keys():
                    raise ProjectFolderInvalid(f"An operation or figure already holds the key {name}.")
                project, shown = change_member_key(
                    source, shown, "elements", member, name, document=command.document,
                )
            else:
                family = command.family
                members = getattr(text.project, family)
                copy_label = None
                if command.kind == "change_key":
                    name = command.key
                elif family == "documents":
                    name, copy_label = duplicate_document_name(text.project.documents, member, command.name)
                else:
                    step = text.project.steps[member]
                    copy = new_step_name(text.project.steps, step.kind, step.label)
                    name, copy_label = command.name or copy.id, copy.label
                if name in members:
                    raise ProjectFileExists(f"{family}/{name}.json")
                if command.kind == "change_key":
                    project, shown = change_member_key(source, shown, family, member, name)
                    if family == "documents" and members[member].schema_id == LOADED_MODEL_SCHEMA_ID:
                        preserve_loaded_archive_paths(text.project.documents, project["documents"], member, name)
                else:
                    source = preserve_dangling_member(source, family, name)
                    if family == "steps":
                        assert copy_label is not None
                        project, shown = duplicate_step(source, shown, member, name, copy_label)
                    else:
                        copy = composed(member)
                        copy["label"] = copy_label
                        if copy["schema"] == ANALYSIS_SCHEMA:
                            copy["payload"] = fresh_element_keys(copy["payload"], element_keys())
                        elif copy["schema"] == LOADED_MODEL_SCHEMA_ID:
                            loaded = LoadedModelDocument.model_validate(copy["payload"])
                            copy["payload"]["archive"] = loaded.archive_path(member)
                        project = {**source, "documents": {**source["documents"], name: copy}}
                        if member in shown.get("documents", {}):
                            shown["documents"] = {**shown["documents"], name: shown["documents"][member]}
            try:
                validated = Project.model_validate(project)
                presentation = PresentationDocument.model_validate(shown)
                _validate_model_presentation(presentation, validated)
                changes = text_project_changes(text, validated, registries=self._stored_regions)
            except ValueError as error:
                raise ProjectFolderInvalid(str(error)) from error
            pending = []
            for relative, data in sorted(changes.items()):
                path = Path(relative)
                pending.append(_PendingFile(
                    address=AuthoredFileAddress(kind="declaration", id=text.configuration.project)
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
            removed = {name for name, data in changes.items() if data is None}
            _check_file_spellings(root, (text.files.keys() | changes.keys()) - removed)

            def guard() -> None:
                check_revision("project", text.revision, self._load(root).revision)
                check_revision("presentation", presentation_revision, _read_presentation(root)[1])

            _publish_patch(folder, pending, guard)
            saved = self._load(root)
            problems = [
                StudioProjectWindowMapProblem.from_diagnostic(item, declaration=saved.declaration)
                for item in self._compile(folder.location, saved).plan.diagnostics
            ] if command.kind != "rename" else []
            return StudioMemberCommandResult(
                member=name, snapshot=self._snapshot(folder.location, saved), problems=problems,
            )

    def apply_model_command(
        self, project_id: str, body: ModelEditRequest, components: ComponentRegistry, store: ModelStore,
        source: dict[str, JsonValue] | None = None,
    ) -> AppliedModelEdit | ModelEditProblem | ModelEditAsked:
        """Apply a named Model operation to the shown document's composed content, or to ``source``, the selected runs'
        copy of it, and commit what ``store`` makes of it with the layout, in one write.

        A document the edit creates gets the layout of its base, the edited layout when its base is the shown one.
        """
        from feedbax.contracts.model_edit import ModelEditAsked, ModelEditProblem, ModelEditState, ModelProjectionRequest, SetField
        from feedbax.compiler.model_document import inspect_model_signals
        from feedbax.compiler.model_edit import ModelEditDraft, apply_model_edit
        from feedbax.compiler.model_ports import ModelRegistryPorts
        from feedbax.compiler.model_signals import intervention_operand_defaults
        from feedbax.compiler.model_templates import instantiate_model_template
        from feedbax.project_model_selection import selected_task_signal

        snapshot = self.read(project_id, revision=body.revision)
        check_revision("presentation", body.presentation_revision, snapshot.presentation_revision)
        if isinstance(operation := body.command.operation, SetField):
            try:
                check_authored_array_budget(
                    operation.value, path=("payload", *operation.address.stored_path()),
                    source=f"documents/{body.document}.json",
                )
            except AuthoredArrayBudgetError as error:
                return ModelEditProblem(address=operation.address, detail=str(error))
        shown = payload, layout = model_edit_source(snapshot, body.document)
        if source is not None:
            payload = source
        state = ModelEditState.model_validate({
            "document": StoredModelDocument.model_validate(payload).lower(components), **layout,
        })
        authority = ModelRegistryPorts(components)

        def reference() -> ModelEditState | None:
            if source is not None:
                restored = shown
            elif isinstance(entry := snapshot.project.documents[body.document], BasedDocument):
                restored = model_edit_source(snapshot, entry.base.document)
            else:
                return None
            return ModelEditState.model_validate({
                "document": StoredModelDocument.model_validate(restored[0]).lower(components), **restored[1],
            })

        def attachment_defaults(owner: ModelAddress, operation: str):
            from feedbax.application.task_signals import matched_task_inputs
            from feedbax.contracts.jobs import StudioSavedTrainKey
            from feedbax.contracts.task_document import TaskDocument
            from feedbax.project_model_selection import selected_model_and_task

            supplied, task = {}, None
            model = state.document
            if operation.lower() == "add" and body.rows and body.step is not None:
                variant = self.read_plan(project_id, revision=body.revision).plan.variants[0]
                run = StudioSavedTrainKey(variant=variant, step=body.step, row=body.rows[0])
                selection = ModelProjectionRequest(revision=body.revision, document=body.document, run=run)
                if selected := selected_model_and_task(self, project_id, selection):
                    _, signals, _, bindings, task = selected
                    supplied = {port.name: signal.type for port, signal in matched_task_inputs(model, signals, bindings)}
            table = inspect_model_signals(model, components, supplied, task=task if isinstance(task, TaskDocument) else None)
            return intervention_operand_defaults(model, table, owner, operation)

        result = apply_model_edit(
            state, body.command, ports=authority.port, component_problem=authority.placement_problem,
            next_input=authority.next_input, functions=self.functions, reference=reference, components=components,
            attachment_defaults=attachment_defaults,
            instantiate_template=lambda name: instantiate_model_template(name, components, state.document.step_seconds),
            task_signal=lambda command: selected_task_signal(self, project_id, ModelProjectionRequest(
                revision=body.revision, run=command.run, document=body.document,
            ), command.signal),
        )
        if isinstance(result, ModelEditProblem):
            return result
        problems = result.problems
        if result.state.document != state.document:
            found = (*problems, *inspect_model_signals(result.state.document, components).problems)
            problems = tuple({(item.address.to_text(), item.reason): item for item in found}.values())
        if body.store_choice is not None and result.state.document == state.document and not body.files:
            raise ProjectRevisionConflict("project", body.revision)
        if any(file.address.kind != "step" for file in body.files):
            raise ProjectFolderInvalid("A Model edit writes step files alongside its command.")
        change = None
        sweep_problems = []
        created: tuple[str, ...] = ()
        if result.state != state or body.files:
            next_payload = (authored_model_payload(result.state.document, components)
                            if result.state.document != state.document else payload)
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
            if isinstance(stored, ModelEditAsked):
                return stored
            assert snapshot.presentation is not None
            layouts: dict[str, dict[str, JsonValue]] = {}
            narrowed = False
            for document in stored.created:
                base = document.document.base.document if isinstance(document.document, BasedDocument) else None
                if base == body.document:
                    narrowed = True
                    layouts[document.id] = next_layout
                elif base is not None and (entry := snapshot.presentation.documents.get(base)) is not None:
                    layouts[document.id] = _runtime_model_layout(entry)
            if not narrowed:
                layouts[body.document] = next_layout
            shown_file = AuthoredFileAddress(kind="document", id=body.document)
            files = list(stored.files)
            if (next_payload is not payload and not isinstance(body.command.operation, SetField)
                    and any(file.address == shown_file for file in files)):
                step_files = [file for file in files if file.address.kind == "step"]
                candidate = preview_authored_files(snapshot.project, step_files)
                before_draft = ModelEditDraft(state, authority.port)
                after_draft = ModelEditDraft(result.state, authority.port)
                moved, sweep_problems = _model_sweep_changes(
                    candidate, body.document, before_draft, after_draft, result.relocations,
                    before_payload=payload, after_payload=next_payload, functions=self.functions,
                    read_plan=lambda: self.read_plan(project_id, revision=body.revision).plan,
                )
                files = merged_files([*files, *moved])
            alone = not stored.created and all(file.address == shown_file for file in files)
            created = tuple(document.id for document in stored.created)
            if patches := _layout_patches(snapshot.presentation, layouts):
                files.append(StudioProjectFilePatch(address=PRESENTATION_ADDRESS, patches=patches))
            before = snapshot
            if files or stored.created:
                snapshot = self._patch(snapshot.project_id, StudioProjectPatchBatch(
                    revision=snapshot.revision, presentation_revision=snapshot.presentation_revision,
                    files=files, created=list(stored.created),
                ), model_operation=True)
            if alone:
                change = ModelEditChange(before, (payload, layout), model_edit_source(snapshot, body.document))
        return AppliedModelEdit(snapshot, result.changed, result.clipboard, change, created, problems, tuple(sweep_problems))

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
    ) -> StudioProjectSnapshot:
        """Apply ``body``'s file patches, after creating the ``created`` documents, in one write."""
        with self._locked_folder(project_id) as folder:
            root = folder.path
            text = self._load(root)
            check_revision("project", body.revision, text.revision)
            files, created = (body.files, body.created) if isinstance(body, StudioProjectPatchBatch) else ([body], [])
            payload = text.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
            configuration = text.configuration.model_dump(mode="json")
            addresses = {}
            declaration_settings = False
            made = set()
            for document in created:
                address = AuthoredFileAddress(kind="document", id=document.id)
                relative = _relative_file(address)
                if document.id in payload.setdefault("documents", {}) or relative in text.files:
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
                    relative = text.declaration if file.address == DECLARATION_ADDRESS else _relative_file(file.address)
                    if relative in addresses and relative not in made:
                        raise ProjectFolderInvalid(f"{relative} is addressed twice")
                    addresses[relative] = file.address
                    if relative == STUDIO_PRESENTATION_FILENAME:
                        assert presentation is not None
                        entry = presentation
                    elif relative == text.declaration:
                        if model_operation:
                            if any(not patch.path.startswith("/sweep/") for patch in file.patches):
                                raise ProjectFolderInvalid("A Model edit only rewrites declaration sweep targets.")
                            entry = payload
                        else:
                            declaration_settings = True
                            entry = configuration
                    else:
                        if relative not in text.files and relative not in made:
                            raise ProjectFolderNotFound(file.address.id)
                        entry = _project_file_entry(payload, file.address)
                    _apply_file_edits(entry, file, model_operation=model_operation)
                project = Project.model_validate(payload)
                changes = text_project_changes(text, project, registries=self._stored_regions, only=addresses)
                if declaration_settings:
                    settings = ProjectConfiguration.model_validate(configuration).settings
                    before = text_project_settings_edit(
                        text.manifest, text.configuration.settings, declaration=text.declaration,
                    )
                    after = text_project_settings_edit(text.manifest, settings, declaration=text.declaration)
                    if before != after:
                        changes[text.declaration] = after
                if presentation is not None:
                    before = _encode_presentation(_decode_presentation(presentation_data))
                    shown = PresentationDocument.model_validate(presentation)
                    _validate_model_presentation(shown, project)
                    after = _encode_presentation(shown)
                    if before != after:
                        changes[STUDIO_PRESENTATION_FILENAME] = after
            except (ValueError, TypeError, RecursionError) as error:
                raise ProjectFolderInvalid(str(error)) from error

            def guard() -> None:
                check_revision("project", text.revision, self._load(root).revision)
                if edits_presentation:
                    check_revision("presentation", body.presentation_revision, _read_presentation(root)[1])

            if changes:
                _publish_patch(folder, [
                    _PendingFile(
                        address=AuthoredFileAddress(kind="declaration", id=text.configuration.project)
                        if relative == text.declaration else addresses[relative],
                        before=(
                            sha256_bytes(presentation_data) if presentation_data is not None else None
                        ) if relative == STUDIO_PRESENTATION_FILENAME else text.file_revisions.get(relative),
                        data=data.decode() if data is not None else None,
                    )
                    for relative, data in changes.items()
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
            _validate_model_presentation(body.presentation, text.project)
            replacement = _encode_presentation(body.presentation)
            if replacement != data:
                path = root / STUDIO_PRESENTATION_FILENAME
                mode = stat.S_IMODE(_regular(path).st_mode) if data is not None else 0o600
                def guard() -> None:
                    check_revision("presentation", revision, _read_presentation(root)[1])

                folder.write(path, replacement, mode=mode, before_publish=guard)
                text = self._load(root)
            return self._snapshot(folder.location, text)
