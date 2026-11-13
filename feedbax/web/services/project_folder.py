"""Adapt core project edits to Studio connection review and undo responses."""

from __future__ import annotations

from base64 import b64decode, urlsafe_b64encode
from binascii import Error as Base64Error
from copy import deepcopy
from hmac import compare_digest, digest
import os
from pathlib import Path
from secrets import token_bytes
from threading import Lock
from typing import TYPE_CHECKING

from feedbax.contracts.base import StrictModel, sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import ProjectValueEdit, Sha256
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project_io import TextProject, TextProjectExternalInputs
from feedbax.contracts.project_resolution import BaseReaders
from feedbax.contracts.studio_project import StudioProjectDocument, StudioProjectSnapshot
from feedbax.project_folder import (
    ProjectFileExists as ProjectFileExists,
    ProjectFolder,
    ProjectFolderLocation,
    ProjectFolderInvalid as ProjectFolderInvalid,
    ProjectFolderNotFound as ProjectFolderNotFound,
    ProjectRevisionConflict as ProjectRevisionConflict,
    SettledProjectPlan as SettledProjectPlan,
    check_revision,
    model_edit_source,
    preview_authored_files as preview_authored_files,
)
from feedbax.project_model_selection import selected_task_signal

if TYPE_CHECKING:
    from feedbax.analysis.operations import OperationRegistry
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.model_edit import (
        ModelConnectionReview, ModelConnectionReviewRequest, ModelEditRequest, ModelEditResponse, ModelUndoRequest,
    )
    from feedbax.expression_registry import FunctionRegistry
    from feedbax.tasks.registry import PythonTaskRegistry


class _ModelInverse(StrictModel):
    project: ProjectName
    document: ProjectName
    payload: tuple[ProjectValueEdit, ...]
    layout: tuple[ProjectValueEdit, ...]
    changed: tuple[ModelAddress, ...]
    before: Sha256
    had_entry: bool
    had_documents: bool


class ProjectFolderService(ProjectFolder):
    """Add window-specific operations to the shared text project folder."""

    def __init__(
        self, root: Path, *, operations: OperationRegistry | None = None,
        components: ComponentRegistry | None = None,
        functions: FunctionRegistry | None = None,
        python_tasks: PythonTaskRegistry | None = None,
    ) -> None:
        super().__init__(
            root, operations=operations, components=components, functions=functions, python_tasks=python_tasks,
        )
        self._model_undo_key = token_bytes(32)
        self._external_projects: dict[str, ProjectFolderLocation] = {}
        self._external_projects_lock = Lock()

    def open_selected_folder(self, folder: Path) -> StudioProjectDocument:
        """Open a folder selected through the private native control socket."""
        if not folder.is_absolute() or ".." in folder.parts:
            raise ProjectFolderInvalid("Choose an absolute Studio project folder.")
        try:
            path = folder.parent.resolve(strict=True) / folder.name
        except FileNotFoundError as error:
            raise ProjectFolderNotFound(str(folder)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise ProjectFolderInvalid(str(error)) from error
        if path == self.root:
            raise ProjectFolderInvalid("Choose an individual Studio project folder.")
        if path.parent == self.root:
            with self._external_projects_lock:
                if path.name in self._external_projects:
                    raise ProjectFolderInvalid("The selected project conflicts with an external project ID.")
            return super().open(path.name)
        grant = ProjectFolderLocation.selected(path)
        if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
            raise ProjectFolderInvalid("The selected project folder must be readable and writable.")
        locator = os.fsencode(grant.path) + b"\0" + str(grant.identity).encode("ascii")
        project_id = f"external-{sha256_bytes(locator)}"
        document = self._open_folder(project_id, grant)
        with self._external_projects_lock:
            if os.path.lexists(self.root / project_id):
                raise ProjectFolderInvalid("The selected project conflicts with a managed project ID.")
            self._external_projects[project_id] = grant
        return document

    def _folder(self, project_id: str) -> ProjectFolderLocation:
        with self._external_projects_lock:
            grant = self._external_projects.get(project_id)
        return grant if grant is not None else super()._folder(project_id)

    def _base_readers(self, project_id: str, text: TextProject) -> BaseReaders:
        with self._external_projects_lock:
            grant = self._external_projects.get(project_id)
        if grant is None:
            return super()._base_readers(project_id, text)
        read_pin = TextProjectExternalInputs(
            text, authority=grant.path, authority_identity=grant.identity,
        ).read
        return BaseReaders(external=read_pin, figure_external=read_pin)

    def review_model_connections(
        self, project_id: str, body: ModelConnectionReviewRequest, components: ComponentRegistry,
    ) -> ModelConnectionReview:
        from feedbax.contracts.model_document import ModelDocument
        from feedbax.contracts.model_edit import ModelProjectionRequest, TaskSignalSource
        from feedbax.studio.model_edit import review_model_connections
        from feedbax.studio.model_ports import ModelRegistryPorts

        snapshot = self.read(project_id, revision=body.revision)
        payload, _ = model_edit_source(snapshot, body.graph.document)
        try:
            authority = ModelRegistryPorts(components)
            task_signal = None
            if isinstance(body.source, TaskSignalSource):
                if body.graph.composites:
                    raise ProjectFolderInvalid("A task signal connects at the Model root")
                task_signal = selected_task_signal(self, project_id, ModelProjectionRequest(
                    revision=body.revision, run=body.source.run, document=body.graph.document,
                ), body.source.signal)
            library = tuple({
                "kind": "leaf", "type": item.identity.type_id if item.identity else item.name,
                "revision": item.revision, "fields": {}, "port_counts": {},
            } for item in components.list_all() if not item.is_composite)
            return review_model_connections(
                ModelDocument.model_validate(payload), body, layout=authority.layout, next_input=authority.next_input,
                library=library, accepts=authority.accepts, task_signal=task_signal,
            )
        except ValueError as error:
            raise ProjectFolderInvalid(str(error)) from error

    def edit_model(
        self, project_id: str, body: ModelEditRequest, components: ComponentRegistry,
    ) -> ModelEditResponse:
        from feedbax.contracts.model_edit import ModelEditApplied, ModelEditProblem, ModelEditRefused, ModelEditResponse

        result = self.apply_model_command(project_id, body, components)
        if isinstance(result, ModelEditProblem):
            return ModelEditResponse(result=ModelEditRefused(status="refused", problem=result))
        change = result.change
        token = self._model_inverse(
            change.before_snapshot, body.document, change.before, change.after, result.changed,
        ) if change is not None else None
        return ModelEditResponse(result=ModelEditApplied(
            status="applied", snapshot=result.snapshot, changed=result.changed, clipboard=result.clipboard,
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
        check_revision("presentation", body.presentation_revision, snapshot.presentation_revision)
        current = model_edit_source(snapshot, edit.document)
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
