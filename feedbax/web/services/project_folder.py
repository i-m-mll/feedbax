"""Adapt core project edits to Studio connection review and undo responses."""

from __future__ import annotations

from base64 import b64decode, urlsafe_b64encode
from binascii import Error as Base64Error
from collections.abc import Callable, Mapping
from copy import deepcopy
from hmac import compare_digest, digest
import os
from pathlib import Path
from secrets import token_bytes
from typing import TYPE_CHECKING

from feedbax.contracts.base import StrictModel, sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Differences, ProjectValueEdit, Sha256, apply_differences, diff
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project_io import PROJECT_DECLARATION_SUFFIX, TextProject, TextProjectExternalInputs
from feedbax.contracts.project_resolution import BaseReaders, RunKey
from feedbax.contracts.stored_model import StoredModelDocument, model_keyed_collection, model_member_order
from feedbax.contracts.studio_project import StudioProjectDocument, StudioProjectSnapshot
from feedbax.project_environment import PackageAvailability
from feedbax.examples import ShippedExample
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

if TYPE_CHECKING:
    from feedbax.analysis.operations import OperationRegistry
    from feedbax.application.bundle import ApplicationRegistryBundle
    from feedbax.plot.templates import FigureTemplateRegistry
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.model_edit import (
        ModelConnectionReview, ModelConnectionReviewRequest, ModelEditRequest, ModelEditResponse, ModelUndoRequest,
    )
    from feedbax.contracts.expression_registry import FunctionRegistry
    from feedbax.tasks.registry import PythonTaskRegistry
from feedbax.synthesis.methods import SynthesisMethodRegistry
from feedbax.training.methods import TrainingMethodRegistry
from feedbax.web.services.trial_expression_menus import TrialExpressionMenus


class _ModelInverse(StrictModel):
    project: ProjectName
    document: ProjectName
    payload: Differences
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
        training_methods: TrainingMethodRegistry | None = None,
        synthesis_methods: SynthesisMethodRegistry | None = None,
        figure_templates: FigureTemplateRegistry | None = None,
        load_examples: Callable[[], Mapping[str, ShippedExample]] = dict,
        packages: PackageAvailability | None = None,
    ) -> None:
        super().__init__(
            root, operations=operations, components=components, functions=functions,
            python_tasks=python_tasks, figure_templates=figure_templates, training_methods=training_methods,
            synthesis_methods=synthesis_methods,
            load_examples=load_examples, packages=packages,
        )
        self._model_undo_key = token_bytes(32)
        self.trial_expression_menus = TrialExpressionMenus()

    def open_selected_folder(self, folder: Path) -> StudioProjectDocument:
        """Open a folder selected through the private native control socket."""
        return self.open_native_folder(folder)[0]

    def open_native_folder(
        self, folder: Path, expected_identity: tuple[int, int] | None = None,
    ) -> tuple[StudioProjectDocument, ProjectFolderLocation]:
        return self._open_native_location(folder, expected_identity, declaration=None)

    def open_native_document(self, file: Path) -> tuple[StudioProjectDocument, ProjectFolderLocation]:
        if not file.is_absolute() or ".." in file.parts or not file.name.endswith(PROJECT_DECLARATION_SUFFIX):
            raise ProjectFolderInvalid("Choose a .feedbax project declaration.")
        return self._open_native_location(file.parent, None, declaration=file.name)

    def _open_native_location(
        self, folder: Path, expected_identity: tuple[int, int] | None, *, declaration: str | None,
    ) -> tuple[StudioProjectDocument, ProjectFolderLocation]:
        if not folder.is_absolute() or ".." in folder.parts:
            raise ProjectFolderInvalid("Choose an absolute Studio project folder.")
        try:
            path = folder.parent.resolve(strict=True) / folder.name
        except FileNotFoundError as error:
            raise ProjectFolderNotFound(str(folder)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise ProjectFolderInvalid(str(error)) from error
        location = ProjectFolderLocation.selected(path)
        if location.path == self.root:
            raise ProjectFolderInvalid("Choose an individual Studio project folder.")
        if expected_identity is not None and location.identity != expected_identity:
            raise ProjectFolderNotFound("The saved project folder was replaced")
        if not os.access(location.path, os.R_OK | os.W_OK | os.X_OK):
            raise ProjectFolderInvalid("The selected project folder must be readable and writable.")
        return self._open_folder(location, declaration=declaration), location

    def open_native_id(self, project_id: str) -> tuple[StudioProjectDocument, ProjectFolderLocation]:
        location = self._folder(project_id)
        return self._open_folder(location), location

    def _base_readers(self, location: ProjectFolderLocation, text: TextProject) -> BaseReaders:
        if location.path.parent == self.root:
            return super()._base_readers(location, text)
        read_pin = TextProjectExternalInputs(
            text, authority=location.path, authority_identity=location.identity,
        ).read
        return BaseReaders(external=read_pin, figure_external=read_pin)

    def review_model_connections(
        self, project_id: str, body: ModelConnectionReviewRequest, bundle: ApplicationRegistryBundle,
    ) -> ModelConnectionReview:
        from feedbax.compiler.model_document import inspect_model_signals
        from feedbax.contracts.model_edit import TaskSignalSource
        from feedbax.contracts.task_document import TaskDocument
        from feedbax.compiler.model_edit import review_model_connections
        from feedbax.compiler.model_ports import ModelRegistryPorts
        from feedbax.application.task_signals import matched_task_inputs
        from feedbax.project_model_selection import selected_model_and_task, task_signal
        from feedbax.web.services.model_projection import evaluate_projection
        from feedbax.web.services.trial_scope import selection_scope

        snapshot = self.read(project_id, revision=body.revision)
        payload, _ = model_edit_source(snapshot, body.document)
        components = bundle.components
        model = StoredModelDocument.model_validate(payload).lower(components)
        task, overrides, source = (), {}, None
        task_source = body.source if isinstance(body.source, TaskSignalSource) else None
        try:
            if body.run is None and (body.rows or body.axes or body.evaluate_scope is not None):
                raise ProjectFolderInvalid("Choose a run for this Model selection.")
            run = body.run or (task_source.run if task_source is not None else None)
            if run is not None:
                selection = body.model_copy(update={"run": run})
                step = snapshot.project.steps.get(run.step)
                if step is not None and step.kind == "evaluate":
                    resolved, task, _, overrides, source = evaluate_projection(self, project_id, selection, bundle)
                    if resolved is None:
                        raise ProjectFolderInvalid("The selected Evaluate has no readable Model.")
                    model = resolved
                else:
                    if body.evaluate_scope is not None:
                        raise ProjectFolderInvalid("Choose an Evaluate run for this Model selection.")
                    key = RunKey(run.variant, run.step, run.row)
                    plan = self.read_plan(project_id, revision=body.revision, source_runs=(key,)).plan
                    if body.rows or body.axes:
                        selection_scope(snapshot.project, plan, key, tuple(body.rows), body.axes or None)
                    definition = plan.selected_definitions.get(key)
                    if definition is not None:
                        paired = selected_model_and_task(self, project_id, selection, definition=definition)
                        if paired is not None:
                            model, task, _, overrides, source = paired
            authority = ModelRegistryPorts(components)
            offered = None
            if task_source is not None:
                if body.graph.composites:
                    raise ProjectFolderInvalid("A task signal connects at the Model root")
                offered = next((item for item in task if item.id == task_source.signal), None)
                if offered is None:
                    raise ProjectFolderInvalid("The selected task does not expose this signal")
            library = tuple({
                "kind": "leaf", "type": item.identity.type_id if item.identity else item.name,
                "revision": item.revision, "fields": {}, "port_counts": {},
            } for item in components.list_all() if not item.is_composite)
            supplied = {port.name: item.type for port, item in matched_task_inputs(model, task, overrides)}
            return review_model_connections(
                model, body, layout=authority.layout, next_input=authority.next_input,
                library=library, task_signal=task_signal(offered) if offered is not None else None,
                task_type=offered.type if offered is not None else None,
                signals=inspect_model_signals(
                    model, components, supplied, task=source if isinstance(source, TaskDocument) else None,
                ),
            )
        except ValueError as error:
            raise ProjectFolderInvalid(str(error)) from error

    def edit_model(
        self, project_id: str, body: ModelEditRequest, components: ComponentRegistry,
    ) -> ModelEditResponse:
        from feedbax.contracts.model_edit import (
            ModelEditApplied, ModelEditAsked, ModelEditProblem, ModelEditRefused, ModelEditResponse,
        )

        from feedbax.web.services.model_scope import model_edit_store
        from feedbax.web.services.trial_scope import row_model

        project_id = self.project_id(project_id)
        row = None
        if body.rows:
            if body.files:
                raise ProjectFolderInvalid("A Model edit for selected runs writes no step files alongside it.")
            row = row_model(self, project_id, self.read(project_id, revision=body.revision).project, body)

        def store(project, body, before, after):
            return model_edit_store(self, project_id, project, body, before, after, row)

        result = self.apply_model_command(
            project_id, body, components, store, source=row.content if row is not None else None,
        )
        if isinstance(result, ModelEditAsked):
            return ModelEditResponse(result=result)
        if isinstance(result, ModelEditProblem):
            return ModelEditResponse(result=ModelEditRefused(status="refused", problem=result))
        change = result.change
        token = self._model_inverse(
            change.before_snapshot, body.document, change.before, change.after, result.changed,
        ) if change is not None else None
        return ModelEditResponse(result=ModelEditApplied(
            status="applied", snapshot=result.snapshot, changed=result.changed, clipboard=result.clipboard,
            undo_token=token, created=result.created, problems=result.problems,
            sweep_problems=result.sweep_problems,
        ))

    def undo_model(self, project_id: str, body: ModelUndoRequest) -> ModelEditResponse:
        from feedbax.compiler.model_document import inspect_model_signals
        from feedbax.contracts.model_edit import ModelEditApplied, ModelEditResponse
        from feedbax.compiler.model_ports import ModelRegistryPorts
        from feedbax.studio.model_undo import apply_delta

        try:
            encoded = b64decode(body.token, altchars=b"-_", validate=True)
            signature, data = encoded[:32], encoded[32:]
            if not compare_digest(signature, digest(self._model_undo_key, data, "sha256")):
                raise ValueError("Invalid model undo signature")
            edit = _ModelInverse.model_validate_json(data)
        except (Base64Error, ValueError) as error:
            raise ProjectFolderInvalid("The model edit history entry is unavailable") from error
        project_id = self.project_id(project_id)
        if edit.project != project_id:
            raise ProjectFolderInvalid("The model edit history entry belongs to another project")
        snapshot = self.read(project_id, revision=body.revision)
        check_revision("presentation", body.presentation_revision, snapshot.presentation_revision)
        current = model_edit_source(snapshot, edit.document)
        if self._model_edit_digest(current) != edit.before:
            raise ProjectRevisionConflict("project", snapshot.revision)
        payload = apply_differences(current[0], edit.payload, keyed_collection=model_keyed_collection)
        layout = deepcopy(current[1])
        apply_delta(layout, edit.layout)
        token = self._model_inverse(snapshot, edit.document, current, (payload, layout), edit.changed)
        snapshot = self._write_model_change(
            snapshot, edit.document, payload, layout,
            had_entry=edit.had_entry, had_documents=edit.had_documents,
        )
        problems = ()
        if self.components is not None:
            model = StoredModelDocument.model_validate(payload).lower(self.components)
            found = (
                *ModelRegistryPorts(self.components).problems(model),
                *inspect_model_signals(model, self.components).problems,
            )
            problems = tuple({(item.address.to_text(), item.reason): item for item in found}.values())
        return ModelEditResponse(result=ModelEditApplied(
            status="applied", snapshot=snapshot, changed=edit.changed, undo_token=token, problems=problems,
        ))

    @staticmethod
    def _model_edit_digest(value: tuple[dict, dict]) -> str:
        orders = [[list(path), keys] for path, keys in sorted(model_member_order(value[0]))]
        return sha256_bytes(canonical_json_v2_bytes([*value, orders]))

    def _model_inverse(
        self, snapshot: StudioProjectSnapshot, document: ProjectName,
        before: tuple[dict, dict], after: tuple[dict, dict],
        changed: tuple[ModelAddress, ...],
    ) -> str:
        from feedbax.studio.model_undo import inverse_delta

        assert snapshot.presentation is not None
        inverse = _ModelInverse(
            project=snapshot.project_id, document=document, before=self._model_edit_digest(after),
            payload=diff(after[0], before[0], keyed_collection=model_keyed_collection),
            layout=tuple(inverse_delta(before[1], after[1])),
            changed=changed,
            had_entry=document in snapshot.presentation.documents,
            had_documents="documents" in snapshot.presentation.model_fields_set,
        )
        data = inverse.model_dump_json().encode()
        return urlsafe_b64encode(digest(self._model_undo_key, data, "sha256") + data).decode()
