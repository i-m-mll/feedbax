"""Read the Model and Task signals of one selected authored run."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter, ValidationError

from feedbax.application.task_signals import model_task_signals
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import ScopedValue
from feedbax.contracts.model_document import ModelDocument, SignalType
from feedbax.contracts.model_edit import ModelProjectionRequest, ModelTaskSignal
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.planner.project_plan import RunKey, SelectedRunDefinition
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument

if TYPE_CHECKING:
    from feedbax.project_folder import ProjectFolder


def task_signal(item: ScopedValue) -> ModelTaskSignal:
    type_ = item.type
    unit = next((name for name, unit in UNITS.items()
                 if name and name != "st" and unit.scale == 1 and unit.dimensions == type_.dimensions), None)
    return ModelTaskSignal(id=item.id, label=item.name, type=SignalType(
        base=type_.base, shape=type_.shape, unit=unit, space=type_.space,
    ))


def selected_model(definition: SelectedRunDefinition, document: str) -> object:
    managed, prefix = trial_document_slots(definition.payload)
    path = f"payload.{prefix}model"
    if (
        definition.document_ids.get(path) != document
        or definition.document_pointers.get(path)
    ):
        return None
    return managed.get("model")


def trial_document_slots(payload: Mapping) -> tuple[Mapping, str]:
    """Return the authored document slots and their path below a step payload."""
    nested = payload.get("trial_documents")
    return (nested, "trial_documents.") if isinstance(nested, dict) else (payload, "")


_TRIAL_TASK = TypeAdapter(TaskDocument | PythonTaskSelection)


def trial_task_slot(value: object) -> TaskDocument | PythonTaskSelection:
    from feedbax.project_folder import ProjectFolderInvalid

    try:
        return _TRIAL_TASK.validate_python(value)
    except ValidationError as error:
        raise ProjectFolderInvalid("The task slot does not contain a Task or Python task selection.") from error


def _trial_document[Document: BaseModel](binding: str, value: object, schema: type[Document]) -> Document:
    from feedbax.project_folder import ProjectFolderInvalid

    try:
        return schema.model_validate(value)
    except ValidationError as error:
        raise ProjectFolderInvalid(
            f"trial_documents.{binding} does not contain a valid {schema.__name__}."
        ) from error


def selected_model_and_task(
    folders: ProjectFolder, project_id: str, body: ModelProjectionRequest, *, definition: SelectedRunDefinition | None = None,
):
    if body.run is None:
        return None
    if definition is None:
        key = RunKey(body.run.variant, body.run.step, body.run.row)
        plan = folders.read_plan(project_id, revision=body.revision, source_runs=(key,)).plan
        definition = plan.selected_definitions.get(key)
    if definition is None or (model := selected_model(definition, body.document)) is None:
        return None
    managed, _ = trial_document_slots(definition.payload)
    if managed.get("python_task") is not None:
        source = _trial_document("python_task", managed["python_task"], PythonTaskSelection)
    elif managed.get("task") is not None:
        source = trial_task_slot(managed["task"])
    else:
        source = None
    objective = managed.get("objective")
    return (
        _trial_document("model", model, ModelDocument), model_task_signals(source, folders.python_tasks),
        _trial_document("objective", objective, ObjectiveDocument) if objective is not None else None,
        managed.get("input_bindings", {}), source,
    )


def selected_task_signal(folders: ProjectFolder, project_id: str, body: ModelProjectionRequest, signal: str) -> ModelTaskSignal:
    from feedbax.project_folder import ProjectFolderInvalid

    selected = selected_model_and_task(folders, project_id, body)
    if selected is None:
        raise ProjectFolderInvalid("The selected run does not use this Model")
    _, task, _, _, _ = selected
    item = next((item for item in task if item.id == signal), None)
    if item is None:
        raise ProjectFolderInvalid("The selected task does not expose this signal")
    return task_signal(item)
