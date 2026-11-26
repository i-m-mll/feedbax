"""Read the Model and Task signals of one selected authored run."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter, ValidationError

from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import Level, ScopedValue
from feedbax.contracts.model_document import ModelDocument, SignalInput, SignalType
from feedbax.contracts.model_edit import ModelProjectionRequest, ModelTaskSignal
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.planner.project_plan import RunKey, SelectedRunDefinition
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
from feedbax.contracts.trial_bindings import selected_signal_id

if TYPE_CHECKING:
    from feedbax.project_folder import ProjectFolder
    from feedbax.tasks.registry import PythonTaskRegistry


def task_signal(item: ScopedValue) -> ModelTaskSignal:
    type_ = item.type
    unit = next((name for name, unit in UNITS.items()
                 if name and name != "st" and unit.scale == 1 and unit.dimensions == type_.dimensions), None)
    return ModelTaskSignal(id=item.id, label=item.name, type=SignalType(
        base=type_.base, shape=type_.shape, unit=unit, space=type_.space,
    ))


def model_task_signals(
    source: TaskDocument | PythonTaskSelection | None, python_tasks: PythonTaskRegistry | None,
) -> tuple[ScopedValue, ...]:
    from feedbax.application.composition import application_python_tasks
    from feedbax.execution.trial_resolution import resolve_python_task

    if isinstance(source, PythonTaskSelection):
        template = resolve_python_task(source, python_tasks or application_python_tasks())
        if not isinstance(template, tuple):
            return tuple(ScopedValue(id=name, name=name, type=type_, level=Level.STEP)
                         for name, type_ in template.signals.items())
    if isinstance(source, TaskDocument):
        return tuple(ScopedValue(id=item.id, name=item.name, type=item.type, level=Level.STEP) for item in source.signals)
    return ()


def matched_task_inputs(
    model: ModelDocument, signals: tuple[ScopedValue, ...], overrides: Mapping[str, str | None],
) -> tuple[tuple[SignalInput, ScopedValue], ...]:
    values = {item.id: item for item in signals}
    names = {item.name: item.id for item in signals}
    return tuple(
        (port, values[signal]) for port in model.ports.inputs
        if (signal := selected_signal_id(port.id, port.name, names, overrides)) is not None and signal in values
    )


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


def trial_task_slot(value: object, *, path: str = "task") -> TaskDocument | PythonTaskSelection:
    from feedbax.project_folder import ProjectFolderInvalid

    try:
        return _TRIAL_TASK.validate_python(value)
    except ValidationError as error:
        raise ProjectFolderInvalid(f"{path} does not contain a Task or Python task selection.") from error


def _trial_document[Document: BaseModel](path: str, value: object, schema: type[Document]) -> Document:
    from feedbax.project_folder import ProjectFolderInvalid

    try:
        return schema.model_validate(value)
    except ValidationError as error:
        raise ProjectFolderInvalid(
            f"{path} does not contain a valid {schema.__name__}."
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
    managed, prefix = trial_document_slots(definition.payload)
    if managed.get("python_task") is not None:
        source = _trial_document(f"{prefix}python_task", managed["python_task"], PythonTaskSelection)
    elif managed.get("task") is not None:
        source = trial_task_slot(managed["task"], path=f"{prefix}task")
    else:
        source = None
    objective = managed.get("objective")
    return (
        _trial_document(f"{prefix}model", model, ModelDocument), model_task_signals(source, folders.python_tasks),
        _trial_document(f"{prefix}objective", objective, ObjectiveDocument) if objective is not None else None,
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
