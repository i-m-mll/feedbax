"""Read the Model and Task signals of one selected authored run."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import Level, ScopedValue
from feedbax.contracts.model_document import ModelDocument, SignalInput, SignalType
from feedbax.contracts.model_edit import ModelProjectionRequest, ModelTaskSignal
from feedbax.planner.project_plan import RunKey, SelectedRunDefinition
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
from feedbax.contracts.trial_bindings import selected_signal_id

if TYPE_CHECKING:
    from feedbax.project_folder import ProjectFolder


def task_signal(item: ScopedValue) -> ModelTaskSignal:
    type_ = item.type
    unit = next((name for name, unit in UNITS.items()
                 if name and name != "st" and unit.scale == 1 and unit.dimensions == type_.dimensions), None)
    return ModelTaskSignal(id=item.id, label=item.name, type=SignalType(
        base=type_.base, shape=type_.shape, unit=unit, space=type_.space,
    ))


def model_task_signals(folders: ProjectFolder, source: TaskDocument | PythonTaskSelection | None) -> tuple[ScopedValue, ...]:
    from feedbax.application.composition import application_python_tasks
    from feedbax.execution.trial_resolution import resolve_python_task

    if isinstance(source, PythonTaskSelection):
        template = resolve_python_task(source, folders.python_tasks or application_python_tasks())
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
    managed = definition.payload.get("trial_documents")
    path = "payload.trial_documents.model"
    if (
        not isinstance(managed, dict)
        or definition.document_ids.get(path) != document
        or definition.document_pointers.get(path)
    ):
        return None
    return managed.get("model")


def selected_model_and_task(
    folders: ProjectFolder, project_id: str, body: ModelProjectionRequest, *, definition: SelectedRunDefinition | None = None,
):
    from feedbax.project_folder import ProjectFolderInvalid

    if body.run is None:
        raise ProjectFolderInvalid("Choose a run for this Model selection.")
    if definition is None:
        key = RunKey(body.run.variant, body.run.step, body.run.row)
        plan = folders.read_plan(project_id, revision=body.revision, source_runs=(key,)).plan
        definition = plan.selected_definitions.get(key)
    if definition is None or (model := selected_model(definition, body.document)) is None:
        raise ProjectFolderInvalid("The selected run does not use this Model")
    managed = definition.payload.get("trial_documents")
    assert isinstance(managed, dict)
    if managed.get("python_task") is not None:
        source = PythonTaskSelection.model_validate(managed["python_task"])
    elif managed.get("task") is not None:
        source = TaskDocument.model_validate(managed["task"])
    else:
        source = None
    return (
        ModelDocument.model_validate(model), model_task_signals(folders, source),
        managed.get("objective"), managed.get("input_bindings", {}),
    )


def selected_task_signal(folders: ProjectFolder, project_id: str, body: ModelProjectionRequest, signal: str) -> ModelTaskSignal:
    from feedbax.project_folder import ProjectFolderInvalid

    _, task, _, _ = selected_model_and_task(folders, project_id, body)
    item = next((item for item in task if item.id == signal), None)
    if item is None:
        raise ProjectFolderInvalid("The selected task does not expose this signal")
    return task_signal(item)
