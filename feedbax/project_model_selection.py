"""Read the Model and Task signals of one selected authored run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import ValueType
from feedbax.contracts.graph import StudioTaskBindingSpec, StudioTaskDataSpec
from feedbax.contracts.model_document import ModelDocument, SignalType
from feedbax.contracts.model_edit import ModelProjectionRequest, ModelTaskSignal
from feedbax.planner.project_plan import RunKey, SelectedRunDefinition
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument

if TYPE_CHECKING:
    from feedbax.project_folder import ProjectFolder


def task_signal(item: StudioTaskDataSpec) -> ModelTaskSignal:
    shape = item.expected_shape
    if shape and shape[0] == "time":
        shape = shape[1:]
    space = item.metadata.get("space")
    return ModelTaskSignal(id=item.id, label=item.label, type=SignalType(
        base=item.dtype, shape=tuple(shape) if shape is not None else None,
        unit=item.units, space=space if space in ("point", "free") else None,
    ))


def model_task_signals(folders: ProjectFolder, source: TaskDocument | PythonTaskSelection | None) -> StudioTaskBindingSpec:
    from feedbax.application.composition import application_python_tasks
    from feedbax.execution.trial_resolution import resolve_python_task

    signals: list[tuple[str, str, ValueType]] = []
    if isinstance(source, PythonTaskSelection):
        template = resolve_python_task(source, folders.python_tasks or application_python_tasks())
        if not isinstance(template, tuple):
            signals = [(name, name, type_) for name, type_ in template.signals.items()]
    elif isinstance(source, TaskDocument):
        signals = [(signal.id, signal.name, signal.type) for signal in source.signals]
    return StudioTaskBindingSpec(exposed_data=[
        StudioTaskDataSpec(
            id=id_, label=name, kind="signal", path=id_,
            dtype={"integer": "int", "real": "float", "boolean": "bool"}[type_.base],
            expected_shape=["time", *type_.shape], bindable=True,
            units=next((name for name, unit in UNITS.items()
                        if name and name != "st" and unit.scale == 1 and unit.dimensions == type_.dimensions), None),
            metadata={"space": type_.space} if type_.space is not None else {},
        ) for id_, name, type_ in signals
    ])


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


def selected_model_and_task(folders: ProjectFolder, project_id: str, body: ModelProjectionRequest):
    from feedbax.project_folder import ProjectFolderInvalid

    if body.run is None:
        raise ProjectFolderInvalid("Choose a run for this Model selection.")
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
    item = next((item for item in task.exposed_data if item.bindable and item.id == signal), None)
    if item is None:
        raise ProjectFolderInvalid("The selected task does not expose this signal")
    return task_signal(item)
