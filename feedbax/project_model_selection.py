"""Read the Model and Task signals of one selected authored run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.graph import StudioTaskBindingSpec, StudioTaskDataSpec
from feedbax.contracts.model_document import ModelDocument, SignalType
from feedbax.contracts.model_edit import ModelProjectionRequest, ModelTaskSignal
from feedbax.contracts.project_plan import RunKey, SelectedRunDefinition
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument

if TYPE_CHECKING:
    from feedbax.project_folder import ProjectFolder


def task_signal(item: StudioTaskDataSpec) -> ModelTaskSignal:
    shape = item.expected_shape
    if shape and shape[0] == "time":
        shape = shape[1:]
    return ModelTaskSignal(
        id=item.id, label=item.label, type=SignalType(base=item.dtype, shape=shape, unit=item.units),
    )


def selected_model(definition: SelectedRunDefinition, document: str) -> object:
    managed = definition.payload.get("trial_documents")
    path = "payload.trial_documents.model" if isinstance(managed, dict) else "payload.graph"
    if definition.document_ids.get(path) != document or definition.document_pointers.get(path):
        return None
    return managed.get("model") if isinstance(managed, dict) else definition.payload.get("graph")


def selected_model_and_task(folders: ProjectFolder, project_id: str, body: ModelProjectionRequest):
    from feedbax.compiler.trial_execution import resolve_python_task
    from feedbax.project_folder import ProjectFolderInvalid
    from feedbax.tasks.reaching import builtin_python_tasks

    key = RunKey(body.run.variant, body.run.step, body.run.row)
    plan = folders.read_plan(project_id, revision=body.revision, source_runs=(key,)).plan
    definition = plan.selected_definitions.get(key)
    if definition is None or (model := selected_model(definition, body.document)) is None:
        raise ProjectFolderInvalid("The selected run does not use this Model")
    managed = definition.payload.get("trial_documents")
    objective = definition.payload.get("objective")
    if isinstance(managed, dict):
        if managed.get("python_task") is not None:
            selection = PythonTaskSelection.model_validate(managed["python_task"])
            template = resolve_python_task(selection, folders.python_tasks or builtin_python_tasks())
            signals = {} if isinstance(template, tuple) else template.signals
        else:
            task = TaskDocument.model_validate(managed["task"])
            signals = {signal.name: signal.type for signal in task.signals}
        binding = StudioTaskBindingSpec(exposed_data=[
            StudioTaskDataSpec(
                id=name, label=name, kind="signal", path=name,
                dtype={"integer": "int", "real": "float", "boolean": "bool"}[type_.base],
                expected_shape=["time", *type_.shape], bindable=True,
                units=next((name for name, unit in UNITS.items()
                            if name and name != "st" and unit.scale == 1 and unit.dimensions == type_.dimensions), None),
            ) for name, type_ in signals.items()
        ])
        objective = managed.get("objective")
    else:
        binding = StudioTaskBindingSpec.model_validate(definition.payload.get("task_binding_spec") or {})
    overrides = managed.get("input_overrides", {}) if isinstance(managed, dict) else {}
    return ModelDocument.model_validate(model), binding, objective, isinstance(managed, dict), overrides


def selected_task_signal(folders: ProjectFolder, project_id: str, body: ModelProjectionRequest, signal: str) -> ModelTaskSignal:
    from feedbax.project_folder import ProjectFolderInvalid

    _, task, _, _, _ = selected_model_and_task(folders, project_id, body)
    item = next((item for item in task.exposed_data if item.bindable and item.id == signal), None)
    if item is None:
        raise ProjectFolderInvalid("The selected task does not expose this signal")
    projected = task_signal(item)
    return projected.model_copy(update={"type": projected.type.model_copy(update={"base": None})})
