"""Training documents and optimizer settings for registered methods."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import Field, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.objective_document import ObjectiveDocument, StoredObjectiveDocument
from feedbax.contracts.stored_model import StoredModelDocument
from feedbax.contracts.studio_refinements import validate_cross_field_refinements
from feedbax.contracts.task_document import PythonTaskSelection, StoredTaskDocument, TaskDocument
from feedbax.contracts.trial_bindings import selected_signal_id

if TYPE_CHECKING:
    from feedbax.component_registry import ComponentRegistry


class TrialDocuments(StrictModel):
    model: ModelDocument
    task: TaskDocument | None
    objective: ObjectiveDocument
    input_bindings: dict[str, str | None] = Field(default_factory=dict)
    python_task: PythonTaskSelection | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="before")
    @classmethod
    def _python_task_without_document(cls, value: Any) -> Any:
        if isinstance(value, dict) and "task" not in value and value.get("python_task") is not None:
            return {"task": None} | value
        return value

    @property
    def task_source(self) -> TaskDocument | PythonTaskSelection:
        if self.task is not None:
            return self.task
        assert self.python_task is not None
        return self.python_task

    def resolved_signal_bindings(
        self, signal_keys: Collection[str], overrides: Mapping[str, str | None] | None = None,
    ) -> dict[str, str | None]:
        """Record the chosen signal ID at each Model input."""
        bindings = self.input_bindings | overrides if overrides is not None else self.input_bindings
        return bindings | {
            port.id: selected_signal_id(port.id, signal_keys, bindings) for port in self.model.ports.inputs
        }

    def without_signal_labels(self) -> TrialDocuments:
        """Use stable signal IDs as names while retaining the resolved input choices."""
        if self.task is None or not self.task.signals:
            return self
        signal_keys = {signal.id for signal in self.task.signals}
        bindings = self.resolved_signal_bindings(signal_keys)
        for port in self.model.ports.inputs:
            default = selected_signal_id(port.id, signal_keys, {})
            if bindings[port.id] == default:
                del bindings[port.id]
        task = self.task.with_signal_ids_as_names()
        if bindings == self.input_bindings and task is self.task:
            return self
        return self.model_copy(update={"task": task, "input_bindings": bindings})

    def bindings_for_same_task(self, task: TaskDocument | PythonTaskSelection) -> Mapping[str, str | None]:
        """Keep a model producer's input choices when an operation uses its Task."""
        if isinstance(task, PythonTaskSelection):
            return self.input_bindings if task == self.python_task else {}
        original = self.task
        if original is None or original.without_markers().with_signal_ids_as_names() != (
            task.without_markers().with_signal_ids_as_names()
        ):
            return {}
        return self.resolved_signal_bindings({signal.id for signal in original.signals})

    @model_validator(mode="after")
    def _one_task(self) -> "TrialDocuments":
        validate_cross_field_refinements("TrialDocuments", self)
        return self


def lower_stored_trial_documents(value: object, *, components: ComponentRegistry | None = None) -> TrialDocuments:
    """Lower resolved authored Train slots once, before pairing and request encoding."""
    if not isinstance(value, dict):
        raise ValueError("Choose Model, Task and Objective documents for Train.")
    task = value.get("task")
    return TrialDocuments.model_validate(value | {
        "model": StoredModelDocument.model_validate(value.get("model")).lower(components),
        "task": StoredTaskDocument.model_validate(task).lower() if task is not None else None,
        "objective": StoredObjectiveDocument.model_validate(value.get("objective")).lower(),
    })
