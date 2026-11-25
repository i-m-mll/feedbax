"""Training documents and optimizer settings for registered methods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.studio_refinements import validate_cross_field_refinements
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
from feedbax.contracts.trial_bindings import selected_signal_id


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
        self, names: Mapping[str, str], overrides: Mapping[str, str | None] | None = None,
    ) -> dict[str, str | None]:
        """Record the chosen signal ID at each Model input."""
        bindings = self.input_bindings | overrides if overrides is not None else self.input_bindings
        return bindings | {
            port.id: selected_signal_id(port.id, port.name, names, bindings) for port in self.model.ports.inputs
        }

    def without_signal_labels(self) -> TrialDocuments:
        """Use stable signal IDs as names while retaining the resolved input choices."""
        if self.task is None or not self.task.signals:
            return self
        bindings = self.resolved_signal_bindings({signal.name: signal.id for signal in self.task.signals})
        signal_ids = {signal.id for signal in self.task.signals}
        for port in self.model.ports.inputs:
            default = port.name if port.name in signal_ids else None
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
        return self.resolved_signal_bindings({signal.name: signal.id for signal in original.signals})

    @model_validator(mode="after")
    def _one_task(self) -> "TrialDocuments":
        validate_cross_field_refinements("TrialDocuments", self)
        return self
