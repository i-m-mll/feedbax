"""Training documents and optimizer settings for registered methods."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.studio_refinements import validate_cross_field_refinements
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument


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

    @model_validator(mode="after")
    def _one_task(self) -> "TrialDocuments":
        validate_cross_field_refinements("TrialDocuments", self)
        return self
