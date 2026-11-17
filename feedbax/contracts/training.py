"""Training documents and optimizer settings for registered methods."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.graph import ParamValue
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.studio_refinements import validate_cross_field_refinements
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument


class LrScheduleSpec(StrictModel):
    """Schema-versioned learning-rate schedule specification.

    Attributes:
        kind: Schedule family. ``"constant"`` holds ``learning_rate_0``;
            ``"warmup_cosine"`` linearly warms from
            ``warmup_init_fraction * learning_rate_0`` over
            ``constant_lr_iterations`` steps, then cosine-anneals to
            ``cosine_annealing_alpha * learning_rate_0`` at ``total_steps``;
            ``"delayed_cosine"`` holds ``learning_rate_0`` for
            ``constant_lr_iterations`` steps, then cosine-anneals to the same
            terminal fraction at ``total_steps``.
        learning_rate_0: Peak learning rate. This maps from rlrmp
            ``controller_lr``.
        total_steps: Origin-relative step at which cosine schedules reach
            their terminal value. Cosine schedules hold that terminal value at
            every later applied coordinate. This maps from rlrmp
            ``n_batches_condition``.
        constant_lr_iterations: Warmup length for ``"warmup_cosine"`` and
            constant-prefix length for ``"delayed_cosine"``. This maps from
            rlrmp ``lr_warmup_batches``.
        warmup_init_fraction: Initial learning-rate fraction for
            ``"warmup_cosine"``. This maps from rlrmp
            ``lr_warmup_init_fraction``.
        cosine_annealing_alpha: Terminal learning-rate fraction for cosine
            schedules. This maps from rlrmp ``lr_cosine_alpha``.
    """

    kind: Literal["constant", "warmup_cosine", "delayed_cosine"] = "constant"
    learning_rate_0: float = Field(gt=0.0, allow_inf_nan=False)
    total_steps: int | None = Field(default=None, gt=0)
    constant_lr_iterations: int = Field(default=0, ge=0)
    warmup_init_fraction: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    cosine_annealing_alpha: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_schedule_shape(self) -> "LrScheduleSpec":
        if self.kind == "constant":
            return self
        if self.total_steps is None:
            raise ValueError(f"/total_steps is required when lr_schedule.kind={self.kind!r}")
        if self.kind == "warmup_cosine":
            if self.constant_lr_iterations < 1:
                raise ValueError("/constant_lr_iterations must be >= 1 for warmup_cosine schedules")
            if self.constant_lr_iterations >= self.total_steps:
                raise ValueError("/constant_lr_iterations must be < /total_steps for warmup_cosine")
        if self.kind == "delayed_cosine" and self.constant_lr_iterations >= self.total_steps:
            raise ValueError("/constant_lr_iterations must be < /total_steps for delayed_cosine")
        return self


class OptimizerSpec(StrictModel):
    """Specification for an optimizer.

    Attributes:
        type: Optimizer factory name. The public builder supports ``"adamw"``,
            ``"adam"``, ``"sgd"``, and ``"rmsprop"``.
        params: Static optimizer parameters other than scheduled learning rate.
        lr_schedule: Optional learning-rate schedule evaluated on the Train clock.
    """

    type: str
    params: dict[str, ParamValue] = Field(default_factory=dict)
    lr_schedule: LrScheduleSpec | None = None


class TrialDocuments(StrictModel):
    model_name: str
    model: ModelDocument
    task: TaskDocument | None
    objective: ObjectiveDocument
    input_overrides: dict[str, str] = Field(default_factory=dict)
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
