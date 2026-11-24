"""Preview authored trials from the same source used by the rollout."""

from __future__ import annotations

import jax
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from feedbax.execution.trial_resolution import PreparedTrialExecution, compile_trial_execution
from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.studio_refinements import validate_cross_field_refinements
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
from feedbax.contracts.trial_record import (
    TaskEpochShare, TaskSignalAggregate, TaskTimelineCue, TaskVariableRange,
    TrialAggregate, TrialDisplay, TrialRecord, TrialRecords,
)
from feedbax.runtime._graph_key_schedule import sample_training_trials
from feedbax.tasks.document import TaskParameters

router = APIRouter()


class TaskTrialSampleRequest(StrictModel):
    model: ModelDocument
    task: TaskDocument | None = None
    python_task: PythonTaskSelection | None = None
    input_overrides: dict[str, str] = Field(default_factory=dict)
    objective: ObjectiveDocument | None = None
    evaluation_trials: list[str] | None = Field(default=None, min_length=1)
    """Evaluate trial IDs to preview with the same seed schedule as the run."""
    seed: int = 0
    count: int = Field(default=8, ge=1)
    """How many trials to draw; Studio offers up to the step's batch size."""
    aggregate: bool = False
    """Whether to return the trials' aggregate instead of the trials."""

    @model_validator(mode="after")
    def _one_task(self):
        validate_cross_field_refinements("TaskTrialSampleRequest", self)
        return self


class TaskTrialSampleResponse(BaseModel):
    schema_version: str = "feedbax.execution.sampled_task_trials.v4"
    task: str
    seed: int
    count: int
    n_steps: int | None
    drawn: TrialDisplay


PERCENTILES = (5, 25, 50, 75, 95)


def aggregate_trials(epochs: list[str], labels: dict[str, str], samples, n_steps: int) -> TrialAggregate:
    """The batch `samples` as shares of trials per epoch and step, and its signals' presence and percentiles."""
    starts, ends, _ = (np.asarray(values) for values in samples.epochs)
    steps = np.arange(n_steps)
    inside: dict[str, np.ndarray] = {}
    for index, epoch in enumerate(epochs):
        covered = (steps >= starts[:, index, None]) & (steps < ends[:, index, None])
        inside[epoch] = inside[epoch] | covered if epoch in inside else covered
    signals = {}
    for name, values in samples.inputs.items():
        values = np.asarray(values)
        present = np.asarray(samples.present[name]).astype(bool)
        percentiles = None
        if np.issubdtype(values.dtype, np.number):
            flat = values.reshape(*values.shape[:2], -1).astype(float)
            masked = np.where(present[..., None], flat, np.nan)
            any_present = present.any(axis=0)
            with np.errstate(all="ignore"):
                quantiles = np.nanpercentile(masked[:, any_present], PERCENTILES, axis=0) if any_present.any() else None
            at = iter(np.moveaxis(quantiles, 0, -1).tolist() if quantiles is not None else ())
            percentiles = [next(at) if here else None for here in any_present]
        signals[name] = TaskSignalAggregate(presence=present.mean(axis=0).tolist(), percentiles=percentiles)
    return TrialAggregate(
        kind="aggregate",
        epochs=[TaskEpochShare(epoch=epoch, label=labels.get(epoch, epoch), share=covered.mean(axis=0).tolist())
                for epoch, covered in inside.items()],
        signals=signals,
        variables={
            name: TaskVariableRange(least=np.min(values, axis=0).tolist(), greatest=np.max(values, axis=0).tolist())
            for name, values in ((name, np.asarray(values)) for name, values in samples.variables.items())
        },
    )


@router.post("/task-trials/sample", response_model=TaskTrialSampleResponse)
async def sample_task_trials(
    payload: TaskTrialSampleRequest,
    request: Request,
) -> TaskTrialSampleResponse:
    bundle = request.app.state.bootstrap_state.bundle
    selected = payload.task if payload.task is not None else payload.python_task
    assert selected is not None
    paired = compile_trial_execution(
        payload.model,
        selected,
        bundle.components,
        bundle.expression_functions,
        python_tasks=bundle.python_tasks,
        input_overrides=payload.input_overrides,
    )
    if not isinstance(paired, PreparedTrialExecution):
        raise HTTPException(
            status_code=422, detail=[item.model_dump(mode="json") for item in paired]
        )
    task = paired.execution.task
    if task.reads_model:
        raise HTTPException(
            status_code=422, detail="This task needs model outputs to produce a trial."
        )

    key = jax.random.PRNGKey(payload.seed)
    samples = jax.device_get(
        sample_training_trials(
            lambda trial_key: task.ahead(TaskParameters(), key, trial_key),
            key,
            payload.count,
        )
    )
    labels = {epoch.id: epoch.name for epoch in selected.epochs} if isinstance(selected, TaskDocument) else {}
    head = dict(task=selected.id if isinstance(selected, TaskDocument) else selected.type,
                seed=payload.seed, count=payload.count, n_steps=task.n_steps)
    if payload.aggregate:
        return TaskTrialSampleResponse(**head, drawn=aggregate_trials(list(task.epochs), labels, samples, task.n_steps))
    trials = []
    for index in range(payload.count):
        trial = jax.tree.map(lambda values: values[index], samples)
        timeline = [
            TaskTimelineCue(
                epoch=epoch,
                label=labels.get(epoch, epoch),
                start=int(start),
                end=int(end),
                outcome=int(outcome),
            )
            for epoch, start, end, outcome in zip(
                task.epochs,
                *trial.epochs,
                strict=True,
            )
        ]
        trials.append(
            TrialRecord(
                id=f"trial:{index}",
                inits={key: value.tolist() for key, value in trial.inits.items()},
                inputs={key: value.tolist() for key, value in trial.inputs.items()},
                present={key: value.tolist() for key, value in trial.present.items()},
                variables={key: value.tolist() for key, value in trial.variables.items()},
                variable_defined={
                    key: value.tolist() for key, value in (trial.variable_defined or {}).items()
                },
                timeline=timeline,
            )
        )
    return TaskTrialSampleResponse(**head, drawn=TrialRecords(kind="trials", trials=trials))
