"""Preview authored trials from the same source used by the rollout."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_values import Level, Problem, ProblemCode, ValueBase
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.rollout import trial_indices
from feedbax.contracts.studio_refinements import validate_cross_field_refinements
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
from feedbax.contracts.trial_record import TrialDisplay, TrialRecords, TrialTerm, TrialUnavailable
from feedbax.execution.trial_execution import sample_evaluation_trials
from feedbax.execution.trial_record import aggregate_trials, trial_record
from feedbax.execution.trial_resolution import preview_trial_source
from feedbax.objectives.document import ObjectiveExecutor, ObjectiveUse, compile_objective
from feedbax.runtime._graph_key_schedule import map_trial_samples, sample_training_trials
from feedbax.tasks.document import TaskParameters

router = APIRouter()


class TaskTrialSampleRequest(StrictModel):
    model: ModelDocument | None = None
    task: TaskDocument | None = None
    python_task: PythonTaskSelection | None = None
    input_bindings: dict[str, str | None] = Field(default_factory=dict)
    objective: ObjectiveDocument | None = None
    evaluation_trials: list[str] | None = Field(default=None, min_length=1)
    """Evaluate trial IDs to preview with the same seed schedule as the run."""
    nominal: bool = False
    """Preview one representative Task trial independently of the sampler seed."""
    seed: int = 0
    count: int = Field(default=8, ge=1)
    """How many trials to draw; Studio offers up to the step's batch size."""
    aggregate: bool = False
    """Whether to return the trials' aggregate instead of the trials."""

    @model_validator(mode="before")
    @classmethod
    def _normalize_model_without_period(cls, value):
        if isinstance(value, dict) and isinstance(model := value.get("model"), dict):
            if model.get("step_seconds") is None:
                return value | {"model": None}
        return value

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


_DTYPES = {ValueBase.BOOLEAN: jnp.bool_, ValueBase.INTEGER: jnp.int32, ValueBase.REAL: jnp.float32}


@router.post("/task-trials/sample", response_model=TaskTrialSampleResponse)
async def sample_task_trials(
    payload: TaskTrialSampleRequest,
    request: Request,
) -> TaskTrialSampleResponse:
    bundle = request.app.state.bootstrap_state.bundle
    selected = payload.task if payload.task is not None else payload.python_task
    assert selected is not None
    task_id = selected.id if isinstance(selected, TaskDocument) else selected.type
    count = 1 if payload.nominal else (
        len(payload.evaluation_trials) if payload.evaluation_trials is not None else payload.count
    )
    seed = 0 if payload.nominal else payload.seed

    def unavailable(*problems: Problem):
        return TaskTrialSampleResponse(
            task=task_id, seed=seed, count=count, n_steps=None,
            drawn=TrialUnavailable(kind="problems", problems=list(problems)),
        )

    if payload.model is None:
        return unavailable(Problem(
            code=ProblemCode.STRUCTURE, path=("model", "step_seconds"),
            reason="A Model step length is needed to draw a trial.",
        ))
    preview = preview_trial_source(
        payload.model, selected, bundle.components, bundle.expression_functions,
        python_tasks=bundle.python_tasks, input_bindings=payload.input_bindings, nominal=payload.nominal,
    )
    if isinstance(preview, tuple):
        return unavailable(*preview)
    task = preview.task
    if task.reads_model:
        return unavailable(Problem(
            code=ProblemCode.STRUCTURE, path=("task",),
            reason="This task needs model outputs to produce a trial.",
        ))

    key = jax.random.PRNGKey(seed)
    parameters = TaskParameters()
    if payload.nominal or payload.evaluation_trials is None:
        ids = ("nominal",) if payload.nominal else tuple(f"trial:{index}" for index in range(count))
        samples, keys = sample_training_trials(lambda sample: (task.ahead(parameters, key, sample), sample), key, count)
    else:
        ids = tuple(payload.evaluation_trials)
        try:
            trial_indices(ids, task.condition_ids)
        except ValueError as error:
            return unavailable(Problem(
                code=ProblemCode.STRUCTURE, path=("evaluation_trials",), reason=str(error),
            ))
        samples, keys = sample_evaluation_trials(task, parameters, key=key, trial_ids=ids)

    term_traces = {}
    if payload.objective is not None:
        scorer = compile_objective(
            payload.objective, (*preview.outputs, *task.interface.variables, *task.interface.signals),
            bundle.expression_functions, task=task, use=ObjectiveUse.PREVIEW, nominal=payload.nominal,
        )
        assert isinstance(scorer, ObjectiveExecutor)
        outputs = {
            output.id: jnp.zeros((task.n_steps, *output.type.shape), _DTYPES[output.type.base])
            for output in scorer.names
            if output.level == Level.STEP and output in preview.outputs
        }
        term_traces = map_trial_samples(
            lambda item: scorer.trial_terms(outputs, item[0], key, item[1]), (samples, keys), keys,
        )

    samples, term_traces = jax.device_get((samples, term_traces))
    labels = {epoch.id: epoch.name for epoch in selected.epochs} if isinstance(selected, TaskDocument) else {}
    trials = [
        trial_record(
            jax.tree.map(lambda values: values[index], samples),
            id=id, epoch_ids=task.epochs, epoch_labels=labels,
            terms={name: TrialTerm(steps=trace.steps[index].tolist(), covered=bool(trace.covered[index]))
                   for name, trace in term_traces.items()},
        )
        for index, id in enumerate(ids)
    ]
    drawn = (aggregate_trials(trials, task.n_steps) if payload.aggregate and not payload.nominal
             else TrialRecords(kind="trials", trials=trials))
    return TaskTrialSampleResponse(task=task_id, seed=seed, count=count, n_steps=task.n_steps, drawn=drawn)
