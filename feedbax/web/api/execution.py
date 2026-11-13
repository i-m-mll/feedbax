"""Preview authored trials from the same source used by the rollout."""

from __future__ import annotations

from typing import Any

import jax
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_refinements import validate_cross_field_refinements
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
from feedbax.runtime._graph_key_schedule import sample_training_trials
from feedbax.tasks.document import TaskParameters

router = APIRouter()


class SimulationRequest(BaseModel):
    graph: GraphSpec
    n_steps: int
    inputs: dict[str, Any] | None = None


@router.post("/simulate")
async def simulate(_: SimulationRequest):
    raise HTTPException(status_code=501, detail="Simulation not implemented yet")


class TaskTrialSampleRequest(BaseModel):
    model_name: ProjectName
    model: ModelDocument
    task: TaskDocument | None = None
    python_task: PythonTaskSelection | None = None
    input_overrides: dict[str, str] = Field(default_factory=dict)
    seed: int = 0
    count: int = Field(default=8, ge=1, le=64)

    @model_validator(mode="after")
    def _one_task(self):
        validate_cross_field_refinements("TaskTrialSampleRequest", self)
        return self


class TaskTimelineCue(BaseModel):
    epoch: str
    label: str
    start: int
    end: int
    outcome: int


class SampledTaskTrial(BaseModel):
    id: str
    inits: dict[str, Any]
    inputs: dict[str, Any]
    present: dict[str, Any]
    variables: dict[str, Any]
    variable_defined: dict[str, Any]
    timeline: list[TaskTimelineCue]


class TaskTrialSampleResponse(BaseModel):
    schema_version: str = "feedbax.execution.sampled_task_trials.v2"
    task: str
    seed: int
    n_steps: int
    trials: list[SampledTaskTrial]


@router.post("/task-trials/sample", response_model=TaskTrialSampleResponse)
async def sample_task_trials(
    payload: TaskTrialSampleRequest,
    request: Request,
) -> TaskTrialSampleResponse:
    bundle = request.app.state.bootstrap_state.bundle
    selected = payload.task if payload.task is not None else payload.python_task
    assert selected is not None
    paired = compile_trial_execution(
        payload.model_name,
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
            SampledTaskTrial(
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
    return TaskTrialSampleResponse(
        task=selected.id if isinstance(selected, TaskDocument) else selected.type,
        seed=payload.seed, n_steps=task.n_steps, trials=trials,
    )
