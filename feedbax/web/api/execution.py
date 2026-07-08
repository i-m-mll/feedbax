from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.graphs.builders import build_component
from feedbax.contracts.training import TaskSpec
from feedbax.tasks import TaskTrialSpec

router = APIRouter()


class SimulationRequest(BaseModel):
    graph: GraphSpec
    n_steps: int
    inputs: Optional[dict[str, Any]] = None


@router.post('/simulate')
async def simulate(_: SimulationRequest):
    raise HTTPException(status_code=501, detail='Simulation not implemented yet')


class TaskTrialSampleRequest(BaseModel):
    task_spec: TaskSpec
    seed: int = 0
    count: int = 8


class TaskTimelineCue(BaseModel):
    label: str
    step: int
    kind: str


class SampledTaskTrial(BaseModel):
    id: str
    index: int
    start: list[float]
    goal: list[float]
    n_steps: int
    timeline: list[TaskTimelineCue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskTrialSampleResponse(BaseModel):
    schema_version: str = "feedbax.execution.sampled_task_trials.v1"
    task_type: str
    seed: int
    count: int
    trials: list[SampledTaskTrial]


def _normal_task_type(task_type: str) -> str:
    if task_type in {"ReachingTask", "feedbax.task.ReachingTask", "feedbax.task.SimpleReaches"}:
        return "SimpleReaches"
    if task_type == "feedbax.task.DelayedReaches":
        return "DelayedReaches"
    return task_type


def _as_pair(value: Any, *, path: str) -> list[float]:
    array = np.asarray(value, dtype=float)
    if array.shape[-1:] != (2,):
        raise ValueError(f"{path} must end in a 2D position, got shape {array.shape}")
    point = array.reshape((-1, 2))[-1]
    return [float(point[0]), float(point[1])]


def _timeline_cues(trial: TaskTrialSpec) -> list[TaskTimelineCue]:
    timeline = trial.timeline
    cues: list[TaskTimelineCue] = []
    bounds = np.asarray(timeline.epoch_bounds)
    if bounds.size and bounds.ndim >= 1:
        flat_bounds = bounds.reshape((-1, bounds.shape[-1]))[0]
        names = list(timeline.epoch_names)
        for index, step in enumerate(flat_bounds[:-1]):
            label = names[index] if index < len(names) else f"epoch {index + 1}"
            cues.append(TaskTimelineCue(label=label, step=int(step), kind="epoch"))
    event_steps = timeline.event_steps
    if event_steps is not None:
        flat_steps = np.asarray(event_steps).reshape((-1,))
        names = list(timeline.event_names)
        for index, step in enumerate(flat_steps):
            label = names[index] if index < len(names) else f"event {index + 1}"
            cues.append(TaskTimelineCue(label=label, step=int(step), kind="event"))
    return cues


def _sampled_trial_payload(trial: TaskTrialSpec, *, index: int) -> SampledTaskTrial:
    try:
        start = _as_pair(trial.inits["mechanics.effector"].pos, path="inits.mechanics.effector.pos")
        goal = _as_pair(
            trial.targets["mechanics.effector.pos"].value,
            path="targets.mechanics.effector.pos.value",
        )
    except KeyError as exc:
        raise ValueError(
            "Task trial does not expose mechanics.effector start/goal positions for preview"
        ) from exc

    metadata: dict[str, Any] = {}
    if trial.extra:
        for key, value in trial.extra.items():
            array = np.asarray(value)
            metadata[key] = array.item() if array.shape == () else array.tolist()

    return SampledTaskTrial(
        id=f"trial:{index}",
        index=index,
        start=start,
        goal=goal,
        n_steps=int(trial.timeline.n_steps),
        timeline=_timeline_cues(trial),
        metadata=metadata,
    )


@router.post("/task-trials/sample", response_model=TaskTrialSampleResponse)
async def sample_task_trials(payload: TaskTrialSampleRequest) -> TaskTrialSampleResponse:
    count = int(payload.count)
    if count < 1 or count > 64:
        raise HTTPException(status_code=422, detail="count must be between 1 and 64")

    task_type = _normal_task_type(payload.task_spec.type)
    if task_type not in {"SimpleReaches", "DelayedReaches"}:
        raise HTTPException(
            status_code=422,
            detail=f"Task type {payload.task_spec.type!r} is not supported for sampled preview",
        )

    try:
        component = build_component("preview_task", task_type, payload.task_spec.params)
    except (NotImplementedError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    task = getattr(component, "task", None)
    if task is None or not hasattr(task, "sample_trial"):
        raise HTTPException(
            status_code=422,
            detail=f"Task type {payload.task_spec.type!r} cannot sample preview trials",
        )

    keys = jax.random.split(jax.random.PRNGKey(int(payload.seed)), count)
    trials: list[SampledTaskTrial] = []
    for index, key in enumerate(keys):
        try:
            trial = task.sample_trial(jnp.asarray(key))
            trials.append(_sampled_trial_payload(trial, index=index))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return TaskTrialSampleResponse(
        task_type=task_type,
        seed=int(payload.seed),
        count=count,
        trials=trials,
    )
