"""Portable owners for built-in reach and stabilization tasks."""

from __future__ import annotations
from typing import Annotated, Any, Literal
import jax.numpy as jnp
import jax.random as jr
from pydantic import Field
from feedbax.contracts.component import PortType
from feedbax.contracts.portable import Ports, ConfigBinding, configuration_model
from feedbax.runtime.graph import Component
from feedbax.objectives.loss import CompositeLoss
from feedbax.tasks.task import (
    TaskComponent,
    SimpleReaches as SimpleReachTask,
    DelayedReaches as DelayedReachTask,
    Stabilization as StabilizationTask,
)
from feedbax._delayed_reaches import (
    DEFAULT_DELAYED_N_STEPS,
    DEFAULT_DELAYED_EPOCH_LEN_RANGES,
    DEFAULT_DELAYED_MOVE_EPOCHS,
    delayed_reaches_n_steps_from_params,
    apply_delayed_reaches_preset,
)

_TASK_PORTS = Ports(
    inputs={},
    outputs={
        "inputs": PortType(dtype="any"),
        "targets": PortType(dtype="state"),
        "inits": PortType(dtype="state"),
    },
)
Workspace = Annotated[Any, Field(json_schema_extra={"feedbax": {"type": "bounds2d"}})]
ArrayConfig = Annotated[Any, Field(json_schema_extra={"feedbax": {"type": "array"}})]
PositiveInt = Annotated[int, Field(ge=1)]


def _initialize_task(owner, task):
    TaskComponent.__init__(
        owner, task=task, trial_spec=task.sample_trial(key=jr.PRNGKey(0)), mode="open_loop"
    )


class SimpleReaches(TaskComponent, Component):
    """Random reach endpoints in a workspace."""

    ports = _TASK_PORTS

    def __init__(
        self,
        n_steps: PositiveInt = 200,
        workspace: Workspace = ((-1.0, -1.0), (1.0, 1.0)),
        eval_n_directions: PositiveInt = 7,
        eval_reach_length: float = 0.5,
        eval_grid_n: PositiveInt = 1,
    ):
        _initialize_task(
            self,
            SimpleReachTask(
                loss_func=CompositeLoss(()),
                n_steps=n_steps,
                workspace=jnp.asarray(workspace),
                eval_n_directions=eval_n_directions,
                eval_reach_length=eval_reach_length,
                eval_grid_n=eval_grid_n,
            ),
        )


class Stabilization(TaskComponent, Component):
    """Hold position against perturbations."""

    ports = _TASK_PORTS

    def __init__(
        self, n_steps: PositiveInt = 200, workspace: Workspace = ((-1.0, -1.0), (1.0, 1.0))
    ):
        _initialize_task(
            self,
            StabilizationTask(
                loss_func=CompositeLoss(()), n_steps=n_steps, workspace=jnp.asarray(workspace)
            ),
        )


class DelayedReaches(TaskComponent, Component):
    """Reaches with a delay period before movement."""

    ports = _TASK_PORTS

    def __init__(
        self,
        n_steps: PositiveInt = DEFAULT_DELAYED_N_STEPS,
        n_control_stages: PositiveInt | None = None,
        workspace: Workspace = ((-1.0, -1.0), (1.0, 1.0)),
        preset: Literal["default", "delayed_center_out"] = "default",
        train_endpoint_mode: Literal["workspace", "center_out"] = "workspace",
        epoch_len_ranges: ArrayConfig = DEFAULT_DELAYED_EPOCH_LEN_RANGES,
        epoch_names: ArrayConfig = ("hold", "target_on", "movement"),
        target_on_epochs: ArrayConfig = (1, 2),
        hold_epochs: ArrayConfig = (0, 1),
        move_epochs: ArrayConfig = DEFAULT_DELAYED_MOVE_EPOCHS,
        target_visible_from_start: bool = False,
        go_cue_event_name: str | None = None,
        p_catch_trial: Annotated[float, Field(ge=0, le=1)] = 0.5,
        catch_metadata_policy: Literal["none", "flag"] = "none",
        eval_n_directions: PositiveInt = 7,
        eval_reach_length: float = 0.5,
        eval_grid_n: PositiveInt = 1,
    ):
        params = {name: value for name, value in locals().items() if name != "self"}
        params = apply_delayed_reaches_preset(params) if preset == "delayed_center_out" else params
        params["n_steps"] = delayed_reaches_n_steps_from_params(params)
        params.pop("n_control_stages", None)
        params["workspace"] = jnp.asarray(params["workspace"])
        _initialize_task(self, DelayedReachTask(loss_func=CompositeLoss(()), **params))

    @property
    def n_control_stages(self):
        return int(self.task.n_steps) - 1


for _owner in (SimpleReaches, Stabilization, DelayedReaches):
    _owner.config_bindings = tuple(
        ConfigBinding(path=f"task.{name}", name=name)
        for name in configuration_model(_owner).model_fields
        if name != "n_control_stages"
    )
