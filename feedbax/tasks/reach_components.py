"""Graph component factories for the built-in reach and stabilization tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import jax.random as jr

from feedbax._delayed_reaches import (
    DEFAULT_DELAYED_EPOCH_LEN_RANGES,
    DEFAULT_DELAYED_MOVE_EPOCHS,
    DEFAULT_DELAYED_N_STEPS,
    apply_delayed_reaches_preset,
    delayed_reaches_n_steps_from_params,
)
from feedbax.component_registry.declarations import (
    DeclaredComponent,
    ParameterField,
    declare_component,
)
from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.representation import RepresentationSpec
from feedbax.objectives.loss import CompositeLoss
from feedbax.tasks.task import (
    AbstractTask,
    DelayedReaches,
    SimpleReaches,
    Stabilization,
    TaskComponent,
)


def _open_loop_task(task: AbstractTask, params: Mapping[str, Any]) -> TaskComponent:
    if params.get("mode", "open_loop") != "open_loop":
        raise ValueError("Only open_loop TaskComponent is supported from GraphSpec")
    trial_spec = task.sample_trial(key=jr.PRNGKey(0))
    return TaskComponent(task=task, trial_spec=trial_spec, mode="open_loop")


def _build_simple_reaches(params: Mapping[str, Any]) -> TaskComponent:
    task = SimpleReaches(
        loss_func=CompositeLoss(()),
        n_steps=int(params.get("n_steps", 200)),
        workspace=jnp.asarray(params.get("workspace", [[-1.0, -1.0], [1.0, 1.0]])),
        eval_n_directions=int(params.get("eval_n_directions", 7)),
        eval_reach_length=float(params.get("eval_reach_length", 0.5)),
        eval_grid_n=int(params.get("eval_grid_n", 1)),
    )
    return _open_loop_task(task, params)


def _build_delayed_reaches(params: Mapping[str, Any]) -> TaskComponent:
    task = DelayedReaches(
        loss_func=CompositeLoss(()),
        n_steps=delayed_reaches_n_steps_from_params(params),
        workspace=jnp.asarray(params.get("workspace", [[-1.0, -1.0], [1.0, 1.0]])),
        preset=params.get("preset", None),
        train_endpoint_mode=str(params.get("train_endpoint_mode", "workspace")),
        epoch_len_ranges=params.get("epoch_len_ranges", DEFAULT_DELAYED_EPOCH_LEN_RANGES),
        epoch_names=tuple(str(value) for value in params.get("epoch_names", []))
        or ("hold", "target_on", "movement"),
        target_on_epochs=tuple(int(value) for value in params.get("target_on_epochs", [1, 2])),
        hold_epochs=tuple(int(value) for value in params.get("hold_epochs", [0, 1])),
        move_epochs=params.get("move_epochs", DEFAULT_DELAYED_MOVE_EPOCHS),
        p_catch_trial=float(params.get("p_catch_trial", 0.5)),
        target_visible_from_start=bool(params.get("target_visible_from_start", False)),
        go_cue_event_name=params.get("go_cue_event_name", None),
        catch_metadata_policy=str(params.get("catch_metadata_policy", "none")),
        eval_n_directions=int(params.get("eval_n_directions", 7)),
        eval_reach_length=float(params.get("eval_reach_length", 0.5)),
        eval_grid_n=int(params.get("eval_grid_n", 1)),
    )
    return _open_loop_task(task, params)


def _build_stabilization(params: Mapping[str, Any]) -> TaskComponent:
    task = Stabilization(
        loss_func=CompositeLoss(()),
        n_steps=int(params.get("n_steps", 200)),
        workspace=jnp.asarray(params.get("workspace", [[-1.0, -1.0], [1.0, 1.0]])),
    )
    return _open_loop_task(task, params)


def _project_delayed_reaches_params(params: Mapping[str, Any]) -> dict[str, Any]:
    if params.get("preset") == "delayed_center_out":
        return apply_delayed_reaches_preset(params)
    return dict(params)


def _extract_delayed_reaches_params(component: Any, model: Any) -> Any:
    task = component.task
    return type(model).model_validate(
        {
            "n_steps": int(task.n_steps),
            "n_control_stages": int(task.n_steps) - 1,
            "workspace": jnp.asarray(task.workspace).tolist(),
            "preset": task.preset,
            "train_endpoint_mode": task.train_endpoint_mode,
            "epoch_len_ranges": [list(item) for item in task.epoch_len_ranges],
            "epoch_names": list(task.epoch_names),
            "target_on_epochs": jnp.asarray(task.target_on_epochs).tolist(),
            "hold_epochs": jnp.asarray(task.hold_epochs).tolist(),
            "move_epochs": jnp.asarray(task.move_epochs).tolist(),
            "target_visible_from_start": bool(task.target_visible_from_start),
            "go_cue_event_name": task.go_cue_event_name,
            "p_catch_trial": float(task.p_catch_trial),
            "catch_metadata_policy": task.catch_metadata_policy,
            "eval_n_directions": int(task.eval_n_directions),
            "eval_reach_length": float(task.eval_reach_length),
            "eval_grid_n": int(task.eval_grid_n),
        }
    )


def _reach_task_representation(*, delayed: bool) -> RepresentationSpec:
    from feedbax.component_registry.representation import (
        _param_binding,
        _representation,
        _trial_binding,
    )

    temporality = (
        {
            "kind": "scheduled",
            "target_on_epochs_param": "target_on_epochs",
            "move_epochs_param": "move_epochs",
            "target_visible_from_start_param": "target_visible_from_start",
        }
        if delayed
        else {"kind": "static"}
    )
    endpoint_mode = (
        {"kind": "param", "param": "train_endpoint_mode", "center_out_value": "center_out"}
        if delayed
        else {"kind": "validation_center_out", "training": "workspace_uniform"}
    )
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "metadata": {
                "schematic": True,
                "temporality": temporality,
                "canonical_goal_anchor": "goal",
                "endpoint_mode": endpoint_mode,
            },
            "anchors": [
                {
                    "id": "start",
                    "semantic_role": "origin",
                    "interaction_roles": ["selectable", "hoverable"],
                    "label": "Start",
                    "binding": _trial_binding("inits.effector.pos", dim=2),
                    "units": "m",
                    "dim": 2,
                    "metadata": {"task_role": "initial_effector_position"},
                },
                {
                    "id": "goal",
                    "semantic_role": "target",
                    "interaction_roles": ["selectable", "hoverable", "editable"],
                    "label": "Goal",
                    "binding": _trial_binding("targets.effector.pos", dim=2),
                    "units": "m",
                    "dim": 2,
                    "metadata": {
                        "canonical_goal": True,
                        "task_role": "effector_target",
                    },
                },
            ],
            "elements": [
                {
                    "id": "workspace",
                    "archetype": "region",
                    "bindings": {
                        "bounds": _param_binding("workspace", expected_type="bounds2d", dim=2)
                    },
                    "style": [
                        {"channel": "fill", "value": "var(--workspace-region-fill)"},
                        {"channel": "opacity", "value": 0.08},
                    ],
                    "metadata": {"region_kind": "workspace_bounds"},
                },
                {
                    "id": "start-marker",
                    "archetype": "marker",
                    "anchors": ["start"],
                    "style": [{"channel": "glyph", "value": "start"}],
                },
                {
                    "id": "goal-marker",
                    "archetype": "marker",
                    "anchors": ["goal"],
                    "style": [{"channel": "glyph", "value": "target"}],
                    "metadata": {"canonical_goal": True},
                },
                {
                    "id": "reach-distribution",
                    "archetype": "distribution_glyph",
                    "anchors": ["start", "goal"],
                    "bindings": {
                        "workspace": _param_binding("workspace", expected_type="bounds2d", dim=2),
                        "eval_reach_length": _param_binding(
                            "eval_reach_length",
                            expected_type="float",
                        ),
                        "eval_n_directions": _param_binding(
                            "eval_n_directions",
                            expected_type="int",
                        ),
                        "eval_grid_n": _param_binding("eval_grid_n", expected_type="int"),
                    },
                    "metadata": {
                        "distribution": "center_out" if delayed else "workspace_uniform",
                        "canonical_objective": "goal",
                    },
                },
                {
                    "id": "objective",
                    "archetype": "objective_link",
                    "anchors": ["start", "goal"],
                    "metadata": {
                        "canonical_goal": "goal",
                        "objective": "effector_position",
                    },
                },
            ],
        }
    )


def simple_reaches_definition() -> DeclaredComponent:
    return declare_component(
        name="SimpleReaches",
        builder=_build_simple_reaches,
        override_reason="constructor aggregates task, trial, loss, and workspace objects",
        attribute_paths={
            "n_steps": "task.n_steps",
            "workspace": "task.workspace",
            "eval_n_directions": "task.eval_n_directions",
            "eval_reach_length": "task.eval_reach_length",
            "eval_grid_n": "task.eval_grid_n",
        },
        category="Tasks",
        description="Random reach endpoints in a workspace.",
        parameter_fields=[
            ParameterField(name="n_steps", type="int", default=200, min=1, required=True),
            ParameterField(
                name="workspace",
                type="bounds2d",
                default=[[-1.0, -1.0], [1.0, 1.0]],
                required=True,
            ),
            ParameterField(name="eval_n_directions", type="int", default=7, min=1, required=False),
            ParameterField(name="eval_reach_length", type="float", default=0.5, required=False),
            ParameterField(name="eval_grid_n", type="int", default=1, min=1, required=False),
        ],
        input_ports=TaskComponent.input_ports,
        output_ports=TaskComponent.output_ports,
        icon="Target",
        port_types=PortTypeSpec(
            inputs={},
            outputs={
                "inputs": PortType(dtype="any"),
                "targets": PortType(dtype="state"),
                "inits": PortType(dtype="state"),
            },
        ),
        representation=_reach_task_representation(delayed=False),
    )


def delayed_reaches_definition() -> DeclaredComponent:
    return declare_component(
        name="DelayedReaches",
        builder=_build_delayed_reaches,
        extractor=_extract_delayed_reaches_params,
        param_projection=_project_delayed_reaches_params,
        override_reason="n_control_stages is transformed into the task step count",
        category="Tasks",
        description="Reaches with a delay period before movement.",
        parameter_fields=[
            ParameterField(
                name="n_steps",
                type="int",
                default=DEFAULT_DELAYED_N_STEPS,
                min=1,
                required=False,
            ),
            ParameterField(
                name="n_control_stages",
                type="int",
                default=None,
                min=1,
                required=False,
            ),
            ParameterField(
                name="workspace",
                type="bounds2d",
                default=[[-1.0, -1.0], [1.0, 1.0]],
                required=True,
            ),
            ParameterField(
                name="preset",
                type="enum",
                options=["default", "delayed_center_out"],
                default="default",
                required=False,
            ),
            ParameterField(
                name="train_endpoint_mode",
                type="enum",
                options=["workspace", "center_out"],
                default="workspace",
                required=False,
            ),
            ParameterField(
                name="epoch_len_ranges",
                type="array",
                default=[list(pair) for pair in DEFAULT_DELAYED_EPOCH_LEN_RANGES],
                required=False,
            ),
            ParameterField(
                name="epoch_names",
                type="array",
                default=["hold", "target_on", "movement"],
                required=False,
            ),
            ParameterField(name="target_on_epochs", type="array", default=[1, 2], required=False),
            ParameterField(name="hold_epochs", type="array", default=[0, 1], required=False),
            ParameterField(
                name="move_epochs",
                type="array",
                default=list(DEFAULT_DELAYED_MOVE_EPOCHS),
                required=False,
            ),
            ParameterField(
                name="target_visible_from_start",
                type="bool",
                default=False,
                required=False,
            ),
            ParameterField(name="go_cue_event_name", type="str", default=None, required=False),
            ParameterField(
                name="p_catch_trial",
                type="float",
                default=0.5,
                min=0.0,
                max=1.0,
                required=False,
            ),
            ParameterField(
                name="catch_metadata_policy",
                type="enum",
                options=["none", "flag"],
                default="none",
                required=False,
            ),
            ParameterField(name="eval_n_directions", type="int", default=7, min=1, required=False),
            ParameterField(name="eval_reach_length", type="float", default=0.5, required=False),
            ParameterField(name="eval_grid_n", type="int", default=1, min=1, required=False),
        ],
        input_ports=TaskComponent.input_ports,
        output_ports=TaskComponent.output_ports,
        icon="Timer",
        port_types=PortTypeSpec(
            inputs={},
            outputs={
                "inputs": PortType(dtype="any"),
                "targets": PortType(dtype="state"),
                "inits": PortType(dtype="state"),
            },
        ),
        representation=_reach_task_representation(delayed=True),
    )


def stabilization_definition() -> DeclaredComponent:
    return declare_component(
        name="Stabilization",
        builder=_build_stabilization,
        override_reason="constructor aggregates task, trial, loss, and workspace objects",
        attribute_paths={"n_steps": "task.n_steps", "workspace": "task.workspace"},
        category="Tasks",
        description="Hold position against perturbations.",
        parameter_fields=[
            ParameterField(name="n_steps", type="int", default=200, min=1, required=True),
            ParameterField(
                name="workspace",
                type="bounds2d",
                default=[[-1.0, -1.0], [1.0, 1.0]],
                required=True,
            ),
        ],
        input_ports=TaskComponent.input_ports,
        output_ports=TaskComponent.output_ports,
        icon="Anchor",
        port_types=PortTypeSpec(
            inputs={},
            outputs={
                "inputs": PortType(dtype="any"),
                "targets": PortType(dtype="state"),
                "inits": PortType(dtype="state"),
            },
        ),
    )
