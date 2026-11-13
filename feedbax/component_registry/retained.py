"""Frozen component owners and presentation for retained task graphs."""

from feedbax.component_migrations.tasks import HISTORIES
from feedbax.contracts.representation import RepresentationSpec
from feedbax.studio.component_presentation import PresentationOverride
from feedbax.tasks.reach_components import DelayedReaches, SimpleReaches

from .discovery import ComponentPack
from .registry import ComponentRegistry


def retained_component_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.ingest(
        ComponentPack(
            namespace="feedbax",
            components=tuple(HISTORIES),
            histories=HISTORIES,
            presentation={
                SimpleReaches: PresentationOverride(
                    representation=_reach_task_representation(delayed=False),
                ),
                DelayedReaches: PresentationOverride(
                    representation=_reach_task_representation(delayed=True),
                ),
            },
        ),
        provenance="feedbax",
    )
    return registry


def _reach_task_representation(*, delayed: bool) -> RepresentationSpec:
    from feedbax.studio.representation import (
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
