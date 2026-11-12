"""Resolve the qualified saved Evaluate diff against its authenticated source."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from pydantic import ValidationError

from feedbax.contracts.graph import ComponentSpec, GraphSpec, StudioEpochValueSpec
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import apply_override_patch_in_place
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.training import TaskSpec
from feedbax.studio.canonical_evaluate import CannotRun


@dataclass(frozen=True)
class EvaluationVariant:
    graph: GraphSpec
    task: TaskSpec
    source_overrides: dict[str, float]


def _finite_float32(value: object) -> bool:
    limit = 3.4028234663852886e38
    return type(value) in (int, float) and -limit <= value <= limit


def _fixed_reach_target(entry: StudioEpochValueSpec) -> bool:
    value = entry.value_spec
    return (
        entry.target_id == "reach_target"
        and value.mode == "constant" and value.value_form == "literal"
        and value.variation.scope == "fixed"
        and value.dtype in {None, "float32"}
        and value.units in {None, "m"} and value.frame in {None, "world"}
        and isinstance(value.value, list) and len(value.value) == 2
        and all(_finite_float32(component) for component in value.value)
    )


def _qualify_task(source: TaskSpec, task: TaskSpec, path: str) -> None:
    if task == source:
        return
    original, candidate = source.timeline, task.timeline
    if (
        task.type != source.type or task.params != source.params
        or original is None or candidate is None
        or [(item.id, item.index, item.length) for item in candidate.epochs]
        != [(item.id, item.index, item.length) for item in original.epochs]
        or candidate.signals != original.signals
    ):
        raise CannotRun(path, "task replacement must preserve the horizon, sampled epochs and signal interface")
    changed = [
        item for left, right in (
            (original.epoch_value_specs, candidate.epoch_value_specs),
            (candidate.epoch_value_specs, original.epoch_value_specs),
        )
        for item in left if item not in right
    ]
    if not all(_fixed_reach_target(item) for item in changed):
        raise CannotRun(
            f"{path}/timeline/epoch_value_specs",
            "task replacement requires fixed reach_target epoch values in world Cartesian metres",
        )


def resolve_evaluation_variant(
    graph: GraphSpec, task: TaskSpec, patches: Sequence[OverridePatch],
) -> EvaluationVariant:
    """Apply the existing project patch grammar with a bounded compatibility policy."""
    payload = {"graph": graph.model_dump(mode="json"), "task": task.model_dump(mode="json")}
    overrides: dict[str, float] = {}
    task_path: str | None = None
    for index, patch in enumerate(patches):
        path = f"/params/variation/{index}"
        parts = patch.path.split(".")
        if len(parts) == 4 and parts[0] == "nodes" and parts[2:] == ["params", "mass"]:
            # The signed observations and raw-state candidates authored mass relative to the graph.
            parts.insert(0, "graph")
        try:
            match parts, patch.op:
                case ["task"], "replace":
                    task_path = f"{path}/value"
                case ["graph", "wires"], "replace":
                    pass
                case ["graph", "nodes", name], "add":
                    if name in payload["graph"]["nodes"]:
                        raise ValueError("cannot replace a retained owner with an addition")
                    component = ComponentSpec.model_validate(patch.value)
                    scale = component.params.get("scale")
                    if (
                        component.type != "feedbax.Scale" or component.params.get("fixed") is not True
                        or not _finite_float32(scale)
                    ):
                        raise ValueError("addition requires a registered fixed finite scalar Scale")
                case ["graph", "nodes", name, "params", parameter], "replace":
                    owner = graph.nodes.get(name)
                    if owner is None:
                        raise ValueError("parameter patch must name a retained owner")
                    if owner.type == "feedbax.PointMass" and parameter == "mass":
                        value = patch.value
                        if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                            raise ValueError("requires a positive finite PointMass mass")
                        overrides[semantic_key("model", "value", (name,), "mass")] = float(value)
                    elif (owner.type, parameter) not in {
                        ("feedbax.GRU", "hidden_size"), ("feedbax.GRU", "dtype"),
                        ("feedbax.Linear", "input_size"), ("feedbax.Linear", "dtype"),
                    }:
                        raise ValueError("parameter has no qualified Evaluate override")
                case _:
                    raise ValueError("patch is outside the qualified task and graph variation profile")
            effective_patch = patch.model_copy(update={"path": ".".join(parts)})
            apply_override_patch_in_place(payload, effective_patch)
        except (ValueError, KeyError, OverflowError) as error:
            raise CannotRun(path, str(error)) from error

    try:
        variant, replacement = GraphSpec.model_validate(payload["graph"]), TaskSpec.model_validate(payload["task"])
    except ValidationError as error:
        first = error.errors()[0]
        raise CannotRun("/params/variation" + pointer(tuple(map(str, first["loc"]))), first["msg"]) from error
    if task_path is not None:
        _qualify_task(task, replacement, task_path)
    return EvaluationVariant(variant, replacement, overrides)
