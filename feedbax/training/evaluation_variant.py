"""Resolve the qualified saved Evaluate diff against its authenticated source."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from feedbax.contracts.graph import ComponentSpec, GraphSpec, StudioEpochValueSpec, WireSpec
from feedbax.contracts.model_occurrence import graph_leaf_specs
from feedbax.contracts.matrix_core import apply_override_patch_in_place
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.rollout import VariationEdits, authored_variation
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.training import TaskSpec
from feedbax.runtime.model_values import MODEL_OWNERS
from feedbax.studio.canonical_evaluate import CannotRun


@dataclass(frozen=True)
class EvaluationVariant:
    graph: GraphSpec
    task: TaskSpec
    source_overrides: dict[str, float]
    edit_origins: dict[tuple[str, ...], str] = field(compare=False)

    def edit_location(self, model_path: str) -> str | None:
        """Trace a resolved graph diagnostic back to the contributing edit."""
        tokens = tuple(
            token.replace("~1", "/").replace("~0", "~") for token in model_path.split("/")[1:]
        )
        _, wires = _addressable_graph(self.graph)
        return _edit_location(tokens, wires, self.edit_origins)


def _edit_location(
    tokens: tuple[str, ...], wires: dict[tuple[str, ...], tuple[str, ...]],
    origins: dict[tuple[str, ...], str],
) -> str | None:
    if len(tokens) < 3:
        return None
    scope = 1
    while tokens[scope:scope + 1] == ("subgraphs",):
        scope += 2
    if tokens[scope:scope + 1] == ("wires",) and len(tokens) > scope + 1:
        index = tokens[scope + 1]
        wire_ids = wires.get(tokens[:scope], ())
        if index.isdigit() and int(index) < len(wire_ids):
            tokens = (*tokens[:scope + 1], wire_ids[int(index)], *tokens[scope + 2:])
    matches = {
        origin for target, origin in origins.items()
        if target[:len(tokens)] == tokens or tokens[:len(target)] == target
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _addressable_graph(graph: GraphSpec) -> tuple[dict, dict[tuple[str, ...], tuple[str, ...]]]:
    payload = graph.model_dump(mode="json")
    wire_ids = {}

    def bind(current: GraphSpec, raw: dict, path: tuple[str, ...]) -> None:
        wires = {wire.authored_id(): value for wire, value in zip(current.wires, raw["wires"], strict=True)}
        if len(wires) != len(current.wires):
            raise CannotRun(pointer(("inputs", "subject", *path, "wires")), "wire addresses are ambiguous")
        raw["wires"] = wires
        wire_ids[path] = tuple(wires)
        for name, child in (current.subgraphs or {}).items():
            if isinstance(child, GraphSpec):
                bind(child, raw["subgraphs"][name], (*path, "subgraphs", name))

    bind(graph, payload, ("graph",))
    return payload, wire_ids


def validate_evaluation_source(graph: GraphSpec) -> None:
    for path, node in graph_leaf_specs(graph):
        declaration = MODEL_OWNERS.get(node.type)
        if (
            declaration is not None and declaration.fixed_configuration and node.params.get("fixed") is True
            and node.revision in {None, declaration.revision}
        ):
            continue
        if (
            declaration is None or not declaration.checkpointable or declaration.states is None
            or node.revision not in {None, declaration.revision}
        ):
            raise CannotRun(
                pointer(("inputs", "subject", "graph", *(
                    part for owner in path[:-1] for part in ("subgraphs", owner)
                ), "nodes", path[-1])),
                f"owner {node.type!r} has no declared raw-state layout for this revision",
            )


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
    graph: GraphSpec, task: TaskSpec, patches: VariationEdits,
) -> EvaluationVariant:
    """Apply the existing project patch grammar with a bounded compatibility policy."""
    graph_payload, wire_ids = _addressable_graph(graph)
    payload = {"graph": graph_payload, "task": task.model_dump(mode="json")}
    overrides: dict[str, float] = {}
    task_path: str | None = None
    edit_origins = {}
    for path, patch in authored_variation(patches):
        parts = patch.path.split(".")
        try:
            if parts == ["task"] and patch.op == "replace":
                task_path = f"{path}/value"
                apply_override_patch_in_place(payload, patch)
                edit_origins[("task",)] = path
                continue
            if parts[0] != "graph":
                raise ValueError("expected an addressed model or task patch")
            current, raw, owners = graph, graph_payload, []
            offset = 1
            while parts[offset:offset + 1] == ["subgraphs"] and len(parts) > offset + 1:
                name = parts[offset + 1]
                child = (current.subgraphs or {}).get(name)
                if not isinstance(child, GraphSpec):
                    raise ValueError(f"unknown causal composite {name!r}")
                current, raw = child, raw["subgraphs"][name]
                owners.append(name)
                offset += 2
            match parts[offset:], patch.op:
                case ["wires", wire_id], ("add" | "replace"):
                    wire = WireSpec.model_validate(patch.value)
                    if wire.id not in (None, wire_id):
                        raise ValueError("wire ID differs from the edited address")
                    wire = WireSpec.model_validate(wire.model_dump(mode="json") | {"id": wire_id})
                    patch = patch.model_copy(update={"value": wire.model_dump(mode="json")})
                case ["wires", _], "remove":
                    pass
                case ["wires", wire_id, field], "replace" if field != "id":
                    if field not in WireSpec.model_fields:
                        raise ValueError("unknown wire field")
                    raw["wires"][wire_id]["id"] = wire_id
                case ["nodes", name], "add":
                    if name in raw["nodes"]:
                        raise ValueError("cannot replace a retained owner with an addition")
                    component = ComponentSpec.model_validate(patch.value)
                    owner = MODEL_OWNERS.get(component.type)
                    if owner is None or owner.states is None or component.revision not in {
                        None, owner.revision,
                    }:
                        raise ValueError(f"added owner {component.type!r} has no qualified declaration")
                case ["nodes", name, "params", parameter], ("replace" | "add"):
                    node = current.nodes.get(name)
                    if node is None:
                        raise ValueError("parameter patch must name a retained owner")
                    owner = MODEL_OWNERS.get(node.type)
                    if owner is None or owner.states is None or not owner.checkpointable:
                        raise ValueError(f"retained owner {node.type!r} has no qualified declaration")
                    declared = next((value for value in owner.values if value.symbol == parameter), None)
                    if declared is not None:
                        if declared.kind == "learned":
                            raise ValueError("M2 Evaluate preserves retained learned values")
                        if not _finite_float32(patch.value):
                            raise ValueError("expected a finite scalar for a declared source value")
                        overrides[semantic_key("model", "value", (*owners, name), parameter)] = float(patch.value)
                case _:
                    raise ValueError("patch is outside the qualified task and graph variation profile")
            apply_override_patch_in_place(payload, patch)
            edit_origins[tuple(parts)] = path
        except (ValueError, KeyError, OverflowError) as error:
            raise CannotRun(path, str(error)) from error

    try:
        for graph_path in wire_ids:
            raw = payload
            for part in graph_path:
                raw = raw[part]
            wire_ids[graph_path] = tuple(raw["wires"])
            raw["wires"] = list(raw["wires"].values())
        variant = GraphSpec.model_validate(payload["graph"])
    except ValidationError as error:
        first = error.errors()[0]
        tokens = ("graph", *map(str, first["loc"]))
        location = _edit_location(tokens, wire_ids, edit_origins) or "/inputs/subject" + pointer(tokens)
        raise CannotRun(location, first["msg"]) from error
    try:
        replacement = TaskSpec.model_validate(payload["task"])
    except ValidationError as error:
        first = error.errors()[0]
        raise CannotRun(task_path or "/inputs/subject/task", first["msg"]) from error
    if task_path is not None:
        _qualify_task(task, replacement, task_path)
    return EvaluationVariant(variant, replacement, overrides, edit_origins)
