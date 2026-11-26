"""Resolve the qualified saved Evaluate diff against its authenticated source."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from feedbax.contracts.expression_values import Problem
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
from feedbax.contracts.model_attachment import is_generated_attachment_node, is_trial_binding
from feedbax.contracts.matrix_core import apply_override_patch_in_place
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.rollout import VariationEdits, authored_variation
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.model_values import component_model_facts, is_finite_source_scalar, model_owner_problem
from feedbax.contracts.rollout import CannotRun

if TYPE_CHECKING:
    from feedbax.component_registry import ComponentRegistry


_MANAGED_TASK = TypeAdapter(TaskDocument | PythonTaskSelection)


@dataclass(frozen=True)
class EvaluationVariant:
    graph: GraphSpec
    task: TaskDocument | PythonTaskSelection
    source_overrides: dict[str, float]
    edit_origins: dict[tuple[str, ...], str] = field(compare=False)

    def edit_location(self, model_path: str) -> str | None:
        """Trace a resolved graph diagnostic back to the contributing edit."""
        tokens = tuple(
            token.replace("~1", "/").replace("~0", "~") for token in model_path.split("/")[1:]
        )
        _, wires = _addressable_graph(self.graph)
        return _edit_location(tokens, wires, self.edit_origins)

    def task_refusal(self, problem: Problem) -> CannotRun:
        """Locate a failed trial pairing at the responsible Task edit when identifiable."""
        path = tuple(map(str, problem.path))
        at = (
            ("task", *path[1:]) if path[:1] in {("task",), ("python_task",)}
            else ("task", *path) if isinstance(self.task, TaskDocument) and path[:1] and path[0] in TaskDocument.model_fields
            else ()
        )
        edits = [
            target for target in self.edit_origins
            if len(target) > 1 and at and (target == at[:len(target)] or at == target[:len(at)])
        ]
        if edits:
            return CannotRun(self.edit_origins[max(edits, key=len)], problem.reason)
        if at or problem.path[:2] == ("model", "inputs"):
            task_edits = [origin for target, origin in self.edit_origins.items() if target[0] == "task"]
            if len(task_edits) == 1:
                return CannotRun(task_edits[0], problem.reason)
        return CannotRun(pointer(("inputs", "subject", *problem.path)), problem.reason)


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


def validate_evaluation_source(graph: GraphSpec, components: ComponentRegistry) -> None:
    if problem := model_owner_problem(graph, "evaluate", components):
        raise CannotRun(pointer(("inputs", "subject", *problem.path)), problem.reason)


def _public_input_ports(ports):
    return [name for name in ports if not is_trial_binding(name)]


def _replacement_task(patch, path):
    try:
        return _MANAGED_TASK.validate_python(patch.value)
    except ValidationError as error:
        raise CannotRun(path, error.errors()[0]["msg"]) from error


def _edit_task(task, patch, path, origins):
    parts = patch.path.split(".")
    if parts == ["task"] and patch.op == "replace":
        task = _replacement_task(patch, f"{path}/value")
        for target in tuple(origins):
            if target[0] == "task":
                del origins[target]
    elif len(parts) > 1 and (isinstance(task, TaskDocument) or len(parts) == 3 and parts[1] == "params"):
        selected = {"task": task.model_dump(mode="json")}
        apply_override_patch_in_place(selected, patch)
        task = type(task).model_validate(selected["task"])
    else:
        raise CannotRun(path, "expected an addressed task patch")
    origins[tuple(parts)] = path
    return task


def resolve_evaluation_task(
    task: TaskDocument | PythonTaskSelection, patches: VariationEdits,
) -> TaskDocument | PythonTaskSelection:
    """Apply the Task's part of a rollout variation independently of graph resolution."""
    origins = {}
    for path, patch in authored_variation(patches):
        if patch.path.split(".", 1)[0] == "task":
            try:
                task = _edit_task(task, patch, path, origins)
            except (ValueError, KeyError, OverflowError) as error:
                raise CannotRun(path, str(error)) from error
    return task


def resolve_evaluation_variant(
    graph: GraphSpec, task: TaskDocument | PythonTaskSelection, patches: VariationEdits,
    *, components: ComponentRegistry,
) -> EvaluationVariant:
    """Apply the existing project patch grammar with a bounded compatibility policy."""
    graph_payload, wire_ids = _addressable_graph(graph)
    payload = {"graph": graph_payload}
    overrides: dict[str, float] = {}
    replacement = task
    edit_origins = {}
    for path, patch in authored_variation(patches):
        parts = patch.path.split(".")
        try:
            if parts[0] == "task":
                replacement = _edit_task(replacement, patch, path, edit_origins)
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
                case ["output_bindings", name], "replace":
                    if name not in current.output_bindings:
                        raise ValueError("output patch must retain the declared port")
                    if (
                        not isinstance(patch.value, (list, tuple)) or len(patch.value) != 2
                        or not all(isinstance(part, str) for part in patch.value)
                    ):
                        raise ValueError("output binding requires a node and port")
                case ["input_ports"], "replace":
                    if (
                        not isinstance(patch.value, list) or not all(isinstance(name, str) for name in patch.value)
                        or _public_input_ports(patch.value) != _public_input_ports(current.input_ports)
                    ):
                        raise ValueError("Only generated intervention inputs can change.")
                case ["input_bindings", name], ("add" | "replace" | "remove") if is_trial_binding(name):
                    if patch.op != "remove" and (
                        not isinstance(patch.value, (list, tuple)) or len(patch.value) != 2
                        or not all(isinstance(part, str) for part in patch.value)
                        or not patch.value[0].startswith("_model_input_") or patch.value[1] != "input"
                    ):
                        raise ValueError("An intervention input requires its generated relay.")
                case ["nodes", name], "add":
                    if name in raw["nodes"]:
                        raise ValueError("cannot replace a retained owner with an addition")
                    component = ComponentSpec.model_validate(patch.value)
                    owner = component_model_facts(component, components)
                    if owner is None or owner.states is None:
                        raise ValueError(f"added owner {component.type!r} has no qualified declaration")
                case ["nodes", name], ("remove" | "replace"):
                    node = current.nodes.get(name)
                    if node is None or not is_generated_attachment_node(name, node.type):
                        raise ValueError("Expected a generated Model attachment occurrence")
                    if patch.op == "replace":
                        target = ComponentSpec.model_validate(patch.value)
                        if target.type != node.type or target.revision not in {None, node.revision}:
                            raise ValueError("An attachment field edit must retain its component type and revision")
                case ["nodes", name, "input_ports"], "replace":
                    node = current.nodes.get(name)
                    if (
                        node is None or node.type != "feedbax.Subgraph" or not isinstance(patch.value, list)
                        or not all(isinstance(port, str) for port in patch.value)
                        or _public_input_ports(node.input_ports) != _public_input_ports(patch.value)
                    ):
                        raise ValueError("Only generated inputs of a retained composite can change.")
                case ["nodes", name, "params", parameter], ("replace" | "add"):
                    node = current.nodes.get(name)
                    if node is None:
                        raise ValueError("parameter patch must name a retained owner")
                    owner = component_model_facts(node, components)
                    if owner is None or owner.states is None or not owner.checkpointable:
                        raise ValueError(f"retained owner {node.type!r} has no qualified declaration")
                    declared = next((value for value in owner.values if value.symbol == parameter), None)
                    if declared is not None:
                        if declared.kind == "parameter":
                            raise ValueError("M2 Evaluate preserves retained learned values")
                        if not is_finite_source_scalar(patch.value):
                            raise ValueError("expected a finite scalar for a declared source value")
                        if patch.op == "add" and parameter in node.params:
                            # Compilation materializes a default absent from the authored Model.
                            patch = patch.model_copy(update={"op": "replace"})
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
    required = {name for name in variant.input_bindings.keys() - graph.input_bindings.keys() if is_trial_binding(name)}
    present = {signal.id for signal in replacement.signals} if isinstance(replacement, TaskDocument) else set()
    if required - present:
        raise CannotRun("/params/variation", "A generated intervention input has no bound Task signal.")
    return EvaluationVariant(variant, replacement, overrides, edit_origins)
