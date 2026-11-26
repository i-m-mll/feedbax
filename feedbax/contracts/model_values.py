"""Semantic model facts declared by component owners."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from feedbax.contracts.expression_values import SignalSpace

if TYPE_CHECKING:
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.acausal import AcausalGraphSpec
    from feedbax.contracts.graph import ComponentSpec, GraphSpec
    from feedbax.contracts.portable import ComponentOwner


@dataclass(frozen=True)
class ModelValue:
    symbol: str
    field: str
    kind: Literal["source", "parameter"]


type SignalLeafPath = tuple[str | int, ...]


@dataclass(frozen=True)
class StateValue:
    symbol: str
    fields: tuple[str, ...]
    units: str
    axis: Literal["xy", "unit", "muscle", "joint"]
    prototype: tuple[str, ...]
    """The resolved output port and path that provide the storage shape and dtype."""
    aliases: tuple[tuple[str, ...], ...] = ()
    """Storage paths initialized with this value when they have no explicit initializer."""
    reset: StateReset | None = None
    configuration: bool = False
    space: SignalSpace | None = None
    subtree: bool = False
    """Use the runtime signal's array leaves and axes below this storage path."""
    output_leaves: tuple[SignalLeafPath, ...] = ()
    """Port-first paths whose values equal this state after the step."""


@dataclass(frozen=True)
class StateReset:
    override_port: str | None = None
    required: bool = False
    """The component has no meaningful start of its own, so a task must initialize the field."""


@dataclass(frozen=True)
class TrialParameter:
    name: str
    kind: Literal["real", "boolean"]
    units: str = "1"
    dtype_from: str | None = None
    """Output port whose array dtype the configuration scalar uses."""


@dataclass(frozen=True)
class ModelFacts:
    """Checkpoint, observation and per-trial storage of a component."""

    values: tuple[ModelValue, ...] = ()
    states: tuple[StateValue, ...] | None = ()
    checkpointable: bool = True
    physical_checkpoint: Callable[[ComponentSpec, AcausalGraphSpec], bool] | None = None
    configuration_slots: tuple[str, ...] = ()
    fixed_configuration_flag: str | None = None
    optional_state_flag: str | None = None
    trial_parameters: tuple[TrialParameter, ...] = ()
    trial_parameter_override_port: str | None = None

    def is_fixed_configuration(self, node: ComponentSpec) -> bool:
        return (
            self.fixed_configuration_flag is not None
            and node.params.get(self.fixed_configuration_flag) is True
        )

    def has_state(self, node: ComponentSpec) -> bool:
        return self.optional_state_flag is None or node.params.get(self.optional_state_flag) is not False

    def supports_checkpoint(self, node: ComponentSpec, physical: AcausalGraphSpec | None) -> bool:
        if self.is_fixed_configuration(node):
            return True
        if not self.checkpointable:
            return False
        if self.physical_checkpoint is None:
            return physical is None
        return physical is not None and self.physical_checkpoint(node, physical)


def model_owner_class(node: ComponentSpec, components: ComponentRegistry) -> type[ComponentOwner] | None:
    """Look up the owner of a current, already migrated component."""
    return components.owner_class(node.type, node.revision)


def component_model_facts(node: ComponentSpec, components: ComponentRegistry) -> ModelFacts | None:
    owner = model_owner_class(node, components)
    return None if owner is None else owner.model_facts


def is_finite_source_scalar(value: object) -> bool:
    limit = 3.4028234663852886e38
    return type(value) in (int, float) and -limit <= value <= limit


def scalar_source_parameters(node: ComponentSpec, components: ComponentRegistry) -> dict[str, float]:
    """The scalar configuration values a compatible Model may choose independently of its weights."""
    owner = component_model_facts(node, components)
    if owner is None or not owner.checkpointable or owner.is_fixed_configuration(node):
        return {}
    return {
        value.symbol: float(config)
        for value in owner.values if value.kind == "source"
        if is_finite_source_scalar(config := node.params.get(value.symbol))
    }


@dataclass(frozen=True)
class ModelOwnerProblem:
    path: tuple[str, ...]
    reason: str

    @property
    def occurrence(self) -> tuple[str, ...]:
        return self.path[2::2]


def model_owner_problem(
    graph: GraphSpec, use: Literal["checkpoint", "evaluate"], components: ComponentRegistry,
) -> ModelOwnerProblem | None:
    """Check a resolved graph's declared support before realizing its owners."""
    return next(model_owner_problems(graph, use, components), None)


def model_owner_problems(
    graph: GraphSpec, use: Literal["checkpoint", "evaluate"], components: ComponentRegistry,
) -> Iterator[ModelOwnerProblem]:
    from feedbax.contracts.model_occurrence import graph_leaf_occurrences

    for path, node, physical in graph_leaf_occurrences(graph):
        facts = component_model_facts(node, components)
        if use == "evaluate":
            if physical is not None or (facts is not None and facts.physical_checkpoint is not None):
                reason = "native or physical owner has no qualified Evaluate realization"
            elif facts is not None and (
                facts.is_fixed_configuration(node) or (facts.checkpointable and facts.states is not None)
            ):
                continue
            else:
                reason = f"owner {node.type!r} has no declared Evaluate layout for this revision"
        elif facts is not None and facts.supports_checkpoint(node, physical):
            continue
        elif physical is not None:
            reason = (
                f"owner {node.type!r} has no typed model checkpoint owner "
                f"for solver {physical.solver.solver_type!r} and this configuration"
            )
        else:
            reason = f"owner {node.type!r} has no typed model checkpoint owner for this revision"
        yield ModelOwnerProblem(
            ("graph", *(token for name in path[:-1] for token in ("subgraphs", name)), "nodes", path[-1]),
            reason,
        )
