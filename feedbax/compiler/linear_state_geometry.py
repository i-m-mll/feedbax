"""The authored effector position of a planar linear state-space occurrence."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from feedbax.builtins.mechanics.linear_state_space import LinearStateSpace
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.model_address import ComponentPortAddress, FieldElement, ModelAddress, PortElement
from feedbax.contracts.model_document import GraphInterior, LeafComponent, ModelDocument
from feedbax.contracts.semantic_values import semantic_key


@dataclass(frozen=True)
class LinearStatePoint:
    owner: tuple[str, ...]
    span: slice

    @property
    def vector(self) -> str:
        return semantic_key("model", "state", self.owner, "vector")

    @property
    def address(self) -> ModelAddress:
        port = PortElement(kind="port", end=ComponentPortAddress(
            owner="component", component=self.owner[-1], direction="output", port="effector",
        ))
        return ModelAddress(composites=self.owner[:-1], element=FieldElement(kind="field", parent=port, path=("pos",)))


def linear_state_point(
    component: LeafComponent, path: tuple[str, ...], registry: ComponentRegistry,
    overrides: Mapping[str, object],
) -> LinearStatePoint | None:
    owner = registry.owner_class(component.type, component.revision)
    if owner is None or not issubclass(owner, LinearStateSpace):
        return None
    resolved = registry.resolve_component_spec(component.type, component.fields, revision=component.revision)
    scope = tuple(part for name in path[:-1] for part in ("subgraphs", name))
    parameter = ".".join(("graph", *scope, "nodes", path[-1], "params", "pos_slice"))
    bounds = overrides.get(parameter, resolved.params["pos_slice"])
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        return None
    start, stop = bounds
    if type(start) is not int or type(stop) is not int or start < 0 or stop != start + 2:
        return None
    return LinearStatePoint(path, slice(start, stop))


def linear_state_points(
    model: ModelDocument, registry: ComponentRegistry, overrides: Mapping[str, object],
) -> Iterator[LinearStatePoint]:
    def visit(interior: GraphInterior, scope: tuple[str, ...]) -> Iterator[LinearStatePoint]:
        for name, component in interior.components.items():
            path = (*scope, name)
            if isinstance(component, LeafComponent):
                if point := linear_state_point(component, path, registry, overrides):
                    yield point
            elif isinstance(component.interior, GraphInterior):
                yield from visit(component.interior, path)

    return visit(model.interior, ())
