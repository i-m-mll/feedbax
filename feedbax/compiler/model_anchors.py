"""Resolve the drawn places declared by a Model's leaf components."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from feedbax.component_registry import ComponentRegistry
from feedbax.component_registry.registry import ComponentResolution
from feedbax.compiler.model_signals import ResolvedSignalLeaf
from feedbax.contracts.model_address import ComponentElement, FieldElement, FieldPath, ModelAddress
from feedbax.contracts.model_document import GraphInterior, LeafComponent, ModelDocument
from feedbax.contracts.model_values import StateValue
from feedbax.contracts.portable import ComponentOwner, StateSignal
from feedbax.contracts.representation import RepresentationStateAnchorSelectorBinding


@dataclass(frozen=True)
class ModelAnchor:
    address: ModelAddress
    signal: ModelAddress | None
    state: StateValue | None


def geometry_address(path: tuple[str, ...], field: FieldPath) -> ModelAddress:
    return ModelAddress(
        composites=path[:-1],
        element=FieldElement(
            kind="field", parent=ComponentElement(kind="component", id=path[-1]),
            path=("geometry", *field),
        ),
    )


def portable_model_anchors(
    path: tuple[str, ...], resolved: ComponentResolution,
) -> tuple[ModelAnchor, ...]:
    representation = resolved.meta.representation
    owner = resolved.meta.owner_class
    if representation is None or owner is None or not issubclass(owner, ComponentOwner):
        return ()
    facts = owner.model_facts
    ports = owner.ports
    if facts is None or facts.states is None or ports is None:
        return ()
    geometry = next(
        (element for element in representation.elements if element.archetype in ("point_body", "planar_chain")),
        None,
    )
    if geometry is None:
        return ()
    declared = {anchor.id: anchor for anchor in representation.anchors}
    states = {state.symbol: state for state in facts.states if state.space == "point"}
    anchors = []
    for anchor_id in geometry.anchors:
        binding = declared[anchor_id].binding
        state = None
        if isinstance(binding, RepresentationStateAnchorSelectorBinding):
            selector = binding.selector
            if selector.target_id is not None and selector.path is not None:
                leaf = ports.output_leaf(
                    (selector.target_id, *selector.path.split(".")), resolved.params,
                )
                if isinstance(leaf, StateSignal):
                    state = states.get(leaf.symbol)
        if geometry.archetype == "point_body":
            if state is None:
                continue
            field = ("point",) if len(geometry.anchors) == 1 else ("anchor", anchor_id)
        else:
            field = ("anchor", anchor_id)
        anchors.append(ModelAnchor(
            address=geometry_address(path, field),
            signal=ModelAddress.at_state(path[:-1], path[-1], state.symbol) if state else None,
            state=state,
        ))
    return tuple(anchors)


def model_anchors(model: ModelDocument, registry: ComponentRegistry) -> Iterator[ModelAnchor]:
    def visit(interior: GraphInterior, scope: tuple[str, ...]) -> Iterator[ModelAnchor]:
        for id, component in interior.components.items():
            path = (*scope, id)
            if isinstance(component, LeafComponent):
                meta = registry.get(component.type)
                if meta is not None and meta.representation is not None:
                    yield from portable_model_anchors(path, registry.resolve_component_spec(
                        component.type, component.fields, revision=component.revision,
                    ))
            elif isinstance(component.interior, GraphInterior):
                yield from visit(component.interior, path)

    yield from visit(model.interior, ())


def anchored_outputs(
    model: ModelDocument, registry: ComponentRegistry, outputs: Mapping[str, ResolvedSignalLeaf],
) -> dict[str, ModelAddress]:
    by_origin = {}
    for anchor in model_anchors(model, registry):
        if anchor.signal is not None:
            by_origin.setdefault(anchor.signal.to_text(), anchor.address)
    return {
        name: by_origin[origin]
        for name, output in outputs.items()
        if output.origin is not None and (origin := output.origin.to_text()) in by_origin
    }
