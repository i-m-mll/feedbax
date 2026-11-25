"""Project retained Model state from the graph's resolved owner declarations."""

from operator import attrgetter

from feedbax.compiler.resolution import GraphParameterResolution
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.model_values import component_model_facts
from feedbax.contracts.retained_state import RetainedStateField, owner_state_field


def resolve_retained_state_fields(
    resolution: GraphParameterResolution, components: ComponentRegistry,
) -> tuple[RetainedStateField, ...]:
    """Read fields with declared array prototypes at every qualified Model occurrence."""
    fields = []

    def visit(current: GraphParameterResolution, scope: tuple[str, ...]) -> None:
        for name, node in current.graph.nodes.items():
            owner = (*scope, name)
            if (child := current.subgraphs.get(name)) is not None:
                visit(child, owner)
                continue
            facts = component_model_facts(node, components)
            if facts is None or not facts.has_state(node):
                continue
            for declaration in facts.states or ():
                # An opaque storage subtree declares its leaves only when the owner is realized.
                if declaration.subtree:
                    continue
                port, *path = declaration.prototype
                prototype = current.node_outputs.get((name, port))
                sample = attrgetter(".".join(path))(prototype) if path else prototype
                fields.append(owner_state_field(owner, declaration, sample.shape))

    visit(resolution, ())
    return tuple(fields)
