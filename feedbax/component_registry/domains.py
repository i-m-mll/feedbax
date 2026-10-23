"""Domain registry for Studio graph-domain metadata."""

from __future__ import annotations

from typing import Iterable

from feedbax.contracts.acausal import ACAUSAL_GRAPH_SCHEMA_ID
from feedbax.contracts.domain import (
    CAUSAL_DOMAIN_ID,
    ACAUSAL_DOMAIN_ID,
    MECHANICS_DOMAIN_ID,
    PENZAI_DOMAIN_ID,
    DomainMeta,
    EditorCapability,
    DomainTheme,
)
from feedbax.contracts.graph import GRAPH_SPEC_SCHEMA_ID
from feedbax.compiler.penzai_compiler import PENZAI_COMPILER_ID


class DomainRegistry:
    """Imperative registry for graph domains."""

    def __init__(self, domains: Iterable[DomainMeta] = ()) -> None:
        self._domains: dict[str, DomainMeta] = {}
        for domain in domains:
            self.register(domain)

    def register(self, domain: DomainMeta) -> None:
        """Register one domain.

        Raises:
            ValueError: If the domain id is already registered.
        """
        if domain.id in self._domains:
            raise ValueError(f"Domain id already registered: {domain.id!r}")
        self._domains[domain.id] = domain

    def get(self, domain_id: str) -> DomainMeta | None:
        """Return a domain by id, if registered."""
        return self._domains.get(domain_id)

    def list_all(self) -> list[DomainMeta]:
        """Return all domains in registration order."""
        return list(self._domains.values())


def builtin_domain_registry() -> DomainRegistry:
    """Build the Feedbax builtin domain registry."""
    registry = DomainRegistry()
    registry.register(
        DomainMeta(
            id=CAUSAL_DOMAIN_ID,
            display_name="Causal",
            interior_schema_id=GRAPH_SPEC_SCHEMA_ID,
            edge_semantics="directed",
            allows_multi_edge_per_port=False,
            nestable_domains=[
                CAUSAL_DOMAIN_ID,
                ACAUSAL_DOMAIN_ID,
                MECHANICS_DOMAIN_ID,
                PENZAI_DOMAIN_ID,
            ],
            editor=EditorCapability(kind="canvas", editable=True),
            theme=DomainTheme(color="causal", icon="Layers", edge_style="directed"),
            compiler_id=None,
        )
    )
    registry.register(
        DomainMeta(
            id=ACAUSAL_DOMAIN_ID,
            display_name="Acausal",
            interior_schema_id=ACAUSAL_GRAPH_SCHEMA_ID,
            edge_semantics="undirected",
            allows_multi_edge_per_port=True,
            nestable_domains=[ACAUSAL_DOMAIN_ID, MECHANICS_DOMAIN_ID],
            editor=EditorCapability(kind="canvas", editable=True),
            theme=DomainTheme(color="acausal", icon="Spline", edge_style="undirected"),
            compiler_id="feedbax.compiler.acausal",
        )
    )
    registry.register(
        DomainMeta(
            id=MECHANICS_DOMAIN_ID,
            display_name="Mechanics",
            interior_schema_id=ACAUSAL_GRAPH_SCHEMA_ID,
            edge_semantics="undirected",
            allows_multi_edge_per_port=True,
            nestable_domains=[ACAUSAL_DOMAIN_ID, MECHANICS_DOMAIN_ID],
            editor=EditorCapability(kind="tree", editable=True),
            theme=DomainTheme(color="mechanics", icon="Cog", edge_style="undirected"),
            compiler_id="feedbax.compiler.acausal",
        )
    )
    registry.register(
        DomainMeta(
            id=PENZAI_DOMAIN_ID,
            display_name="Penzai",
            interior_schema_id=None,
            edge_semantics="directed",
            allows_multi_edge_per_port=False,
            nestable_domains=[],
            editor=EditorCapability(kind="inspector", editable=False),
            theme=DomainTheme(color="penzai", icon="Hexagon", edge_style="directed"),
            compiler_id=PENZAI_COMPILER_ID,
        )
    )
    return registry
