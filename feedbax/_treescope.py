"""Treescope integration for PyTree visualization.

Provides Treescope-based rendering for feedbax models with special handling
for cyclic graphs (feedback loops).

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from jaxtyping import PyTree

from feedbax._graph import detect_cycles_and_sort

if TYPE_CHECKING:
    from feedbax.graph import Graph

# Track whether treescope has been configured
_treescope_configured = False


@dataclass
class CycleAnnotation:
    """Annotation for a back edge that was cut to break a cycle.

    Attributes:
        source_node: The node where the edge originates.
        target_node: The node where the edge terminates.
        source_port: The output port on the source node.
        target_port: The input port on the target node.
    """

    source_node: str
    target_node: str
    source_port: str
    target_port: str

    def __repr__(self) -> str:
        return (
            f"CycleAnnotation({self.source_node}.{self.source_port} -> "
            f"{self.target_node}.{self.target_port})"
        )


@dataclass
class TreeProjection:
    """Result of projecting a cyclic graph to a tree.

    Attributes:
        tree: The tree representation of the graph (same Graph but with
            cycle wires conceptually cut).
        execution_order: The topological execution order of nodes.
        cycle_annotations: List of back edges that were cut.
    """

    tree: "Graph"
    execution_order: tuple[str, ...]
    cycle_annotations: list[CycleAnnotation]

    @property
    def has_cycles(self) -> bool:
        """Return True if the original graph contained cycles."""
        return len(self.cycle_annotations) > 0


def setup_treescope() -> None:
    """Configure Treescope with feedbax-specific renderers.

    This function sets up custom rendering handlers for feedbax types.
    It is idempotent and can be called multiple times safely.

    Note: In non-IPython environments, we skip interactive setup since
    treescope's basic_interactive_setup requires IPython.
    """
    global _treescope_configured

    if _treescope_configured:
        return

    try:
        import treescope
    except ImportError as e:
        raise ImportError(
            "treescope is required for visualization. "
            "Install it with: pip install treescope"
        ) from e

    # Configure treescope with default settings
    # The library handles most PyTree types automatically via JAX integration
    # Only call basic_interactive_setup in IPython environments
    try:
        # Check if we're in an IPython environment
        import IPython

        if IPython.get_ipython() is not None:
            treescope.basic_interactive_setup(autovisualize_arrays=True)
    except (ImportError, RuntimeError):
        # Not in IPython - skip interactive setup
        # Treescope will still work for render_to_html
        pass

    _treescope_configured = True


def graph_to_tree(
    graph: "Graph",
    cut_strategy: Literal["back_edges", "spanning_tree"] = "back_edges",
) -> TreeProjection:
    """Project a cyclic graph to a tree with cycle annotations.

    This function analyzes the graph's wiring to detect cycles (feedback loops)
    and returns a projection that identifies which edges were cut.

    Args:
        graph: The feedbax Graph to project.
        cut_strategy: Strategy for breaking cycles.
            - "back_edges": Cut edges that form back-references in DFS (default).
            - "spanning_tree": Not yet implemented.

    Returns:
        A TreeProjection containing the tree structure and cycle annotations.

    Raises:
        ValueError: If an unsupported cut_strategy is provided.
    """
    if cut_strategy != "back_edges":
        raise ValueError(
            f"Unsupported cut_strategy: {cut_strategy}. "
            "Currently only 'back_edges' is supported."
        )

    # Build adjacency list from wires
    adjacency: dict[str, set[str]] = {name: set() for name in graph.nodes}
    for wire in graph.wires:
        adjacency[wire.source_node].add(wire.target_node)

    # Use the existing cycle detection utility
    execution_order, back_edges = detect_cycles_and_sort(adjacency)

    # Build cycle annotations from back edges
    cycle_annotations: list[CycleAnnotation] = []
    for src, tgt in back_edges:
        # Find all wires that form this back edge
        for wire in graph.wires:
            if wire.source_node == src and wire.target_node == tgt:
                cycle_annotations.append(
                    CycleAnnotation(
                        source_node=wire.source_node,
                        target_node=wire.target_node,
                        source_port=wire.source_port,
                        target_port=wire.target_port,
                    )
                )

    return TreeProjection(
        tree=graph,  # The graph itself; cycles are annotated, not removed
        execution_order=tuple(execution_order),
        cycle_annotations=cycle_annotations,
    )


def render_model_html(
    model: PyTree,
    max_depth: int = 10,
    project_cycles: bool = True,
    roundtrip_mode: bool = False,
) -> str:
    """Render a PyTree model to Treescope HTML.

    Args:
        model: The PyTree (typically a feedbax Graph or Component) to render.
        max_depth: Maximum depth to expand in the tree view. Defaults to 10.
        project_cycles: If True and model is a Graph with cycles, include
            cycle annotations. Defaults to True.
        roundtrip_mode: If True, render in a format that can be copy-pasted
            to reconstruct the object. Defaults to False.

    Returns:
        An HTML string containing the Treescope visualization.
        This HTML is self-contained and can be embedded in an iframe.
    """
    setup_treescope()

    import treescope

    # Import Graph here to avoid circular imports at module level
    from feedbax.graph import Graph

    # Build context information for cycles if applicable
    cycle_info: dict[str, Any] = {}
    if project_cycles and isinstance(model, Graph):
        projection = graph_to_tree(model)
        if projection.has_cycles:
            cycle_info = {
                "has_cycles": True,
                "cycle_count": len(projection.cycle_annotations),
                "cycles": [
                    {
                        "source": f"{ann.source_node}.{ann.source_port}",
                        "target": f"{ann.target_node}.{ann.target_port}",
                    }
                    for ann in projection.cycle_annotations
                ],
                "execution_order": projection.execution_order,
            }

    # Configure rendering
    with treescope.active_autovisualizer.set_scoped(
        treescope.ArrayAutovisualizer()
    ):
        # Render to HTML
        rendered = treescope.render_to_html(
            model,
            roundtrip_mode=roundtrip_mode,
        )

    # If we have cycle info, prepend the cycle header
    if cycle_info:
        cycle_header = _build_cycle_header_html(cycle_info)
        # Wrap the result with the cycle header
        rendered = f'<div class="treescope-with-cycles">{cycle_header}{rendered}</div>'

    return rendered


def _build_cycle_header_html(cycle_info: dict[str, Any]) -> str:
    """Build HTML header for cycle annotations.

    Args:
        cycle_info: Dictionary containing cycle information.

    Returns:
        HTML string for the cycle annotation header.
    """
    import html

    cycle_count = cycle_info.get("cycle_count", 0)
    cycles = cycle_info.get("cycles", [])

    header_parts = [
        '<div style="padding: 8px 12px; background: #fef3c7; border-bottom: 1px solid #fcd34d; '
        'font-family: system-ui, sans-serif; font-size: 13px;">',
        f'<strong style="color: #92400e;">Feedback Loops Detected:</strong> '
        f'<span style="color: #78350f;">{cycle_count} cycle(s)</span>',
        '<ul style="margin: 4px 0 0 0; padding-left: 20px; color: #78350f;">',
    ]

    for cycle in cycles[:5]:  # Limit display to first 5 cycles
        source = html.escape(cycle["source"])
        target = html.escape(cycle["target"])
        header_parts.append(f"<li>{source} &rarr; {target}</li>")

    if len(cycles) > 5:
        header_parts.append(f"<li>... and {len(cycles) - 5} more</li>")

    header_parts.extend(["</ul>", "</div>"])

    return "".join(header_parts)


def render_to_text(
    model: PyTree,
    max_depth: int = 10,
) -> str:
    """Render a PyTree model to plain text.

    This is useful for debugging and logging where HTML is not appropriate.

    Args:
        model: The PyTree to render.
        max_depth: Maximum depth to show in the output. Defaults to 10.

    Returns:
        A string representation of the PyTree structure.
    """
    setup_treescope()

    import treescope

    # Use render_to_text which works outside IPython
    return treescope.render_to_text(model)


def get_treescope_status() -> dict[str, Any]:
    """Get the current Treescope configuration status.

    Returns:
        Dictionary with configuration status and capabilities.
    """
    status: dict[str, Any] = {
        "configured": _treescope_configured,
        "available": False,
    }

    try:
        import treescope

        status["available"] = True
        status["version"] = getattr(treescope, "__version__", "unknown")
    except ImportError:
        pass

    return status
