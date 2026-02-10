"""API endpoints for model inspection and Treescope visualization."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from feedbax.web.models.graph import GraphSpec
from feedbax.web.models.inspection import (
    CycleAnnotationModel,
    InlineTreescopeRequest,
    InspectionStatusResponse,
    TreescopeRequest,
    TreescopeResponse,
)
from feedbax.web.services.graph_service import GraphService

router = APIRouter()
graph_service = GraphService()


def _graph_spec_to_tree_dict(graph_spec: GraphSpec) -> dict:
    """Convert GraphSpec to a treescope-friendly nested dict.

    Treescope renders plain dicts as readable trees, whereas Pydantic models
    produce unhelpful raw-object dumps.  This helper builds a concise nested
    structure suitable for ``render_model_html()``.

    Args:
        graph_spec: The graph specification to convert.

    Returns:
        A nested dict representing the graph structure.
    """
    tree: dict = {}
    for name, node in graph_spec.nodes.items():
        node_dict: dict = {"type": node.type}
        if node.params:
            node_dict["params"] = dict(node.params)
        if node.input_ports:
            node_dict["inputs"] = node.input_ports
        if node.output_ports:
            node_dict["outputs"] = node.output_ports
        tree[name] = node_dict

    if graph_spec.wires:
        tree["__wires__"] = [
            f"{w.source_node}.{w.source_port} -> {w.target_node}.{w.target_port}"
            for w in graph_spec.wires
        ]

    if graph_spec.subgraphs:
        for sg_name, sg_spec in graph_spec.subgraphs.items():
            # Ensure the subgraph key exists (it may not if there is no
            # corresponding top-level node with that name).
            tree.setdefault(sg_name, {})
            tree[sg_name]["subgraph"] = _graph_spec_to_tree_dict(sg_spec)

    return tree


def _render_graph_spec(
    graph_spec: GraphSpec,
    max_depth: int = 10,
    project_cycles: bool = True,
    roundtrip_mode: bool = False,
) -> TreescopeResponse:
    """Render a GraphSpec to a TreescopeResponse.

    Shared logic for both the saved-graph and inline-graph endpoints.

    Args:
        graph_spec: The graph specification to render.
        max_depth: Maximum tree depth for rendering.
        project_cycles: Whether to detect and annotate cycles.
        roundtrip_mode: Whether to render in reconstructable format.

    Returns:
        A TreescopeResponse with the rendered HTML and cycle info.

    Raises:
        HTTPException: If treescope is unavailable or rendering fails.
    """
    from feedbax._treescope import (
        get_treescope_status,
        render_model_html,
    )

    status = get_treescope_status()
    if not status.get("available", False):
        raise HTTPException(
            status_code=503,
            detail="Treescope is not installed. Install with: pip install treescope",
        )

    try:
        tree_dict = _graph_spec_to_tree_dict(graph_spec)
        html = render_model_html(
            tree_dict,
            max_depth=max_depth,
            project_cycles=project_cycles,
            roundtrip_mode=roundtrip_mode,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to render Treescope visualization: {str(e)}",
        )

    # Build cycle information from the spec
    cycles: list[CycleAnnotationModel] = []
    execution_order: list[str] | None = None
    has_cycles = False

    if project_cycles:
        from feedbax._graph import detect_cycles_and_sort

        adjacency: dict[str, set[str]] = {
            name: set() for name in graph_spec.nodes.keys()
        }
        for wire in graph_spec.wires:
            adjacency[wire.source_node].add(wire.target_node)

        order, back_edges = detect_cycles_and_sort(adjacency)
        execution_order = list(order)
        has_cycles = len(back_edges) > 0

        for src, tgt in back_edges:
            for wire in graph_spec.wires:
                if wire.source_node == src and wire.target_node == tgt:
                    cycles.append(
                        CycleAnnotationModel(
                            source=f"{wire.source_node}.{wire.source_port}",
                            target=f"{wire.target_node}.{wire.target_port}",
                        )
                    )

    return TreescopeResponse(
        html=html,
        has_cycles=has_cycles,
        cycle_count=len(cycles),
        cycles=cycles,
        execution_order=execution_order,
    )


@router.get("/status")
async def get_inspection_status() -> InspectionStatusResponse:
    """Get the status of inspection capabilities."""
    from feedbax._treescope import get_treescope_status

    status = get_treescope_status()

    return InspectionStatusResponse(
        treescope_available=status.get("available", False),
        treescope_configured=status.get("configured", False),
        treescope_version=status.get("version"),
    )


@router.post("/treescope/inline")
async def render_treescope_from_spec(
    request: InlineTreescopeRequest,
) -> TreescopeResponse:
    """Render treescope visualization from an in-memory graph spec.

    This endpoint accepts a GraphSpec directly in the request body,
    allowing visualization of unsaved graphs and subgraphs.

    Args:
        request: The graph spec and rendering options.

    Returns:
        TreescopeResponse containing the HTML and cycle information.
    """
    return _render_graph_spec(
        request.graph,
        max_depth=request.max_depth,
        project_cycles=request.project_cycles,
    )


@router.post("/treescope/{graph_id}")
async def render_treescope(
    graph_id: str,
    request: TreescopeRequest,
) -> TreescopeResponse:
    """Render a graph as Treescope HTML visualization.

    Args:
        graph_id: The ID of the graph to render.
        request: Rendering options.

    Returns:
        TreescopeResponse containing the HTML and cycle information.
    """
    try:
        record = graph_service.get_graph(graph_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Graph not found")

    return _render_graph_spec(
        record.project.graph,
        max_depth=request.max_depth,
        project_cycles=request.project_cycles,
        roundtrip_mode=request.roundtrip_mode,
    )
