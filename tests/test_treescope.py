"""Tests for Treescope integration.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

import pytest

from feedbax._graph import detect_cycles_and_sort
from feedbax._treescope import (
    CycleAnnotation,
    TreeProjection,
    get_treescope_status,
    graph_to_tree,
    render_model_html,
    setup_treescope,
)


class TestCycleDetection:
    """Tests for cycle detection in graphs."""

    def test_acyclic_graph(self):
        """Test that an acyclic graph has no back edges."""
        adjacency = {
            "a": {"b", "c"},
            "b": {"d"},
            "c": {"d"},
            "d": set(),
        }
        order, back_edges = detect_cycles_and_sort(adjacency)

        assert len(back_edges) == 0
        # d must come after a, b, c
        assert order.index("d") > order.index("a")
        assert order.index("d") > order.index("b")
        assert order.index("d") > order.index("c")

    def test_single_cycle(self):
        """Test detection of a single cycle."""
        adjacency = {
            "a": {"b"},
            "b": {"c"},
            "c": {"a"},  # back edge forming a cycle
        }
        order, back_edges = detect_cycles_and_sort(adjacency)

        assert len(back_edges) == 1
        assert ("c", "a") in back_edges

    def test_feedback_loop_pattern(self):
        """Test a typical feedback loop pattern like in neural control."""
        # network -> mechanics -> feedback -> network (cycle)
        adjacency = {
            "task": {"network"},
            "network": {"mechanics"},
            "mechanics": {"feedback"},
            "feedback": {"network"},  # feedback loop
        }
        order, back_edges = detect_cycles_and_sort(adjacency)

        assert len(back_edges) == 1
        assert ("feedback", "network") in back_edges

        # Verify topological order of acyclic portion
        assert order.index("task") < order.index("network")
        assert order.index("network") < order.index("mechanics")
        assert order.index("mechanics") < order.index("feedback")


class TestCycleAnnotation:
    """Tests for CycleAnnotation dataclass."""

    def test_repr(self):
        """Test string representation."""
        ann = CycleAnnotation(
            source_node="mechanics",
            target_node="network",
            source_port="effector",
            target_port="feedback",
        )
        result = repr(ann)
        assert "mechanics.effector" in result
        assert "network.feedback" in result


class TestTreeProjection:
    """Tests for TreeProjection dataclass."""

    def test_has_cycles_property(self):
        """Test has_cycles property."""
        # Without cycles
        proj_no_cycles = TreeProjection(
            tree=None,  # type: ignore
            execution_order=("a", "b", "c"),
            cycle_annotations=[],
        )
        assert not proj_no_cycles.has_cycles

        # With cycles
        proj_with_cycles = TreeProjection(
            tree=None,  # type: ignore
            execution_order=("a", "b", "c"),
            cycle_annotations=[
                CycleAnnotation("c", "a", "out", "in"),
            ],
        )
        assert proj_with_cycles.has_cycles


class TestTreescopeStatus:
    """Tests for treescope availability checking."""

    def test_get_status(self):
        """Test that status can be retrieved."""
        status = get_treescope_status()

        assert "configured" in status
        assert "available" in status
        assert isinstance(status["configured"], bool)
        assert isinstance(status["available"], bool)


class TestTreescopeSetup:
    """Tests for treescope setup."""

    def test_setup_is_idempotent(self):
        """Test that setup can be called multiple times."""
        # First call
        setup_treescope()

        # Second call should not raise
        setup_treescope()

        status = get_treescope_status()
        assert status["configured"]


class TestGraphToTree:
    """Tests for graph_to_tree function."""

    def test_invalid_strategy(self):
        """Test that invalid cut strategy raises error."""
        # We need a minimal Graph-like object
        from feedbax.graph import Component, Graph, Wire

        # Create a simple graph (using the real Graph class)
        class DummyComponent(Component):
            input_ports = ("input",)
            output_ports = ("output",)

            def __call__(self, inputs, state, *, key):
                return {}, state

        graph = Graph(
            nodes={"a": DummyComponent()},
            wires=(),
        )

        with pytest.raises(ValueError, match="Unsupported cut_strategy"):
            graph_to_tree(graph, cut_strategy="invalid")  # type: ignore

    def test_acyclic_graph_projection(self):
        """Test projection of an acyclic graph."""
        from feedbax.graph import Component, Graph, Wire

        class DummyComponent(Component):
            input_ports = ("input",)
            output_ports = ("output",)

            def __call__(self, inputs, state, *, key):
                return {}, state

        graph = Graph(
            nodes={"a": DummyComponent(), "b": DummyComponent()},
            wires=(Wire("a", "output", "b", "input"),),
        )

        projection = graph_to_tree(graph)

        assert not projection.has_cycles
        assert len(projection.cycle_annotations) == 0
        assert projection.tree is graph

    def test_cyclic_graph_projection(self):
        """Test projection of a cyclic graph."""
        from feedbax.graph import Component, Graph, Wire

        class DummyComponent(Component):
            input_ports = ("input",)
            output_ports = ("output",)

            def __call__(self, inputs, state, *, key):
                return {}, state

        graph = Graph(
            nodes={"a": DummyComponent(), "b": DummyComponent()},
            wires=(
                Wire("a", "output", "b", "input"),
                Wire("b", "output", "a", "input"),  # Back edge
            ),
        )

        projection = graph_to_tree(graph)

        assert projection.has_cycles
        assert len(projection.cycle_annotations) == 1

        # The back edge should be from b to a
        ann = projection.cycle_annotations[0]
        assert ann.source_node == "b"
        assert ann.target_node == "a"


class TestRenderModelHtml:
    """Tests for HTML rendering."""

    def test_render_simple_dict(self):
        """Test rendering a simple PyTree (dict)."""
        model = {"a": 1, "b": [2, 3, 4]}

        html = render_model_html(model)

        assert isinstance(html, str)
        assert len(html) > 0
        # Should be valid HTML
        assert "<" in html
        assert ">" in html

    def test_render_nested_structure(self):
        """Test rendering a nested structure."""
        model = {
            "layer1": {"weights": [1.0, 2.0], "bias": 0.5},
            "layer2": {"weights": [3.0, 4.0], "bias": 1.0},
        }

        html = render_model_html(model)

        assert isinstance(html, str)
        assert len(html) > 0

    def test_render_with_max_depth(self):
        """Test that max_depth parameter is accepted."""
        model = {"a": {"b": {"c": {"d": 1}}}}

        # Should not raise
        html = render_model_html(model, max_depth=2)
        assert isinstance(html, str)

    def test_render_graph_with_cycles(self):
        """Test rendering a graph with cycles shows annotations."""
        from feedbax.graph import Component, Graph, Wire

        class DummyComponent(Component):
            input_ports = ("input",)
            output_ports = ("output",)

            def __call__(self, inputs, state, *, key):
                return {}, state

        graph = Graph(
            nodes={"a": DummyComponent(), "b": DummyComponent()},
            wires=(
                Wire("a", "output", "b", "input"),
                Wire("b", "output", "a", "input"),  # Back edge
            ),
        )

        html = render_model_html(graph, project_cycles=True)

        assert isinstance(html, str)
        # Should contain cycle information in the header
        assert "Feedback" in html or "cycle" in html.lower()


class TestPerformance:
    """Tests for performance requirements."""

    def test_render_typical_model_under_2_seconds(self):
        """Test that rendering completes in reasonable time."""
        import time

        # Create a moderately complex structure
        model = {
            f"layer_{i}": {
                "weights": list(range(10)),
                "bias": float(i),
                "config": {"type": "linear", "activation": "relu"},
            }
            for i in range(20)
        }

        start = time.time()
        html = render_model_html(model, max_depth=5)
        elapsed = time.time() - start

        assert elapsed < 2.0, f"Rendering took {elapsed:.2f}s, expected < 2s"
        assert len(html) > 0
