"""Tests for PenzaiSubgraph component.

Tests the Penzai model wrapper for feedbax Graphs, including:
- Input/output mapping
- State management
- Graph integration
- Treescope rendering (when available)
"""

import pytest
import jax
import jax.numpy as jnp
from equinox.nn import State

from feedbax.graph import Component, Graph, Wire
from equinox.nn import StateIndex

from feedbax.graph import init_state_from_component
from feedbax.penzai_component import (
    PENZAI_AVAILABLE,
    TREESCOPE_AVAILABLE,
    PenzaiSubgraph,
    InputMapping,
    OutputMapping,
    PortSpec,
    PenzaiStateManager,
    register_penzai_builder,
    get_penzai_builder,
    list_penzai_builders,
    build_penzai_subgraph,
)


# =============================================================================
# Mock Penzai-like classes for testing without penzai installed
# =============================================================================


class MockPenzaiLayer:
    """Mock Penzai layer for testing without penzai installed."""

    def __init__(self, weight: jnp.ndarray, bias: jnp.ndarray):
        self.weight = weight
        self.bias = bias

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.weight + self.bias


class MockPenzaiSequential:
    """Mock Penzai Sequential for testing."""

    def __init__(self, layers: list):
        self.layers = layers

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


class MockPenzaiMLP:
    """Mock Penzai MLP for testing."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, *, key):
        keys = jax.random.split(key, 2)
        self.layer1 = MockPenzaiLayer(
            weight=jax.random.normal(keys[0], (input_size, hidden_size)) * 0.1,
            bias=jnp.zeros(hidden_size),
        )
        self.layer2 = MockPenzaiLayer(
            weight=jax.random.normal(keys[1], (hidden_size, output_size)) * 0.1,
            bias=jnp.zeros(output_size),
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.layer1(x))
        return self.layer2(x)


# =============================================================================
# InputMapping Tests
# =============================================================================


class TestInputMapping:
    """Tests for InputMapping class."""

    def test_single_input(self):
        """Test mapping single input port."""
        mapping = InputMapping.single("input")
        inputs = {"input": jnp.array([1.0, 2.0, 3.0])}

        result = mapping(inputs)

        assert (result == jnp.array([1.0, 2.0, 3.0])).all()

    def test_multi_input(self):
        """Test mapping multiple input ports."""
        mapping = InputMapping.multi("a", "b")
        inputs = {"a": jnp.array([1.0]), "b": jnp.array([2.0])}

        result = mapping(inputs)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert (result[0] == jnp.array([1.0])).all()
        assert (result[1] == jnp.array([2.0])).all()

    def test_structured_input(self):
        """Test mapping ports into structured input."""
        mapping = InputMapping.structured(
            ["x", "y"],
            lambda x, y: {"x": x, "y": y},
        )
        inputs = {"x": jnp.array([1.0]), "y": jnp.array([2.0])}

        result = mapping(inputs)

        assert isinstance(result, dict)
        assert (result["x"] == jnp.array([1.0])).all()
        assert (result["y"] == jnp.array([2.0])).all()

    def test_port_spec_extract_from_inputs(self):
        """Test PortSpec extraction with key path from inputs dict."""
        spec = PortSpec("nested", key_path=("data", "value"))
        inputs = {"nested": {"data": {"value": jnp.array([1.0, 2.0])}}}

        result = spec.extract_from_inputs(inputs)

        assert (result == jnp.array([1.0, 2.0])).all()

    def test_port_spec_extract_for_output(self):
        """Test PortSpec extraction for output mapping (no port name lookup)."""
        spec = PortSpec("output", key_path=("data",))
        data = {"data": jnp.array([3.0, 4.0])}

        result = spec.extract(data)

        assert (result == jnp.array([3.0, 4.0])).all()


# =============================================================================
# OutputMapping Tests
# =============================================================================


class TestOutputMapping:
    """Tests for OutputMapping class."""

    def test_single_output(self):
        """Test mapping single output value."""
        mapping = OutputMapping.single("output")
        layer_output = jnp.array([1.0, 2.0, 3.0])

        result = mapping(layer_output)

        assert "output" in result
        assert (result["output"] == jnp.array([1.0, 2.0, 3.0])).all()

    def test_multi_output(self):
        """Test mapping tuple output to multiple ports."""
        mapping = OutputMapping.multi("a", "b")
        layer_output = (jnp.array([1.0]), jnp.array([2.0]))

        result = mapping(layer_output)

        assert "a" in result
        assert "b" in result
        assert (result["a"] == jnp.array([1.0])).all()
        assert (result["b"] == jnp.array([2.0])).all()

    def test_structured_output(self):
        """Test extracting fields from structured output."""
        mapping = OutputMapping.structured([
            PortSpec("x", key_path=("x",)),
            PortSpec("y", key_path=("y",)),
        ])
        layer_output = {"x": jnp.array([1.0]), "y": jnp.array([2.0])}

        result = mapping(layer_output)

        assert "x" in result
        assert "y" in result
        assert (result["x"] == jnp.array([1.0])).all()
        assert (result["y"] == jnp.array([2.0])).all()


# =============================================================================
# PenzaiSubgraph Tests
# =============================================================================


def _make_penzai_subgraph(pz_model, input_mapping, output_mapping):
    """Helper to create PenzaiSubgraph with explicit empty state manager."""
    return PenzaiSubgraph(
        pz_model=pz_model,
        input_mapping=input_mapping,
        output_mapping=output_mapping,
        state_manager=PenzaiStateManager.empty(),
    )


class TestPenzaiSubgraph:
    """Tests for PenzaiSubgraph component."""

    def test_basic_wrapping(self):
        """Test wrapping a simple callable as PenzaiSubgraph."""
        key = jax.random.PRNGKey(0)
        layer = MockPenzaiLayer(
            weight=jnp.eye(4),
            bias=jnp.zeros(4),
        )

        component = _make_penzai_subgraph(
            pz_model=layer,
            input_mapping=InputMapping.single("input"),
            output_mapping=OutputMapping.single("output"),
        )

        inputs = {"input": jnp.array([1.0, 2.0, 3.0, 4.0])}
        state = init_state_from_component(component)

        outputs, new_state = component(inputs, state, key=key)

        assert "output" in outputs
        assert outputs["output"].shape == (4,)
        assert (outputs["output"] == jnp.array([1.0, 2.0, 3.0, 4.0])).all()

    def test_from_layer_factory(self):
        """Test from_layer class method."""
        layer = MockPenzaiLayer(
            weight=jnp.eye(2),
            bias=jnp.ones(2),
        )

        # Mock penzai availability check
        import feedbax.penzai_component as pc
        original_require = pc._require_penzai

        # Temporarily bypass penzai requirement for mock testing
        pc._require_penzai = lambda: None

        try:
            component = PenzaiSubgraph.from_layer(layer, input_port="x", output_port="y")

            assert component.input_ports == ("x",)
            assert component.output_ports == ("y",)

            inputs = {"x": jnp.array([1.0, 0.0])}
            outputs, _ = component(inputs, init_state_from_component(component), key=jax.random.PRNGKey(0))

            assert "y" in outputs
            assert (outputs["y"] == jnp.array([2.0, 1.0])).all()  # [1,0] + [1,1]
        finally:
            pc._require_penzai = original_require

    def test_mlp_execution(self):
        """Test PenzaiSubgraph with mock MLP."""
        key = jax.random.PRNGKey(42)
        mlp = MockPenzaiMLP(input_size=4, hidden_size=8, output_size=2, key=key)

        component = _make_penzai_subgraph(
            pz_model=mlp,
            input_mapping=InputMapping.single("input"),
            output_mapping=OutputMapping.single("output"),
        )

        inputs = {"input": jnp.ones(4)}
        state = init_state_from_component(component)

        outputs, new_state = component(inputs, state, key=jax.random.PRNGKey(0))

        assert "output" in outputs
        assert outputs["output"].shape == (2,)

    def test_multi_input_multi_output(self):
        """Test component with multiple inputs and outputs."""

        class MultiIOLayer:
            def __call__(self, x):
                a, b = x
                return a + b, a * b

        layer = MultiIOLayer()

        component = _make_penzai_subgraph(
            pz_model=layer,
            input_mapping=InputMapping.multi("a", "b"),
            output_mapping=OutputMapping.multi("sum", "prod"),
        )

        inputs = {"a": jnp.array([2.0]), "b": jnp.array([3.0])}
        outputs, _ = component(inputs, init_state_from_component(component), key=jax.random.PRNGKey(0))

        assert "sum" in outputs
        assert "prod" in outputs
        assert (outputs["sum"] == jnp.array([5.0])).all()
        assert (outputs["prod"] == jnp.array([6.0])).all()

    def test_graph_integration(self):
        """Test PenzaiSubgraph works within a Graph."""
        key = jax.random.PRNGKey(0)

        layer = MockPenzaiLayer(
            weight=2.0 * jnp.eye(2),
            bias=jnp.zeros(2),
        )

        penzai_node = _make_penzai_subgraph(
            pz_model=layer,
            input_mapping=InputMapping.single("input"),
            output_mapping=OutputMapping.single("output"),
        )

        graph = Graph(
            nodes={"encoder": penzai_node},
            wires=(),
            input_ports=("input",),
            output_ports=("output",),
            input_bindings={"input": ("encoder", "input")},
            output_bindings={"output": ("encoder", "output")},
        )

        inputs = {"input": jnp.array([1.0, 2.0])}
        state = init_state_from_component(graph)

        outputs, new_state = graph(inputs, state, key=key)

        assert "output" in outputs
        assert (outputs["output"] == jnp.array([2.0, 4.0])).all()

    def test_repr(self):
        """Test string representation."""
        layer = MockPenzaiLayer(weight=jnp.eye(2), bias=jnp.zeros(2))
        component = _make_penzai_subgraph(
            pz_model=layer,
            input_mapping=InputMapping.single("input"),
            output_mapping=OutputMapping.single("output"),
        )

        repr_str = repr(component)

        assert "PenzaiSubgraph" in repr_str
        assert "input_ports" in repr_str
        assert "output_ports" in repr_str


# =============================================================================
# PenzaiStateManager Tests
# =============================================================================


class TestPenzaiStateManager:
    """Tests for PenzaiStateManager."""

    def test_empty_state_manager(self):
        """Test creating empty state manager."""
        manager = PenzaiStateManager.empty()

        assert manager._initial_state == {}

    def test_state_initialization(self):
        """Test state is properly initialized."""
        # Create a simple component with state manager
        layer = MockPenzaiLayer(weight=jnp.eye(2), bias=jnp.zeros(2))
        component = _make_penzai_subgraph(
            pz_model=layer,
            input_mapping=InputMapping.single("input"),
            output_mapping=OutputMapping.single("output"),
        )

        state = init_state_from_component(component)

        # State should be initialized (may be empty dict for stateless models)
        assert state is not None


# =============================================================================
# Factory Registry Tests
# =============================================================================


class TestFactoryRegistry:
    """Tests for the Penzai builder registry."""

    def test_register_and_get_builder(self):
        """Test registering and retrieving a builder."""
        import feedbax.penzai_component as pc

        # Bypass penzai requirement
        original_require = pc._require_penzai
        pc._require_penzai = lambda: None

        try:
            def mock_builder(params):
                return MockPenzaiLayer(
                    weight=jnp.eye(params["size"]),
                    bias=jnp.zeros(params["size"]),
                )

            register_penzai_builder(
                "test_mock_layer",
                mock_builder,
                {"size": 4},
            )

            result = get_penzai_builder("test_mock_layer")

            assert result is not None
            builder_fn, defaults = result
            assert defaults["size"] == 4

            # Test builder works
            model = builder_fn({"size": 3})
            assert model.weight.shape == (3, 3)

        finally:
            pc._require_penzai = original_require
            # Clean up registry
            if "test_mock_layer" in pc._PENZAI_MODEL_BUILDERS:
                del pc._PENZAI_MODEL_BUILDERS["test_mock_layer"]

    def test_list_builders(self):
        """Test listing registered builders."""
        import feedbax.penzai_component as pc

        original_require = pc._require_penzai
        pc._require_penzai = lambda: None

        try:
            register_penzai_builder("test_list_a", lambda p: None, {})
            register_penzai_builder("test_list_b", lambda p: None, {})

            builders = list_penzai_builders()

            assert "test_list_a" in builders
            assert "test_list_b" in builders

        finally:
            pc._require_penzai = original_require
            # Clean up
            for name in ["test_list_a", "test_list_b"]:
                if name in pc._PENZAI_MODEL_BUILDERS:
                    del pc._PENZAI_MODEL_BUILDERS[name]

    def test_build_penzai_subgraph(self):
        """Test building PenzaiSubgraph from registered builder."""
        import feedbax.penzai_component as pc

        original_require = pc._require_penzai
        pc._require_penzai = lambda: None

        try:
            def mock_builder(params):
                return MockPenzaiLayer(
                    weight=jnp.eye(params["size"]),
                    bias=jnp.zeros(params["size"]),
                )

            register_penzai_builder("test_build", mock_builder, {"size": 2})

            component = build_penzai_subgraph(
                "test_build",
                params={"size": 3},
                input_port="x",
                output_port="y",
            )

            assert isinstance(component, PenzaiSubgraph)
            assert component.input_ports == ("x",)
            assert component.output_ports == ("y",)

        finally:
            pc._require_penzai = original_require
            if "test_build" in pc._PENZAI_MODEL_BUILDERS:
                del pc._PENZAI_MODEL_BUILDERS["test_build"]

    def test_build_unknown_builder_raises(self):
        """Test that building from unknown builder raises ValueError."""
        with pytest.raises(ValueError, match="Unknown Penzai builder"):
            build_penzai_subgraph("nonexistent_builder")


# =============================================================================
# Treescope Integration Tests
# =============================================================================


@pytest.mark.skipif(not TREESCOPE_AVAILABLE, reason="treescope not installed")
class TestTreescopeIntegration:
    """Tests for treescope HTML rendering."""

    def test_treescope_html(self):
        """Test treescope HTML generation."""
        layer = MockPenzaiLayer(weight=jnp.eye(2), bias=jnp.zeros(2))
        component = PenzaiSubgraph(
            pz_model=layer,
            input_mapping=InputMapping.single("input"),
            output_mapping=OutputMapping.single("output"),
        )

        html = component.treescope_html()

        assert isinstance(html, str)
        assert len(html) > 0


# =============================================================================
# JAX Transformation Compatibility Tests
# =============================================================================


class TestJAXTransformations:
    """Tests for JAX transformation compatibility."""

    def test_jit_compatible(self):
        """Test PenzaiSubgraph works with jax.jit."""
        layer = MockPenzaiLayer(weight=jnp.eye(4), bias=jnp.zeros(4))
        component = _make_penzai_subgraph(
            pz_model=layer,
            input_mapping=InputMapping.single("input"),
            output_mapping=OutputMapping.single("output"),
        )

        @jax.jit
        def run(inputs, state, key):
            return component(inputs, state, key=key)

        inputs = {"input": jnp.ones(4)}
        outputs, _ = run(inputs, init_state_from_component(component), jax.random.PRNGKey(0))

        assert "output" in outputs
        assert outputs["output"].shape == (4,)

    def test_vmap_compatible(self):
        """Test PenzaiSubgraph works with jax.vmap (batched inputs)."""
        layer = MockPenzaiLayer(weight=jnp.eye(4), bias=jnp.zeros(4))
        component = _make_penzai_subgraph(
            pz_model=layer,
            input_mapping=InputMapping.single("input"),
            output_mapping=OutputMapping.single("output"),
        )

        def run_single(x, key):
            inputs = {"input": x}
            outputs, _ = component(inputs, init_state_from_component(component), key=key)
            return outputs["output"]

        batch_size = 8
        batch_inputs = jnp.ones((batch_size, 4))
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

        batch_outputs = jax.vmap(run_single)(batch_inputs, keys)

        assert batch_outputs.shape == (batch_size, 4)

    def test_grad_compatible(self):
        """Test gradients flow through PenzaiSubgraph."""
        layer = MockPenzaiLayer(
            weight=jnp.eye(2),
            bias=jnp.zeros(2),
        )
        component = _make_penzai_subgraph(
            pz_model=layer,
            input_mapping=InputMapping.single("input"),
            output_mapping=OutputMapping.single("output"),
        )

        def loss_fn(x):
            inputs = {"input": x}
            outputs, _ = component(inputs, init_state_from_component(component), key=jax.random.PRNGKey(0))
            return jnp.sum(outputs["output"] ** 2)

        x = jnp.array([1.0, 2.0])
        grad = jax.grad(loss_fn)(x)

        # Gradient should exist and be finite
        assert grad.shape == (2,)
        assert jnp.isfinite(grad).all()
        # For identity + square loss, grad should be 2*x
        assert jnp.allclose(grad, 2 * x)
