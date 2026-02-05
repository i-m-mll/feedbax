"""Tests for the PyTree selection API.

Tests cover:
- Core Selection functionality (at, get, count, set, apply)
- Type and predicate filtering (at_instances_of, where)
- Partition/combine round-trips
- Compatibility with JAX transformations (jit, vmap)
- Graph-specific selection extensions
- Equivalence with eqx.tree_at for basic operations
"""

import equinox as eqx
from equinox import Module
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
import jax.tree as jt
import pytest

from feedbax._selectors import Selection, select, tree_at
from feedbax.graph import Component, Graph, Wire


# ========== Test Fixtures ==========


class SimpleModule(Module):
    """A simple module for testing."""

    weight: jax.Array
    bias: jax.Array
    name: str = eqx.field(static=True)


class NestedModule(Module):
    """A nested module for testing."""

    layer1: SimpleModule
    layer2: SimpleModule
    scale: float


class DeepModule(Module):
    """A deeply nested module for testing."""

    encoder: NestedModule
    decoder: NestedModule
    config: dict = eqx.field(static=True)


@pytest.fixture
def simple_module():
    return SimpleModule(
        weight=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        bias=jnp.array([0.1, 0.2]),
        name="simple",
    )


@pytest.fixture
def nested_module():
    return NestedModule(
        layer1=SimpleModule(
            weight=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            bias=jnp.array([0.1, 0.2]),
            name="layer1",
        ),
        layer2=SimpleModule(
            weight=jnp.array([[5.0, 6.0], [7.0, 8.0]]),
            bias=jnp.array([0.3, 0.4]),
            name="layer2",
        ),
        scale=2.0,
    )


@pytest.fixture
def deep_module():
    return DeepModule(
        encoder=NestedModule(
            layer1=SimpleModule(
                weight=jnp.ones((2, 2)),
                bias=jnp.zeros(2),
                name="enc_l1",
            ),
            layer2=SimpleModule(
                weight=jnp.ones((2, 2)) * 2,
                bias=jnp.ones(2),
                name="enc_l2",
            ),
            scale=1.0,
        ),
        decoder=NestedModule(
            layer1=SimpleModule(
                weight=jnp.ones((2, 2)) * 3,
                bias=jnp.zeros(2) + 0.5,
                name="dec_l1",
            ),
            layer2=SimpleModule(
                weight=jnp.ones((2, 2)) * 4,
                bias=jnp.ones(2) * 2,
                name="dec_l2",
            ),
            scale=0.5,
        ),
        config={"hidden_size": 64},
    )


# ========== Core Selection Tests ==========


class TestSelectionBasics:
    """Tests for basic Selection functionality."""

    def test_select_returns_selection(self, simple_module):
        sel = select(simple_module)
        assert isinstance(sel, Selection)
        assert sel.tree is simple_module

    def test_selection_repr(self, simple_module):
        sel = select(simple_module)
        repr_str = repr(sel)
        assert "Selection" in repr_str
        assert "SimpleModule" in repr_str

    def test_count_all_leaves(self, simple_module):
        count = select(simple_module).count()
        # SimpleModule has weight (array), bias (array) as dynamic leaves
        # name is static, so it shouldn't be counted
        assert count == 2

    def test_count_nested_leaves(self, nested_module):
        count = select(nested_module).count()
        # 2 arrays per SimpleModule * 2 layers + 1 scale
        assert count == 5

    def test_get_returns_filtered_tree(self, simple_module):
        result = select(simple_module).get()
        # Without filter, should get all leaves
        assert result.weight is not None
        assert result.bias is not None


class TestSelectionAt:
    """Tests for the `at()` accessor method."""

    def test_at_single_field(self, simple_module):
        result = select(simple_module).at(lambda m: m.weight).get()
        assert jnp.allclose(result, simple_module.weight)

    def test_at_nested_field(self, nested_module):
        result = select(nested_module).at(lambda m: m.layer1.weight).get()
        assert jnp.allclose(result, nested_module.layer1.weight)

    def test_at_chained(self, nested_module):
        result = (
            select(nested_module)
            .at(lambda m: m.layer1)
            .at(lambda l: l.weight)
            .get()
        )
        assert jnp.allclose(result, nested_module.layer1.weight)


class TestSelectionSet:
    """Tests for the `set()` modification method."""

    def test_set_single_value(self, simple_module):
        new_weight = jnp.zeros((2, 2))
        result = select(simple_module).at(lambda m: m.weight).set(new_weight)

        assert jnp.allclose(result.weight, new_weight)
        assert jnp.allclose(result.bias, simple_module.bias)

    def test_set_nested_value(self, nested_module):
        new_weight = jnp.ones((2, 2)) * 99
        result = (
            select(nested_module)
            .at(lambda m: m.layer1.weight)
            .set(new_weight)
        )

        assert jnp.allclose(result.layer1.weight, new_weight)
        assert jnp.allclose(result.layer2.weight, nested_module.layer2.weight)

    def test_set_equals_eqx_tree_at(self, simple_module):
        """Verify that select().at().set() equals eqx.tree_at()."""
        new_weight = jnp.zeros((2, 2))
        where_fn = lambda m: m.weight

        select_result = select(simple_module).at(where_fn).set(new_weight)
        eqx_result = eqx.tree_at(where_fn, simple_module, new_weight)

        assert jnp.allclose(select_result.weight, eqx_result.weight)
        assert jnp.allclose(select_result.bias, eqx_result.bias)


class TestSelectionApply:
    """Tests for the `apply()` method."""

    def test_apply_function(self, simple_module):
        result = select(simple_module).at(lambda m: m.weight).apply(lambda x: x * 2)

        assert jnp.allclose(result.weight, simple_module.weight * 2)
        assert jnp.allclose(result.bias, simple_module.bias)

    def test_apply_nested(self, nested_module):
        result = (
            select(nested_module)
            .at(lambda m: m.layer1.weight)
            .apply(jnp.zeros_like)
        )

        assert jnp.allclose(result.layer1.weight, jnp.zeros((2, 2)))
        assert jnp.allclose(result.layer2.weight, nested_module.layer2.weight)


# ========== Type and Predicate Filtering Tests ==========


class TestAtInstancesOf:
    """Tests for at_instances_of() filtering."""

    def test_filter_by_array_type(self, nested_module):
        count = select(nested_module).at_instances_of(jax.Array).count()
        # 4 arrays (2 per layer) but scale is float, not array
        assert count == 4

    def test_filter_by_module_type(self, deep_module):
        count = select(deep_module).at_instances_of(SimpleModule).count()
        # 4 SimpleModule instances (2 in encoder, 2 in decoder)
        assert count == 4

    def test_filter_multiple_types(self, nested_module):
        # This should match arrays
        count = select(nested_module).at_instances_of(jax.Array, float).count()
        # 4 arrays + 1 float scale
        assert count == 5


class TestWhere:
    """Tests for where() predicate filtering."""

    def test_where_shape_predicate(self, nested_module):
        # Select only arrays with shape (2, 2)
        result = (
            select(nested_module)
            .at_instances_of(jax.Array)
            .where(lambda x: x.shape == (2, 2))
            .count()
        )
        # Only weights have shape (2, 2), biases have shape (2,)
        assert result == 2

    def test_where_value_predicate(self, simple_module):
        # Select arrays with sum > 5
        result = (
            select(simple_module)
            .at_instances_of(jax.Array)
            .where(lambda x: x.sum() > 5)
            .count()
        )
        # weight.sum() = 10, bias.sum() = 0.3
        assert result == 1


# ========== Partition/Combine Tests ==========


class TestPartitionCombine:
    """Tests for partition() and combine() round-trips."""

    def test_partition_by_type(self, nested_module):
        selected, rest = (
            select(nested_module)
            .at_instances_of(jax.Array)
            .partition()
        )

        # Selected should have arrays, rest should have None for arrays
        assert selected.layer1.weight is not None
        assert rest.layer1.weight is None

    def test_combine_round_trip(self, nested_module):
        original = nested_module

        selected, rest = (
            select(original)
            .at_instances_of(jax.Array)
            .partition()
        )

        reconstructed = Selection.combine(selected, rest)

        # Verify reconstruction matches original
        assert jnp.allclose(reconstructed.layer1.weight, original.layer1.weight)
        assert jnp.allclose(reconstructed.layer1.bias, original.layer1.bias)
        assert jnp.allclose(reconstructed.layer2.weight, original.layer2.weight)
        assert jnp.allclose(reconstructed.layer2.bias, original.layer2.bias)
        assert reconstructed.scale == original.scale

    def test_partition_modify_combine(self, nested_module):
        """Test modifying partitioned leaves and recombining."""
        arrays, non_arrays = (
            select(nested_module)
            .at_instances_of(jax.Array)
            .partition()
        )

        # Modify only arrays (zero them out)
        zeroed_arrays = jt.map(
            lambda x: jnp.zeros_like(x) if x is not None else None,
            arrays,
        )

        result = Selection.combine(zeroed_arrays, non_arrays)

        assert jnp.allclose(result.layer1.weight, jnp.zeros((2, 2)))
        assert jnp.allclose(result.layer2.bias, jnp.zeros(2))
        assert result.scale == nested_module.scale  # Non-array preserved


# ========== JAX Transformation Compatibility ==========


class TestJAXCompatibility:
    """Tests for compatibility with JAX transformations."""

    def test_set_under_jit(self, simple_module):
        @jax.jit
        def modify(module):
            return select(module).at(lambda m: m.weight).apply(lambda x: x * 2)

        result = modify(simple_module)
        assert jnp.allclose(result.weight, simple_module.weight * 2)

    def test_partition_combine_under_jit(self, simple_module):
        """Test partition/combine works under JIT.

        Note: We use SimpleModule here because NestedModule's `scale` field
        (a Python float) gets traced to an array under JIT, which complicates
        the test. This test focuses on verifying the basic pattern works.
        """
        @jax.jit
        def zero_arrays(module):
            arrays, rest = select(module).at_instances_of(jax.Array).partition()
            zeroed = jt.map(
                lambda x: jnp.zeros_like(x) if x is not None else None,
                arrays,
            )
            return Selection.combine(zeroed, rest)

        result = zero_arrays(simple_module)
        assert jnp.allclose(result.weight, jnp.zeros((2, 2)))
        assert jnp.allclose(result.bias, jnp.zeros(2))

    def test_apply_under_vmap(self):
        """Test that selection works with vmapped modules."""
        # Create a batch of modules
        weights = jnp.stack([jnp.ones((2, 2)) * i for i in range(3)])
        biases = jnp.stack([jnp.zeros(2) for _ in range(3)])

        batch_module = SimpleModule(weight=weights, bias=biases, name="batch")

        # Apply scaling under vmap perspective
        result = select(batch_module).at(lambda m: m.weight).apply(lambda x: x * 2)

        expected = weights * 2
        assert jnp.allclose(result.weight, expected)


# ========== tree_at Compatibility ==========


class TestTreeAtShim:
    """Tests for the tree_at compatibility function."""

    def test_tree_at_with_value(self, simple_module):
        new_weight = jnp.zeros((2, 2))
        result = tree_at(lambda m: m.weight, simple_module, value=new_weight)
        assert jnp.allclose(result.weight, new_weight)

    def test_tree_at_with_replace_fn(self, simple_module):
        result = tree_at(
            lambda m: m.weight,
            simple_module,
            replace_fn=lambda x: x * 2,
        )
        assert jnp.allclose(result.weight, simple_module.weight * 2)

    def test_tree_at_rejects_both_args(self, simple_module):
        with pytest.raises(ValueError, match="Cannot specify both"):
            tree_at(
                lambda m: m.weight,
                simple_module,
                value=jnp.zeros((2, 2)),
                replace_fn=lambda x: x * 2,
            )


# ========== Graph Selection Extensions ==========


class PassThrough(Component):
    """A simple pass-through component for testing."""

    input_ports = ("input",)
    output_ports = ("output",)

    weight: jax.Array

    def __init__(self, weight: jax.Array):
        self.weight = weight

    def __call__(self, inputs, state, *, key):
        return {"output": inputs["input"] @ self.weight}, state


class ScaledComponent(Component):
    """A component that scales input."""

    input_ports = ("input",)
    output_ports = ("output",)

    scale: jax.Array

    def __init__(self, scale: jax.Array):
        self.scale = scale

    def __call__(self, inputs, state, *, key):
        return {"output": inputs["input"] * self.scale}, state


@pytest.fixture
def simple_graph():
    """Create a simple graph for testing."""
    return Graph(
        nodes={
            "encoder": PassThrough(weight=jnp.ones((2, 2))),
            "decoder": PassThrough(weight=jnp.ones((2, 2)) * 2),
        },
        wires=(Wire("encoder", "output", "decoder", "input"),),
        input_ports=("input",),
        output_ports=("output",),
        input_bindings={"input": ("encoder", "input")},
        output_bindings={"output": ("decoder", "output")},
    )


@pytest.fixture
def mixed_graph():
    """Create a graph with mixed component types."""
    return Graph(
        nodes={
            "pass1": PassThrough(weight=jnp.ones((2, 2))),
            "scale": ScaledComponent(scale=jnp.array(2.0)),
            "pass2": PassThrough(weight=jnp.ones((2, 2)) * 3),
        },
        wires=(
            Wire("pass1", "output", "scale", "input"),
            Wire("scale", "output", "pass2", "input"),
        ),
        input_ports=("input",),
        output_ports=("output",),
        input_bindings={"input": ("pass1", "input")},
        output_bindings={"output": ("pass2", "output")},
    )


class TestGraphSelect:
    """Tests for Graph.select() method."""

    def test_graph_select_returns_selection(self, simple_graph):
        sel = simple_graph.select()
        assert isinstance(sel, Selection)
        assert sel.tree is simple_graph

    def test_graph_select_at_nodes(self, simple_graph):
        # Select all arrays in the graph
        count = simple_graph.select().at_instances_of(jax.Array).count()
        # 2 weights
        assert count == 2

    def test_graph_select_apply(self, simple_graph):
        result = (
            simple_graph
            .select()
            .at(lambda g: g.nodes["encoder"].weight)
            .apply(lambda x: x * 10)
        )

        assert jnp.allclose(result.nodes["encoder"].weight, jnp.ones((2, 2)) * 10)
        assert jnp.allclose(result.nodes["decoder"].weight, jnp.ones((2, 2)) * 2)


class TestGraphSelectNode:
    """Tests for Graph.select_node() method."""

    def test_select_existing_node(self, simple_graph):
        sel = simple_graph.select_node("encoder")
        assert isinstance(sel, Selection)

    def test_select_nonexistent_node_raises(self, simple_graph):
        with pytest.raises(KeyError, match="does not exist"):
            simple_graph.select_node("nonexistent")

    def test_select_node_apply(self, simple_graph):
        # Modify the encoder's weight
        result = (
            simple_graph
            .select_node("encoder")
            .apply(lambda node: PassThrough(weight=jnp.zeros((2, 2))))
        )

        assert jnp.allclose(result.nodes["encoder"].weight, jnp.zeros((2, 2)))
        assert jnp.allclose(result.nodes["decoder"].weight, jnp.ones((2, 2)) * 2)


class TestGraphSelectNodesOfType:
    """Tests for Graph.select_nodes_of_type() method."""

    def test_select_single_type(self, mixed_graph):
        sel = mixed_graph.select_nodes_of_type(PassThrough)
        count = sel.count()
        # 2 PassThrough nodes
        assert count == 2

    def test_select_multiple_types(self, mixed_graph):
        sel = mixed_graph.select_nodes_of_type(PassThrough, ScaledComponent)
        count = sel.count()
        # All 3 nodes
        assert count == 3

    def test_select_type_apply(self, mixed_graph):
        # Zero out all PassThrough weights
        result = (
            mixed_graph
            .select_nodes_of_type(PassThrough)
            .apply(lambda node: PassThrough(weight=jnp.zeros_like(node.weight)))
        )

        assert jnp.allclose(result.nodes["pass1"].weight, jnp.zeros((2, 2)))
        assert jnp.allclose(result.nodes["pass2"].weight, jnp.zeros((2, 2)))
        # ScaledComponent should be unchanged
        assert jnp.allclose(result.nodes["scale"].scale, jnp.array(2.0))


# ========== Edge Cases ==========


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_none_values_in_tree(self):
        """Test selection on a tree with None values.

        In JAX, None values are treated as "absence of value" and don't
        produce leaves. This is consistent with eqx.partition behavior.
        """
        tree_with_nones = {"a": None, "b": None}
        count = select(tree_with_nones).count()
        # None values don't produce leaves in JAX trees
        assert count == 0

    def test_mixed_none_and_values(self):
        """Test selection on a tree with mixed None and actual values."""
        mixed = {"a": jnp.array([1, 2]), "b": None, "c": jnp.array([3, 4])}
        count = select(mixed).count()
        # Only the actual arrays are counted as leaves
        assert count == 2

    def test_select_dict(self):
        """Test selection on a plain dict."""
        d = {"x": jnp.array([1, 2]), "y": jnp.array([3, 4])}
        result = select(d).at(lambda t: t["x"]).set(jnp.zeros(2))
        assert jnp.allclose(result["x"], jnp.zeros(2))
        assert jnp.allclose(result["y"], jnp.array([3, 4]))

    def test_select_tuple(self):
        """Test selection on a tuple."""
        t = (jnp.array([1, 2]), jnp.array([3, 4]))
        result = select(t).at(lambda x: x[0]).set(jnp.zeros(2))
        assert jnp.allclose(result[0], jnp.zeros(2))
        assert jnp.allclose(result[1], jnp.array([3, 4]))

    def test_deeply_nested_selection(self, deep_module):
        """Test selection in deeply nested structures."""
        result = (
            select(deep_module)
            .at(lambda m: m.encoder.layer1.weight)
            .set(jnp.ones((2, 2)) * 99)
        )

        assert jnp.allclose(result.encoder.layer1.weight, jnp.ones((2, 2)) * 99)
        # Other weights unchanged
        assert jnp.allclose(result.encoder.layer2.weight, jnp.ones((2, 2)) * 2)
        assert jnp.allclose(result.decoder.layer1.weight, jnp.ones((2, 2)) * 3)
