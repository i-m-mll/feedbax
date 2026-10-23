from equinox.nn import StateIndex
import jax.numpy as jnp
import pytest

from feedbax.runtime import align_state_indices_like


def test_align_state_indices_like_aligns_markers_and_preserves_target_init() -> None:
    target_index = StateIndex(jnp.array([1.0, 2.0], dtype=jnp.float32))
    template_index = StateIndex(jnp.array([3.0, 4.0], dtype=jnp.float32))
    passthrough = object()
    target_tree = {"index": target_index, "passthrough": passthrough}
    template_tree = {
        "index": template_index,
        "passthrough": object(),
    }

    aligned = align_state_indices_like(target_tree, template_tree)

    assert aligned["index"] is not target_index
    assert aligned["index"].marker is template_index.marker
    assert jnp.array_equal(aligned["index"].init, target_index.init)
    assert aligned["passthrough"] is passthrough


@pytest.mark.parametrize(
    ("target_tree", "template_tree"),
    [
        ({"index": StateIndex(jnp.array(1.0))}, {"index": object()}),
        ({"index": object()}, {"index": StateIndex(jnp.array(1.0))}),
    ],
)
def test_align_state_indices_like_rejects_state_index_leaf_mismatch(
    target_tree,
    template_tree,
) -> None:
    with pytest.raises(TypeError, match="matching StateIndex leaves"):
        align_state_indices_like(target_tree, template_tree)


def test_align_state_indices_like_rejects_tree_structure_mismatch() -> None:
    target_tree = {"target": StateIndex(jnp.array(1.0))}
    template_tree = {"template": StateIndex(jnp.array(2.0))}

    with pytest.raises(ValueError, match="same PyTree structure"):
        align_state_indices_like(target_tree, template_tree)
