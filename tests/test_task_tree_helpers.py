import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax_cookbook.tree as jtree

from feedbax.config.mapping import WhereDict
from feedbax.tasks._tree import is_marked_shared, mark_shared, tree_call_with_keys


class _Pair(eqx.Module):
    left: jnp.ndarray
    right: jnp.ndarray


def test_cookbook_filter_spec_expands_selected_subtree_leaves() -> None:
    tree = {"pair": _Pair(jnp.array([1.0]), jnp.array([2.0])), "other": jnp.array([3.0])}

    spec = jtree.filter_spec_leaves(tree, lambda item: item["pair"])

    assert spec["pair"].left is True
    assert spec["pair"].right is True
    assert spec["other"] is False


def test_cookbook_array_set_handles_none_non_array_leaves() -> None:
    tree = {"x": jnp.zeros((2,)), "name": "unchanged"}
    values = {"x": jnp.array(4.0), "name": "replacement"}

    updated = jtree.array_set(tree, values, 1)

    assert jnp.array_equal(updated["x"], jnp.array([0.0, 4.0]))
    assert updated["name"] == "replacement"


def test_tree_call_with_keys_honors_shared_key_markers() -> None:
    def make_value(*, key):
        return jr.uniform(key, ())

    callables = mark_shared({"shared": make_value, "split": make_value}, lambda item: item["shared"])
    shared_key = jr.PRNGKey(101)

    values = tree_call_with_keys(
        callables,
        key=jr.PRNGKey(0),
        shared_key=shared_key,
        use_shared=is_marked_shared,
    )

    assert values["shared"] == make_value(key=shared_key)
    assert values["split"] != make_value(key=shared_key)


def test_wheredict_string_and_callable_keys_match() -> None:
    spec = WhereDict({lambda item: item.state.hidden: "hidden"})

    assert spec["state.hidden"] == "hidden"
    assert spec[lambda item: item.state.hidden] == "hidden"
