from operator import attrgetter
from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
import feedbax._vendor.jax_cookbook.tree as jtree

from feedbax.runtime.mapping import WhereDict
from feedbax.objectives.loss import TermTree
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


def test_cookbook_array_set_handles_termtree_none_value_leaves() -> None:
    history = TermTree.branch(
        "loss",
        {
            "position": TermTree.leaf("position", jnp.zeros((3,)), weight=1.0),
            "velocity": TermTree.leaf("velocity", jnp.zeros((3,)), weight=0.5),
        },
        weight=1.0,
    )
    values = TermTree.branch(
        "loss",
        {
            "position": TermTree.leaf("position", jnp.array(2.0), weight=1.0),
            "velocity": TermTree.leaf("velocity", jnp.array(3.0), weight=0.5),
        },
        weight=1.0,
    )

    updated = jtree.array_set(history, values, 1)

    assert jnp.array_equal(updated["position"].value, jnp.array([0.0, 2.0, 0.0]))
    assert jnp.array_equal(updated["velocity"].value, jnp.array([0.0, 3.0, 0.0]))
    assert updated["position"].weight == 1.0
    assert updated["velocity"].weight == 0.5


def test_tree_call_with_keys_honors_shared_key_markers() -> None:
    def make_value(*, key):
        return jr.uniform(key, ())

    callables = mark_shared(
        {"shared": make_value, "split": make_value}, lambda item: item["shared"]
    )
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



def test_wheredict_jit_reuses_equivalent_selectors_and_evaluates_distinct_paths() -> None:
    traced = []

    @eqx.filter_jit
    def apply_init(mapping, state):
        traced.append(None)
        where, value = next(iter(mapping.items()))
        return eqx.tree_at(where, state, value)

    state = _Pair(jnp.array(1), jnp.array(2))
    first = apply_init(WhereDict({lambda item: item.left: jnp.array(3)}), state)
    equivalent = apply_init(WhereDict({lambda pair: pair.left: jnp.array(4)}), state)
    getter = apply_init(WhereDict({attrgetter("left"): jnp.array(6)}), state)

    assert (first.left, first.right) == (3, 2)
    assert (equivalent.left, equivalent.right) == (4, 2)
    assert (getter.left, getter.right) == (6, 2)
    assert len(traced) == 1

    distinct = apply_init(WhereDict({lambda pair: pair.right: jnp.array(5)}), state)

    assert (distinct.left, distinct.right) == (1, 5)
    assert len(traced) == 2



def test_wheredict_functional_update_preserves_tagged_indexed_selectors() -> None:
    source = WhereDict({
        (lambda item: item.nodes["net"], "first"): 1,
        (lambda item: item.nodes["net"], "second"): 2,
        "plain": 3,
    })

    updated = eqx.tree_at(lambda mapping: mapping["nodes['net']#first"], source, 4)

    (where, tag), _, plain = updated
    assert where(SimpleNamespace(nodes={"net": "selected"})) == "selected"
    assert tag == "first"
    assert plain == "plain"
    assert source["nodes['net']#first"] == 1
    assert updated[(lambda item: item.nodes["net"], "first")] == 4
    assert updated["nodes['net']#second"] == 2


def test_wheredict_rejects_a_conditional_selector_before_it_can_alias_a_path() -> None:
    spec = WhereDict({lambda item: item.left: "left"})

    with pytest.raises(ValueError, match="WhereDict selector.*path"):
        spec[lambda item: item.left if item.flag else item.right] = "ambiguous"

    assert spec["left"] == "left"
