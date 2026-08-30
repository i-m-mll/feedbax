import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax_cookbook.tree as jtree

from feedbax.runtime.mapping import WhereDict
from feedbax.intervene.schedule import evaluate_intervenor_params
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


def test_intervenor_param_evaluator_splits_callable_leaf_keys() -> None:
    def make_value(trial_spec, batch_info, *, key):
        del trial_spec, batch_info
        return jr.uniform(key, ())

    values = evaluate_intervenor_params(
        {"left": make_value, "right": make_value},
        trial_spec=None,
        batch_info=None,
        key=jr.PRNGKey(0),
    )

    assert values["left"] != values["right"]


def test_wheredict_string_and_callable_keys_match() -> None:
    spec = WhereDict({lambda item: item.state.hidden: "hidden"})

    assert spec["state.hidden"] == "hidden"
    assert spec[lambda item: item.state.hidden] == "hidden"
