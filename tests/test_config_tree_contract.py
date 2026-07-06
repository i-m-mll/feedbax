from __future__ import annotations

from types import SimpleNamespace

import pytest
from jax_cookbook import LDict

import feedbax.config as config
from feedbax.config.batch import load_batch_config
from feedbax.config.namespace import TreeNamespace
from feedbax.config.tree import (
    _align_trees_to_structure,
    _expand_missing_levels,
    ldict_level_to_bottom,
    ldict_level_to_top,
    move_ldict_level_above,
    rearrange_ldict_levels,
    tree_level_labels,
)
from feedbax.config.utils import deep_merge

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.no_silent_substitution_contract]


def _condition_model_tree() -> LDict:
    return LDict.of("condition")(
        {
            "easy": LDict.of("model")({"linear": 1, "gru": 2}),
            "hard": LDict.of("model")({"linear": 3, "gru": 4}),
        }
    )


def test_missing_ldict_level_without_reference_keys_rejected() -> None:
    with pytest.raises(ValueError, match="Cannot expand missing LDict level"):
        _expand_missing_levels(
            {"leaf": 1},
            target_levels=["condition"],
            current_levels=[],
            reference_trees=None,
        )


def test_ldict_level_reordering_preserves_leaf_assignments() -> None:
    tree = _condition_model_tree()

    reordered = rearrange_ldict_levels(tree, ["model", "condition"])

    assert tree_level_labels(reordered) == ["model", "condition"]
    assert reordered["linear"]["easy"] == 1
    assert reordered["linear"]["hard"] == 3
    assert reordered["gru"]["easy"] == 2
    assert reordered["gru"]["hard"] == 4


def test_ldict_level_move_helpers_preserve_relative_order() -> None:
    tree = _condition_model_tree()

    assert tree_level_labels(ldict_level_to_top("model", tree)) == ["model", "condition"]
    assert tree_level_labels(ldict_level_to_bottom("condition", tree)) == [
        "model",
        "condition",
    ]
    assert tree_level_labels(move_ldict_level_above("model", "condition", tree)) == [
        "model",
        "condition",
    ]


def test_align_trees_expands_missing_levels_from_reference_keys() -> None:
    aligned = _align_trees_to_structure(
        {"params": {"leaf": 1}},
        target_structure=["condition"],
        reference_trees={"reference": _condition_model_tree()},
    )

    assert tree_level_labels(aligned["params"]) == ["condition", "dict"]
    assert aligned["params"]["easy"] == {"leaf": 1}
    assert aligned["params"]["hard"] == {"leaf": 1}


def test_deep_merge_none_semantics_are_explicit_and_recursive() -> None:
    base = {"train": {"batch_size": 32, "n_batches": 10}, "seed": 0}
    override = {"train": {"batch_size": None}, "seed": None}

    assert deep_merge(base, override) == base
    assert deep_merge(base, override, ignore_none=False) == {
        "train": {"batch_size": None, "n_batches": 10},
        "seed": None,
    }


def test_tree_namespace_merge_matches_deep_merge_none_semantics() -> None:
    base = TreeNamespace(train=TreeNamespace(batch_size=32, n_batches=10), seed=0)
    override = {"train": {"batch_size": None}, "seed": None}

    ignored = base | override
    replaced = base.merge(override, ignore_none=False)

    assert ignored.train.batch_size == 32
    assert ignored.train.n_batches == 10
    assert ignored.seed == 0
    assert replaced.train.batch_size is None
    assert replaced.train.n_batches == 10
    assert replaced.seed is None


def test_config_globals_context_restores_mutated_namespaces() -> None:
    original_sep = config.STRINGS.hps_level_label_sep

    with config.config_globals_context():
        config.STRINGS.hps_level_label_sep = "mutated"
        assert config.STRINGS.hps_level_label_sep == "mutated"

    assert config.STRINGS.hps_level_label_sep == original_sep


def test_batch_package_probe_does_not_swallow_unexpected_errors(monkeypatch) -> None:
    registry = SimpleNamespace(
        _packages={
            "bad": SimpleNamespace(
                package_module=SimpleNamespace(__name__="bad_pkg"),
                config_resource_root="config",
            )
        },
        single_package_name=lambda: None,
    )

    def raise_unexpected(_: str) -> object:
        raise RuntimeError("unexpected probe failure")

    monkeypatch.setattr("feedbax.config.batch.resources.files", raise_unexpected)

    with pytest.raises(RuntimeError, match="unexpected probe failure"):
        load_batch_config("training", "missing", registry=registry)
