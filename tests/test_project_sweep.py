from __future__ import annotations

from random import Random
from time import perf_counter

import pytest

from feedbax.contracts.matrix_core import apply_override_patches
from feedbax.contracts.project import Project
from feedbax.contracts.project_resolution import ResolvedStep, resolve_step
from feedbax.contracts.project_sweep import (
    Sweep,
    SweepExpansionCode,
    SweepExpansionError,
    SweepInputMember,
    count_sweep,
    expand_sweep,
    sweep_patches,
)


def test_inherited_rows_and_axis_alternatives_use_the_resolved_sweep():
    project = Project.model_validate(
        {
            "steps": {
                "parent": {
                    "kind": "train",
                    "base": {"inline": {"gain": 1}},
                    "sweep": {
                        "axes": {"gain": {"values": {name: {} for name in ("low", "mid", "high")}}},
                        "rows": {
                            "prototype": {"patches": {"tag": {"op": "add", "value": "parent"}}}
                        },
                    },
                },
                "child": {
                    "kind": "train",
                    "base": {"step": "parent"},
                    "sweep": {
                        "inherit_rows": False,
                        "axes": {
                            "gain": {
                                "values": {
                                    "low": None,
                                    "mid": None,
                                    "high": {"patches": {"gain": {"value": 5}}},
                                }
                            },
                        },
                        "rows": {
                            "derived": {
                                "from": "prototype",
                                "patches": {"tag": {"value": "child"}},
                            },
                            "next": {"from": "derived", "patches": {"gain": {"value": 2}}},
                        },
                    },
                },
            },
        }
    )
    child = resolve_step(project, "child")
    assert isinstance(child, ResolvedStep)
    rows = expand_sweep(child.sweep)
    assert count_sweep(child.sweep) == len(rows) == 3
    assert [row.id for row in rows] == ["derived", "next", "gain=high"]
    assert [apply_override_patches(child.payload, sweep_patches(row.patches)) for row in rows] == [
        {"gain": 1, "tag": "child"},
        {"gain": 2, "tag": "child"},
        {"gain": 5},
    ]


def test_groups_zip_and_manual_are_ordered_and_use_named_values():
    axes = {
        "b": {"values": {"p": {}, "q": {}}},
        "a": {"values": {"x": {}, "y": {}}},
        "c": {"values": {"m": {}, "n": {}}},
    }
    grouped = Sweep.model_validate(
        {
            "axes": axes,
            "combination": {
                "groups": {
                    "pair": {"axes": ["b", "a"], "mode": "zip"},
                    "other": {"axes": ["c"], "mode": "cross"},
                }
            },
        }
    )
    assert count_sweep(grouped) == 4
    assert [row.id for row in expand_sweep(grouped)] == [
        "a=x--b=p--c=m",
        "a=x--b=p--c=n",
        "a=y--b=q--c=m",
        "a=y--b=q--c=n",
    ]
    manual = Sweep.model_validate(
        {
            "axes": axes,
            "combination": {
                "manual": [{"c": "m", "b": "q", "a": "x"}, {"a": "y", "b": "p", "c": "n"}]
            },
        }
    )
    assert [row.id for row in expand_sweep(manual)] == ["a=x--b=q--c=m", "a=y--b=p--c=n"]
    zipped = Sweep.model_validate({"axes": axes, "combination": "zip"})
    assert [row.id for row in expand_sweep(zipped)] == ["a=x--b=p--c=m", "a=y--b=q--c=n"]


def test_fanout_carries_coordinates_through_two_steps():
    training = expand_sweep(
        Sweep.model_validate(
            {
                "axes": {
                    "rate": {"values": {f"r{i}": {} for i in range(4)}},
                    "seed": {"values": {f"s{i}": {} for i in range(3)}},
                }
            }
        )
    )
    known = [
        SweepInputMember(row.id, row.coordinates, row.label)
        for row in training
        if row.coordinates["rate"].value_id in {"r1", "r2"}
    ]
    evaluation = Sweep.model_validate(
        {
            "axes": {
                "training": {"input": "subject"},
                "task": {"values": {f"t{i}": {} for i in range(3)}},
            }
        }
    )
    roles = {"subject": known}
    evaluated = expand_sweep(evaluation, inputs=roles)
    assert count_sweep(evaluation, inputs=roles) == len(evaluated) == 18
    assert list(evaluated[0].coordinates) == ["rate", "seed", "task"]
    assert evaluated[0].input_keys == {"subject": known[0].key}

    analysis = Sweep.model_validate({"axes": {"evaluation": {"input": "data"}}})
    analyzed = expand_sweep(
        analysis,
        inputs={
            "data": [SweepInputMember(row.id, row.coordinates, row.label) for row in evaluated]
        },
    )
    assert [row.coordinates for row in analyzed] == [row.coordinates for row in evaluated]
    assert [row.input_keys["data"] for row in analyzed] == [row.id for row in evaluated]
    assert count_sweep(evaluation, inputs={"subject": []}) == 0
    assert expand_sweep(evaluation, inputs={"subject": []}) == ()


def test_generated_values_have_stable_keys_and_sampling_is_explicit():
    first = Sweep.model_validate({"axes": {"seed": {"path": "seed", "values": [1, 2**64 - 1]}}})
    second = Sweep.model_validate({"axes": {"seed": {"path": "seed", "values": [2**64 - 1, 1.0]}}})
    assert [row.id for row in expand_sweep(first)] == [
        row.id for row in reversed(expand_sweep(second))
    ]
    assert expand_sweep(first)[1].coordinates["seed"].value == 2**64 - 1

    ranges = Sweep.model_validate(
        {
            "axes": {
                "linear": {"path": "linear", "range": {"start": 0, "stop": 2, "count": 3}},
                "log": {
                    "path": "log",
                    "range": {"start": 1, "stop": 100, "count": 3, "scale": "log"},
                },
            },
            "combination": "zip",
        }
    )
    assert [
        (row.coordinates["linear"].value, row.coordinates["log"].value)
        for row in expand_sweep(ranges)
    ] == [(0, 1), (1, 10), (2, 100)]

    sampled = Sweep.model_validate(
        {
            "axes": {
                "draw": {
                    "path": "sample",
                    "sampler": {"distribution": "prior", "count": 4, "seed": 9},
                },
            }
        }
    )
    assert count_sweep(sampled) == 4
    with pytest.raises(SweepExpansionError) as error:
        expand_sweep(sampled)
    assert error.value.code == SweepExpansionCode.UNRESOLVED_SAMPLER
    calls = []

    def resolve(distribution, count, seed):
        calls.append((distribution, count, seed))
        return Random(seed).sample(range(1_000), count)

    with pytest.raises(SweepExpansionError) as error:
        expand_sweep(sampled, sampler=resolve, max_rows=3)
    assert error.value.code == SweepExpansionCode.ROW_BUDGET
    assert calls == []
    assert expand_sweep(sampled, sampler=resolve) == expand_sweep(sampled, sampler=resolve)
    assert calls == [("prior", 4, 9)] * 2


def test_invalid_combinations_and_conflicts_name_the_location():
    axes = {"a": {"values": {"x": {}}}, "b": {"values": {"y": {}, "z": {}}}}
    cases = [
        ({"axes": axes, "combination": "zip"}, "invalid_combination", "sweep.combination"),
        (
            {
                "axes": axes,
                "combination": {"groups": {"bad": {"axes": ["a", "a", "b"], "mode": "cross"}}},
            },
            "invalid_combination",
            "sweep.combination.groups.bad.axes",
        ),
        (
            {"axes": axes, "combination": {"manual": [{"a": "no", "b": "y"}]}},
            "unknown_value",
            "sweep.combination.manual.0.a",
        ),
        (
            {"axes": axes, "combination": {"manual": [{"a": "x", "b": "y"}] * 2}},
            "duplicate_row",
            "sweep",
        ),
        ({"axes": {"fanout": {"input": "data"}}}, "unknown_cardinality", "sweep.axes.fanout.input"),
        (
            {
                "axes": {
                    "a": {"values": {"x": {"patches": {"root": {"value": {}}}}}},
                    "b": {"values": {"y": {"patches": {"root.child": {"value": 2}}}}},
                }
            },
            "patch_conflict",
            "sweep.axes.b",
        ),
    ]
    for data, code, path in cases:
        with pytest.raises(SweepExpansionError) as error:
            expand_sweep(Sweep.model_validate(data))
        assert (error.value.code, error.value.path) == (code, path)
    assert expand_sweep(None)[0].id == "default"


def test_sweep_editor_expansion_budget():
    for width, seconds in ((10, 0.2), (100, 2.0)):
        sweep = Sweep.model_validate(
            {
                "axes": {
                    "a": {"path": "params.a", "values": list(range(width))},
                    "b": {"path": "params.b", "values": list(range(width))},
                }
            }
        )
        start = perf_counter()
        assert count_sweep(sweep) == len(expand_sweep(sweep)) == width**2
        elapsed = perf_counter() - start
        assert elapsed < seconds, f"{width**2} rows expanded in {elapsed:.3f}s"
