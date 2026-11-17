from __future__ import annotations

from time import perf_counter

import pytest

from feedbax.contracts.matrix_core import apply_override_patches
from feedbax.contracts.project import Project
from feedbax.contracts.project_resolution import ResolvedStep, resolve_step
from feedbax.contracts.project_sweep import (
    ManualCombination,
    NamedAxis,
    RangeAxis,
    SampledAxis,
    SetValue,
    Sweep,
    SweepExpansionCode,
    SweepExpansionError,
    SweepInputMember,
    authored_row,
    axis_to_list,
    count_sweep,
    expand_sweep,
    inherit_sweep,
    sweep_patches,
    sweep_to_rows,
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


def test_generated_values_have_stable_keys_and_recorded_sampling():
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
    for compile_rows in (count_sweep, expand_sweep):
        with pytest.raises(SweepExpansionError) as error:
            compile_rows(sampled)
        assert error.value.code == SweepExpansionCode.MISSING_DRAWS
        assert error.value.path == "sweep.axes.draw.draws"

    axis = sampled.axes["draw"]
    assert isinstance(axis, SampledAxis)
    axis.draws = {"first": 7, "second": 4, "third": 7, "fourth": 9}
    reloaded = Sweep.model_validate_json(sampled.model_dump_json(exclude_unset=True))
    reread = reloaded.axes["draw"]
    assert isinstance(reread, SampledAxis)
    reread.sampler.distribution = "unregistered"
    reread.sampler.seed = 0
    assert count_sweep(sampled) == len(expand_sweep(sampled)) == 4
    assert expand_sweep(sampled) == expand_sweep(reloaded)
    assert [row.id for row in expand_sweep(sampled)] == [
        "draw=first",
        "draw=second",
        "draw=third",
        "draw=fourth",
    ]
    with pytest.raises(SweepExpansionError) as error:
        expand_sweep(sampled, max_rows=3)
    assert error.value.code == SweepExpansionCode.ROW_BUDGET
    reread.draws = {"first": 7}
    with pytest.raises(SweepExpansionError) as error:
        expand_sweep(reloaded)
    assert error.value.code == SweepExpansionCode.INVALID_GENERATION


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


def test_to_list_and_to_rows_preserve_coordinates_and_patches():
    rule = Sweep.model_validate(
        {
            "axes": {
                "gain": {"path": "gain", "range": {"start": 1, "stop": 3, "count": 3}},
                "sample": {
                    "path": "sample",
                    "sampler": {"distribution": "prior", "count": 2, "seed": 9},
                    "draws": {"one": 13, "two": 21},
                },
            },
            "exceptions": {"skip": {"where": {"axes": {"sample": ["two"]}}, "exclude": True}},
        }
    )
    loaded = Sweep.model_validate_json(rule.model_dump_json(exclude_unset=True))
    assert isinstance(loaded.axes["gain"], RangeAxis)
    assert isinstance(loaded.axes["sample"], SampledAxis)
    resolved = inherit_sweep(None, loaded, author="source")
    assert resolved is not None
    gain, sample = resolved.axes.values()
    assert isinstance(gain, RangeAxis)
    assert isinstance(sample, SampledAxis)
    listed = resolved.model_copy(
        update={"axes": {"gain": axis_to_list(gain), "sample": axis_to_list(sample)}}
    )
    original = expand_sweep(resolved)
    assert expand_sweep(listed) == original
    manual = sweep_to_rows(listed)
    assert isinstance(manual.combination, ManualCombination)
    assert len(manual.combination.manual) == 3
    assert expand_sweep(manual) == original

    gain_values = listed.axes["gain"]
    assert isinstance(gain_values, NamedAxis)
    first = original[0].coordinates["gain"].value_id
    value = gain_values.values[first]
    assert value is not None
    value.patches["gain"] = SetValue(value=12)
    edited = expand_sweep(listed)
    assert [row.id for row in edited] == [row.id for row in original]
    assert [row.coordinates for row in edited] == [row.coordinates for row in original]
    assert edited[0].patches["gain"] == SetValue(value=12)
    assert edited[1:] == original[1:]


def test_ordered_exceptions_intersect_rows_exclude_and_count_final_members():
    sweep = Sweep.model_validate(
        {
            "axes": {
                "gain": {"values": {"low": {}, "mid": {}, "high": {}}},
                "mode": {"values": {"a": {}, "b": {}}},
            }
        }
    )
    base = expand_sweep(sweep)
    altered = Sweep.model_validate(
        {
            **sweep.model_dump(exclude_unset=True),
            "exceptions": {
                "fix": {
                    "where": {
                        "axes": {"mode": ["a"]},
                        "rows": ["gain=low--mode=a", "gain=mid--mode=a", "gain=high--mode=b"],
                    },
                    "patches": {"setting": {"op": "add", "value": 7}},
                },
                "refine": {
                    "where": {"rows": ["gain=mid--mode=a"]},
                    "patches": {"setting": {"value": 9}},
                },
                "remove": {
                    "where": {"axes": {"gain": ["high"]}, "rows": ["gain=high--mode=b"]},
                    "exclude": True,
                },
                "empty": {"where": {"rows": []}, "patches": {"setting": {"value": 99}}},
            },
        }
    )
    final = expand_sweep(altered)
    assert count_sweep(altered) == len(final) == 5
    assert [row.id for row in final] == [row.id for row in base if row.id != "gain=high--mode=b"]
    assert [row.coordinates for row in final] == [row.coordinates for row in base[:-1]]
    changed = {row.id: row for row in final if row.patches}
    assert set(changed) == {"gain=low--mode=a", "gain=mid--mode=a"}
    assert apply_override_patches({}, sweep_patches(changed["gain=mid--mode=a"].patches)) == {
        "setting": 9
    }
    assert changed["gain=mid--mode=a"].patch_origins == {"setting": ("exception", "refine")}


def test_to_rows_pins_survivors_and_keeps_patch_exception_ids():
    sweep = Sweep.model_validate(
        {
            "axes": {"x": {"values": {"a": {}, "b": {}}}},
            "rows": {"manual": {}},
            "exceptions": {
                "patch": {"where": {"rows": ["x=a", "x=b"]}, "patches": {"setting": {"value": 7}}},
                "exclude": {"where": {"rows": ["x=b", "manual"]}, "exclude": True},
            },
        }
    )
    manual = sweep_to_rows(sweep)
    assert isinstance(manual.combination, ManualCombination)
    assert manual.combination.manual == [{"x": "a"}]
    patch = manual.exceptions["patch"]
    assert patch is not None and patch.where is not None and patch.where.rows == ["x=a"]
    assert expand_sweep(manual) == expand_sweep(sweep)

    del manual.exceptions["exclude"]
    rows = expand_sweep(manual)
    assert [row.id for row in rows] == ["manual", "x=a"]
    assert rows[1].patch_origins == {"setting": ("exception", "patch")}


def test_exception_write_after_ancestor_replacement_uses_its_own_order_and_op():
    sweep = Sweep.model_validate(
        {
            "rows": {"one": {}},
            "exceptions": {
                "add": {"where": {}, "patches": {"root.child": {"op": "add", "value": 1}}},
                "reset": {"where": {}, "patches": {"root": {"value": {"child": 2, "kept": 3}}}},
                "replace": {"where": {}, "patches": {"root.child": {"value": 4}}},
            },
        }
    )
    (row,) = expand_sweep(sweep)
    assert list(row.patches) == ["root", "root.child"]
    assert row.patches["root.child"] == SetValue(value=4)
    assert row.patch_origins == {
        "root": ("exception", "reset"),
        "root.child": ("exception", "replace"),
    }
    assert apply_override_patches({"root": {}}, sweep_patches(row.patches)) == {
        "root": {"child": 4, "kept": 3},
    }


def test_inherited_patch_origins_follow_exact_authored_writes():
    parent = inherit_sweep(
        None,
        Sweep.model_validate(
            {
                "axes": {
                    "gain": {
                        "values": {
                            "a": {
                                "patches": {
                                    "model.gain": {"value": 2},
                                    "model.offset": {"value": 3},
                                }
                            }
                        }
                    }
                },
                "rows": {"base": {"patches": {"setup": {"op": "add", "value": True}}}},
                "exceptions": {
                    "adjust": {
                        "where": {"rows": ["gain=a"]},
                        "patches": {"old": {"value": 1}, "keep": {"value": 8}},
                    },
                    "removed": {"where": {"rows": ["gain=a"]}, "exclude": True},
                },
            }
        ),
        author="parent",
    )
    child = inherit_sweep(
        parent,
        Sweep.model_validate(
            {
                "inherit_rows": False,
                "axes": {"gain": {"values": {"a": {"patches": {"model.gain": {"value": 2}}}}}},
                "rows": {
                    "new": {"from": "base", "patches": {"tag": {"op": "add", "value": "child"}}}
                },
                "exceptions": {
                    "adjust": {"patches": {"old": None, "new": {"value": 4}}},
                    "removed": None,
                },
            }
        ),
        author="child",
    )
    assert child is not None
    row, generated = expand_sweep(child)
    assert row.id == "new"
    assert row.patch_origins == {"setup": ("row", "base"), "tag": ("row", "new")}
    assert row.patch_authors == {"setup": "parent", "tag": "child"}
    assert generated.patch_origins == {
        "model.gain": ("sweep_value", "gain", "a"),
        "model.offset": ("sweep_value", "gain", "a"),
        "keep": ("exception", "adjust"),
        "new": ("exception", "adjust"),
    }
    assert generated.patch_authors == {
        "model.gain": "child",
        "model.offset": "parent",
        "keep": "parent",
        "new": "child",
    }
    assert "_patch_authors" not in child.model_dump_json()
    assert "_patch_sources" not in child.model_dump_json()


@pytest.mark.parametrize(
    ("where", "code", "path"),
    [
        ({"best": {"by": "loss", "k": 1}}, "invalid_exception", "sweep.exceptions.fix.where.best"),
        ({"rows": ["missing"]}, "invalid_exception", "sweep.exceptions.fix.where.rows"),
        (
            {"axes": {"gain": ["wrong"]}},
            "invalid_exception",
            "sweep.exceptions.fix.where.axes.gain",
        ),
        ({"axes": {"gain": ["a"]}}, "patch_conflict", "sweep.exceptions.fix.patches.gain"),
    ],
)
def test_exception_refuses_dynamic_selectors_unknown_ids_and_axis_patch_conflicts(
    where, code, path
):
    sweep = Sweep.model_validate(
        {
            "axes": {"gain": {"values": {"a": {"patches": {"gain": {"value": 2}}}}}},
            "exceptions": {"fix": {"where": where, "patches": {"gain": {"value": 3}}}},
        }
    )
    with pytest.raises(SweepExpansionError) as error:
        expand_sweep(sweep)
    assert (error.value.code, error.value.path) == (code, path)


def test_authored_base_row_observes_exceptions_and_unknown_generated_members():
    sweep = Sweep.model_validate(
        {
            "axes": {"fanout": {"input": "subject"}},
            "rows": {
                "kept": {"patches": {"value": {"value": 2}}},
                "hidden": {},
            },
            "exceptions": {
                "change": {"where": {"rows": ["kept"]}, "patches": {"value": {"value": 3}}},
                "hide": {"where": {"rows": ["hidden"]}, "exclude": True},
            },
        }
    )
    resolved = inherit_sweep(None, sweep, author="base-step")
    kept = authored_row(resolved, "kept")
    assert kept is not None
    assert kept.patches == {"value": SetValue(value=3)}
    assert kept.patch_origins == {"value": ("exception", "change")}
    assert kept.patch_authors == {"value": "base-step"}
    assert authored_row(resolved, "hidden") is None
    assert authored_row(resolved, "generated") is None

    inspect = Sweep.model_validate(
        {
            **sweep.model_dump(exclude_unset=True),
            "exceptions": {
                "inspect": {"where": {"rows": ["fanout=first"]}, "exclude": True},
            },
        }
    )
    with pytest.raises(SweepExpansionError) as error:
        authored_row(inspect, "kept")
    assert (error.value.code, error.value.path) == (
        SweepExpansionCode.UNKNOWN_CARDINALITY,
        "sweep.axes.fanout.input",
    )


def test_to_rows_can_pin_empty_fanout_without_inventing_an_implicit_row():
    sweep = Sweep.model_validate({"axes": {"fanout": {"input": "subject"}}})
    manual = sweep_to_rows(sweep, inputs={"subject": []})
    assert isinstance(manual.combination, ManualCombination)
    assert manual.combination.manual == []
    assert count_sweep(manual) == 0
    assert expand_sweep(manual) == ()
    assert expand_sweep(manual, inputs={"subject": [SweepInputMember("new")]}) == ()


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
