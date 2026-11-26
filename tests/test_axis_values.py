"""A step's axes gather its runs by value, whatever key each producer stored the value under, in the window map and in
a selection of runs by their values."""

from types import SimpleNamespace

from feedbax.contracts.project_sweep import SweepCoordinate
from feedbax.planner.project_plan import PlannedRow, RunKey
from feedbax.web.services.axis_values import gather_axes
from feedbax.web.services.trial_scope import SweepValuesRuns, runs_scope


def _row(step, row, **coordinates):
    return PlannedRow(RunKey("base", step, row), "evaluate", None, coordinates)


def _seeds(train, seeds):
    """Evaluate rows reading ``train``'s runs, whose seeds it stores as ``value``, ``value-1``, … in order."""
    return [
        _row("evaluate", f"{train}-{seed}", seed=SweepCoordinate("value" if index == 0 else f"value-{index}", None, seed, axis_label="Seed"))
        for index, seed in enumerate(seeds)
    ]


def test_runs_of_one_seed_from_two_producers_share_a_value_and_seeds_are_in_order():
    rows = [*_seeds("train", [0, 2, 3]), *_seeds("train-2", [0, 1, 2, 3])]
    (axis,), values = gather_axes(rows)
    assert axis.name == "Seed"
    assert [(value.name, value.value_label) for value in axis.values] == [
        ("Seed 0", "0"), ("Seed 1", "1"), ("Seed 2", "2"), ("Seed 3", "3"),
    ]
    ids = {value.value_label: value.id for value in axis.values}
    assert values[RunKey("base", "evaluate", "train-0"), "seed"] == values[RunKey("base", "evaluate", "train-2-0"), "seed"] == ids["0"]
    assert values[RunKey("base", "evaluate", "train-2"), "seed"] == values[RunKey("base", "evaluate", "train-2-2"), "seed"] == ids["2"]
    assert len(set(ids.values())) == 4


def test_set_entries_with_one_key_and_different_names_stay_apart():
    rows = [
        _row("evaluate", "a", size=SweepCoordinate("entry", "small")),
        _row("evaluate", "b", size=SweepCoordinate("entry", "large")),
    ]
    (axis,), values = gather_axes(rows)
    assert [value.name for value in axis.values] == ["small", "large"]
    assert values[RunKey("base", "evaluate", "a"), "size"] != values[RunKey("base", "evaluate", "b"), "size"]


def test_a_seed_of_an_evaluate_reading_two_producers_is_selected_as_a_rule():
    rows = [*_seeds("train", [0, 2, 3]), *_seeds("train-2", [0, 1, 2, 3])]
    (axis,), _ = gather_axes(rows)
    two = next(value.id for value in axis.values if value.value_label == "2")
    project = SimpleNamespace(steps={"evaluate": SimpleNamespace(sweep=None)})
    scope = runs_scope(
        project, SimpleNamespace(rows=rows), RunKey("base", "evaluate", "train-2"), ("train-2", "train-2-2"),
        {"seed": [two]},
    )
    assert isinstance(scope, SweepValuesRuns)
    assert scope.rule == "seed=2"
