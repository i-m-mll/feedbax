"""Stored trial members retain scientific identities through validation."""

import jax.random as jr
import numpy as np
import pytest

from feedbax.compiler.model_document import ResolvedModelInterface
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.contracts.task_document import StoredTaskDocument, StoredTaskDraftError, ValidationTrial
from feedbax.expression_builtins import register_builtins
from feedbax.tasks.document import DocumentTask, TaskParameters, compile_task


def _task(trials):
    return StoredTaskDocument.model_validate({
        "id": "trial-rules",
        "fixed_steps": 8,
        "variables": {
            "drawn": {"label": "Drawn", "type": {"base": "real"}, "value": "U(3, 7)"},
            "assigned": {"label": "Assigned", "type": {"base": "real"}, "value": 0},
        },
        "epochs": {
            "prep": {"label": "Preparation", "end": {"kind": "length", "length": "4 st"}},
            "move": {"label": "Movement", "end": {"kind": "rest"}},
        },
        "signals": {},
        "validation_trials": trials,
    })


def _compile(task):
    functions = FunctionRegistry()
    register_builtins(functions)
    functions.seal()
    return compile_task(task, functions, model=ResolvedModelInterface(0.01, (), (), (), ()))


def test_listed_trial_draws_follow_ids_and_distinguish_epoch_lengths():
    trials = {
        "early": {"label": "Short preparation", "variables": {"assigned": "1 + 2"}, "epoch_steps": {"prep": 1}},
        "late": {"label": "Long preparation", "variables": {"assigned": "1 + 2"}, "epoch_steps": {"prep": 3}},
    }
    program = _compile(_task(trials).lower())
    assert isinstance(program, DocumentTask), program
    assert program.condition_ids == ("early", "late")
    keys = jr.key(19)
    validation = program.validation(TaskParameters(), keys)
    np.testing.assert_array_equal(validation.variables["assigned"], [3, 3])
    np.testing.assert_array_equal(validation.epochs.ends[:, 0], [1, 3])
    assert validation.variables["drawn"][0] != validation.variables["drawn"][1]

    reversed_program = _compile(_task(dict(reversed(tuple(trials.items())))).lower())
    assert isinstance(reversed_program, DocumentTask), reversed_program
    assert reversed_program.condition_ids == ("late", "early")
    reversed_validation = reversed_program.validation(TaskParameters(), keys)
    np.testing.assert_array_equal(reversed_validation.variables["drawn"], validation.variables["drawn"][::-1])
    np.testing.assert_array_equal(reversed_validation.epochs.ends, validation.epochs.ends[::-1])


def test_empty_trial_member_retains_its_key_but_the_default_is_sampled():
    sampled = _task({}).lower()
    assert sampled.validation_trials == (ValidationTrial(),)
    assert sampled.condition_ids == ()
    with pytest.raises(ValueError, match="label"):
        _task({"control": {}})
    listed = _task({"control": {"label": "Baseline"}}).lower()
    assert listed.condition_ids == ("control",)

    with pytest.raises(ValueError, match="Only the default empty validation trial",):
        type(listed).model_validate(listed.model_dump() | {"validation_trials": [{"epoch_steps": {"prep": 1}}]})


@pytest.mark.parametrize("expression", ["U(1, 2)", "drawn + 1", "1 / 0"])
def test_validation_values_must_be_defined_constants(expression):
    task = _task({"check": {"label": "Check", "variables": {"assigned": expression}}}).lower()
    problems = _compile(task)
    assert not isinstance(problems, DocumentTask)
    assert any(problem.path[:4] == ("validation_trials", "check", "variables", "assigned") for problem in problems)


def test_family_and_listed_trials_lower_and_run_identically_in_generated_order():
    family = _task({"reach": {
        "label": "Reach {direction}, repeat {repeat}",
        "over": {"direction": "0..1", "repeat": "0..1"},
        "variables": {"assigned": "polar(0.15 m, direction * 90°)"},
        "epoch_steps": {"prep": "1 + repeat * 2"},
    }})
    listed = _task({
        f"reach-{direction}-{repeat}": {
            "label": f"Direction {direction} repeat {repeat}",
            "variables": {"assigned": f"polar(0.15 m, {direction} * 90°)"},
            "epoch_steps": {"prep": 1 + repeat * 2},
        }
        for direction in (0, 1) for repeat in (0, 1)
    })

    def positions(task):
        raw = task.model_dump(mode="json")
        raw["variables"]["assigned"].update(
            type={"base": "real", "shape": [2], "dimensions": {"length": 1}},
            value="[0 m, 0 m]",
        )
        return StoredTaskDocument.model_validate(raw)

    family, listed = positions(family), positions(listed)
    expanded = family.lower()
    together = listed.lower()
    assert expanded == together
    assert expanded.condition_ids == ("reach-0-0", "reach-0-1", "reach-1-0", "reach-1-1")
    assert family.validation_trial_families() == {"reach": expanded.condition_ids}
    assert family.lower(first=True).condition_ids == ("reach-0-0",)

    programmes = (_compile(expanded), _compile(together))
    assert all(isinstance(programme, DocumentTask) for programme in programmes), programmes
    from_family, from_rows = (programme.validation(TaskParameters(), jr.key(3)) for programme in programmes)
    starts = np.array([[0.15, 0], [0.15, 0], [0, 0.15], [0, 0.15]])
    np.testing.assert_allclose(from_family.variables["assigned"], starts, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(from_family.epochs.ends[:, 0], [1, 3, 1, 3])
    np.testing.assert_array_equal(from_family.variables["assigned"], from_rows.variables["assigned"])
    np.testing.assert_array_equal(from_family.variables["drawn"], from_rows.variables["drawn"])
    np.testing.assert_array_equal(from_family.epochs.ends, from_rows.epochs.ends)


def test_structural_error_uses_the_authored_member_and_its_failing_combination():
    task = _task({"check": {"label": "Check {k}", "over": {"k": "1..2"}, "epoch_steps": {"prep": "4 % (k - 2)"}}})
    assert task.lower(first=True).validation_trials[0].epoch_steps == {"prep": 0}
    with pytest.raises(StoredTaskDraftError, match=r"zero.*k=2") as error:
        task.lower()
    assert error.value.path == ("validation_trials", "check", "epoch_steps", "prep")


def test_authored_trial_rule_size_only_changes_with_the_count_digits():
    def authored(count):
        return _task({"cohort": {
            "label": "Check {k}",
            "over": {"k": f"linspace(0, 1, {count})"},
            "variables": {"assigned": "k"},
        }})

    small, large = authored(10), authored(1000)
    encoded = large.model_dump_json()
    assert len(encoded) - len(small.model_dump_json()) == 2
    assert StoredTaskDocument.model_validate_json(encoded) == large
    assert large.lower(first=True).condition_ids == ("cohort-0",)


def test_generated_labels_must_be_distinct_and_leave_trial_identity_unchanged():
    members = {"reach": {"label": "Reach {k}", "over": {"k": "0..1"}}}
    original = _task(members)
    renamed = _task({"reach": members["reach"] | {"label": "Goal {k}"}})
    assert renamed.lower() == original.lower()

    duplicate = _task(members | {"plain": {"label": "Reach 1"}})
    with pytest.raises(StoredTaskDraftError, match="Several validation trials have the label") as error:
        duplicate.lower()
    assert error.value.path == ("validation_trials", "plain", "label")
