"""Stored trial members retain scientific identities through validation."""

import jax.random as jr
import numpy as np
import pytest

from feedbax.compiler.model_document import ResolvedModelInterface
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.contracts.task_document import StoredTaskDocument, ValidationTrial
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
        "early": {"variables": {"assigned": "1 + 2"}, "epoch_steps": {"prep": 1}},
        "late": {"variables": {"assigned": "1 + 2"}, "epoch_steps": {"prep": 3}},
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
    listed = _task({"control": {}}).lower()
    assert listed.condition_ids == ("control",)

    with pytest.raises(ValueError, match="Only the default empty validation trial",):
        type(listed).model_validate(listed.model_dump() | {"validation_trials": [{"epoch_steps": {"prep": 1}}]})


@pytest.mark.parametrize("expression", ["U(1, 2)", "drawn + 1"])
def test_validation_values_refuse_draws_and_other_variables(expression):
    task = _task({"check": {"variables": {"assigned": expression}}}).lower()
    problems = _compile(task)
    assert not isinstance(problems, DocumentTask)
    assert any(problem.path[:4] == ("validation_trials", "check", "variables", "assigned") for problem in problems)
