from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from feedbax.objectives.loss import EffectorStraightPathLoss, TargetSpec
from feedbax.runtime.mapping import WhereDict
from feedbax.runtime.state import CartesianState
from feedbax.tasks.task import SimpleReaches, TaskTrialSpec


class _TimeMajorTarget(TargetSpec):
    @property
    def batch_axes(self):
        return TargetSpec(value=1)


@pytest.mark.parametrize(
    "normalize_by, target_type",
    [("actual", TargetSpec), ("goal", TargetSpec), ("goal", _TimeMajorTarget)],
)
def test_straight_path_loss_and_gradient_on_mixed_batch(normalize_by, target_type):
    moving_end = [0, 0] if normalize_by == "actual" else [2, 1]
    positions = jnp.array(
        [
            [[0, 0], [0, 0], [1, 0], [2, 0]],  # Initial hold.
            [[0, 0], [3, 0], [3, 2], [3, 4]],  # Length 7, reference length 5.
            [[1, 2], [1, 2], [1, 2], [1, 2]],  # Stationary.
            [[0, 0], [1, 0], [1, 1], moving_end],  # Motion with zero reference length.
            [[0, 0], [0, 0], [0, 0], [0, 0]],  # Stationary away from goal.
        ],
        dtype=float,
    )
    goals = jnp.array([[2, 0], [0, 5], [1, 2], [0, 0], [2, 0]], dtype=float)
    loss = EffectorStraightPathLoss(normalize_by=normalize_by)

    def objective(pos, goal):
        states = SimpleNamespace(mechanics=SimpleNamespace(effector=CartesianState(pos=pos)))
        if target_type is _TimeMajorTarget:
            goal = jnp.stack((goal + 1, goal + 2, goal))
        specs = TaskTrialSpec(
            inits=WhereDict(),
            targets=WhereDict({"mechanics.effector.pos": target_type(value=goal)}),
            inputs=None,
        )
        result = loss(states, specs, None)
        return result.total, result.value

    if normalize_by == "actual":
        hold_grad = [[1 / 2, 0], [-1 / 2, 0], [0, 0], [0, 0]]
        corner_grad = [[-4 / 125, 28 / 125], [1 / 5, -1 / 5], [0, 0], [-21 / 125, -3 / 125]]
    else:
        hold_grad = [[1 / 2, 0], [-1 / 2, 0], [0, 0], [1 / 2, 0]]
        corner_grad = [[-1 / 5, 7 / 25], [1 / 5, -1 / 5], [0, 0], [0, 1 / 5]]
    expected_loss = jnp.array([1, 7 / 5, 0, float("inf"), 0])
    zero_grad = [[0, 0]] * 4
    expected_grad = jnp.array([hold_grad, corner_grad, zero_grad, zero_grad, zero_grad])
    evaluate = jax.jit(jax.value_and_grad(objective, has_aux=True))
    for order in (slice(None), slice(None, None, -1)):
        (total, per_trial), grad = evaluate(positions[order], goals[order])
        assert_allclose(per_trial, expected_loss[order])
        assert total == float("inf")
        assert_allclose(grad, expected_grad[order] / len(positions), atol=1e-7)


def test_simple_reaches_straight_path_normalizes_by_task_goal():
    task = SimpleReaches(
        n_steps=4,
        loss_func=EffectorStraightPathLoss(normalize_by="goal"),
        workspace=[[-3, -3], [3, 3]],
        fixed_endpoints=[[0, 0], [2, 0]],
    )
    specs = task.get_validation_trials(jax.random.key(0))
    positions = jnp.array([[[0.0, 0.0], [0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]])

    def objective(pos):
        states = SimpleNamespace(mechanics=SimpleNamespace(effector=CartesianState(pos=pos)))
        result = task.loss_func(states, specs, None)
        return result.total, result.value

    (value, per_trial), grad = jax.jit(jax.value_and_grad(objective, has_aux=True))(positions)
    assert value == pytest.approx(0.5)
    assert per_trial.shape == (1,)
    assert_allclose(per_trial, [0.5])
    assert_allclose(grad, [[[0.25, 0], [-0.5, 0], [0, 0], [0.5, 0]]])
