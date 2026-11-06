import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from feedbax.objectives.loss import CompositeLoss, StopAtGoalLoss, TargetSpec
from feedbax.runtime.mapping import WhereDict
from feedbax.runtime.state import CartesianState
from feedbax.tasks.task import TaskTrialSpec, TrialTimeline


def _states(pos, vel=None):
    effector = CartesianState(pos=pos, vel=jnp.zeros_like(pos) if vel is None else vel)
    return SimpleNamespace(mechanics=SimpleNamespace(effector=effector))


def _specs(goal, start=1):
    return TaskTrialSpec(
        inits=WhereDict(),
        targets=WhereDict({"mechanics.effector.pos": TargetSpec(value=goal)}),
        inputs=None,
        timeline=TrialTimeline(target_start_step=start),
    )


@pytest.mark.parametrize("method", ["softmin", "soft-or"])
def test_stop_at_goal_scores_only_complete_windows_within_each_trial(method):
    goals = jnp.array([[2.0, 4.0], [5.0, -3.0]])
    pos = jnp.broadcast_to(goals[:, None, :], (2, 5, 2)).at[1, 4, 0].add(2.0)
    vel = jnp.zeros_like(pos).at[0, 3, 0].set(1.0)
    if method == "soft-or":
        drift = jnp.array([[0.0, 1.0], [2.0, 0.0], [-1.0, 3.0]])
        goals = goals[:, None, :] + drift
        pos = pos.at[:, 2:, :].add(drift)
    loss = StopAtGoalLoss(
        method=method, window_len=2, std_dist=1.0, std_vel=1.0, vel_weight=4.0, tau=1.0
    )
    objective = CompositeLoss(terms={"stop": loss}, weights={"stop": 3.0})
    result = objective(_states(pos, vel), _specs(goals, start=2), None)

    # The scored step costs are [0, 4, 0] and [0, 0, 4], so only trial 1
    # has a complete window with zero cost.
    if method == "softmin":
        expected = [2 - math.log(2), -math.log1p(math.exp(-2))]
    else:
        miss = 1 / (1 + math.exp(-2))
        expected = [-math.log(1 - (1 - loss.eps) * p) for p in (miss**2, 0.5 * miss)]
    assert result["stop"].value == pytest.approx(jnp.array(expected))
    assert result.total == pytest.approx(3 * sum(expected) / 2)


def test_soft_or_is_finite_when_no_window_nears_a_shared_goal():
    loss = StopAtGoalLoss(method="soft-or", window_len=2)
    specs = _specs(jnp.zeros(2, dtype=jnp.float16))

    @jax.jit
    @jax.value_and_grad
    def objective(pos):
        return loss(_states(pos), specs, None).total

    near, _ = objective(jnp.zeros((2, 4, 2), dtype=jnp.float16))
    far, grad = objective(jnp.ones((2, 4, 2), dtype=jnp.float16))
    assert near < far
    assert far == pytest.approx(-math.log(loss.eps))
    assert jnp.all(jnp.isfinite(grad))


def test_window_length_must_fit_scored_steps():
    with pytest.raises(ValueError, match="window_len"):
        StopAtGoalLoss(window_len=0)

    loss = StopAtGoalLoss(window_len=2)
    with pytest.raises(ValueError, match="exceeds the number of scored steps"):
        loss(_states(jnp.zeros((1, 3, 2))), _specs(jnp.zeros(2), start=2), None)
