from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from feedbax._mapping import WhereDict
from feedbax.loss import TargetSpec
from feedbax.xabdeef.losses import EffectorFixationLoss, delayed_reach_loss


def _state_with_effector_pos(pos):
    return SimpleNamespace(
        mechanics=SimpleNamespace(
            effector=SimpleNamespace(pos=jnp.asarray(pos, dtype=jnp.float32))
        )
    )


def _trial_with_effector_target(target, hold):
    return SimpleNamespace(
        inputs=SimpleNamespace(hold=jnp.asarray(hold, dtype=jnp.float32)),
        targets=WhereDict(
            {
                (lambda state: state.mechanics.effector.pos): TargetSpec(
                    jnp.asarray(target, dtype=jnp.float32)
                )
            }
        ),
    )


def test_effector_fixation_loss_masks_by_hold_signal():
    states = _state_with_effector_pos(
        [
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        ]
    )
    trial_specs = _trial_with_effector_target(
        target=[
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ],
        hold=[[[1.0], [0.0], [1.0]]],
    )

    result = EffectorFixationLoss().term(states, trial_specs, model=None)

    assert result.shape == (1,)
    assert result[0] == pytest.approx(20.0)


def test_delayed_reach_loss_constructs_fixation_term():
    loss = delayed_reach_loss()

    assert isinstance(loss.terms["effector_fixation"], EffectorFixationLoss)
    assert loss.weights["effector_fixation"] == pytest.approx(1.0)
