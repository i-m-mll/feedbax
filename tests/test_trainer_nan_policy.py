from __future__ import annotations

import optax
import pytest

from feedbax.training.trainer import TaskTrainer

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.no_silent_substitution_contract]


def test_task_trainer_nan_policy_defaults_to_raise() -> None:
    trainer = TaskTrainer(optimizer=optax.sgd(0.1), checkpointing=False)

    assert trainer.on_nan == "raise"


def test_task_trainer_nan_policy_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="on_nan"):
        TaskTrainer(
            optimizer=optax.sgd(0.1),
            checkpointing=False,
            on_nan="restore_last_good",  # type: ignore[arg-type]
        )
