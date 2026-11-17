from __future__ import annotations

import math

import pytest

from feedbax.contracts.training import LrScheduleSpec
from feedbax.training.optimizers import learning_rate_schedule


pytestmark = pytest.mark.feedbax_contract


def _expected_cosine(peak: float, *, alpha: float, progress: float) -> float:
    terminal = peak * alpha
    return terminal + 0.5 * (1.0 + math.cos(math.pi * progress)) * (peak - terminal)


@pytest.mark.parametrize("step", [0, 5, 12])
def test_constant_lr_schedule_holds_its_rate(step: int) -> None:
    schedule = LrScheduleSpec(kind="constant", learning_rate_0=0.03)

    assert float(
        learning_rate_schedule(schedule)(100 + step)
    ) == pytest.approx(0.03)


def test_warmup_cosine_schedule_pins_representative_values() -> None:
    schedule = LrScheduleSpec(
        kind="warmup_cosine",
        learning_rate_0=0.1,
        total_steps=10,
        constant_lr_iterations=4,
        warmup_init_fraction=0.1,
        cosine_annealing_alpha=0.2,
    )

    samples = {
        0: 0.01,
        4: 0.1,
        7: _expected_cosine(0.1, alpha=0.2, progress=0.5),
        10: 0.02,
    }
    for step, expected in samples.items():
        assert float(
            learning_rate_schedule(schedule)(step)
        ) == pytest.approx(expected)


def test_warmup_cosine_holds_terminal_rate_after_authored_endpoint() -> None:
    schedule = LrScheduleSpec(
        kind="warmup_cosine",
        learning_rate_0=0.03,
        total_steps=3500,
        constant_lr_iterations=1000,
        warmup_init_fraction=0.1,
        cosine_annealing_alpha=0.001,
    )

    for step in (3500, 3501, 4499, 4500):
        assert float(
            learning_rate_schedule(schedule)(step)
        ) == pytest.approx(3e-5)


def test_delayed_cosine_schedule_pins_representative_values() -> None:
    schedule = LrScheduleSpec(
        kind="delayed_cosine",
        learning_rate_0=0.1,
        total_steps=8,
        constant_lr_iterations=3,
        cosine_annealing_alpha=0.2,
    )

    samples = {
        0: 0.1,
        3: 0.1,
        5: _expected_cosine(0.1, alpha=0.2, progress=2 / 5),
        8: 0.02,
    }
    for step, expected in samples.items():
        assert float(
            learning_rate_schedule(schedule)(step)
        ) == pytest.approx(expected)
