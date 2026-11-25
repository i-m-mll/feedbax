"""Check recorded Mean replay and figure samples against the same independent reference."""

from contextlib import nullcontext

import numpy as np
import pytest

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.jobs import JobReceipt
from feedbax.contracts.trial_clock import TrialClockChoice, TrialMeanSelection
from feedbax.dataset import Dataset, Variable
from feedbax.persistence.canonical_evaluate import EvaluationResult
from feedbax.persistence.evaluation_trial_records import RecordedTrialLayout

_POSITION = "/v1/model/state/hand/pos"
_TIME = np.arange(9) * 0.5
_TRIALS = ("first", "second", "third")
_START = (2, 4, 1)
_END = (7, 6, 8)


def _sample(trial, step):
    return (10 * trial + step**2, 5 * trial - step)


@pytest.fixture
def unequal_epochs():
    """Two trials reach Move at different times. Samples exist after their last epoch."""
    axes = ("condition", "trial")
    layout = RecordedTrialLayout(
        epochs=(("hold", "Hold"), ("move", "Move")),
        positions=(_POSITION,), configurations=(),
        discrete_variables=(TrialClockChoice(id="direction", label="Direction"),),
    )
    return Dataset.from_arrays(
        {
            "epoch_start": Variable(
                np.array([[(0, move) for move in _START]]), (*axes, "epoch"),
            ),
            "epoch_end": Variable(
                np.array([list(zip(_START, _END, strict=True))]), (*axes, "epoch"),
            ),
            "epoch_outcome": Variable(np.ones((1, 3, 2), dtype=int), (*axes, "epoch")),
            "/v1/task/variable/direction": Variable(np.array([[0, 0, 1]]), axes),
            _POSITION: Variable(
                np.array([[[_sample(trial, step) for step in range(len(_TIME))] for trial in range(3)]]),
                (*axes, "state_time", "xy"), units="m", frame="world",
            ),
        },
        coords={
            "condition": Variable(np.array(["condition-id"]), ("condition",)),
            "trial": Variable(np.array(_TRIALS), ("trial",)),
            "state_time": Variable(_TIME, ("state_time",), units="s"),
        },
        attrs={"trial_record": layout.model_dump(mode="json")},
    )


def _reference(indices, starts):
    """Intersect integer-step ranges independently, with no padded samples."""
    overlap = range(max(-starts[i] for i in indices), min(_END[i] - starts[i] for i in indices) + 1)
    rows = [list(zip(*(_sample(i, step + starts[i]) for i in indices), strict=True)) for step in overlap]
    return (
        [step * 0.5 for step in overlap],
        [[sum(values) / len(values) for values in row] for row in rows],
        [[min(values) for values in row] for row in rows],
        [[max(values) for values in row] for row in rows],
    )


def test_saved_mean_groups_discrete_values_and_intersects_aligned_times(unequal_epochs, tmp_path, monkeypatch):
    sha = "a" * 64
    result = EvaluationResult(
        tmp_path, "job", "request", JobReceipt(manifest_id="manifest", manifest_sha256=sha),
        manifest_object=ContentObjectRef(object_id=f"content-object://sha256/{sha}", sha256=sha, size_bytes=0),
    )
    monkeypatch.setattr(result, "open_states", lambda: nullcontext(unequal_epochs))

    one = result.replay("second")
    assert one.drawn.kind == "trials"
    assert [trial.id for trial in one.drawn.trials] == ["second"]
    assert one.mean is None
    assert one.mean_options is not None
    assert [(choice.id, choice.label) for choice in one.mean_options.by] == [("direction", "Direction")]
    assert [choice.id for choice in one.mean_options.align] == ["hold", "move"]

    aligned = result.replay(mean=TrialMeanSelection(align="move"))
    assert aligned.mean is not None and aligned.position is not None
    assert aligned.drawn.kind == "trials" and len(aligned.drawn.trials) == 3
    assert [condition.values for condition in aligned.mean.conditions] == [{"direction": 0}, {"direction": 1}]
    for condition, indices in zip(aligned.mean.conditions, ((0, 1), (2,)), strict=True):
        time, mean, lower, upper = _reference(indices, _START)
        assert list(condition.trials) == [_TRIALS[i] for i in indices]
        assert condition.span is not None
        assert (condition.span.start, condition.span.end) == (time[0], time[-1])
        assert condition.time == time
        assert condition.states[aligned.position].mean == mean
        assert condition.states[aligned.position].lower == lower
        assert condition.states[aligned.position].upper == upper
    move = aligned.mean.conditions[0].epochs[1]
    assert (move.start.lower, move.start.mean, move.start.upper) == (0.0, 0.0, 0.0)
    assert (move.end.lower, move.end.mean, move.end.upper) == (1.0, 1.75, 2.5)

    pooled = result.replay(mean=TrialMeanSelection(by=[]))
    assert pooled.mean is not None
    (condition,) = pooled.mean.conditions
    assert condition.values == {}
    assert condition.time == _reference((0, 1, 2), (0, 0, 0))[0]
