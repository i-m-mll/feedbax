"""Check recorded Mean replay and figure samples against the same independent reference."""

from contextlib import nullcontext
import json

import numpy as np
import pytest

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.dataset import TrialFigureTime
from feedbax.contracts.jobs import JobReceipt
from feedbax.contracts.trial_clock import TrialClockChoice, TrialMeanOptions, TrialMeanSelection
from feedbax.contracts.trial_record import TaskTimelineCue, TrialRecord, TrialTerm
from feedbax.execution.trial_clock import mean_trial_paths
from feedbax.execution.trial_record import aggregate_trials
from feedbax.dataset import Dataset, Variable
from feedbax.persistence.canonical_evaluate import EvaluationResult
from feedbax.persistence.evaluation_trial_records import RecordedTrialLayout

_POSITION = "/v1/model/state/hand/pos"
_TIME = np.arange(9) * 0.5
_TRIALS = ("first", "second", "third")
_START = (2, 4, 1)
_END = (7, 6, 8)
_START_POINT = np.array([-0.1, 0.2], dtype=np.float32)
_TARGETS = np.array([[0.3, -0.2], [0.3, 0.1]], dtype=np.float32)


def _sample(trial, step):
    return (10 * trial + step**2, 5 * trial - step)


@pytest.fixture
def unequal_epochs():
    """Two trials reach Move at different times. Samples exist after their last epoch."""
    axes = ("condition", "trial")
    layout = RecordedTrialLayout(
        epochs=(("hold", "Hold"), ("move", "Move")),
        positions=(_POSITION,), configurations=(),
        discrete_variables=(
            TrialClockChoice(id="direction", label="Direction"),
            TrialClockChoice(id="start", label="Start"),
            TrialClockChoice(id="target", label="Target"),
        ),
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
            "/v1/task/variable/start": Variable(np.broadcast_to(_START_POINT, (1, 3, 2)), (*axes, "xy")),
            "/v1/task/variable/target": Variable(_TARGETS[[0, 0, 1]][None], (*axes, "xy")),
            _POSITION: Variable(
                np.array([[[_sample(trial, step) for step in range(len(_TIME))] for trial in range(3)]]),
                (*axes, "state_time", "xy"), units="m", frame="world",
            ),
        },
        coords={
            "condition": Variable(np.array(["condition-id"]), ("condition",)),
            "trial": Variable(np.array(_TRIALS), ("trial",)),
            "state_time": Variable(_TIME, ("state_time",), units="s", trial_time=TrialFigureTime()),
            "clock": Variable(_TIME, ("state_time",), units="s", trial_time=TrialFigureTime()),
            "xy": Variable(np.array(["x", "y"]), ("xy",)),
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


def test_mean_shelf_pools_epoch_rows_and_held_samples_on_the_alignment_clock():
    def trial(name, start, end, signal, term):
        return TrialRecord(
            id=name, inits={}, inputs={"target": [0] * 4}, present={"target": signal},
            variables={"group": name}, variable_defined={},
            timeline=[TaskTimelineCue(epoch="move", label="Move", start=start, end=end, outcome=1)],
            terms={"cost": TrialTerm(steps=term, covered=True)},
        )

    first = trial("first", 1, 4, [False, True, True, False], [True, False, True, False])
    second = trial("second", 2, 3, [False, False, True, False], [False, True, True, False])
    first.intervention_windows["pulse"] = [False, True, True, False]
    second.intervention_windows["only-second"] = [False, False, True, False]
    aggregate = aggregate_trials([item.model_copy(update={"variables": {}}) for item in (first, second)], n_steps=4)
    assert aggregate.interventions == {
        "pulse": [0, .5, .5, 0], "only-second": [0, 0, .5, 0],
    }
    options = TrialMeanOptions(
        by=[TrialClockChoice(id="group", label="Group")],
        align=[TrialClockChoice(id="move", label="Move")],
    )
    result = mean_trial_paths([first, second], np.arange(5), options, TrialMeanSelection(align="move"))
    assert len(result.conditions) == 2
    assert result.shelf.times == [-2, -1, 0, 1, 2, 3]
    assert result.shelf.signals["target"] == [0, 0, 1, .5, 0]
    assert result.shelf.terms["cost"].steps == [0, 1, .5, .5, 0]
    assert result.shelf.interventions == {
        "pulse": [0, 0, .5, .5, 0], "only-second": [0, 0, .5, 0, 0],
    }
    (move,) = result.shelf.epochs
    assert (move.start.mean, move.end.mean, move.end.lower, move.end.upper) == (0, 2, 1, 3)


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
    assert [(choice.id, choice.label) for choice in one.mean_options.by] == [
        ("direction", "Direction"), ("start", "Start"), ("target", "Target"),
    ]
    assert [choice.id for choice in one.mean_options.align] == ["hold", "move"]

    aligned = result.replay(mean=TrialMeanSelection(align="move", by=["direction"]))
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


def test_saved_figure_mean_uses_the_same_aligned_sample_reference(unequal_epochs, tmp_path):
    from feedbax.contracts.dataset import DATASET_MEDIA_TYPE
    from feedbax.contracts.effective_success import BoundMember
    from feedbax.contracts.figure_document import FigureDocument
    from feedbax.contracts.request_identity import decode_request, encode_request
    from feedbax.contracts.request_refs import InputMember, ProductRef
    from feedbax.contracts.request_specs import RenderRequestSpec
    from feedbax.host.light_worker import render_saved_figure
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.persistence.dataset import publish_dataset

    document = FigureDocument.model_validate({
        "data": ["states"],
        "layout": [{"name": "path", "row": 1, "col": 1}],
        "layers": [
            {
                "name": "path", "panel": "path", "role": "states", "mark": "line",
                "group": ["condition", "trial"],
                "encode": {"x": "clock", "y": {
                    "variable": _POSITION, "select": {"xy": {"index": 0}},
                }},
            },
            {
                "name": "authored", "panel": "path", "role": "states", "mark": "line",
                "encode": {"x": "state_time", "y": "state_time"},
            },
        ],
    })
    store = ImmutableArtifactBlobProvider(tmp_path)
    ref = publish_dataset(store, unequal_epochs)
    encoded = encode_request(RenderRequestSpec(document=document), {"states": (InputMember(
        key="states", reference=ProductRef(request_id="evaluate:" + "a" * 64, role="states"),
    ),)})
    request = decode_request(encoded.data, encoded.documents.__getitem__)
    bindings = {"states": (BoundMember(
        key="states", content=ref.content.model_copy(update={"media_type": DATASET_MEDIA_TYPE}),
        descriptor={"descriptor_sha256": ref.descriptor_sha256},
    ),)}
    plain = json.loads(render_saved_figure(store, request, bindings))
    rendered = json.loads(render_saved_figure(store, request, bindings, mean=TrialMeanSelection(align="move")))
    assert rendered["layout"]["meta"]["panels"]["path"]["time"]["zero_epoch"] == "move"
    traces = rendered["data"]
    assert traces[-1] == plain["data"][-1]
    assert [trace["meta"]["group"]["trial"] for trace in traces[:-1] if "trial" in trace["meta"]["group"]] == list(_TRIALS)
    labels = ("Direction=0 · Target=[0.3, -0.2]", "Direction=1 · Target=[0.3, 0.1]")
    for direction, (indices, label) in enumerate(zip(((0, 1), (2,)), labels, strict=True)):
        group = {
            "condition": "condition-id", "direction": direction,
            "start": canonical_json_v2_bytes(_START_POINT.tolist()).decode(),
            "target": canonical_json_v2_bytes(_TARGETS[direction].tolist()).decode(),
        }
        upper, lower, average = [trace for trace in traces[:-1] if trace["meta"]["group"] == group]
        assert {trace["name"] for trace in (upper, lower, average)} == {label}
        assert len({trace["legendgroup"] for trace in (upper, lower, average)}) == 1
        time, mean, mins, maxes = _reference(indices, _START)
        for trace, values in ((average, mean), (lower, mins), (upper, maxes)):
            assert trace["x"] == time
            assert trace["y"] == [xy[0] for xy in values]
        assert lower["fill"] == "tonexty"


def test_stacked_figure_matches_the_bound_member_with_repeated_trial_ids(unequal_epochs):
    from feedbax.contracts.project_sweep import SweepCoordinate, SweepInputMember
    from feedbax.dataset_sweep import stack_dataset_members
    from feedbax.host.light_worker import _figure_recorded_dataset
    from feedbax.plot.figure_trial_clock import FigureTrialSource

    descriptor = unequal_epochs.descriptor
    start = unequal_epochs.data("epoch_start").copy()
    start[0, 1, 1] = 3
    other = Dataset.bind(
        descriptor,
        tensor_shapes={name: descriptor.shape(name) for name in descriptor.tensor_names},
        load_tensor=lambda name, index: start[index] if name == "epoch_start" else unequal_epochs.data(name, index),
    )
    plotted = stack_dataset_members([
        (SweepInputMember("first-run", {"mass": SweepCoordinate("light", value=1)}), unequal_epochs),
        (SweepInputMember("second-run", {"mass": SweepCoordinate("heavy", value=2)}), other),
    ])
    source = FigureTrialSource(plotted, [
        _figure_recorded_dataset(unequal_epochs, frozenset({"first-run"})),
        _figure_recorded_dataset(other, frozenset({"second-run"})),
    ], plotted.data)

    trial, _ = source.project({"mass": (1,), "trial": (1,)})
    assert trial.id == "second"
    assert trial.timeline[1].start == 3
