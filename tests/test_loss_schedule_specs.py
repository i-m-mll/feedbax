"""Tests for durable objective/loss schedule specifications."""

import pytest
from pydantic import ValidationError

from feedbax.objective_spec import (
    EpochMaskSpec,
    FiniteDifferenceLossSpec,
    MetricSpec,
    MovementEpochRampScheduleSpec,
    ObjectiveSpec,
    PowerLawScheduleSpec,
    SelectorAddressSpec,
    TargetStateLossSpec,
    TargetValueSpec,
    TaskTimelineSpec,
    TimelineEpochSpec,
    TimelineEventSpec,
    canonical_objective_payload,
)
from feedbax.provider import provider_manifest


def _delayed_reach_timeline() -> TaskTimelineSpec:
    return TaskTimelineSpec(
        n_steps=81,
        epochs=[
            TimelineEpochSpec(name="hold", index=0, length_range=(5, 15)),
            TimelineEpochSpec(name="target_on", index=1, length_range=(10, 20)),
            TimelineEpochSpec(name="movement", index=2),
        ],
        events=[TimelineEventSpec(name="go", epoch="movement")],
        movement_epoch="movement",
        go_event="go",
        metadata={"task_type": "DelayedReaches"},
    )


def _state_selector(selector: str, *, units: str | None = None) -> SelectorAddressSpec:
    return SelectorAddressSpec(
        selector=selector,
        kind="state",
        value_units=units,
        temporal_axis="time",
    )


def test_b_set_like_objective_spec_covers_masks_schedules_and_differences() -> None:
    spec = ObjectiveSpec(
        timeline=_delayed_reach_timeline(),
        terms=[
            TargetStateLossSpec(
                label="effector_pos_full_trial_powerlaw",
                selector=_state_selector("state.mechanics.effector.pos", units="m"),
                target=TargetValueSpec(
                    kind="task_target",
                    target_key="mechanics.effector.pos",
                ),
                metric=MetricSpec(kind="squared_l2", axis="coordinate"),
                schedule=PowerLawScheduleSpec(
                    exponent=6,
                    window="full_trial",
                    start_value=0.0,
                    end_value=1.0,
                ),
            ),
            TargetStateLossSpec(
                label="effector_pos_post_go_powerlaw",
                selector=_state_selector("state.mechanics.effector.pos", units="m"),
                target=TargetValueSpec(
                    kind="task_target",
                    target_key="mechanics.effector.pos",
                ),
                metric=MetricSpec(kind="squared_l2", axis="coordinate"),
                mask=EpochMaskSpec(epochs=["movement"]),
                schedule=PowerLawScheduleSpec(
                    exponent=6,
                    window="post_go",
                    start_value=0.0,
                    end_value=1.0,
                ),
            ),
            TargetStateLossSpec(
                label="effector_pos_movement_ramp",
                selector=_state_selector("state.mechanics.effector.pos", units="m"),
                target=TargetValueSpec(
                    kind="task_target",
                    target_key="mechanics.effector.pos",
                ),
                metric=MetricSpec(kind="squared_l2", axis="coordinate"),
                mask=EpochMaskSpec(epochs=["movement"]),
                schedule=MovementEpochRampScheduleSpec(
                    anchor_epoch="movement",
                    duration_steps=60,
                    shape="power",
                    power=4,
                    start_value=0.0,
                    end_value=1.0,
                ),
            ),
            TargetStateLossSpec(
                label="pre_go_output_penalty",
                selector=_state_selector("state.net.output"),
                target=TargetValueSpec(kind="constant", value=0.0),
                mask=EpochMaskSpec(epochs=["hold", "target_on"]),
                metric=MetricSpec(kind="squared_l2", axis="unit"),
            ),
            FiniteDifferenceLossSpec(
                label="hidden_derivative",
                selector=_state_selector("state.net.hidden"),
                order=1,
                quantity="hidden_derivative",
                mask=EpochMaskSpec(
                    epochs=["hold", "target_on", "movement"],
                    alignment="right_edge",
                ),
                metric=MetricSpec(kind="squared_l2", axis="unit"),
            ),
            FiniteDifferenceLossSpec(
                label="output_velocity_jerk",
                selector=_state_selector("state.mechanics.effector.vel", units="m/s"),
                order=2,
                quantity="velocity_jerk",
                mask=EpochMaskSpec(
                    epochs=["hold", "target_on", "movement"],
                    alignment="right_edge",
                ),
                metric=MetricSpec(kind="squared_l2", axis="coordinate"),
            ),
        ],
    )

    payload = canonical_objective_payload(spec)

    assert payload["schema_version"] == "feedbax.spec.objective.v1"
    assert [term["label"] for term in payload["terms"]] == [
        "effector_pos_full_trial_powerlaw",
        "effector_pos_post_go_powerlaw",
        "effector_pos_movement_ramp",
        "pre_go_output_penalty",
        "hidden_derivative",
        "output_velocity_jerk",
    ]
    assert payload["terms"][2]["schedule"] == {
        "type": "movement_epoch_ramp",
        "anchor_epoch": "movement",
        "duration_steps": 60,
        "shape": "power",
        "power": 4.0,
        "start_value": 0.0,
        "end_value": 1.0,
        "hold_after": True,
    }


def test_flat_position_schedule_uses_constant_schedule_default() -> None:
    spec = ObjectiveSpec(
        timeline=_delayed_reach_timeline(),
        terms=[
            TargetStateLossSpec(
                label="effector_pos_flat",
                selector=_state_selector("state.mechanics.effector.pos", units="m"),
                target=TargetValueSpec(
                    kind="task_target",
                    target_key="mechanics.effector.pos",
                ),
                metric=MetricSpec(kind="squared_l2", axis="coordinate"),
            )
        ],
    )

    assert canonical_objective_payload(spec)["terms"][0]["schedule"] == {
        "type": "constant",
        "value": 1.0,
    }


def test_validation_rejects_finite_difference_without_temporal_selector() -> None:
    with pytest.raises(ValidationError, match="temporal_axis"):
        FiniteDifferenceLossSpec(
            label="bad_hidden_derivative",
            selector=SelectorAddressSpec(
                selector="state.net.hidden",
                kind="state",
                temporal_axis=None,
            ),
            order=1,
        )


def test_validation_rejects_sample_aligned_difference_mask() -> None:
    with pytest.raises(ValidationError, match="right_edge alignment"):
        FiniteDifferenceLossSpec(
            label="bad_output_jerk",
            selector=_state_selector("state.mechanics.effector.vel"),
            order=2,
            mask=EpochMaskSpec(epochs=["movement"]),
        )


def test_validation_rejects_movement_ramp_mask_that_excludes_anchor_epoch() -> None:
    with pytest.raises(ValidationError, match="anchor_epoch"):
        TargetStateLossSpec(
            label="bad_ramp",
            selector=_state_selector("state.mechanics.effector.pos"),
            target=TargetValueSpec(
                kind="task_target",
                target_key="mechanics.effector.pos",
            ),
            mask=EpochMaskSpec(epochs=["hold"]),
            schedule=MovementEpochRampScheduleSpec(
                anchor_epoch="movement",
                duration_steps=20,
            ),
        )


def test_validation_rejects_unknown_timeline_epoch_reference() -> None:
    with pytest.raises(ValidationError, match="unknown mask epochs"):
        ObjectiveSpec(
            timeline=_delayed_reach_timeline(),
            terms=[
                TargetStateLossSpec(
                    label="bad_epoch",
                    selector=_state_selector("state.net.output"),
                    target=TargetValueSpec(kind="constant", value=0.0),
                    mask=EpochMaskSpec(epochs=["pre_go"]),
                )
            ],
        )


def test_validation_rejects_duplicate_term_labels() -> None:
    term = TargetStateLossSpec(
        label="duplicate",
        selector=_state_selector("state.net.output"),
        target=TargetValueSpec(kind="constant", value=0.0),
    )

    with pytest.raises(ValidationError, match="labels must be unique"):
        ObjectiveSpec(
            timeline=_delayed_reach_timeline(),
            terms=[term, term],
        )


def test_provider_manifest_exposes_objective_schema_models() -> None:
    schemas = provider_manifest().schemas

    assert "ObjectiveSpec" in schemas
    assert "FiniteDifferenceLossSpec" in schemas
    assert "MovementEpochRampScheduleSpec" in schemas
