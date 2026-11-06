"""Authored timing and presets for DelayedReaches task specs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any


DEFAULT_DELAYED_EPOCH_LEN_RANGES = ((5, 15), (10, 20))
DEFAULT_DELAYED_MOVE_EPOCHS = (2,)
DELAYED_CENTER_OUT_PRESET = "delayed_center_out"
DEFAULT_DELAYED_N_STEPS = 140

_DELAYED_CENTER_OUT_DEFAULTS: dict[str, Any] = {
    "preset": DELAYED_CENTER_OUT_PRESET,
    "train_endpoint_mode": "center_out",
    "epoch_names": ["prep", "movement"],
    "epoch_len_ranges": [[5, 15]],
    "target_on_epochs": [0, 1],
    "hold_epochs": [0],
    "move_epochs": [1],
    "target_visible_from_start": True,
    "go_cue_event_name": "go_cue",
    "catch_metadata_policy": "flag",
}


@dataclass(frozen=True)
class DelayedTimingIssue:
    code: str
    path: str
    message: str


def delayed_timing_issues(
    n_steps: int | None, epoch_len_ranges: object, move_epochs: object
) -> list[DelayedTimingIssue]:
    """Check that every draw fits the rollout and has a movement stage."""
    if not isinstance(epoch_len_ranges, (list, tuple)):
        return [
            DelayedTimingIssue(
                "invalid_epoch_len_ranges",
                "epoch_len_ranges",
                "DelayedReaches epoch_len_ranges must be a list of [min, max] pairs",
            )
        ]

    issues = []
    lower_bounds = []
    upper_bounds = []
    for index, pair in enumerate(epoch_len_ranges):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            message = "Each DelayedReaches epoch_len_ranges entry must be [min, max]"
        elif not all(isinstance(value, Integral) and not isinstance(value, bool) for value in pair):
            message = "DelayedReaches epoch lengths must be integers"
        elif not 0 <= pair[0] <= pair[1]:
            message = "DelayedReaches epoch length ranges must satisfy 0 <= min <= max"
        else:
            lo, hi = map(int, pair)
            lower_bounds.append(lo)
            upper_bounds.append(hi - 1 if lo < hi else lo)
            continue
        issues.append(
            DelayedTimingIssue("invalid_epoch_len_range", f"epoch_len_ranges/{index}", message)
        )

    remainder_epoch = len(epoch_len_ranges)
    movement = set()
    if not isinstance(move_epochs, (list, tuple)):
        issues.append(
            DelayedTimingIssue(
                "invalid_epoch_index_list",
                "move_epochs",
                "DelayedReaches move_epochs must be a list of epoch indexes",
            )
        )
    else:
        for index, epoch in enumerate(move_epochs):
            if not isinstance(epoch, Integral) or isinstance(epoch, bool):
                message = "DelayedReaches move_epochs entries must be integer epoch indexes"
            elif not 0 <= epoch <= remainder_epoch:
                message = (
                    f"DelayedReaches move_epochs entry {epoch} is outside "
                    f"the configured epoch range [0, {remainder_epoch}]"
                )
            else:
                movement.add(epoch)
                continue
            issues.append(
                DelayedTimingIssue("invalid_epoch_index", f"move_epochs/{index}", message)
            )

    if issues or n_steps is None:
        return issues

    n_control_stages = n_steps - 1
    maximum_sampled = sum(upper_bounds)
    if maximum_sampled > n_control_stages:
        message = (
            f"DelayedReaches epoch lengths can require {maximum_sampled} control stages "
            f"but only {n_control_stages} are available"
        )
    else:
        if remainder_epoch in movement:
            # Selected sampled epochs and the selected remainder share the stages
            # left over after non-movement epochs take their largest draws.
            minimum_movement = n_control_stages - sum(
                hi for epoch, hi in enumerate(upper_bounds) if epoch not in movement
            )
        else:
            minimum_movement = sum(lower_bounds[epoch] for epoch in movement)
        if minimum_movement >= 1:
            return issues
        message = (
            f"DelayedReaches timing must leave at least one movement control stage "
            f"for every draw with {n_control_stages} control stages"
        )
    return [DelayedTimingIssue("invalid_epoch_budget", "epoch_len_ranges", message)]


def delayed_center_out_reaches_params(
    *,
    n_control_stages: int | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Return compact params for delayed center-out reaches.

    ``n_control_stages`` is the rollout length. A :class:`feedbax.tasks.DelayedReaches`
    instance uses ``n_steps = n_control_stages + 1`` to include the initial state.

    The default prep range ``[5, 15)`` requires at least 15 control stages.
    Set ``epoch_len_ranges`` explicitly for a shorter rollout, for example
    ``[[2, 7]]`` for eight control stages.
    """

    params = dict(_DELAYED_CENTER_OUT_DEFAULTS)
    if n_control_stages is not None:
        params["n_control_stages"] = int(n_control_stages)
    params.update(overrides)
    return params


def apply_delayed_reaches_preset(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return DelayedReaches params with any named preset defaults applied."""

    preset = params.get("preset")
    if preset in (None, "", "default"):
        return dict(params)
    if preset != DELAYED_CENTER_OUT_PRESET:
        raise ValueError(
            f"Unknown DelayedReaches preset {preset!r}; expected {DELAYED_CENTER_OUT_PRESET!r}"
        )

    merged = delayed_center_out_reaches_params()
    merged.update(params)
    return merged


def delayed_reaches_n_steps_from_params(
    params: Mapping[str, Any], default: int = DEFAULT_DELAYED_N_STEPS
) -> int:
    """Materialize DelayedReaches constructor ``n_steps`` from compact params."""

    if params.get("n_control_stages") is not None:
        return int(params["n_control_stages"]) + 1
    return int(params.get("n_steps", default))
