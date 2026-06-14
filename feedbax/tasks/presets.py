"""Reusable task-spec presets for public Feedbax task APIs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DELAYED_CENTER_OUT_PRESET = "delayed_center_out"

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


def delayed_center_out_reaches_params(
    *,
    n_control_stages: int | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Return compact params for delayed center-out reaches.

    ``n_control_stages`` is a serialized task-spec alias for the rollout/control
    length.  Materializing a :class:`feedbax.tasks.DelayedReaches` instance from
    these params maps it to ``n_steps = n_control_stages + 1`` because the task
    trial includes an initial state plus control transitions.
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
            f"Unknown DelayedReaches preset {preset!r}; "
            f"expected {DELAYED_CENTER_OUT_PRESET!r}"
        )

    merged = delayed_center_out_reaches_params()
    merged.update(params)
    return merged


def delayed_reaches_n_steps_from_params(params: Mapping[str, Any], default: int = 140) -> int:
    """Materialize DelayedReaches constructor ``n_steps`` from compact params."""

    if "n_steps" in params:
        return int(params["n_steps"])
    if "n_control_stages" in params:
        return int(params["n_control_stages"]) + 1
    return default
