"""Task timeline helpers for named epoch masks.

This module is the provider-side counterpart to Studio's task timeline editor.
It accepts the durable Studio task timeline shape when present, falls back to
the compact ``DelayedReaches`` params for current fixtures, and returns masks
that can be used by GraphSpec retained-observable/loss lowering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import jax.numpy as jnp


@dataclass(frozen=True)
class TaskTimelineMask:
    """Executable mask for one named task timeline segment."""

    segment_name: str
    mask: Any
    epoch_ids: tuple[str, ...]
    n_steps: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TaskTimelineMaskError(ValueError):
    """Pathful timeline-mask validation error."""

    def __init__(self, message: str, *, path: str, segment_name: Optional[str] = None) -> None:
        super().__init__(message)
        self.path = path
        self.segment_name = segment_name


def build_task_timeline_mask(
    task_spec: Mapping[str, Any] | Any | None,
    *,
    segment_name: str,
    n_steps: int,
    path: str = "/task_spec/timeline",
) -> TaskTimelineMask:
    """Build a graph-worker-aligned mask for a named task timeline segment.

    Only fixed or already-resolved epochs are executable here. Variable-length
    and adaptive epochs need per-trial resolved bounds from task generation,
    so they fail pathfully instead of silently lowering an arbitrary preview.
    """

    if not segment_name:
        raise TaskTimelineMaskError(
            "Segment time aggregation requires segment_name",
            path=f"{path}/segment_name",
        )
    task = _as_mapping(task_spec)
    timeline = _timeline_from_task(task)
    if timeline is None:
        raise TaskTimelineMaskError(
            "Segment time aggregation requires task timeline mask lowering from "
            "task_spec.timeline or a supported task timeline",
            path=path,
            segment_name=segment_name,
        )
    if n_steps <= 0:
        raise TaskTimelineMaskError(
            "Timeline mask lowering requires a positive n_steps",
            path=f"{path}/n_steps",
            segment_name=segment_name,
        )

    epochs = list(timeline.get("epochs") or [])
    if not epochs:
        raise TaskTimelineMaskError(
            "Task timeline has no epochs to use as segment masks",
            path=f"{path}/epochs",
            segment_name=segment_name,
        )

    bounds = _epoch_bounds(epochs, n_steps=n_steps, path=f"{path}/epochs")
    selected_indexes = _segment_epoch_indexes(timeline, segment_name, path=path)
    if not selected_indexes:
        available = ", ".join(_epoch_names(timeline)) or "none"
        raise TaskTimelineMaskError(
            f"Unknown task timeline segment {segment_name!r}; available segments: {available}",
            path=f"{path}/segment_name",
            segment_name=segment_name,
        )

    time = jnp.arange(n_steps)
    mask = jnp.zeros((n_steps,), dtype=bool)
    for index in selected_indexes:
        if index < 0 or index + 1 >= len(bounds):
            raise TaskTimelineMaskError(
                f"Timeline segment {segment_name!r} references missing epoch index {index}",
                path=f"{path}/segments/{segment_name}",
                segment_name=segment_name,
            )
        start = int(bounds[index])
        end = int(bounds[index + 1])
        mask = jnp.logical_or(mask, jnp.logical_and(time >= start, time < end))

    if not bool(jnp.any(mask)):
        raise TaskTimelineMaskError(
            f"Timeline segment {segment_name!r} selected no timesteps",
            path=f"{path}/segment_name",
            segment_name=segment_name,
        )

    return TaskTimelineMask(
        segment_name=segment_name,
        mask=mask,
        epoch_ids=tuple(str(epochs[index].get("id", f"epoch:{index}")) for index in selected_indexes),
        n_steps=n_steps,
        metadata={
            "timeline_schema_version": timeline.get("schema_version"),
            "bounds": bounds,
            "source": timeline.get("metadata", {}).get("source", "task_spec.timeline"),
        },
    )


def align_time_mask(mask: Any, values: Any, *, path: str) -> Any:
    """Align a timeline mask to a retained trace/loss density time axis."""

    mask_arr = jnp.asarray(mask, dtype=bool)
    values_arr = jnp.asarray(values)
    if values_arr.ndim == 0:
        raise TaskTimelineMaskError(
            "Segment time aggregation requires a time dimension",
            path=path,
        )
    time_len = int(values_arr.shape[0])
    if time_len <= 0:
        raise TaskTimelineMaskError(
            "Segment time aggregation received an empty time axis",
            path=path,
        )
    if mask_arr.shape[0] < time_len:
        raise TaskTimelineMaskError(
            "Timeline mask is shorter than the retained trace time axis",
            path=path,
        )
    if mask_arr.shape[0] > time_len:
        # Cross-timestep terms such as differences lose the first `order`
        # timesteps. Align to the right edge, matching the existing epoch-mask
        # loss convention.
        mask_arr = mask_arr[-time_len:]
    if not bool(jnp.any(mask_arr)):
        raise TaskTimelineMaskError(
            "Segment time aggregation selected no timesteps after alignment",
            path=path,
        )
    return mask_arr


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json", exclude_none=True)
        return dumped if isinstance(dumped, Mapping) else {}
    return {}


def _timeline_from_task(task: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    timeline = task.get("timeline")
    if isinstance(timeline, Mapping):
        return timeline
    task_type = str(task.get("type", ""))
    if task_type not in {"DelayedReaches", "feedbax.task.DelayedReaches"}:
        return None
    params = task.get("params") if isinstance(task.get("params"), Mapping) else {}
    return _delayed_reaches_timeline(params)


def _delayed_reaches_timeline(params: Mapping[str, Any]) -> Mapping[str, Any]:
    ranges = _as_ranges(params.get("epoch_len_ranges", [[5, 15], [10, 20]]))
    epoch_names = tuple(
        str(name)
        for name in params.get("epoch_names", ("hold", "target_on", "movement"))
    )
    epoch_count = max(len(epoch_names), len(ranges) + 1)
    epochs: list[dict[str, Any]] = []
    for index in range(epoch_count):
        label = epoch_names[index] if index < len(epoch_names) else f"epoch {index + 1}"
        if index < len(ranges):
            lower, upper = ranges[index]
            value: dict[str, Any] | None = {"min": lower, "max": upper}
            metadata = {"range_upper_bound": "exclusive"}
        else:
            value = None
            metadata = {"inferred_from_remaining_steps": True}
        epochs.append(
            {
                "id": f"epoch:{index}",
                "label": label,
                "index": index,
                "length": {
                    "schema_version": "feedbax.spec.studio.value.v1",
                    "mode": "constant",
                    "value": value,
                    "units": "steps",
                    "metadata": metadata,
                },
                "metadata": {},
            }
        )
    return {
        "schema_version": "feedbax.spec.studio.task_timeline.v1",
        "epochs": epochs,
        "segments": [],
        "metadata": {
            "source": "compact_delayed_reaches_params",
            "n_steps": params.get("n_steps"),
        },
    }


def _as_ranges(value: Any) -> list[tuple[int, int]]:
    if not isinstance(value, list):
        return []
    ranges: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        try:
            ranges.append((int(item[0]), int(item[1])))
        except (TypeError, ValueError):
            continue
    return ranges


def _epoch_bounds(epochs: list[Mapping[str, Any]], *, n_steps: int, path: str) -> list[int]:
    bounds = [0]
    cursor = 0
    for index, epoch in enumerate(epochs):
        epoch_path = f"{path}/{index}/length"
        length_spec = _as_mapping(epoch.get("length"))
        duration = _fixed_epoch_duration(length_spec, n_steps - cursor, path=epoch_path)
        if duration is None:
            duration = n_steps - cursor
        if duration < 0:
            raise TaskTimelineMaskError(
                "Timeline epoch lengths exceed n_steps",
                path=epoch_path,
            )
        cursor += duration
        if cursor > n_steps:
            raise TaskTimelineMaskError(
                "Timeline epoch lengths exceed n_steps",
                path=epoch_path,
            )
        bounds.append(cursor)
    if bounds[-1] < n_steps:
        bounds[-1] = n_steps
    return bounds


def _fixed_epoch_duration(
    length_spec: Mapping[str, Any],
    remaining_steps: int,
    *,
    path: str,
) -> Optional[int]:
    mode = length_spec.get("mode", "constant")
    metadata = _as_mapping(length_spec.get("metadata"))
    if metadata.get("inferred_from_remaining_steps"):
        return None
    if mode != "constant":
        raise TaskTimelineMaskError(
            f"Timeline epoch length mode {mode!r} is not executable as a fixed mask",
            path=path,
        )
    value = length_spec.get("value")
    if value is None:
        return remaining_steps
    if isinstance(value, Mapping):
        if "steps" in value:
            return int(value["steps"])
        if "min" in value and "max" in value:
            lower = int(value["min"])
            upper = int(value["max"])
            if upper == lower:
                return lower
            if upper == lower + 1:
                return lower
            raise TaskTimelineMaskError(
                "Variable-length timeline epochs need resolved per-trial bounds before "
                "segment mask lowering",
                path=path,
            )
    if isinstance(value, int | float):
        return int(value)
    raise TaskTimelineMaskError(
        "Timeline epoch length must be a fixed step count or fixed range",
        path=path,
    )


def _segment_epoch_indexes(
    timeline: Mapping[str, Any],
    segment_name: str,
    *,
    path: str,
) -> tuple[int, ...]:
    epochs = list(timeline.get("epochs") or [])
    for index, epoch in enumerate(epochs):
        epoch_id = str(epoch.get("id", f"epoch:{index}"))
        label = str(epoch.get("label", epoch_id))
        candidates = {
            epoch_id,
            label,
            label.replace(" ", "_"),
            label.replace("-", "_"),
        }
        if segment_name in candidates:
            return (index,)

    for index, segment in enumerate(list(timeline.get("segments") or [])):
        if not isinstance(segment, Mapping):
            continue
        if segment.get("id") != segment_name and segment.get("label") != segment_name:
            continue
        epoch_ids = segment.get("epoch_ids", [])
        indexes = []
        for epoch_id in epoch_ids:
            match = next(
                (
                    epoch_index
                    for epoch_index, epoch in enumerate(epochs)
                    if str(epoch.get("id", f"epoch:{epoch_index}")) == epoch_id
                ),
                None,
            )
            if match is None:
                raise TaskTimelineMaskError(
                    f"Timeline segment {segment_name!r} references unknown epoch {epoch_id!r}",
                    path=f"{path}/segments/{index}/epoch_ids",
                    segment_name=segment_name,
                )
            indexes.append(match)
        return tuple(indexes)
    return ()


def _epoch_names(timeline: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for index, epoch in enumerate(list(timeline.get("epochs") or [])):
        names.append(str(epoch.get("label") or epoch.get("id") or f"epoch:{index}"))
    for segment in list(timeline.get("segments") or []):
        if isinstance(segment, Mapping) and segment.get("id"):
            names.append(str(segment["id"]))
    return tuple(dict.fromkeys(names))
