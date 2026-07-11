"""Resolution helpers for typed batch-schedule clocks."""

from __future__ import annotations

from dataclasses import dataclass

from feedbax.contracts.checkpoints import CheckpointSegmentLineage
from feedbax.contracts.training import BatchScheduleOriginSpec


@dataclass(frozen=True)
class ResolvedScheduleWindow:
    """Absolute batch window resolved once for schedule evaluation."""

    start_batch: int
    end_batch: int | None


def lr_origin_for_continuation(mode: str) -> BatchScheduleOriginSpec:
    """Map the retained LR continuation axis onto typed schedule clocks."""
    if mode == "continue":
        return BatchScheduleOriginSpec(kind="run_start")
    if mode == "restart":
        return BatchScheduleOriginSpec(kind="segment_start")
    raise ValueError(f"unsupported lr_continuation mode {mode!r}")


def resolve_schedule_window(
    origin: BatchScheduleOriginSpec,
    *,
    lineage: CheckpointSegmentLineage,
    duration: int | None,
    allow_inert: bool = False,
) -> ResolvedScheduleWindow:
    """Resolve a typed origin and reject deliberately inert absolute windows."""
    if duration is not None and duration < 0:
        raise ValueError("schedule duration must be non-negative")
    if origin.kind == "segment_start":
        start_batch = lineage.start_batch
    elif origin.kind == "run_start":
        start_batch = 0
    else:
        assert origin.batch is not None
        start_batch = origin.batch
    end_batch = None if duration is None else start_batch + duration
    if (
        origin.kind == "absolute"
        and end_batch is not None
        and end_batch <= lineage.start_batch
        and not allow_inert
    ):
        raise ValueError(
            "absolute schedule window lies entirely before the segment start; "
            f"window=[{start_batch}, {end_batch}], segment_start={lineage.start_batch}; "
            "set allow_inert=true to admit it deliberately"
        )
    return ResolvedScheduleWindow(start_batch=start_batch, end_batch=end_batch)


def schedule_progress(
    current_batch: int,
    *,
    window: ResolvedScheduleWindow,
) -> float:
    """Return clipped linear progress through a finite resolved window."""
    if window.end_batch is None:
        raise ValueError("schedule progress requires a finite window")
    duration = window.end_batch - window.start_batch
    if duration <= 0:
        return 1.0
    return min(1.0, max(0.0, (current_batch - window.start_batch) / duration))
