"""Task timeline helpers for named epoch masks.

The implementation lives in :mod:`feedbax.runtime.timeline_masks` so the graph
runtime can lower segment masks without importing concrete task definitions.
This module is the task-facing name for the same contract.
"""

from __future__ import annotations

from feedbax.runtime.timeline_masks import (
    TaskTimelineMask,
    TaskTimelineMaskError,
    align_time_mask,
    build_task_timeline_mask,
)

__all__ = [
    "TaskTimelineMask",
    "TaskTimelineMaskError",
    "align_time_mask",
    "build_task_timeline_mask",
]
