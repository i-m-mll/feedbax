"""Drive managed Train blocks and their committed checkpoint boundaries."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from feedbax.contracts.method_checkpoint import MethodCheckpointClock

if TYPE_CHECKING:
    from feedbax.contracts.request_specs import TrainContinuation


TRAIN_PREFIX_UPDATES = 5
TRAIN_DRIVER_REVISION = 1


def carries_method_state(continuation: TrainContinuation | None, *, resume: bool) -> bool:
    return resume or (continuation is not None and continuation.state == "preserve")


def train_start_clock(
    seed: int, *, source: MethodCheckpointClock | None, resume: bool, continuation: TrainContinuation | None,
) -> MethodCheckpointClock:
    if resume:
        if source is None:
            raise ValueError("Resume requires the exact prior Train clock")
        return source
    if continuation is None:
        return MethodCheckpointClock(completed_updates=0, segment_start=0, schedule_origin=0, seed=seed)
    if source is None or source.completed_updates != continuation.source_completed_updates:
        raise ValueError("/checkpoint_initialization/source_completed_updates: bound predecessor differs.")
    completed = source.completed_updates
    return MethodCheckpointClock(
        completed_updates=completed, segment_start=completed,
        schedule_origin=source.schedule_origin if continuation.schedule == "continue" else completed,
        seed=source.seed if continuation.rng == "carry" else seed,
    )


@dataclass(frozen=True)
class TrainBlock:
    completed: int
    count: int
    checkpoint: bool


@dataclass(frozen=True)
class TrainSchedule:
    start: int
    target: int
    segment_start: int
    checkpoint_interval: int
    checkpoints: tuple[int, ...]

    @property
    def prefix_end(self) -> int:
        return min(self.start + TRAIN_PREFIX_UPDATES, self.target)

    def blocks(self, *, end: int | None = None) -> Iterator[TrainBlock]:
        """Yield the blocks of a successful segment from its actual start."""
        end = self.target if end is None else end
        if not self.segment_start <= self.start <= end <= self.target or self.checkpoint_interval < 1:
            raise ValueError("Train block coordinates do not follow the segment schedule")
        completed = self.start
        requested = iter(sorted({step for step in self.checkpoints if completed < step < self.target}))
        next_requested = next(requested, self.target)
        while completed < end:
            segment_updates = completed - self.segment_start
            cadence = completed + self.checkpoint_interval - segment_updates % self.checkpoint_interval
            checkpoint = min(cadence, next_requested, self.target)
            boundary = min(checkpoint, end, self.prefix_end) if completed < self.prefix_end else min(checkpoint, end)
            count = min(16, boundary - completed)
            yield TrainBlock(completed, count, completed + count == checkpoint)
            completed += count
            if completed == next_requested:
                next_requested = next(requested, self.target)


class CommittedBlock[State](Protocol):
    @property
    def state(self) -> State: ...

    @property
    def completed(self) -> int: ...

    @property
    def rejected(self) -> bool: ...


@dataclass(frozen=True)
class TrainProgress[State]:
    state: State
    completed: int
    stopped: bool
    rejected: bool


def drive_train[State, Block: CommittedBlock](
    schedule: TrainSchedule, state: State, *,
    step: Callable[[State, TrainBlock], Block],
    record: Callable[[TrainBlock, Block], None],
    checkpoint: Callable[[bool], None],
    should_stop: Callable[[], bool],
    end: int | None = None,
) -> TrainProgress[State]:
    """Commit the accepted prefix of each block before exposing its checkpoint."""
    completed = schedule.start
    last_checkpoint = None
    for block in schedule.blocks(end=end):
        if completed > schedule.start and should_stop():
            if completed != last_checkpoint:
                checkpoint(True)
            return TrainProgress(state, completed, stopped=True, rejected=False)

        result = step(state, block)
        state, completed = result.state, result.completed
        if not block.completed <= completed <= block.completed + block.count:
            raise ValueError("Train block committed updates outside its scheduled span")
        record(block, result)
        stopped = completed < schedule.target and should_stop()
        if not (result.rejected or stopped) and completed != block.completed + block.count:
            raise ValueError("Train block accepted fewer updates without recording an interruption")
        if completed > block.completed and (block.checkpoint or stopped or result.rejected):
            checkpoint(completed == schedule.target or stopped or result.rejected)
            last_checkpoint = completed
        if stopped or result.rejected:
            return TrainProgress(state, completed, stopped, result.rejected)

    if completed != (schedule.target if end is None else end):
        raise ValueError("Train finished at a different coordinate from its scheduled end")
    return TrainProgress(state, completed, stopped=False, rejected=False)
