"""Group recorded trials and summarize samples on their shared time domain."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import JsonValue

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.rollout import CannotRun
from feedbax.contracts.trial_clock import (
    TrialBoundarySummary,
    TrialClockAlignment,
    TrialClockSpan,
    TrialMean,
    TrialMeanCondition,
    TrialMeanEpoch,
    TrialMeanOptions,
    TrialMeanSelection,
    TrialSeriesSummary,
)
from feedbax.contracts.trial_record import TaskTimelineCue, TrialRecord


@dataclass(frozen=True)
class TrialClockRecord:
    id: str
    values: Mapping[str, JsonValue]
    timeline: Sequence[TaskTimelineCue]
    state_time: np.ndarray


@dataclass(frozen=True)
class AlignedTrial:
    record: TrialClockRecord
    alignment: TrialClockAlignment

    def times(self, recorded: np.ndarray) -> np.ndarray:
        return recorded - self.alignment.origin


@dataclass(frozen=True)
class AlignedCondition:
    values: dict[str, JsonValue]
    trials: list[AlignedTrial]


def align_trials(
    records: Iterable[TrialClockRecord], options: TrialMeanOptions, selection: TrialMeanSelection,
) -> list[AlignedCondition]:
    available = {choice.id for choice in options.by}
    by = [choice.id for choice in options.by] if selection.by is None else selection.by
    if len(set(by)) != len(by) or not set(by) <= available:
        raise CannotRun("/mean/by", "choose distinct discrete variables recorded for these trials")
    if selection.align is not None and selection.align not in {choice.id for choice in options.align}:
        raise CannotRun("/mean/align", "choose an epoch recorded for these trials")

    groups: dict[bytes, AlignedCondition] = {}
    for record in records:
        time = record.state_time
        reached = [cue for cue in record.timeline if cue.start >= 0]
        end = max((cue.end for cue in reached), default=len(time) - 1)
        start = 0
        if selection.align is not None:
            anchor = next((cue for cue in reached if cue.epoch == selection.align), None)
            if anchor is None:
                raise CannotRun("/mean/align", f"trial {record.id!r} did not reach {selection.align!r}")
            start = anchor.start
        origin = float(time[start])
        alignment = TrialClockAlignment(
            origin=origin, span=TrialClockSpan(start=float(time[0]) - origin, end=float(time[end]) - origin),
        )
        values = {name: record.values[name] for name in by}
        condition = groups.setdefault(canonical_json_v2_bytes(values), AlignedCondition(values, []))
        condition.trials.append(AlignedTrial(record, alignment))
    return list(groups.values())


def common_span(spans: Iterable[TrialClockSpan]) -> TrialClockSpan | None:
    spans = tuple(spans)
    start = max(span.start for span in spans)
    end = min(span.end for span in spans)
    return TrialClockSpan(start=start, end=end) if start <= end else None


@dataclass(frozen=True)
class AlignedSeries:
    time: np.ndarray
    values: np.ndarray
    span: TrialClockSpan

    @property
    def extent(self) -> TrialClockSpan:
        return TrialClockSpan(start=max(float(self.time[0]), self.span.start), end=min(float(self.time[-1]), self.span.end))


def aligned_sample_times(axes: Sequence[np.ndarray], *, span: TrialClockSpan) -> np.ndarray:
    candidates = np.unique(np.concatenate(([span.start, span.end], *axes)))
    inside = candidates[(candidates >= span.start) & (candidates <= span.end)]
    # Subtracting integer epoch starts from recorded float seconds can perturb otherwise
    # coincident samples by a few ulps. Preserve the boundaries and a single interior sample.
    scale = max(float(np.max(np.abs(axis))) for axis in axes)
    tolerance = np.spacing(scale) * 8
    interior = inside[(inside > span.start + tolerance) & (inside < span.end - tolerance)]
    separated = interior[np.r_[True, np.diff(interior) > tolerance]] if interior.size else interior
    return np.r_[span.start, separated, span.end] if span.start != span.end else inside


def summarize_series(series: Sequence[AlignedSeries], time: np.ndarray) -> TrialSeriesSummary:
    """Interpolate over common coverage, then take the pointwise mean and observed range."""
    samples = np.stack([
        np.stack([np.interp(time, item.time, component) for component in item.values.T], axis=-1)
        for item in series
    ])
    return TrialSeriesSummary(
        mean=samples.mean(axis=0).tolist(), lower=samples.min(axis=0).tolist(), upper=samples.max(axis=0).tolist(),
    )


def _bounds(values: list[float]) -> TrialBoundarySummary:
    return TrialBoundarySummary(mean=sum(values) / len(values), lower=min(values), upper=max(values))


def _mean_epochs(trials: Sequence[AlignedTrial]) -> list[TrialMeanEpoch]:
    bounds: dict[tuple[str, int, str], tuple[list[float], list[float]]] = {}
    for trial in trials:
        occurrences: Counter[str] = Counter()
        for cue in trial.record.timeline:
            occurrence = occurrences[cue.epoch]
            occurrences[cue.epoch] += 1
            if cue.start < 0:
                continue
            starts, ends = bounds.setdefault((cue.epoch, occurrence, cue.label), ([], []))
            start, end = trial.times(trial.record.state_time[[cue.start, cue.end]])
            starts.append(float(start))
            ends.append(float(end))
    return [
        TrialMeanEpoch(epoch=epoch, label=label, occurrence=occurrence, start=_bounds(starts), end=_bounds(ends))
        for (epoch, occurrence, label), (starts, ends) in bounds.items()
    ]


def mean_trial_paths(
    records: Sequence[TrialRecord], state_time: np.ndarray, options: TrialMeanOptions, selection: TrialMeanSelection,
) -> TrialMean:
    by_id = {record.id: record for record in records}
    clocks = [TrialClockRecord(record.id, record.variables, record.timeline, state_time) for record in records]
    conditions = []
    for group in align_trials(clocks, options, selection):
        span = common_span(trial.alignment.span for trial in group.trials)
        aligned_times = [trial.times(state_time) for trial in group.trials]
        time = np.array([], dtype=float)
        summaries = {}
        if span is not None:
            tracks = [by_id[trial.record.id].states for trial in group.trials]
            time = aligned_sample_times(aligned_times, span=span)
            for name in tracks[0]:
                series = [
                    AlignedSeries(axis, np.asarray(track[name].samples), trial.alignment.span)
                    for axis, trial, track in zip(aligned_times, group.trials, tracks, strict=True)
                ]
                summaries[name] = summarize_series(series, time)
        conditions.append(TrialMeanCondition(
            values=group.values, trials={trial.record.id: trial.alignment for trial in group.trials},
            span=span, time=time.tolist(), states=summaries, epochs=_mean_epochs(group.trials),
        ))
    return TrialMean(selection=selection, conditions=conditions)
