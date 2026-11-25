"""Project trial execution records into Studio's transport contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import TYPE_CHECKING

import numpy as np

from feedbax.contracts.trial_record import (
    TaskEpochShare, TaskSignalAggregate, TaskTimelineCue, TaskVariableRange, TrialAggregate, TrialRecord,
    TrialTerm, TrialTermAggregate,
)
if TYPE_CHECKING:
    from feedbax.contracts.task_document import TaskDocument
    from feedbax.tasks.document import Trial


_PERCENTILES = (5, 25, 50, 75, 95)


def trial_record(
    trial: Trial, *, id: str, epoch_ids: Sequence[str], epoch_labels: Mapping[str, str],
    terms: dict[str, TrialTerm],
    document: TaskDocument | None = None,
) -> TrialRecord:
    timeline = [
        TaskTimelineCue(epoch=epoch, label=epoch_labels.get(epoch, epoch),
                        start=int(start), end=int(end), outcome=int(outcome))
        for epoch, start, end, outcome in zip(epoch_ids, *trial.epochs, strict=True)
    ]
    return TrialRecord(
        id=id,
        inits={key: np.asarray(value).tolist() for key, value in trial.inits.items()},
        inputs={key: np.asarray(value).tolist() for key, value in trial.inputs.items()},
        present={key: np.asarray(value).tolist() for key, value in trial.present.items()},
        variables={key: np.asarray(value).tolist() for key, value in trial.variables.items()},
        variable_defined={key: np.asarray(value).tolist() for key, value in (trial.variable_defined or {}).items()},
        timeline=timeline,
        terms=terms,
        intervention_windows={address: mask.tolist() for address, mask in intervention_windows(trial.inputs, document).items()},
    )


def intervention_windows(inputs: Mapping[str, object], document: TaskDocument | None) -> dict[str, np.ndarray]:
    windows = {}
    if document is not None:
        for signal in document.signals:
            if signal.intervention is None or signal.id not in inputs:
                continue
            address = signal.intervention
            active = np.asarray(inputs[signal.id], dtype=bool)
            windows[address] = active | windows.get(address, False)
    return windows


def trial_weight_values(values: np.ndarray, defined: np.ndarray) -> list[float | None]:
    """Project a weight trace onto its valid trial steps for transport."""
    valid = defined & np.isfinite(values)
    return [value if keep else None for value, keep in zip(values.tolist(), valid.tolist(), strict=True)]


def _aggregate_signal(values: np.ndarray, present: np.ndarray) -> TaskSignalAggregate:
    percentiles = None
    if np.issubdtype(values.dtype, np.number):
        flat = values.reshape(*values.shape[:2], prod(values.shape[2:])).astype(float)
        masked = np.where(present[..., None], flat, np.nan)
        any_present = present.any(axis=0)
        with np.errstate(all="ignore"):
            quantiles = np.nanpercentile(masked[:, any_present], _PERCENTILES, axis=0) if any_present.any() else None
        at = iter(np.moveaxis(quantiles, 0, -1).tolist() if quantiles is not None else ())
        percentiles = [next(at) if here else None for here in any_present]
    return TaskSignalAggregate(presence=present.mean(axis=0).tolist(), percentiles=percentiles)


def _aggregate_term(terms: Sequence[TrialTerm]) -> TrialTermAggregate:
    steps = np.asarray([term.steps for term in terms], dtype=bool)
    reduced = [term.value for term in terms if term.value is not None]
    samples = [term.values for term in terms]
    percentiles = None
    if all(value is not None for value in samples):
        bands = _aggregate_signal(np.asarray(samples), steps).percentiles
        if bands is not None:
            percentiles = [band[0] if band is not None else None for band in bands]
    return TrialTermAggregate(
        problems=terms[0].problems,
        steps=steps.mean(axis=0).tolist(),
        covered=float(np.mean([term.covered for term in terms])),
        value=float(np.mean(reduced)) if reduced else None,
        percentiles=percentiles,
    )


def aggregate_trials(trials: Sequence[TrialRecord], n_steps: int) -> TrialAggregate:
    """Shares and spread over a cohort of trials from the same pairing."""
    if not trials:
        return TrialAggregate(kind="aggregate", epochs=[], signals={}, variables={})

    steps = np.arange(n_steps)
    epochs = []
    for cues in zip(*(trial.timeline for trial in trials), strict=True):
        first = cues[0]
        starts = np.asarray([cue.start for cue in cues])[:, None]
        ends = np.asarray([cue.end for cue in cues])[:, None]
        inside = (steps >= starts) & (steps < ends)
        epochs.append(TaskEpochShare(epoch=first.epoch, label=first.label, share=inside.mean(axis=0).tolist()))
    return TrialAggregate(
        kind="aggregate",
        epochs=epochs,
        signals={
            name: _aggregate_signal(np.asarray([trial.inputs[name] for trial in trials]),
                                    np.asarray([trial.present[name] for trial in trials], dtype=bool))
            for name in trials[0].inputs
        },
        variables={
            name: TaskVariableRange(least=values.min(axis=0).tolist(), greatest=values.max(axis=0).tolist())
            for name in trials[0].variables
            for values in (np.asarray([trial.variables[name] for trial in trials]),)
        },
        terms={
            name: _aggregate_term([trial.terms[name] for trial in trials])
            for name in trials[0].terms
        },
    )
