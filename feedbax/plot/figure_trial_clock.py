"""Recover recorded trial clocks for the samples selected by a figure layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from feedbax.contracts.member_stack import MEMBER
from feedbax.contracts.trial_clock import TrialMeanOptions
from feedbax.dataset import Dataset
from feedbax.execution.trial_clock import AlignedTrial, TrialClockRecord
from feedbax.persistence.evaluation_trial_records import RecordedTrialLayout, read_trial_clock


type Positions = Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class FigureTrialSource:
    plotted: Dataset
    recorded: Mapping[str, Dataset]
    read: Callable[[str], np.ndarray]

    def _coordinate(self, name: str, positions: Positions):
        descriptor = self.plotted.descriptor
        indices = []
        for dim in descriptor.coords[name].dims:
            selected = positions[dim] if dim in positions else range(descriptor.dimensions[dim])
            if len(selected) != 1:
                raise ValueError(f"Clock requires one {dim!r} for a plotted trial.")
            indices.append(selected[0])
        return self.read(name)[tuple(indices)].item()

    def project(self, positions: Positions) -> tuple[TrialClockRecord, TrialMeanOptions]:
        plotted = self.plotted.descriptor
        key = self._coordinate(MEMBER, positions) if MEMBER in plotted.coords else next(iter(self.recorded))
        recorded = self.recorded[key]
        descriptor = recorded.descriptor
        layout = RecordedTrialLayout.model_validate(descriptor.attrs["trial_record"])
        names = descriptor.variables["epoch_start"].dims
        indices = []
        for dim in names:
            if dim == "epoch":
                break
            candidates = ("update", "checkpoint") if dim in {"update", "checkpoint"} else (dim,)
            source = next((name for name in candidates if name in plotted.coords), None)
            if source is None:
                if descriptor.dimensions[dim] != 1:
                    raise ValueError(f"Plotted trial does not identify its recorded {dim!r}.")
                indices.append(0)
                continue
            value = self._coordinate(source, positions)
            matches = np.flatnonzero(recorded.data(dim) == value)
            if len(matches) != 1:
                raise ValueError(f"Plotted {source!r} {value!r} does not identify one recorded trial member.")
            indices.append(int(matches[0]))
        trial = self._coordinate("trial", positions)
        return read_trial_clock(recorded, trial, tuple(indices), layout), layout.mean_options()


def recorded_sample_times(
    trial: AlignedTrial, dimension: str | None, positions: tuple[int, ...] | None,
) -> np.ndarray:
    """The reset-inclusive state samples or the transitions whose left edge they are."""
    match dimension:
        case "state_time":
            time = trial.record.state_time
        case "transition":
            time = trial.record.state_time[:-1]
        case None:
            return np.asarray([trial.alignment.origin])
        case _:
            raise ValueError(f"Series dimension {dimension!r} is not a recorded trial sample axis.")
    return time if positions is None else time[np.asarray(positions)]
