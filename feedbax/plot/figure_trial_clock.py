"""Recover recorded trial clocks for the samples selected by a figure layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from feedbax.contracts.member_stack import MEMBER
from feedbax.contracts.trial_clock import TrialMeanOptions
from feedbax.dataset import Dataset
from feedbax.execution.trial_clock import TrialClockRecord


type Positions = Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class FigureRecordedDataset:
    dataset: Dataset
    member_keys: frozenset[str]
    trial_axes: tuple[str, ...]
    read_trial: Callable[[str, tuple[int, ...]], tuple[TrialClockRecord, TrialMeanOptions]]


@dataclass(frozen=True)
class FigureTrialSource:
    plotted: Dataset
    recorded: Sequence[FigureRecordedDataset]
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

    def _selection(
        self, recorded: FigureRecordedDataset, positions: Positions,
    ) -> tuple[int, ...] | None:
        plotted = self.plotted.descriptor
        descriptor = recorded.dataset.descriptor
        indices = []
        for dim in recorded.trial_axes:
            candidates = (dim, "checkpoint" if dim == "update" else "update") if dim in {"update", "checkpoint"} else (dim,)
            source = next((name for name in candidates if name in plotted.coords), None)
            if source is None:
                if descriptor.dimensions[dim] != 1:
                    return None
                indices.append(0)
                continue
            value = self._coordinate(source, positions)
            matches = np.flatnonzero(recorded.dataset.data(dim) == value)
            if len(matches) != 1:
                return None
            indices.append(int(matches[0]))
        return tuple(indices)

    def project(self, positions: Positions) -> tuple[TrialClockRecord, TrialMeanOptions]:
        trial = self._coordinate("trial", positions)
        if not isinstance(trial, str):
            raise ValueError("A recorded trial ID must be a string.")
        member = self._coordinate(MEMBER, positions) if MEMBER in self.plotted.descriptor.coords else None
        matches = [
            (source, indices)
            for source in self.recorded
            if member is None or member in source.member_keys
            if (indices := self._selection(source, positions)) is not None
        ]
        if len(matches) != 1:
            raise ValueError(f"Plotted trial {trial!r} requires one matching retained trial clock.")
        (recorded, indices), = matches
        return recorded.read_trial(trial, indices)
