"""Runtime PyTree wrappers for batch-indexed checkpoint histories."""

from __future__ import annotations

from typing import Generic, TypeVar

import equinox as eqx


BatchHistoryValueT = TypeVar("BatchHistoryValueT")


class Granularity(eqx.Module):
    """Static sampling granularity for a batch-indexed history."""

    interval: int = eqx.field(static=True)

    def __init__(self, interval: int = 1):
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("BatchHistory granularity interval must be a positive integer")
        object.__setattr__(self, "interval", interval)

    @classmethod
    def per_batch(cls) -> "Granularity":
        """Return one history entry per training batch."""
        return cls(1)

    @classmethod
    def per_interval(cls, interval: int) -> "Granularity":
        """Return one history entry per ``interval`` training batches."""
        return cls(interval)

    def expected_entries(self, batch_count: int) -> int:
        """Return the segment-local entry count for ``batch_count`` batches."""
        if batch_count < 0:
            raise ValueError("BatchHistory segment batch count must be non-negative")
        return (batch_count + self.interval - 1) // self.interval


class BatchHistory(eqx.Module, Generic[BatchHistoryValueT]):
    """Mark an array as a segment-local batch-indexed checkpoint history."""

    value: BatchHistoryValueT
    batch_axis: int = eqx.field(static=True)
    granularity: Granularity = eqx.field(static=True)

    def __init__(
        self,
        value: BatchHistoryValueT,
        *,
        batch_axis: int = -1,
        granularity: Granularity | None = None,
    ):
        if isinstance(batch_axis, bool) or not isinstance(batch_axis, int):
            raise ValueError("BatchHistory batch_axis must be an integer")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "batch_axis", batch_axis)
        object.__setattr__(self, "granularity", granularity or Granularity.per_batch())


# Preserve the established public and pickle identity while keeping Equinox out
# of validation-only imports of ``feedbax.contracts.checkpoints``.
Granularity.__module__ = "feedbax.contracts.checkpoints"
BatchHistory.__module__ = "feedbax.contracts.checkpoints"
