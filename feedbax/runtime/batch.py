"""Batch progress metadata shared by tasks and training."""

import equinox as eqx


class BatchInfo(eqx.Module):
    size: int
    current: int
    total: int
    start: int = 0

    @property
    def progress(self) -> float:
        return self.current / self.total

    @property
    def run_progress(self) -> float:
        return (self.current - self.start) / (self.total - self.start)
