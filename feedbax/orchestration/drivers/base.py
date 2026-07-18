"""Driver protocol for synchronous orchestration backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from feedbax.orchestration.bundle import RunBundle, RunRowSpec
from feedbax.orchestration.state import RunSetState


@dataclass(frozen=True)
class DriverRowProbe:
    """Driver-observed row status."""

    status: str
    pid: int | None = None
    detail: str | None = None
    metadata: Mapping[str, Any] | None = None


class ProvisioningAttemptError(RuntimeError):
    """One failed provisioning attempt with retry and cleanup evidence."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        attempt_record: Mapping[str, Any],
        stop_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.attempt_record = dict(attempt_record)
        self.stop_reason = stop_reason


class OrchestrationDriver(Protocol):
    """Synchronous idempotent driver interface used by the stage engine."""

    def provision(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        """Provision driver resources."""

    def realize_env(self, bundle: RunBundle, state: RunSetState) -> str:
        """Realize the execution environment and return its fingerprint."""

    def stage_inputs(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        """Stage immutable inputs required by rows."""

    def launch_row(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        """Launch one row idempotently."""

    def probe(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> DriverRowProbe:
        """Probe one row without mutating remote resources beyond local sentinels."""

    def stop_row(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        """Stop one row idempotently."""

    def collect(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, str]:
        """Collect row outputs and return logical-name to path mappings."""

    def teardown(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        """Tear down driver resources."""
