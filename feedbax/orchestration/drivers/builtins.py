"""Built-in orchestration driver registry construction."""

from __future__ import annotations

from feedbax.orchestration.drivers.capabilities import DriverRegistry
from feedbax.orchestration.drivers.local import local_driver_registration
from feedbax.orchestration.drivers.runpod import runpod_driver_registration


def build_builtin_driver_registry() -> DriverRegistry:
    """Return a fresh registry containing Feedbax's built-in drivers."""
    from feedbax.web.services.worker_driver import worker_http_driver_registration

    return DriverRegistry(
        (
            local_driver_registration(),
            runpod_driver_registration(),
            worker_http_driver_registration(),
        )
    )
