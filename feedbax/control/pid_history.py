"""Frozen historical wire endpoints for native component owners."""

from feedbax.component_registry.history import Change, History, LegacyEndpoint

from .pid import PID, PIDDiscrete


def _continuous_v1(params):
    return {"integral_limit": 1000.0, **params}


def _discrete_v1(params):
    return {"output_limit": 1000.0, **params}


HISTORIES = {
    PID: History(
        Change(2, migrate=_continuous_v1),
        legacy=(LegacyEndpoint("PID", "1", 1), LegacyEndpoint("PID", None, 1)),
    ),
    PIDDiscrete: History(
        Change(2, migrate=_discrete_v1),
        legacy=(LegacyEndpoint("PIDDiscrete", "1", 1), LegacyEndpoint("PIDDiscrete", None, 1)),
    ),
}
