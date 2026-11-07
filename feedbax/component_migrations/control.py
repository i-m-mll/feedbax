"""Frozen component migration records for control."""

from feedbax.component_registry.history import Change, History, LegacyEndpoint
from feedbax.control.affine import AffineFeedbackController
from feedbax.control.continuous import Derivative, Integrator, StateSpace, TransferFunction
from feedbax.control.discrete import IntegratorDiscrete, UnitDelay, ZeroOrderHold
from feedbax.control.pid import PID, PIDDiscrete


def _continuous_v1(params):
    return {"integral_limit": 1000.0, **params}


def _discrete_v1(params):
    return {"output_limit": 1000.0, **params}


HISTORIES = {
    Integrator: History(
        legacy=(LegacyEndpoint("Integrator", "1", 1), LegacyEndpoint("Integrator", None, 1)),
    ),
    Derivative: History(
        legacy=(LegacyEndpoint("Derivative", "1", 1), LegacyEndpoint("Derivative", None, 1)),
    ),
    StateSpace: History(
        legacy=(LegacyEndpoint("StateSpace", "1", 1), LegacyEndpoint("StateSpace", None, 1)),
    ),
    TransferFunction: History(
        legacy=(
            LegacyEndpoint("TransferFunction", "1", 1),
            LegacyEndpoint("TransferFunction", None, 1),
        ),
    ),
    IntegratorDiscrete: History(
        legacy=(
            LegacyEndpoint("IntegratorDiscrete", "1", 1),
            LegacyEndpoint("IntegratorDiscrete", None, 1),
        ),
    ),
    UnitDelay: History(
        legacy=(LegacyEndpoint("UnitDelay", "1", 1), LegacyEndpoint("UnitDelay", None, 1)),
    ),
    ZeroOrderHold: History(
        legacy=(LegacyEndpoint("ZeroOrderHold", "1", 1), LegacyEndpoint("ZeroOrderHold", None, 1)),
    ),
    PID: History(
        Change(2, migrate=_continuous_v1),
        legacy=(LegacyEndpoint("PID", "1", 1), LegacyEndpoint("PID", None, 1)),
    ),
    PIDDiscrete: History(
        Change(2, migrate=_discrete_v1),
        legacy=(LegacyEndpoint("PIDDiscrete", "1", 1), LegacyEndpoint("PIDDiscrete", None, 1)),
    ),
    AffineFeedbackController: History(
        legacy=(
            LegacyEndpoint("AffineFeedbackController", "1", 1),
            LegacyEndpoint("AffineFeedbackController", None, 1),
        ),
    ),
}
