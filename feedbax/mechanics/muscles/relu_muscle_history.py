"""Exact pre-revision wire endpoints for portable owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .relu_muscle import ReluMuscle


def _matches_ReluMuscle_v1(params):
    return set(params) <= {
        "dt",
        "tau_deactivation",
        "tau_activation",
        "min_activation",
        "max_isometric_force",
        "initial_activation",
    }


HISTORIES = {
    ReluMuscle: History(legacy=(LegacyEndpoint("ReluMuscle", "1", 1, _matches_ReluMuscle_v1),)),
}
