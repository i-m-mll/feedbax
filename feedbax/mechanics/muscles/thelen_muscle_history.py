"""Exact pre-revision wire endpoints for portable owners."""

from feedbax.component_registry.history import Change, History, LegacyEndpoint
from .thelen_muscle import RigidTendonHillMuscleThelen


def _materialize_v1_defaults(params):
    return {"n_muscles": 6, "tendon_slack_length": 0.2, **params}


def _matches_RigidTendonHillMuscleThelen_v1(params):
    return set(params) <= {
        "dt",
        "tau_deactivation",
        "tau_activation",
        "n_muscles",
        "max_isometric_force",
        "optimal_muscle_length",
        "initial_activation",
        "tendon_slack_length",
    }


HISTORIES = {
    RigidTendonHillMuscleThelen: History(
        Change(2, migrate=_materialize_v1_defaults),
        legacy=(
            LegacyEndpoint(
                "RigidTendonHillMuscleThelen", "1", 1, _matches_RigidTendonHillMuscleThelen_v1
            ),
        ),
    ),
}
