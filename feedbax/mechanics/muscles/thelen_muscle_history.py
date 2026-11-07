"""Exact pre-revision wire endpoints for portable owners."""

from feedbax.component_registry.history import Change, History, LegacyEndpoint
from .thelen_muscle import RigidTendonHillMuscleThelen


def _materialize_v1_defaults(params):
    return {"n_muscles": 6, "tendon_slack_length": 0.2, **params}


HISTORIES = {
    RigidTendonHillMuscleThelen: History(
        Change(2, migrate=_materialize_v1_defaults),
        legacy=(LegacyEndpoint("RigidTendonHillMuscleThelen", "1", 1),),
    ),
}
