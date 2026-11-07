"""Exact historical identities for the physical element owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .mechanics import RigidTendonHillMuscle

HISTORIES = {
    RigidTendonHillMuscle: History(
        legacy=(
            LegacyEndpoint(
                "RigidTendonHillMuscle", "1", 1, matches_unversioned=lambda params: True
            ),
        )
    ),
}
