"""Exact historical identities for the physical element owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .rotational import (
    Inertia,
    TorsionalSpring,
    RotationalDamper,
    TorqueSource,
    RotationalGround,
    GearRatio,
    AngleSensor,
    AngularVelocitySensor,
    TorqueSensor,
)

HISTORIES = {
    Inertia: History(
        legacy=(LegacyEndpoint("Inertia", "1", 1, matches_unversioned=lambda params: True),)
    ),
    TorsionalSpring: History(
        legacy=(LegacyEndpoint("TorsionalSpring", "1", 1, matches_unversioned=lambda params: True),)
    ),
    RotationalDamper: History(
        legacy=(
            LegacyEndpoint("RotationalDamper", "1", 1, matches_unversioned=lambda params: True),
        )
    ),
    TorqueSource: History(
        legacy=(LegacyEndpoint("TorqueSource", "1", 1, matches_unversioned=lambda params: True),)
    ),
    RotationalGround: History(
        legacy=(
            LegacyEndpoint("RotationalGround", "1", 1, matches_unversioned=lambda params: True),
        )
    ),
    GearRatio: History(
        legacy=(LegacyEndpoint("GearRatio", "1", 1, matches_unversioned=lambda params: True),)
    ),
    AngleSensor: History(
        legacy=(LegacyEndpoint("AngleSensor", "1", 1, matches_unversioned=lambda params: True),)
    ),
    AngularVelocitySensor: History(
        legacy=(
            LegacyEndpoint(
                "AngularVelocitySensor", "1", 1, matches_unversioned=lambda params: True
            ),
        )
    ),
    TorqueSensor: History(
        legacy=(LegacyEndpoint("TorqueSensor", "1", 1, matches_unversioned=lambda params: True),)
    ),
}
