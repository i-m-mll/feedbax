"""Exact historical identities for the physical element owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .translational import Mass, LinearSpring, LinearDamper, ForceSource, Ground, PrescribedMotion, PositionSensor, VelocitySensor, ForceSensor, MechanicsMass, MechanicsLinearSpring, MechanicsLinearDamper, MechanicsForceSource, MechanicsGround, MechanicsPositionSensor, MechanicsVelocitySensor, MechanicsForceSensor

HISTORIES = {
    Mass: History(legacy=(LegacyEndpoint("Mass", "1", 1, matches_unversioned=lambda params: True),)),
    LinearSpring: History(legacy=(LegacyEndpoint("LinearSpring", "1", 1, matches_unversioned=lambda params: True),)),
    LinearDamper: History(legacy=(LegacyEndpoint("LinearDamper", "1", 1, matches_unversioned=lambda params: True),)),
    ForceSource: History(legacy=(LegacyEndpoint("ForceSource", "1", 1, matches_unversioned=lambda params: True),)),
    Ground: History(legacy=(LegacyEndpoint("Ground", "1", 1, matches_unversioned=lambda params: True),)),
    PrescribedMotion: History(legacy=(LegacyEndpoint("PrescribedMotion", "1", 1, matches_unversioned=lambda params: True),)),
    PositionSensor: History(legacy=(LegacyEndpoint("PositionSensor", "1", 1, matches_unversioned=lambda params: True),)),
    VelocitySensor: History(legacy=(LegacyEndpoint("VelocitySensor", "1", 1, matches_unversioned=lambda params: True),)),
    ForceSensor: History(legacy=(LegacyEndpoint("ForceSensor", "1", 1, matches_unversioned=lambda params: True),)),
    MechanicsMass: History(legacy=(LegacyEndpoint("MechanicsMass", "1", 1, matches_unversioned=lambda params: True),)),
    MechanicsLinearSpring: History(legacy=(LegacyEndpoint("MechanicsLinearSpring", "1", 1, matches_unversioned=lambda params: True),)),
    MechanicsLinearDamper: History(legacy=(LegacyEndpoint("MechanicsLinearDamper", "1", 1, matches_unversioned=lambda params: True),)),
    MechanicsForceSource: History(legacy=(LegacyEndpoint("MechanicsForceSource", "1", 1, matches_unversioned=lambda params: True),)),
    MechanicsGround: History(legacy=(LegacyEndpoint("MechanicsGround", "1", 1, matches_unversioned=lambda params: True),)),
    MechanicsPositionSensor: History(legacy=(LegacyEndpoint("MechanicsPositionSensor", "1", 1, matches_unversioned=lambda params: True),)),
    MechanicsVelocitySensor: History(legacy=(LegacyEndpoint("MechanicsVelocitySensor", "1", 1, matches_unversioned=lambda params: True),)),
    MechanicsForceSensor: History(legacy=(LegacyEndpoint("MechanicsForceSensor", "1", 1, matches_unversioned=lambda params: True),)),
}
