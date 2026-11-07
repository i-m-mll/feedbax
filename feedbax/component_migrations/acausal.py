"""Frozen component migration records for acausal."""

from feedbax.acausal.boundaries import ActuationInput, BoundaryPort, SensorOutput
from feedbax.acausal.mechanics import RigidTendonHillMuscle
from feedbax.acausal.multibody import (
    Anchor,
    MusclePath,
    PlanarLink,
    PointMarker,
    RevoluteJoint,
    WorldFrame,
)
from feedbax.acausal.rotational import (
    AngleSensor,
    AngularVelocitySensor,
    GearRatio,
    Inertia,
    RotationalDamper,
    RotationalGround,
    TorqueSensor,
    TorqueSource,
    TorsionalSpring,
)
from feedbax.acausal.translational import (
    ForceSensor,
    ForceSource,
    Ground,
    LinearDamper,
    LinearSpring,
    Mass,
    MechanicsForceSensor,
    MechanicsForceSource,
    MechanicsGround,
    MechanicsLinearDamper,
    MechanicsLinearSpring,
    MechanicsMass,
    MechanicsPositionSensor,
    MechanicsVelocitySensor,
    PositionSensor,
    PrescribedMotion,
    VelocitySensor,
)
from feedbax.component_registry.history import History, LegacyEndpoint

HISTORIES = {
    ActuationInput: History(
        legacy=(LegacyEndpoint("ActuationInput", "1", 1, matches_unversioned=lambda params: True),)
    ),
    SensorOutput: History(
        legacy=(LegacyEndpoint("SensorOutput", "1", 1, matches_unversioned=lambda params: True),)
    ),
    BoundaryPort: History(
        legacy=(LegacyEndpoint("BoundaryPort", "1", 1, matches_unversioned=lambda params: True),)
    ),
    RigidTendonHillMuscle: History(
        legacy=(
            LegacyEndpoint(
                "RigidTendonHillMuscle", "1", 1, matches_unversioned=lambda params: True
            ),
        )
    ),
    PlanarLink: History(
        legacy=(LegacyEndpoint("PlanarLink", "1", 1, matches_unversioned=lambda params: True),)
    ),
    RevoluteJoint: History(
        legacy=(LegacyEndpoint("RevoluteJoint", "1", 1, matches_unversioned=lambda params: True),)
    ),
    WorldFrame: History(
        legacy=(LegacyEndpoint("WorldFrame", "1", 1, matches_unversioned=lambda params: True),)
    ),
    Anchor: History(
        legacy=(LegacyEndpoint("Anchor", "1", 1, matches_unversioned=lambda params: True),)
    ),
    MusclePath: History(
        legacy=(LegacyEndpoint("MusclePath", "1", 1, matches_unversioned=lambda params: True),)
    ),
    PointMarker: History(
        legacy=(LegacyEndpoint("PointMarker", "1", 1, matches_unversioned=lambda params: True),)
    ),
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
    Mass: History(
        legacy=(LegacyEndpoint("Mass", "1", 1, matches_unversioned=lambda params: True),)
    ),
    LinearSpring: History(
        legacy=(LegacyEndpoint("LinearSpring", "1", 1, matches_unversioned=lambda params: True),)
    ),
    LinearDamper: History(
        legacy=(LegacyEndpoint("LinearDamper", "1", 1, matches_unversioned=lambda params: True),)
    ),
    ForceSource: History(
        legacy=(LegacyEndpoint("ForceSource", "1", 1, matches_unversioned=lambda params: True),)
    ),
    Ground: History(
        legacy=(LegacyEndpoint("Ground", "1", 1, matches_unversioned=lambda params: True),)
    ),
    PrescribedMotion: History(
        legacy=(
            LegacyEndpoint("PrescribedMotion", "1", 1, matches_unversioned=lambda params: True),
        )
    ),
    PositionSensor: History(
        legacy=(LegacyEndpoint("PositionSensor", "1", 1, matches_unversioned=lambda params: True),)
    ),
    VelocitySensor: History(
        legacy=(LegacyEndpoint("VelocitySensor", "1", 1, matches_unversioned=lambda params: True),)
    ),
    ForceSensor: History(
        legacy=(LegacyEndpoint("ForceSensor", "1", 1, matches_unversioned=lambda params: True),)
    ),
    MechanicsMass: History(
        legacy=(LegacyEndpoint("MechanicsMass", "1", 1, matches_unversioned=lambda params: True),)
    ),
    MechanicsLinearSpring: History(
        legacy=(
            LegacyEndpoint(
                "MechanicsLinearSpring", "1", 1, matches_unversioned=lambda params: True
            ),
        )
    ),
    MechanicsLinearDamper: History(
        legacy=(
            LegacyEndpoint(
                "MechanicsLinearDamper", "1", 1, matches_unversioned=lambda params: True
            ),
        )
    ),
    MechanicsForceSource: History(
        legacy=(
            LegacyEndpoint("MechanicsForceSource", "1", 1, matches_unversioned=lambda params: True),
        )
    ),
    MechanicsGround: History(
        legacy=(LegacyEndpoint("MechanicsGround", "1", 1, matches_unversioned=lambda params: True),)
    ),
    MechanicsPositionSensor: History(
        legacy=(
            LegacyEndpoint(
                "MechanicsPositionSensor", "1", 1, matches_unversioned=lambda params: True
            ),
        )
    ),
    MechanicsVelocitySensor: History(
        legacy=(
            LegacyEndpoint(
                "MechanicsVelocitySensor", "1", 1, matches_unversioned=lambda params: True
            ),
        )
    ),
    MechanicsForceSensor: History(
        legacy=(
            LegacyEndpoint("MechanicsForceSensor", "1", 1, matches_unversioned=lambda params: True),
        )
    ),
}
