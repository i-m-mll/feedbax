# Mechanics

## Units and Frames

Mechanics modules use SI units unless a field name explicitly says it is
normalized or dimensionless:

- lengths are metres, masses are kilograms, time is seconds
- forces are newtons and torques are newton-metres
- angles are radians and angular velocities are radians per second
- Cartesian states are in a right-handed XY frame
- muscle fiber and musculotendon velocities are positive for lengthening and
  negative for shortening

The shared constructor-time checks and numerical constants for these
conventions live in `feedbax.mechanics.units`.

::: feedbax.mechanics.MechanicsState

::: feedbax.mechanics.Mechanics
    options:
        members: [
            '__init__',
            'init',
            'model_spec',
            'dynamics_step',
        ]
