# Commit: [feature/muscle-templates] Add muscle models and effector templates

## Overview

Adds standalone muscle Components and composite effector templates to feedbax,
enabling muscle-driven simulations without requiring the full musculoskeletal
DAE infrastructure. Each muscle manages its own activation state via StateIndex
and uses Euler integration internally.

## Changes

### Standalone Muscle Components (`feedbax/mechanics/muscles/`)

**ReluMuscle** (~80 LOC): Minimal muscle model where force = activation * F_max.
First-order activation dynamics with separate tau_activation/tau_deactivation
time constants. Useful for point-mass tasks where detailed muscle physiology
is not needed.

**RigidTendonHillMuscleThelen** (~200 LOC): Thelen 2003 rigid tendon variant
with pre-computed force-velocity constants following MotorNet's approach. Computes
active force-length (Gaussian), passive force-length (exponential), and
force-velocity (piecewise concentric/eccentric) curves. Fiber length determined
algebraically from musculotendon length minus tendon slack length.

### Composite Effector Templates (`feedbax/mechanics/templates/`)

**Arm6MuscleRigidTendon**: Wraps 6 Thelen muscles with TwoLinkArmMuscleGeometry.
Takes excitation [6], joint angles [2], angular velocities [2] and outputs
joint torques [2], muscle forces [6], activations [6].

**PointMass8MuscleRelu**: Wraps 8 ReluMuscle instances with PointMassRadialGeometry
(4 antagonist pairs at 0/45/90/135 degrees). Takes excitation [8] and outputs
2D net force, individual muscle forces, and activations.

### Point Mass Radial Geometry (`feedbax/mechanics/geometry.py`)

**PointMassRadialGeometry**: Arranges antagonist muscle pairs radially around a
2D point mass. Directions are interleaved [pos0, neg0, pos1, neg1, ...].
Provides `forces_to_force_2d()` to convert individual muscle forces to net 2D force.

### Component Registry

All four new components registered in the web UI component registry with
appropriate parameter schemas, port types, and categories (Muscles, Mechanics).

## Rationale

The existing Hill muscle models in `hill_muscles.py` are tightly coupled to the
DAE solver framework. These new standalone Components follow the standard feedbax
Component protocol (input_ports, output_ports, __call__ with State) making them
composable in Graph topologies. The pre-computed FV constants in the Thelen model
avoid repeated computation and follow MotorNet's validated approach. Using explicit
`float32` dtype for StateIndex initial values prevents weak-type promotion issues
when multiple muscle instances share a State container.

## Files Changed

- `feedbax/mechanics/muscles/__init__.py` - New package init
- `feedbax/mechanics/muscles/relu_muscle.py` - ReluMuscle Component
- `feedbax/mechanics/muscles/thelen_muscle.py` - RigidTendonHillMuscleThelen Component
- `feedbax/mechanics/templates/__init__.py` - New package init
- `feedbax/mechanics/templates/arm_6muscle.py` - Arm6MuscleRigidTendon template
- `feedbax/mechanics/templates/pointmass_muscles.py` - PointMass8MuscleRelu template
- `feedbax/mechanics/geometry.py` - Added PointMassRadialGeometry
- `feedbax/mechanics/__init__.py` - Updated exports
- `feedbax/web/services/component_registry.py` - Registered new components
- `tests/test_muscle_templates.py` - 28 tests (all passing)
