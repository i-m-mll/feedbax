# Commit: Add acausal mechanical modeling framework

## Overview

Implements a Modelica-style acausal modeling framework for feedbax. Physical
elements (masses, springs, dampers, etc.) are described as construction-time
equation descriptors. At `__init__` time, an assembly algorithm compiles all
element equations and connection constraints into a single JAX-traceable
vector field, which then runs as a standard `DAEComponent` inside the feedbax
Graph.

## Changes

### Core types (`feedbax/acausal/base.py`)

Defines the vocabulary of the acausal framework:

- **Domain** enum (translational / rotational) determines the physical
  semantics of ports.
- **AcausalPort** carries across-variable names (shared at a node) and a
  through-variable name (summed at a node).
- **AcausalVar** tracks each scalar variable's status: differential, eliminated
  by a connection, grounded, or driven by a causal input.
- **AcausalEquation** stores a callable RHS and its dependencies.
- **AcausalElement / AcausalConnection** are user-facing descriptors.
- **StateLayout** maps variable names to indices in the flat state vector.

### Assembly algorithm (`feedbax/acausal/assembly.py`)

The `assemble_system()` function implements the following pipeline:

1. Collect all variables from all element ports.
2. Process connections with a Union-Find to identify shared across-variable
   groups. Pick canonical representatives; eliminate aliases.
3. Handle special elements: Ground (across vars = 0), ForceSource /
   TorqueSource (through var = causal input), PrescribedMotion (across vars
   = causal input), Sensors (read-only output), GearRatio (algebraic
   constraint).
4. Build the differential variable list (state vector).
5. Compile a vector field function that:
   - Reads state from `y`, inputs and params from `args`.
   - Evaluates through-variable equations in topological order.
   - Sums through-vars at each mass/inertia node for force balance.
   - Returns `dy` with position derivatives (= velocity) and velocity
     derivatives (= net force / mass).

All Python-level indexing is pre-computed so the vector field body contains
only `jnp` operations and integer-literal array access, making it fully
JAX-traceable.

### Sign convention

Through-variables at a port represent the force/torque exerted **on the
external body** connected at that port. For a spring with `pos_a < pos_b`
(stretched), `force_b = k*(pos_a - pos_b) < 0` pulls B back toward A,
which is physically correct. The net force on a mass is the sum of all
connected through-vars (excluding the mass's own).

### Translational elements (`feedbax/acausal/translational.py`)

Mass, LinearSpring, LinearDamper, Ground, ForceSource, PrescribedMotion,
PositionSensor, VelocitySensor, ForceSensor.

### Rotational elements (`feedbax/acausal/rotational.py`)

Inertia, TorsionalSpring, RotationalDamper, RotationalGround, TorqueSource,
GearRatio.

### DAEComponent bridge (`feedbax/acausal/system.py`)

`AcausalSystem` subclasses `DAEComponent[AcausalSystemState]`. It runs the
assembly algorithm in `__init__`, stores the compiled vector field and layout,
and delegates to diffrax for integration. Input routing collects named causal
inputs into a flat array.

### Component registry

Registered `AcausalSystem` in the web UI component registry under Mechanics.

### Tests

23 tests covering:
- Assembly correctness (layout, elimination, grounding, params)
- Physics (force acceleration, damped oscillation, energy conservation <1%)
- Multi-connection force balance (3 springs at one node)
- JIT compilation and gradient flow
- Sensor readings match state vector
- Long-horizon stability (10k steps without NaN or divergence)
- Rotational domain (torsional spring oscillation, gear ratio)
- Graph integration (constant force, P-controller feedback loop,
  prescribed motion)

## Rationale

Modelica-style acausal modeling is the standard approach for multi-domain
physical systems in engineering. By separating element description (pure
Python dataclasses) from execution (JAX vector field), we get:

1. **Composability**: Users snap together elements without worrying about
   causality or variable ordering.
2. **Differentiability**: The compiled vector field is fully JAX-traceable,
   enabling gradient-based parameter fitting and neural network integration.
3. **Graph compatibility**: `AcausalSystem` is a standard `Component`, so it
   can be wired into feedbax Graphs alongside neural networks, controllers,
   and other components.

The through-variable sign convention ("force on external body") was chosen
because it makes force balance at mass nodes a simple sum, avoiding the
Modelica convention of "flow into component" which requires sign flips.

## Files Changed

- `feedbax/acausal/__init__.py` -- Package init with public exports
- `feedbax/acausal/base.py` -- Core types (Domain, Port, Var, Equation, Element, Connection, Layout)
- `feedbax/acausal/assembly.py` -- Union-Find + assembly algorithm + vector field builder
- `feedbax/acausal/system.py` -- AcausalSystem (DAEComponent bridge)
- `feedbax/acausal/translational.py` -- Translational-domain elements
- `feedbax/acausal/rotational.py` -- Rotational-domain elements
- `feedbax/web/services/component_registry.py` -- Register AcausalSystem
- `tests/test_acausal.py` -- Core tests (19 tests)
- `tests/test_acausal_graph.py` -- Graph integration tests (4 tests)
