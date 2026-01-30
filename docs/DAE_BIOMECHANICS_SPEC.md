# DAE & Biomechanics Implementation Spec

This spec outlines the implementation path for DAE (Differential-Algebraic Equation)
subgraphs, starting with preformed mechanics nodes.

**Related GitHub Issue:** [#49 - DAE/Acausal subgraphs with domain-specific component palettes](https://github.com/i-m-mll/feedbax/issues/49)

---

## Overview

Rather than implementing full acausal/DAE editing from scratch, we start with
**preformed DAE nodes** for our existing mechanics types. These are components that
internally use DAE solving but present a simple causal interface externally.

---

## Phase 1: Preformed DAE Mechanics Nodes

### PointMassDAE

A point mass with optional muscle-like actuators, using implicit integration internally.

```yaml
name: PointMassDAE
category: Mechanics
description: Point mass with implicit DAE integration (handles stiff dynamics)
params:
  - dt: float (timestep)
  - mass: float
  - damping: float
  - integration_method: enum [euler, rk4, implicit_midpoint]
inputs:
  - force: vector (external force input)
outputs:
  - effector: state (position, velocity)
  - state: state (full internal state)
```

### TwoLinkArmDAE

A two-link arm with proper inertia matrix and optional compliant tendon muscles.

```yaml
name: TwoLinkArmDAE
category: Mechanics
description: Two-link arm with implicit DAE integration
params:
  - dt: float
  - link_lengths: array[2]
  - link_masses: array[2]
  - damping: float
  - integration_method: enum [euler, rk4, implicit_midpoint]
inputs:
  - torque: vector (joint torques) OR
  - muscle_excitation: vector (if muscles attached)
outputs:
  - effector: state (end-effector position, velocity)
  - joint_state: state (joint angles, velocities)
  - state: state (full internal state)
```

### MusculoskeletalArmDAE

A complete arm model with Hill-type muscles, using DAE for compliant tendon dynamics.

```yaml
name: MusculoskeletalArmDAE
category: Mechanics
description: Two-link arm with 6 Hill-type muscles (compliant tendon)
params:
  - dt: float
  - muscle_params: struct (max_force, optimal_length, etc. per muscle)
  - integration_method: enum [implicit_midpoint, bdf]
inputs:
  - excitation: vector[6] (muscle excitations 0-1)
outputs:
  - effector: state
  - joint_state: state
  - muscle_state: state (activations, fiber lengths, forces)
```

---

## Phase 2: DAE Solver Integration

### Backend Changes

1. **Add `diffrax` integration** for implicit ODE/DAE solving
   ```python
   from diffrax import diffeqsolve, ImplicitEuler, Kvaerno5
   ```

2. **Create `DAEComponent` base class**
   ```python
   class DAEComponent(Component):
       """Component using implicit DAE integration internally."""
       solver: AbstractSolver = field(static=True)

       def __call__(self, inputs, state, *, key):
           # Use diffrax for internal integration
           solution = diffeqsolve(
               self.ode_fn, self.solver, t0, t1, dt0, y0
           )
           ...
   ```

3. **Muscle model implementations** (see MOTORNET_COMPARISON.md)
   - `RigidTendonHillMuscle` - explicit ODE, can use existing solver
   - `CompliantTendonHillMuscle` - stiff DAE, needs implicit solver

### Files to Create/Modify

- `feedbax/mechanics/dae.py` - DAE component base class
- `feedbax/mechanics/arm_dae.py` - TwoLinkArmDAE implementation
- `feedbax/mechanics/muscles.py` - Hill muscle models
- `feedbax/web/services/component_registry.py` - Register new components

---

## Phase 3: Editable DAE Subgraphs (Future)

Once preformed DAE nodes work, we can expose their internals as editable subgraphs:

1. **Enter DAE node** → See internal acausal structure
2. **Component palette changes** → Shows acausal components (Spring, Damper, Mass, Muscle)
3. **Bidirectional wires** → Physical connections, not data flow
4. **Exit subgraph** → Compiles back to causal interface

This is the full vision from issue #49, but preformed nodes come first.

---

## Implementation Order

| Step | Component | Complexity | Dependencies |
|------|-----------|------------|--------------|
| 1 | Add `diffrax` to dependencies | Low | None |
| 2 | Create `DAEComponent` base class | Medium | diffrax |
| 3 | Implement `PointMassDAE` | Low | DAEComponent |
| 4 | Implement `TwoLinkArmDAE` | Medium | DAEComponent |
| 5 | Implement `RigidTendonHillMuscle` | Medium | None (explicit ODE) |
| 6 | Implement `CompliantTendonHillMuscle` | High | DAEComponent |
| 7 | Implement `MusculoskeletalArmDAE` | High | All above |
| 8 | Register in component_registry | Low | Implementations |

---

## Why Preformed First?

1. **Incremental progress** - Get working DAE mechanics without full UI overhaul
2. **Validates solver integration** - Ensures diffrax works with our architecture
3. **Useful immediately** - Can train models with realistic muscle dynamics
4. **Informs acausal design** - Learn what internal structure users need to edit

---

## References

- [GitHub Issue #49](https://github.com/i-m-mll/feedbax/issues/49)
- [MOTORNET_COMPARISON.md](./MOTORNET_COMPARISON.md) - Hill muscle model details
- [diffrax documentation](https://docs.kidger.site/diffrax/)

---

*Co-authored by Claude Opus 4.5*
