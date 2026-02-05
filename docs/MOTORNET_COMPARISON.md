# MotorNet Comparison & Component Analysis

This document compares MotorNet's architecture to Feedbax and identifies biomechanics
components we should consider implementing.

---

## Part 1: MotorNet Overview

### What is MotorNet?

MotorNet is a Python/PyTorch toolbox for training recurrent neural networks to control
biomechanically realistic effectors. It focuses on differentiable musculoskeletal simulation
for motor control research.

**Key characteristics:**
- Built on PyTorch (we use JAX/equinox)
- Gymnasium-compatible environments
- Differentiable end-to-end for gradient-based training
- Focus on upper limb motor control

### Architecture Comparison

| Aspect | MotorNet | Feedbax |
|--------|----------|---------|
| **Framework** | PyTorch | JAX/equinox |
| **Graph model** | Fixed hierarchy: Environment → Effector → (Skeleton + Muscle) | Flexible component graph with wires |
| **State** | Distributed across objects (joint_state, muscle_state, geometry_state) | Unified state PyTree |
| **Composition** | Hard-coded nesting (Effector contains Skeleton and Muscle) | Wired components in arbitrary graphs |
| **Differentiability** | Yes (PyTorch autograd) | Yes (JAX autodiff) |
| **Integration** | Euler or RK4, explicit | Currently explicit Euler (DAE planned) |

### MotorNet's Computational Paradigm

MotorNet uses a **fixed three-level hierarchy**:

```
Environment (gymnasium.Env)
    └── Effector (torch.nn.Module)
            ├── Skeleton (defines kinematics, dynamics)
            └── Muscle (defines force generation)
```

**Execution flow:**
1. Environment receives action from policy
2. Effector.step() integrates skeleton + muscle dynamics
3. Skeleton computes joint accelerations from forces
4. Muscle computes forces from excitation + length/velocity
5. Geometry state (moment arms, tendon lengths) computed online

**Limitations:**
- Can't easily add components outside this hierarchy
- Skeleton and Muscle must match (hard-coded effector presets)
- No visual graph editor or UI
- Limited to arm/point-mass effectors currently

**Advantages:**
- Well-validated biomechanics (Hill muscle models)
- Clean separation of skeleton kinematics vs. muscle dynamics
- Moment arm computation from muscle path geometry
- Multiple integration methods (Euler, RK4)

---

## Part 2: MotorNet Component Classes

### Skeleton Classes

| Class | Description | We Have? |
|-------|-------------|----------|
| `Skeleton` | Base class for kinematic chains | (base class) |
| `PointMass` | 2D point mass with direct force input | ✅ PointMass |
| `TwoDofArm` | 2-link planar arm with shoulder + elbow | ✅ TwoLinkArm |

### Muscle Classes

| Class | Description | We Have? |
|-------|-------------|----------|
| `Muscle` | Base class for muscle models | ❌ |
| `ReluMuscle` | Simple linear force = activation × max_force | ❌ |
| `MujocoHillMuscle` | Rigid tendon Hill model (MuJoCo-style) | ❌ |
| `RigidTendonHillMuscle` | Rigid tendon Hill model with force-length-velocity | ❌ |
| `RigidTendonHillMuscleThelen` | Thelen 2003 rigid tendon model | ❌ |
| `CompliantTendonHillMuscle` | Full compliant tendon Hill model (stiff DAE) | ❌ |

### Pre-built Effector Classes

| Class | Description | We Have? |
|-------|-------------|----------|
| `Effector` | Base class combining skeleton + muscle | ❌ |
| `ReluPointMass24` | Point mass with 2×4=8 ReluMuscles | ❌ |
| `RigidTendonArm26` | 2-link arm with 6 rigid tendon Hill muscles | ❌ |
| `CompliantTendonArm26` | 2-link arm with 6 compliant tendon Hill muscles | ❌ |

---

## Part 3: Hill Muscle Model Details

MotorNet's Hill muscle models are the most significant biomechanics components we lack.

### What is a Hill Muscle Model?

A phenomenological model of muscle force production with three elements:
1. **Contractile Element (CE)** - active force from muscle fibers
2. **Series Elastic Element (SE)** - tendon elasticity
3. **Parallel Elastic Element (PE)** - passive muscle elasticity

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `max_isometric_force` | Maximum force at optimal length (N) |
| `optimal_muscle_length` | Length at which CE produces max force (m) |
| `tendon_slack_length` | Tendon length at zero force (m) |
| `pennation_angle` | Fiber angle relative to tendon (rad) |
| `tau_activation` | Activation time constant (s) |
| `tau_deactivation` | Deactivation time constant (s) |
| `vmax` | Maximum shortening velocity (lengths/s) |

### Force-Length-Velocity Relationships

**Active force-length:** Bell-shaped curve, max at optimal length
```
f_L(l) = exp(-((l - l_opt) / width)^2)
```

**Force-velocity:** Hyperbolic relationship (Hill equation)
```
f_V(v) = (1 - v/vmax) / (1 + v/(vmax * a))  for shortening
f_V(v) = (1 + v * b) / (1 - v / vmax_ecc)   for lengthening
```

**Passive force:** Exponential, engages when stretched
```
f_PE(l) = exp(k * (l - l_slack)) - 1  for l > l_slack
```

### Rigid vs. Compliant Tendon

| Type | Description | Numerical Difficulty |
|------|-------------|---------------------|
| **Rigid tendon** | Tendon length constant; muscle fiber length = total - tendon | Easy (ODE) |
| **Compliant tendon** | Tendon stretches under load; equilibrium constraint | Hard (DAE/stiff ODE) |

Compliant tendon models are more realistic but create stiff equations that benefit from
implicit integration (relates to our DAE subgraph discussion).

---

## Part 4: Components We Should Implement

### Priority 1: Basic Muscle Models

```python
# Proposed component registry entries

ReluMuscle:
  category: Biomechanics/Muscles
  description: Simple linear muscle (force = activation × max_force)
  params: [max_isometric_force, tau_activation, tau_deactivation]
  inputs: [excitation]
  outputs: [force, activation]

RigidTendonHillMuscle:
  category: Biomechanics/Muscles
  description: Hill-type muscle with rigid tendon
  params: [max_isometric_force, optimal_length, tendon_slack_length,
           pennation_angle, tau_activation, tau_deactivation, vmax]
  inputs: [excitation, muscle_length, muscle_velocity]
  outputs: [force, activation, fiber_length, fiber_velocity]
```

### Priority 2: Muscle-Skeleton Integration

```python
MuscleWrap:
  category: Biomechanics/Geometry
  description: Defines muscle path around skeleton (fixation points)
  params: [fixation_bodies, fixation_coordinates]
  inputs: [joint_state]
  outputs: [musculotendon_length, musculotendon_velocity, moment_arms]

MomentArmTransform:
  category: Biomechanics/Geometry
  description: Converts muscle forces to joint torques via moment arms
  inputs: [muscle_forces, moment_arms]
  outputs: [joint_torques]
```

### Priority 3: Pre-built Effector Subgraphs

Rather than hard-coded effector classes, we could provide subgraph templates:

- **6-muscle arm subgraph**: TwoLinkArm + 6 Hill muscles with anatomical paths
- **Point mass with muscles**: PointMass + 4-8 muscles in antagonist pairs

These would be DAE subgraphs (when implemented) for proper handling of compliant tendons.

---

## Part 5: Architectural Implications

### Why Our Architecture is Better for Extensibility

MotorNet's fixed hierarchy (Environment → Effector → Skeleton + Muscle) means:
- Can't easily add components between Skeleton and Muscle
- Can't mix different muscle types on same skeleton
- No visual representation of the computation graph

Our wire-based graph allows:
- Arbitrary muscle-skeleton configurations
- Mixing component types
- Visual editing of biomechanical models
- Barnacles for probing internal muscle states

### What We Need to Match Their Capabilities

1. **Muscle models** - Hill-type with force-length-velocity curves
2. **Geometry computation** - Muscle path → moment arms
3. **DAE support** - For compliant tendon models (stiff equations)
4. **Pre-built templates** - Common arm configurations as subgraphs

---

## Part 6: Implementation Roadmap

### Phase 1: Basic Muscles (no DAE needed)
- ReluMuscle (linear)
- RigidTendonHillMuscle (explicit ODE)
- MuscleWrap for geometry
- MomentArmTransform

### Phase 2: DAE Subgraphs
- CompliantTendonHillMuscle (requires implicit solving)
- Package as DAE subgraph with acausal internal wiring

### Phase 3: Pre-built Effector Templates
- 6-muscle arm template (subgraph)
- 8-muscle point mass template (subgraph)
- Anatomically-based muscle path presets

---

## References

- [MotorNet GitHub](https://github.com/motornet-org/MotorNet)
- [MotorNet Documentation](https://www.motornet.org/)
- [MotorNet Paper (eLife)](https://elifesciences.org/articles/88591)
- Thelen DG (2003). Adjustment of muscle mechanics model parameters to simulate dynamic
  contractions in older adults. J Biomech Eng.

---

*Co-authored by Claude Opus 4.5*
