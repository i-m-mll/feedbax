# Collimator.ai Comparison & Component Analysis

This document compares Collimator.ai's approach to ours, analyzes their component library,
and identifies components we should consider adding.

---

## Part 1: Collimator.ai Overview

### What is Collimator?

Collimator is a cloud-based simulation platform for model-based design of multi-domain
dynamical systems. Built on Python and JAX (like us!), it supports:
- Differentiable simulation
- Neural network integration (uses equinox for MLP blocks)
- PID tuning via auto-differentiation
- State-space linearization
- Model-predictive control

### How Collimator Handles State and Wires

Collimator uses a **block diagram paradigm** where:
1. **Wires are explicit data flow** - they represent actual signal connections
2. **State is distributed** - each block with continuous state (e.g., Integrator) maintains its own
3. **Composite systems** are built via DiagramBuilder connecting leaf systems
4. **No unified state object** - state is implicit in the block hierarchy

This is fundamentally different from our approach where:
- We pass a **unified state PyTree** through all components
- Wires are **keyed slots** into that state (publish/subscribe pattern)
- The state contains everything; wires visualize which keys are being routed

### How Collimator Handles Probes/Monitoring

Collimator does **NOT** have separate "probe" or "barnacle" concepts. Instead:
- **Every block output can be recorded** - you choose "All outputs" or specific ones
- **Post-hoc visualization** - the Visualizer can be turned on/off for any block after simulation
- **No special probe nodes needed** - monitoring is a simulation setting, not a graph element

This is simpler but less flexible than our barnacle approach:
- ✅ Simpler: no special UI elements needed
- ❌ Limited: can only observe block outputs, not internal state paths
- ❌ No intervention capability built into probes

### How Collimator Handles Interventions

Collimator treats interventions as **regular blocks**:
- Force fields, noise injection, etc. are just components in the graph
- No special "intervention" category with timing control
- Parameters can vary via input signals (wires from other blocks)

Our barnacle approach offers:
- ✅ Access to full state (not just ports)
- ✅ Timing control (before/after component execution)
- ✅ Cleaner separation of "model" vs "experimental manipulation"
- ❌ More complex UI/conceptual model

---

## Part 2: Collimator Block Library

### Categories

1. **Continuous Blocks** - require continuous solver (ODE integration)
2. **Discrete Blocks** - operate on discrete time steps
3. **Agnostic Blocks** - can be either (Gain, Saturation, etc.)
4. **Acausal Blocks** - physical domain modeling (electrical, mechanical, thermal)
5. **High Level Blocks** - complex composite blocks
6. **State Machine** - finite state machine modeling
7. **Python Script Block** - custom code
8. **Predictor Block** - ML inference

### Complete Block List by Category

#### Math Operations
| Block | Description | We Have? |
|-------|-------------|----------|
| Gain | Multiply by constant | ❌ (was in old registry) |
| Sum/Adder | Add/subtract signals | ❌ (was in old registry) |
| Product/Multiply | Element-wise multiplication | ❌ (was in old registry) |
| Abs | Absolute value | ❌ |
| Sqrt | Square root | ❌ |
| Sin/Cos/Tan | Trigonometric functions | ❌ |
| Exp/Log | Exponential/logarithm | ❌ |
| Power | Raise to power | ❌ |
| MinMax | Element-wise min/max | ❌ |
| Sign | Sign function | ❌ |
| Mod | Modulo operation | ❌ |

#### Signal Sources
| Block | Description | We Have? |
|-------|-------------|----------|
| Constant | Constant value output | ❌ (was in old registry) |
| Step | Step function | ❌ |
| Ramp | Linear ramp | ❌ (was in old registry) |
| Sine | Sinusoidal signal | ❌ (was in old registry) |
| Pulse | Pulse/square wave | ❌ (was in old registry) |
| Clock | Time signal | ❌ |
| RandomNoise | Random noise source | ❌ |

#### Continuous Dynamics
| Block | Description | We Have? |
|-------|-------------|----------|
| Integrator | Continuous integration | ❌ |
| Derivative | Time derivative | ❌ |
| TransferFunction | LTI transfer function | ❌ |
| StateSpace | State-space system | ❌ |
| PID | PID controller | ❌ |

#### Discrete Dynamics
| Block | Description | We Have? |
|-------|-------------|----------|
| IntegratorDiscrete | Discrete integration | ❌ |
| DerivativeDiscrete | Discrete derivative | ❌ |
| UnitDelay | z^-1 delay | ❌ |
| TransferFunctionDiscrete | Discrete transfer function | ❌ |
| ZeroOrderHold | Sample and hold | ❌ |
| PIDDiscrete | Discrete PID | ❌ |

#### Signal Routing
| Block | Description | We Have? |
|-------|-------------|----------|
| Mux | Combine signals | ❌ |
| Demux | Split signals | ❌ |
| Switch | Conditional routing | ❌ |
| Selector | Index into vector | ❌ |

#### Nonlinearities
| Block | Description | We Have? |
|-------|-------------|----------|
| Saturation | Clamp to range | ❌ (was in old registry) |
| DeadZone | Zero output in range | ❌ |
| RateLimiter | Limit rate of change | ❌ |
| Quantizer | Discretize amplitude | ❌ |
| Backlash | Hysteresis | ❌ |
| LookupTable | Interpolated table | ❌ |

#### Neural Networks / ML
| Block | Description | We Have? |
|-------|-------------|----------|
| MLP | Multi-layer perceptron | ❌ (was in old registry) |
| (Custom NN) | Import from PyTorch/TensorFlow | ❌ |

#### Filtering
| Block | Description | We Have? |
|-------|-------------|----------|
| FirstOrderFilter | Low-pass filter | ✅ FirstOrderFilter |
| LowPassFilter | Configurable LP filter | (covered by above) |
| HighPassFilter | HP filter | ❌ |
| BandPassFilter | BP filter | ❌ |
| KalmanFilter | State estimation | ❌ |

#### Acausal/Physical Domains

**Electrical:**
| Block | Description |
|-------|-------------|
| Resistor | Ohmic resistance |
| Capacitor | Capacitance |
| Inductor | Inductance |
| VoltageSource | Ideal voltage source |
| CurrentSource | Ideal current source |
| Ground | Electrical ground |
| Battery | Battery cell model |

**Mechanical (Translational):**
| Block | Description | We Have? |
|-------|-------------|----------|
| Mass | Point mass | ✅ PointMass |
| Spring | Linear spring | ❌ (was in old registry) |
| Damper | Viscous damper | ❌ (was in old registry) |
| ForceSource | Force input | ❌ |

**Mechanical (Rotational):**
| Block | Description |
|-------|-------------|
| Inertia | Rotational inertia |
| RotationalSpring | Torsional spring |
| RotationalDamper | Rotational damping |
| GearRatio | Gear reduction |
| TorqueSource | Torque input |

**Thermal:**
| Block | Description |
|-------|-------------|
| ThermalMass | Heat capacity |
| ThermalResistor | Thermal resistance |
| HeatSource | Heat input |

---

## Part 3: Our Current Registry vs. Pre-Merge

### Components We Currently Have
- Network (with GRU/LSTM/Linear options)
- Subgraph
- TwoLinkArm
- PointMass
- Mechanics (generic)
- Channel
- FirstOrderFilter
- CurlField
- FixedField
- AddNoise
- NetworkClamp
- NetworkConstantInput
- ConstantInput
- SimpleReaches
- DelayedReaches
- Stabilization

### Components Lost in Merge (should restore)
- **MLP** - standalone multi-layer perceptron
- **GRU** - standalone GRU cell/network
- **LSTM** - standalone LSTM cell/network
- **Gain** - multiply by constant
- **Sum** - add/subtract signals
- **Multiply** - element-wise product
- **Constant** - constant value source
- **Spring** - linear spring dynamics
- **Damper** - viscous damper dynamics
- **Saturation** - clamp to range
- **DelayLine** - discrete delay buffer
- **Noise** - noise source
- **Ramp** - linear ramp signal
- **Sine** - sinusoidal signal
- **Pulse** - pulse/square wave
- **Probe** - observation point (now → barnacles)
- **Clamp** - value clamping
- **ForceField** - generic force field
- **Perturbation** - generic perturbation
- **NoiseInjector** - noise injection
- **HoldTask, ReachingTask, TrackingTask** - alternative task types

---

## Part 4: Key Architectural Questions

### The `intervene` Output Port on Tasks

In the old staged model architecture, the Task would generate intervention parameters
that varied trial-by-trial and feed them to Intervenor components separately.

With the unified state object:
- The `intervene` output could populate a part of the state that interventions read from
- OR: interventions could be parameterized directly (not via Task)
- OR: this port is vestigial and should be removed

**Recommendation**: The Task should still be able to provide trial-varying parameters
(e.g., curl field direction varies per trial). These go into state, and barnacle-style
interventions read from state to get their parameters. The wiring is implicit via state
keys rather than explicit port connections.

### Barnacles vs. Regular Components for Interventions

Current approach: CurlField, FixedField, etc. are regular components with ports.

Barnacle approach would:
1. Attach to nodes (not wires)
2. Access full state at that execution point
3. Specify timing (before/after component)
4. Be parameterized via state paths (including trial-varying params from Task)

**The key insight**: Barnacles solve the problem of accessing state that ISN'T exposed
as ports. For things that ARE naturally port-based (like adding force to a mechanics
input), regular components are fine. Barnacles are for:
- Observing internal state (e.g., network hidden activity)
- Modifying state paths that aren't ports (e.g., muscle activation)

### Task + Model Subgraph Structure

You're right that the reaching model should be:
```
[Task] ──────────────────────────> [Model (Subgraph)]
         inputs, targets, inits      ↓
                                   [Inside Model:]
                                   Network ↔ Mechanics
                                   (feedback loop)
```

The Task is outside, the closed-loop controller is inside a subgraph.

---

## Part 5: Collimator Advantages vs. Ours

### Collimator Advantages
1. **Rich acausal modeling** - electrical, thermal, rotational domains
2. **Standard control blocks** - PID, transfer functions, state-space
3. **Mature tooling** - linearization, code generation
4. **Simpler mental model** - wires = data flow, no hidden state routing

### Our Advantages
1. **Full state access** - barnacles can observe/modify anything
2. **Trial-varying parameters** - deep integration with Task system
3. **Neural network focus** - RNNs, training loops, loss functions built-in
4. **Neuroscience domain** - arm mechanics, reaching tasks, muscle models
5. **JAX ecosystem** - same stack, but more specialized

### Potential Issues Implementing Collimator-style Blocks

1. **Acausal blocks** - require equation-based solving (DAE), not just feed-forward
2. **Continuous integrators** - we use discrete stepping; true ODE solving is different
3. **Transfer functions** - need scipy.signal integration for tf2ss conversion
4. **State machines** - different execution model than our graph

---

## Part 6: Recommendations

### Immediate (restore lost components)
1. Gain, Sum, Multiply - basic math
2. Constant, Ramp, Sine, Pulse - signal sources
3. MLP, GRU, LSTM - standalone neural modules
4. Spring, Damper - mechanical primitives
5. Saturation, DelayLine - signal processing

### Medium-term (new components)
1. PID controller (discrete)
2. Mux/Demux for vector manipulation
3. Switch for conditional routing
4. Kalman filter for state estimation

### Long-term (architectural)
1. Full barnacle implementation for probes/interventions
2. Acausal domain modeling (if needed for muscle/tendon models)
3. State machine support for task phases

---

## Sources

- [Collimator Main Site](https://www.collimator.ai/)
- [Collimator Docs](https://docs.collimator.ai/)
- [Collimator Python API](https://py.collimator.ai/)
- [Block Library Reference](https://py.collimator.ai/library/)
- [Acausal Blocks](https://docs.collimator.ai/using-the-model-editor/block-library/acausal-blocks)
- [ML Integration Blog Post](https://www.collimator.ai/post/machine-learning-control-and-hil-testing-with-collimator-and-jax)

---

*Co-authored by Claude Opus 4.5*
