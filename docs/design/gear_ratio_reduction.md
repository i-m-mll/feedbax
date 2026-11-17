# Ideal gear reduction

`GearRatio` retains its signed kinematic convention: angle_b = ratio * angle_a
and angular_velocity_b = ratio * angular_velocity_a. A ratio is finite, nonzero,
and constant during a simulation; it remains a differentiable model parameter.

Each physical shaft has a coordinate map q_i = s_i(parameters) q. Assembly uses
that same map for constitutive laws, observations, virtual work, and kinetic
energy. Applied shaft efforts pull back as sum(s_i * torque_i), and rotational
inertias pull back as sum(s_i² * J_i). One acceleration is solved per independent
coordinate. Internal ideal gear reactions perform zero total power.

Angle and velocity sensors report physical shaft quantities. A TorqueSensor
reports the net torque on its physical node, including transmission reactions:
the sum of local inertias times that shaft’s physical acceleration. It reports
zero on a massless shaft. It does not expose individual gear reactions.

The reducer supports nonsingular gear trees, including chains, branches, negative
ratios, and grounded shafts. Cyclic constraints and multiple independently fixed
coordinates in a gear component are rejected explicitly. Authoring reports count
the reduced evolution equations while preserving missing-equation diagnostics
for unsupported free massless coordinates.

## Saved graphs

Saved gear graphs use `feedbax.spec.acausal_graph.v2`. The correction implements the existing ideal-lossless gear intent and ratio convention and preserves the authored type, ports, parameters, units, topology, and serialization.

The consumer audit examined rlrmp2 revision
`7d60e2b36f5c163c2d5acdce9d5ac8788a61bd34`, 23,738 local rlrmp2 JSON/YAML/JSONL
artifacts, 1,091 local Feedbax JSON/YAML candidates, and four Studio saves. No
GearRatio or acausal graph payload was identified. Direct and nested serialized
gear graphs exercise the current admission path. The audit covered local custody.

Co-Authored-By: Codex (GPT) <codex@openai.com>
