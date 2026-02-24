# Commit: Fix 5 acausal framework issues from Codex review

## Overview

Post-merge Codex review of the acausal mechanics framework (Workstream 3)
identified 5 issues: 1 critical, 3 major, 1 minor. All are now fixed and
verified with 103 passing tests.

## Changes

### Issue 1 (Critical): Input routing for multi-input elements

`AcausalSystem.__call__` was appending the entire element input vector for
every input variable, duplicating values for elements with multiple inputs
(e.g. PrescribedMotion has pos+vel). Fixed by adding `input_specs` mapping
(`var_fqn -> (element_name, slot_index)`) to `StateLayout`, then building
the flat input array by indexing into each element's input at the correct
slot.

### Issue 2 (Major): Gear ratio constraints not enforced

Two bugs combined to produce zero torque through gears:

1. **Node merging**: Gear across-variable elimination (`canon_b -> canon_a`
   in the `eliminated` dict) caused force balance code to resolve node B's
   key to node A, mixing through-vars from both physical nodes. Spring
   torques `tau_a = -tau_b` then cancelled to zero. Fixed by saving
   `eliminated_pre_gear` before gear processing and using it for all
   node resolution.

2. **Undefined torque**: The old gear equation `tau_b = tau_a / ratio`
   referenced `tau_a` which was never defined. Replaced with proper
   node-B balance equations:
   - `gear.flange_b.torque = -(sum of non-gear through vars at node B)`
   - `gear.flange_a.torque = ratio * gear.flange_b.torque`

3. **Empty gear_ratio_scale**: The across-variable scaling dict was also
   built with post-gear `eliminated`, making `canon_b == canon_a` and the
   dict empty. Fixed by using `_epg` for this lookup too.

### Issue 3 (Major): ForceSensor outputs never emitted

`output_indices` only included sensors targeting differential state
variables. ForceSensor targets through-variables (not in state). Added a
compiled sensor evaluation function (`_compiled_sensor_fn`) that computes
through-variable sensor values from `(y, input_vals, params)` using the
same alias resolution and through-equation evaluation as the vector field.
Called post-ODE-step in `__call__`.

### Issue 4 (Major): AcausalParams API mismatch

Changed `AcausalParams.values` from flat array + `_names` to
`dict[str, Array]`. Vector field reads params by key name directly.
Dict of Arrays is a valid PyTree, so `jax.grad` flows through.

### Issue 5 (Minor): Missing spec tests

Added: analytical damped MSD solution comparison, vmap over spring
stiffness parameters, ForceSensor validation (spring force = -k*x),
gear ratio behavior test, and multi-PrescribedMotion input routing test.

## Rationale

The gear ratio fix required careful analysis of how acausal modeling
handles force balance across kinematic constraints. In Modelica-style
systems, gear elimination creates an across-variable alias (kinematic
constraint) but must NOT merge the physical force-balance nodes. The
through-variable transmission must go through the gear's power-conservation
equation (`tau_a = ratio * tau_b`) rather than direct node summation.

The correct acceleration for the test system (inertia + gear + spring) is
`(1 + ratio) * k * (ratio - 1) * theta / J` because the inertia sees both
the direct spring torque at its node AND the gear-transmitted torque from
the other side of the spring.

## Files Changed

- `feedbax/acausal/assembly.py` -- Major: `eliminated_pre_gear`, gear torque
  equations, `input_specs`, sensor eval fn, params as dict, `_build_vals`
- `feedbax/acausal/base.py` -- Added `_input_specs` field to `StateLayout`
- `feedbax/acausal/system.py` -- `AcausalParams` dict API, sensor fn call,
  input routing via `input_specs`
- `tests/test_acausal.py` -- 5 new tests, params API update
- `tests/test_acausal_graph.py` -- 1 new test (multi-PrescribedMotion)
