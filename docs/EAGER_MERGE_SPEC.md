# Eager Graph Architecture Merge Spec

This document describes the state of the codebase, what needs to be merged, and how the UI should align with the eager graph architecture.

---

## Current Situation

### Branch State

- **`feature/web-ui`** (this branch): Built against the old staged model architecture. Still has `feedbax/_staged.py`, does not have `feedbax/graph.py`.
- **`eager-models`** (commit `1e5d8ec`): Implemented the eager graph cutover. Deleted `_staged.py`, added `graph.py` with Component/Graph execution.
- **`develop`** (commit `5411510`): Merged `eager-models`. Has the new architecture.

The web UI was built before the eager cutover was merged, so it's operating against deprecated architecture.

### What the Eager Cutover Did

From commit `1e5d8ec`:
- Replaced staged/iterator model flow with explicit Component/Graph execution
- Added `feedbax/graph.py` (~600 lines) with Component, Wire, Graph classes
- Unified state via StateIndex pattern (keyed immutable index)
- Updated tasks, training, interventions to work with new architecture
- Deleted `feedbax/_staged.py`

---

## Required Work

### 1. Merge `develop` into `feature/web-ui`

Rebase or merge to bring the eager graph architecture into this branch. Resolve conflicts, particularly in:
- `feedbax/bodies.py` (heavily changed)
- `feedbax/nn.py` (SimpleStagedNetwork → new pattern)
- `feedbax/task.py` (TaskComponent added)
- Any web layer code that references old APIs

### 2. Align UI Wire Semantics with State Slots

**The key insight**: In the eager architecture, all components operate on a shared state PyTree. Wires in the UI should represent **named slots** in this state — a publish/subscribe pattern.

#### How It Works

1. **Components declare state keys** they read from (inputs) and write to (outputs)
2. **Wires connect slots**: A wire from `ComponentA.output_x` → `ComponentB.input_y` means:
   - ComponentA writes to `state["output_x"]`
   - ComponentB reads from `state["output_x"]` (aliased as its `input_y`)
3. **Default keys**: Components have sensible defaults (e.g., Network reads from `state["target"]` and `state["feedback"]`)
4. **Override via wiring**: User can rewire to change which slot a component reads/writes

#### Example

```
Task (SimpleReaches)          Network                    Mechanics
├─ outputs:                   ├─ inputs:                 ├─ inputs:
│  ├─ targets ──────────────────▶ target                 │  └─ force ◀────────┐
│  └─ inputs                  │  └─ feedback ◀───────────│                    │
└─ (writes to state slots)    ├─ outputs:                ├─ outputs:          │
                              │  ├─ output ──────────────────────────────────┘
                              │  └─ hidden                │  └─ effector
                              └─ (reads/writes state)     └─ (reads/writes state)
```

Each wire establishes that two ports share the same state slot.

### 3. Intervention Access to Arbitrary State

Users may want to intervene on state that isn't explicitly wired. The UI should allow:
- Selecting from the full state tree (introspection)
- Creating intervention nodes that read/write arbitrary paths
- This connects to the "state browser" concept (GitHub issue #47)

### 4. Component Registry Updates

Update `feedbax/web/services/component_registry.py` to:
- Reflect the new Component types from `graph.py`
- Declare which state keys each component reads/writes
- Remove deprecated staged model references

### 5. Task Component Ports

Tasks like SimpleReaches should expose outputs:
- `inputs` — per-timestep model input (target trajectory)
- `targets` — loss targets (effector position)
- `inits` — initial state overrides

These become state slots that the model reads from.

---

## Wire Semantics Summary

| Concept | Meaning |
|---------|---------|
| Port | A named slot a component reads from (input) or writes to (output) |
| Wire | Connects two ports, making them share the same state slot |
| State slot | A keyed path in the shared state PyTree |
| Default binding | Components have default slot names; wiring can override |

**Key principle**: Wires are not just visualization — they configure which state slots components use.

---

## Technical Notes

### State as Keyed Index

The eager architecture uses a StateIndex pattern where state is accessed by keys:
```python
state["network.hidden"]  # Read hidden state
state = state.set("mechanics.force", new_force)  # Immutable update
```

Components declare their input/output keys, and the graph executor routes accordingly.

### Graph Execution

The Graph class in `feedbax/graph.py` handles:
1. Topological ordering of components
2. State initialization
3. Per-component execution with state routing
4. Intervention injection at specified points

The UI's wire list should map to the Graph's internal wiring.

---

## Files to Review

After merge, ensure these align with eager architecture:
- `feedbax/graph.py` — Core Component/Wire/Graph classes
- `feedbax/bodies.py` — SimpleFeedback now uses Graph internally
- `feedbax/nn.py` — Network as Component
- `feedbax/task.py` — TaskComponent and trial spec generation
- `feedbax/web/services/component_registry.py` — Component metadata
- `feedbax/web/models/graph.py` — GraphSpec/WireSpec (should match graph.py semantics)

---

## Success Criteria

1. `feature/web-ui` has all changes from `develop` (eager architecture)
2. UI components match the new Component types
3. Wires in the UI correspond to state slot bindings
4. Task components have appropriate output ports
5. Existing UI functionality (canvas, properties, training panel) still works

---

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
