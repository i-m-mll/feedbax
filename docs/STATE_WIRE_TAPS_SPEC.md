# State Wire Taps Spec

This spec replaces the "barnacles" and "user-exposed ports" concepts with a simpler,
more intuitive model: **taps on state-flow wires**.

---

## Concept

The state-flow wire (thick wire between nodes) represents the entire state PyTree
passing from component to component. A **tap** is a small inline element on this wire
that extracts or modifies state paths.

```
                         ┌─────────┐
                         │  Probe  │
                         └────┬────┘
                              │ (thin wire)
     ┌─────────┐         ┌───●───┐         ┌─────────┐
     │ Network │━━━━━━━━━┤  tap  ├━━━━━━━━━│Mechanics│
     └─────────┘         └───────┘         └─────────┘
                    (tap on state wire)
```

---

## Why This Is Better

### vs. Barnacles (attaching to nodes)

| Barnacles | Wire Taps |
|-----------|-----------|
| Attach to nodes, specify "at input" or "at output" | Naturally positioned on wire between nodes |
| Conceptually awkward (barnacles on a hull?) | Intuitive (tap a pipe to extract flow) |
| Node UI gets cluttered | Nodes stay clean, taps live on wires |

### vs. User-Exposed Ports (adding ports to nodes)

| User-Exposed Ports | Wire Taps |
|--------------------|-----------|
| Adds extra ports to node box | Doesn't modify node appearance |
| Confusing: which ports are "real" vs "added"? | Clear: taps are separate elements |
| Requires node to know about external observers | Taps are independent of nodes |

### Unified Model

Wire taps unify probing and intervention:
- **Probe tap**: Extracts state path(s), outputs on perpendicular wire
- **Intervention tap**: Modifies state path(s) inline, state continues flowing

---

## Visual Design

### Tap Element

A small circle (or rounded rectangle) sitting inline on the state-flow wire:

```
━━━━━━●━━━━━━   Simple tap (just a dot)

━━━━┫tap┣━━━━   Labeled tap (shows what it extracts)
       │
       ▼
   (output)
```

### Tap with Output

When a tap extracts data, a thin wire extends perpendicular to the state wire:

```
                    ┌──────────┐
                    │ LossNode │
                    └────┬─────┘
                         │
━━━━━━━━━━━━━━━━━━━●━━━━━━━━━━━━━━━━━━
              (tap: effector.pos)
```

### Tap Types

| Type | Symbol | Description |
|------|--------|-------------|
| **Probe** | `●` or `◯` | Read-only extraction |
| **Intervention** | `◆` or `⬥` | Inline modification |

---

## Tap Configuration

When you click a tap, the sidebar shows:

1. **State path selector** - Browse available paths at this point in execution
2. **Output port(s)** - Which extracted values to expose
3. **For interventions**: transformation parameters

Example for a probe tap:
```
┌─────────────────────────────┐
│ Tap: probe_1                │
├─────────────────────────────┤
│ Position: after Network     │
│                             │
│ Extract:                    │
│   ☑ mechanics.effector.pos  │
│   ☑ mechanics.effector.vel  │
│   ☐ network.hidden          │
│                             │
│ Output ports:               │
│   • pos (mechanics.effector.pos)
│   • vel (mechanics.effector.vel)
└─────────────────────────────┘
```

---

## Execution Semantics

### Probe Taps

1. State flows into tap
2. Tap extracts specified paths (read-only)
3. Extracted values available on output port(s)
4. Full state continues unchanged to next component

```python
def probe_tap(state, paths):
    extracted = {name: get_path(state, path) for name, path in paths.items()}
    return state, extracted  # state unchanged, extracted goes to output
```

### Intervention Taps

1. State flows into tap
2. Tap reads state, applies transformation
3. Modified state continues to next component

```python
def intervention_tap(state, path, transform_fn):
    value = get_path(state, path)
    new_value = transform_fn(value)
    return set_path(state, path, new_value)
```

---

## Interaction with Port Wires

Taps exist on state-flow wires, but their outputs connect via regular (thin) port wires:

```
┌─────────┐                           ┌─────────┐
│ Network │                           │ Loss    │
├─────────┤                           ├─────────┤
│  output─●─────────────────────────▶─●─pred    │
│         │                     ┌────▶●─target  │
│         ◉━━━━━●━━━━━━━━━━━━━━━│━━━━━◉         │
└─────────┘     │               │     └─────────┘
           ┌────┴────┐          │
           │ pos tap │──────────┘
           └─────────┘
```

Here:
- Thick wire (━) is state flow
- Thin wire from tap to Loss.target is a regular port connection
- Network.output → Loss.pred is also a regular port connection

---

## Graph Spec Representation

```typescript
interface Tap {
  id: string;
  type: 'probe' | 'intervention';
  position: {
    afterNode: string;  // tap is on wire leaving this node
  };
  paths: {
    [outputName: string]: string;  // output port name → state path
  };
  // For interventions:
  transform?: {
    type: string;
    params: Record<string, unknown>;
  };
}

interface GraphSpec {
  nodes: Record<string, NodeSpec>;
  wires: Wire[];
  taps: Tap[];  // NEW: taps on state-flow wires
}
```

---

## Replacing Previous Concepts

| Old Concept | New Equivalent |
|-------------|----------------|
| Barnacle (probe) | Probe tap on state wire |
| Barnacle (intervention) | Intervention tap on state wire |
| User-exposed port | Probe tap with output wired to consumer |
| Probe component | Can be replaced by probe tap (simpler) |

**Removed from design:**
- Barnacles attaching to nodes
- User-exposed ports on nodes
- "at input" / "at output" timing (position on wire is sufficient)

---

## Implementation Notes

### UI Components

1. **TapNode** - New React Flow node type for taps
   - Smaller than regular nodes
   - Sits inline on state-flow edge
   - Click to configure in sidebar

2. **State-flow edge with taps** - Edge renders taps as inline elements
   - Or: taps are nodes positioned on edge path

3. **Tap sidebar panel** - Configure paths and outputs

### Positioning

Taps position themselves on the state-flow wire:
- Horizontally: between source and target nodes
- Vertically: aligned with state-flow wire path
- Can be dragged along the wire

### State Path Browser

Need UI to browse available state paths at tap position:
- Introspect state structure from Python backend
- Show tree of available paths
- Filter/search functionality

---

## Summary

Wire taps are a cleaner mental model than barnacles or user-exposed ports:

1. **Visual clarity** - Taps live on wires, not cluttering nodes
2. **Intuitive metaphor** - Tap a pipe to extract/modify flow
3. **Unified concept** - Probes and interventions are both taps
4. **Position = timing** - Where tap sits determines when it executes
5. **Simpler implementation** - One concept instead of three

---

*Co-authored by Claude Opus 4.5*
