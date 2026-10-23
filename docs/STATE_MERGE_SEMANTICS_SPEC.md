# State Merge Semantics Spec

This spec clarifies the relationship between state-flow wires and port wires, and defines
what happens when multiple state sources converge on a single node.

---

## The Two Wire Types

### State-Flow Wires (Thick)

The thick wire represents the **entire state PyTree** flowing from one component to the next.
When you see:

```
┌───────┐         ┌───────┐
│   A   │━━━━━━━━▶│   B   │
└───────┘         └───────┘
```

This means: "A's complete output state is the default source for B's input state."

### Port Wires (Thin)

The thin wires show **specific key routing**. When nodes are expanded:

```
┌───────────┐                 ┌───────────┐
│     A     │                 │     B     │
├───────────┤                 ├───────────┤
│    pos ───●────────────────▶●── pos     │
│    vel ───●────────────────▶●── vel     │
│           ◉━━━━━━━━━━━━━━━━━◉           │
└───────────┘                 └───────────┘
```

Each thin wire says: "This specific key comes from this specific source port."

---

## The Multi-Source Scenario

### What Happens When You Mix Sources

Suppose A's state-flow wire goes to B (the default), but you want B's `force` input
to come from C instead of A:

```
EXPANDED VIEW:
┌───────────┐                 ┌───────────┐
│     A     │                 │     B     │
├───────────┤                 ├───────────┤
│    pos ───●────────────────▶●── pos     │
│    vel ───●────────────────▶●── vel     │
│  force ───●        ┌───────▶●── force   │
│           ◉━━━━━━━━│━━━━━━━━◉           │
└───────────┘        │        └───────────┘
                     │
┌───────────┐        │
│     C     │        │
├───────────┤        │
│  force ───●────────┘
│           ◉
└───────────┘
```

**The question**: What does this mean for state-flow wires?

### The Collapsed View Reveals Merge

When collapsed, you see **two thick wires** converging on B:

```
┌───────┐
│   A   │━━━━━━━━┓
└───────┘        ┃
                 ▼
┌───────┐    ┌───────┐
│   C   │━━━▶│   B   │
└───────┘    └───────┘
```

This visual immediately communicates: "B receives state from multiple sources."

---

## State Merge Semantics

### How Merge Works

When multiple state-flow wires converge on a node, the system performs an **implicit
state merge** before execution. The merge is defined by **port-level wiring**:

1. Each input port can have exactly **one** source wire
2. The source wire determines which node provides that specific key
3. The merged input state is the **union** of all port values from their respective sources

```python
def implicit_state_merge(port_sources: dict[str, tuple[str, str]]) -> State:
    """
    port_sources maps: input_port_name -> (source_node, source_port)

    Result: a state PyTree containing values from the appropriate sources.
    """
    merged = {}
    for input_port, (source_node, source_port) in port_sources.items():
        merged[input_port] = get_port_value(source_node, source_port)
    return merged
```

### Key Property: Port Exclusivity Defines the Partition

Because each input port can only have one wire connected to it, the wiring itself
naturally defines **which parts come from where**. There's no ambiguity:

- If B.pos is wired from A.pos → that value comes from A
- If B.force is wired from C.force → that value comes from C
- The merge is fully determined by the port-level wiring

---

## Distinction From Array Merge

### The Existing Merge Node

The `Merge` component in the component registry performs **array concatenation**:

```python
# Merge node: concatenates arrays along an axis
merged_array = jnp.concatenate([input_a, input_b], axis=-1)
```

Use case: Combining network feedback + target into a single input tensor.

### State Merge (This Concept)

State merge is about **selecting different parts of a structured state** from
different sources:

```python
# State merge: pick which keys come from which source
merged_state = {
    "pos": state_from_A["pos"],
    "vel": state_from_A["vel"],
    "force": state_from_C["force"],  # Override with C's value
}
```

Use case: A provides most state, but C overrides a specific intervention.

### Naming Clarity

| Concept | What it does | Node type |
|---------|--------------|-----------|
| **Array Merge** | Concatenate arrays along axis | `Merge` component |
| **State Merge** | Select keys from multiple sources | Implicit (via wiring) |

The key distinction:
- **Array Merge** is explicit (you place a Merge node)
- **State Merge** is implicit (emerges from multi-source wiring)

---

## Visual Design

### Collapsed View: Multiple Thick Wires

When a node has inputs from multiple sources, show multiple state-flow wires:

```
┌───────┐
│   A   │━━━━━━━━┓
└───────┘        ┣━━━▶┌───────┐
┌───────┐        ┃    │   B   │
│   C   │━━━━━━━━┛    └───────┘
└───────┘
```

The converging wires visually indicate "merge point."

### Expanded View: Port-Level Detail

When expanded, the thin wires show exactly which keys come from where:

```
┌───────────┐                    ┌───────────┐
│     A     │                    │     B     │
├───────────┤                    ├───────────┤
│    pos ───●───────────────────▶●── pos     │
│    vel ───●───────────────────▶●── vel     │
│  force ───●                    │           │
│           ◉━━━━━━━━━━━━━━━━━━━━◉           │
└───────────┘                    └───────────┘
                     ┌──────────▶●── force
┌───────────┐        │           └───────────┘
│     C     │        │
├───────────┤        │
│  force ───●────────┘
│           ◉
└───────────┘
```

Each thin wire tells the full story of where each input comes from.

### State-Flow Wire Rendering Rules

1. **Primary source**: The node with the most port connections gets the "main" thick wire
2. **Secondary sources**: Other contributing nodes also get thick wires, but potentially
   thinner or styled differently (dashed, lighter)
3. **Merge indicator**: Where wires converge, a small merge glyph could appear (optional)

---

## Interaction Design: Dragging State-Flow Wires

### The Fundamental Model

**Port wiring is primary; state-flow wires are derived.**

State-flow wires are a visual summary of port-level connections. They show "which nodes
contribute inputs to which targets" but are computed from the underlying port wires.

However, for convenience, users can **drag state-flow wires as a shortcut** for bulk
port wiring operations.

### The Ambiguity Problem

When B already has all its input ports wired from A, and you drag C's state-flow output
to B's state-flow input, the system faces ambiguity:

- You're saying "I want C to contribute to B"
- But B's ports are already fully wired from A
- Which ports should switch to C? All of them? Some of them?

### Rejected Alternatives

| Approach | Problem |
|----------|---------|
| **Displacement** (C replaces A entirely) | Dangerous—one drag destroys all A→B wiring |
| **Refusal** (disallow the action) | Frustrating—user clearly wants to do something |
| **Auto-match by name** (wire matching port names) | Surprising—user loses control |

### The Solution: Port Mapping UI

When you drag a state-flow wire from C to B (where B already has connections), a
**port mapping dialog** appears:

```
┌─────────────────────────────────────────────────────┐
│  Connect C → B                                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Select which of B's inputs should come from C:    │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  B's Input    Current Source    Wire from C │   │
│  ├─────────────────────────────────────────────┤   │
│  │  pos          A.pos             ☐           │   │
│  │  vel          A.vel             ☐           │   │
│  │  force        A.force           ☑ → C.force │   │
│  │  target       A.target          ☐           │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Selecting a port will disconnect it from its      │
│  current source and wire it from C instead.        │
│                                                     │
│                        [Cancel]       [Connect]    │
└─────────────────────────────────────────────────────┘
```

### Interaction Flow

1. **User drags** from C's state-flow output handle to B's state-flow input handle
2. **System detects** that B already has port connections
3. **Port mapping UI appears** showing:
   - All of B's input ports
   - Current source for each port (e.g., "A.pos")
   - Checkbox to rewire from C
   - Auto-matched C output port (if names match) or dropdown to select
4. **User selects** which ports to rewire
5. **On confirm**:
   - Selected ports get new wires from C
   - Unselected ports keep their existing wires from A
   - Old wires for selected ports are removed

### Special Cases

#### Case 1: B Has No Existing Connections

If B's ports are not yet wired, dragging C→B could either:

- **Auto-wire matching names**: Wire C.foo → B.foo for all matching port names
- **Show port mapping UI**: Let user explicitly choose mappings

Recommendation: Auto-wire matching names, but show a toast/notification of what was
connected, with an "Undo" option.

#### Case 2: C Has Fewer Outputs Than B Has Inputs

The port mapping UI only shows ports where C has a compatible output. Ports that C
cannot provide remain wired to their current source (or unwired).

#### Case 3: Complete Replacement

If user selects ALL ports in the mapping UI, the result is equivalent to displacement:
A is fully disconnected, C is now the sole source. The UI should show this clearly:

```
⚠ This will disconnect all wires from A to B.
```

### Port-Level Wiring Remains Primary

Users can always:
1. **Expand nodes** to see port-level detail
2. **Drag thin wires** between specific ports for precise control
3. **Delete individual port wires** without affecting others

The state-flow drag is a **convenience shortcut**, not a replacement for port-level
editing. Power users may prefer to always work at the port level.

### Visual Feedback During Drag

While dragging a state-flow wire:

1. **Valid targets highlight**: Nodes that could receive the connection glow/highlight
2. **Potential conflicts shown**: If target has existing connections, show a subtle
   indicator (e.g., small badge showing "3 ports connected")
3. **Preview on hover**: When hovering over a target, briefly show which ports would
   be affected (optional, may be too noisy)

### Keyboard Modifiers (Optional Enhancement)

For power users, modifier keys could bypass the dialog:

| Modifier | Behavior |
|----------|----------|
| **No modifier** | Show port mapping UI (default) |
| **Shift+drag** | Replace all connections (displacement) |
| **Alt+drag** | Add connections for unconnected ports only |

These are optional enhancements; the dialog-based flow is the safe default.

---

## Implementation Notes

### Determining State-Flow Edges

When computing state-flow edges for the collapsed view:

```typescript
function computeStateFlowEdges(nodes: Node[], portWires: Wire[]): StateFlowEdge[] {
  // Group port wires by target node
  const sourcesByTarget = new Map<string, Set<string>>();

  for (const wire of portWires) {
    const sources = sourcesByTarget.get(wire.targetNode) || new Set();
    sources.add(wire.sourceNode);
    sourcesByTarget.set(wire.targetNode, sources);
  }

  // Create state-flow edge for each source->target pair
  const stateFlowEdges: StateFlowEdge[] = [];
  for (const [target, sources] of sourcesByTarget) {
    for (const source of sources) {
      stateFlowEdges.push({ source, target });
    }
  }

  return stateFlowEdges;
}
```

### Execution Order with Multi-Source

When B has inputs from A and C, execution order must ensure both A and C
complete before B:

```
Topological order: [A, C, B] or [C, A, B]
```

The graph executor already handles this via dependency analysis.

---

## User Mental Model

### The Intuition

Think of state-flow wires as "pipes" carrying the full state. When multiple pipes
converge:

1. **The valves (ports) determine the mix** - Each input port is a valve that selects
   which pipe's value to use for that specific key
2. **No conflicts possible** - Each valve can only connect to one pipe
3. **The result is a blend** - The output is a state composed of parts from each source

### When This Happens

Common scenarios:
- **Interventions**: Main model provides most state, but an intervention node overrides
  one specific value (e.g., force perturbation)
- **Multi-modal inputs**: Different sources provide different modalities to a consumer
- **Partial overrides**: Testing a component with modified inputs while keeping others

---

## Summary

| Concept | Description |
|---------|-------------|
| **State-flow wire** | Full state PyTree from source to target (derived from port wiring) |
| **Port wire** | Specific key routing between ports (primary/fundamental) |
| **Multi-source scenario** | Multiple state-flow wires converge on one node |
| **Implicit state merge** | Port wiring determines which keys come from which source |
| **Port exclusivity** | Each input port → one source, defining the partition |
| **Visual indicator** | Collapsed view shows multiple thick wires converging |
| **State-flow drag** | Convenience shortcut that triggers port mapping UI |
| **Port mapping UI** | Dialog for selecting which ports to rewire when adding a source |

The elegance: **port-level wiring IS the merge specification**. No separate merge
configuration needed—the thin wires tell the complete story.

State-flow wires are a derived visualization and bulk-wiring convenience, but port
wires remain the source of truth.

---

*Co-authored by Claude Opus 4.5*
