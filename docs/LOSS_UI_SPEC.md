# Loss Function UI Specification

This document specifies the design for interactive loss function configuration in the Feedbax web UI.

---

## Overview

The loss function UI enables users to:
1. **Target state variables** for loss computation via probe nodes
2. **View and edit** the hierarchical loss tree structure
3. **Configure** time aggregation, norm functions, and weights
4. **Bind** loss terms to probed signals in the model graph

---

## Core Concepts

### Probes as Taps

A **Probe** is a graph component that "taps" a signal for downstream use. Probes are generic—they don't encode behavior themselves. Instead, consumers (loss terms, plots, loggers) reference probes by name.

**Benefits:**
- Orthogonal: probe is just an output handle
- Reusable: one probe can feed multiple consumers (loss, plot, validation)
- Avoids component proliferation (no LossProbe, PlotProbe, etc.)

**Probe component:**
```typescript
interface ProbeNode {
  id: string;
  name: string;           // User-visible label, e.g., "effector_pos"
  input_port: string;     // Single input that receives the tapped signal
  // No output ports—probes are sinks in the graph
}
```

### Selectors

Selectors identify which part of the state a loss term operates on. Two forms are supported:

1. **Probe reference**: `"probe:effector_pos"` — references a probe node by name
2. **String path**: `"mechanics.effector.pos"` — direct state path (for advanced use)

Both are serialized as strings. The Python backend reconstructs `where` callables via `attr_str_tree_to_where_func`.

---

## State Selection Mechanism

### Primary: Probe Nodes

Users drop a **Probe** component onto the canvas and wire it to any output port. The probe captures that signal for use in loss terms, plots, etc.

```
┌─────────────┐         ┌─────────────┐
│  Mechanics  │         │   Probe     │
│             │──pos───▶│ "effector"  │
│             │         └─────────────┘
└─────────────┘
```

### Shortcut: Port Context Menu

Right-click an output port → **"Add probe here"**:
1. Creates a new Probe node near the port
2. Automatically wires the port to the probe's input
3. Names the probe after the port (e.g., `mechanics_pos`)

This reduces the two-step process (create probe, wire it) to a single action.

### Advanced: State Browser Panel

See GitHub issue #47. A hierarchical state tree browser for power users who need to target internal state not exposed via ports. Lower priority.

---

## Loss Tree Structure

### Data Model

The loss tree mirrors Python's `TermTree` / `CompositeLoss`:

```typescript
interface LossTermSpec {
  type: string;                           // "TargetStateLoss", "Composite", etc.
  label: string;                          // Display name
  weight: number;                         // Weighting factor
  selector?: string;                      // Probe reference or state path
  norm?: string;                          // "squared_l2" | "l2" | "l1" | "huber"
  time_agg?: TimeAggregationSpec;         // Time aggregation settings
  children?: Record<string, LossTermSpec>; // For composite nodes
}

interface TimeAggregationSpec {
  mode: "all" | "final" | "range" | "segment" | "custom";
  // For "range":
  start?: number;
  end?: number;
  // For "segment":
  segment_name?: string;  // References TaskSpec.timeline
  // For "custom":
  time_idxs?: number[];
  // Discount:
  discount?: "none" | "power" | "linear";
  discount_exp?: number;  // For power discount
}
```

### Example

```json
{
  "type": "Composite",
  "label": "reach_loss",
  "weight": 1.0,
  "children": {
    "position": {
      "type": "TargetStateLoss",
      "label": "Effector Position",
      "weight": 1.0,
      "selector": "probe:effector_pos",
      "norm": "squared_l2",
      "time_agg": { "mode": "all", "discount": "power", "discount_exp": 6 }
    },
    "final_velocity": {
      "type": "TargetStateLoss",
      "label": "Final Velocity",
      "weight": 0.5,
      "selector": "probe:effector_vel",
      "norm": "squared_l2",
      "time_agg": { "mode": "final" }
    },
    "regularization": {
      "type": "TargetStateLoss",
      "label": "Network Activity",
      "weight": 0.01,
      "selector": "probe:network_hidden",
      "norm": "squared_l2",
      "time_agg": { "mode": "all" }
    }
  }
}
```

---

## UI Components

### Loss Tree Panel

Located in the **Training** tab of the bottom shelf.

#### Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Loss Function                                          [+]  │
├─────────────────────────────────────────────────────────────┤
│ L = 1.0·L_pos + 0.5·L_vel + 0.01·L_reg                     │  ← Equation summary
├─────────────────────────────────────────────────────────────┤
│ ▼ reach_loss (Composite)                           ×1.0    │
│   ├─ ▶ Effector Position                           ×1.0    │
│   │     Probe: effector_pos │ Norm: Squared L2            │
│   │     Time: All steps, power discount (exp=6)           │
│   ├─ ▶ Final Velocity                              ×0.5    │
│   │     Probe: effector_vel │ Norm: Squared L2            │
│   │     Time: Final step only                             │
│   └─ ▶ Network Activity                            ×0.01   │
│         Probe: network_hidden │ Norm: Squared L2          │
│         Time: All steps                                   │
└─────────────────────────────────────────────────────────────┘
```

#### Interactions

| Action | Result |
|--------|--------|
| Click term row | Select and show details in properties panel |
| Click weight | Inline edit (number input) |
| Click [+] button | Add new loss term (dropdown: term type) |
| Drag term | Reorder within parent |
| Right-click term | Context menu: Delete, Duplicate, Wrap in Composite |
| Expand/collapse (▶/▼) | Show/hide children and details |

### Equation Summary

A compact inline equation at the top of the loss panel:

```
L = 1.0·L_pos + 0.5·L_vel + 0.01·L_reg
```

- Derived from the tree structure and weights
- Non-editable (just a summary)
- Clicking a term in the equation scrolls to and highlights it in the tree

### Term Detail Panel

When a term is selected, the properties panel (right sidebar) shows editable details:

```
┌─────────────────────────────────┐
│ Effector Position               │
├─────────────────────────────────┤
│ Type: TargetStateLoss           │
│                                 │
│ Target                          │
│ ┌─────────────────────────────┐ │
│ │ Probe: [effector_pos    ▼] │ │  ← Dropdown of available probes
│ └─────────────────────────────┘ │
│                                 │
│ Norm Function                   │
│ ┌─────────────────────────────┐ │
│ │ [Squared L2            ▼]  │ │  ← Dropdown
│ └─────────────────────────────┘ │
│                                 │
│ Time Aggregation                │
│ ┌─────────────────────────────┐ │
│ │ Mode: [All steps       ▼]  │ │
│ │                             │ │
│ │ Discount: [Power       ▼]  │ │
│ │ Exponent: [6           ]   │ │
│ └─────────────────────────────┘ │
│                                 │
│ Weight: [1.0            ]       │
└─────────────────────────────────┘
```

---

## Time Aggregation

### Presets (Primary UI)

| Mode | Description | `time_idxs` equivalent |
|------|-------------|------------------------|
| All steps | Evaluate at every timestep | `None` (no mask) |
| Final step | Evaluate only at the last timestep | `[-1]` |
| Range | Evaluate from step `start` to `end` | `range(start, end)` |
| Segment | Use a named segment from `TaskSpec.timeline` | Callable lookup |

### Discount Options

| Option | Description |
|--------|-------------|
| None | Uniform weighting over time |
| Power | `(t/T)^exp` — emphasizes later timesteps |
| Linear | `t/T` — linear ramp |

### Advanced (Custom)

Behind an "Advanced" toggle:
- Raw `time_idxs` array input
- Custom mask editor (lower priority)

---

## Norm Functions

### Registry

The UI exposes a fixed set of named norms; custom norms require Python.

| Name | Formula | Python |
|------|---------|--------|
| Squared L2 (default) | `sum(x²)` | `lambda x: jnp.sum(x**2, axis=-1)` |
| L2 | `sqrt(sum(x²))` | `jnp.linalg.norm(x, axis=-1)` |
| L1 | `sum(|x|)` | `lambda x: jnp.sum(jnp.abs(x), axis=-1)` |
| Huber | Huber loss with δ=1 | `jax.nn.huber_loss` |

---

## Serialization

### JSON Schema

Loss configuration is serialized as part of `TrainingSpec`:

```typescript
interface TrainingSpec {
  optimizer: OptimizerSpec;
  loss: LossTermSpec;       // Root of the loss tree
  n_batches: number;
  batch_size: number;
  // ...
}
```

### Probe References

Probes are referenced by name with a `probe:` prefix:
- `"probe:effector_pos"` → looks up probe node named "effector_pos"

### String Selectors

For advanced use (state browser), direct paths are supported:
- `"mechanics.effector.pos"` → converted to `where` callable via `attr_str_tree_to_where_func`

---

## Python Backend Integration

### Reconstructing Loss Objects

```python
def loss_spec_to_loss(spec: dict, probes: dict[str, str]) -> AbstractLoss:
    """Convert UI LossTermSpec to Python AbstractLoss."""
    if spec["type"] == "Composite":
        children = {
            name: loss_spec_to_loss(child, probes)
            for name, child in spec["children"].items()
        }
        weights = {name: child["weight"] for name, child in spec["children"].items()}
        return CompositeLoss(children, weights, label=spec["label"])

    elif spec["type"] == "TargetStateLoss":
        selector = spec["selector"]
        if selector.startswith("probe:"):
            probe_name = selector[6:]
            path = probes[probe_name]  # Resolve probe to state path
        else:
            path = selector

        where_fn = attr_str_tree_to_where_func(path)
        norm_fn = NORM_REGISTRY[spec.get("norm", "squared_l2")]
        time_spec = build_target_spec(spec.get("time_agg", {}))

        return TargetStateLoss(
            label=spec["label"],
            where=where_fn,
            norm=norm_fn,
            spec=time_spec,
        )
```

### Norm Registry

```python
NORM_REGISTRY = {
    "squared_l2": lambda x: jnp.sum(x**2, axis=-1),
    "l2": lambda x: jnp.linalg.norm(x, axis=-1),
    "l1": lambda x: jnp.sum(jnp.abs(x), axis=-1),
    "huber": lambda x: jnp.sum(jax.nn.huber_loss(x, jnp.zeros_like(x)), axis=-1),
}
```

---

## Implementation Phases

### Phase 1: Read-Only Display
- Render existing loss tree structure from Python
- Display weights and term types
- Equation summary line

### Phase 2: Weight Editing
- Inline weight editing in tree view
- Sync changes to Python via API

### Phase 3: Add/Remove Terms
- Add new loss terms via [+] button
- Delete terms via context menu
- Wrap terms in Composite

### Phase 4: Full Configuration
- Probe selection dropdown
- Time aggregation presets
- Norm function dropdown

### Phase 5: Probe Integration
- Port context menu shortcut
- Visual indicators (probe nodes highlight when loss term selected)
- Bidirectional navigation (click loss term → highlight probe on canvas)

---

## Open Questions

1. **Target values**: Where do target values (e.g., goal position) come from?
   - Currently from `TaskSpec.targets`
   - Should the UI allow specifying constant targets per loss term?

2. **Multiple terms per probe**: Can multiple loss terms target the same probe?
   - Yes, supported via the `#` suffix convention in `WhereDict`

3. **Validation feedback**: Should the UI validate that selected probes are wired?
   - Probably yes—show warning if probe has no input connection

---

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
