# Web UI Issues & Improvements

Tracking unresolved issues and planned features.

---

## Edge Drawing UX

**Current behavior**: Global toggle between curved and elbow edge styles (top-right button).

**Desired behavior**: Per-edge control with drawing tool semantics:
- Default drawing mode creates curved (bezier) edges
- Holding Shift (or similar modifier) creates straight/elbow edges
- Clicking on canvas mid-draw introduces an elbow/waypoint
- Clicking on a port initiates or terminates the wire
- Each edge retains its own style (not global toggle)

**Notes**: Need to investigate React Flow capabilities for custom edge routing with user-defined waypoints, per-edge metadata, and modifier key detection during drawing.

---

## Node Title Editing Causes Resize

**Current behavior**: Component boxes change size when editing their titles, which is jarring.

**Desired behavior**: Node size should remain stable during title editing. Options:
- Fixed minimum width that accommodates reasonable title lengths
- Title truncation with ellipsis when exceeding box width
- Tooltip showing full title on hover if truncated
- Possibly user-controllable: allow manual node resize, with title truncating to fit

---

## Two-Shelf Layout

**Current behavior**: Single-shelf layout with Training/Inspector tabs in the right sidebar.

**Desired behavior**: Two-shelf layout spanning full page width:

- **Top shelf**: Model canvas + component library (left) + properties (right)
  - Sidebars collapse with the shelf
- **Bottom shelf**: Tabs for Validation | Training | Analysis
  - Draggable height / drawer style
- **Constraint**: At least one shelf open at a time (NAND on collapsed state)
- **Default**: Top shelf maximized, bottom shelf collapsed/minimized

**Open questions**:
- Where should shelf toggle affordances live? (drag handle, collapse buttons, etc.)
- Default heights when both open?
- Should bottom shelf tabs have their own internal sidebars/panels?

---

## CompositeLoss / Loss Tree Editor

**Current behavior**: "Composite loss tree" is a placeholder string in Training panel.

**Desired behavior**:
- Show actual loss term tree structure (matching `TermTree` in Python)
- Editable: drag/drop terms, change weights, configure parameters
- Loss targets bound to graph outputs

**Design decision — how to bind loss targets**:

### Option A: Probe/Metric Nodes (recommended)
Add explicit nodes in the graph whose outputs are used as loss targets.
- **Pros**: Visual, inspectable, consistent with graph-first UI. Loss targets reusable for logging, analysis, validation. Easier to debug.
- **Cons**: Requires new component type (probe/metric). Slightly more steps (create probe, then bind loss).

### Option B: State/Port Selectors (where-function style)
Loss terms refer directly to a state path or port selector string.
- **Pros**: Faster to specify in text; no extra nodes. Closer to current code patterns.
- **Cons**: Less discoverable in UI; harder to validate visually. Brittle if state paths change.

**Recommendation**: Option A scales better for UI + analysis workflows.

---

## Port Type System

**Current behavior**: No type checking on port connections.

**Desired behavior**: Validate port compatibility at connection time. Optionally color-code port types.

**Design decision — type system complexity**:

### Simple Type System (lightweight)
A few coarse labels: `float`, `vector`, `state`, `key`, `control`.
- **Pros**: Fast to implement, good early guardrails.
- **Cons**: Low specificity; won't catch shape/size mismatches.

### Rich Type System (dtype + shape/rank)
Dtype plus optional rank/shape constraints: `float[2]`, `state[6]`.
- **Pros**: Prevents subtle wiring errors; better compile-time validation.
- **Cons**: More upfront schema work per component; more UI complexity.

**Recommendation**: Start with simple system, but design schema to expand to shape constraints later. Define per-port dtype (optional shape) in `ComponentDefinition`, enforce in `isValidConnection`.

---

## Component Catalog Coverage

**Current behavior**: Component registry only includes minimal subset:
- SimpleStagedNetwork
- Mechanics
- FeedbackChannel
- Constant
- Gain

**Missing categories**:
- **Tasks**: ReachingTask, TrackingTask, etc. — not implemented in UI
- **Interventions**: CurlField, ForceField, Clamp, Perturbation — not wired
- **Signals**: Ramp, Sine, Pulse, etc. — not wired

**Work needed**: Expand `component_registry.py` to register all built-in Feedbax components with proper param schemas.

---

## (Add more issues below as they arise)
