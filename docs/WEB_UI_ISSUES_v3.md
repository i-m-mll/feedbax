# Web UI Issues & Improvements (Batch 3)

---

## 1. Node Box Manual Resizing

**Current behavior**: Nodes can be resized via drag handles; sizes persist in UI state.

**Desired behavior**: Users can resize node boxes by clicking and dragging borders.

**Implementation notes**:
- Drag handles on all four edges/corners
- Exclude port handle areas from resize interaction (ports should remain clickable for wiring)
- Minimum size constraints to ensure ports and title remain visible
- Persist custom sizes in UI state per node

**React Flow consideration**: React Flow has a `NodeResizer` component that may help, or we implement custom resize handles.

---

## 2. Collapse Behavior for Single-Port Nodes

**Current behavior**: Collapse is hidden/disabled for ≤1-port nodes; collapsed layout never exceeds expanded height.

**Desired behavior**:
- Collapsed height should never exceed expanded height
- If a node has only 1 input and 1 output (or 1 of one and 0 of the other), collapsing should either:
  - Be a no-op (already minimal), or
  - Hide port labels but keep dots visible, resulting in a shorter box

**Analysis**: The collapse toggle is meant to reduce visual clutter by hiding port labels. For single-port nodes, there's minimal clutter to begin with. Consider:
- Auto-hiding the collapse button for nodes with ≤1 input and ≤1 output
- Or ensuring collapsed state is always visually smaller than expanded

---

## 3. Port Dot Vertical Positioning

**Current behavior**: Port dots are row-aligned based on max(input, output) count.

**Desired behavior**: Proper row-based layout.

**Algorithm**:
1. Node box = header (title bar) + body (port area)
2. Number of rows = max(input_count, output_count)
3. Divide body height evenly into `n` rows
4. Place each port dot at the vertical midpoint of its corresponding row
5. If body has border-radius at bottom, may need small padding adjustment

**Example** (2 inputs, 3 outputs → 3 rows):
```
┌─────────────────────┐
│ Title               │  ← header
├─────────────────────┤
│ ●─in1      out1─●   │  ← row 1, dots at row midpoint
│ ●─in2      out2─●   │  ← row 2
│            out3─●   │  ← row 3 (no input, but row still exists)
└─────────────────────┘
```

---

## 4. Two-Shelf Collapse: Top Shelf Not Collapsible

**Current behavior**: Both shelves can be collapsed via header toggles; NAND constraint enforced.

**Desired behavior**: Both shelves should be independently collapsible, with NAND constraint (at least one must be open).

**What was tried**: Header has chevron buttons for toggling shelves, but top shelf toggle may not be working or may not be present.

**Action**: Create GitHub issue to track this. Then investigate:
- Is `toggleTop` wired correctly in Header?
- Does `layoutStore` enforce the NAND constraint properly?
- Is there a visual affordance (button) for collapsing the top shelf?

---

## 5. Two-Shelf Resize: 50% Minimum for Top Shelf

**Current behavior**: Bottom shelf can exceed 50% (min top height enforced), full range to collapsed.

**Desired behavior**: Full range of splits, from top-collapsed to bottom-collapsed, with reasonable minimums (e.g., 44px collapsed height for either).

**What was tried**: Previous commits adjusted `MIN_BOTTOM_HEIGHT`, `MAX_BOTTOM_HEIGHT`, etc.

**Analysis**: The constraint is likely in:
- `setBottomHeight` clamping logic in `layoutStore.ts`
- Or in the drag handler that calculates new height from mouse position
- Or CSS constraints on the shelf containers

**Action**: Add to GitHub issue. Investigate clamping logic and ensure bottom shelf can grow to (viewport height - top collapsed height).

---

## 6. Canvas Scaling Asymmetry on Shelf Resize

**Current behavior**: Canvas zoom scales proportionally both directions during resize.

**Desired behavior**: Graph maintains relative size in both directions. When canvas area changes, graph should scale proportionally to fill similar visual proportion.

**Implementation approach**:
- Track canvas dimensions before and after resize
- Calculate scale factor = new_dimension / old_dimension
- Apply inverse zoom adjustment to maintain apparent size
- Or: use `fitView` with consistent padding after resize

**Dependency**: Fixing issues #4 and #5 first, since full range of collapse/resize is needed.

---

## 7. Composite Loss Tree: Still Placeholder

**Current behavior**: Training panel renders a read-only loss tree structure with weights.

**Desired behavior**:
- Display actual `TermTree` structure from Python
- Editable: add/remove terms, adjust weights, configure parameters
- Bind loss terms to graph outputs (probe nodes or port selectors)

**Questions to resolve**:
1. Does the current Python `TermTree` / loss API need changes to support UI-driven construction?
2. How are loss targets specified? (See batch 2 discussion: probe nodes vs where-functions)
3. Should loss configuration live in the Training tab of the bottom shelf (per two-shelf design)?

**Implementation phases**:
1. Read-only display of loss tree structure
2. Weight editing
3. Add/remove terms
4. Parameter configuration per term
5. Target binding UI

---

## 8. "Feedback Channel" → "Channel"

**Current behavior**: Component renamed to "Channel".

**Problem**: Channels provide delay and noise, and can be used anywhere—not just in feedback loops. Naming it "Feedback Channel" incorrectly implies it's only for feedback.

**Example**: A channel between network output and mechanics input would add delay/noise to the motor command. This is not feedback; it's feedforward delay.

**Desired behavior**: Rename to simply "Channel" (or "Delay Channel" / "Signal Channel" if more specificity is wanted).

**Action**: Update `component_registry.py` and any UI references.

---

## 9. "Simple Staged Network" Naming & Dynamic Subgraph Ports

**Current behavior**: Component named "Network". Subgraph ports are derived from internal bindings/unwired ports on exit.

**Note**: Subgraph structure is still stored in UI state only (not serialized into the root graph yet).

**Problems**:
1. "Staged" is anachronistic—we no longer use staged models
2. Hardcoded input/output names (target, feedback, output, hidden) assume a specific use case
3. Not flexible for different network configurations

**Desired behavior**: Subgraph ports should be *dynamic*, derived from unconnected internal ports.

**Concept**:
- A subgraph (composite node) contains an internal graph
- Any internal port that is *not* wired to another internal node becomes an *external port* of the subgraph
- The external port inherits its name/label from the internal port
- Users can rename ports if desired

**Example**:
```
Inside "Network" subgraph:
  - Encoder node: input port "x" (unconnected)
  - Decoder node: output port "y" (unconnected)
  - Encoder.output → Hidden.input (connected)
  - Hidden.output → Decoder.input (connected)

Result: "Network" subgraph has:
  - Input port "x" (from Encoder.x)
  - Output port "y" (from Decoder.y)
```

**Benefits**:
- Fully flexible—any internal graph structure
- No hardcoded assumptions about input/output names
- Port labels flow naturally from internal structure
- Users can relabel at the subgraph level if desired

**Naming**: Consider renaming "SimpleStagedNetwork" to something like:
- "Network" (generic)
- "RecurrentNetwork" (if it specifically uses RNN cells)
- "NeuralController" (if it's specifically for control)
- Or just let users name their own subgraphs

**Implementation notes**:
- Subgraph port derivation happens when internal graph changes
- `input_bindings` and `output_bindings` in `GraphSpec` already support this conceptually
- UI needs to reflect dynamic port lists on subgraph nodes
- Python `Graph` class may need utility methods to compute "unbound ports"

---

## 10. Barnacles: Probes and Interventions

**Problem**: Probes and interventions need access to the full state at a point in execution, but wires only represent single state slots. There's no way to observe or modify state paths that aren't exposed as ports.

**Desired behavior**: Barnacles are small UI elements that attach to **nodes** (not wires) and can access the full state at that execution point.

**Two types**:
- **Probes**: Observe state (read-only) for loss targets, plotting, logging
- **Interventions**: Modify state for perturbations (e.g., CurlField), noise injection, clamps

**Key properties**:
- Attach to nodes, not wires (since wires only show one state slot)
- Specify timing: "at input" (before component runs) or "at output" (after)
- Parameterized: select which state path(s) to observe/modify
- Visually distinct: appear as small attachments on node sides (like barnacles on a hull)

**Implementation notes**:
- Barnacle data stored in graph spec per node
- State path selector UI (introspect available paths at that point)
- Distinct visual styling from regular ports/wires
- CurlField and similar interventions become parameterized barnacles, not hardcoded components

**See also**: [Wiki: Barnacles](https://github.com/i-m-mll/feedbax/wiki/Barnacles-(Probes-and-Interventions))

---

## 11. User-Exposed Ports

**Problem**: Components have default ports, but users may want to wire state paths that aren't exposed by default. Currently no way to "break off" an arbitrary piece of state and route it elsewhere.

**Desired behavior**: Users can promote arbitrary state paths to visible, wire-able ports on a node.

**Two kinds of ports**:
- **Defined ports**: From component type definition, not removable, normal styling
- **User-added ports**: From user selection, removable, visually distinct (e.g., dashed outline, muted color)

**Workflow**:
1. Right-click node → "Expose state path..."
2. State browser shows available paths at that execution point
3. User selects path (e.g., `state.foo.bar.baz`)
4. New port appears on node, can be wired
5. Persists in graph spec (not component definition)
6. Removable via sidebar or context menu

**Implementation notes**:
- Node box grows to accommodate new ports
- User-added ports stored separately in graph spec
- State introspection API needed to show available paths
- Visual distinction so users know which ports are built-in vs added

**Relation to barnacles**: Both access state beyond default ports. Barnacles work in-place (observe/modify); user-exposed ports enable wiring to other components.

**See also**: [Wiki: User-Exposed Ports](https://github.com/i-m-mll/feedbax/wiki/Barnacles-(Probes-and-Interventions)#user-exposed-ports)

---

## Summary of Actions

| Issue | Type | Action |
|-------|------|--------|
| Node resizing | Feature | Implement drag-to-resize with NodeResizer or custom handles |
| Single-port collapse | Bug/UX | Fix collapsed height logic; consider hiding collapse for minimal nodes |
| Port dot positioning | Bug | Implement row-based vertical layout |
| Top shelf collapse | Bug | **Create GitHub issue**, investigate toggleTop wiring |
| 50% resize limit | Bug | **Add to GitHub issue**, fix clamping logic |
| Canvas scaling asymmetry | Bug | Fix bidirectional scaling after shelf issues resolved |
| Loss tree placeholder | Feature | Implement tree display and editing (phased) |
| "Feedback Channel" name | Naming | Rename to "Channel" |
| Dynamic subgraph ports | Architecture | Implement unbound-port derivation for subgraphs |
| Barnacles | Architecture | Implement probe/intervention attachments on nodes with full state access |
| User-exposed ports | Feature | Allow users to promote state paths to wire-able ports |
