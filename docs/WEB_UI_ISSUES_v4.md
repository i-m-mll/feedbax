# Web UI Issues & Improvements (Batch 4)

This document specifies issues to be addressed. Items are ordered by priority.

---

## 1. Subgraph Navigation (HIGH PRIORITY)

**Current behavior**: Only nodes with `type === 'Network'` show the enter button and can be navigated into.

**Desired behavior**: ANY node that has an associated subgraph should be navigable.

**Root cause**: In `CustomNode.tsx` line 27:
```tsx
const isComposite = spec.type === 'Network';
```

**Fix required**:
1. Change the check to detect if a node has an associated subgraph, not just if it's type "Network"
2. A node is composite if:
   - `spec.type === 'Network'`, OR
   - `spec.type === 'Subgraph'`, OR
   - The current graph has a subgraph associated with this node's ID (check `graph.subgraphs?.[nodeId]`)

**Files to modify**:
- `web/src/components/canvas/CustomNode.tsx` - update `isComposite` logic
- `web/src/stores/graphStore.ts` - ensure `enterSubgraph` works for any node with a subgraph
- May need to pass graph context to CustomNode or derive from node data

**Test**:
1. Create a graph, wrap it in a parent (using + button)
2. The child should now be a Subgraph node
3. Should be able to click the enter button on that Subgraph node to navigate in

---

## 2. Collapse Behavior Redesign

**Current behavior**: When collapsed:
- Port labels disappear
- Port dots remain in their original positions (multiple dots per side)
- Box size doesn't change
- Shows "N in • M out" summary text

**Desired behavior**: When collapsed:
- Port labels disappear
- ALL port dots disappear
- Show a SINGLE unlabeled dot on left side (if any inputs exist)
- Show a SINGLE unlabeled dot on right side (if any outputs exist)
- Box height shrinks to minimal size (header + small padding for the single dots)

**Visual example**:
```
EXPANDED:                    COLLAPSED:
┌─────────────────┐          ┌─────────────────┐
│ Network         │          │ Network         │
├─────────────────┤          ├─────────────────┤
│ ●─input         │          │ ●           ●   │
│ ●─feedback  out─●│    →    └─────────────────┘
│             hid─●│
└─────────────────┘
```

**Implementation**:
1. When `collapsedEffective` is true:
   - Don't render individual port handles
   - Render at most ONE handle on left (if `inputCount > 0`) at vertical center
   - Render at most ONE handle on right (if `outputCount > 0`) at vertical center
   - These handles should still be functional for wiring (connect to first port)
   - Don't render port labels
   - Remove the "N in • M out" summary (the single dots make it clear)
2. Collapsed height should be approximately `HEADER_HEIGHT + COLLAPSED_BODY_HEIGHT` where body is ~20-24px

**Files to modify**:
- `web/src/components/canvas/CustomNode.tsx`

**Edge case**: For collapsed nodes, wires should connect to the single dot. When expanded, wires reconnect to their actual ports. This may require edge routing logic to handle gracefully, or we simply keep all wires connected to their real port IDs and visually render them going to the single dot position when collapsed.

---

## 3. Network as a Subgraph Type

**Current behavior**: "Network" is a special component type with hardcoded ports (`input`, `feedback`, `output`, `hidden`).

**Desired behavior**: Network should be a pre-configured Subgraph with a specific internal structure, not a fundamentally different type.

**Conceptual change**:
- `Network` creates a Subgraph containing: Encoder → Hidden (RNN) → Decoder
- The Subgraph's external ports are derived from the internal structure
- Users can enter the Network to see/modify its internal wiring

**For now (minimal change)**:
- Keep Network as-is in the component registry
- BUT make it navigable like any Subgraph
- When entering a Network, show its internal structure (or create one on first entry)

**Future work**: Full dynamic subgraph port derivation (see issue #4 below)

---

## 4. Dynamic Subgraph Port Derivation (LOWER PRIORITY)

**Current behavior**: Subgraph ports are manually defined or inherited from wrapped graph.

**Desired behavior**: A subgraph's external ports are automatically derived from unconnected internal ports.

**Algorithm**:
1. For each node inside the subgraph:
   - For each input port: if no internal wire targets it → expose as subgraph input
   - For each output port: if no internal wire sources from it → expose as subgraph output
2. External port names = `{nodeId}.{portName}` or just `{portName}` if unambiguous

**Example**:
```
Inside subgraph:
  Encoder: inputs=[x], outputs=[encoded]
  Hidden:  inputs=[in], outputs=[out]
  Decoder: inputs=[in], outputs=[y]

  Wires: Encoder.encoded → Hidden.in
         Hidden.out → Decoder.in

Unconnected: Encoder.x (input), Decoder.y (output)
Result: Subgraph has input_ports=['x'], output_ports=['y']
```

**Files to modify**:
- `web/src/stores/graphStore.ts` - add `deriveSubgraphPorts()` function
- Call this when entering/exiting subgraphs and when internal wiring changes

---

## 5. Verify Channel Naming in UI

**Registry status**: `component_registry.py` line 134 shows `name='Channel'` ✓

**Task**: Verify the UI displays "Channel" not "Feedback Channel" in:
- Component palette (left sidebar)
- Node type label on canvas
- Any other UI references

If it shows "Feedback Channel" anywhere, trace where that string comes from.

---

## Summary

| # | Issue | Priority | Complexity |
|---|-------|----------|------------|
| 1 | Subgraph navigation for any composite node | HIGH | Medium |
| 2 | Collapse behavior (single dot per side) | HIGH | Medium |
| 3 | Network as navigable subgraph | MEDIUM | Low (piggybacks on #1) |
| 4 | Dynamic subgraph port derivation | LOW | High |
| 5 | Verify Channel naming | LOW | Low |

Focus on #1 and #2 first. #3 follows from #1. #4 is architectural and can wait.

---

## Technical Notes

### Subgraph Detection Logic

Replace:
```tsx
const isComposite = spec.type === 'Network';
```

With something like:
```tsx
const isComposite = spec.type === 'Network' || spec.type === 'Subgraph' || hasSubgraph(nodeId);
```

Where `hasSubgraph` checks if `graph.subgraphs?.[nodeId]` exists.

### Collapsed Node Wiring

When a node is collapsed and shows single dots:
- Option A: Keep real port handles but position them all at the single dot location
- Option B: Create synthetic "collapsed" handles that map to the first port
- Option A is simpler and maintains wire integrity

### Height Calculation for Collapsed Nodes

```tsx
const COLLAPSED_BODY_HEIGHT = 24; // just enough for centered dots
const collapsedHeight = HEADER_HEIGHT + COLLAPSED_BODY_HEIGHT;
```
