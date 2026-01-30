# Commit: [feature/web-ui] Implement subgraph navigation and collapse redesign

## Overview
This commit enables navigation into any composite node (not just Network), redesigns
the collapse behavior to show a single dot per side with a shrunk box, and implements
dynamic subgraph port derivation from unconnected internal ports.

## Changes

### Subgraph Navigation
- `isComposite` now checks `type === 'Network' || type === 'Subgraph' || hasSubgraph`
- Any node with an entry in `graph.subgraphs[nodeId]` shows the enter button
- `enterSubgraph` creates an empty subgraph on first entry for non-Network nodes

### Collapse Behavior Redesign
- When collapsed: all port dots except the first are hidden (single dot per side)
- Labels are completely hidden when collapsed
- Box height shrinks to `HEADER_HEIGHT + COLLAPSED_BODY_HEIGHT (24px)`
- Hidden handles remain in DOM but are invisible and non-interactive

### Dynamic Subgraph Ports
- Renamed `deriveExternalPorts` → `deriveSubgraphPorts` for clarity
- Called automatically when wiring changes inside a subgraph
- Derives external ports from internal ports that have no internal connections

## Rationale
Subgraph navigation was limited to "Network" type, but any node can have an associated
subgraph. The collapse redesign reduces visual clutter to a minimal single-dot
representation while maintaining wire connectivity. Dynamic port derivation ensures
subgraph interfaces stay in sync with their internal structure.

## Files Changed
- `web/src/components/canvas/CustomNode.tsx` - collapse logic, isComposite check
- `web/src/stores/graphStore.ts` - enterSubgraph, deriveSubgraphPorts, createEmptySubgraph
