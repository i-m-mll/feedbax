# Commit: [feature/web-ui] Fix shelf layout and subgraph persistence

## Overview
This commit stabilizes the two-shelf layout with consistent headers, collapse logic,
and resize behavior, and it persists subgraph graphs/UI state in saved project data.
It also tightens node port layout/spacing and panel formatting to match the new shelf
UX.

## Changes

### Two-shelf layout
- Add dedicated headers for model and workbench shelves with collapse controls
- Enforce NAND collapse logic and allow full-range splits; hide resize handle when
  collapsed
- Remove global header toggles to keep shelf controls in-place

### Subgraph persistence
- Add `subgraphs` to GraphSpec and `subgraph_states` to GraphUIState (TS + Pydantic)
- Store and propagate subgraph graphs/UI state on enter/exit, rename, and delete
- Normalize subgraph UI state recursively during hydration

### Node and panel polish
- Correct port dot placement based on node geometry and align labels
- Ensure node header spacing avoids name/type collisions
- Constrain Validation panel callout width for readability
- Hide edge-style toggle pending full routing tools (tracked separately)

## Rationale
Keeping shelf controls within each shelf avoids disappearing affordances and makes
collapse/resize behavior predictable. Persisting subgraphs in the graph model fixes
UI-only state and enables serialization. Port positioning now derives from geometry
to keep edges and labels aligned under resize.

## Files Changed
- `web/src/App.tsx`, `web/src/stores/layoutStore.ts`
- `web/src/components/layout/TopShelf.tsx`
- `web/src/components/layout/BottomShelf.tsx`
- `web/src/components/layout/Header.tsx`
- `web/src/stores/graphStore.ts`
- `web/src/types/graph.ts`
- `feedbax/web/models/graph.py`
- `web/src/components/canvas/CustomNode.tsx`
- `web/src/components/canvas/Canvas.tsx`
- `web/src/components/canvas/RoutedEdge.tsx`
- `web/src/components/panels/ValidationPanel.tsx`
- `web/src/components/panels/PropertiesPanel.tsx`
