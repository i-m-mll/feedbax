# Commit: [feature/web-ui] Expand edge routing + UI shelves

## Overview
This update implements the two-shelf layout (model editor + workbench) and adds
per-edge routing with editable elbow points. It also expands the component
catalog and introduces port type metadata for safer wiring validation.

## Changes

### Two-shelf layout
Top/bottom shelves now split the canvas and workbench, with NAND-style collapse
behavior, header toggles, and a draggable bottom shelf. Workbench tabs are
horizontally scrollable with edge fades for clarity.

### Edge routing + node stability
Edge routing lives in UI state per edge. Edges support elbow waypoints with
point handles, per-edge style toggling, and persistence in saved graphs.
Collapsed nodes keep handles mounted so edges remain visible.

### Component catalog & port typing
Component registry and local catalog are expanded with additional neural
network, mechanics, intervention, and task components. Icons are aligned to
available lucide glyphs. Port types (dtype + optional rank/shape) now drive
connection validation.

## Rationale
Separating model editing from training/analysis improves focus and layout
flexibility. Persisted edge routing enables precise diagram control without
global style toggles. A lightweight, extensible port typing scheme reduces
wiring errors while allowing future shape constraints.

## Files Changed
- feedbax/web/services/component_registry.py: expand built-in components
- web/src/data/components.ts: align local catalog + port types
- web/src/stores/graphStore.ts: persist edge routing + new shelf state
- web/src/components/canvas/RoutedEdge.tsx: per-edge routing rendering
- web/src/components/canvas/Canvas.tsx: routing + type validation
- web/src/components/canvas/CustomNode.tsx: stable collapsed handles
- web/src/components/layout/TopShelf.tsx: new model shelf container
- web/src/components/layout/BottomShelf.tsx: new workbench shelf + fades
- web/src/components/layout/Header.tsx: shelf toggle controls
- web/src/components/panels/ComponentLibrary.tsx: expanded icon map
- web/src/types/graph.ts: edge routing UI state types
- docs/WEB_UI_ISSUES.md: status updates
- docs/WEB_UI_RESPONSE_2026-01-27.md: appended progress report

---

Co-Authored-By: Codex (GPT-5) <codex@openai.com>
