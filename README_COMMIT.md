# Commit: [feature/web-ui] Improve canvas UX and settings

## Overview
This commit tightens canvas UX by fixing delete sync, refining port layout, and
adding a wrap‑to‑parent graph action. It also introduces a basic settings popover
for app‑level toggles, including the minimap visibility.

## Changes

### Canvas behavior
- Keep graph data in sync when nodes are removed via UI changes
- Add a “wrap in parent” action to lift the current graph into a new outer graph
- Improve port label spacing and vertical centering

### Settings and controls
- Add an app settings popover with a minimap visibility toggle
- Move resize mode control into a compact canvas toolbar
- Ensure header menus render above the canvas and show disabled state clearly

## Rationale
Fixing delete synchronization prevents “ghost” nodes from reappearing when dragging
new components. A wrap‑to‑parent action supports top‑down modeling workflows, while
the settings popover and toolbar make global and canvas‑level controls discoverable.

## Files Changed
- `web/src/stores/graphStore.ts`
- `web/src/components/canvas/Canvas.tsx`
- `web/src/components/canvas/CustomNode.tsx`
- `web/src/components/layout/Header.tsx`
- `web/src/stores/settingsStore.ts`
- `feedbax/web/services/component_registry.py`
- `web/src/data/components.ts`
- `web/src/components/panels/ComponentLibrary.tsx`
