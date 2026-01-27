# Commit: [feature/web-ui] Implement Phase 1 web UI foundation

## Overview

Initial implementation of the canvas-based web UI for Feedbax model construction. This commit delivers the Phase 1 foundation: a working React Flow canvas with custom nodes/edges, FastAPI backend with component registry, and bidirectional JSON serialization.

## Changes

### Frontend (`web/`) — ~1950 lines TypeScript/React

**Canvas System** (`components/canvas/`)
- `Canvas.tsx` — React Flow wrapper with background, controls, minimap
- `CustomNode.tsx` — Component nodes with port visualization (input=blue, output=green)
- `CustomEdge.tsx` — Wire rendering with cycle detection styling

**Layout** (`components/layout/`)
- `Header.tsx` — Project name, file menu placeholder
- `Sidebar.tsx` — Collapsible left panel container
- `StatusBar.tsx` — Connection status, validation status

**Panels** (`components/panels/`)
- `ComponentLibrary.tsx` — Searchable, categorized component palette with drag-to-add
- `PropertiesPanel.tsx` — Parameter editing for selected nodes
- `TrainingPanel.tsx` — Optimizer/loss configuration UI (scaffold)
- `InspectorPanel.tsx` — State inspection (scaffold)
- `RightPanel.tsx` — Tabbed container for properties/training/inspector

**State Management** (`stores/`)
- `graphStore.ts` — Zustand store for graph state, React Flow integration, undo/redo
- `trainingStore.ts` — Training configuration state

**Graph Logic** (`features/graph/`)
- `operations.ts` — Pure functions: addNode, removeNode, addWire, removeWire, insertBetween
- `validation.ts` — Graph validation, cycle detection, error/warning collection

**Types** (`types/`)
- `graph.ts` — ComponentSpec, WireSpec, GraphSpec matching Python structures
- `components.ts` — ComponentDefinition, ParamSchema
- `training.ts` — OptimizerSpec, LossTermSpec, TrainingSpec

**API Client** (`api/`)
- `client.ts` — REST client with TanStack Query integration

### Backend (`feedbax/web/`) — ~960 lines Python

**FastAPI Application**
- `app.py` — Application factory with CORS, router registration
- `config.py` — Configuration (ports, paths, user component directory)

**API Endpoints** (`api/`)
- `graphs.py` — CRUD for graph projects (list, create, get, update, delete, validate)
- `components.py` — Component registry (list, get, refresh user components)
- `training.py` — Training job management (start, status, stop)
- `execution.py` — Simulation execution endpoint

**WebSocket Handlers** (`ws/`)
- `training.py` — Training progress streaming
- `simulation.py` — Simulation state streaming

**Services** (`services/`)
- `component_registry.py` — Component discovery, built-in + user component registration
- `graph_service.py` — Graph file operations, validation
- `training_service.py` — Training job execution in background thread

**Serialization**
- `serialization.py` — Pydantic models, Graph ↔ JSON conversion
- `decorators.py` — `@register_component` for user-defined components

### Configuration

- `web/vite.config.ts` — Vite with React, path aliases, API proxy
- `web/tailwind.config.js` — Tailwind with custom colors
- `web/tsconfig.json` — TypeScript strict mode
- `scripts/dev.sh` — Start frontend + backend concurrently
- `scripts/build.sh` — Production build script
- `pyproject.toml` — Added FastAPI, uvicorn, websockets dependencies

## Rationale

**Why implement before finalizing spec?**
The spec provided sufficient detail to begin implementation. Building the foundation surfaces practical issues early and validates the architecture.

**Graph store design**
The Zustand store maintains both `graphSpec` (canonical data) and React Flow's `nodes`/`edges` (derived for rendering). Changes update both synchronously to avoid drift.

**Component registry pattern**
Backend discovers components at startup, exposing metadata (ports, param schemas) via API. Frontend fetches this once and caches. User components loaded from `~/.feedbax/components/`.

## Files Changed

- `web/` — Complete frontend application (25 files)
- `feedbax/web/` — Complete backend module (14 files)
- `scripts/` — Dev and build scripts
- `pyproject.toml`, `uv.lock` — New dependencies

*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*
