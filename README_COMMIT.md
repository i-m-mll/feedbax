# Commit: [feature/web-ui] Add comprehensive web UI specification

## Overview

This commit introduces a detailed specification for a canvas-based web application that will allow interactive construction, visualization, and training of Feedbax models. The spec draws inspiration from Collimator.ai while adapting to Feedbax's eager graph architecture.

## Changes

### Web UI Specification (`docs/WEB_UI_SPEC.md`)

A ~900-line comprehensive specification covering:

**Technology Stack**
- Frontend: React 18 + React Flow 12 + TypeScript + Vite 5 + Tailwind CSS + Zustand
- Backend: FastAPI + WebSocket + Pydantic
- Rationale: React Flow is the mature, battle-tested choice for node-based editors; FastAPI integrates naturally with the JAX/Python backend

**Data Model & Serialization**
- TypeScript types mirroring Python `graph.py` structures (`ComponentSpec`, `WireSpec`, `GraphSpec`)
- Complete JSON project file format including graph, task, training config, and UI state
- Bidirectional serialization: UI ↔ JSON ↔ Python objects

**UI/UX Design**
- Clean, minimal aesthetic inspired by modern tools (Claude Code, Linear, Figma)
- Three-panel layout: component library (left), canvas (center), properties/training (right)
- Custom node design with visual ports (input=blue, output=green)
- Cycle wires rendered distinctly (dashed purple)

**Canvas & Graph Editing**
- Pure TypeScript implementations of graph surgery operations matching Python API
- Undo/redo system with 50-state history
- Validation with inline error display
- Keyboard shortcuts for power users

**Training Configuration**
- Side panel for optimizer and loss function configuration
- Hierarchical loss builder UI matching `TermTree` structure
- WebSocket streaming for real-time training progress
- Start/stop/pause controls with checkpoint management

**Component System**
- Registry pattern with parameter schemas for auto-generated property editors
- User component discovery from `~/.feedbax/components/`
- Future path for in-browser code editor

**Implementation Phases**
- Phase 1: Foundation (canvas, nodes, wiring)
- Phase 2: Component system (registry, properties)
- Phase 3: Training integration
- Phase 4: Execution & debugging
- Phase 5: Polish
- Phase 6: Future enhancements (dashboard integration, collaboration)

## Rationale

**Why React + React Flow over Svelte?**
While Svelte offers better raw performance (~30% faster loads, 1.6KB vs 40KB runtime), React Flow is significantly more mature and battle-tested for node-based editors. The ecosystem support and library quality outweigh the performance delta for this use case.

**Why JSON serialization instead of Python code generation?**
JSON serialization is the modern pattern—bidirectional, version-friendly, and doesn't require string manipulation or syntax handling. The Python side deserializes JSON and instantiates objects. Code generation is fragile and unnecessary.

**Why training config in a panel instead of on canvas?**
TaskTrainer operates at a meta-level—it orchestrates training of the model, not part of the computation graph itself. Putting it on canvas would conflate levels of abstraction. The Task (environment) can be a canvas component since it participates in data flow.

**Why user components via filesystem discovery?**
Injecting code into the library would be invasive and create versioning issues. Filesystem discovery (`~/.feedbax/components/`) is clean, explicit, and follows standard plugin patterns.

## Files Changed

- `docs/WEB_UI_SPEC.md` - New comprehensive specification (~900 lines)

*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*
