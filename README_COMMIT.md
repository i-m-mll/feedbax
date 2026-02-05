# Commit: [feature/loss-ui] Add loss term configuration UI with probe management

## Overview

Implements a comprehensive UI for configuring loss terms in the feedbax web interface. Users can now add, edit, and remove loss terms, select probes from the graph, configure time aggregation settings, and see visual feedback linking loss terms to their corresponding graph ports.

## Changes

### Backend: Loss Service (`feedbax/web/services/loss_service.py`)

New service providing:
- **Probe extraction**: `get_available_probes()` discovers probes from barnacles, taps, and implicit output ports
- **Selector resolution**: `resolve_probe_selector()` converts selector strings to probe specifications
- **Time aggregation building**: `build_time_aggregation()` processes time aggregation specs
- **Validation**: `validate_loss_spec()` validates loss configurations against the graph
- **Spec-to-config conversion**: `spec_to_loss_config()` produces configuration for feedbax loss objects

### Backend: Training API (`feedbax/web/api/training.py`)

New endpoints:
- `GET /api/training/probes/{graph_id}` - List available probes for a graph
- `POST /api/training/loss/validate` - Validate a loss specification
- `POST /api/training/loss/resolve-selector` - Resolve a probe selector

### Frontend: Components

**New components:**
- `ProbeSelector` - Dropdown for selecting probes, grouped by node
- `TimeAggregationEditor` - Editor for time aggregation mode, range, segment, discount settings
- `LossTermDetail` - Detail panel for editing individual loss terms
- `AddLossTermModal` - Modal dialog for adding new loss terms
- `PortContextMenu` - Right-click menu on ports for quick probe creation

**Modified components:**
- `TrainingPanel` - Integrated loss term management with add/remove buttons, detail panel, validation error display
- `CustomNode` - Added visual highlighting for ports linked to selected loss terms, context menu support

### Frontend: State Management (`web/src/stores/trainingStore.ts`)

Extended store with:
- `availableProbes` - Cached list of probes from backend
- `selectedLossPath` - Currently selected loss term path
- `lossValidationErrors` - Validation errors from backend
- `highlightedProbeSelector` - Selector for visual highlighting
- Actions: `updateLossTerm`, `addLossTerm`, `removeLossTerm`

### Frontend: Utilities (`web/src/features/loss/`)

- `operations.ts` - Loss term manipulation utilities (get/update/add/remove at path, collect leaves, clone)
- `validation.ts` - Client-side validation for loss specifications

## Rationale

The loss function is central to training neural models, and users need fine-grained control over:
1. **What to measure**: Probe selection determines which signals contribute to the loss
2. **When to measure**: Time aggregation specifies which timesteps matter
3. **How to measure**: Norm functions and weights control the loss computation

The visual linking feature (highlighting graph ports when hovering over loss terms) helps users understand the connection between the abstract loss configuration and the concrete graph structure.

The port context menu provides a fast workflow: right-click on any output port to immediately create a loss term targeting that signal.

## Files Changed

**Backend:**
- `feedbax/web/services/loss_service.py` - New (180 LOC)
- `feedbax/web/api/training.py` - Modified (+70 LOC)
- `tests/test_loss_service.py` - New (200 LOC)

**Frontend:**
- `web/src/components/panels/ProbeSelector.tsx` - New (80 LOC)
- `web/src/components/panels/TimeAggregationEditor.tsx` - New (170 LOC)
- `web/src/components/panels/LossTermDetail.tsx` - New (200 LOC)
- `web/src/components/modals/AddLossTermModal.tsx` - New (170 LOC)
- `web/src/components/canvas/PortContextMenu.tsx` - New (90 LOC)
- `web/src/components/canvas/CustomNode.tsx` - Modified (+60 LOC)
- `web/src/components/panels/TrainingPanel.tsx` - Modified (+80 LOC)
- `web/src/stores/trainingStore.ts` - Modified (+80 LOC)
- `web/src/types/training.ts` - Modified (+40 LOC)
- `web/src/api/client.ts` - Modified (+20 LOC)
- `web/src/features/loss/operations.ts` - New (180 LOC)
- `web/src/features/loss/validation.ts` - New (180 LOC)
