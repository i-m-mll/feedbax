# Commit: [feature__web-ui] Refresh loss function UI

## Overview

This commit implements the loss function UI based on the new spec, replacing the placeholder "Composite loss tree" text with an interactive tree editor and equation summary. The TypeScript types and stores have been updated to align with the Python `TermTree` / `CompositeLoss` structure.

## Changes

### Loss Term Schema (`web/src/types/training.ts`)

Added `TimeAggregationSpec` and updated `LossTermSpec` to include:
- `label` for display names
- `selector` for probe references or state paths (e.g., `"probe:effector_pos"`)
- `norm` for distance function selection (`squared_l2`, `l2`, `l1`, `huber`)
- `time_agg` for time aggregation settings (mode, discount, etc.)

### Training Store (`web/src/stores/trainingStore.ts`)

Updated the default loss spec to match the new schema with realistic example terms:
- `position` — Effector position with power discount
- `final_velocity` — Final velocity (final step only)
- `regularization` — Network activity regularization

### Training Panel (`web/src/components/panels/TrainingPanel.tsx`)

Major UI additions:
- **Equation summary**: Compact `L = w₁·L_pos + w₂·L_vel + ...` display at top
- **Loss tree**: Hierarchical tree view with expand/collapse, selection, and inline weight editing
- **Detail lines**: Shows probe/selector, norm, and time aggregation per term
- **Navigation**: Click equation term to scroll and highlight in tree

### Specification (`docs/LOSS_UI_SPEC.md`)

New comprehensive spec documenting:
- Probe-as-tap architecture (probes are generic taps, behaviors reference them)
- Hybrid display (tree + equation)
- Time aggregation presets and discount options
- Norm function registry
- Serialization format and Python backend integration
- 5-phase implementation plan

### Issues Tracker (`docs/WEB_UI_ISSUES_v3.md`)

Batch 3 issues including node resizing, collapse behavior, port positioning, shelf layout bugs, and loss tree improvements.

## Rationale

The previous placeholder loss display provided no insight into the training objective. The new implementation:
- Makes loss structure visible and editable without leaving the UI
- Uses string selectors (`probe:name` or state paths) that serialize cleanly to JSON
- Aligns with Python's existing `WhereDict` pattern for bidirectional conversion
- Separates probe nodes (generic taps) from behaviors (loss, plotting) for maximum flexibility

## Files Changed

- `docs/LOSS_UI_SPEC.md` — New loss UI specification
- `docs/WEB_UI_ISSUES_v3.md` — Batch 3 issues tracker
- `web/src/types/training.ts` — Added TimeAggregationSpec, updated LossTermSpec
- `web/src/stores/trainingStore.ts` — Updated default loss with new schema
- `web/src/components/panels/TrainingPanel.tsx` — LossEquation + LossTree components
- Other modified files — Related refinements to graph store, node rendering, layout
