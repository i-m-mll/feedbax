# feedbax Project Instructions

**Protected branch: `develop`**

## Worktree Layout

- **Main worktree** (`/feedbax/`): tracks the `main` branch, used for releases only. **Never interact with it for feature work.** All development targets `develop`.
- **`worktrees/develop/`**: the integration target; feature branches base off this and merge back here via auth request.
- Feature worktrees: `worktrees/feature__<name>/` — created with `wt feature/<name>` from inside `worktrees/develop/`.

## Repository Structure

- **Python library**: `feedbax/` — core JAX/Equinox components, graph execution engine, networks
  - CDE network: `feedbax/nn_cde.py`
  - Other networks (SimpleStagedNetwork, LeakyRNNCell): `feedbax/nn.py`
  - Graph execution: `feedbax/graph.py`
  - Component base: `feedbax/components.py`
- **Studio backend** (FastAPI): `feedbax/web/`
  - Training service (stub): `feedbax/web/services/training_service.py`
  - WebSocket handlers: `feedbax/web/ws/`
  - API routes: `feedbax/web/api/`
- **Studio frontend** (React/TypeScript): `web/`
  - Canvas + nodes: `web/src/components/canvas/`, `web/src/components/nodes/`
  - Bottom shelf panels: `web/src/components/panels/`
  - Zustand stores: `web/src/stores/`
- **Docs**: `docs/WEB_UI_SPEC.md` (87 KB comprehensive spec), `docs/COLLIMATOR_COMPARISON.md`
- **Design specs**: `feedbax_merge_spec.md`, `SPEC_EAGER_MODELS.md`

## Core Principle

**The graph is the model.** What is rendered in the Studio canvas is the literal model that is built and trained. No node type is decorative, templated, or a placeholder for something constructed elsewhere. The worker builds exactly what the graph spec describes — node types, params, and topology — without hardcoding or inferring any architectural choices. Any deviation from this is a bug, not a known limitation.

Corollaries that must be respected without exception:

- **No background construction.** Nothing in the build pipeline may construct architecture that the canvas does not describe. If a composite node has a subgraph, that subgraph is the source of truth — the outer/stale params stored on the node itself are not authoritative and must not be used to construct anything.
- **Absence of a subgraph is an error, not a condition to work around.** If a composite node has not had its subgraph populated (e.g., the user has never opened it in Studio), that is an incomplete model state. Raise a clear error rather than falling back to outer params or synthesising a default subgraph.
- **"Just for now" workarounds are bugs.** Temporary shims, display-only nodes that shadow real architectural choices, and fallback paths that substitute stale values silently are all bugs regardless of how they are labelled in the code.

## Backward Compatibility

**Backward compatibility is not a concern.** There is a single developer. When the architecture improves, old saved graphs are expected to be re-created from Studio. We do not maintain legacy code paths, fallback logic, or compatibility shims for older graph formats. When something is wrong, raise a clear error rather than silently substituting a stale value.

## UI Conventions

**No-jitter**: Interactive/editable page elements must not change geometry (size, position, spacing) when interacted with, except as explicitly intended (e.g. expand/collapse). Hover states, focus rings, edit mode transitions must preserve element dimensions.

**No-volatility**: Everything the user sees in Studio must survive save/load/refresh cycles. If a UI element displays state, that state must be persisted. There are no exceptions — if it's visible, it's saved.

## Active Feature Context

- `feature/differentiable-mjx`: CDE hidden-state stability experiments (v6→v9b), AnalyticalMusculoskeletalPlant, DiffraxBackend. Latest: hybrid fixed-decay + Anti-NF gate (v9b).
- Issue d8de481: Feedbax Studio cloud training orchestration + CDE graph editing. Deep context at `~/.claude/projects/-Users-mll-Main-10-Projects-10-PhD-20-Feedbax-feedbax/memory/studio-cloud-training-context.md`.
