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

## Active Feature Context

- `feature/differentiable-mjx`: CDE hidden-state stability experiments (v6→v9b), AnalyticalMusculoskeletalPlant, DiffraxBackend. Latest: hybrid fixed-decay + Anti-NF gate (v9b).
- Issue d8de481: Feedbax Studio cloud training orchestration + CDE graph editing. Deep context at `~/.claude/projects/-Users-mll-Main-10-Projects-10-PhD-20-Feedbax-feedbax/memory/studio-cloud-training-context.md`.
