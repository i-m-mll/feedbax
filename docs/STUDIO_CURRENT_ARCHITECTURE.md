# Feedbax Studio Current Architecture

This is the short orientation path for agents working on Feedbax Studio. The
older `docs/WEB_UI_SPEC.md` is a historical 2026-01 draft, not the current
architecture contract.

## Source Of Truth

The graph is the model. The Studio canvas and its subgraphs describe the model
that the worker builds and trains. Do not synthesize background architecture,
fall back to stale outer params, or treat missing subgraphs as a recoverable
default. If the graph is incomplete, raise or surface a clear error.

Durable Studio specs and emitted payloads must keep explicit schema identity or
an explicit migration/rejection path. Validation-only frontend shapes are not a
substitute for durable schema handling.

## Frontend Shape

Studio is a React/Vite/TypeScript app in `web/`. It uses Zustand for local
stores, TanStack Query for server state, React Flow (`@xyflow/react`) for the
model canvas, generated Zod contracts, Tailwind CSS, lucide icons, Plotly, and
Recharts. `web/package.json` is the dependency authority; Radix UI and React
DnD are not current dependencies.

The main surfaces are:

- `web/src/components/layout/`: top shelf, bottom shelf, header, sidebars,
  settings overlay, and shelf sizing behavior.
- `web/src/components/canvas/`: React Flow canvas, node renderers, routed
  edges, state-flow edges, tap nodes, subgraph nodes, and port context menus.
- `web/src/components/panels/`: training/evaluation/analysis/report stage
  panels, properties, validation, console, trajectory/statistics/figure views,
  and pipeline-stage workspaces.
- `web/src/components/scenario/`: the top-pane Workspace view, task projection,
  observable/objective projections, and scenario inspection.
- `web/src/components/ui/`, `web/src/components/primitives/` if present, and
  nearby files: shared primitives and visual tokens.
- `web/src/stores/`: graph, workspace, layout, training, run, analysis,
  trajectory, statistics, settings, and project-tab state.

The top shelf owns the model/task/scenario workspace. The bottom shelf owns the
stage workspace and console mode. Stage panels should read and write through
`workspaceStore` and the generated contract types rather than inventing local
payload shapes.

Generated frontend contracts live in
`web/src/generated/studioContracts.ts` and are produced by
`scripts/generate_studio_contracts.py` from Python/Pydantic contract models.
Do not edit the generated file by hand.

## Save And Persistence

Project save state is optimistic-concurrency protected. The backend increments
`save_revision` on every accepted project write. Frontend saves send the current
revision through `expected_save_revision`/`If-Match`; stale saves return HTTP
409 with expected/current revision details. The frontend fetches the server copy
where possible and shows concrete local-versus-server conflict sections.

Autosave is debounced in `web/src/App.tsx`. The pagehide path uses the beacon
save endpoint so refresh/close events still attempt to persist graph, UI state,
workspace, and analysis pages. Same-project multi-tab editing is warned through
`BroadcastChannel`.

Local UI preferences are intentionally persisted where visible state should
survive refreshes. Examples include shelf/sidebar sizing and mode in
`layoutStore`, local project-tab restore state in `projectsStore`, and
shortcut-triggered save/delete/fit-view behavior in `useShortcuts`.

## Backend Shape

The Studio backend is FastAPI under `feedbax/web/`.

- `feedbax/web/app.py` wires REST and WebSocket routes.
- `feedbax/web/api/` contains graph, component, provider/schema, training,
  execution, run, analysis, figure, trajectory, statistics, inspection, and
  orchestration routes.
- `feedbax/web/services/graph_service.py` stores graph projects as filesystem
  JSON files under the configured graphs directory. It normalizes/migrates
  project payloads, validates graph connectivity, and enforces `save_revision`.
  SQLite is not the current Studio graph/project store.
- `feedbax/web/api/provider.py` exposes provider registries, schema
  enumeration, execution planning, Studio training preparation/local execution,
  and pipeline materialization.
- `feedbax/web/services/training_service.py` starts a local
  `feedbax.web.worker` subprocess on demand or forwards to a configured remote
  worker. It relays worker SSE progress, tracks monotonic event/status state,
  emits schema-versioned error/resync events, and proxies checkpoint metadata
  and checkpoint downloads.
- `feedbax/web/worker/client.py` is the HTTP/SSE client for worker health,
  start/stop/status/checkpoint endpoints, streaming, reconnect, and resync.
- `feedbax/web/ws/training.py` relays the worker SSE stream to browser
  WebSocket clients and closes defensively on disconnects.

The simulation WebSocket at `feedbax/web/ws/simulation.py` currently sends one
empty `simulation_state` message and closes. Do not document it as a production
simulation preview.

## Design Routing

Use this file for current orientation, then route to narrower authorities:

- Model graph semantics: `docs/design/SPEC_EAGER_MODELS.md`,
  `docs/COMPONENTS_AND_WIRES_SPEC.md`, `docs/STATE_MERGE_SEMANTICS_SPEC.md`,
  and `docs/STATE_WIRE_TAPS_SPEC.md`.
- Typed subgraph domains, including acausal/mechanics/penzai interiors and
  compile-status routing: `docs/design/typed_subgraph_domains.md`; umbrella
  `6116155`.
- Loss and training UI details: `docs/LOSS_UI_SPEC.md` and
  `docs/CLOUD_TRAINING_DISPATCH_SPEC.md`.
- Historical UI design conversation and old issue lists:
  `docs/WEB_UI_RESPONSE_2026-01-27.md` and `docs/WEB_UI_ISSUES*.md`.
- Pipeline pane and Workspace view behavior: current code in
  `web/src/components/panels/PipelineStageWorkspace.tsx`,
  `web/src/components/panels/RunCollectionStagePanel.tsx`,
  `web/src/components/scenario/ScenarioProjectionWorkspace.tsx`, and
  `web/src/stores/workspaceStore.ts` is the current authority unless a newer
  checked-in design doc appears.
- Studio platform-health umbrella planning: read the Mandible umbrella and
  child issues, especially `e59ed00`, when working in that umbrella. This
  checkout does not currently contain a standalone platform-health design doc.

## Verification Entry Points

Use the narrowest check that matches the change while iterating. Common Studio
checks are:

- Full integration bar at lane closeout: `scripts/full_suite.sh`.
- Generated contract freshness:
  `uv run --no-sync pytest tests/test_studio_api_contracts.py::test_generated_studio_contracts_are_current -q`.
  To regenerate, run `cd web && npm run generate:contracts`.
- Backend focused tests: relevant `uv run --no-sync pytest ...` paths such as
  `tests/test_studio_api_contracts.py`, `tests/test_studio_workspace.py`,
  `tests/test_studio_execution.py`, `tests/test_studio_runs_api.py`, and
  `tests/test_studio_analysis_jobs.py`.
- Frontend build: `cd web && npm run build`.
- Frontend focused tests: `cd web && npm run test -- <path-or-pattern>`.
- Instruction-policy mirror check, only when the marked AGENTS/CLAUDE test
  policy block changes:
  `uv run --no-sync python scripts/check_instruction_policy.py`.

For docs-only updates, `git diff --check` plus targeted stale-term searches are
usually enough unless the docs touch generated or mirrored instruction blocks.

## Known Non-Claims

Do not claim these are complete unless the code changes first:

- The simulation WebSocket is not a production preview stream.
- Python export from `GraphService.export_graph(..., "python")` is still a
  TODO placeholder.
- Studio graph/project storage is filesystem JSON, not SQLite.
- Training is not an in-process thread in the backend. The current local path
  is a worker subprocess with HTTP/SSE, relayed to browser WebSockets.
