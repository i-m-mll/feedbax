# External Surface Inventory

Umbrella issue: `6128292`.

## Preserve

- Python package identity: `feedbax` version `0.1.2`, Python `>=3.12`, wheel/sdist package `feedbax`.
- Root package exports from `feedbax/__init__.py`, including IO helpers, graph/model types, selectors, tree helpers, CDE types, and version/log-level values.
- CLI entry points: `feedbax-run`, `feedbax-analysis`, `feedbax-train`, `feedbax-provider`.
- Plugin entry point group: `feedbax.plugins`.
- FastAPI and WebSocket paths under `/api/*` and `/ws/*`, including graphs, components, provider, training, trajectories, runs, figures, analysis, inspection, orchestration, execution, and training/simulation websockets.
- Environment/config names: `FEEDBAX_DEBUG`, `FEEDBAX_LOG_LEVEL`, `FEEDBAX_TQDM`, `FEEDBAX_EXPERIMENTS_CONFIG_DIR`, `FEEDBAX_WEB_DATA`, `FEEDBAX_TRAJECTORIES_DIR`, `FEEDBAX_WORKER_URL`, `FEEDBAX_RUNS_DIR`, `FEEDBAX_STUDIO_WORKSPACE_ID`, `FEEDBAX_STUDIO_STAGE_ID`, `FEEDBAX_STUDIO_SCENARIO_ID`.
- Provider/manifest/execution schemas: `feedbax.manifest.v1`, `feedbax.provider.v1`, `feedbax.registry.v1`, `feedbax.execution.v1`.
- SQLite/web data tables used by routes: model/evaluation/figure records and manifest index tables.
- Frontend Vite dev surface: port `3008`, `/api` proxy to `localhost:8000`, `/ws` proxy to `localhost:8000`.
- Frontend browser storage keys: `feedbax:studio-local-tabs`, `feedbax:lastProjectId`, `feedbax-studio-layout`.
- Frontend drag MIME types: `application/feedbax-component`, `application/feedbax-analysis`.

## Explicitly Changeable

- Internal implementation details behind the public package/API/CLI/schema/storage surfaces above.
- Saved Studio graph format compatibility. Repo instructions say to raise clear errors rather than preserving saved graph fallback shims.

