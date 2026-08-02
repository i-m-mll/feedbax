# External Surface Inventory

Umbrella issue: `ce814d1`.
Branch: `feature/compat-internal-cleanup-residuals`.

## Preserve

- Python package identity: `feedbax` version `0.1.2`, Python `>=3.12`,
  wheel/sdist package `feedbax`.
- Root package exports from `feedbax/__init__.py`, including IO helpers,
  graph/model types, selectors, tree helpers, CDE types, graph templates, and
  version/log-level values.
- CLI entry points: the unified `feedbax` command, plus `feedbax-run`,
  `feedbax-analysis`, `feedbax-figure`, `feedbax-train`, `feedbax-provider`,
  and `feedbax-orchestrate`.
- Plugin entry point group: `feedbax.plugins`.
- FastAPI and WebSocket paths under `/api/*` and `/ws/*`, including graphs,
  components, provider, training, trajectories, runs, figures, analysis,
  inspection, orchestration, execution, and training/simulation websockets.
- Environment/config names: `FEEDBAX_DEBUG`, `FEEDBAX_LOG_LEVEL`,
  `FEEDBAX_TQDM`, `FEEDBAX_EXPERIMENTS_CONFIG_DIR`, `FEEDBAX_WEB_DATA`,
  `FEEDBAX_TRAJECTORIES_DIR`, `FEEDBAX_WORKER_URL`, `FEEDBAX_RUNS_DIR`,
  `FEEDBAX_STUDIO_WORKSPACE_ID`, `FEEDBAX_STUDIO_STAGE_ID`,
  `FEEDBAX_STUDIO_SCENARIO_ID`.
- Provider/manifest/execution schema versions and records:
  `feedbax.manifest.v1`, `feedbax.provider.v1`, `feedbax.registry.v1`,
  `feedbax.execution.v1`, `ProviderManifest`, `CapabilitySpec`,
  `MandibleManifestMapping`, `MandibleArtifactMapping`, `GraphSpecManifest`,
  `ModelArtifactManifest`, `TrainingRunSetManifest`, `TrainingRunManifest`,
  `EvaluationRunManifest`, `AnalysisRunManifest`, and `ReportManifest`.
- Mandible-facing provider manifest mapping fields exposed through
  `provider_manifest().mandible_manifest_mappings`, including manifest kinds,
  `subject_node_type`, artifact field mappings, custody hints, action names,
  related issue refs, parent refs, and opaque Feedbax-owned domain fields.
- SQLite/web data tables used by routes: model, evaluation, figure, manifest,
  and manifest-index records.
- Frontend Vite dev surface: port `3008`, `/api` proxy to `localhost:8000`,
  `/ws` proxy to `localhost:8000`.
- Frontend browser storage keys: `feedbax:studio-local-tabs`,
  `feedbax:lastProjectId`, `feedbax-studio-layout`.
- Frontend drag MIME types: `application/feedbax-component`,
  `application/feedbax-analysis`.

## Explicitly Changed Without Breaking External Contracts

- Local Studio execution now uses the current interpreter path for local backend
  runs so the worktree environment is used. Remote/cloud execution specs still
  use `python`.
- Development CORS now allows the inventoried Vite origin
  `http://localhost:3008`.
- Internal web-source aliases with no callers were removed; web source modules
  are not published SDK surfaces.

## Explicitly Changeable

- Internal implementation details behind the public package/API/CLI/schema/
  storage surfaces above.
- Dead temporary modules that are neither exported nor referenced in active
  docs, tests, routes, manifests, or package entry points.
- Local developer tooling implementation when script entry points and documented
  workflow remain intact.
- Saved Studio graph format compatibility. Repo instructions say to raise clear
  errors rather than preserving saved graph fallback shims.
