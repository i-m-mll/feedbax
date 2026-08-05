# feedbax Project Instructions

**Protected branch: `develop`**

## Worktree Layout

- **Main worktree** (`/feedbax/`): tracks the protected `develop` branch and is
  the integration target. Start implementation work in feature worktrees from
  the repo root rather than committing directly here.
- Feature worktrees: `worktrees/feature__<name>/` — created with
  `wt feature/<name>` from the repo root.
- Release/default `main`, when needed, lives in a named worktree such as
  `worktrees/main`.

## Repository Structure

- **Python library**: `feedbax/` — core JAX/Equinox components, graph execution engine, networks
  - CDE network: `feedbax/models/cde.py`
  - Other networks (SimpleStagedNetwork, LeakyRNNCell): `feedbax/models/networks.py`
  - Graph execution and Component base: `feedbax/runtime/graph.py`
- **Studio backend** (FastAPI): `feedbax/web/`
  - Training service: `feedbax/web/services/training_service.py` manages local
    worker subprocesses, remote worker forwarding, SSE relay state, and
    checkpoint proxy/download helpers
  - Worker client/subprocess path: `feedbax/web/worker/client.py`,
    `feedbax/web/worker/`, and `feedbax/web/ws/training.py`
  - WebSocket handlers: `feedbax/web/ws/`
  - API routes: `feedbax/web/api/`
- **Studio frontend** (React/TypeScript): `web/`
  - Canvas renderers: `web/src/components/canvas/`
  - Shelves, sidebars, and panels: `web/src/components/layout/`,
    `web/src/components/panels/`
  - Zustand stores: `web/src/stores/`
- **Docs**: `docs/STUDIO_CURRENT_ARCHITECTURE.md`,
  `docs/WEB_UI_SPEC.md` (historical 2026-01 draft),
  `docs/COLLIMATOR_COMPARISON.md`
- **Design specs**: `docs/design/feedbax_merge_spec.md`, `docs/design/SPEC_EAGER_MODELS.md`

## Core Principle

**The graph is the model.** What is rendered in the Studio canvas is the literal model that is built and trained. No node type is decorative, templated, or a placeholder for something constructed elsewhere. The worker builds exactly what the graph spec describes — node types, params, and topology — without hardcoding or inferring any architectural choices. Any deviation from this is a bug, not a known limitation.

Corollaries that must be respected without exception:

- **No background construction.** Nothing in the build pipeline may construct architecture that the canvas does not describe. If a composite node has a subgraph, that subgraph is the source of truth — the outer/stale params stored on the node itself are not authoritative and must not be used to construct anything.
- **Absence of a subgraph is an error, not a condition to work around.** If a composite node has not had its subgraph populated (e.g., the user has never opened it in Studio), that is an incomplete model state. Raise a clear error rather than falling back to outer params or synthesising a default subgraph.
- **"Just for now" workarounds are bugs.** Temporary shims, display-only nodes that shadow real architectural choices, and fallback paths that substitute stale values silently are all bugs regardless of how they are labelled in the code.

<!-- feedbax-downstream-stability:start -->
## Backward Compatibility and Downstream Stability

Internal and unregistered helpers are free to change; compatibility aliases and
silent legacy fallbacks are not maintained. The owner-ratified stable downstream
contract is `feedbax.downstream-interface-stability.v1` in
`docs/design/downstream_interface_stability.md`: extension protocol current
version `1`, minimum supported version `1`, effective Feedbax release `0.2.0`.
Only the import paths, behavior, and durable schemas enumerated there are
guaranteed.

Supported protocol versions coexist or migrate at an explicit version boundary.
Unknown, removed, unsafe, or otherwise unsupported versions fail closed with an
actionable error; never infer a version or retry through a compatibility shim.
GraphSpec, Studio-persisted state, manifests, checkpoints, and emitted specs
remain governed by explicit schema identity plus tested migration or explicit
rejection. A breaking change follows the policy's deprecation, duration,
release, external-fixture, and owner-ratification gates.
<!-- feedbax-downstream-stability:end -->

## Artifact Schema And Migrations

Durable artifact/schema changes require explicit migration handling. This is
not a request for silent backward-compatibility shims. It is a requirement that
Feedbax-owned saved formats remain semantically migratable as the library
evolves.

If a change alters GraphSpec semantics, component type IDs, parameter or state
roles, selector meanings, manifest formats, storage layouts, or
checkpoint/artifact codecs, the same implementation/auth request must either:

- preserve the existing semantic schema; or
- add a versioned migration rule/API plus focused tests for the affected
  schema transition.

When changing durable formats, record the migration issue and validation
strategy in the implementation issue or auth spec. Do not leave
schema-affecting refactors as agent archaeology for downstream projects.

Any new or changed structured spec emitter must declare a stable schema
identity, such as an explicit version field or registered schema ID, and must
integrate with the migration path or explicitly reject older versions with a
clear error. This applies to provider APIs, manifests, Studio save/load,
workers, analysis/evaluation/report execution, registries, and downstream
extension hooks. Validation-only Pydantic or TypeScript shapes are not
sufficient for durable emitted specs; implementation issues and auth specs must
include focused acceptance evidence for old-version accept, migrate, or reject
behavior.

## Naming Scope

A name states a concept. Parameters live in the body of the thing named, where
they can be read, diffed, and migrated; a name that restates a parameter becomes
false as soon as that parameter is revised. This applies to specs, envelopes,
locks, matrices, figures, analyses, reports, design documents, and fixtures —
anything durable enough to be referenced by something else.

- **Concept identity only.** Never a report-slot code (`k4`, `d1`), a model or
  cohort index (`m2`), or a parameter value or descriptor (knot count, grid
  span, scalar setpoint). Those change; the concept does not.
  `m2_robustness_report.v4` → `task_aware_robustness_report.v4`.
- **No campaign or coordination strings.** Wave and stage ordinals (`wave1`,
  `stage2`), umbrella or issue handles, branch, worktree, and session names say
  when and how work was organized, not what the thing is; that provenance
  belongs in the ledger and in commit history, where it stays accurate.
  `wave1_stage.lock` → `controller_geometry.lock`.
- **Adopted defaults version on succession; contrast variants may name their
  dimension.** Revising a recipe's parameters keeps the concept and bumps the
  version, so the succession is legible. Name a varied dimension only when the
  variation *is* the experiment. `near_zero_knots.lock` → `sampling.lock.v2`,
  but `target2x.lock` → `sampling_target2x.lock` keeps `target2x`.
- **Study scoping is structural.** A study-specific document lives in a
  subdirectory named for the study, under its layer directory, and does not
  repeat the study in the filename; generic documents stay at the layer root.
  `post_run/<study>_induced_gain.base.json` →
  `post_run/<study>/induced_gain.base.json`.
- **Presentation never names substance.** A figure or analysis identity must not
  depend on its slot or role in a report — reports bind figures, not the reverse
  — or reordering a report invalidates a name.
- **Schema and type identities are durable interface.** A registered schema id,
  component type id, or template id is renamed only through the migration policy
  above, never as a side effect of renaming the file carrying it; consumers bind
  the identity, not the path.

The test for a candidate name: if a value in the body changed tomorrow, would
the name become wrong? Then it is describing a value. Name the concept.

These rules are exported to downstream projects verbatim in
`feedbax/governance/templates/agent_instructions.v1.md`, installed by
`feedbax instructions install`. Change both together.

## UI Conventions

**No-jitter**: Interactive/editable page elements must not change geometry (size, position, spacing) when interacted with, except as explicitly intended (e.g. expand/collapse). Hover states, focus rings, edit mode transitions must preserve element dimensions.

**No-volatility**: Everything the user sees in Studio must survive save/load/refresh cycles. If a UI element displays state, that state must be persisted. There are no exceptions — if it's visible, it's saved.

## Development

### Python/JAX Coding Conventions

#### Coding Style & Naming

- Follow PEP 8: 4-space indentation and a 100-character soft line limit.
- Use type hints for public APIs.
- Keep imports at the top of files unless a local import is needed for
  performance, optional dependencies, or typing.
- Use `lower_snake_case` for modules, packages, functions, and variables;
  `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants.
- Use Google-style docstrings when docstrings are useful; include shapes and
  dtypes for JAX arrays when relevant.

#### Environment Management

- Use `uv` for package management. Do not run `pip install` directly.

#### Equinox Modules

- Subclass `equinox.Module` for dataclasses-that-are-PyTrees; do not also add
  `@dataclass`.
- Treat `Module` instances as immutable. Use `equinox.tree_at` or
  `eqx.tree_at` for out-of-place updates; avoid direct attribute assignment.
- Use `eqx.field` for defaults and converters. Rely on `Module`'s default
  PyTree behavior unless custom flattening is truly needed.
- Module instances are frozen: never assign to `self.field` after `__init__`.
  Use `eqx.tree_at` for out-of-place updates. Never use `dataclasses.replace`
  on Modules with computed fields.

#### JAX Tree API

- Import once as `import jax.tree as jt` and use `jt.*` consistently
  (`jt.map`, `jt.leaves`, `jt.structure`, `jt.flatten`, `jt.unflatten`).
- Do not use deprecated `jax.tree_*` helpers such as `jax.tree_map` or
  `jax.tree_leaves`.

#### jax_cookbook Helpers

- Use `import jax_cookbook.tree as jtree` for PyTree utilities not in core JAX,
  such as `jtree.unzip` and `jtree.get_ensemble`.
- Use `from jax_cookbook import is_type, is_module, is_none` for common
  `is_leaf` predicates and shorthands.

### Test Policy

<!-- feedbax-test-policy:start -->
- This marked block is mirrored in `AGENTS.md` and `CLAUDE.md`. When changing
  Feedbax test policy, edit both copies in the same commit and run
  `uv run --no-sync python scripts/check_instruction_policy.py`.
- While iterating on a fix, run the narrowest relevant tests first: explicit
  node IDs or paths, `-k`, `pytest --lf`, or the repo's selective runner when
  one exists.
- The routine `python -m pytest tests` bar distributes across cores by default
  (`-n auto` in the `pyproject.toml` addopts). For fast single-node-id
  iteration, add `-n0` to run in-process without pytest-xdist worker startup
  overhead.
- Run the repo's full integration bar only at lane closeout before work lands
  on an integration or auth path, and at most once or twice per lane when a
  rerun is justified. Do not use the full bar to check whether a single fix
  worked. Repo instructions define the integration bar; this norm governs how
  often to pay it.
- Use as few full-suite runs as possible during integration: targeted pytest
  selection (explicit node IDs, `-k`, `--lf`) is the default while iterating,
  and the full bar is paid at lane closeout only. NEVER run two full-suite
  invocations in parallel — not in one checkout, not across worktrees, and
  never via concurrently-dispatched subagents. Delegating sessions must pass
  this constraint down to every subagent they dispatch.
- Run the integration test bar through `scripts/full_suite.sh`. The default core
  profile excludes MJX simulation/integration and PPO rollout/training tests.
  Use `scripts/full_suite.sh --include-mjx`, `--include-ppo`, or
  `--include-optional` to add those explicit tiers. Direct pytest selections use
  `-m optional_mjx` or `-m optional_ppo`; cheap PPO API and structural contract
  tests remain in the core profile.
- The wrapper uses `uv run --no-sync python -m pytest tests -n auto`, configures
  the persistent JAX compilation cache, and records green-tree memo entries only
  for a clean Git tree with the same test profile and passthrough pytest arguments,
  `uv.lock`, Python, JAX, and jaxlib fingerprint. Dirty trees or unresolved
  fingerprint fields run the suite and do not record a green memo.
- The test JAX compilation cache defaults to the shared Git common-dir cache;
  override with
  `FEEDBAX_JAX_COMPILATION_CACHE_DIR` or disable with
  `FEEDBAX_DISABLE_JAX_COMPILATION_CACHE=1`.
- The sealed repo-snapshot cache follows the same precedent: it defaults to
  `<git-common-dir>/feedbax_repo_snapshots` for the running checkout and is
  overridden with `FEEDBAX_REPO_SNAPSHOT_CACHE_DIR`. Tests are pinned to a
  per-worker directory so they never share sealed bytes with production runs.
  Never point it back at a machine-global temporary directory: entries are
  durable and read-only, and an operating-system temporary-file reaper will
  empty them in place and poison later runs.
- New tests must be safe under `pytest-xdist`: write only to `tmp_path` or a
  unique per-test directory, avoid shared checkpoint/custody/cache locations
  unless the path includes a test-unique segment, and restore any process-global
  JAX, registry, environment, or cwd changes before the test exits. Tests must
  not depend on collection or execution order.
<!-- feedbax-test-policy:end -->

### Running Studio

Studio requires two processes:
- Frontend: `cd web && npm run dev` (Vite, default port 3008)
- Backend: `uv run uvicorn feedbax.web.app:app --port 8000` (FastAPI)
Both must be running for full functionality.

### Cloud/Remote Training Practices

- For Feedbax RunPod training, use the orchestration core as the primary paved
  road: run `feedbax-orchestrate preflight --assembly-request <path>` as the
  non-billable gate, then `feedbax-orchestrate launch --assembly-request <path>
  --driver runpod`. Monitor through `feedbax-orchestrate status` or
  `feedbax-orchestrate watch`. Downstream RLRMP launches route through RLRMP's
  `scripts/launch_training.py execute`, not the Feedbax deploy scripts.
- Normal RunPod workflow remains acquire first, deploy second. The RunPod
  driver enforces this discipline by completing preflight checks before pod
  creation, then proving endpoint and GPU readiness before deployment and
  training launch.
- Both drivers export a persistent JAX compilation cache to the rows they
  launch, so the locally run end-to-end smoke that precedes remote acquisition
  does not recompile from scratch every time. The RunPod driver uses
  `<volume-mount>/jax_cache`; the local driver defaults to
  `<git-common-dir>/feedbax_jax_compilation_cache`, so worktrees of one checkout
  share compiled artifacts while sibling checkouts stay separate. Override with
  `FEEDBAX_JAX_COMPILATION_CACHE_DIR` or disable with
  `FEEDBAX_DISABLE_JAX_COMPILATION_CACHE=1`, matching the test cache.
- Keep `scripts/deploy/runpod_deploy.sh` and `scripts/deploy/poll_run.sh` as
  legacy/parity references for debugging and script-refactor work; they are no
  longer the primary launch or monitoring interface.
- Treat hand-rolled `runpodctl` calls, custom SSH readiness loops, ad hoc
  rsync/path patching, manual CUDA JAX install steps, direct nohup/sentinel
  launch snippets, and bespoke polling cadence instructions as fallback,
  debugging, or script-refactor context. If the orchestration core misses a
  required Feedbax/RLRMP RunPod case, improve the orchestration path rather than
  bypassing it long-term.
- Never kill processes on TPU VMs via SSH; `kill`, `pkill`, or signals sent
  during SSH commands can disrupt the SSH session itself. If a process has
  crashed, clear `/tmp/libtpu_lockfile` and launch a new one.
- Always verify the latest code is deployed before running on cloud instances.
  Stale code on TPU/GPU is a recurring source of wasted time.

## Active Feature Context

- `feature/differentiable-mjx`: CDE hidden-state stability experiments (v6→v9b), AnalyticalMusculoskeletalPlant, DiffraxBackend. Latest: hybrid fixed-decay + Anti-NF gate (v9b).
- Issue d8de481: Feedbax Studio cloud training orchestration + CDE graph editing. Deep context at `~/.claude/projects/-Users-mll-Main-10-Projects-10-PhD-20-Feedbax-feedbax/memory/studio-cloud-training-context.md`.
