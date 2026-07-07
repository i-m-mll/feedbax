# feedbax Project Instructions

**Protected branch: `develop`**

## Worktree Layout

- **Main worktree** (`/feedbax/`): tracks the protected `develop` branch and is
  the integration target.
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

## Backward Compatibility

**Backward compatibility is not a concern.** There is a single developer. When the architecture improves, old saved graphs are expected to be re-created from Studio. We do not maintain legacy code paths, fallback logic, or compatibility shims for older graph formats. When something is wrong, raise a clear error rather than silently substituting a stale value.

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

## UI Conventions

**No-jitter**: Interactive/editable page elements must not change geometry (size, position, spacing) when interacted with, except as explicitly intended (e.g. expand/collapse). Hover states, focus rings, edit mode transitions must preserve element dimensions.

**No-volatility**: Everything the user sees in Studio must survive save/load/refresh cycles. If a UI element displays state, that state must be persisted. There are no exceptions — if it's visible, it's saved.

## Development

### Python/JAX Coding Conventions

- Follow PEP 8: 4-space indentation and a 100-character soft line limit.
- Use type hints for public APIs.
- Keep imports at the top of files unless a local import is needed for
  performance, optional dependencies, or typing.
- Use Google-style docstrings when useful; include shapes and dtypes for JAX
  arrays when relevant.
- Use `uv` for package management. Do not run `pip install` directly.
- Subclass `equinox.Module` for dataclasses-that-are-PyTrees; do not also add
  `@dataclass`.
- Treat `Module` instances as immutable. Use `equinox.tree_at` or
  `eqx.tree_at` for out-of-place updates; avoid direct attribute assignment.
- Use `eqx.field` for defaults and converters. Rely on `Module`'s default
  PyTree behavior unless custom flattening is truly needed.
- Import JAX tree utilities once as `import jax.tree as jt` and use `jt.*`
  consistently. Do not use deprecated `jax.tree_*` helpers.
- Use `import jax_cookbook.tree as jtree` for PyTree utilities not in core JAX,
  and `from jax_cookbook import is_type, is_module, is_none` for common
  shorthands.

### Test Policy

<!-- feedbax-test-policy:start -->
- This marked block is mirrored in `AGENTS.md` and `CLAUDE.md`. When changing
  Feedbax test policy, edit both copies in the same commit and run
  `uv run --no-sync python scripts/check_instruction_policy.py`.
- While iterating on a fix, run the narrowest relevant tests first: explicit
  node IDs or paths, `-k`, `pytest --lf`, or the repo's selective runner when
  one exists.
- Run the repo's full integration bar only at lane closeout before work lands
  on an integration or auth path, and at most once or twice per lane when a
  rerun is justified. Do not use the full bar to check whether a single fix
  worked. Repo instructions define the integration bar; this norm governs how
  often to pay it.
- Run the integration test bar through `scripts/full_suite.sh`. The wrapper uses
  `uv run --no-sync python -m pytest tests -n auto`, configures the persistent
  JAX compilation cache, and records green-tree memo entries only for a clean Git
  tree with the same `uv.lock`, Python, JAX, and jaxlib fingerprint. Dirty trees
  or unresolved fingerprint fields run the suite and do not record a green memo.
- The test JAX compilation cache defaults to the shared Git common-dir cache;
  override with
  `FEEDBAX_JAX_COMPILATION_CACHE_DIR` or disable with
  `FEEDBAX_DISABLE_JAX_COMPILATION_CACHE=1`.
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

- For Feedbax/RLRMP RunPod work, use the repository scripts first:
  `scripts/deploy/runpod_deploy.sh` for deployment/training launch and
  `scripts/deploy/poll_run.sh` for status polling. These scripts encode the
  expected pod creation/reuse, Docker tag checks, train-spec confirmation gate,
  SSH and GPU readiness checks, rsync paths, local editable path patching,
  CUDA JAX bootstrapping, remote GPU/JAX verification, train-spec sync, and
  sentinel/nohup launch behavior.
- Normal RunPod workflow is acquire first, deploy second. Use
  `scripts/deploy/runpod_deploy.sh --acquire-only` to create or attach to a pod,
  prove the direct `.ssh` endpoint with `~/.runpod/ssh/RunPod-Key-Go`, and
  verify `nvidia-smi` before any rsync, bootstrap, or training launch. If a
  direct endpoint is already known, pass `--ssh-host` and `--ssh-port` so the
  script skips endpoint discovery while still proving GPU readiness.
- Treat hand-rolled `runpodctl` calls, custom SSH readiness loops, ad hoc
  rsync/path patching, manual CUDA JAX install steps, direct nohup/sentinel
  launch snippets, and bespoke polling cadence instructions as fallback,
  debugging, or script-refactor context. If the scripts miss a required
  Feedbax/RLRMP RunPod case, improve the script rather than bypassing it
  long-term.
- Use `scripts/deploy/poll_run.sh` wherever possible for RunPod monitoring; it
  owns the default early/steady polling cadence and deterministic status-line
  output expected by agents.
- Never kill processes on TPU VMs via SSH; `kill`, `pkill`, or signals sent
  during SSH commands can disrupt the SSH session itself. If a process has
  crashed, clear `/tmp/libtpu_lockfile` and launch a new one.
- Always verify the latest code is deployed before running on cloud instances.
  Stale code on TPU/GPU is a recurring source of wasted time.
- Module instances are frozen: never assign to `self.field` after `__init__`.
  Use `eqx.tree_at` for out-of-place updates. Never use `dataclasses.replace`
  on Modules with computed fields.

## Active Feature Context

- `feature/differentiable-mjx`: CDE hidden-state stability experiments (v6→v9b), AnalyticalMusculoskeletalPlant, DiffraxBackend. Latest: hybrid fixed-decay + Anti-NF gate (v9b).
- Issue d8de481: Feedbax Studio cloud training orchestration + CDE graph editing. Deep context at `~/.claude/projects/-Users-mll-Main-10-Projects-10-PhD-20-Feedbax-feedbax/memory/studio-cloud-training-context.md`.
