# feedbax Project Instructions

## Python/JAX Coding Conventions

### Coding Style & Naming

- Follow PEP 8: 4-space indentation and a 100-character soft line limit.
- Use type hints for public APIs.
- Keep imports at the top of files unless a local import is needed for
  performance, optional dependencies, or typing.
- Use `lower_snake_case` for modules, packages, functions, and variables;
  `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants.
- Use Google-style docstrings when docstrings are useful; include shapes and
  dtypes for JAX arrays when relevant.

### Environment Management

- Use `uv` for package management. Do not run `pip install` directly.

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

### Equinox Modules

- Subclass `equinox.Module` for dataclasses-that-are-PyTrees; do not also add
  `@dataclass`.
- Treat `Module` instances as immutable. Use `equinox.tree_at` or
  `eqx.tree_at` for out-of-place updates; avoid direct attribute assignment.
- Use `eqx.field` for defaults and converters. Rely on `Module`'s default
  PyTree behavior unless custom flattening is truly needed.

### JAX Tree API

- Import once as `import jax.tree as jt` and use `jt.*` consistently
  (`jt.map`, `jt.leaves`, `jt.structure`, `jt.flatten`, `jt.unflatten`).
- Do not use deprecated `jax.tree_*` helpers such as `jax.tree_map` or
  `jax.tree_leaves`.

### jax_cookbook Helpers

- Use `import jax_cookbook.tree as jtree` for PyTree utilities not in core JAX,
  such as `jtree.unzip` and `jtree.get_ensemble`.
- Use `from jax_cookbook import is_type, is_module, is_none` for common
  `is_leaf` predicates and shorthands.

## Project-Specific Rules

- Protected branch: `develop`.
- The repo root tracks the protected `develop` branch. Start implementation work
  in feature worktrees from the repo root. When release/default `main` is needed,
  use a named worktree such as `worktrees/main`.
- The graph is the model. Studio canvas nodes and subgraphs are the source of
  truth; do not synthesize background architecture or silently fall back to
  stale outer params.
- Backward compatibility is not a concern for saved graph formats. Raise clear
  errors rather than preserving fallback paths or compatibility shims.
- Durable artifact/schema changes require explicit migration handling. If a
  Feedbax change alters GraphSpec semantics, component type IDs, parameter or
  state roles, selector meanings, manifest formats, storage layouts, or
  checkpoint/artifact codecs, the same implementation must either preserve the
  existing semantic schema or include versioned migration logic and focused
  migration tests. Do not leave schema-affecting refactors as agent archaeology
  for downstream projects.
- Any new or changed structured spec emitter must declare a stable schema
  identity, such as an explicit version field or registered schema ID, and must
  integrate with the migration path or explicitly reject older versions with a
  clear error. This applies to provider APIs, manifests, Studio save/load,
  workers, analysis/evaluation/report execution, registries, and downstream
  extension hooks. Validation-only Pydantic or TypeScript shapes are not
  sufficient for durable emitted specs; implementation issues and auth specs
  must include focused acceptance evidence for old-version accept, migrate, or
  reject behavior.
- Studio needs both processes: frontend `cd web && npm run dev` and backend
  `uv run uvicorn feedbax.web.app:app --port 8000`.

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
