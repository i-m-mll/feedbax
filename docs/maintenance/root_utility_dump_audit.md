# Root Utility Dump Audit

Issue: `97cdf91` - Audit utility and integration dumps

Parent: `42198b0` - Finish residual root package cleanup

Branch: `feature/97cdf91-utility-dump-audit`

## Scope

This is a read-only recommendation artifact. It covers direct `feedbax/*.py`
utility, helper, and integration modules that are not already clearly owned by
the sibling lanes:

- `f5ebdfb`: thin residual root facade deletion.
- `6de75cd`: real root domain module rehoming.

The audit includes the explicitly requested modules plus other direct root
helpers that are utility-like or integration-like enough to need a disposition.
It does not propose compatibility shims: project policy says backwards
compatible Python import aliases are not required. Durable emitted specs and
storage schemas still require explicit schema identity, migration, or clear
old-version rejection.

## Evidence

- Live Mandible state: `mandible issue report 97cdf91 --json` and
  `mandible umbrella status 42198b0 --deep --json`.
- Root inventory: `rg --files feedbax | rg '^feedbax/[^/]+\.py$' | sort`.
- Call-site scan for candidate root helpers across `feedbax`, `tests`, `docs`,
  and `scripts`.
- Function/class inventory and line counts for the candidate modules.
- Feedbax `jax_cookbook` imports across package, tests, and scripts.
- Local `jax_cookbook` checkout found at
  `/Users/mll/Main/10 Projects/05 Utils/jax-cookbook`; the originally suggested
  `/Users/mll/Main/10 Projects/10 PhD/jax-cookbook` path does not exist.

The highest-coupling roots are `misc.py`, `database.py`, `_tree.py`,
`tree_utils.py`, `types.py`, and the execution-spec modules. These are not dead
files. They are used by tests, training, analysis, dashboard/web APIs, runtime
components, and CLI entrypoints.

## JAX Cookbook Relationship

Feedbax should continue depending on `jax_cookbook`. It is already a declared
dependency and live imports use it broadly for `LDict`, `MaskedArray`, predicate
helpers, `jax_cookbook.tree`, `jax_cookbook.misc`, progress helpers, and function
hashing. Absorbing the cookbook into Feedbax would increase root-package
coupling and duplicate a utility library that already exists locally.

`feedbax/_tree.py` substantially duplicates `jax_cookbook/tree.py`. The local
cookbook already provides the generic predicate helpers, tree filtering,
taking, setting, stacking, concatenating, named dict/tuple factories, zipping,
prefix expansion, path labels, equal-leaf helpers, byte counts, batch-size
inference, and `leaves_of_type`. Feedbax should import those from
`jax_cookbook.tree` or JAX core as implementation follow-up.

The remaining Feedbax-specific PyTree helpers should stay local until proven
generic:

- `filter_spec_leaves` has Feedbax-specific mask/subtree expansion behavior.
- `tree_set` has local handling around `None` leaves during partition/combine.
- `tree_call_with_keys`, shared-key markers, and marker unwrap helpers are tied
  to Feedbax task/intervention RNG semantics.

`tree_utils.py` is mostly analysis/config LDict manipulation rather than generic
JAX utility. Some level-rearrangement helpers may belong upstream in
`jax_cookbook` after tests, but most should move inside Feedbax analysis or
config packages.

## Module Dispositions

| Module | Disposition | Recommendation |
| --- | --- | --- |
| `_io.py` | Merge into `jax_cookbook` or split | The generic Equinox tree save/load and `arrays_to_lists` overlap cookbook IO. Keep only Feedbax-specific dated/commit filename behavior if still used, likely under persistence. |
| `_mapping.py` | Move to subpackage | `WhereDict` is a Feedbax selector/where-function structure used by tasks, objectives, and tests. Move to `feedbax.runtime.where` or `feedbax.selectors`, not `jax_cookbook` unless Feedbax coupling is removed first. |
| `_progress.py` | Delete or replace | Thin `tqdm` selector duplicates the cookbook progress surface. Replace local users with `jax_cookbook.progress` while preserving `FEEDBAX_TQDM` behavior if it is still expected. |
| `_treescope.py` | Move to integration package | Optional Treescope integration plus Feedbax graph cycle projection. Move to `feedbax.integrations.treescope`. |
| `_logging.py` | Move to private support or config | Feedbax process logging policy and Rich handlers. Move to `feedbax.config.logging` or `feedbax._support.logging`; keep private. |
| `_warnings.py` | Move to private support or inline with CLI setup | Small warning de-dup helper used by CLI setup. Keep private or merge into CLI/setup support. |
| `_tree.py` | Split: mostly cookbook, some local runtime | Replace generic helpers with `jax_cookbook.tree` or JAX core. Keep Feedbax-specific mask/call/shared-key helpers under `feedbax.runtime.tree` or private support. Delete unused helpers after public root export removal is coordinated. |
| `types.py` | Split | Reexports of cookbook types should disappear or import from cookbook directly. Move `TreeNamespace` and namespace conversion near config, analysis data types to `feedbax.analysis.types`, and durable enum/spec-facing types near contracts/config. |
| `tree_utils.py` | Split | Move analysis/config LDict helpers to `feedbax.config.tree`. Consider upstreaming reusable LDict level rearrangement helpers to `jax_cookbook` only after tests. Use JAX core for simple path flattening. |
| `misc.py` | Split aggressively | Generic helpers overlap `jax_cookbook.misc`, `jax_cookbook._where`, and `jax_cookbook._print`; domain math belongs in mechanics/analysis; logging, interrupt, filesystem, version, JSON/YAML, and location helpers belong in private support or config. Do not keep a new `utils.misc` dump. |
| `database.py` | Move/split to persistence | Large SQLAlchemy persistence layer for model/evaluation/figure records, dynamic DB schema, old model save/load, and record-to-hyperparameter reconstruction. Move to `feedbax.persistence.database` or `feedbax.storage.database`; treat as high migration risk. |
| `plot_utils.py` | Split into plot package | Move label helpers, figure flattening, annotation, Plotly widget, and save routines into `feedbax.plot.labels`, `feedbax.plot.figure_io`, and notebook-only integration modules. Coordinate `savefig` with persistence because `database.py` calls it. |
| `colors.py` | Move to plot package | Hyperparameter color specs and color-map setup belong in `feedbax.plot.colors`. |
| `setup_utils.py` | Split | Notebook file chooser belongs in `feedbax.integrations.notebook`; model loading, noise editing, task/model pair setup, replicate info, and query helpers belong with training or analysis support. |
| `cloud_backends.py` | Moved under execution package | Canonical module is `feedbax.execution.backends`; Modal and RunPod helpers stay with execution schema tests. |
| `execution_models.py` | Moved under execution package with schema identity | Canonical module is `feedbax.execution.models`. `ExecutionSpec` and `ExecutionPlan` now declare explicit schema versions. |
| `execution_plan.py` | Moved under execution package | Canonical module is `feedbax.execution.planning`, with provider CLI and local execution imports updated. |
| `local_execution.py` | Moved under execution package | Canonical module is `feedbax.execution.local`; focused execution contract tests cover manifest/log emission. |
| `hyperparams.py` | Move to config package | Converts config/YAML dictionaries to `TreeNamespace`, handles LDict-wrapped where specs, flattening, and derived training parameters. Move to `feedbax.config.hyperparams`. |
| `constants.py` | Move to config or analysis defaults | Contains evaluation defaults and replicate criterion, not generic constants. Move to `feedbax.config.defaults` or analysis/training defaults. |
| `environment.py` | Move to training/runtime contract package | Protocols for supervised and RL training environments and tasks. Move to `feedbax.training.environment` or `feedbax.runtime.environment`; it is not a utility dump. |
| `dimred.py` | Move to analysis package | PCA helper belongs in `feedbax.analysis.dimred` or an analysis math package. |
| `iterate.py` | Move to runtime package | Component iteration and streaming-loss scan logic belongs in `feedbax.runtime.iteration` with graph/runtime tests. |
| `manifest_index.py` | Move to contracts/artifact storage | SQLite indexing for Feedbax manifests belongs near manifest contracts or storage, e.g. `feedbax.contracts.manifest_index` or `feedbax.artifacts.index`. It already depends on manifest schema fields. |
| `perturbations.py` | Move to intervention/mechanics package | Helpers construct feedback impulses and time-series intervention params. Move under `feedbax.intervene.perturbations` or `feedbax.mechanics.perturbations`. |

## Sibling-Owned Root Modules

These direct root modules are intentionally not implementation targets for this
audit:

- Facade lane `f5ebdfb`: `artifact_schema.py`, `manifest.py`,
  `migrations.py`, `provider.py`, `retention_artifact_schema.py`,
  `studio_execution.py`, `studio_protocol.py`, `studio_schema.py`, and
  `execution.py` as combined facade.
- Domain lane `6de75cd`: `artifact_materialize.py`, `bodies.py`,
  `dynamics.py`, `environment.py` if that lane claims training protocol
  surfaces, `eqx_components.py`, `filters.py`, `graph_normalization.py`,
  `graph_templates.py`, `nn.py`, `nn_cde.py`, `noise.py`,
  `penzai_component.py`, `schema_namespace.py`, `serialization.py`,
  `serialization_builders.py`, and `serialization_prototypes.py`.

If later implementation finds that one of these is better handled by the utility
cleanup, the umbrella should reassign it explicitly before editing.

## Proposed Follow-Up Issue Slices

1. **Execution package and schema identity**
   - Completed in issue `49806b6`: root execution modules moved into
     `feedbax.execution.*`, in-repo imports updated, and `ExecutionSpec`
     schema identity/rejection coverage added.

2. **Persistence and legacy model database split**
   - Move `database.py` into a persistence/storage package.
   - Separate SQLAlchemy models, dynamic table migration, model tree save/load,
     figure persistence, and record-to-hyperparameter reconstruction.
   - Add focused tests around dynamic schema updates, hash paths, legacy
     load/save, dashboard/web API imports, and figure record retrieval.

3. **Plot and notebook integration split**
   - Move `colors.py` and `plot_utils.py` into `feedbax.plot.*`.
   - Move notebook file chooser and Plotly widget helpers from `setup_utils.py`
     or `plot_utils.py` into optional integration modules.
   - Verify analysis plotting imports and figure save behavior.

4. **Config, hyperparameters, and typed analysis data**
   - Move `hyperparams.py`, `TreeNamespace`, namespace conversion helpers,
     config enums, and analysis data containers to config/analysis homes.
   - Update config loaders, analysis modules, training modules, and tests.
   - Keep old-version schema behavior explicit where emitted specs are touched.

5. **Cookbook de-duplication**
   - Replace `_tree.py`, `_io.py`, `_progress.py`, and generic `misc.py` helpers
     with `jax_cookbook` or JAX core imports.
   - Upstream Feedbax-specific improvements only when they are actually generic,
     with cookbook tests first.
   - Keep Feedbax shared-key and selector behavior local.

6. **Runtime/private support cleanup**
   - Move `_mapping.py`, `_treescope.py`, `_logging.py`, `_warnings.py`,
     remaining private tree helpers, `iterate.py`, and `perturbations.py` into
     runtime, integration, support, or intervention homes.
   - Verify task/objective selector behavior, graph iteration, Treescope web API,
     and CLI setup behavior.

## Risk Notes

- Public root API risk: `feedbax.__init__` currently reexports several `_tree`
  helpers. The facade-removal lane should remove root exports before aggressive
  deletion, or implementation should update tests to assert the canonical API.
- Durable schema risk: execution specs and manifest indexes are user-visible
  data contracts. Moves must preserve semantics or add versioned migration or
  rejection tests.
- Database risk: `database.py` combines dynamic schema mutation, persistence,
  figure IO, legacy model bytes, and analysis reconstruction. It should not be a
  mechanical file move.
- Cookbook risk: the cookbook should receive only generic, tested helpers.
  Feedbax selectors, task/model semantics, graph cycle projection, and
  intervention RNG behavior should stay in Feedbax.
- Optional dependency risk: notebook, Plotly widget, Treescope, Modal, RunPod,
  pyexiv2, and database dependencies should remain lazily imported or isolated
  where possible so core imports stay light.

## Recommended Ordering

Start with execution schema/package work because it has the clearest contract
tests and the smallest bounded surface among large integration dumps. Then split
plot/config helpers, then persistence, then cookbook de-duplication. Runtime and
private support cleanup can proceed in parallel once facade and domain lanes
have removed root import pressure.
