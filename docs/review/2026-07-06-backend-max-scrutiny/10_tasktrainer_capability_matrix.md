# TaskTrainer → Executor Stack: Capability-Parity Matrix

**Purpose**: input to a binding retirement spec for the legacy `TaskTrainer`
(`feedbax/training/trainer.py`) in favor of the executor stack
(`feedbax/training/executor.py` + `checkpoint_custody.py`, plus
`worker_validation.py`, `phase_executor.py`, `contracts/worker.py`,
`contracts/checkpoints.py`).

**Branch analyzed**: `integration/db41e6a-backend-hardening` (about to merge
into `develop`). The branch ref has already been consumed by the merge, so it
no longer resolves directly; it was located via reflog as the second parent of
merge commit `ae207269` on `develop`:

```
COMMIT = d026038b453adcc0ce57979c0bcbe05b3cd386f5
```

All feedbax quotes below are `git show d026038b:<path>` (read-only,
`GIT_OPTIONAL_LOCKS=0` prefixed throughout the investigation; no mutating git
commands were run). rlrmp quotes are from the current working tree at
`/Users/mll/Main/10 Projects/10 PhD/rlrmp` (a separate, non-branch-pinned
repo).

---

## 1. Capability Matrix

| # | Capability | TaskTrainer (legacy) | Executor stack equivalent (file:line) | Verdict |
|---|---|---|---|---|
| 1 | Constructor / config surface | `TaskTrainer.__init__` (`feedbax/training/trainer.py:156-218`): `optimizer`, `checkpointing`, `checkpoint_custody` (mutually exclusive, raises `ValueError` if both set, `trainer.py:186-191`), `on_nan: "raise"\|"restore_checkpoint"`, `chkpt_dir`, `enable_tensorboard`/`tensorboard_logdir`, `model_update_funcs` (gradient-free per-batch state updates) | `execute_training_run_spec(spec, *, run_id, initial_slots, kernel_context, manifest_root, checkpoint_root, registry, loss_service, environment, resume, resume_slot_transform, stop_after_barrier, manifest_conflict_policy, issues, progress_callback, ...)` (`executor.py:174-197`); config is the Pydantic `TrainingRunSpec` (`contracts/training.py`), not constructor kwargs | equivalent-different-API (object→function/spec; `model_update_funcs` has no declared analogue found) |
| 2 | `train()`/run entry-point signature | `TaskTrainer.__call__(task, model, n_batches, batch_size, where_train, idx_start=0, opt_state=None, loss_func=None, ensembled=False, ensemble_random_trials=True, log_step=100, state_reset_iterations=None, save_model_parameters=False, save_trial_specs=None, toggle_model_update_funcs=True, restore_checkpoint=False, disable_progress=False, batch_callbacks=None, run_label=None, verbose_progress=True, loss_update_func=None, loss_update_iterations=True, loss_reduction_fn=None, pre_step_fn=None, rollout_step_hook=None, *, key)` (`trainer.py:220-250`), returns `(model, history, opt_state)` | `PhaseProgramExecutor.run(self, slots, *, run_id, resume_from_barrier=None, stop_after_barrier=None, context=None, progress_callback=None) -> PhaseExecutionResult` (`phase_executor.py:193-289`), invoked by `execute_training_run_spec` | equivalent-different-API (spec/phase-program-driven vs. flat kwarg surface; several kwargs — `loss_func`, `loss_update_func`, `pre_step_fn`, `rollout_step_hook`, `state_reset_iterations` — have no confirmed 1:1 executor hook) |
| 3 | `where_train` mid-run schedule | `where_train: WhereFunc \| dict[int, WhereFunc]`, keyed by global batch index, must include key `0` (`trainer.py:329-330`); mid-run the trainable spec and optimizer state are rebuilt, preserving opt-state for still-trainable params via `update_opt_state` (`trainer.py:127-141, 627-636`); `_get_trainable_params_superset` (`trainer.py:1121-1135`) preallocates history for the union of all stages' masks | **Not found.** Grepped `where_train` across `executor.py`, `checkpoint_custody.py`, `worker_validation.py`, `phase_executor.py`, `contracts/worker.py`, `contracts/checkpoints.py` — zero hits. Only occurrences are in the legacy collaborator `feedbax/training/train.py:134-144,286`, which itself calls `TaskTrainer`. `PhaseSpec.writes/reads/initializes` declare fixed, spec-authored state-slot associations per phase, not a runtime trainable-mask toggle. | **MISSING** |
| 4 | Batch callback mechanism | `batch_callbacks: Mapping[int, Sequence[Callable]]`; host-side, zero-arg, side-effect-only, fired after the optimizer step, before NaN check (`trainer.py:692-694`) | Two layers, both read-only observers, not per-batch-number registered hooks: `PhaseProgramExecutor.run(..., progress_callback: Callable[[ProgressCoordinate], None])` fired once per inner step (`phase_executor.py:255-256`); `execute_training_run_spec(..., progress_callback: Callable[[Mapping[str, Any]], None])` wrapped by `_live_progress_callback`, also accumulating `history_events` (`executor.py:453-463`) | equivalent-different-API (progress observer present; but no mechanism to register an arbitrary callable to fire exactly at a specific target batch index the way `batch_callbacks` does) |
| 5 | History/metrics accumulation shape | `TaskTrainerHistory` (`trainer.py:104-120`): `loss: TermTree[AbstractLoss] \| Array`, `loss_validation`, `learning_rate: Optional[Array]`, `model_parameters: Optional[Component]`, `trial_specs: dict[int, TaskTrialSpec]`. Shape `(n_batches,)` or `(n_batches, n_replicates)` if ensembled — fixed-shape, batch-indexable arrays, already trial-axis-averaged (`trainer.py:703-711`, `1153-1156`) | `TrainingRunExecutionResult.history_events` (`executor.py:82-93`): a **tuple of heterogeneous dicts**, each `{"type": "training_progress", "coordinate": ..., "metrics": dict}` (`_history_event`, `executor.py:445-450`). No fixed-shape stacked array. `_final_metrics` (`executor.py:559-568`) pulls slot keys ending in `"loss"` into a flat dict for a final summary only. | **MISSING** (array-shaped, batch-indexable history has no equivalent; this is the largest concrete gap — see §4/§6 consumer impact) |
| 6 | `train()` return value | `(model, history, opt_state)` tuple | `TrainingRunExecutionResult` object: `run_id, status, manifest, manifest_path, final_slots, final_coordinate, checkpoint_writes, history_events` (`executor.py:82-93`) | equivalent-different-API at the top level, but see #5 for the history-field gap specifically |
| 7 | Validation-set eval cadence | `log_batches_mask` via `np.linspace(idx_start, idx_end, n_batches // log_step, endpoint=False)` plus always the final batch (`trainer.py:596-604`); at each masked batch, delegates to `task.eval_with_loss`/`task.eval_ensemble_with_loss`, written to `history.loss_validation` at **absolute** batch index (`trainer.py:786-842`) — note indexing differs from training loss's `batch - idx_start` | No built-in cadence. `worker_validation.py` only does **pre-flight structural contract validation** once before training (`validate_worker_contract`, `worker_validation.py:330-660`, called at `executor.py:234-240`) — not numerical held-out-loss evaluation. A `"measurement"` `UpdateStepKind` vocabulary item exists (`contracts/worker.py:69-78`) for expressing held-out-eval steps inside an authored phase program, constrained at validation time (must write only metric slots, must declare `data_member`, `worker_validation.py:513-535`), but cadence is whatever the method's phase-program graph encodes — no default. | **MISSING** as a built-in default; equivalent-different-API only if the method author explicitly builds a "measurement" step into their phase program |
| 8 | Progress reporting (human-readable) | `jax_cookbook.progress.piter`/`progress_piter` + stdlib `logging`, no tqdm/rich/print. Per-logged-batch subdescription (`trainer.py:687-690`): `f"training loss: {_as_host_scalar(losses.total.mean()):{LOSS_FMT}}"` (`LOSS_FMT = ".2e"`). Console block every `log_step` via `logger.info` (`trainer.py:846-872`): `"Training iteration {batch}:"` / `"{ensembled_str}training loss: ..."` / per-term breakdown if `verbose_progress` | **None found.** No tqdm, rich, print, or logging progress lines in any of the six executor-stack files (grep-confirmed); only a dead reference `# disable_tqdm=True` in the sibling legacy file `train.py:293`. `git grep -n "BATCH phase"` / `"BATCH "` / `"elapsed="` across the whole tree at this commit finds no Python hits (only unrelated bash in `poll_run.sh`). | **MISSING** (no human-readable output at all; caller must build it from raw `progress_callback`/`ProgressCoordinate` data) |
| 9 | `BATCH phase=... batch=.../n loss=... elapsed=...s` log line | Not emitted by TaskTrainer at all — this format does not exist in feedbax | Also not emitted by the executor stack | **N/A for retirement** — confirmed independently rlrmp-owned; see §5 |
| 10 | Checkpointing cadence | Same `log_batches_mask`, gated by `batch > 0` (`trainer.py:790-791`) | **Checkpoint barriers**: phase-program-declared completion points (`PhaseSpec.checkpoint_barrier`), not a fixed N-step interval (`phase_executor.py:258-274, 291-329`) | equivalent-different-API (spec-authored barriers vs. fixed-N cadence) |
| 11 | Checkpoint serialization | `eqx.tree_serialise_leaves`/`eqx.tree_deserialise_leaves`, single `.eqx` file holding `(model, opt_state, history)` tuple, sidecar `last_batch.txt` bare integer (`trainer.py:1085-1098`); code has a `TODO` flagging an unfixed `opt_state` serialization bug | `pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)` / `pickle.loads` (`checkpoint_custody.py:8,160,314`); schema constant `SerializerVersionRecord.serializer: str = "feedbax.training.checkpoint_custody.pickle.v1"` (`contracts/checkpoints.py:47`); per-state-slot `.pkl` blobs inside a UUID transaction dir with rich manifest metadata (schema id/version, run-contract binding hashes, structural ABI fingerprint, content digest, population identity, parent lineage — see §3 for full diff) | equivalent-different-API, substantially more defensive; see §3 |
| 12 | Checkpoint resume | `_load_last_checkpoint` (`trainer.py:1100-1118`) deserializes against the caller's current `(model, opt_state, history)` PyTree as structural template; **`opt_state` explicitly not restored** (discarded on resume, consistent with the TODO) | `load_latest_checkpoint` (`checkpoint_custody.py:252-345`): re-validates schema identity, content hash, run-contract binding, consistency predicate, slot coordinate consistency, population identities, structural ABI fingerprints; optimizer state and PRNG key are generic named slots (`role="optimizer"`/`"prng"`), validated and **restored** like any other slot | equivalent-different-API, and a net capability **gain** on the executor side (opt_state resume actually works) |
| 13 | Ensemble/vmap handling | `n_replicates` inferred via `tree_infer_batch_size` excluding `StateIndex` nodes (`trainer.py:347-359`); outer `eqx.filter_vmap` over replicate axis wraps `@eqx.filter_jit`-decorated `_train_step`, inner vmap over trial-batch axis inside `_train_step`; `ensemble_random_trials` controls per-replicate key splitting | **No `jax.vmap`/`eqx.filter_vmap` anywhere in the six executor-stack files** (grep-confirmed). Ensembles are represented declaratively via `AxisSpec(role="population"/"member"/"replicate")` and `StateSlotSpec(role="population")`, with identity tracked at checkpoint time (`PopulationIdentityRecord`, `contracts/checkpoints.py:129-142`); actual vectorization presumably lives inside method-supplied update kernels, outside these six files | **MISSING** as an executor-owned mechanism — pushed to method authors; the stack treats population slots as opaque PyTrees rather than owning the vmap itself |
| 14 | NaN policy | Checked once per batch on `losses_mean.total` (mean over replicates if ensembled — no per-replicate isolation) (`trainer.py:762-783`). `on_nan="raise"` → `FloatingPointError`. `on_nan="restore_checkpoint"` → reuses `_load_last_checkpoint`, restores `model`/`history` (not `opt_state`); either branch is a terminal early-return, not skip-and-continue | **Zero occurrences** of `nan`/`isnan`/`NaN` across all six files and collaborators. The only generic mechanism that *could* implement this is `MetricGuardSpec`/transition guards (`contracts/worker.py:254-271`, `phase_executor.py:374-385`) — arbitrary method-supplied guard predicates on metric slots deciding phase transitions — but nothing built-in exists | **MISSING** |
| 15 | Alternate construction from a graph | `TaskTrainer.from_graph` (`trainer.py:1015-1083`) discovers a `TaskComponent` inside a `Component`/`Graph` | Not investigated as a distinct item; the whole executor stack is spec/graph-driven by design (`TrainingRunSpec`), so the "from a graph" use case is closer to the stack's native mode than to a bolt-on | equivalent-different-API (architecturally subsumed, not a discrete port) |
| 16 | Checkpoint mutual-exclusion guard | Constructor raises if both `checkpointing=True` and `checkpoint_custody=True` (`trainer.py:186-191`): *"Feedbax checkpoint custody must be the only active checkpoint writer for a run."* | N/A — the executor stack only ever uses checkpoint_custody; this guard exists on the TaskTrainer side specifically as the legacy/new-stack bridge point | exact (guard already anticipates retirement; no executor-side counterpart needed) |
| 17 | Schema identity / fail-closed version check on checkpoints | None — no schema versioning concept at all in TaskTrainer's format | `TRAINING_CHECKPOINT_TRANSACTION_SCHEMA_ID`/`_SCHEMA_VERSION` constants (`contracts/checkpoints.py:18-29`) plus nested sub-schema IDs; `_validate_schema_identity` (`contracts/checkpoints.py:193-207`) raises `ValueError` on mismatch, wrapped by Pydantic `ValidationError`; named exceptions `CheckpointIntegrityError`, `CheckpointContractBindingError` (explicit message: *"pass allow_new_lineage_override=True to resume as new lineage"*), `CheckpointCompatibilityError` | **capability gain, no legacy equivalent** — but no migration function exists yet (single schema version so far); plumbing is in place to fail closed per project policy |

**Tally**: 17 capability rows. **MISSING (no executor equivalent at all): 6** —
`where_train` mid-run schedule (#3), array-shaped/indexable history (#5),
default validation-eval cadence (#7), human-readable progress output (#8),
executor-owned ensemble vmap (#13), NaN policy (#14). Everything else is
either exact or equivalent-with-a-different-API, with two areas (#12 opt_state
resume, #17 schema versioning) representing net capability **gains** on the
executor side.

---

## 2. Consumer Inventory

### Feedbax (at commit `d026038b`, i.e. the integration branch tip)

| Consumer | file:line | Capabilities exercised |
|---|---|---|
| `TrainingContext.train` | `feedbax/xabdeef/contexts.py:71` | Constructs `TaskTrainer(optimizer=optimizer, checkpointing=True)`; calls with bound `where_train` attribute and `ensembled` toggle; passes arbitrary `**kwargs` through |
| `setup_trainer` | `feedbax/training/train.py:63-80` | Trainer construction wrapper |
| `train_pair` | `feedbax/training/train.py:83-124` | Chains baseline+condition training runs with `opt_state` continuity and `idx_start` — i.e. depends on TaskTrainer's opt_state threading across successive `__call__`s |
| `train_and_save` | `feedbax/training/train.py:~250-270` | Exercises `where_train` as an iteration-keyed dict via `where_strs_to_fns`, plus `save_model_parameters`, `state_reset_iterations`, `loss_update_func` |
| `setup_train_histories` | `feedbax/training/post_training.py:65-95` | Rebuilds a `TaskTrainerHistory` skeleton via `init_task_trainer_history` to deserialize saved history |
| `get_best_iterations_and_losses` / `get_train_history_figures` | `feedbax/training/post_training.py` (same module) | Read `history.loss` directly — depends on the array-shaped history contract (capability #5) |
| `supervised_task_trainer_mapping()` | `feedbax/contracts/worker.py:555-566` | **Declarative mapping table** describing `TaskTrainer.__call__`'s phase/state-slot/checkpoint-transaction shape in worker vocabulary — evidence the new stack *models* rather than *calls* TaskTrainer |
| Tests | `test_checkpoint_custody.py` (mutual-exclusion of `checkpointing`/`checkpoint_custody`), `test_parameter_constraints.py`, `test_trainer_hotpath.py` (host-sync perf regression), `test_trainer_nan_policy.py` | Constructor guard, NaN policy, hotpath perf |
| Examples/docs | `examples/1_train.ipynb`, `examples/4_vmap.ipynb` (ensemble demo), `saving_and_loading.ipynb`, `dash/dash_demo_data_gen.py` | Standard training walkthroughs, ensemble/vmap demo, save/load demo |
| Studio backend stub | `feedbax/web/services/training_service.py` | Stub sketch importing `TaskTrainer` for a planned async progress-streaming service (per project CLAUDE.md, confirmed still a stub) |

### rlrmp (current working tree, `/Users/mll/Main/10 Projects/10 PhD/rlrmp`)

| Consumer | file:line | Capabilities exercised |
|---|---|---|
| Timing benchmark | `src/rlrmp/benchmarks/local_parallel.py:245-320` | `ensembled=True`, progress suppressed |
| `closed_loop_distillation.py` | `src/rlrmp/train/closed_loop_distillation.py:842-857,946-1005` | Custom `loss_func`, custom `where_train`; writes returned `history` via `eqx.tree_serialise_leaves` **directly** to `training_history.eqx`, bypassing any feedbax persistence helper — a direct dependency on `TaskTrainerHistory`'s exact field layout |
| `minimax_native.py` | `src/rlrmp/train/minimax_native.py:373-392` | Warmup phase only; reads `_last_history_loss(history)` to seed rlrmp's own coordinate-based phase machine's `CONTROLLER_LOSS` slot. The adversarial ascent loop itself is fully custom (no TaskTrainer) |
| `cs_nominal_gru.py` — **deepest coupling in the codebase** | `src/rlrmp/train/cs_nominal_gru.py` | (a) `_build_trainer` builds `TaskTrainer` purely as an optimizer/schedule container; (b) a real `__call__` invocation inside a chunked resumable executor (~line 3119-3134), exercising `idx_start=0` + resumed `opt_state`, `pre_step_fn`; (c) `_initial_training_state` reuses only `trainer.optimizer.init` + `get_model_parameters`, bypassing `__call__` entirely; (d) **directly `eqx.filter_vmap`s TaskTrainer's *private* `_train_step`** (~line 6920-6943) to build a bespoke loop, reconstructing a compatible `TaskTrainerHistory` via `init_task_trainer_history`; (e) manually reconstructs `history.learning_rate` bookkeeping via `eqx.tree_at` reading `opt_state.hyperparams` because these paths never call `__call__` (~6424-6431, ~7012-7019); (f) `_emit_checkpoint_progress` reads history-chunk loss scalars via `_latest_loss_scalars(history_chunk, ...)` (~7222) to feed the `BATCH phase=checkpoint` progress line |
| Tests | `test_cs_nominal_gru.py` (smoke test through `train_pair`; confirms checkpoint-phase line absent for non-checkpoint-boundary runs), `test_cs_lss_gru.py` (only `get_model_parameters`+`filter_spec_leaves`, no `__call__`) | Smoke coverage of the above coupling |
| `eval_part2_5_figures.py` | `scripts/eval_part2_5_figures.py:20,77-85` | Post-hoc: rebuilds `init_task_trainer_history` skeleton to `eqx.tree_deserialise_leaves` a saved `train_history.eqx` — a load-time structural dependency on `TaskTrainerHistory`'s exact field set |
| `train_minimax.py` | `scripts/train_minimax.py:52` | Imports `make_batch_log_callbacks` (`# noqa: F401`) — genuinely `TaskTrainer.batch_callbacks`-compatible, but **dead wiring, never actually called** |

**Note on `cs_nominal_gru.py`**: this file is the single highest-risk consumer
for a retirement spec — it reaches past `TaskTrainer.__call__` into the
private `_train_step` staticmethod and reconstructs the history skeleton by
hand. Any retirement plan must treat this file as requiring either (a) a
dedicated migration to the executor stack's history/progress contracts, or (b)
an explicit, time-boxed exception if migrating it is out of scope for the
initial retirement wave.

---

## 3. Checkpoint Format Diff

| | TaskTrainer (legacy) | Executor stack (`checkpoint_custody.py`) |
|---|---|---|
| Serialization | `eqx.tree_serialise_leaves` / `eqx.tree_deserialise_leaves` | `pickle.dumps`/`pickle.loads` (`protocol=pickle.HIGHEST_PROTOCOL`) |
| Unit saved | Single `.eqx` file containing `(model, opt_state, history)` tuple | Per-state-slot `.pkl` blobs inside a UUID transaction directory |
| Layout | `<chkpt_dir>/ckpt_<batch>.eqx` + `<chkpt_dir>/last_batch.txt` (bare integer) | `<checkpoint_root>/checkpoints/<run_id>/{latest.json, transactions/tx-<uuid4hex>/{manifest.json, blobs/<slot>-<sha256>.pkl}}`, written atomically via `tempfile.mkdtemp` + `os.replace` |
| Metadata | None beyond the tuple itself and the bare batch-number sidecar | `CheckpointTransactionManifest` (`contracts/checkpoints.py:171-207`): `transaction_id, run_id, status ("partial"\|"final"), barrier, completed_coordinate (ProgressCoordinate), consistency_predicate, run_contract_binding` (schema IDs + sha256 hashes of spec/method/phase-program/objective/graph/optimizer bindings), `slots` (per-state-slot blob refs with structural ABI fingerprint + content digest + optional population identity), `content_integrity_digest, history_availability, parent_lineage` |
| Optimizer state | Included in the tuple but has a **known unfixed serialization bug** (source `TODO`); explicitly **not restored** on resume (discarded) | Generic named slot (`role="optimizer"`), validated and restored like any other slot |
| RNG key | Not explicitly tracked (whatever's embedded inside `model`/`opt_state`) | Generic named slot (`role="prng"`) |
| Resume constraint | Structural-template equality enforced only by `eqx.tree_deserialise_leaves`'s own shape/dtype checks; no explicit n_batches/ensemble-size assertion | Explicit multi-stage validation: schema identity, content hash, run-contract binding, consistency predicate, slot coordinate consistency, structural ABI, population identity — each with a named exception class |
| Version handling | None — no schema versioning concept | Explicit `schema_id`/`schema_version` fields + fail-closed validator (`_validate_schema_identity`, `contracts/checkpoints.py:193-207`); no migration function exists yet (`migrate_checkpoint_v1_to_v2`-style function not present) because only one schema version has existed so far — but the version-check plumbing is in place |

**What a read-side migration could NOT map** (justifying reject-with-error
rather than silent migration, per the project's no-compat-shim policy): the
legacy format has no schema identity, no content hashing, and no
run-contract binding, and (per the source `TODO`) unreliable optimizer-state
serialization. There is no principled way to reconstruct
`run_contract_binding`, `consistency_predicate`, or a verified
`structural_abi_fingerprint` from an old `.eqx` blob after the fact, because
those fields are hashes of contract objects (phase program, objective, graph)
that never existed for TaskTrainer runs.

**What is mappable in principle**: model weights, and — unreliably, given the
known bug — optimizer state, if a one-off importer pickled the deserialized
`.eqx` leaves into a synthetic v1 transaction manifest. **No such importer
exists in the codebase today.** The retirement spec should explicitly decide
whether to build one or to declare old TaskTrainer checkpoints permanently
unreadable by the new stack (consistent with feedbax's "no legacy fallback
paths" policy).

---

## 4. Return-Value/History Contract

- **TaskTrainer's history shape**: `TaskTrainerHistory` (`trainer.py:104-120`)
  is a fixed-shape, batch-indexable PyTree: `loss`/`loss_validation` as
  `TermTree[AbstractLoss] | Array` of shape `(n_batches,)` or
  `(n_batches, n_replicates)`; `learning_rate: Optional[Array]`;
  `model_parameters: Optional[Component]` with a leading saved-step axis when
  requested; `trial_specs: dict[int, TaskTrialSpec]` (not array-shaped).
- **Direct rlrmp dependents on this exact shape**:
  - `src/rlrmp/train/closed_loop_distillation.py:936-942` — writes the raw
    `TaskTrainerHistory` via `eqx.tree_serialise_leaves` to
    `training_history.eqx`.
  - `scripts/eval_part2_5_figures.py:77-85` — reads it back via
    `init_task_trainer_history` skeleton + `eqx.tree_deserialise_leaves`; a
    load-time structural dependency on the exact field set.
  - `src/rlrmp/train/cs_nominal_gru.py` (multiple sites, see §2) —
    reconstructs `TaskTrainerHistory` by hand for chunks that bypass
    `__call__`, and reads `.learning_rate`/`.loss` fields directly for
    progress-line construction.
  - `src/rlrmp/train/minimax_native.py:386` — `_last_history_loss(history)`
    feeds a phase-machine slot.
- **`training_summary.json` consumers**: no direct evidence found of code
  that both writes `training_summary.json` and unpacks a `TaskTrainerHistory`
  object in the same step — scalar extraction happens earlier (via the sites
  above), and plain dicts/floats are what actually flow into summary
  construction. So the immediate blast radius on `training_summary.json`
  itself looks contained, but the **upstream** scalar-extraction code that
  feeds it is not.
- **Executor stack's contract**: `TrainingRunExecutionResult.history_events`
  is a tuple of loosely-typed dicts (`{"type": "training_progress",
  "coordinate": ..., "metrics": dict}`), with no fixed-shape stacked array
  and no `.loss`/`.learning_rate` attributes. **None of the rlrmp consumers
  above can consume this shape without an adapter.** This is the single
  largest concrete parity gap surfaced by the consumer search — bigger in
  practical impact than the `where_train` or NaN-policy gaps, because it is
  load-bearing in a currently-active training script (`cs_nominal_gru.py`),
  not just a theoretical feature gap.

---

## 5. Progress-Line Contract

- **Contract owner**: `rlrmp/src/rlrmp/train/progress.py`. Its own docstring
  states the contract explicitly (lines 14-17):
  ```
  Line contract (consumed by poll_run.sh — do not reorder phase/batch
  without updating that consumer):
      BATCH phase=warmup batch=42/1000 loss=3.21 elapsed=12.3s
  ```
  Built by `format_batch_line` (lines 83-120); token constant
  `BATCH_LINE_TOKEN = "BATCH"` (line 40).
- **Two code paths**: (1) `make_batch_log_callbacks` (lines 123-169) *is*
  genuinely `TaskTrainer.batch_callbacks`-compatible and documented as such,
  but is imported dead in `scripts/train_minimax.py:52` (`# noqa: F401`) and
  never actually invoked anywhere. (2) `format_batch_line` /
  `should_log_batch` / `batch_log_every` are called **directly inside
  rlrmp's own bespoke chunked training loops** in `cs_nominal_gru.py`
  (`adaptive_epsilon` phase, `policy_adversary` phase, and
  `_emit_checkpoint_progress`) — plain Python loops, not
  `TaskTrainer.__call__`'s callback hook.
- **`poll_run.sh` side** (feedbax working tree, `scripts/deploy/poll_run.sh`):
  does **not** grep the literal `BATCH`/`phase=` tokens. Its extraction is a
  generic, format-agnostic heuristic:
  `grep -oiE '(batch|step|iter|it)[[:space:]=:]+[0-9]+'` piped through
  `grep -oE '[0-9]+' | tail -1`.
- **Conclusion**: the `BATCH phase=...` line is emitted entirely by rlrmp's
  own code, independent of TaskTrainer and of the executor stack. **Retiring
  TaskTrainer does not threaten this contract directly.** But because the
  executor stack emits *no* human-readable progress at all (matrix row #8),
  rlrmp will need to keep building its own progress lines from whatever raw
  data the new `progress_callback`/`history_events` mechanism supplies —
  there is no feedbax-side format string to inherit either way, so this is a
  "rlrmp must adapt its emitter, not a blocked migration" item rather than a
  MISSING-parity item in the strict sense.

---

## 6. Summary Verdict Counts

- Capability rows evaluated: **17**
- **MISSING** (no executor-stack equivalent found at all): **6** —
  `where_train` mid-run schedule, array-shaped/indexable history, default
  validation-eval cadence, human-readable progress output, executor-owned
  ensemble vmap, NaN policy
- **equivalent-different-API**: 9 rows (constructor/config, run-entry
  signature, batch-callback mechanism, return-value top level, checkpoint
  cadence, checkpoint serialization, checkpoint resume, `from_graph`
  construction, and the return-value/history split noted in row 6 which
  restates row 5's gap at the top level)
- **exact**: 1 row (mutual-exclusion guard, which anticipates its own
  retirement) plus 1 explicit **capability gain** (schema-identity fail-closed
  versioning) and a second gain embedded in row 12 (opt_state actually
  resumes correctly on the executor side, unlike TaskTrainer's known bug)
- Feedbax consumers found: **~11** call sites/files (contexts.py, train.py
  ×3 functions, post_training.py ×2, contracts/worker.py mapping table, 4
  test files, 3 example notebooks, 1 Studio-backend stub)
- rlrmp consumers found: **~8** call sites/files, with `cs_nominal_gru.py`
  representing by far the deepest and riskiest coupling (private
  `_train_step` vmap reuse, hand-reconstructed history)
