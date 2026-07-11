# Feedbax backend max-scrutiny review — synthesis

**Date:** 2026-07-06 · **Scope:** everything except the Studio frontend (`web/`) and the UI work owned by the three existing umbrellas ("Feedbax studio pipeline UI", "Feedbax workspace view design", "Feedbax studio improvements"). ~98k lines of Python across 25 subpackages, reviewed by six parallel area agents with independent spot-verification of all critical claims.

**Status note:** originally written pre-umbrella against develop `9ceb9805`. Filed as umbrella `db41e6a` (20 children after the wave-2 extension). Waves 1–3 implemented via auth:6eb99ef6, merged to develop at `ae207269`. See `07_auth_6eb99ef6_verification.md` (branch verification), `08_wave2_remaining_work.md` (remaining work + filing record), `09_losstermspec_inventory.md` and `10_tasktrainer_capability_matrix.md` (binding-spec inventories for the consolidation children). This directory is UNTRACKED and has been lost twice to host-side tree operations — it should be committed.

**Area reports** (evidence, file:line citations, per-finding proposals and effort estimates live there):

| File | Scope | Findings |
|---|---|---|
| `01_core_execution.md` | runtime/, execution/, components/, models/, acausal/ | 34 |
| `02_contracts_persistence.md` | contracts/, persistence/, config/, component_registry/ | 30 |
| `03_training_objectives_tasks.md` | training/, objectives/, intervene/, tasks/ | 34 (+ 3 verification notes) |
| `04_web_backend_studio.md` | web/ (Python), studio/, integrations/, dashboard/, bin/ | 23 |
| `05_analysis_plot_mechanics.md` | analysis/, plot/, mechanics/, control/ | ~63 |
| `06_packaging_tests_hygiene.md` | pyproject, layering, tests, CI, dead weight | 18 + layering map + coverage table |

Findings are referenced as `<report>#<ID>` (e.g. `03#resume-key-index-oob`). Claims marked ✓ were re-verified directly against source during synthesis. Line numbers reference `develop` at `9ceb9805` (pre-db41e6a).

---

## 1. Top findings (fix-first list) — addressed by db41e6a waves 1–3

1. **✓ Resume RNG indexing is out-of-bounds → silent key reuse** (`03#resume-key-index-oob`, blocker). `trainer.py` builds `keys = jr.split(key, n_batches)` but indexes it with the *global* batch counter; JAX clamps OOB silently on resumed runs, so late batches reuse the final key.
2. **✓ `python -m feedbax.web.worker` cannot execute** (`04#C1`, critical). No `__main__.py`; both the local training launch (`training_service.py:129`) and the GCP VM bootstrap (`startup_script.py:30`) use this invocation with `stdout/stderr=DEVNULL`.
3. **✓ `RigidTendonMusculoskeletalArm` configures `Kvaerno3` but never calls it** (`05`, critical). `self.solver = solver_type()` at `musculoskeletal.py:192` is the only reference to `self.solver` in the file.
4. **✓ `CompliantTendonHillMuscle.extract_outputs` returns a hardcoded zero force** (`05`, critical). `hill_muscles.py:723`.
5. **✓ `"l2"` norm is implemented as `jnp.abs`** (`03#l2-norm-implemented-as-abs-not-euclidean`, high). `objectives/service.py:751-752`.
6. **Stateful Equinox layers silently don't work** (`01#A1/A2/A3/A4`). BatchNorm never threads `eqx.nn.State`; Dropout forces inference; MultiheadAttention drops its key; `PenzaiStateManager.bind_state` is a no-op. Root cause `01#A7`: ~21 of ~30 wrappers are byte-identical copy-paste.
7. **CDE templates embed display-only node types** (`02#cde-templates-display-only-nodes-shadow-real-architecture`, critical). Direct "graph is the model" violation.
8. **Self-flagged silent truncation in analysis gradient path** (`05#AN-13`, critical). `analysis/grad.py`.
9. **`/api/orchestration/launch` executes an unvalidated shell string as root on a billed VM** (`04#C2`) + no launch gate (`04#H3`) + in-memory orchestration state (`04#M7`).
10. **✓ Versionless payloads treated as current-schema** (`02#versionless-payload-silently-treated-as-current`, high). `contracts/migrations.py:450-464`.
11. **13 of 16 matplotlib figure sites never `plt.close()`** (`05#PLOT-14`).
12. **Analysis caching defeated on main paths** (`05#AN-6/AN-7/AN-8`).

## 2. Cross-cutting themes

The most consistent result: **the repo's stated principles are good, and the highest-severity findings are places the code violates its own principles silently** — suggesting structural gates, not one-off patches.

**T1 — Silent substitution instead of raising** → gate child `535a32e`. **T2 — Copy-paste registration surfaces with no completeness checks** → child `2f16212`; serves `c0b869a`. **T3 — RNG discipline** (seven independent defects) → child `8467245`. **T4 — Two persistence stacks, one database** → child `a2d9695`, with `e33f487`. **T5 — Cloud/orchestration is optimistic-success end to end** → child `32ec389`. **T6 — God-modules with known seams** (analysis.py 3164, studio/schema.py 2382, provider.py 2224, task.py 2187, builtins.py 2108, migrations.py 2106) → child `beda036`. **T7 — Packaging and CI don't protect any of the above** (CI ran 2/109 test files, undeclared sqlalchemy, eager plotly, missing `feedbax_contract` markers, notebook/CLAUDE.md drift) → child `4be1586`; extends `1201afd`. **T8 — Legacy siblings of better implementations** (TaskTrainer vs executor; legacy `LossTermSpec` pipeline vs `ObjectiveSpec`; standalone `train_ppo` vs batched) → wave-5 children `34fed00`, `dd224bf`, `e80ec80`; see `08_wave2_remaining_work.md`.

**Terminology (important):** in the loss layer, "legacy vs modern" refers to the two spec-authoring generations inside `service.py`/`contracts/training.py` — NOT to `AbstractLoss`/`CompositeLoss`/`TermTree` in `loss.py`, which is the original, well-designed runtime engine both spec generations lower into and which no child modifies.

**Positive findings:** the core `Graph` engine compiles rollouts to a single `lax.scan` (`01#G2`); the migration registry is a real BFS-pathed version graph (`02`); no core→outer layering violations (`06#F6`); analysis topo-sort clean (`05#AN-3/AN-5`); tracked `d8f6b09`, `52774c3`, `a39e7d0` verified resolved (`03`).

## 3. Umbrella structure (as filed: db41e6a, 20 children)

| # | Child | Wave | Scope |
|---|---|---|---|
| 1 | `7714978` | 1 | Correctness hotfixes (§1 items 1–5, 8) |
| 2 | `2f16212` | 1–2 | Stateful layers + wrapper de-dup + registry invariants |
| 3 | `535a32e` | 2 | No-silent-substitution gate + feedbax_contract markers |
| 4 | `8467245` | 2 | RNG discipline audit |
| 5 | `897fc2c` | 1 | CDE display-only nodes + Studio missing-subgraph validation |
| 6 | `32ec389` | 2 | Orchestration hardening |
| 7 | `3fabf61` | 2 | Analysis cache repair |
| 8 | `a2d9695` | 3 | Persistence defect wave (with `e33f487`) |
| 9 | `4be1586` | 2–3 | Packaging + CI (extends `1201afd`) |
| 10 | `beda036` | 3 | God-module decompositions |
| 11 | `54eb43e` | 3 | Plot layer hygiene |
| 12 | `d0ea7bc` | 3 | Mechanics numerical audit (with `525249b`) |
| 13 | `f9a8524` | 4 | Auth-6eb99ef6 merge fixes |
| 14 | `dd224bf` | 5 | Loss-spec unification (closes `63230c3` gap) |
| 15 | `34fed00` | 5 | TaskTrainer retirement (blocked by rlrmp `a0a03bf`) |
| 16 | `e80ec80` | 5 | PPO consolidation |
| 17 | `48f6e40` | 5 | Durable-spec schema identity completion |
| 18 | `0eae385` | 5 | Config subsystem hardening |
| 19 | `c6c8af2` | 5 | Dashboard retirement |
| 20 | `6d7ecd7` | 6 | Residual small-fix sweep (diff-checked) |

## 4. Caveats

Six area agents wrote the reports; critical/blocker claims marked ✓ were independently re-verified; medium/low findings should be re-read in context before further filing. The `03` report notes some tracked issues verified-resolved; the `06` report contains the layering map and per-subpackage test-coverage table. Wave-5 children `dd224bf`/`34fed00`/`48f6e40` carry binding acceptance-criteria addenda grounded in `09_`/`10_` inventories.
