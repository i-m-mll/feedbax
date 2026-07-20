# Feedbax full-suite runtime profile

Date: 2026-07-20  
Issue: [issue:32b6405]  
Protected baseline: `f86912f6ed3ca39b230dd01dc4a11dbaf344edd8`

## Executive finding

The reported increase from roughly 3 minutes to roughly 14 minutes is **not a
reproducible increase in the canonical full-suite wrapper on the protected
baseline**.

On the same shared local environment, a forced, memo-free invocation of
`scripts/full_suite.sh` completed in **250.4 seconds end to end**; pytest
reported **3,240 passed, 2 xfailed in 249.2 seconds**. Preserved comparable
results include **2,995 passed, 2 xfailed in 162.3 seconds** on 2026-07-18 and
**2,565 passed, 2 xfailed in 223.9 seconds** on 2026-07-14. The suite has grown,
but test-count growth is not the main explanation for runtime variance:

- From July 14 to the current baseline, the completed test count grew 26.3%
  while pytest wall time grew only 11.3%.
- From July 18 to the current baseline, the completed test count grew 8.2%
  while pytest wall time grew 53.6%.
- All 33 files changed since the July 18 profiled head contain 1,080 current
  tests and complete together in 14.8 seconds with xdist. This is an upper
  bound on the wall-time contribution of the 245 net-new cases, because it
  includes hundreds of pre-existing tests in those files.

The measured mechanisms large enough to explain a multi-minute outlier are:

1. **Serial versus xdist execution.** The canonical wrapper adds `-n auto` and
   created 14 workers on this machine. The same 347-case JAX/RL/mechanics
   selection took 95.9 pytest seconds with 14 warm-cache workers but 364.7
   pytest seconds serially. Raw `pytest` does not inherit `-n auto` from
   `pyproject.toml`; it is a materially different benchmark.
2. **JAX compilation-cache reuse.** The default cache is isolated by source and
   process ID, so each worker and each new invocation starts with a fresh cache
   directory. A bounded 14-worker tail comparison fell from 144.5 seconds cold
   to 96.2 seconds when the same cache namespace was reused, a 48.3-second
   (33%) reduction. General reuse is not yet safe: [issue:57c94e5] records a
   prior stale/bad executable that produced incorrect fixed-point results.
3. **A long, imbalanced JAX/RL/mechanics tail.** The current suite reached about
   91% completion around 90 seconds, then required roughly another 159 seconds
   to drain the remaining work. A small set of expensive individual tests and
   repeated PPO executions dominate this tail.
4. **Unrecorded host load and invocation details.** The 14-minute observation
   did not retain worker count, command, cache namespace, host load, or
   per-file durations. It cannot be attributed more precisely after the fact.

Collection/import startup (6.8 seconds) and warning reporting (382 warnings,
nearly unchanged from the historical 379) are visible but too small to explain
the outlier.

## Machine and environment

| Field | Value |
| --- | --- |
| Machine | Apple M4 Pro, 14 physical / 14 logical CPUs, 48 GiB RAM |
| OS | macOS 26.5.2, Darwin 25.5.0, arm64 |
| Python | 3.13.5 |
| JAX / jaxlib | 0.5.2 / 0.5.1 |
| pytest / pytest-xdist | 9.0.2 / 3.8.0 |
| JAX devices | one CPU device |
| Branch | `feature/32b6405-suite-runtime-profile` |
| Baseline | signed protected `develop` at `f86912f6` |
| Relevant ambient variables | no `XLA_FLAGS`, cache override, xdist override, or `PYTEST_ADDOPTS` |

All Git reads used `GIT_OPTIONAL_LOCKS=0`. Python commands used
`PYTHONPATH=src uv run --no-sync`. No dependency mutation, provider access,
training, cache cleaning, push, protected auth action, or production/test
semantic edit occurred.

## Canonical wrapper and configuration

`scripts/full_suite.sh` delegates to `scripts/full_suite.py`. On this baseline,
the dry run resolves to:

```text
python -m pytest tests -n auto
```

`pyproject.toml` itself adds only `--strict-markers`; it does not enable xdist.
Therefore:

```text
PYTHONPATH=src uv run --no-sync python -m pytest tests
```

is serial, while:

```text
GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src scripts/full_suite.sh --force --no-memo
```

is the canonical parallel integration bar.

The wrapper provides a machine-wide advisory lock and green-tree memoization.
The memo fingerprint includes Git tree, `uv.lock`, Python, JAX, jaxlib, pytest,
and xdist versions. It does not currently retain runtime, collection time,
worker count, warning count, load, or per-file durations.

## Historical comparison

| Date / head | Test files | Result | Pytest wall | Notes |
| --- | ---: | --- | ---: | --- |
| 2026-07-05 `0d055508` | 106 | no preserved count/runtime used | — | full-suite wrapper introduced |
| 2026-07-14 `a21fc422` | 167 | 2,565 passed, 2 xfailed | 223.88 s | preserved integration run |
| 2026-07-18 `4d2932e` | 174 | 2,995 passed, 2 xfailed | 162.27 s | preserved canonical wrapper output |
| 2026-07-20 `f86912f6` | 178 | 3,240 passed, 2 xfailed | 249.22 s | forced memo-free profile |

The remembered “roughly 3 minutes” is consistent with the 2026-07-18 preserved
result. The remembered “roughly 14 minutes” has no retained comparable command
or environment record and did not reproduce.

These rows are not controlled benchmarks: they differ in Git tree, test set,
possibly host load, and unobserved cache state. They are suitable for rejecting
a simple monotonic test-count explanation, not for attributing every second to
a code change.

### Growth since the preserved July 18 run

Between `4d2932e` and `f86912f6`:

- test files: 174 to 178;
- completed cases: 2,997 to 3,242, including expected failures;
- test source: +7,190 / -131 lines across 33 test files;
- directly added Python test definitions: 155 added, 4 removed (parametrization
  means definition counts are not item counts).

The current versions of all 33 changed files ran together as:

```text
1080 passed, 27 warnings in 14.78s
```

Because this selection includes all old and new tests in the files, 14.78
seconds is a conservative upper bound on the xdist wall occupied by the new
post-July-18 cases when run as one selection. It cannot explain the 86.95-second
difference between the two whole-suite observations.

## Current baseline measurements

### Collection/import

Command:

```text
GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src /usr/bin/time -lp \
  uv run --no-sync python -m pytest tests --collect-only -q
```

Result:

- 3,242 tests collected in 6.80 seconds;
- 8.46 seconds end to end;
- 621 MB maximum resident set size reported by `/usr/bin/time`;
- the same three collection-time JAX-static-array warning sites are repeated
  once per xdist worker during normal execution.

Collection is 2.7% of the profiled pytest wall. Import optimization cannot
recover the missing order of magnitude.

### Canonical full suite, default cache behavior

Command:

```text
GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src /usr/bin/time -lp \
  scripts/full_suite.sh --force --no-memo --durations=200 -q
```

Result:

- 3,240 passed, 2 xfailed;
- 382 warnings;
- pytest: 249.22 seconds;
- end to end: 250.40 seconds;
- user CPU: 1,156.92 seconds;
- system CPU: 168.14 seconds;
- maximum resident set size: 5.48 GB reported by `/usr/bin/time`;
- 14 xdist worker cache directories appeared for this invocation.

The progress shape matters: about 91% of cases completed in the first roughly
90 seconds, while the remaining roughly 9% held the run open for another 159
seconds. The suite is tail-bound rather than collection-bound.

## JAX/RL/mechanics tail attribution

A 13-file selection was used for cache and worker comparisons:

- `test_batched_ppo.py`
- `test_extended_ppo.py`
- `test_2link_muscle_routing.py`
- `test_mjx_plant.py`
- `test_backend.py`
- `test_hill_muscles.py`
- `test_dae.py`
- `test_cde_controller.py`
- `test_components.py`
- `test_acausal.py`
- `test_control_blocks.py`
- `test_worker_execution.py`
- `test_studio_execution.py`

It contains 347 collected cases (345 passed, 2 xfailed).

### Cache and worker comparison

| Mode | Pytest wall | End to end | User CPU | Max RSS |
| --- | ---: | ---: | ---: | ---: |
| 14 workers, cold named namespace | 144.24 s | 144.88 s | 902.78 s | 3.22 GB |
| 14 workers, reused namespace | 95.89 s | 96.39 s | 526.32 s | 4.16 GB |
| 7 workers, reused namespace | 107.07 s | 107.61 s | 388.05 s | 6.93 GB |
| serial, reused namespace | 364.70 s | 378.60 s | 379.84 s | 11.58 GB |

Interpretation:

- Cache reuse reduced 14-worker wall by 48.35 seconds (33.5%) on this tail and
  reduced user CPU by 376 seconds. This is real opportunity, but the previous
  correctness failure forbids simply sharing the cache globally.
- Fourteen workers were 10.4% faster than seven on this idle-machine selection,
  though they used 36% more user CPU. Reducing the default worker count is not
  supported by this evidence.
- Fourteen warm workers were 3.8 times faster than warm serial pytest. A serial
  invocation can plausibly turn the full suite into a many-minute run even
  without any regression.
- The 61.9-second slowest warm test imposes a hard tail floor that scheduling
  alone cannot remove.

### Top files by aggregate call time

Aggregate call time sums tests that run concurrently, so it is workload rather
than wall time. It identifies where CPU and compilation work are concentrated.

| File | Cases | Cold 14-worker call time | Reused-cache call time | Largest reused-cache call |
| --- | ---: | ---: | ---: | ---: |
| `test_extended_ppo.py` | 15 | 223.96 s | 123.56 s | 25.87 s |
| `test_batched_ppo.py` | 9 | 180.45 s | 104.89 s | 24.23 s |
| `test_mjx_plant.py` | 16 | 119.89 s | 84.22 s | 61.86 s |
| `test_backend.py` | 30 | 130.01 s | 78.18 s | 35.30 s |
| `test_2link_muscle_routing.py` | 33 | 99.59 s | 55.37 s | 32.66 s |
| `test_acausal.py` | 26 | 56.44 s | 49.31 s | 22.62 s |
| `test_studio_execution.py` | 28 | 33.37 s | 31.50 s | 4.08 s |
| `test_hill_muscles.py` | 46 | 23.15 s | 17.13 s | 13.26 s |

### Dominant individual tests

Cold 14-worker durations:

| Test | Duration | Assessment |
| --- | ---: | --- |
| `test_mjx_plant.py::TestMJXPlantDiffraxIntegration::test_euler_100_steps` | 71.62 s | useful stability integration; Python dispatch loop is reworkable |
| `test_extended_ppo.py::TestExtendedObsNorm::test_obs_norm_runs` | 60.30 s | duplicates the same configured training used by two sibling assertions |
| `test_extended_ppo.py::TestExtendedCurriculum::test_curriculum_stages_tracked` | 59.50 s | duplicates the same curriculum training used by two sibling assertions |
| `test_batched_ppo.py::TestBatchedCollect::test_shapes` | 59.29 s | duplicates rollout work used by sibling finiteness assertion |
| `test_backend.py::TestMechanicsMJXBackend::test_mjx_backend_multi_step` | 42.91 s | useful public integration; loop/step execution may be reworkable |
| `test_2link_muscle_routing.py::TestMJXPlant2Link::test_euler_50_steps_stable` | 37.18 s | useful stability integration; Python dispatch loop is reworkable |
| `test_acausal.py::TestLongHorizonStability::test_long_horizon_no_nan` | 26.40 s | useful 10k-step stability check; Python dispatch loop is reworkable |
| `test_batched_ppo.py::TestBatchedTrainShort::test_returns_improve` | 27.68 s | name overclaims; test checks finite returns, not improvement |

Durations varied substantially between cold, reused-cache parallel, and warm
serial modes. Rankings and structural duplication are more stable than any one
duration.

## Test usefulness and rework opportunities

### PPO tests: high-confidence duplicate work

`test_extended_ppo.py` contains 15 tests and calls `train_ppo_batched` 16 times.
The baseline, observation-normalization, lattice, curriculum, and combined-
enhancement configurations are each retrained separately for assertions that
could inspect one result: completion, policy shape, metric presence, and
finiteness. `test_batched_ppo.py` likewise executes identical collection calls
twice for shape and finiteness assertions in two families.

Recommended change: consolidate assertions by distinct configuration, not by
single assertion. This should preserve coverage while eliminating more than
half of the extended PPO training invocations and two duplicate batched rollout
executions. Keep failure messages granular inside the consolidated tests.

`TestBatchedTrainShort::test_returns_improve` should either assert a defensible
improvement criterion or be renamed to the finite/non-divergence contract it
actually checks. It is the only test marked `slow`; the full suite does not
exclude `slow`, while several unmarked tests are substantially slower.

Tracking: [issue:a33aa59].

### Long-horizon mechanics tests: useful but host-dispatch-heavy

The 50-step, 100-step, and 10,000-step stability checks protect real numerical
behavior and should not be deleted just because they are slow. Their Python
loops mix the desired numerical horizon with repeated host dispatch. A
JAX-native scan or batched checkpoint extraction may preserve the full horizon
and assertions while reducing dispatch overhead. Any rewrite must compare
fixed-seed trajectories/checkpoints before replacing the old path.

Tracking: [issue:c1015c2].

### Tests not shown to be wasteful

The post-July-18 contract/orchestration growth is large in lines and count but
cheap in parallel wall time: all 33 changed files, including 1,080 current
tests, complete in 14.8 seconds. There is no runtime basis to delete or demote
those tests.

The single-step and short public-integration tests in backend, Studio, worker,
and contract families remain useful despite aggregate totals. Their cost is
mostly many independent small checks, which xdist handles well.

## Cache findings

By default, `tests/conftest.py` selects:

```text
<git-common-dir>/feedbax_test_cache/
  <HEAD-prefix>-<tracked-source-digest>/
  pid-<worker-pid>/jax_compilation
```

This design was introduced after [issue:57c94e5] found that a reused persistent
cache could return an incorrect fixed-point result. It is source-safe and
process-safe, but it is not persistent across invocations in the performance
sense. At profile time the cache root contained 1,493 top-level source
namespaces and 5,643 second-level invocation namespaces. No cache was cleaned.

The measured reused namespace proves that reuse can save meaningful work on
the selected tests, not that unrestricted sharing is correct. A fix must first
reproduce and retain the historical corruption regression, then define a safe
reuse key and bounded retention policy.

Tracking: [issue:f93007c].

## Warning/reporting volume

The current suite emits 382 warnings versus 379 in the preserved July 14 and
July 18 runs. Most come from repeated static-JAX-array warnings and Plotly's
`scattermapbox` deprecation, with collection warning sites repeated once per
xdist worker.

Suppressing or fixing warnings would improve signal quality and shorten the
terminal report, but three additional warnings and a few hundred printed lines
cannot explain a multi-minute or 14-minute runtime. Warning cleanup should not
be presented as the runtime fix.

## Recommendations

Ordered by evidence and expected effect:

1. **Consolidate repeated PPO executions** ([issue:a33aa59]). Expected evidence:
   eliminate more than half of 16 extended PPO training calls plus two duplicate
   batched rollout calls; remeasure the 347-case tail. This is the strongest
   test-design change because it removes duplicated work without deleting
   distinct assertions.
2. **Develop correctness-safe JAX cache reuse and retention** ([issue:f93007c]).
   Measured opportunity: 48.3 seconds / 33% on the selected tail. This must be
   gated by the historical stale-executable regression; no blanket shared-cache
   change is justified yet.
3. **Move long-horizon checks off Python step loops** ([issue:c1015c2]).
   Measured targets: individual cold calls of 71.6, 37.2, and 26.4 seconds.
   Preserve horizons and compare fixed-seed trajectories before adoption.
4. **Persist structured suite profiles** ([issue:d136fad]). Record command,
   xdist worker count, cache mode, environment fingerprint, collection,
   warnings, per-file durations, slow nodes, load, and wall time in or beside
   the green memo. This is the prevention path for unexplained outliers.
5. **Keep `-n auto` on an otherwise idle 14-core machine.** Fourteen workers
   beat seven by 10.4% and serial by 3.8x on the measured tail. Revisit only
   with profiles from a genuinely shared/contended workload.
6. **Use only `scripts/full_suite.sh` for comparable integration timing.** Raw
   pytest is serial unless the caller adds xdist explicitly. Always record
   `--force`/memo behavior when reporting a duration.

No hard runtime budget is recommended yet. The preserved runs show meaningful
variance even with the same wrapper, and the 14-minute observation lacks the
metadata required to decide whether it was serial, contended, or anomalous.

## Reproducible command set

Environment and wrapper command:

```text
uname -a
sysctl -n machdep.cpu.brand_string hw.logicalcpu hw.physicalcpu hw.memsize
PYTHONPATH=src uv run --no-sync python -c \
  'import jax,jaxlib,pytest,xdist; print(jax.__version__,jaxlib.__version__,pytest.__version__,xdist.__version__,jax.devices())'
GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src uv run --no-sync python scripts/full_suite.py --dry-run
```

Collection:

```text
GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src /usr/bin/time -lp \
  uv run --no-sync python -m pytest tests --collect-only -q
```

Canonical profile (one full-suite run):

```text
GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src /usr/bin/time -lp \
  scripts/full_suite.sh --force --no-memo --durations=200 -q
```

Tail selection:

```text
TAIL_TESTS='tests/test_batched_ppo.py tests/test_extended_ppo.py tests/test_2link_muscle_routing.py tests/test_mjx_plant.py tests/test_backend.py tests/test_hill_muscles.py tests/test_dae.py tests/test_cde_controller.py tests/test_components.py tests/test_acausal.py tests/test_control_blocks.py tests/test_worker_execution.py tests/test_studio_execution.py'

GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src \
FEEDBAX_JAX_CACHE_INVOCATION_ID=profile-tail \
  uv run --no-sync python -m pytest $TAIL_TESTS -n auto --durations=100 -q

# Repeating exactly once with the same profile-only namespace measures reuse.
# This is not a recommendation to make the override the default: see issue 57c94e5.
```

Worker comparison:

```text
GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src \
FEEDBAX_JAX_CACHE_INVOCATION_ID=profile-tail \
  uv run --no-sync python -m pytest $TAIL_TESTS -n 7 -q

GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src \
FEEDBAX_JAX_CACHE_INVOCATION_ID=profile-tail \
  uv run --no-sync python -m pytest $TAIL_TESTS -q
```

Historical read-only proxies:

```text
GIT_OPTIONAL_LOCKS=0 git ls-tree -r --name-only <rev> tests \
  | rg '^tests/test_.*\.py$' | wc -l
GIT_OPTIONAL_LOCKS=0 git diff --stat 4d2932e..f86912f6 -- tests
GIT_OPTIONAL_LOCKS=0 git diff --numstat 4d2932e..f86912f6 -- tests
```

## Limitations

- The 14-minute run itself was not available as a structured profile, so its
  cause remains unproved.
- Historical rows are preserved execution evidence, not same-tree repeated
  benchmarks.
- Aggregate per-file call time is not additive wall time under xdist.
- The cache-reuse experiment was deliberately bounded and did not include the
  historical fixed-point corruption regression.
- No caches were cleaned, so a controlled “empty cache” benchmark was not run.
  “Cold” here means a newly named invocation namespace.
- Full serial pytest was not run because the selected 347-case tail already
  established the worker effect while avoiding another redundant complete
  suite.

---
Co-Authored-By: Codex (GPT-5) <codex@openai.com>
