# Training-Run Lifecycle Contract

Status: accepted design contract for feedbax issue `590d24b`. This document is the authority
for the orchestration migration (umbrella `1c6293a`); the bash deploy scripts and Studio
training service remain the operative surfaces until the migration waves replace them.

Repository-state citations refer to feedbax and rlrmp as read on 2026-07-09 (feedbax `develop`
at `1137e0f`). A detailed survey of the current surfaces is archived in Mandible custody with
the review bundle.

---

## 0. Decisions made by this document

| # | Decision |
|---|---|
| D1 | One orchestrator: a Python library `feedbax.orchestration`, consumed by a CLI, by the Studio backend, and (during migration) by the existing bash scripts as thin shims. The two current systems — the RunPod nohup+sentinel CLI path and the Studio subprocess+SSE path — converge on it rather than being maintained in parallel. |
| D2 | Run identity is minted by the orchestrator at bundle-assembly time (`run_set_id`, per-row `row_id`), never by a worker. Workers accept identity as input. |
| D3 | The lifecycle is a fixed stage model (§2) executed by a persisted, idempotent run-set state machine (§7). Every stage is re-entrant; a fresh process can resume any run set from its state document. |
| D4 | Progress, readiness, and terminal status are carried by a structured run-event protocol (`feedbax.run_event.v1`, §3) emitted by executors. JSONL files are the canonical transport; SSE/WS streaming and human-readable `BATCH` lines are renderings. File sentinels remain as the driver-level coarse truth and must agree with events. |
| D5 | Every run set ends with a spec-conformance certificate (`feedbax.run_conformance.v1`, §4): an automatic red/green artifact verifying that realized behavior matched the declared spec. A run without a certificate is not a completed run. |
| D6 | Environments are realized hermetically from the bundle (lockfiles + repo revisions + declared overlay steps) and summarized by an environment fingerprint recorded in the state document and the `TrainingRunManifest` (§5). Fingerprint mismatch fails before launch. |
| D7 | The orchestrator owns teardown and budget guards (§6). Successful completion tears the pod down by default; today's behavior (pod deliberately left running after a successful deploy) is inverted, with `keep_alive` as the explicit opt-out. |
| D8 | Backend specifics live behind a driver interface (§8): `local`, `runpod`, `modal` (future). Secrets and provider APIs are confined to drivers. |
| D9 | The CLI is a new fifth feedbax entry point, `feedbax-orchestrate` (`feedbax.bin.orchestrate:main`), following the existing `feedbax.bin.*:main` convention. |
| D10 | Studio's `TrainingService` becomes a client of `feedbax.orchestration` (§9): externally minted job identity, persisted job state, run-event vocabulary. The worker HTTP/SSE surface is retained as the live-streaming transport for local/remote workers. |
| D11 | feedbax is ledger-agnostic: REGISTER emits payloads only (tracked specs, terminal-status documents, certificate references); commits, auth requests, and any issue-tracker actions belong to sessions or to a future user-configurable terminal-hook interface (deferred). No ledger client is baked into the orchestrator. |

## 1. Current state (what this contract replaces or absorbs)

Two disjoint orchestration systems exist today.

**System A — RunPod CLI path.** `scripts/deploy/runpod_deploy.sh` (driver, ~1130 lines) with
`lib_acquire.sh` (pure pod-state/rows-manifest helpers) and `lib_run_prep.sh` (provider-
abstracted run-prep: train-spec gate, baseline staging, row launch, warm-compile gate). Rows
are launched as `nohup` processes with file sentinels (`<row>.started/.pid/.done/.failed`);
progress is inferred by `poll_run.sh` grepping logs with a loose regex; readiness for warm-
compile fan-out is a log-regex probe; teardown fires only if acquisition never completed; the
run's operational state lives in the shell process and is lost when it exits. rlrmp's
`scripts/post_run.sh` performs artifact sync, manifest parity checks, run-spec commit, and the
terminal ledger checkpoint.

**System B — Studio path.** `feedbax/web/services/training_service.py` spawns (or connects to)
a `feedbax.web.worker` FastAPI process; the worker mints a `uuid4()` job id, runs the shared
training executor in a thread, and emits typed JSON events (`training_log`, `training_progress`,
`training_trajectory`, `training_complete`, `training_error`) over an in-process queue → SSE →
service normalization (`feedbax.spec.studio.api_transport.v2`) → browser WebSocket. All live
job state is in-memory in two processes and unrecoverable across a backend restart; a job
killed before terminal status never writes a `TrainingRunManifest` and becomes invisible to
`/api/runs/training`.

Shared durable substrate (kept, not replaced): `TrainingRunManifest`
(`feedbax/contracts/manifest.py`; two producers — Studio "pending/planned" pre-dispatch and the
executor "completed" post-run), checkpoint custody transactions with `latest.json` pointers
(`feedbax/training/checkpoint_custody.py`; two-step atomic publication), the migrations
registry (`feedbax.contracts.migrations`), and the SQLite manifest index.

Load-bearing constraints this contract respects: manifests have two legitimate producers at
two lifecycle points; `latest.json` is the only valid way to resolve "current checkpoint";
checkpoint publication crash-safety semantics are already correct and are consumed as-is;
`FEEDBAX_RUNS_DIR` pinning in rlrmp's post-run path is fail-closed and stays that way.

## 2. Stage model

A **run set** is the unit of orchestration: one spec bundle, one backend target, N rows.
Stages execute in order; MONITOR overlaps LAUNCH per row. Every stage is idempotent: re-running
a completed stage is a no-op verified against recorded outputs, and re-running an interrupted
stage either completes it or fails cleanly without duplicating side effects.

| Stage | Purpose | Driver hooks | Terminal failure class |
|---|---|---|---|
| ASSEMBLE | Resolve the spec bundle: run specs per row, task/graph specs, declared baselines, environment declaration, budget, launch policy. Mint `run_set_id` and `row_id`s. Write the initial state document. | none | spec-invalid |
| PREFLIGHT | All statically decidable checks, before any billable action (§2.2). | none | preflight-failed |
| PROVISION | Acquire compute (pod create/reuse, worker VM, or no-op for local). Record endpoint + billing start. | `provision` | provision-failed |
| REALIZE_ENV | Build the declared environment on the target; compute and verify the environment fingerprint (§5). | `realize_env` | env-failed |
| STAGE_INPUTS | Sync code and resolve input custody: pinned checkpoint transactions, baselines, spec files. Verify digests remotely. | `stage_inputs` | staging-failed |
| LAUNCH | Start rows per launch policy (§2.3): warm-first, parallelism cap, stagger. | `launch_row` | launch-failed |
| MONITOR | Consume run events + sentinels; maintain per-row status; enforce budget guards; feed live views. | `probe` | row-failed / budget-exceeded |
| COLLECT | Pull artifacts, manifests, event logs; verify payload (summary present, JSON valid, manifest parity). | `collect` | collect-failed |
| CERTIFY | Produce the spec-conformance certificate (§4) from collected artifacts. | none | conformance-failed |
| TEARDOWN | Release compute per policy (§6). Runs even when earlier stages failed (best-effort, recorded). | `teardown` | teardown-failed (non-blocking) |
| REGISTER | Durable registration: tracked run specs, ledger run-status checkpoint, index updates. Project-specific registration (e.g. rlrmp `post_run.sh` semantics) plugs in here. | none | register-failed |

Failure taxonomy and retry policy: `provision`, `realize_env`, and `stage_inputs` failures are
retryable with bounded attempts (default 3) and are expected to be transient; `preflight`,
`conformance`, and spec-invalid failures are never retried automatically (they indicate a wrong
bundle, not a flaky environment); row training failures are surfaced, never auto-retried
(scientific runs must not silently re-execute); `collect` retries are safe and unbounded-ish
(default 5) because collection is read-only. Every retry is recorded in the state document with
timestamps and reasons.

### 2.1 Identity

`run_set_id`: `<utc-date>-<8-hex>` minted at ASSEMBLE (e.g. `20260709-1f3a9c2e`). `row_id`:
taken from the bundle's row labels, validated against `^[A-Za-z0-9_.-]+$` (the existing rows-
manifest constraint). The pair `(run_set_id, row_id)` is carried into: the state document, every
run event, `TrainingRunManifest.run_set_id`/`job_id`, checkpoint transaction metadata, and the
conformance certificate. Workers and executors never mint identity (supersedes the worker-side
`uuid4()`; the worker `/start` route gains a required `job_id` parameter).

### 2.2 PREFLIGHT contents

All of the following, each producing a named pass/fail entry in the state document:

1. Spec schema validation and migration currency for every spec payload in the bundle
   (via `feedbax.contracts.migrations`; unknown or stale versions fail here).
2. Manifest-payload normalization: construct the exact `TrainingRunManifest` spec-payload
   embedding the executor will write at run end — using the same payload-assembly function the
   executor uses, factored to be callable statically — and validate it. (Owned by issue
   `559f97a`; the orchestrator calls that gate.)
3. Schedule realization: for every row with a declared LR schedule and resume/fork context,
   build the schedule through the same optimizer-builder call path the executor will use
   (`feedbax.training.optimizers.build_optimizer` with the row's actual resume parameters) and
   check the realized learning rate at no fewer than three points (start, mid-warmup or
   mid-decay, terminal) against the declared schedule. A gate that recomputes the schedule
   through any other code path is non-conforming.
4. Input custody resolution: every source checkpoint resolves to a pinned transaction id whose
   manifest hash verifies; mutable `latest.json` pointers are resolved to transactions at
   ASSEMBLE and only the pinned form enters the bundle.
5. Environment declaration completeness (§5): lockfiles present and clean, repo revisions
   resolvable, overlay steps declared.
6. Driver preconditions: image tag exists (registry check), GPU class available or fallback
   policy declared, credentials present, balance floor satisfied.
7. Budget declaration present (§6).

PREFLIGHT is pure: no compute acquired, no remote mutation, no billable action.

### 2.3 Launch policy

Bundle-declared: `max_parallel_rows` (int), `warm_first` (bool, default true),
`stagger_seconds`. Warm-first launches one representative row and blocks fan-out until that
row's readiness event (§3) or its terminal success; a terminal failure of the warm row aborts
fan-out. Readiness never depends on log-regex scraping. Compile-class-aware scheduling
(launching one representative per compilation class; feedbax issue `738a221`) is a future
refinement of this policy, not a replacement.

## 3. Run-event protocol — `feedbax.run_event.v1`

### 3.1 Schema

One JSON object per event. Common envelope fields, all required unless noted:

| Field | Type | Meaning |
|---|---|---|
| `schema_id` | str | `"feedbax.run_event"` |
| `schema_version` | str | `"feedbax.run_event.v1"` |
| `run_set_id` | str | §2.1 |
| `row_id` | str | §2.1 |
| `seq` | int | Monotonic per `(run_set_id, row_id)`, starting at 0, no gaps from the emitter's perspective. |
| `emitted_at_ms` | int | Unix epoch milliseconds, emitter clock. |
| `type` | str | One of the types below. |
| `payload` | object | Type-specific; unknown payload keys must be preserved by consumers. |

Event types and required payload fields:

| `type` | Payload | Semantics |
|---|---|---|
| `ready` | `{phase}` | Executor initialized; JIT/tracing complete enough that training steps are executing. Releases warm-first fan-out. |
| `progress` | `{phase, batch, total_batches, loss?, loss_terms?, metrics?, elapsed_s}` | Periodic; cadence per emitter config (default every 10 batches; every batch when `total_batches <= 50`, matching the existing rlrmp convention). |
| `heartbeat` | `{last_seq}` | Emitted by a sidecar timer thread when no other event has been written for `heartbeat_seconds` (default 60). Distinguishes "slow step" from "hung process". |
| `checkpoint_written` | `{transaction_id, coordinate, batch}` | After each custody transaction publishes. `batch` is the true batch count; `coordinate` is the custody coordinate — the two are distinct by contract (see feedbax issue `6004265`). |
| `phase_changed` | `{phase, batch}` | Optional; multi-phase executors. |
| `complete` | `{batch, summary_metrics, manifest_id?}` | Terminal success. Exactly one per row. |
| `failed` | `{batch?, error, diagnostics?}` | Terminal failure. Exactly one per row. |

Versioning: the family is registered in `feedbax.contracts.migrations`
(`register_family(kind="RunEvent", ...)`); consumers reject versions they cannot migrate, with
a clear error. Adding an event type or optional payload field is a minor revision within v1;
changing envelope fields or required payload fields is v2 with a migration or explicit
rejection.

### 3.2 Transports

**Canonical: JSONL file.** The emitter appends events to
`<row_run_dir>/events/<row_id>.events.jsonl`, one event per line, line-buffered writes of
complete lines only (a single `write()` of `line + "\n"` on a line-buffered handle; no partial
lines). This works identically for detached remote rows (nohup), local subprocesses, and Studio
workers, and survives every process death by construction.

**Streaming rendering: worker SSE.** The Studio worker's event queue carries the same event
objects; the existing SSE route and `training_resync` reconnect logic are retained. The
existing worker vocabulary maps onto run events (`training_progress` → `progress`,
`training_complete` → `complete`, `training_error` → `failed`, `training_log` → `progress`
metadata or a dedicated `log` type if needed; `training_trajectory` remains a Studio-specific
extension type carried in the same envelope). The Studio api_transport normalization keeps its
own version and wraps run events rather than replacing them.

**Human rendering: BATCH lines.** `BATCH phase=<p> batch=<i>/<n> [loss=<x>] [elapsed=<s>s]`
lines are generated FROM `progress` events (same values, same cadence) for greppability and
backward compatibility with existing polling. They are a rendering, not a source of truth;
nothing in the orchestrator parses them.

### 3.3 Emission

`feedbax.orchestration.events` provides `RunEventEmitter` with methods mirroring the event
types (`emitter.progress(...)`, `emitter.complete(...)`, etc.), constructed from
`(run_set_id, row_id, events_path)`. The training executor accepts an emitter (generalizing the
existing `progress_callback` parameter of `execute_training_run_spec`); the worker constructs
one per job; detached rows construct one from environment variables set by `launch_row`
(`FEEDBAX_RUN_SET_ID`, `FEEDBAX_ROW_ID`, `FEEDBAX_RUN_EVENTS_DIR`). When the env vars are
absent (ad hoc local runs), the emitter degrades to BATCH-line rendering only, so existing
entry points keep working unbundled.

The emitter never raises into the training loop: I/O errors are counted, retried with backoff
on the sidecar thread, and surfaced as a warning in the run log.

### 3.4 Consumption

`feedbax.orchestration.events` also provides `RunEventReader` (tail a JSONL file from a seq
offset; validate envelope; yield typed events) used by: the MONITOR stage, the warm-first gate,
the status CLI, and the Studio live table (for detached rows). Sentinel reconciliation rule:
`.done`/`.failed` sentinels and `complete`/`failed` events must agree; when a sentinel exists
without its terminal event (executor died between the two writes), MONITOR synthesizes the
terminal status from the sentinel and records the discrepancy — a row is never left
indeterminate, and a completed row is never classified as failed on the basis of missing
progress output alone.

## 4. Spec-conformance certificate — `feedbax.run_conformance.v1`

Produced by CERTIFY for every run set; written next to the collected artifacts
(`<run_set_dir>/conformance.json`) and referenced from the REGISTER stage output. Schema
envelope: `schema_id="feedbax.run_conformance"`, `schema_version="feedbax.run_conformance.v1"`,
`run_set_id`, `generated_at`, `overall: "pass"|"fail"`, `rows: {row_id: {checks: [...]}}`.

Each check entry: `{check_id, status: "pass"|"fail"|"skipped", expected, observed, detail}`.
`skipped` requires a `detail` naming why the check does not apply; a check that cannot run
because inputs are missing is `fail`, not `skipped`.

Core checks (feedbax-owned):

| check_id | Verifies |
|---|---|
| `completed_batches` | Realized batch count equals the declared batch count. |
| `lr_trace` | Realized learning-rate trace (from training diagnostics) matches the declared schedule at ≥3 points (start / mid / terminal), tolerance 1e-6 relative, schedule constructed through the executor's own builder path with the row's actual resume context. |
| `seeds` | Realized seeds recorded in the manifest equal the declared seeds. |
| `checkpoint_cadence` | Custody transactions exist at the declared interval; final transaction coordinate consistent with the realized batch count. |
| `environment_fingerprint` | Fingerprint recorded at REALIZE_ENV equals the fingerprint recorded in the row's manifest. |
| `manifest_valid` | Final `TrainingRunManifest` loads, migrates, and matches the PREFLIGHT-normalized payload. |
| `events_terminal` | Event log ends in exactly one terminal event consistent with the sentinel. |

Project-specific checks register through the existing plugin mechanism (entry-point group
`feedbax.plugins`): a plugin may contribute check callables keyed by `check_id`, receiving the
collected row artifacts and the bundle row spec. Certificates are additive evidence: a failing
project check fails the certificate exactly like a failing core check.

Consequences: REGISTER refuses to emit a `phase=completed` ledger checkpoint for a run set
whose certificate is `fail`; it emits `phase=failed` with the certificate attached. Experiment
interpretation cites the certificate rather than re-deriving conformance ad hoc.

## 5. Hermetic environment realization

The bundle's environment declaration:

```json
{
  "python": "3.12",
  "repos": [{"name": "rlrmp", "revision": "<sha>", "dirty_allowed": false},
            {"name": "feedbax", "revision": "<sha>", "dirty_allowed": false}],
  "lockfiles": [{"path": "rlrmp/uv.lock", "sha256": "..."}],
  "overlay": [{"id": "cuda_jax", "command": "uv pip install -U 'jax[cuda12]'"}],
  "image": "runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204"
}
```

REALIZE_ENV builds the environment as a function of this declaration only: sync repos at the
declared revisions (dirty working trees are refused unless `dirty_allowed`, in which case the
dirt is captured as a patch artifact and recorded), `uv sync --frozen` against the declared
lockfile, then the declared overlay steps in order. The **environment fingerprint** is
`sha256` over: the resolved package list (`uv pip freeze` output, sorted), repo revisions (+
patch hashes when dirty), image identifier, and Python version. It is recorded in the state
document, exported to the row processes (`FEEDBAX_ENV_FINGERPRINT`), and stamped into
`TrainingRunManifest.metadata.environment_fingerprint` by the executor.

Failure semantics: if a reused target (pod, VM, local venv) fails a fingerprint probe, the
environment is rebuilt from scratch (the current probe-and-rebuild behavior, now keyed to the
fingerprint rather than an import smoke test); if rebuild cannot reach the declared
fingerprint, REALIZE_ENV fails before LAUNCH. The `uv run --no-sync` convention on pods remains
valid *after* REALIZE_ENV because the fingerprint pins what `--no-sync` preserves.

## 6. Budget and teardown guards

Bundle-declared budget: `max_wall_clock_seconds` per run set, optional per row;
`max_spend_usd` when the driver can price time (RunPod: hourly rate × elapsed since pod
creation — billing begins at creation, not container start). MONITOR enforces both: on breach
it stops rows (driver `stop_row` if available, else records the breach), runs COLLECT
best-effort, and proceeds to TEARDOWN with terminal state `budget-exceeded`.

Teardown policy: default `on_terminal` — when every row is terminal and COLLECT has verified
the payload, TEARDOWN releases the compute. A pod created by this run is removable when
`auto_teardown` is enabled; `keep_alive: true` or disabled `auto_teardown` records a skipped
teardown without removing it. A supplied pod ID and a supplied SSH endpoint are never owned or
removed by the run, regardless of either flag. `stop_after_stage` is a successful, resumable
pause: it does not trigger abort teardown when it stops before TEARDOWN. On exceptions,
`SystemExit`, SIGINT, or SIGTERM after an owned pod is observable, the run collects failure
evidence first and then performs bounded teardown. Success requires an exact provider query that
verifies the pod is absent; otherwise TEARDOWN durably records the pod ID, its last known state,
and the unresolved reason while preserving the primary failure. Main-thread signal handlers are
installed only for the run call and restored afterward; off the main thread this signal layer is
an intentional no-op. RunPod failure-log collection and provider teardown subprocesses have
finite timeouts, so handled signals are deferred only through that bounded cleanup window.

This observable-exit contract does not cover SIGKILL. Before each provider create invocation,
the engine durably records a name-tagged acquisition intent. A later local process or operator
must reconcile that intent if the creating process dies before recording the outcome or receives
an ambiguous response. Inventory absence alone never authorizes retry because an in-flight create
may materialize later. No process-level guarantee can make cleanup unkillable; the contract is
bounded best effort with either verified absence or durable unresolved-pod evidence.

Dead-man switch (optional, per-bundle, driver-level): drivers that bill for idle compute may
co-launch a watchdog process ON the compute target. The RunPod watchdog is installed over SSH
immediately after endpoint configuration and before GPU readiness. There is no autonomous
coverage between create and endpoint availability; SIGKILL in that window is covered only by the
durable intent plus a later local reconciliation or a human acting on its evidence. From
installation onward the watchdog watches the recency
of run-event/heartbeat output (event-file mtimes and row sentinels — signals the runners must
emit anyway) and, after a configurable silence window with no live rows, terminates the target
from the inside (RunPod: `runpodctl remove pod $RUNPOD_POD_ID` with in-pod credentials). This
failsafe holds even when the local orchestrator has crashed or lost connectivity, which
MONITOR-side budget enforcement cannot guarantee. The watchdog must refuse to fire while any
row is emitting, must log an imminent-termination warning into the run dir before firing, and
is disabled by `keep_alive`. Bundle fields: `deadman_enabled`, `deadman_silence_seconds`
(default 1800). Implemented per driver; specified here so every remote driver offers the same
policy surface.

## 7. Run-set state machine

State document: `run_set_state.json`, schema `feedbax.orchestration.run_set_state.v4`,
registered with the migrations registry. Location: `<orchestration_root>/<run_set_id>/`, where
`orchestration_root` defaults to `~/.cache/feedbax/orchestration/` and is overridable per
project (rlrmp: under `_artifacts/`). Writes are atomic (write temp, `os.replace`), following
the existing `OrchestrationState` precedent.

Contents: bundle digest and path; stage status map (`{stage: {status, started_at, ended_at,
attempts, outputs, error?}}`); per-row status (`pending → launched → ready → running →
completed|failed|stopped`, with seq high-water mark and last-event timestamp); provision record
(endpoint, driver, billing-start); environment fingerprint; budget counters; certificate
reference. The state document is operational state — the ledger still receives exactly one
terminal run-status checkpoint per run (existing convention), emitted by REGISTER.

Resume semantics: `feedbax-orchestrate resume --run-set <run_set_id>` reloads the document,
verifies the provision record against the driver (endpoint probe), reconciles per-row status
from sentinels + event logs (§3.4), and continues from the first non-completed stage. Any process
holding the document uses an advisory lockfile; a stale lock (dead pid) is breakable with an
explicit flag. Two MONITOR processes on one run set are refused.

Relationship to manifests (kept two-writer): a "planned" manifest (Studio staging) may exist
before ASSEMBLE completes; the executor's "completed" manifest is written per row at run end.
The state document references both and is authoritative for *orchestration* status only;
manifests remain authoritative for run *content*.

## 8. Backend driver interface

`feedbax.orchestration.drivers` defines a `Protocol`:

```python
class OrchestrationDriver(Protocol):
    name: str
    def provision(self, bundle: RunBundle, state: RunSetState) -> ProvisionRecord: ...
    def realize_env(self, bundle, state) -> EnvFingerprint: ...
    def stage_inputs(self, bundle, state) -> StagingRecord: ...
    def launch_row(self, bundle, state, row_id: str) -> LaunchRecord: ...
    def probe(self, state) -> ProbeReport: ...          # endpoint/process liveness, cheap
    def stop_row(self, state, row_id: str) -> None: ... # best-effort
    def collect(self, bundle, state, dest: Path) -> CollectRecord: ...
    def teardown(self, state) -> TeardownRecord: ...
```

All methods are synchronous and idempotent with respect to the state document (each checks
recorded outputs before acting). Drivers receive credentials/config via their own constructor
config objects; nothing above the driver layer touches provider APIs, SSH, or secrets.

**`local`** (wave 2): `provision` is a no-op record; `realize_env` verifies/builds the local
venv fingerprint; `launch_row` spawns a subprocess with the event env vars; `collect` is a
local copy; `teardown` terminates row processes only. This driver is also the test double for
the stage engine.

**`runpod`** (wave 2): extraction of the existing script logic — pod classify/acquire with DC
ranking and balance floor, SSH endpoint discovery/classification, rsync + literal path patching,
sentinel-launched rows — behind the driver methods, with behavior changes ONLY where this
contract requires them (readiness via events, teardown policy, fingerprint-based env probe).
The existing bash scripts become shims that call the CLI, then are deleted at parity.

**`modal`** (wave 3): bundle→image build, volume-based collect. Out of scope until the driver
interface has survived the runpod extraction.

## 9. Library, CLI, and Studio reconciliation

Package layout (new code):

```
feedbax/orchestration/
    __init__.py        # public API: assemble, preflight, launch, resume, status, certify
    bundle.py          # RunBundle: specs, rows, environment, budget, launch policy
    stages.py          # stage engine over the state machine (§2, §7)
    state.py           # RunSetState (de)serialization, locking, atomic writes
    events.py          # RunEventEmitter / RunEventReader (§3)
    conformance.py     # certificate framework + core checks (§4)
    drivers/
        base.py        # Protocol + records
        local.py
        runpod.py
feedbax/bin/orchestrate.py   # CLI: preflight | launch | status | watch | resume | collect | certify | teardown
```

CLI: `feedbax-orchestrate launch --assembly-request <path> --driver <driver>` starts a run;
`status --run-set <id>`, `watch --run-set <id>`, and `resume --run-set <id>` operate on an
existing run set. The CLI is a fifth `[project.scripts]` entry. `status` prints one
machine-readable line per row (stable field order) plus the stage map; `watch` follows events.
rlrmp keeps no orchestration logic of its own: `post_run.sh`'s registration semantics become
the rlrmp REGISTER plugin, and its sync/verify mechanics are absorbed by COLLECT.

Studio adoption: `TrainingService` delegates run lifecycle to `feedbax.orchestration` with the
`local` driver (or a remote worker driver wrapping the existing HTTP surface): job identity
comes from ASSEMBLE (§2.1); job state persists in the state document, making runs recoverable
across backend restarts (reconcile-on-startup replaces in-memory-only dicts); the live table
consumes run events (worker SSE for attached workers, JSONL tail via `RunEventReader` for
detached rows launched through any driver). The worker keeps its FastAPI surface; its `/start`
gains required `job_id`/`run_set_id` parameters and its event objects adopt the run-event
envelope. One orchestrator, two frontends (CLI, Studio), N drivers.

Known Studio-side defect recorded here for sequencing: a browser-WS reconnect does not resume
from the last delivered `seq` even though the worker SSE layer supports `from_seq` replay —
events during a WS-only gap are silently lost. The run-event `seq` makes the fix mechanical;
it is filed separately and does not block the orchestration core.

## 10. Migration waves

Wave 1 (open, umbrella `1c6293a`): stopgap fixes on the current scripts — warm-first gate
success path (`ae2eeae`), manifest preflight (`559f97a`), checkpoint-coordinate naming
(`6004265`), deploy residuals (`3b001e1`), import guard (`bc79f8c`) — plus this document.

Wave 2 (children specced with this document; unblocked by review):

| Child | Scope | Depends on |
|---|---|---|
| W2-1 | Run-event protocol: contract types, migrations registration, `RunEventEmitter`/`RunEventReader`, executor integration, BATCH-line rendering, worker envelope adoption. | — |
| W2-2 | Orchestration core: `RunBundle`, state machine + stage engine, PREFLIGHT composition, `local` driver, advisory locking, resume. | W2-1 |
| W2-3 | RunPod driver extraction (existing issue `0ab7d01`, scope refined): script logic behind the driver interface; scripts to shims; parity tests; retire rlrmp's interim `cbb5b66` realized-LR guard only after this migration and fail-closed certification both land. | W2-2 |
| W2-4 | Conformance certificate framework + core checks + plugin check registration. | W2-1 (events_terminal check), else independent of W2-2 internals |
| W2-5 | CLI `feedbax-orchestrate` + script shims + machine-readable status contract. | W2-2 |
| W2-6 | Studio adoption: external job identity, persisted job state, run-event envelope in the worker, reconcile-on-startup. | W2-1, W2-2 |

Wave 3 (enumerated, not filed): `modal` driver; browser-WS `from_seq` resume in the frontend;
compile-class launch policy (`738a221`) as a launch-policy plugin; rlrmp REGISTER plugin
replacing `post_run.sh` (requires W2-2 + W2-4 in production use first); retirement of
`poll_run.sh` once `feedbax-orchestrate status/watch` reaches parity.

Sequencing rule: wave-1 stopgaps must not be blocked on wave 2; wave-2 children land behind
this accepted document; no wave-3 work starts before W2-3 has run a real billable training run
end-to-end.

## 11. Out of scope

Training semantics (loss functions, schedules, executors' numerical behavior); rlrmp
experimental design; evaluation/analysis pipelines (`EvaluationRunManifest` consumers); the
Studio frontend beyond the integration points named in §9; multi-node/distributed training.

## 12. Resolved review questions (2026-07-08)

1. `orchestration_root` default: `~/.cache/feedbax/orchestration/`, with the per-project
   override (rlrmp places it under `_artifacts/`). RESOLVED as proposed.
2. Trajectory snapshots: run-event envelope type with the payload externalized to a sidecar
   file reference above a size threshold; worker SSE may inline small payloads. RESOLVED as
   proposed.
3. Budget on unpriceable drivers: wall-clock only. Additionally, the optional driver-level
   dead-man switch (§6) is the post-install failsafe layer for remote drivers: once installed,
   it survives loss of the local orchestrator by running on the compute target and keying off
   the signals runners must emit anyway. RESOLVED.
4. REGISTER stops at payload emission. feedbax stays ledger-agnostic (D11): no commit, auth,
   or issue-tracker action is baked into the orchestrator; those remain with sessions, and a
   user-configurable terminal-hook interface (through which, e.g., an external plugin could
   submit an auth request automatically at run end) is deferred to its own issue. RESOLVED.
