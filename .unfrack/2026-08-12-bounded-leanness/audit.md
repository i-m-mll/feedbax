# Bounded leanness audit

Date: 2026-08-12

Baseline: protected `develop` at `9455af0a`

Accepted integration snapshot: `31d9628c1eb13c5314bb49f6a674e0a937b2df78`

Coordination issue: [issue:925f4ad]

Synthesis issue: [issue:f4237a7]

## Verdict and evidence vocabulary

The bounded portfolio is **accepted for the final integration gate** at
`31d9628c`. This is a high-review verdict on the integrated code and does not mean
that the broad suite, protected-branch auth, push, or protected merge has occurred.
Those steps remain pending and are owned by the umbrella coordinator.

- **Verified integrated fact** was checked against `31d9628c`, its Git history, or
  current Mandible evidence.
- **Survey evidence** is a read-only audit finding pinned to its named snapshot.
  File:line references in survey comments are historical evidence, not unqualified
  claims about `31d9628c`.
- **Held** means the evidence requires an owner decision, a durable migration, or a
  wider trust/lifecycle contract. It does not mean the finding is disproved.
- **Accepted** means focused review and the final integration re-review accepted the
  implementation. It does not substitute for the pending full suite.

## Method and audit waves

This synthesis consolidates three deliberately distinct read-only audit waves, the
bounded implementations selected from them, independent high reviews, repair cycles,
and the final cross-lane review.

### Wave 1: repository surface survey

Eight children surveyed the protected snapshot at `9455af0a`:

- [issue:282a55d] core/runtime construction and parsimony;
- [issue:003e59e] backend/API and worker lifecycle;
- [issue:ea5d63a] Studio state, API coupling, and duplication;
- [issue:de0b456] external and downstream compatibility;
- [issue:9e49002] security, custody, concurrency, and failure boundaries;
- [issue:bd9c94b] tests, tooling, performance, and dead stock;
- [issue:21d41a1] ledger and protected-history reconciliation; and
- [issue:17d40e2] bounded portfolio planning.

Wave 1 produced eleven accepted integrated lanes and one rejected implementation.
[issue:eb2ee06] was rejected because caller-local save generations could not provide
document-scoped acknowledgement across all save paths; it remains held rather than
being represented as partial success.

### Wave 2: negative-space audit

Eight independent children searched for omissions not covered by Wave 1:
[issue:35af65a], [issue:fc51915], [issue:e873c66], [issue:c80aeff],
[issue:858107a], [issue:492b92f], [issue:2452f1f], and [issue:14d822a].
They covered numerical edges, backend contract/error behavior, Studio interaction,
Python and frontend parsimony, packaging/release drift, test quality, and sensitive
trust boundaries. Broader findings were held instead of being converted into local
patches.

### Wave 3: cross-surface residual audit

Eight children audited the then-integrated head `0e26cf17` while subtracting the first
two waves' findings: [issue:2c384ec], [issue:a33c72a], [issue:f1c454c],
[issue:6c31db4], [issue:0e23b89], [issue:835771d], [issue:7f76266], and
[issue:cb0f373]. They traced schema parity, lifecycle state machines, resource
ownership, operational truth, efficiency, frontend concurrency, diagnostics, and
deletion proof. [issue:d5dbde2] synthesized the final bounded matrix.

No audit wave ran the broad suite. Static surveys did not claim runtime proof.

## Exact maintained-source metrics

CLOC 2.04 was run with identical baseline and final scopes:

- Python/library scope: `cloc feedbax`.
- Web scope: `cloc web/src web/package.json web/tsconfig.json
  web/tsconfig.node.json`.
- Combined scope: the union of those inputs.

The checked-in baseline receipts are the exact bytes copied from the umbrella
integration worktree and remain pinned to `9455af0a`. The final receipts were generated
from the live integration worktree at `31d9628c`. The baseline web scope included
`web/src/assets/logo.svg`; its accepted deletion is therefore reflected naturally in
the final language and file totals.

| surface | baseline code | final code | delta | baseline files | final files | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `feedbax/` | 160,732 | 161,139 | +407 | 436 | 438 | +2 |
| maintained web source | 68,428 | 69,355 | +927 | 211 | 214 | +3 |
| combined | 229,160 | 230,494 | +1,334 | 647 | 652 | +5 |

The portfolio intentionally does not claim net line reduction. Most added lines are
focused tests and fail-closed custody/concurrency checks. Leanness here means fewer
false authorities, duplicated decisions, silent wrong results, and misleading paved
roads, not a smaller raw total at any cost.

## Severity-ranked final findings

### Blocker resolved: final high review rejected three cross-lane defects

The first final high review rejected integration `4a365750` for three defects
attributable to the accumulated changes:

1. fresh run-set locks rejected the empty inode created by their own acquisition;
2. worker teardown could miss a stream admitted after its ownership snapshot; and
3. checkpoint eviction could race an already-authorized streaming download.

All three were repaired and independently accepted:

- `24c59947` makes fresh lock acquisition distinguish the inode created by the current
  `O_CREAT|O_EXCL` attempt while keeping pre-existing empty, malformed, or PID-less
  locks fail-closed; the final lock matrix passed 28 focused tests.
- `1936b4f0` closes stream admission atomically before teardown snapshots ownership,
  rejects driver reuse, and preserves repeat teardown of surviving streams; 23 focused
  tests passed and high review returned ACCEPT.
- `27308e71` completes the checkpoint download lease: request-scoped cleanup covers
  response streaming and pre-response cancellation, releases exactly once, and lets
  deferred eviction retry after lease release. Its final independent lifecycle review
  passed four focused cases and returned ACCEPT.

The final integration re-review at `31d9628c` returned **ACCEPT**: all three blockers
remain closed after merge composition and no accepted lane was overwritten. The tree
may proceed to, but has not yet passed, the single full integration gate.

### High held: downstream consumers invalidate three cleanup premises

The external survey found live rlrmp or rlrmp2 consumers that contradict the
zero-consumer or no-facade premises of [issue:b853bc6], [issue:07c8428], and
[issue:529c401]. These scopes remain invalid as written. The exact evidence and
preservation rules are in `external-surface.md`.

### High held: event fanout, cancellation, and process authority remain wider contracts

The worker event path still needs one ordered broadcast/replay authority before queue
bounding or browser resume can be called correct. Manifest cancellation, worker stop,
startup reconciliation, and analysis task shutdown still need one explicit lifecycle
contract under [issue:6c4dea0], [issue:d63b0ac], [issue:a33c72a], and related product
ownership. The accepted `1936b4f0` repair is deliberately limited to HTTP stream
teardown admission and does not claim to solve those wider contracts.

PID reuse and concurrent repository snapshot authority remain held under
[issue:9d3b127] and [issue:a77ba1c]. The accepted path, lock, and checkpoint custody
repairs do not invent process identity or source-authentication rules.

### High held: durable Studio and cross-language schema decisions

[issue:9c85879] holds Studio version admission, Python/TypeScript constraint parity,
and non-finite GraphSpec scalar encoding. [issue:eb2ee06] holds document-scoped save
acknowledgement. [issue:e0dbef3] and [issue:bb0105c] hold Analysis layout custody and
timeline epoch-value semantics. These are persistence/schema decisions and were not
implemented by local validation patches.

### High held: remote-worker trust boundary

[issue:899b809] holds non-loopback worker authentication, URL/redirect/stream budgets,
credential transport, and secret-safe cloud diagnostics. The bounded worker-stream
teardown work does not ratify remote-worker trust.

### Medium: ledger and maintenance remain separate owner actions

The survey found done-but-open, duplicate, superseded, and stranded-work candidates,
including [issue:1479f9c]. This synthesis does not close issues or delete branches or
worktrees. `deferred.md` records the exact decision boundary.

## Accepted integrated portfolio

### Wave 1 bounded implementations

The following Wave 1 lanes are accepted and present in `31d9628c`:

| issue | accepted result |
| --- | --- |
| [issue:686aae9] | repair the import-dead analysis activity module without broad retirement |
| [issue:ddb7cf0] | preserve supported Plotly RGB forms and validate alpha fills |
| [issue:dc1d4ba] | consolidate figure MIME labeling while preserving the six-format contract |
| [issue:19bd70b] | correct the RunPod authority note without response-schema change |
| [issue:270b176] | reject non-scalar rollout inputs shorter than `n_steps` |
| [issue:058a7f5] | persist semantic Analysis deletion and close adjacent interaction holes |
| [issue:4bb08be] | align worktree frontend sync with npm lockfile authority |
| [issue:30bcc37] | remove fail-open task-component collection and dead fixture stock |
| [issue:593742e] | confine worker job IDs and checkpoint output paths |
| [issue:2e5d1e0] | harden reserve publication and run-set lock ownership, including `24c59947` |
| [issue:c6d1181] | synchronize downstream-policy ratification and evidence domains |

[issue:eb2ee06] is not in this table because its implementation was rejected and not
integrated.

### Final bounded implementation wave

The five planner-selected lanes are accepted and integrated:

| issue | accepted commits and result |
| --- | --- |
| [issue:bc20ddd] | `d0d7329c`, `147f61d0`: validate evaluation parents, return canonical empty evaluation lists, and preserve the real SQLite legacy fallback |
| [issue:6c4dea0] | `0fedd13d`, `f4cfe19a`, `1936b4f0`: bounded HTTP stream teardown, race/failure repair, and teardown admission closure |
| [issue:8eb9492] | `97dac046`, `fc4fa9b4`: bind figure polling to request identity and terminate empty-result completions with an actionable error |
| [issue:bc88134] | `efa68e83`, `27a5eeec`: ignore stale evaluation loads and stale error feedback using selection epochs |
| [issue:71a61ca] | `4a088fd5`, `7e923068`, `30dbbb15`: correct orchestration commands and current/historical guidance, then remove the proven-unused Studio logo and stale ignore rule |

The final review additionally admitted checkpoint-download custody under
[issue:55a527b] through `7a12f594` and accepted repair `27308e71`.

## OOM and shared-gate incident

During maximal audit fanout the host crashed from memory pressure. The post-restart
inventory found no active Feedbax full suite; aggregate task overhead, especially 117
idle per-task Playwright MCP processes, was the plausible dominant contributor and had
driven aggregate RSS to roughly 53 GiB. The coordinators stopped fanout, retired idle
helpers, reduced work to measured batches, and established the shared machine gate at
`/private/tmp/codex-large-test-gate/held` for broad or greater-than-2-GiB work, with a
32-GiB launch threshold and 36-GiB recursive Codex-child no-start ceiling.

This incident is operational evidence, not a test result. It explains why focused
work was serialized and why the final broad suite is deliberately deferred until the
documented tree is stable. No broad Feedbax gate is claimed in this packet.

## Remaining pre-auth sequence

1. Integrate this synthesis commit into the umbrella integration branch.
2. Confirm the documented head and tested head are identical.
3. Acquire the shared heavy-test gate and run `scripts/full_suite.sh` once.
4. If the tree changes in a way that can affect the gate, update receipts and rerun only
   as required by policy.
5. Reconcile issue closeouts and run the Mandible independence/auth checks.
6. Submit one protected-branch auth request for the umbrella integration branch.

At this snapshot, steps 1 through 6 are pending. This document does not claim auth,
push, protected merge, or full-suite success.

## Identity preservation rule

No cleanup may collapse these authorities:

1. authored identity and authored bytes;
2. raw stored or transferred bytes and their raw-file hash;
3. canonical document bytes and the algorithm/version producing their hash;
4. compiled realization bytes or structure and its compiled hash; and
5. runtime receipt identity describing what actually executed.

Equality in one domain is not evidence of equality in another. Compression settings,
canonical JSON changes, schema renames, file moves, and compiler refactors must state
which identity changes and which remains stable.

Co-Authored-By: Codex (GPT-5) <codex@openai.com>
