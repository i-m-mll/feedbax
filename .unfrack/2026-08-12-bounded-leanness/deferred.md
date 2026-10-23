# Deferred and owner-decision work

Date: 2026-08-12

Baseline: `develop` at `9455af0a`; accepted integration evidence at `31d9628c`

This file records concrete work deliberately held outside the bounded implementation
portfolio. A deferred entry is not a claim that the problem is unimportant. It means
the current evidence requires an owner decision, wider migration, shared-contract
sequencing, or separate custody action before implementation can be honest.

## Explicit owner decisions

### [issue:9c85879] — Studio persistence and cross-language schema admission

**Verified current fact:** save ingress, generated TypeScript constraints, and
non-finite GraphSpec scalar encoding do not yet share one versioned admission rule.

**Decision needed:** ratify accept/migrate/reject behavior and shared Python/TypeScript
vectors before changing durable Studio bytes. This work remains held.

### [issue:899b809] — remote-worker trust and credential boundary

**Verified current fact:** non-loopback authentication, arbitrary worker URL handling,
stream budgets, and secret-safe cloud diagnostics require one trust model.

**Decision needed:** ratify authentication, URL/redirect, credential transport,
timeout/byte budgets, and redaction while preserving a documented local path.

### [issue:eb2ee06] — document-scoped save acknowledgement

**Verified current fact:** high review rejected the caller-local generation patch; a
save acknowledgement must coordinate all document mutations and save callers.

**Decision needed:** design one persistence transaction across autosave, manual save,
tabs, undo/redo, create, reload, and conflict handling. The rejected commit is not in
the accepted integration tree.

### [issue:a33c72a] — cancellation, retry, and idempotency lifecycle

**Verified current fact:** manifest cancellation, live worker stop, restart
reconciliation, and figure rerun promises do not share one lifecycle authority.

**Decision needed:** define cancellation and retry semantics across API, durable state,
worker execution, and UI before implementing local state transitions.

### [issue:9d3b127] — durable process identity

**Verified current fact:** local and RunPod recovery use PID liveness as adoption
authority, and stop paths can later signal that PID or process group. PID reuse can
therefore bind to an unrelated process.

**Decision needed:** choose an identity proof such as PID plus launch token and
platform start identity, or fail closed when identity cannot be established. This is
broader than parsing or rejecting unusual PID values because crash recovery and stop
semantics change together.

### [issue:a77ba1c] — repo snapshot source authority

**Verified current fact:** snapshot sealing authenticates the copied result but does
not prove that each copied pathname stayed bound to the Git-observed source during the
copy.

**Decision needed:** choose descriptor-relative, no-follow source custody with
before/after identity checks, or choose a different Git/object authority. Dirty-tree
semantics and runtime provenance must be explicit.

### [issue:e0dbef3] — Analysis Canvas layout custody

**Verified current fact:** accepted [issue:058a7f5] work persists semantic deletion
but intentionally does not add node-position fields to the durable page schema. Drag
layout still disappears when the graph is reconstructed.

**Decision needed:** version and persist UI layout outside semantic graph identity, or
make the canvas deliberately non-draggable. Do not fold presentation coordinates into
the graph's semantic hash by accident.

### [issue:bb0105c] — timeline epoch-value semantics

**Verified current fact:** the frontend authors `epoch_value_specs`, while generated
TypeScript and backend contracts omit it and execution consumes only the top-level
value. The normal editor can also emit modes rejected by timeline-mask lowering.

**Decision needed:** define per-epoch values as executable contract with a versioned
migration, define them as preview-only persisted state, or reject the unsupported
authoring. Reconcile the historical claims in [issue:5977ece], [issue:6fb098d], and
[issue:bd5aefb].

## High-priority broader implementation held from this portfolio

### Backend event fanout and lifecycle

The worker event stream has competing consumers rather than broadcast delivery. A
durable persistence consumer and browser consumers can observe disjoint sequences.
Client clean EOF and swallowed failures can then hide or indefinitely retry the
incomplete stream.

This needs a dedicated design and implementation lane coordinated across
[issue:6c4dea0] and [issue:d63b0ac]. Acceptance must prove identical ordered delivery
to simultaneous consumers, gap-free replay/live handoff, terminal-event delivery,
bounded retention, explicit failure propagation, and deterministic shutdown. Queue
bounding alone is insufficient.

The accepted `1936b4f0` teardown-admission repair is intentionally narrower. It closes
stream admission during teardown but does not establish broadcast, replay, or manifest
cancellation semantics.

### Graph and model authority

- [issue:8247f87]: decide representable authored planar-multibody topology and build it
  literally, or explicitly reject it; do not replace it with a stock plant.
- [issue:afbdcfb]: establish registry-wide non-default GraphSpec round-trip evidence.
- [issue:2b9dac3]: derive build and serialization from one typed parameter declaration
  after the enforcement gate above.
- [issue:2f8dd61]: decide derived versus detached composite parameter semantics and
  provenance.
- [issue:63eb0a9]: retire `GRUOracle` only through durable component-identity migration.
- [issue:c539ba5] and [issue:8378254]: repair invariant-level tests and remaining silent
  substitutions rather than closing on symptom-based strict xfails.

### Identity, canonicalization, and secure-path consolidation

- [issue:0720161]: four canonical-JSON byte domains feed durable identity; convergence
  requires an algorithm/version and migration decision.
- [issue:f3e5928]: secure directory and containment primitives have different rigor and
  error tiers; consolidate only after preserving each authority boundary.
- [issue:c1f2e05]: complete the durable identity map from authored bytes through
  canonicalization and compilation to downstream assertion.

Authored hashes, raw-file hashes, canonical-document hashes, compiled hashes, and
runtime receipts remain separate. No deferred cleanup may substitute one for another.

### Product topology and persistent UI

The disconnected Figure Gallery, Trajectory Viewer, and Statistics Panel prove dead
frontend entrypoints, not dead backend routes or stores. Settings also contains
volatile or inert controls. Decide reconnect, implement, or remove at the product
level; do not infer backend deletion from frontend reachability alone.

## Cleanup scopes rejected as currently written

- [issue:b853bc6]: live rlrmp/rlrmp2 consumers invalidate part of the proposed deletion
  set. Split out only independently proved-safe bin cleanup or migrate consumers.
- [issue:07c8428]: live rlrmp imports invalidate a no-facade `contracts/graphs` move.
- [issue:529c401]: extensive rlrmp/rlrmp2 imports require preserved paths or coordinated
  migration during a manifest split.

These are not indefinitely blocked. They require an updated scope whose downstream
custody and migration plan matches `external-surface.md`.

## Tooling and admission decisions

- [issue:e84bf42]: known package Pyright defects must be resolved or baselined before
  replacing the misleading one-file CI probe with an honest ratchet.
- [issue:f93007c] and [issue:d136fad]: define effective runtime fingerprint and evidence
  before broadening green full-suite memo reuse.
- [issue:a33aa59]: align slow-marker policy and repeated CI selections using measured
  profiles rather than deleting fast-fail gates by inspection.
- [issue:6e1a36e] and [issue:651d381]: expand conformance and decide versionless analysis
  loader policy at explicit schema boundaries.

The host OOM/shared-gate incident is not deferred code work: it established the
operating constraint that broad or greater-than-2-GiB jobs serialize through
`/private/tmp/codex-large-test-gate/held`, with measured launch/no-start thresholds.
The Feedbax full suite remains pending and must run once against the stable integrated
and documented tree.

## Ledger closure candidates — no closure authorized here

**Verified current fact:** the following entries have evidence supporting a later
owner-authorized reconciliation batch. This synthesis does not close them.

- [issue:4f7014b] is already marked duplicate of closed [issue:b37cf00], whose
  `TypeAliasType` fix is on `develop`.
- [issue:52774c3] was superseded by protected TaskTrainer retirement.
- [issue:c92b547] and [issue:8e58557] are an exact-title duplicate pair;
  [issue:c92b547] is already fixed by closed [issue:558bb00].
- [issue:b3ce3ae], [issue:6de75cd], [issue:97cdf91], and [issue:f5ebdfb] were recorded
  by the umbrella as done by file absence and require current re-verification.
- [issue:1ebb38c], [issue:3ab8f9a], [issue:641cc18], [issue:7be9c5d], and
  [issue:4390d11] were recorded as delivered by the generic envelope/fulfillment seam
  and require body-by-body confirmation before closure.
- [issue:12e94bc] has been answered as not redundant. Preserve it as a tracking
  reference or dispose it only with explicit owner intent; do not close it as a
  duplicate.
- [issue:c539ba5] is not a closure candidate yet. Its commit reached `develop`, but the
  current tests do not enforce the stated invariant.

Recommended action: re-read each live report, confirm open lifecycle immediately
before mutation, then apply one explicit ledger-maintenance decision log. Auth
association or feature-branch implementation alone is never closure evidence.

## Stranded implemented work: [issue:1479f9c]

**Verified current fact on 2026-08-12:** worker status is done and the integration work
unit is resolved, but `integration/feedbax-studio-iab` has no auth request, was 1,112
commits behind `develop`, and retained five patch-distinct commits across nine Studio
files.

**Owner decision required:** either revalidate and reimplement the intended behavior on
current Studio, or explicitly retire the branch with reversible custody of every
unique tracked and untracked byte. Do not infer delivery from the resolved work unit,
and do not delete the old worktree or branch as routine cleanup.

## Branch and worktree maintenance

The survey named merged ancestor worktrees or branches `help`,
`feature/b74330d-census-p2`, `feature/b74330d-census-p6b`, `list`, and
`integration/feedbax-studio-live-preview` as maintenance candidates only. Route any
cleanup through [issue:ce0b823] after exact status and unique-byte inspection. This
document does not authorize removal.

Co-Authored-By: Codex (GPT-5) <codex@openai.com>
