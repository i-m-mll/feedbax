# ASSEMBLE Contract: Authored Intent to Executable Run Bundle

Status: proposed contract for feedbax issue `5afbdfb`, for coordinator review before
implementation.

This document closes the missing boundary between authored training intent and the
`RunBundle` consumed by orchestration. It refines, and does not replace,
`docs/design/TRAINING_RUN_LIFECYCLE.md`.

## 1. Decision

The orchestration engine accepts a typed, durable `RunAssemblyRequest`. Its persisted
ASSEMBLE stage resolves and compiles that request into `RunBundle` v3. ASSEMBLE, not a
caller or worker, mints `run_set_id`; the compiler supplies validated row labels, which
become `row_id`s. The completed stage records the request, compiled bundle, custody
references, and their hashes.

This is the only normal launch path. A caller must not compile a bundle and then ask the
engine to pretend that ASSEMBLE already completed.

### 1.1 Why this strengthens the accepted lifecycle

The decision makes the implementation match the lifecycle contract:

- D2 assigns identity minting to the orchestrator at bundle-assembly time. Keeping
  compilation inside ASSEMBLE makes the minted identity, the authored input, and the
  compiled output part of one durable transition.
- D3 requires every stage to be persisted, idempotent, and resumable. An ASSEMBLE failure
  is therefore visible in run-set state and can be re-entered without reconstructing work
  performed by an unrecorded caller.
- D5 requires independent evidence that realized behavior matched declared intent. The
  immutable compiled-row envelope written by ASSEMBLE supplies expected identity; the
  final `TrainingRunManifest`, emitted by the executor, supplies independently produced
  observed identity.
- Mutable input locators such as `latest.json` are resolved and pinned inside ASSEMBLE,
  as required by lifecycle PREFLIGHT rule 4. Only immutable identities enter the bundle.
- Studio and the CLI converge on the same identity-minting and assembly path instead of
  preserving a second hand-built bundle path.

### 1.2 Rejected alternative

Rejected: compile `RunBundle` before entering the engine, then import it with ASSEMBLE
already marked completed.

That alternative splits lifecycle authority. Compiler failures, mutable-reference
resolution, identity minting, and custody emission occur outside the resumable state
machine. A fresh process cannot re-enter or verify the first stage without a second
protocol duplicating ASSEMBLE provenance. It also makes D5 easier to satisfy vacuously by
comparing a final manifest only with values copied from a caller-built bundle. A future
precompiled-bundle import, if ever needed, must be a separately named, fail-closed import
mode that verifies a complete ASSEMBLE evidence record; it is not part of this contract.

## 2. Contract boundaries

The boundary has three durable records:

| Record | Lifecycle role | Contains | Must not contain |
|---|---|---|---|
| `RunAssemblyRequest` | Authored input to ASSEMBLE | Authored artifact reference, compiler identity, unresolved input declarations, orchestration and launch policy | Compiled row payloads or `TrainingRunMatrixSpec` copied onto rows |
| `RunBundle` v3 | Output of ASSEMBLE and input to later stages | Per-row execution envelope and separate launch instructions; bundle-wide environment, launch, budget, and custody policy | Opaque `run_spec`; mutable input pointers; undeclared architecture synthesis |
| `TrainingRunManifest` | Executor-emitted realized record | Final run identity, spec payloads, artifacts, and execution-identity fields | Authority to redefine identities fixed by ASSEMBLE |

`TrainingRunMatrixSpec` and `FlattenedIntent` are authoring/compiler inputs. They may be
referenced by `RunAssemblyRequest` and consumed by a registered compiler, but they are
explicitly rejected from `RunRowSpec`. After compilation, a row is Feedbax-generic: it
names a registered executable payload and the evidence binding that payload to authored
intent and resolved semantics.

Launch mechanics are separate from scientific execution identity. Commands, entry
points, collection globs, backend choice, environment realization, concurrency,
warm-first policy, budget, dead-man policy, and keep-alive policy do not enter
`execution_hash`. If one of those values is scientifically identity-bearing, the compiler
must declare it as an immutable input or include it in the registered executable payload;
implicit coupling is forbidden.

## 3. Authored assembly request

Add a governed spec family:

- schema identity: `feedbax.spec.run_assembly_request`
- current version: `feedbax.spec.run_assembly_request.v1`
- owner: `feedbax.orchestration.assembly`
- old-version policy: explicit rejection of v0 until a deterministic migration exists

The public model is `RunAssemblyRequest`:

| Field | Contract |
|---|---|
| `schema_id`, `schema_version` | Exact governed family identity above. |
| `authored` | `SchemaArtifactRef` for the authored document: payload schema ID/version, registered artifact reference, and canonical content SHA-256. |
| `compiler` | Stable `compiler_id` and `compiler_version`; dispatch is explicit, never inferred from payload contents. |
| `inputs` | Input declarations that ASSEMBLE resolves to immutable identities. Mutable locators may appear here, but not in `RunBundle`. |
| `driver`, `environment`, `launch_policy`, `budget` | Bundle-wide operational policy passed through after validation. |
| `orchestration_root`, `keep_alive`, `deadman_*` | Lifecycle policy, excluded from scientific identity. |
| `metadata` | Non-authoritative annotations only. Compilers must not read it to change executable semantics. |

`SchemaArtifactRef` is stricter than the current permissive `ArtifactRef`. It requires:

- the referenced payload's `schema_id` and `schema_version`;
- a registered `artifact_id` or other registry-resolvable immutable locator;
- `sha256` for the referenced bytes; and
- optional materialization URI as a locator only, excluded from identity.

ASSEMBLE verifies the bytes against `sha256`, migrates the payload through
`SpecSchemaRegistry`, validates the current typed model, and selects the compiler by
`(schema_id, compiler_id, compiler_version)`.

### 3.1 Assembly API

Add `feedbax/orchestration/assembly.py` with these public surfaces:

```python
class CompiledExecutionRow(StrictModel):
    row_id: str
    payload: RegisteredSchemaPayload
    resolved_semantics: dict[str, JsonValue]
    immutable_inputs: list[ImmutableInputIdentity]
    launch: RowLaunchSpec


class CompiledRunSet(StrictModel):
    rows: list[CompiledExecutionRow]


class ExecutionIdentityAdapter(Protocol):
    def intent_hash(self, authored: RegisteredSchemaPayload) -> str: ...
    def build_capsule(self, row: CompiledExecutionRow, *, identities: RowIdentities,
                      context: AssemblyContext) -> RegisteredSchemaPayload: ...
    def capsule_identities(self, capsule: RegisteredSchemaPayload) -> RowIdentities: ...


class AssemblyCompiler(Protocol):
    def compile(
        self,
        request: RunAssemblyRequest,
        *,
        authored: RegisteredSchemaPayload,
        run_set_id: str,
        context: AssemblyContext,
    ) -> CompiledRunSet: ...


class AssemblyCompilerRegistry: ...


def persist_compiled_row(
    row: CompiledExecutionRow,
    *,
    authored: RegisteredSchemaPayload,
    identity_adapter: ExecutionIdentityAdapter,
    context: AssemblyContext,
) -> CompiledRowStorageResult: ...


def assemble_run_bundle(
    request: RunAssemblyRequest,
    *,
    run_set_id: str,
    context: AssemblyContext,
    registry: AssemblyCompilerRegistry,
) -> RunBundle: ...
```

`AssemblyContext` carries machine-local resolvers and roots needed to materialize
artifacts. It is not serialized into the request and is not part of scientific identity.
The compiler registry rejects missing, duplicate, or ambiguous registrations.

This intermediate boundary is deliberate: a compiler resolves authored semantics into
typed in-memory rows, while `assemble_run_bundle()` alone owns custody writes and final
envelope/`RunBundle` construction. A compiler must not return artifact references that
have not been registered by ASSEMBLE.

The registry entry also supplies an `ExecutionIdentityAdapter` for the authored/capsule
schema families. This is the family-specific semantic hook needed by a generic assembler:
it computes authored intent identity, builds a capsule after generic row/snapshot/input
identities exist, and extracts capsule identities for independent validation. The default
training adapter delegates to the existing training spec-storage hash and capsule APIs.

The default training compiler accepts the registered `TrainingRunMatrixSpec` family,
uses the existing migration and materialization APIs, and returns a generic
`CompiledRunSet`. No orchestration module imports RLRMP or other project-owned scientific
types.

## 4. Persisted ASSEMBLE transition

New launches use `StageEngine.from_request(...)`; resumption uses the already persisted
request, state, and compiled bundle. The current public path that requires callers to pass
`bundle=...` is retired for new launches.

ASSEMBLE executes in this order:

1. Mint `run_set_id` and initialize run-set state with an empty row map and ASSEMBLE
   marked running. A failed compilation therefore has durable identity and diagnostics.
2. Canonically persist `assembly-request.json`; record its SHA-256 in ASSEMBLE outputs.
3. Resolve and verify the authored artifact and every input declaration.
4. Pass the exact verified, migrated authored payload from step 3 to the explicitly
   registered compiler and produce `CompiledRunSet`—typed rows without custody
   references. The compiler never resolves the authored reference a second time.
5. For each row, canonicalize inputs, build and validate its resolved-semantics snapshot,
   ask the registered identity adapter for its capsule, then persist the compiled payload,
   snapshot, and capsule through content-addressed custody.
6. Validate `RunBundle` v3, populate state rows from the compiled row IDs, and persist
   `bundle.json` atomically.
7. Mark ASSEMBLE completed with request/bundle hashes and all emitted artifact refs.

Re-entry with the same stored run set reuses its `run_set_id`. Content-addressed writes
are idempotent, and a completed ASSEMBLE verifies recorded request and bundle hashes
before becoming a no-op. A new launch of byte-identical intent mints a new run-set
identity; identity reuse occurs only through explicit resume.

Later stages require a completed ASSEMBLE and load its persisted bundle. Driver creation
therefore becomes a `driver_factory(bundle)` step after ASSEMBLE, rather than requiring a
fully selected driver before compilation. PREFLIGHT continues to be non-billable and now
validates every referenced envelope artifact before PROVISION.

## 5. Compiled-row execution envelope

Add a governed spec family:

- schema identity: `feedbax.spec.execution_identity_envelope`
- current version: `feedbax.spec.execution_identity_envelope.v1`
- owner: `feedbax.orchestration.bundle`

`RunBundle` advances to v3. `RunRowSpec.run_spec` is removed and replaced by a required
`execution: ExecutionIdentityEnvelope`. Row launch values move under a required
`launch: RowLaunchSpec`. This nesting makes the scientific/operational boundary
structural instead of conventional.

### 5.1 Envelope fields

`ExecutionIdentityEnvelope` contains exactly these identity-bearing fields:

| Field | Type and meaning |
|---|---|
| `schema_id`, `schema_version` | The envelope family and version above. |
| `payload` | `SchemaArtifactRef` to the registered executable row payload. The payload's own schema identity/version determines how executors validate it. |
| `authored_intent` | `IdentityArtifactRef` to the authored document plus `intent_hash`. For a matrix compiler this is the existing matrix-wide `training_run_intent_hash`; rows from one authored matrix intentionally share it. |
| `resolved_snapshot` | `IdentityArtifactRef` to a row-specific resolved-semantics snapshot plus `root_hash`. The snapshot contains only that row's executable resolved tree, not the whole run set. |
| `execution_capsule` | `IdentityArtifactRef` to a registered capsule plus `execution_hash`. The default training compiler emits `TrainingRunExecutionCapsule` v1 per row. The envelope does not require that capsule class, so other Feedbax-owned compilers may register another capsule family. |
| `immutable_inputs` | Typed, canonical records for every external input that affects execution. An explicit empty list is valid. |

`IdentityArtifactRef` requires a resolvable artifact identity and byte SHA-256. The named
semantic hash (`intent_hash`, `root_hash`, or `execution_hash`) is distinct from the
artifact byte hash and both are verified.

Each `ImmutableInputIdentity` has:

- `role`: stable consumer-facing role;
- `kind`: namespaced identity kind such as checkpoint transaction, dataset artifact, or
  source revision;
- `identifier`: immutable provider/registry identifier;
- `digest`: algorithm (`sha256` in v1) and lowercase digest value; and
- optional `schema_id` and `schema_version` when the input is a structured record.

ASSEMBLE sorts these records canonically by `(role, kind, identifier, digest)` and rejects
duplicate keys with different content. The sorted JSON projection is supplied to
the existing `training_run_execution_hash(root_hash, input_data_identities)` function.
Thus v1 execution identity is deterministic without changing that hash algorithm.

The shared API is
`canonicalize_immutable_input_identities(...) -> list[dict[str, JsonValue]]`. ASSEMBLE,
capsule construction, executor manifest emission, and CERTIFY all use this exact
projection; no consumer maintains a parallel normalization rule.

`execution_hash` is always recomputed and must equal the capsule value:

```text
sha256(canonical_json({
  "resolved_root_hash": resolved_snapshot.root_hash,
  "input_data_identities": canonical_immutable_inputs
}))
```

`RowLaunchSpec` owns `command`, `entry`, `collect`, launch-payload routing, and
non-authoritative launch metadata. The registered executable payload is never smuggled
through launch metadata.

### 5.2 Cross-artifact binding invariants

Artifact byte hashes prove custody, not semantic linkage. ASSEMBLE and PREFLIGHT also
enforce these invariants through the registered `ExecutionIdentityAdapter`:

1. Recompute `intent_hash` from the authored artifact using its registered family rule;
   require it to equal `authored_intent.intent_hash`.
2. Decode the resolved snapshot, recompute every node hash and its root hash, and require
   the result to equal `resolved_snapshot.root_hash`.
3. Extract capsule identity and require its intent hash, resolved root, and canonical
   immutable inputs to equal the other three envelope fields.
4. Recompute `execution_hash` from the verified root and inputs and require both the
   capsule and envelope values to equal it.
5. Verify the registered executable payload's byte hash and schema identity/version.

An adapter may define how a schema family derives authored intent, but it may not weaken
the required root/input/execution equalities. Missing family adapters fail ASSEMBLE; there
is no generic "hash the whole dict" fallback.

### 5.3 Training compiler/storage changes

The current `emit_training_run_spec_storage()` writes one resolved snapshot and one
capsule for the entire materialized run set. Attaching those records to every row would
not satisfy this contract.

Refactor `feedbax.training.spec_storage` into:

- pure authored/matrix intent hashing and training identity-adapter helpers; and
- `compile_training_run_matrix(...) -> CompiledRunSet`, which returns typed row payloads,
  resolved semantics, immutable inputs, and launch declarations without writing custody.

`feedbax.orchestration.assembly.persist_compiled_row(...) -> CompiledRowStorageResult`
then performs the generic content-addressed writes and supplies all references and hashes
needed to build `ExecutionIdentityEnvelope`. The existing run-set storage result may
remain as an aggregate convenience, but it must be derived from these row results rather
than reused as row evidence. The compiler and identity adapter remain pure with respect
to custody; ASSEMBLE owns the writes.

Every compiled payload must declare a schema family/version registered in
`SpecSchemaRegistry`. Opaque mappings and strings fail ASSEMBLE.

## 6. Core execution-identity conformance

Add `execution_identity` as the eighth check in
`build_core_check_registry()`. It is Feedbax-owned and cannot be disabled by choosing
`include_plugins=False`.

`ConformanceRowArtifacts` receives the typed envelope directly from the persisted bundle.
CERTIFY must not reconstruct it from launch metadata, `run_spec`, or the final manifest.

The check uses this expected/observed mapping:

| Expected from compiled envelope | Observed from final `TrainingRunManifest` |
|---|---|
| `authored_intent.intent_hash` | `intent_hash` |
| `resolved_snapshot.root_hash` | `resolved_semantics_root_hash` |
| `execution_capsule.execution_hash` | `execution_hash` |
| canonical `immutable_inputs` | canonical `input_data_identities` |

Before comparing, the check:

1. requires the typed envelope and a final manifest;
2. dereferences the payload, authored intent, snapshot, and capsule, verifies their byte
   hashes, validates their registered schema identities/versions, and reruns the
   cross-artifact binding invariants from §5.2;
3. inspects the raw manifest mapping and requires every identity key before model
   validation—missing values fail rather than being filled by
   `TrainingRunManifest` defaults or its derived-hash validator;
4. recomputes the envelope execution hash from the resolved root and immutable inputs;
5. loads/migrates/validates the manifest as `TrainingRunManifest`; and
6. compares all four identity values exactly.

An explicit empty immutable-input list is present evidence and may pass. `None == None`,
missing envelope fields, an unavailable artifact, a byte-digest mismatch, an unknown
schema, or a skipped comparison can never pass.

A non-vacuous pass means ASSEMBLE committed before launch to the authored intent, exact
row semantics, immutable inputs, and their derived execution hash; an independently
emitted final manifest repeated those values; and the certificate contains populated
`expected` and `observed` maps. A field-level mismatch map is recorded in `detail` on
failure. `feedbax.run_conformance.v1` need not change because its existing schema permits
additive check IDs and evidence values.

REGISTER retains its existing rule: a failing certificate cannot register the run set as
completed.

## 7. Acceptance tests

### 7.1 Fake-driver end-to-end proof

Extend `tests/test_orchestration_core.py` with a purpose-built identity fake driver. The
test must not derive the emitted manifest by copying the envelope at runtime.

The test:

1. writes independent authored-intent, executable-payload, resolved-snapshot, and capsule
   fixtures under `tmp_path`, each with fixed expected hashes;
2. submits a `RunAssemblyRequest` through a registered fixture compiler;
3. lets ASSEMBLE produce the typed row envelope and bundle;
4. has the fake driver emit all evidence required by the seven existing checks: terminal
   events, completed-batch/LR/seed diagnostics, checkpoint custody records, environment
   fingerprint, PREFLIGHT-normalized payload, and a valid `TrainingRunManifest` built
   from separately supplied constants;
5. runs the complete engine with `build_default_check_registry(include_plugins=False)`;
6. asserts CERTIFY and REGISTER complete, the certificate is `pass`, and
   the exact eight core check IDs are present, with `execution_identity.expected` and
   `.observed` populated and equal; and
7. repeats with a valid manifest carrying a different `intent_hash`, asserting CERTIFY
   fails with that field named and REGISTER does not complete.

The tampered fixture's constants must be changed independently; the test fails review if
the fake executor reads the envelope and reflects its values back.

### 7.2 Focused conformance and migration tests

Extend `tests/test_run_conformance.py` for:

- missing envelope and missing manifest identity fields;
- authored-intent mismatch;
- resolved-root drift with a correspondingly recomputed manifest execution hash;
- immutable-input drift with a correspondingly recomputed manifest execution hash;
- execution-hash inconsistency;
- artifact byte-digest or schema-version failure; and
- exact explicit-empty-input pass.

Extend `tests/test_structured_spec_migrations.py` for both new schema families, provider
emitter-policy inventory, current-version acceptance, v0 rejection, and RunBundle old-
version behavior. Extend `tests/test_training_jobs.py` for the Studio cutover.

## 8. Exact implementation surfaces

The implementation pass is expected to touch these modules and APIs:

| Module | Required change |
|---|---|
| `feedbax/orchestration/assembly.py` (new) | `RunAssemblyRequest`, strict artifact/identity refs, input declarations, `CompiledExecutionRow`/`CompiledRunSet`, `AssemblyContext`, `ExecutionIdentityAdapter`, `AssemblyCompiler`, registries, `persist_compiled_row`, and `assemble_run_bundle`. |
| `feedbax/orchestration/bundle.py` | RunBundle v3; `ExecutionIdentityEnvelope`, typed immutable inputs, `RowLaunchSpec`; remove `run_spec`. |
| `feedbax/orchestration/stages.py` | Request-based engine construction; real persisted ASSEMBLE; post-assembly driver factory; envelope-based PREFLIGHT and CERTIFY inputs. |
| `feedbax/orchestration/conformance.py` | `ConformanceRowArtifacts.execution`, `check_execution_identity`, eighth core-check registration, field-level evidence. |
| `feedbax/orchestration/__init__.py` | Export the public assembly, envelope, and conformance APIs. |
| `feedbax/contracts/spec_storage.py` | Shared strict identity helpers and `canonicalize_immutable_input_identities`; retain the existing execution-hash definition. |
| `feedbax/training/spec_storage.py` | Factor the current run-set emitter into pure matrix/row compilation, intent hashing, and the default training identity adapter; no compiler-owned custody writes. |
| `feedbax/contracts/migrations.py` | Register both new families and RunBundle v3 old-version policy. |
| `feedbax/bin/orchestrate.py` | New launches use `--assembly-request`; resume/status load persisted state/bundle. Remove normal-launch reliance on `--bundle`. |
| `feedbax/contracts/studio_training.py` (new) | Registered `StudioTrainingAssemblySpec` authored/payload family and its one-row compiler/identity adapter contract. |
| `feedbax/web/services/training_service.py` | Replace `_build_worker_bundle` with assembly-request construction and read status from typed payload/state. |
| `feedbax/web/services/worker_driver.py` | Resolve the registered execution payload; remove `worker_start` from row metadata. |
| `tests/test_orchestration_core.py` | Assembly lifecycle and non-vacuous fake-driver end-to-end proof. |
| `tests/test_run_conformance.py` | Identity-check pass/failure matrix. |
| `tests/test_orchestration_cli.py` | Request launch, resume, and rejection behavior. |
| `tests/test_structured_spec_migrations.py` | Schema registration and version policy. |
| `tests/test_training_jobs.py` | Studio service/worker payload migration. |

The compiler may continue to use `TrainingRunMatrixSpec`,
`materialize_run_matrix()`/`materialize_adapted_run_matrix()`, and existing spec-storage
hash functions internally. Drivers receive only the compiled generic row contract.

## 9. Migration and compatibility

### 9.1 RunBundle

RunBundle v1/v2 rows cannot be upgraded truthfully: optional opaque `run_spec` and launch
metadata do not contain enough evidence to reconstruct authored, resolved, capsule, and
immutable-input identities. The migration registry must explicitly reject execution of
v1/v2 with a clear "reassemble from authored request" diagnostic. It must not synthesize
hashes or wrap `run_spec` in a nominal envelope.

Persisted run sets that completed under v2 remain historical artifacts; this contract does
not reinterpret their certificates as having passed `execution_identity`.

### 9.2 Studio

The Studio backend currently hand-builds a minimal bundle and stores the worker `/start`
body in `row.metadata["worker_start"]`. That path carries no scientific execution identity
and must be replaced atomically with the assembler cutover.

Add `StudioTrainingAssemblySpec` in `feedbax.contracts.studio_training`, registered as
`feedbax.spec.studio.training_assembly.v1`, plus compiler identity
`feedbax.studio.worker.v1`. `TrainingService` normalizes the current `/start` body into
that governed authored document, persists it, and places its `SchemaArtifactRef` in a
`RunAssemblyRequest`.

The one-row Studio compiler validates that document and returns `CompiledExecutionRow`
with the normalized worker request as its registered executable payload, the fully
defaulted/resolved request as row semantics, launch routing for `WorkerHttpDriver`, and
immutable identities extracted from checkpoint and other artifact inputs. Its identity
adapter hashes the governed authored document, builds the same per-row training execution
capsule used by the default training compiler, and validates all §5.2 bindings. Authored
and payload refs may point to identical bytes when no lowering changed them, but they are
separate semantic roles and remain independently validated.

`WorkerHttpDriver` resolves and validates the payload before sending it. Status reads
`total_batches` from the typed payload or normalized state, never launch metadata. The
Studio HTTP/WebSocket UI contract is out of scope and need not change.

Existing Studio v2 bundles cannot be made conformant by manufacturing hashes from
`worker_start`. They are recreated for execution. Whether a read-only legacy status loader
is worth retaining is an open decision below.

## 10. Non-goals

- RLRMP adoption, storage cutover, or KPI/anti-reaccretion gates.
- Carrying RLRMP scientific payload types in orchestration contracts.
- Modal driver work.
- Studio UI changes.
- Changing scientific defaults or `TrainingRunMatrixSpec` semantics.
- Treating launch commands, backend selection, or environment mechanics as scientific
  identity unless explicitly declared by a compiler.

## 11. Coordinator decisions still required

1. **Legacy Studio visibility.** Recommended: reject v1/v2 for launch but permit a small,
   read-only v2 loader for status/history until existing local state ages out. Alternative:
   reject all v2 loading and require recreation.
2. **Precompiled import mode.** Recommended: omit it from the implementation pass. If a
   real consumer requires it, design a separate verified-import contract with a complete
   ASSEMBLE evidence record; do not overload `--bundle`.
3. **Capsule generalization.** Recommended: keep the envelope generic and let the default
   compiler reference the existing per-row `TrainingRunExecutionCapsule` v1. Introduce a
   new generic capsule family only when a second Feedbax compiler proves shared fields.

None of these decisions changes the central contract: authored request enters the engine,
ASSEMBLE produces the typed bundle, and CERTIFY compares its immutable execution identity
against the final manifest.
