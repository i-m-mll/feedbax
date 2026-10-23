# Max-Scrutiny Review: feedbax core execution (runtime/, execution/, components/, models/, acausal/)

Scope reviewed: `feedbax/runtime/graph.py` (1631 lines), `feedbax/runtime/components.py` (766 lines),
`feedbax/runtime/retained_observables.py` (1063 lines), `feedbax/execution/backends.py` (722 lines),
`feedbax/execution/planning.py` (668 lines), `feedbax/components/equinox.py` (1500 lines),
`feedbax/components/penzai.py` (737 lines), `feedbax/models/networks.py` (1248 lines),
`feedbax/acausal/assembly.py` (707 lines), plus supporting context files (`runtime/_graph.py`,
`execution/models.py`, `execution/container.py`, `execution/local.py`, `models/cde.py`,
`acausal/base.py`, `acausal/system.py`).

All findings below were verified against file contents at review time (no speculation from
memory/training data). Already-tracked issues (96a0725, 29c89d8, 563cda8, 3fb7f56, 44183de,
7550a17, 2f8dd61, 56dfd97) are referenced where relevant but not re-reported as new findings.

---

## Findings

### Graph execution engine (`feedbax/runtime/graph.py`)

**G1** `cached_property` mutates a frozen Equinox `Module`'s `__dict__`, bypassing immutability by construction | severity: **medium** | area: Equinox correctness
Evidence: `graph.py:349` `@cached_property def _cycle_analysis`, `:361` `_execution_order`/`_cycle_wires` (via `@property` over `_cycle_analysis`, fine), `:362` `@cached_property def _cycle_wire_set`, `:476` `@cached_property def _node_output_initializer_upstream_node_set`, `:579` `@cached_property def _outgoing_wires`. `Graph` subclasses `Component(Module)`, and Equinox's `Module` is `dataclass(frozen=True)` (verified in `equinox/_module.py:137,327`) with a `__setattr__` override that raises after `__init__`. `functools.cached_property` writes directly to `instance.__dict__`, which is why this "works" — it bypasses `__setattr__` entirely rather than complying with it.
Why it matters: this contradicts the repo's own stated convention ("Treat `Module` instances as immutable... avoid direct attribute assignment") in spirit, even though the literal `self.field = x` pattern isn't used. It happens to be harmless *today* because `eqx.tree_at`/graph-surgery methods (`add_node`, `remove_node`, etc.) always construct genuinely new `Graph` instances via `type(self)(**field_values)` or `eqx.tree_at`, so the cache starts empty on every mutation path. But it is fragile: any future code that clones a `Graph` via a mechanism that preserves `__dict__` (e.g. `copy.copy`, or a pytree unflatten path that reuses `__dict__`), or that accesses `_cycle_analysis` before vs. after a `jax.jit` retrace boundary, risks a stale cache silently surviving a structural change (e.g. `wires` changed but `_cycle_wire_set` still reflects the old wire set). The same footgun was independently found in `models/networks.py`'s `LeakyRNNCell.alpha`/`noise_std` (see N-cluster below) — this is a repo-wide pattern, not a one-off.
Fix: Precompute these values once in `__check_init__`/`__post_init__` as declared (frozen) dataclass fields set via the `_InitableModule` window, or convert to plain (uncached) `@property` if recomputation cost is acceptable (topological sort over typically-small node counts), or explicitly document why the cache-invalidation argument above is safe and add a test that mutates `wires` and asserts `_cycle_wire_set` updates.
Effort: S–M.

**G2** `Graph._execute_step`'s per-node Python dispatch loop is trace-time-only (positive finding, but scaling caveat) | severity: **low** | area: architecture
Evidence: `graph.py:886-962` (`_execute_step`) is a plain Python `for node_name, node_key in zip(self._execution_order, keys)` loop, invoked from `step_body`/`step_streaming` (`:1434`, `:1466`), which are themselves the bodies passed to `lax.scan` at `:1458`/`:1490`/`:1509`. This means the Python-level node dispatch happens once per `jax.jit`/`lax.scan` **trace**, not once per **timestep** at runtime — the actual per-step execution is a single compiled `scan`. This is the architecturally correct design and should be called out as such (it directly addresses the "does execution compile to a single scan or re-dispatch in Python" question in the review brief: it compiles to a single scan).
Caveat: nested `Graph` nodes recurse into their own `step()`/`_execute_step()` at the same trace time (`:938-946`), so trace time (and compiled HLO size) grows with total node count across all nesting levels, and every `_execute_step` call constructs fresh Python dicts (`port_values`, `node_inputs`) per node at trace time. This is fine for compiled-program correctness but means large/deeply-nested graphs will have proportionally larger compile times and larger unrolled-per-step HLO (since each node's ops are inlined once per scan body, not looped). For typical models (tens of nodes) this is a non-issue; for very large auto-generated graphs (hundreds+ of nodes, deep composite nesting) this is a latent compile-time/code-size concern worth monitoring.
Fix: none needed now; flag as a scaling watchpoint if Studio starts generating very large composite graphs.
Effort: N/A (informational).

**G3** `_execute_step` skip condition for partially-fed `node_output` initializer upstream nodes reads oddly restrictive and undertested | severity: **medium** | area: correctness / readability
Evidence: `graph.py:910-914`:
```python
if (
    node_name in self._node_output_initializer_upstream_node_set
    and len(node_inputs) < len(node.input_ports)
):
    continue
```
Why it matters: this silently skips executing a node mid-scan if it's part of a `node-output` recurrent-initializer's upstream sub-DAG and doesn't yet have all its inputs populated in `port_values` for *this* step. The intent (per the surrounding cycle-initializer machinery) appears to be to avoid re-evaluating nodes whose only purpose was pre-step initializer evaluation, but this same node may also have a legitimate role in the *main* per-step execution order, and this condition would then silently skip its real per-step execution if its inputs aren't wired in a way the check anticipates — the check is structural (does the node happen to have fewer wired inputs than its total input_ports) rather than semantic (is this specifically an initializer-only invocation). No comment explains why "fewer inputs than input_ports" is a safe proxy for "already-evaluated-during-initializer-phase," and no test file in scope demonstrates this edge case.
Fix: Add an explicit inline comment describing the invariant this relies on (or better, thread an explicit "already evaluated as initializer" flag instead of inferring it from input-count), and add a regression test with a `node-output` recurrent initializer whose upstream node also legitimately receives a subset of its inputs from a different (non-initializer) wire in the main step.
Effort: S (docs) to M (structural fix + test).

**G4** `RolloutStepHook` mechanism adds a full `RolloutStepContext` construction (with the entire live `port_values`, `state`, and `step_inputs`) per node, per step, only used by debugging/analysis paths | severity: **low** | area: JAX performance
Evidence: `graph.py:915-937` — inside the main per-node loop, if `rollout_step_hook is not None`, a `RolloutStepContext` `Module` instance is constructed per node (carrying `graph=self`, full `state`, full `port_values` dict, etc.) and passed through `_apply_rollout_step_hook`, which re-validates all returned port keys (`_validate_rollout_hook_port_keys`, `:990-1002`) via Python-level dict iteration and `set` membership checks — inside the scanned/traced function body.
Why it matters: this is fine when `rollout_step_hook is None` (the conditional is skipped and costs nothing after tracing — Python `if` on a static `None` is eliminated). But when a hook *is* provided (e.g., during a Studio trace-collection run), the entire mechanism runs at trace time; the concern is less about the (correctly trace-time) construction and more that the hook contract requires copying/replacing whole `port_values` dicts per node (`RolloutStepHookResult.port_values` is documented as "the complete replacement mapping for the current graph step... Hooks that only need to update one port should copy `context.port_values` first"). For debugging paths this is acceptable, but the docstring's advice to "copy `context.port_values` first" means a naive hook author will trigger an O(n_ports) dict copy per node per step at trace time — for large graphs with many ports this could measurably slow down trace/compile time for hook-instrumented runs (never for uninstrumented production training, which is the common case).
Fix: Document the trace-time-only cost tradeoff explicitly; consider offering a narrower hook contract (e.g. return only changed entries, merged internally) to avoid encouraging full-dict copies.
Effort: S.

**G5** `_get_initial_cycle_values`/`_evaluate_node_output_initializer` re-derive a second, parallel execution engine for pre-step evaluation | severity: **medium** | area: architecture / duplication
Evidence: `graph.py:1307-1372` (`_evaluate_node_output_initializer`) re-implements a node-execution loop (`for node_name, node_key in zip(self._execution_order, keys): ... node(node_inputs, eval_state, key=node_key)`) that is structurally near-identical to the main `_execute_step` loop (`:903-957`), but without the rollout-hook plumbing, without cycle-wire handling, and iterating over `upstream_nodes` instead of the full node set. This is a second, hand-maintained copy of "run nodes in topological order, threading port_values" logic.
Why it matters: any future change to how outputs propagate along instant wires in the main loop (e.g. a bugfix to `_outgoing_wires` handling, or new port-value semantics) must be manually mirrored here or the two loops will silently diverge — e.g., `_execute_step` explicitly skips propagating cycle wires (`if wire in self._cycle_wire_set: continue`, `:955-956`), while `_evaluate_node_output_initializer`'s equivalent line (`:1360`) checks `if outgoing_wire.temporality != "instant": continue` — a *different* (though currently equivalent for this use case) condition expressed independently. A `Wire.temporality` value other than `"instant"`/`"recurrent"` introduced in the future (there's no `Literal` constraining `Wire.temporality` — see G6) would make these two conditions diverge without either loop erroring.
Fix: Factor the "run these nodes in this order, propagating instant wires, threading port_values" primitive into one shared helper method parameterized by "which nodes to run" and "whether to apply cycle wires," used by both `_execute_step` and `_evaluate_node_output_initializer`.
Effort: M.

**G6** `Wire.temporality: str = field(default="instant", static=True)` is an unconstrained string, not a `Literal["instant", "recurrent"]` | severity: **low** | area: code quality / type safety
Evidence: `graph.py:216` `temporality: str = field(default="instant", static=True)`. Every consumer (`_analyze_cycles:589`, `_execute_step:955`, `_evaluate_node_output_initializer:1360`) compares against literal strings `"recurrent"`/`"instant"` with no shared enum/constant and no static type constraining valid values.
Why it matters: a typo'd `temporality="Recurrent"` (wrong case) or a new value introduced by a future feature would silently fall through every `== "recurrent"` check as `False`, being treated as an ordinary instant wire without any validation error — exactly the kind of "silent fallback" the project's conventions warn against, here at the type-safety level rather than the architecture level.
Fix: Change to `Literal["instant", "recurrent"]` (already imported at `:14`) and let Pydantic-equivalent/dataclass validation catch bad values at `Wire` construction.
Effort: S.

**G7** `init_state_from_component`'s reflective `StateIndex` discovery swallows all exceptions during attribute access | severity: **low** | area: code quality / error handling
Evidence: `graph.py:71-78`:
```python
def _iter_state_indices(obj) -> list[StateIndex]:
    indices: list[StateIndex] = []
    if dataclasses.is_dataclass(obj):
        for field_obj in dataclasses.fields(obj):
            try:
                value = getattr(obj, field_obj.name)
            except Exception:
                continue
```
Why it matters: a bare `except Exception: continue` here means any property/descriptor on a `Component` subclass that raises during state initialization (e.g. a computed field that depends on something not yet set, or a genuine bug in a user-defined `@property`) is silently treated as "not a StateIndex, skip it" rather than surfacing the underlying error. Combined with G1 (cached_property mutating `__dict__`), if a `cached_property`-based field raises partway through evaluation, this reflective walk would swallow that error during state construction rather than reporting it — a debugging trap for anyone adding a computed property to a `Component` subclass.
Fix: Narrow to `except AttributeError: continue` (the only exception type that should legitimately mean "field access is not simple," e.g. from a lazily-failing property) and let other exceptions propagate.
Effort: S.

**G8** `Component.initial_outputs` uses `hasattr`/`attrgetter` reflection to bridge state values to output ports, with no validation that all declared `output_ports` are actually derivable | severity: **low** | area: code quality
Evidence: `graph.py:142-150`:
```python
def initial_outputs(self, state_value: PyTree | None) -> dict[str, PyTree]:
    if state_value is None:
        return {}
    outputs: dict[str, PyTree] = {}
    for port in self.output_ports:
        if hasattr(state_value, port):
            outputs[port] = attrgetter(port)(state_value)
    return outputs
```
Why it matters: if a component declares `output_ports = ("output", "hidden")` but its state object only exposes `hidden` (e.g. a naming mismatch introduced during refactoring), `initial_outputs` silently returns a partial dict rather than raising — and callers of this (e.g. `_get_initial_cycle_values`, `:1173-1178`) treat a missing key as "fall through to recurrent_initializer metadata" rather than "component is misconfigured." This makes a genuine port/state naming bug indistinguishable from the legitimate "no initial value derivable, use recurrent_initializer" case.
Fix: Not urgent given the fallback path is a deliberate design (recurrent_initializer as an escape hatch), but consider a debug-mode assertion or Studio-side validation that a component's `output_ports` are always resolvable from its `state_view`/`initial_outputs` when no recurrent initializer is present, to catch naming drift early.
Effort: S.

---

### Runtime components (`feedbax/runtime/components.py`)

**C1** `MLP.__call__` and several `_StepSource` subclasses are missing a blank-line/formatting consistency issue that's cosmetic but signals low test coverage on this file | severity: **low** | area: code quality
Evidence: `components.py:463-464`:
```python
        )
    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
```
Missing blank line between `__init__` and `__call__` (every other class in the file has one). Trivial, but consistent with the file not having gone through a lint pass recently.
Fix: `ruff format`.
Effort: S.

**C2** `Constant.value: PyTree = field(static=True)` — a JAX-array-valued static field, matching the already-tracked static-metadata-array class of bugs (96a0725/29c89d8) | severity: **medium** (already-tracked pattern — do not file as new, but flag as another concrete instance for the tracked issue's remediation scope) | area: JAX performance
Evidence: `components.py:173-176`:
```python
value: PyTree = field(static=True)

def __init__(self, value: PyTree = 0.0):
    self.value = jt.map(jnp.asarray, value)
```
This is a textbook instance of tracked issue 96a0725/29c89d8 ("JAX arrays stored as static PyTree metadata") — `Constant.value` is explicitly converted to `jnp.asarray` and then stored as a `static=True` field, meaning every distinct `Constant` value used anywhere in a graph forces a new `jax.jit` cache entry (arrays aren't hashable/comparable the way static fields require, so Equinox falls back to identity or raises depending on version — in practice this causes silent recompilation on every distinct constant value/every training step if `Constant.value` is ever swept). Also present in `_StepSource` subclasses: `Ramp.slope`/`intercept` (`:216-217`), `Sine.amplitude`/`offset` (`:235,238`), `Pulse.amplitude`/`offset` (`:270,273`), all `field(static=True)` wrapping `jt.map(jnp.asarray, ...)` results.
Why it matters: flagging explicitly because this is the single most repeated instance of the tracked pattern across the reviewed scope — six distinct component classes in one file exhibit it. Worth appending to the tracked issue as concrete remediation targets rather than treating as resolved by fixing one occurrence elsewhere.
Fix: per the tracked issue's remediation (make these genuine PyTree leaves, not static fields, or store the pre-array Python scalar as static and construct the array inside `__call__`).
Effort: (tracked elsewhere — effort estimate deferred to that issue).

**C3** `DelayLine`'s `_initial_state` couples a `PyTree` (possibly array-containing) tuple as a `static=True` field | severity: **medium** (same tracked-pattern family as C2) | area: JAX performance
Evidence: `components.py:358,366-369`:
```python
_initial_state: tuple[PyTree, tuple[PyTree, ...]] = field(static=True)
...
output = jt.map(lambda x: jnp.full_like(x, self.init_value), input_proto)
queue = self.delay * (output,)
self._initial_state = (output, queue)
```
`output`/`queue` are JAX arrays (from `jnp.full_like`) stored inside a `static=True` field — same class as C2, and additionally this static tuple's *length* depends on `self.delay` (a runtime-configurable int), meaning two `DelayLine` instances with different `delay` values already recompile (expected, since `delay` legitimately changes structure) — but even two instances with the *same* `delay` and different `init_value` will also force separate compiled programs, unnecessarily.
Fix: same remediation family as C2/96a0725.
Effort: deferred to tracked issue.

**C4** `Noise.__call__` reads `key` correctly but the component provides no way to seed `mean`/`std` as trainable arrays — likely intentional but undocumented | severity: **low** | area: code quality
Evidence: `components.py:302-313` — `mean: float`, `std: float` are plain Python floats (not `Array`), so unlike `Gain`/`Spring`/`Damper` (which also use plain `float` fields), `Noise` cannot be made trainable via the graph's parameter-selection API (`select_nodes_of_type` + gradient-based training) without a refactor. Not a bug, but worth a docstring note since several sibling classes (`Linear`, `MLP`) support `dtype`/trainable arrays while these simple math components uniformly do not — a graph author might reasonably expect `Gain.gain` to be trainable the way `Linear.layer`'s weights are, and be surprised it silently isn't part of any gradient (it will be treated as a static Python float baked into the compiled program, again a variant of the C2/C3 pattern but the reverse direction — a value the user might *want* to be traced/trainable is static instead).
Fix: docstring clarification; if trainability is desired, migrate to `Array` fields.
Effort: S (docs).

---

### Retention/loss plan lowering (`feedbax/runtime/retained_observables.py`)

**R1** `evaluate_loss_term`'s "no target" fallback silently defaults to `target_value=0.0` rather than requiring an explicit choice | severity: **low** | area: code quality / API clarity
Evidence: `retained_observables.py:311-320`:
```python
def evaluate_loss_term(term: LossTermPlan, trace: Mapping[str, Any]) -> Any:
    source = _lookup_trace(trace, term.source.selector, f"/loss/{term.key}/selector")
    if term.target is not None:
        target = _lookup_trace(trace, term.target.selector, f"/loss/{term.key}/target_selector")
    elif term.target_value is not None:
        target = term.target_value
    else:
        target = 0.0
```
Why it matters: a loss term authored in Studio with neither `target_selector` nor `target_value` set (e.g. an incomplete/in-progress spec, or a UI bug that fails to populate the field) silently becomes "minimize squared distance to zero" rather than raising a validation error at plan-lowering time. This is a real risk given `lower_retention_plan`/`_lower_loss_terms` (`:467-580`) otherwise does careful validation (e.g. raising if both `target_selector` and `target_value` are set, `:522-527`) — the missing-both case is the one gap in that validation.
Fix: In `_lower_loss_terms`, raise `RetentionPlanError` if `term.type` requires a target (most norm-based losses do) and neither `target_selector` nor `target_value` is set, rather than deferring to a silent zero-default at evaluation time. This is squarely a "the graph is the model" / no-silent-fallback violation: a Studio-authored loss with no target is an incomplete spec, not a valid "target zero" spec.
Effort: S.

**R2** `_validate_executable_retention` unconditionally rejects `"stream"`/`"window"` retention modes, meaning the `RetentionMode` type's other two values are permanently dead code paths in the current worker | severity: **low** | area: code quality / dead code
Evidence: `retained_observables.py:43` declares `RetentionMode = Literal["stream", "window", "trajectory"]`, and `_RETENTION_RANK` (`:54-58`) ranks all three, but `_validate_executable_retention` (`:715-726`) unconditionally raises for anything other than `"trajectory"`:
```python
def _validate_executable_retention(requirement: _ObservableRequirement) -> None:
    mode = requirement.retention.mode
    if mode == "trajectory":
        return
    raise RetentionPlanError(...)
```
Why it matters: `"stream"` and `"window"` retention (plus the associated `order`/`window_size`/streaming-loss machinery elsewhere in `graph.py`'s `streaming_loss_fn` path) are modeled throughout this file's types (`RetentionPolicyPlan.window_size`, `.order`, `_merge_retention`'s ranking logic) but are unreachable in practice because this single validation gate rejects them before they'd ever reach an evaluator. This is not wrong (correctly fails closed rather than silently downgrading to trajectory), but it means a meaningful fraction of this file's type surface (window/stream modes, their merge-ranking logic) is currently unexercised dead functionality — worth flagging so a reviewer doesn't assume streaming/windowed retention is production-ready just because the types model it richly.
Fix: either implement window/stream support in the worker (tracked separately if planned) or add a comment at the `RetentionMode` definition noting that `"stream"`/`"window"` are reserved/unimplemented, so the rich merge/ranking logic isn't mistaken for a supported feature.
Effort: S (docs) or L (if streaming/window support is meant to land here).

**R3** `retention_plan_to_json`'s `time_mask` serialization silently drops non-array-convertible masks via unchecked `jnp.asarray(..., dtype=bool).tolist()` | severity: **low** | area: robustness
Evidence: `retained_observables.py:381-385`:
```python
"time_mask": (
    jnp.asarray(item.time_mask, dtype=bool).tolist()
    if item.time_mask is not None
    else None
),
```
Why it matters: if `item.time_mask` were ever a JAX tracer (e.g. this serialization function called from inside a jitted context, which is plausible given it's used for manifest/artifact writing that might be invoked mid-pipeline) this would raise a `ConcretizationTypeError` — a host-device sync/tracer-leak risk. Given this function's purpose (serializing a plan to JSON for a manifest), it should only ever be called with concrete arrays, but nothing enforces that at the type level, and no docstring states the "must be called outside any jit context" requirement.
Fix: add a docstring note or an explicit `jax.core.concrete_or_error` guard.
Effort: S.

---

### Execution planning/backends (`feedbax/execution/backends.py`, `feedbax/execution/planning.py`)

*(Full sub-review conducted by delegated agent; representative high-value findings included below — see summary table for complete list.)*

**E1** `ExecutionPlan.cloud_payload`/`.reproducibility` are unschemad `dict[str, Any]` despite being part of a versioned, durable artifact | severity: **high** | area: CLAUDE.md migration policy violation
Evidence: `execution/models.py:479-481`:
```python
cloud_payload: dict[str, Any] = Field(default_factory=dict)
reproducibility: dict[str, Any] = Field(default_factory=dict)
```
`ExecutionPlan` itself is versioned (`EXECUTION_PLAN_SCHEMA_VERSION = "feedbax.manifest.execution.v3"`) and persisted as a tracked artifact (`local.py:184-190`, `planning.py:663-668` writes `execution-plan.json`), but its two largest, most backend-specific fields have no internal schema/version and are hand-assembled from string literals in `backends.py:107-173`.
Why it matters: this is precisely the gap the CLAUDE.md Artifact Schema policy is written to close — "any new or changed structured spec emitter must declare a stable schema identity... Validation-only Pydantic shapes are not sufficient for durable emitted specs." A downstream consumer parsing `cloud_payload["worker_transport"]["port"]` has no contract guaranteeing that key survives a `backends.py` refactor.
Fix: Promote `cloud_payload` to a discriminated-union Pydantic model per backend (`RunPodCloudPayload | ModalCloudPayload`) with its own schema id/version.
Effort: M–L.

**E2** Command-construction and provenance-collection logic is triplicated across `backends.py`, `planning.py`, and `container.py` with small, unverified divergences | severity: **high** | area: architecture / duplication
Evidence: `_execution_command`/`_training_run_spec_path` (`backends.py:597-635` vs `planning.py:103-132`) are structurally identical but independently derive `run_directory` differently; `_source_provenance_record`/`_local_embed_source_provenance` (`backends.py:686-705`) vs `_local_embed_source_record` (`planning.py:627-641`) build the same provenance dict shape with one path expanding env vars (`expand_local_path`) and the other not; `rewrite_embedded_paths` exists as real importable code in `container.py:177-189` and is *also* re-implemented as a string template embedded in generated Modal source at `backends.py:458-465` — character-for-character identical logic maintained as two independent copies.
Why it matters: none of these have observably diverged yet, but there is no shared source of truth and no test pinning the invariant that they agree — this is exactly the kind of "second, textually-forked implementation of the same behavior" the project's "no legacy shims / no background construction" ethos warns against in spirit, generalized to executor tooling.
Fix: Extract one shared helper per duplicated concern (command rendering, provenance collection, path rewriting) used by both files.
Effort: M.

**E3** `cloud_payload()`'s cell-command records are a hand-maintained, independent mirror of what the Modal app renderer actually emits, with no shared derivation | severity: **medium** | area: correctness / integrity of previewed plan
Evidence: `backends.py:130-142` (`cloud_payload`) independently rebuilds cell command records rather than calling the same helper `_render_modal_app_pip_install`/`_render_modal_app_local_embed` use to build the actual generated app's `CELLS` literal.
Why it matters: `cloud_payload` is presented to users/tooling as "what will run," but is computed by an entirely separate code path from the actual generated app source — a future change to one renderer's cell-command wrapping will not automatically show up in the previewed payload, letting the inspectable plan drift from the artifact that actually executes.
Fix: Derive `cloud_payload["cells"]` from the same cell-record-building helper the renderers use.
Effort: M.

**E4** `SCHEMA_VERSION` bumps have no in-repo changelog/migration trace | severity: **medium** | area: CLAUDE.md migration policy
Evidence: `models.py:317` `schema_version: Literal[EXECUTION_SPEC_SCHEMA_VERSION]` (currently v2) alongside `ExecutionPlan`'s v3 — no comment anywhere ties either version bump to an issue or describes what changed.
Why it matters: CLAUDE.md requires recording "the migration issue and validation strategy" for durable-format changes; a future reader hitting a v2/v3 mismatch has no way to know if it's intentional (independent versioning axes) or a bug.
Fix: Add a one-line comment/changelog mapping at each `SCHEMA_VERSION` constant.
Effort: S.

**E5** `RunPodBackendConfig.gpu_type_ids` empty-list handling diverges between the CLI renderer (silently substitutes a default GPU) and the REST payload renderer (sends the empty list as-is) | severity: **low** | area: robustness
Evidence: `models.py:268-269` guards with a fallback default; `models.py:237-258`'s `pod_request` passes `self.gpu_type_ids` directly with no equivalent guard.
Why it matters: a plan can "look fine" in its CLI-rendered form while its REST payload would fail against the live RunPod API — same divergent-renderer pattern as E2/E3.
Fix: Validate `gpu_type_ids` non-empty in a model validator so both renderers see a guaranteed-non-empty list.
Effort: S.

*(Additional lower-severity findings from the full sub-review: local-run `model_dump()`/reconstruct round-trip in `local.py:42` instead of `model_copy(update=...)`; a needlessly double-encoded `_json_loads_literal` helper; an `ExecutionCell.id` regex that permits path-traversal-shaped ids like `".."`; redundant defense-in-depth validation of `command is None` already guaranteed by a Pydantic validator. All S-effort; see delegated sub-review detail retained in session for follow-up if wanted.)*

---

### Component adapters (`feedbax/components/equinox.py`, `feedbax/components/penzai.py`)

**A1** `BatchNorm` wrapper never threads `State` through `eqx.nn.BatchNorm`, so running statistics are never updated | severity: **critical** | area: correctness
Evidence: `equinox.py:889-897`:
```python
output = eqx.nn.inference_mode(self.layer)(inputs["input"])
return {"output": output}, state
```
Equinox's `BatchNorm.__call__` requires `state` as a positional argument and returns `(output, updated_state)`; this wrapper always forces `inference_mode` and never passes/receives state. Running mean/variance are never threaded through feedbax's `State`, and no test in scope exercises `BatchNorm`.
Why it matters: any graph using this node for training-mode batch norm gets silently wrong behavior — normalization is always done in inference mode (using whatever the layer's frozen initial statistics are, never updated from data) regardless of what a Studio user configures. This is a leaf-level instance of "what's configured silently diverges from what runs," the same class of bug the project's Core Principle exists to prevent at the graph level.
Fix: Thread state via a `StateIndex` mirroring other stateful wrappers (`GRU`, `LSTM`, `DelayLine` in `runtime/components.py` show the correct pattern); call `self.layer(inputs["input"], state=bn_state)` and store the returned updated state.
Effort: M.

**A2** `MultiheadAttention` wrapper drops the `key` argument entirely, silently unseeding dropout | severity: **high** | area: correctness
Evidence: `equinox.py:1355-1367` — `key` is received (required by the `Component.__call__` contract) but never forwarded to `self.layer(...)`.
Why it matters: any node configured with `dropout_p > 0` silently loses its dropout regularization — a configured hyperparameter with zero effect, invisible without reading the wrapper's source.
Fix: pass `key=key` through to `self.layer(...)`.
Effort: S.

**A3** `Dropout` wrapper unconditionally forces `inference_mode`, making the `inference`/`p` constructor params dead | severity: **high** | area: correctness / "graph is the model" leaf-level violation
Evidence: `equinox.py:1449-1476` unconditionally wraps in `eqx.nn.inference_mode(...)` regardless of the `inference` flag a Studio user sets.
Why it matters: identical bug class to A1/A2 — a Studio-configurable knob (`inference=False`, meant to enable dropout during training) has no effect on execution. Docstrings at `equinox.py:848,1450` claim behavior is "by default" configurable when it is in fact hardcoded — actively misleading documentation.
Fix: `return self.layer(inputs["input"], key=key)` and let equinox's own inference toggle apply per the constructed `inference` flag; fix docstrings.
Effort: S.

**A4** `PenzaiStateManager.bind_state` is a non-functional stub that silently no-ops rather than raising | severity: **critical** | area: "graph is the model" violation
Evidence: `penzai.py:303-323` — `bind_state` never re-injects stored state values back into the Penzai model; the code returns the model unmodified, with a comment: "For simplicity, we return the model as-is if no special handling needed." Any Penzai model carrying `StateVariable`s (cached KV, running stats, etc.) silently loses state across every graph step, despite the surrounding plumbing (`unbind_state`, `state_index`, `init_state`) presenting a complete-looking state-management API.
Why it matters: this is a textbook instance of the CLAUDE.md Core Principle's explicit prohibition: "'Just for now' workarounds are bugs... display-only nodes that shadow real architectural choices, and fallback paths that substitute stale values silently are all bugs regardless of how they are labelled in the code." A Studio user wiring a stateful Penzai model into a graph has no signal that state isn't actually being threaded — it looks wired, it silently isn't.
Fix: implement real state re-binding via the same tree-walk `unbind_state` already performs (in reverse), or raise `NotImplementedError` clearly at construction/call time until implemented, per the CLAUDE.md instruction to raise rather than silently substitute.
Effort: M–L.

**A5** `PenzaiSubgraph.__call__` silently passes input through as identity when the wrapped model isn't callable | severity: **high** | area: "graph is the model" violation
Evidence: `penzai.py:544-550` — `else: layer_output = layer_input`, with a comment claiming the code "tries to find a `__call__` method" when it does not attempt anything; it silently no-ops to an identity pass-through.
Why it matters: same silent-fallback class as A4 — a misconfigured or unsupported Penzai subgraph node produces plausible-looking (identity) output rather than a clear, located error.
Fix: raise `TypeError(f"PenzaiSubgraph node {...} model is not callable")`.
Effort: S.

**A6** `state_view` in `penzai.py` catches both `KeyError` and `ValueError` broadly, masking real bugs as "no state" | severity: **medium** | area: error handling
Evidence: `penzai.py:596-600`.
Why it matters: compounds A4 — a genuinely broken state binding (wrong `State` object passed, a subgraph reused across incompatible state trees) is indistinguishable from "this component legitimately has no state."
Fix: narrow the exception type, or log/annotate on catch.
Effort: S.

**A7** ~21 of the ~30 generated `equinox.py` Component wrapper classes share byte-identical bodies with no shared base, meaning bugs like A1–A3 must be independently caught per class rather than fixed once | severity: **medium** | area: architecture / code quality
Evidence: at least 21 classes share the literal body `output = self.layer(inputs["input"]); return {"output": output}, state` (verified by the delegated sub-review's read of the full file). This is presumably generated (there is a `scripts/generate_eqx_components.py`-style generator implied by the mechanical repetition), but the generator does not appear to validate each wrapped layer's actual `__call__` signature against the emitted wrapper — which is exactly why `BatchNorm` (needs `state`), `MultiheadAttention` (needs `key`), and `Dropout` (needs `key`, no forced `inference_mode`) each independently deviate from correct behavior.
Why it matters: this explains the *root cause* of A1–A3 as a class, not three unrelated bugs — the generation/wrapping process has no per-layer signature check, so any equinox layer whose `__call__` needs something beyond `(x)` silently gets the wrong wrapper shape.
Fix: Either (a) have the generator introspect each wrapped layer's real signature and emit the correct forwarding call (state-threading for stateful layers, key-forwarding for stochastic layers), or (b) add a test that asserts, for every generated wrapper class, that all of the wrapped equinox layer's non-default `__call__` parameters are actually threaded through.
Effort: M (generator fix) + tests.

**A8** No version marker ties `equinox.py`'s generated wrapper set to the equinox version it was generated against | severity: **medium** | area: CLAUDE.md migration policy
Evidence: no `schema_version`/generator-version constant found anywhere in `equinox.py`; `BatchNorm`'s wrapper is missing the `mode: Literal['ema','batch','legacy']` parameter that current `eqx.nn.BatchNorm.__init__` accepts, direct evidence the generator/wrapper pair has drifted from the installed equinox version without any check catching it.
Why it matters: a saved graph's component params are shaped by whatever `equinox.py` looked like when a component was added to a subgraph; regenerating this file against a newer/older equinox release could silently reinterpret old saved constructor params without any version gate.
Fix: emit and check a version constant tying generated wrappers to the equinox version/generator run that produced them.
Effort: M.

**A9** Callable static fields (`combine_fn`/`unpack_fn` in `penzai.py`'s `InputMapping`/`OutputMapping`, and `Graph.state_view_fn`/`state_consistency_fn` in `graph.py:300-301`) risk hash churn if constructed as fresh closures per Studio rebuild | severity: **low-medium** | area: JAX performance (distinct from the tracked array-as-static-field issue — this is callable-as-static, not array-as-static)
Evidence: `penzai.py:141-146` (`InputMapping.multi(...)`) builds a new lambda per call; `graph.py:300-301` fields are correctly-typed callables but nothing prevents a caller from passing a freshly-constructed closure on every graph rebuild.
Why it matters: static fields participate in `jax.jit`'s cache key; a fresh Python closure (new identity, and lambdas don't define custom `__eq__`/`__hash__`) forces a new compiled program every time, even if the closure is behaviorally identical to the previous one.
Fix: hoist to module-level named functions where possible; document the reuse expectation for callers who must pass closures.
Effort: S (docs) to M (refactor call sites).

*(Additional lower-severity findings from the full sub-review: `RotaryPositionalEmbedding`'s shape contract silently mismatches the graph's per-step execution convention; `Embedding` forwards dims with zero shape validation; `PenzaiSubgraph.init_state` reaches into a collaborator's private `_initial_state` attribute instead of a public accessor; a redundant local re-import of `pzl` already bound at module scope.)*

---

### Networks (`feedbax/models/networks.py`, `feedbax/models/cde.py`)

**N1** `n_positional_args`-based binary dispatch for hidden-cell calling convention has no explicit error path for cells that don't fit the two supported shapes | severity: **high** | area: architecture / silent-limitation pattern (generalization of tracked 44183de, not a re-report of the LSTM-specific case)
Evidence: `networks.py:975-978`:
```python
if n_positional_args(self.hidden) == 1:
    hidden = self.hidden(x_hidden)
else:
    hidden = self.hidden(x_hidden, net_state.hidden)
```
Why it matters: this binary heuristic assumes every stateful `hidden_type` accepts exactly `(input, state)` positionally. A custom cell with any other signature (multiple state tensors, a `state=None` default, etc.) is silently mis-routed — producing a wrong value or an inscrutable shape error deep inside a matmul, not a clear "unsupported cell type" error at construction time. This is the same *pattern* as the tracked LSTM issue (44183de) but is the general mechanism producing it — fixing LSTM alone would not fix the next cell type that doesn't fit this shape.
Fix: Replace the heuristic with an explicit registered adapter/protocol per accepted `hidden_type`, validated at `__init__` time (raise clearly if the signature doesn't match a known shape) rather than branching silently at call time.
Effort: M.

**N2** `cached_property` on `LeakyRNNCell` (a frozen `equinox.Module`) mutates `__dict__` directly | severity: **medium** | area: Equinox correctness — same root-cause pattern as G1, found independently in a different file
Evidence: `networks.py:1113-1121`:
```python
@cached_property
def alpha(self):
    return self.dt / self.tau

@cached_property
def noise_std(self):
    if self.use_noise:
        return math.sqrt(2 / self.alpha) * self.noise_strength
    else:
        return None
```
Why it matters: identical mechanism to G1 — `functools.cached_property` bypasses the frozen-dataclass `__setattr__` enforcement by writing straight to `instance.__dict__`. It "works" only because `eqx.tree_at`/copy paths always produce fresh instances with empty `__dict__` caches; it is not a declared PyTree field, so it is invisible to any tooling that inspects `Module`'s dataclass fields, and equinox's own documentation recommends against this pattern in favor of precomputing at `__init__` time. Given this same anti-pattern appears independently in `graph.py` (G1) and here, it looks like a repo-wide idiom rather than a one-off — worth a single sweep/lint rule.
Fix: precompute `alpha`/`noise_std` as regular fields in `__init__` (both are cheap, available from `dt`/`tau`/`use_noise`/`noise_strength` at construction time already).
Effort: S (this file) — consider a repo-wide grep-based sweep given G1 shows the same pattern elsewhere.

**N3** `LeakyRNNCell.use_noise=True` is unreachable/broken when used as `SimpleStagedNetwork.hidden_type`, because the dispatch in N1 never threads a `key` | severity: **medium** | area: correctness
Evidence: `networks.py:975-978` (see N1) never passes `key=` to `self.hidden(...)`; `LeakyRNNCell.__call__` (`networks.py:1109-1123`) raises `ValueError("LeakyRNNCell requires an RNG key when use_noise=True")` if `key is None`. So a `SimpleStagedNetwork(hidden_type=LeakyRNNCell, ...)` configuration with `use_noise=True` will always raise at call time (not at construction), the first time the network is actually run.
Why it matters: this is a configuration that looks valid at construction (no error raised when the network is built) and only fails deep into a training loop's first forward pass — should be caught at construction time instead, per the project's "raise a clear error" preference over deferred/silent failures.
Fix: either thread `key` through the hidden-cell dispatch (requires generalizing N1's fix anyway), or validate at `SimpleStagedNetwork.__init__` that `hidden_type`'s stochasticity requirements are compatible with the calling convention actually used, raising there instead.
Effort: M (bundled with N1).

**N4** Quadruplicated encoder/hidden-construction branches in `SimpleStagedNetwork.__init__` | severity: **medium** | area: code quality / maintainability
Evidence: `networks.py:759-849` — the encoder-present/absent × population-structure-present/absent 2×2 branching duplicates the "construct hidden layer, optionally mask `weight_ih`" block nearly verbatim across all four branches (~90 lines total).
Why it matters: quadruples the surface area for a bugfix (e.g. gate-count inference for GRU/LSTM cells) to be applied to one branch and missed in the others — directly relevant given N1's dispatch heuristic already shows this file has correctness bugs around cell-type handling.
Fix: factor a single `_build_hidden(input_dim, hidden_size, hidden_type, population_structure, ...)` helper shared by all four branches.
Effort: M.

**N5** Dead `encoder_mask = jnp.zeros(...)` assignment immediately overwritten, misleadingly implying population-structure masking is applied to the encoder layer | severity: **medium** | area: code quality / misleading-code (adjacent to "looks configured, isn't")
Evidence: `networks.py:766-769`:
```python
# Create mask for encoder: only input-receiving units get non-zero columns
encoder_mask = jnp.zeros((encoding_size, layer_input_size))
# For simplicity, allow all encoder units to receive all inputs
# The masking will happen at the encoder->hidden connection instead
encoder_mask = jnp.ones((encoding_size, layer_input_size))
```
Why it matters: the first line computes a real per-unit connectivity mask that is then discarded and replaced with all-ones — encoder connectivity is never actually population-constrained despite `population_structure` being passed in and a mask literally computed for it. A reader (or a Studio user relying on `PopulationStructure` documentation) could reasonably but wrongly conclude the encoder layer respects population wiring.
Fix: delete the dead first assignment; if encoder-level population masking is intended, implement it; otherwise document explicitly that population constraints apply only to input→hidden and hidden→readout connections.
Effort: S (cleanup) to M (if masking should be extended).

**N6** `CDENetwork` (the documented "drop-in replacement" for `SimpleStagedNetwork`) has no `dtype` field/parameter at all | severity: **medium** | area: architecture consistency / dtype churn
Evidence: `networks.py:958-961` casts `flat_input` to `self.dtype` in `SimpleStagedNetwork.__call__`; `models/cde.py`'s `CDENetwork.__call__` has no equivalent cast and no `dtype` constructor parameter anywhere in the class.
Why it matters: `cde.py` is documented as a drop-in replacement for `SimpleStagedNetwork`; a caller doing mixed-precision training who swaps between the two network types under an otherwise-identical graph config gets silently different dtype behavior (CDE always runs at whichever dtype `eqx.nn.MLP`/`jr.normal` default to, ignoring any `dtype` the surrounding graph/task was configured with).
Fix: add a `dtype` field to `CDENetwork` mirroring `SimpleStagedNetwork`'s, threaded through `vector_field`, `readout`, and state/`h0` construction.
Effort: M.

**N7** Silent dtype fallback when a custom `hidden_type`/`encoder_type`/`readout_type` factory doesn't accept a `dtype` kwarg | severity: **medium** | area: dtype churn / silent-fallback
Evidence: `_trainable_dtype_kwargs` (referenced near `networks.py:617-624`) returns `{}` (no `dtype` kwarg passed) when `_supports_keyword(factory, "dtype")` is `False`, with no warning.
Why it matters: `SimpleStagedNetwork.dtype` is meant to describe the network's numeric precision, but any sub-layer factory lacking a `dtype` parameter silently constructs at its own default (typically float32) regardless of the network's configured `dtype` — a correctness footgun for mixed-precision experiments, and a silent (not raised/warned) divergence between declared and actual behavior.
Fix: assert/warn loudly if any constructed leaf's dtype doesn't match `self.dtype` after construction, or raise clearly at construction time when a factory can't honor the configured dtype.
Effort: S–M.

*(Additional lower-severity findings from the full sub-review: dead unused `input` parameter in `_add_hidden_noise`; `MaskedLinear` recomputes `weight * mask` every forward call rather than once, with an implicit and undocumented reliance on external `ParameterConstraintSpec` enforcement to keep the underlying weight actually zero between training steps; a live unresolved `TODO` in the exported `gru_weight_idxs_func`; an unresolved `#!`-flagged rhetorical-question comment in the readout bias-zeroing branch; a speculative "future API" aside embedded in `PopulationStructure.create`'s docstring rather than tracked as an issue.)*

---

### Acausal assembly (`feedbax/acausal/assembly.py`)

**AC1** `AcausalSystem` is registered as a Studio-visible composite node with no subgraph-to-`AcausalElement`/`AcausalConnection` builder wired anywhere in the codebase | severity: **high** | area: "graph is the model" violation
Evidence: `feedbax/component_registry/builtins.py:1032-1053` registers `ComponentMeta(name='AcausalSystem', ..., is_composite=True)` with only a `dt`/`domain` param schema — no `template_graph`, `template_id`, or `builder`, unlike genuine composite templates (e.g. `templates.py:31-50`'s `recurrent_controller`, which always sets `template_graph=`). No construction site for `AcausalSystem` exists outside `feedbax/acausal/` and its own tests.
Why it matters: `is_composite=True` signals to Studio/the build pipeline that this node's real structure lives in a subgraph, but no code path translates a Studio subgraph into the `dict[str, AcausalElement]`/`list[AcausalConnection]` `AcausalSystem.__init__` actually requires. If a user drags this node onto the canvas today, the build either fails unhandled or (worse) some unaudited path constructs it with empty/default elements — exactly the "background construction"/"stale outer params" pattern the CLAUDE.md Core Principle prohibits. Per that principle, "absence of a subgraph is an error, not a condition to work around," and right now there is no evidence this case is even detected, let alone raised as a located error.
Fix: either remove the `builtins.py` registration until a real subgraph→element/connection builder exists, or (if partially implemented elsewhere) make it raise a clear, specific error rather than silently failing/falling back.
Effort: M (mostly registry/build-pipeline work, not this file).

**AC2** `_topo_sort_through_eqs` has no cycle detection, silently producing an invalid evaluation order for cyclic through-equation dependencies | severity: **high** | area: correctness / silent-fallback
Evidence: `assembly.py:692-702` — a standard DFS post-order sort with a `visited` set but no `in_progress`/recursion-stack tracking; a cyclic dependency between two through-equations simply returns early on re-entry rather than raising.
Why it matters: for a genuinely cyclic through-variable definition (a modeling bug where two elements each define the other's force in terms of the other), the assembler emits a silently wrong (non-topologically-valid) vector field rather than raising — and because `_build_vals` uses `vals.get(tv, 0.0)` (see AC3) for unresolved dependencies, the wrong order manifests as physically-plausible-looking but incorrect zero-valued force contributions rather than a crash. This compounds into a hard-to-diagnose silent correctness bug in a physics simulation — the exact failure mode ("fallback path that substitutes stale/default values silently... is a bug regardless of how it's labelled") the project's conventions call out. Note `_resolve` (the alias-resolution sibling function, `assembly.py:277-280`) already does correct cycle detection for aliases — this asymmetry suggests the through-equation sort was simply never given the same treatment, not a deliberate design choice.
Fix: track an `in_progress` set separate from `visited`; raise `ValueError(f"Cyclic through-variable dependency involving '{var}'")` on re-entry to an in-progress node, mirroring `_resolve`'s existing correct pattern.
Effort: S.

**AC3** Silent zero-fallback (`vals.get(tv, 0.0)`) for unresolved through-variable contributions masks assembly bugs as physically-plausible zero force | severity: **medium** | area: silent-fallback (compounds AC2)
Evidence: `assembly.py:377` and recurring at `:636-639`, `:661-664` — any through-variable that fails to resolve (naming bug, missing port registration, or an unhandled future element type) silently contributes `0.0` rather than raising.
Why it matters: in a mechanics domain, a silently-dropped force/torque term produces a plausible-looking but physically wrong simulation that passes ordinary dimensional/shape checks — the hardest kind of bug to catch, and again a direct instance of the "silent fallback to stale/default value" anti-pattern.
Fix: validate at `_make_vector_field` build time (not call time) that every `through_sources`/`sum_vars`/`node_b_sources` entry is statically resolvable (state, ground, input, param, or an earlier through-eq LHS) and raise `ValueError` naming any that are not, instead of deferring to a runtime `.get(tv, 0.0)`.
Effort: M.

**AC4** Dead code: `var_lookup` is fully built (with an `O(n²)` construction cost via `.index()` in a loop) but never read anywhere afterward | severity: **low** | area: code quality
Evidence: `assembly.py:314-330` — verified via full-file grep that `var_lookup` is never referenced again after construction; `_build_vals`/`vector_field` read `diff_vars`/`grounded`/`input_vars` directly instead.
Why it matters: pure waste, and its presence misleads a future reader into assuming it's load-bearing.
Fix: delete the "Pre-compute index maps" block and its stale explanatory comment.
Effort: S.

**AC5** Three independent re-implementations of alias-resolution fixed-point iteration, only one of which (`_build_vals`'s) is confirmed to apply `gear_ratio_scale` | severity: **medium** | area: correctness / duplication
Evidence: `assembly.py:588-600` (`_build_vals`, applies gear-ratio scaling), `:633-635` (`vector_field`'s `net_force` loop), `:658-660` (`sensor_fn`) — the latter two re-implement "walk `eliminated` until resolved" independently of the shared `_resolve` helper (`:274`) and independently of each other, rather than calling one shared routine.
Why it matters: if the two unshared copies are in fact redundant (since `vals` should already be fully resolved by the time they run), they're dead-but-risky code; if they are *not* redundant (i.e. they handle a case `_build_vals` doesn't), then the divergence in whether `gear_ratio_scale` is applied is a live correctness gap for gear-scaled through-variables reached via those two paths. Either way this needs to be resolved, not left ambiguous.
Fix: route all three call sites through one shared, parameterized `_resolve_alias(...)` helper; verify with a gear-ratio test that exercises all three paths identically.
Effort: S–M.

**AC6** `AcausalConnection`'s hand-written `__init__` signature doesn't match its declared `@dataclass` fields, breaking dataclass tooling assumptions | severity: **low** | area: code quality / future migration risk
Evidence: `acausal/base.py:137-159` — `@dataclass class AcausalConnection` declares fields `(element_a, port_a, element_b, port_b)` but overrides `__init__` to take `(port_a: tuple, port_b: tuple)`, a completely different signature.
Why it matters: dataclass tooling (`dataclasses.replace`, `dataclasses.asdict`, generic dataclass-based serialization) assumes the constructor signature matches the field list; `dataclasses.replace(conn, element_a="x")` would fail today because `replace` calls `__init__(**changed_fields)` with keywords that don't match the actual constructor. This is a latent trap for any future migration/serialization work on acausal specs (relevant given AC1/no schema versioning yet exists for these types).
Fix: either drop `@dataclass` (since its generated `__init__` isn't used anyway) or keep it and rename the convenience two-tuple constructor to a `@classmethod from_ports(...)`.
Effort: S.

**AC7** No schema/version identity on `AcausalElement`/`AcausalConnection`/`StateLayout` despite being the construction-time spec surface for a composite node intended for Studio | severity: **low-medium** (elevated by AC1 — once a builder exists, this becomes urgent) | area: CLAUDE.md migration policy
Evidence: `acausal/base.py` defines these as plain dataclasses with no `schema_version`/`schema_id` field anywhere.
Why it matters: if/when AC1 is resolved and these become persisted (Studio save/load, training-run manifests), there is currently no version field to gate future changes to `element_type` discriminator strings or connection tuple ordering.
Fix: add an explicit schema version once persistence lands (tied to AC1); until then, document that these types are construction-time-only and never directly serialized.
Effort: S now / M later.

---

## Architecture assessment

**The graph execution engine (`runtime/graph.py`) is the strongest-designed piece in scope.** It correctly compiles per-step Python node dispatch to trace time only, producing a single `lax.scan` for the actual rollout (`_call_with_iteration`, confirmed at `graph.py:1458/1490/1509`) — this directly answers the review brief's central architecture question in the affirmative: execution compiles to a scan, it does not re-dispatch in Python at runtime. Cycle detection, nested-graph recursion, and recurrent-initializer resolution are handled with real care and mostly-good error messages (raising `ValueError` with actionable text in the overwhelming majority of failure paths). The weakest points are (a) a repeated `cached_property`-on-frozen-Module pattern (G1, independently found again in `models/networks.py`'s `LeakyRNNCell` as N2) that technically violates the project's immutability convention even though it's currently safe by accident of how graph-surgery methods always construct fresh instances, and (b) a second, hand-maintained parallel implementation of the node-execution loop for pre-step cycle-initializer evaluation (G5) that risks silent divergence from the main loop.

**The execution/planning layer is architecturally disjoint from the JAX engine** — it is a cloud-orchestration DSL (RunPod/Modal/local shell-command and Python-source generation) that touches no JAX arrays or Equinox Modules at all, meeting the graph engine only through a CLI shell-out (`execute-training-run-spec`). Its dominant defect pattern is duplication without a shared source of truth: the same command-construction, provenance-collection, and path-rewrite logic is implemented two or three times across `backends.py`/`planning.py`/`container.py`, with small unverified divergences that are latent drift risks rather than active bugs today (E1–E5). The most CLAUDE.md-relevant gap here is that `ExecutionPlan`'s two largest payload fields are unschemad dicts despite the envelope itself being a versioned, durable, tracked artifact.

**The component adapter layer (`components/equinox.py`, `components/penzai.py`) has the most severe concrete correctness bugs found in this review.** Three of ~30 generated Equinox wrappers (`BatchNorm`, `MultiheadAttention`, `Dropout`) silently fail to thread a required argument (`state` or `key`) to the wrapped layer, meaning a Studio-configured knob has zero effect on execution with no error or warning — precisely the "what's configured silently diverges from what runs" failure mode the project's Core Principle exists to prevent at the graph-topology level, found here reproduced at the individual-leaf level. The root cause (A7) is architectural: the generator producing these ~30 wrapper classes does not appear to introspect each wrapped layer's actual call signature, so any equinox layer needing more than `(x)` risks silent mis-wrapping. Penzai's adapter is worse in kind: `bind_state` is an admitted no-op stub (A4) and `PenzaiSubgraph.__call__` silently no-ops to identity pass-through for uncallable models (A5) — both are textbook "just for now" workarounds the CLAUDE.md Core Principle explicitly calls bugs regardless of code comments labeling them as simplifications.

**`models/networks.py` is not a god-module in the coupling sense but does conflate three concerns** (generic staged-network architecture, a self-contained population-connectivity sub-system that's ~30% of the file and could be its own module, and graph-node adapter boilerplate). Its most consequential defect is the same silent-cell-dispatch pattern already tracked for LSTM (44183de) but shown here (N1) to be a general heuristic-dispatch mechanism, not an LSTM-specific gap — meaning the tracked issue's fix needs to address the mechanism, not just add an LSTM branch, or the next unsupported cell type will reproduce the same silent failure.

**`acausal/assembly.py` is a well-isolated, mostly trace-time-only equational compiler** that correctly pushes topology/indexing work to construction time rather than re-deriving it per call — a genuine architecture strength given the review's specific concern about per-call graph traversal. Its two real defects are a cycle-detection gap in through-equation topological sorting (AC2, silently produces wrong-but-plausible physics rather than raising) and a fully-unwired Studio registration (AC1) — `AcausalSystem` is presented to the UI as a composite node with no code path that can actually build one from a subgraph, a direct instance of the "absence of a subgraph is an error, not a condition to work around" principle currently having no enforcement point at all.

---

## Quick wins (small effort, real value)

1. **G6**: Change `Wire.temporality: str` to `Literal["instant", "recurrent"]` — one-line type tightening, closes a silent-typo footgun.
2. **G7**: Narrow `except Exception: continue` to `except AttributeError: continue` in `init_state_from_component`'s `_iter_state_indices` (`graph.py:77`).
3. **A2**: Forward `key=key` in the `MultiheadAttention` wrapper (`equinox.py:1355-1367`) — one line, restores dropout.
4. **A3**: Remove the unconditional `eqx.nn.inference_mode(...)` wrap in the `Dropout` component so the `inference` flag actually does something (`equinox.py:1449-1476`).
5. **A5**: Raise `TypeError` instead of silently identity-passing-through in `PenzaiSubgraph.__call__`'s uncallable-model branch (`penzai.py:544-550`).
6. **AC4**: Delete the dead `var_lookup` block (`assembly.py:314-330`).
7. **AC2**: Add cycle detection to `_topo_sort_through_eqs`, mirroring the already-correct pattern in `_resolve` (`assembly.py:277-280` vs `:692-702`).
8. **N5**: Delete the dead `encoder_mask = jnp.zeros(...)` line immediately overwritten in `SimpleStagedNetwork.__init__` (`networks.py:766-769`).
9. **N2 / G1**: Precompute `LeakyRNNCell.alpha`/`noise_std` as regular `__init__`-time fields instead of `cached_property` — small, mechanical, removes a frozen-Module mutation footgun in two places at once if applied as a repo-wide sweep.
10. **R1**: Raise in `_lower_loss_terms` when a loss leaf has neither `target_selector` nor `target_value`, instead of silently defaulting to zero at evaluation time (`retained_observables.py:311-320`).
11. **E4**: Add a one-line changelog comment at each `SCHEMA_VERSION` constant in `execution/models.py` noting what changed and why.
12. **AC6**: Rename `AcausalConnection`'s two-tuple convenience constructor to a `@classmethod from_ports(...)` so the dataclass-declared `__init__` matches its fields.
