# `LossTermSpec` Retirement Inventory

**Branch state used:** `integration/db41e6a-backend-hardening`. This branch's own ref
was deleted after merge (standard post-merge cleanup), but its tip commit is
recoverable as the second parent of the merge commit that landed it on
`develop`:

```
ae207269 Merge branch 'integration/db41e6a-backend-hardening' into develop
  parent 1: d330ea32  (develop before merge)
  parent 2: d026038b  (tip of integration/db41e6a-backend-hardening)
```

`git merge-base d026038b develop` returns `d026038b` itself, confirming `d026038b`
is on develop's ancestry and is exactly the pre-merge branch tip. **All citations
below are `git show d026038b:<path>`**, i.e. the state of files on
`integration/db41e6a-backend-hardening` immediately before its merge into
`develop`. Any explicit comparison to `develop`'s current tip is called out
separately and is not otherwise conflated.

**Correction to the task's assumed layout:** `LossTermSpec` is **not** defined
in `feedbax/objectives/spec.py`. That file (`feedbax/objectives/spec.py` on
this branch) only contains the *modern* `ObjectiveSpec`/`ReductionSpec`
machinery — there is no legacy class there at all. `LossTermSpec` is defined in
`feedbax/contracts/training.py:62`. `feedbax/objectives/service.py` imports it
from there (`feedbax/objectives/service.py:12`).

---

## 1. Field inventory

### 1.1 `LossTermSpec` — `feedbax/contracts/training.py:62-76`

Plain `pydantic.BaseModel` (not a `StrictModel`/`ObjectiveSpecModel`), no
`extra="forbid"`, no discriminator, no schema-identity field of its own.

| Field | Type | Default | Semantics (from usage in `service.py`) |
|---|---|---|---|
| `type` | `str` | required | Free-form string tag. Only two families are actually recognized during lowering (`_lower_loss_term`, `feedbax/objectives/service.py:573`): a leaf must have `type` in `{"TargetStateLoss", "target_state", "MatrixQuadraticLoss", "matrix_quadratic"}`; anything else raises `ObjectiveLoweringError` at `/type`. A term with non-empty `children` is treated as a composite container regardless of its own `type` string (the `type` value itself, e.g. `"Composite"`, is never checked for composites — see `service.py:573-576`, children take priority over type). |
| `label` | `str` | required | Display/identity label. Becomes the key under which the term's `AbstractLoss` is registered inside a `CompositeLoss.terms`/`weights` dict (`service.py:576-577`, `service.py:585-618`). |
| `weight` | `float` | `1.0` | Per-term scalar weight. For composites, `_lower_loss_term` takes each **child's own** `weight` to build the parent `CompositeLoss.weights` dict (`service.py:576`) — the composite node's own `weight` field is not applied by this path. `validate_loss_spec` (`service.py:336-423`) flags negative weights as invalid. |
| `selector` | `Optional[str]` | `None` | String selector address into runtime state, e.g. `"state.output"`, `"port:effector.position"`, `"probe:<id>"`, `"path:<dotted.path>"`. Required on any non-composite leaf (`service.py:578`: `if not term.selector: raise ...`). Resolved at runtime by `_select_runtime_value` (`service.py:696-743`), which special-cases string prefixes `state.`/`states.`/`trial_specs.`/`task_data.`/`model.`/`path:`; `port:`/`probe:` selectors raise `ObjectiveLoweringError` at lowering time because they require "retained-observable executor binding" not implemented in this service (`service.py:722-727`). |
| `target_selector` | `Optional[str]` | `None` | Alternative selector string pointing at a *runtime* target value (as opposed to a literal). Mutually exclusive with `target_value` (`service.py:579-583`: both set is an error). |
| `target_value` | `Optional[Any]` | `None` | Literal/constant target value (e.g. `[1.0, 0.0]`). For `target_state`-family terms, either `target_selector` or `target_value` must be set (`service.py:584-588`). For `matrix_quadratic`-family terms, `target_value` defaults to `0.0` when unset (`service.py:591-594`). |
| `retention` | `Optional[RetentionPolicySpec]` | `None` | How long a selected observable must be retained during graph execution (`mode: stream\|window\|trajectory`, `window_size`, `order`) — defined in `feedbax/contracts/graph.py:158-165`. **Not read anywhere in `feedbax/objectives/service.py`'s lowering path** — it is round-tripped through the Studio TS layer (`web/src/features/scenario/objectives.ts:330-339` reads/writes it) but has no effect on `_lower_loss_term`. |
| `norm` | `Optional[Literal["squared_l2","l2","l1","huber"]]` | `None` | Norm/metric kind. Defaults to `"squared_l2"` at lowering if unset (`service.py:596`: `norm=term.norm or "squared_l2"`). Consumed by `_metric_values` (`service.py:753-774`) and `_legacy_reduce_feature_metric` (`service.py:776-785`, see §3). Note the legacy `norm` enum is a strict subset of the modern `MetricKind` (`spec.py`'s `MetricKind` also allows `"squared"`, `"absolute"`) — `_metric_values` accepts `"squared"`/`"absolute"` as synonyms of `"squared_l2"`/`"l1"` but `LossTermSpec.norm`'s Pydantic `Literal` cannot express those aliases. |
| `matrix` | `Optional[Any]` | `None` | Raw nested-list/array matrix payload for quadratic terms. Validated by `_validate_matrix_payload` (`service.py:820-831`) against `matrix_kind`. |
| `matrix_kind` | `Optional[Literal["dense","diagonal"]]` | `None` | Shape discriminator for `matrix`. Defaults to `"dense"` when `matrix` is set but `matrix_kind` is not (`service.py:600`, `service.py:823`). |
| `time_agg` | `Optional[TimeAggregationSpec]` | `None` | Legacy time-reduction directive (fields below). Defaults to `TimeAggregationSpec(mode="mean")` at lowering if unset (`service.py:601`). |
| `children` | `Optional[Dict[str, "LossTermSpec"]]` | `None` | Recursive map of named child terms. Presence of any children makes the node a `CompositeLoss` container; `_lower_loss_term` recurses per child and ignores this node's own `selector`/`target_*`/`norm`/etc. entirely (`service.py:574-577`). |

### 1.2 `TimeAggregationSpec` — `feedbax/contracts/training.py:52-60`

| Field | Type | Default | Semantics |
|---|---|---|---|
| `mode` | `Literal["all","mean","sum","final","range","segment","custom"]` | `"all"` | Selects the time-axis reduction strategy. Only `"all"/"mean"`, `"sum"`, `"final"`, and `"range"` are actually implemented in the executable lowering path `_reduce_legacy_values` (`service.py:965-982`); `"segment"` and `"custom"` are recognized by `LossService.build_time_aggregation`/`validate_loss_spec`/`spec_to_loss_config` (the older dict-config path, `service.py:294-321`, `service.py:336-423`) but have **no implementation** in `_reduce_legacy_values` — passing `mode="segment"` or `mode="custom"` to the executable lowering path (`lower_loss_term_spec`) hits the `else: raise ObjectiveLoweringError(..., f"unsupported time aggregation {mode!r}")` branch at `service.py:980-981`. |
| `start` | `Optional[int]` | `None` | Start index for `mode="range"`. |
| `end` | `Optional[int]` | `None` | End index for `mode="range"`. |
| `segment_name` | `Optional[str]` | `None` | Named segment reference for `mode="segment"` (validated for presence only; unimplemented in executable lowering — see above). |
| `time_idxs` | `Optional[List[int]]` | `None` | Explicit index list for `mode="custom"` (validated for presence only; unimplemented in executable lowering). |
| `discount` | `Optional[Literal["none","power","linear"]]` | `None` | Discount-curve selector. **Not read anywhere in `_reduce_legacy_values`** — only surfaced through `build_time_aggregation`'s dict-config output (`service.py:317-320`), which is a dead-ish side path (see §2). No executable discount weighting exists for the legacy pipeline. |
| `discount_exp` | `Optional[float]` | `None` | Exponent for `discount="power"`. Same caveat as `discount`. |

### 1.3 Modern `ObjectiveSpec`/`ReductionSpec`/`SelectorAddressSpec` family — `feedbax/objectives/spec.py`

The modern selector-addressing type is named **`SelectorAddressSpec`** (confirmed at `feedbax/objectives/spec.py:44-52`) — matches the task's guess.

**`SelectorAddressSpec`** (`spec.py:44-52`):

| Field | Type | Default | Semantics |
|---|---|---|---|
| `selector` | `str` (`min_length=1`) | required | Same string-address role as legacy `selector`, but always required (no optional/None state). |
| `kind` | `SelectorKind` (`"state"\|"port"\|"wire"\|"graph_output"\|"recurrent_carry"\|"task_data"\|"objective"\|"probe"`) | `"state"` | Declares which selector namespace the address lives in — this is metadata; the *lowering* code (`_select_runtime_value`) still dispatches purely on the string prefix of `selector`, not on `kind` (`service.py:696-727`). |
| `value_dtype` | `Optional[str]` | `None` | Declarative dtype metadata (not consumed by lowering). |
| `value_shape` | `Optional[list[Any]]` | `None` | Declarative shape metadata (not consumed by lowering). |
| `value_units` | `Optional[str]` | `None` | Declarative units metadata. |
| `temporal_axis` | `Optional[Literal["time","step","sample"]]` | `"time"` | Declares which axis is temporal. `FiniteDifferenceLossSpec` requires this be non-`None` (`spec.py:224-225`). |
| `feature_axis` | `Optional[Literal["feature","coordinate","unit","channel"]]` | `None` | Declares which axis is the feature axis. Required for `MatrixQuadraticLossSpec` (`spec.py:255-256`). |
| `metadata` | `dict[str, Any]` | `{}` | Free-form metadata bag. |

**`ReductionSpec`** (`spec.py:180-186`):

| Field | Type | Default | Semantics |
|---|---|---|---|
| `time` | `TimeReductionKind` (`"mean"\|"sum"\|"none"\|"final"`) | `"mean"` | Time-axis reduction, applied via `_reduce_objective_values` (`service.py:923-963`) at `axis=1` of the (trial, time, ...) value tensor. |
| `trial` | `ReductionKind` (`"mean"\|"sum"\|"none"\|"tail"`) | `"mean"` | Trial/batch-axis reduction, applied at `axis=0`, **after** time reduction. This is the field that is entirely absent from the legacy shape — legacy always means over trials unconditionally (see §3). |
| `feature` | `ReductionKind` | `"sum"` | Feature-axis reduction, applied at `axis=-1`, **before** time reduction (see order in `_reduce_objective_values`, `service.py:929-936`). |
| `tail_fraction` | `float` (`0 < x <= 1`) | `0.1` | Fraction of (sorted, ascending) values kept for `kind="tail"` reductions — used for CVaR-like risk aggregation (`_apply_reduction`, `service.py:984-1004`). No legacy equivalent exists. |
| `empty_mask` | `Literal["zero","error"]` | `"zero"` | Declared behavior for an empty post-mask window; **not read anywhere in `service.py`** — masks are always converted to zero/one weights via `_timeline_mask` (`service.py:856-880`), so an "all excluded" mask silently multiplies by zero regardless of this field's value. This field currently has no enforcement path in the lowering service.

**`ObjectiveSpec`** container (`spec.py:311-319`): `kind: Literal["objective_spec"]`, `schema_version: str = OBJECTIVE_SCHEMA_VERSION` (`"feedbax.spec.objective.v1"`), `terms: list[ObjectiveTermSpec]`, `timeline: Optional[TaskTimelineSpec]`, `metadata: dict`. This is the container-level schema-identity field that `LossTermSpec` entirely lacks (see §5).

Also relevant to mapping (all in `spec.py`):
- **`MetricSpec`** (`spec.py:146-158`): `kind: MetricKind` (`"squared_l2"|"l2"|"l1"|"squared"|"absolute"|"huber"`, default `"squared_l2"`), `axis`, `huber_delta: Optional[float]` (required iff `kind="huber"` — this is the configurable Huber threshold that `LossTermSpec.norm="huber"` has no equivalent field for; the legacy huber path hardcodes the threshold to `1.0`, see §3).
- **`EpochMaskSpec`** (`spec.py:83-93`): `epochs: list[str]`, `mode: "include"|"exclude"`, `alignment: "sample"|"right_edge"`. No legacy equivalent at all (see §6).
- **`ScheduleSpec`** union (`ConstantScheduleSpec`/`PowerLawScheduleSpec`/`MovementEpochRampScheduleSpec`, `spec.py:96-144`): time-varying weighting curves. No legacy equivalent — `LossTermSpec.time_agg.discount` is the closest legacy analog but is unimplemented in the executable path (see 1.2).
- **`TargetValueSpec`** (`spec.py:55-68`): `kind: "constant"|"selector"|"task_target"`, `value`, `selector: Optional[SelectorAddressSpec]`, `target_key: Optional[str]`. Generalizes legacy's separate `target_selector`/`target_value` fields into one tagged union, plus adds a third `task_target` kind (keyed lookup into `trial_specs.targets.<target_key>`, `service.py:690-693`) that has no legacy equivalent.
- **`MatrixPayloadSpec`** (`spec.py:161-177`): `kind: "dense"|"diagonal"`, `value`, `dtype`, `metadata`. Same semantic role as legacy's separate `matrix`/`matrix_kind` fields, merged into one object.

---

## 2. Dispatch map

Dispatch into legacy vs. modern reduction happens **inside a single shared
executable-loss class**, `SelectorObjectiveLoss` (`feedbax/objectives/service.py:91-159`),
not via two separate classes. The class carries both a `reduction:
ReductionSpec | None` field and a `time_agg: TimeAggregationSpec | None` field;
exactly one is populated for a given lowered term, and `term()`'s body branches
on which is present:

```python
# feedbax/objectives/service.py:140-158
if self.reduction is None:
    values = _legacy_reduce_feature_metric(values, self.norm)
values = _apply_time_weights(
    values, mask=self.mask, schedule=self.schedule, timeline=self.timeline, path=self.path,
)
if self.reduction is not None:
    return _reduce_objective_values(
        values, self.reduction, norm=self.norm, path=f"{self.path}/reduction",
    )
return _reduce_legacy_values(values, self.time_agg, path=f"{self.path}/time_agg")
```

So the actual gate is `self.reduction is None` (legacy path) vs. `self.reduction
is not None` (modern path) — **confirmed real names, matching the task's
guesses**: `_reduce_legacy_values` (`service.py:965`) and
`_reduce_objective_values` (`service.py:923`) both exist verbatim.
`_legacy_reduce_feature_metric` (`service.py:776`) also exists verbatim and has
no modern counterpart function name of the same shape — the modern feature
reduction is just the generic `_apply_reduction` (`service.py:984`) called with
`axis=-1` inside `_reduce_objective_values`.

Which caller sets which: the two `SelectorObjectiveLoss` construction sites are
mutually exclusive by construction, not by any runtime `isinstance` check on
the incoming spec inside `SelectorObjectiveLoss` itself:

- `LossService._lower_loss_term` (`service.py:573-618`), called only from
  `lower_loss_term_spec` (`service.py:514-538`), always builds:
  ```python
  # service.py:604-618
  return SelectorObjectiveLoss(
      ..., time_agg=term.time_agg or TimeAggregationSpec(mode="mean"),
      reduction=None, mask=None, schedule=None, timeline=None,
      difference_order=0, path=path,
  )
  ```
  i.e. `reduction` is hardcoded to `None` — this is the only place `LossTermSpec`
  objects reach `SelectorObjectiveLoss`.

- `LossService._lower_objective_term` (`service.py:619-655`), called only from
  `lower_objective_spec` (`service.py:540-571`), always builds:
  ```python
  # service.py:643-655
  return SelectorObjectiveLoss(
      ..., time_agg=None, reduction=term.reduction, mask=term.mask,
      schedule=term.schedule, timeline=timeline,
      difference_order=term.order if isinstance(term, FiniteDifferenceLossSpec) else 0,
      path=path,
  )
  ```
  i.e. `time_agg` is hardcoded to `None` and `reduction=term.reduction` is
  always a real `ReductionSpec` (required, non-optional field on
  `ObjectiveTermBase`, `spec.py:190`).

**One level up**, the actual entry-point dispatch that decides which of these
two lowering functions gets called at all is on `ObjectiveSlotSpec.kind`
(`feedbax/contracts/training.py:187-197`), consumed by
`LossService.lower_objective_slot` (`service.py:482-512`):

```python
# service.py:490-509
if slot.kind == "loss_term":
    if slot.loss is None:
        raise ObjectiveLoweringError(f"{path}/loss", "loss_term slot is missing loss")
    return self.lower_loss_term_spec(slot.loss, graph=graph, trial_axis=trial_axis, path=f"{path}/loss")
if slot.kind == "objective_spec":
    if slot.payload is None:
        raise ObjectiveLoweringError(f"{path}/payload", "objective_spec slot is missing payload")
    return self.lower_objective_spec(slot.payload, trial_axis=trial_axis, path=f"{path}/payload")
raise ObjectiveLoweringError(path, f"objective kind {slot.kind!r} is external and cannot be lowered by Feedbax")
```

This is the real, top-level "does this training run's objective route into
the legacy pipeline or the modern one" gate. It is a plain string-literal
`==` comparison on `ObjectiveSlotSpec.kind: Literal["loss_term","objective_spec","external"]`
(default `"loss_term"`, `feedbax/contracts/training.py:189`), not an
`isinstance`/`hasattr` check on the payload shape itself.

**Runtime entry point** into this dispatch: `feedbax/training/executor.py:392-408`
(`_lower_objective`) calls `loss_service.lower_objective_slot(spec.objective, ...)`
where `spec: TrainingRunSpec` — i.e. the modern
`TrainingRunSpec.objective: ObjectiveSlotSpec` field is what the executor
actually reads. The separate, older `TrainingSpec.loss: LossTermSpec` field
(`feedbax/contracts/training.py:91`) is a different, parallel container used
only by the Studio backend's `/loss/validate` and `/loss/resolve-selector`
endpoints (`feedbax/web/api/training.py:184-220`) and by
`feedbax/runtime/retained_observables.py` — it is not consumed by
`feedbax/training/executor.py` at all. **Both `TrainingSpec.loss` (direct
`LossTermSpec`) and `TrainingRunSpec.objective` (`ObjectiveSlotSpec`, which can
itself wrap a `LossTermSpec` via `kind="loss_term"`) are live legacy entry
points into `LossTermSpec`; the migration must account for both containers**,
not just `ObjectiveSlotSpec`.

`AbstractLoss`/`CompositeLoss`/`TermTree` in `feedbax/objectives/loss.py` sit
strictly downstream of this dispatch: every lowered term, legacy or modern,
becomes a `SelectorObjectiveLoss(AbstractLoss)` or a `CompositeLoss` built
from those (`service.py:576-577`, `service.py:585-590`). Neither pipeline
constructs `TargetStateLoss`/`NthDifferenceLoss`/etc. from `loss.py` directly —
`loss.py`'s richer term classes are unused by this service and are out of
scope per the task's framing; confirmed no further inventory effort was spent
there beyond this dispatch-boundary note.

---

## 3. Semantic diff table

All rows below reflect what the *executable* lowering computes
(`SelectorObjectiveLoss.term`, `service.py:109-159`), not the older
dict-producing `spec_to_loss_config`/`build_time_aggregation` path
(`service.py:294-321`, `424-479`), which is a separate, non-executing
config-dict output used only by `validate_loss_spec`'s sibling helper and is
not wired into any `AbstractLoss` — flagged separately at the end of this
section since it is a second, smaller legacy surface.

Runtime value tensor shape convention observed in the tests
(`tests/test_loss_service.py`): `(trial, time, feature)`, i.e. axis 0 = trial,
axis 1 = time, axis -1 = feature. This matches `_apply_time_weights`'s
`arr.shape[1]` time-length assumption (`service.py:833-847`) and
`_reduce_objective_values`'s axis choices (`-1` then `1` then `0`,
`service.py:923-963`).

| Metric (norm) | Legacy computation (`reduction=None` path) | Modern computation (`reduction` set) | Numerically identical when? |
|---|---|---|---|
| `squared_l2` / `squared` | `_metric_values` squares element-wise (`service.py:764-765`, `770-771`) → `_legacy_reduce_feature_metric`: for `ndim>=3` (has a feature axis), **sums over feature axis** unconditionally (`service.py:781-783`); for `ndim<3`, feature step is a no-op → time-weighted (`_apply_time_weights`) → `_reduce_legacy_values`: time step is `mean` unless `time_agg.mode` is `sum`/`final`/`range` (`service.py:965-981`) → **always finishes with an unconditional `jnp.mean(arr)` over the trial axis** (`service.py:982`, no branch — every legacy term is trial-mean regardless of `time_agg`). | `_metric_values` squares element-wise (identical) → `_reduce_objective_values`: feature axis reduced first via `_apply_reduction(axis=-1, kind=reduction.feature, ...)` (configurable: `mean`/`sum`/`none`/`tail`) → time axis reduced via `reduction.time` (configurable: `mean`/`sum`/`none`/`final`) → trial axis reduced via `reduction.trial` (configurable: `mean`/`sum`/`none`/`tail`). | Identical **only** when `reduction.feature="sum"`, `reduction.trial="mean"`, and `reduction.time` matches the legacy `time_agg.mode` (`"all"`/`"mean"`→`"mean"`, `"sum"`→`"sum"`, `"final"`→`"final"`), and no mask/schedule is applied (legacy has no mask/schedule support at all — see below). This exact case (`time="sum", trial="mean", feature="sum"`) is asserted equal to a hand-written legacy-style reduction in `tests/test_loss_service.py:432-448` (`test_loss_term_lowers_to_executable_loss_matching_hand_reduction`) and again for `l2` at `tests/test_loss_service.py:471-484,486-501`. |
| `l2` | Same feature step as `squared_l2` but `_legacy_reduce_feature_metric` additionally takes `jnp.sqrt` after the feature-axis sum (`service.py:781-783`; for `ndim<3` it takes `sqrt` of the raw squared value, `service.py:779-780`) — i.e. **sqrt is applied once, immediately after the feature-sum, before time weighting/reduction.** | `_metric_values` squares (same as squared_l2, `service.py:770-771`) → `_reduce_objective_values` applies feature reduction (sum) **then** takes `sqrt` immediately after the feature reduction if `arr.ndim>=3` (`service.py:929-933`: `if norm=="l2": arr = jnp.sqrt(arr)`), or takes `sqrt` directly if `arr.ndim<3` and `norm=="l2"` (`service.py:934-935`) — same point in the pipeline (right after feature sum, before time reduction) as legacy. | Identical under the same conditions as `squared_l2` (feature=`"sum"`, trial=`"mean"`, time mode matched) — explicitly verified by `test_loss_term_l2_uses_euclidean_feature_norm` vs. `test_objective_spec_l2_uses_euclidean_feature_norm` (`tests/test_loss_service.py:471-501`), both computing `jnp.mean(jnp.sum(jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1)), axis=1))`. No discrepancy found for this metric under matched reduction settings. |
| `l1` / `absolute` | `_metric_values` takes `jnp.abs` (`service.py:766-767`) → `_legacy_reduce_feature_metric` sums over feature axis for `ndim>=3` (no sqrt branch for l1, `service.py:781-785`: only `l2` gets special-cased; `l1` falls through to the plain `jnp.sum(arr, axis=-1)`) → same trial-mean finish. | `_metric_values` takes `jnp.abs` (identical) → `_reduce_objective_values` feature-reduces (sum), no sqrt branch triggers since `norm != "l2"` → time/trial reduced per `ReductionSpec`. | Identical when `feature="sum"`, `trial="mean"`, `time` matches legacy mode. No test explicitly exercises `l1`/`absolute` end-to-end in `test_loss_service.py`, but the code paths are structurally parallel to the verified `squared_l2` case — no divergent branch was found in either function for this norm. |
| `huber` | `_metric_values` huber branch (`service.py:772-774`) hardcodes the Huber threshold to a **literal `1.0`**: `jnp.where(abs_arr <= 1.0, 0.5*jnp.square(arr), abs_arr - 0.5)`. `LossTermSpec` has no field to configure this threshold. | Same `_metric_values` function is shared (huber branch is norm-string-gated, not spec-shape-gated) — **also hardcodes `1.0`** in the current lowering, even though `MetricSpec.huber_delta` exists as a configurable field on the modern spec (`spec.py:150-158`, required when `kind="huber"`). `SelectorObjectiveLoss._lower_objective_term` passes `norm=term.metric.kind` (`service.py:632`) but **never passes `term.metric.huber_delta` through to `_metric_values`** — the delta is validated at the spec layer but silently dropped before reaching the shared metric function. | **FLAGGED: numerically identical today only because the modern path also ignores `huber_delta` and hardcodes 1.0 — this is itself a latent bug in the modern pipeline, not a migration-introduced discrepancy.** Any future fix to thread `huber_delta` through `_metric_values` before this migration lands would silently change modern-huber behavior for any `huber_delta != 1.0`; the migration mapping must not assume `huber_delta` is currently honored. |
| Matrix quadratic (`matrix_kind` dense/diagonal) | `_metric_values` routes to `_matrix_quadratic_values` when `matrix is not None` (`service.py:757-758`) — same function object used by both dispatch paths (norm/metric string is not consulted at all once a matrix is present). Feature axis is already collapsed by the einsum/dot inside `_matrix_quadratic_values` (`service.py:787-819`), so `_legacy_reduce_feature_metric`'s `ndim>=3` branch is never reached for matrix terms (the quadratic form leaves `ndim` one lower already) — legacy's feature-reduction step is effectively a no-op post-matrix. | Same `_matrix_quadratic_values` function; same effective no-op for `reduction.feature` post-matrix (the array is already feature-reduced by the einsum). `MatrixQuadraticLossSpec` additionally validates `metric.kind == "squared_l2"` is required (`spec.py:257-262`) — a constraint `LossTermSpec` has no equivalent enforcement for (legacy simply ignores `norm` once `matrix` is set, per `_metric_values`'s early-return at `service.py:757-758`). | Identical value computation (verified in `test_objective_spec_lowers_matrix_mask_and_schedule_terms`, `tests/test_loss_service.py:504-541`, against a hand-written `jnp.sum(jnp.square(diff) * diag, axis=-1)` reduction) when `trial="mean"` and no mask/schedule (legacy has neither). No numeric divergence found beyond the trial/mask/schedule gaps already noted. |
| Time mode `"all"`/`"mean"` vs `reduction.time="mean"` | `_reduce_legacy_values`: `jnp.mean(arr, axis=1)` (`service.py:975-976`). | `_apply_reduction(axis=1, kind="mean", ...)`: `jnp.mean(values, axis=1)` (`service.py:993-994`). | Identical. |
| Time mode `"sum"` vs `reduction.time="sum"` | `jnp.sum(arr, axis=1)` (`service.py:977-978`). | `jnp.sum(values, axis=1)` (`service.py:995-996`). | Identical. |
| Time mode `"final"` vs `reduction.time="final"` | `jnp.take(arr, -1, axis=1)` (`service.py:979-980`). | `jnp.take(arr, -1, axis=1)` (`service.py:940-941`) — legacy's `"final"` branch was evidently copied verbatim into the modern path. | Identical. |
| Time mode `"range"` | `jnp.mean(arr[:, start:end], axis=1)` (`service.py:980-981`) — **always a mean over the slice**, `start`/`end` required. | **No `"range"` option in `TimeReductionKind`** (`"mean"|"sum"|"none"|"final"` only, `spec.py:19`). | **FLAGGED — no-equivalent case.** A legacy term using `time_agg.mode="range"` has no direct modern `ReductionSpec.time` value to migrate to. The closest modern approximation (masking the desired range via `EpochMaskSpec`/`TaskTimelineSpec` epochs, then `time="mean"`) requires the timeline to declare an epoch with a matching `length_range`, which is a structurally different mechanism (named epochs vs. raw integer indices) and is not a value-preserving 1:1 field copy — needs a migration rule that either fabricates a synthetic timeline epoch or is rejected as `equivalent-with-migration-rule` at best. |
| Time mode `"segment"` / `"custom"` | **Unimplemented** in the executable path — `_reduce_legacy_values`'s `mode` variable takes the literal value from `time_agg.mode`, and neither `"segment"` nor `"custom"` matches any of the `if/elif` branches (`service.py:970-981`), so it falls through to `else: raise ObjectiveLoweringError(path, f"unsupported time aggregation {mode!r}")` at `service.py:980-981`. | No modern equivalent (`TimeReductionKind` has no segment/custom analog). | **FLAGGED — legacy itself cannot execute these modes today; not a migration regression, but any saved graph with `time_agg.mode in {"segment","custom"}` cannot be lowered by either pipeline currently and needs explicit handling (reject-with-clear-error is the safe migration stance, matching existing legacy behavior).** |
| Trial axis (all norms) | **Always `jnp.mean(arr)` over axis 0, unconditionally** (`service.py:982`) — there is no legacy field that can request `sum`/`none`/`tail` over trials. | `reduction.trial` is a first-class configurable field (`mean`/`sum`/`none`/`tail`), applied via `_apply_reduction(axis=0, ...)` (`service.py:958-963`). | **FLAGGED — only equivalent when the target `ReductionSpec.trial` is explicitly forced to `"mean"`.** Any migration that defaults new specs to a different `trial` reduction (e.g. `"tail"` for CVaR-style objectives) would silently change every previously-`LossTermSpec`-authored term's aggregate semantics if the migration rule does not hardcode `trial="mean"` for legacy-sourced terms. This is the single largest structural gap between the two pipelines. |
| Mask / schedule weighting | **No mask or schedule support at all.** `SelectorObjectiveLoss.term` only applies `_apply_time_weights` with `mask=None, schedule=None` when lowered from `_lower_loss_term` (`service.py:610-611`); `_apply_time_weights` is a no-op multiply-by-ones when both are `None` (`service.py:833-847`: `weights = jnp.ones(...)`, neither `if` branch triggers). | `EpochMaskSpec`/`ScheduleSpec` are first-class, applied via `_timeline_mask`/`_schedule_weights` (`service.py:856-921`) before time reduction. | **FLAGGED as a one-directional gap, not a numeric-divergence-on-shared-input case**: legacy specs never carry mask/schedule data, so there is nothing to migrate incorrectly here — but it means legacy-authored objectives that a user *wants* to add epoch-gating or ramp-scheduling to have no legacy-side representation; this is a `no-equivalent-needs-design` field addition, not a value-changing reinterpretation of existing data. |
| `tail_fraction` / `reduction.*="tail"` | No equivalent; legacy trial axis is hardcoded mean, feature/time have no `"tail"` option in `TimeAggregationSpec.mode`/legacy feature reduction. | `_apply_reduction`'s `"tail"` branch (`service.py:998-1003`) sorts ascending and means the top `ceil(size * tail_fraction)` elements — a CVaR/worst-case-tail statistic. | **No shared representation** — `no-equivalent-needs-design`, not a numeric-difference case (nothing to compare against since legacy cannot express it). |

**Separate flag — the non-executing `spec_to_loss_config`/`build_time_aggregation`
dict path** (`service.py:294-321`, `424-479`): this is a second, smaller
legacy surface that converts `LossTermSpec` → a plain `dict` (not an
`AbstractLoss`). It is used only by `LossService.spec_to_loss_config` and
tested in `TestSpecToLossConfig` (`tests/test_loss_service.py:349-407`); no
call site in `feedbax/training/executor.py` or `feedbax/web/api/training.py`
was found consuming this dict's output for actual loss execution (searched
`feedbax/` and `tests/` for `spec_to_loss_config(` call sites — only the test
file constructs it). **This function appears to be dead/unused for training
execution and should be explicitly scoped in or out of the migration** rather
than silently ported, since it duplicates `_lower_loss_term`'s config
extraction with different, incomplete semantics (e.g. it never validates
`matrix`/`matrix_kind` shape, and its `time_aggregation` dict never actually
executes a reduction — it only stages the fields).

---

## 4. Producer inventory

### 4.a feedbax repo (`integration/db41e6a-backend-hardening`, via `git show d026038b`)

No `LossTermSpec(` constructor call site exists in feedbax's non-test source —
Studio never directly instantiates the class in Python; it always arrives as
a deserialized Pydantic payload from an HTTP request body or a saved
manifest. Full grep-confirmed call-site inventory:

**Tests only** (all `LossTermSpec(` constructor calls in the tree):

| File:line | Snippet | What it's doing |
|---|---|---|
| `tests/test_checkpoint_custody.py:92` | `loss=LossTermSpec(type="target_state", label="target", selector="output")` | Builds a minimal `TrainingSpec.loss` for a checkpoint-custody manifest round-trip test. |
| `tests/test_execution_contract.py:138` | `loss=LossTermSpec(type="target_state", label="target", selector="output")` | Same minimal pattern, for the execution-contract test suite. |
| `tests/test_loss_service.py` (18 call sites: lines 278, 290, 301, 312, 322, 327, 333, 350, 369, 385, 390, 396, 450, 469, 568, 577, 583, 592) | e.g. `LossTermSpec(type="TargetStateLoss", label="Position Error", weight=1.0, selector="port:effector.position", norm="squared_l2", time_agg=TimeAggregationSpec(mode="all"))` (`:278-284`) | Direct unit coverage of `LossService.validate_loss_spec`, `spec_to_loss_config`, and `lower_loss_term_spec`/`_lower_loss_term` — this is the authoritative test surface for legacy lowering semantics (source of the semantic-diff evidence in §3). |
| `tests/test_retained_observables.py` (17 call sites: lines 138, 314, 337, 367, 397, 413, 426, 444, 461, 532, 559, 585, 605, 626, 630, 638, 667, 698) | e.g. `LossTermSpec(type="target_state", label="pos", selector="state.output", target_value=[0.0, 0.0], time_agg=TimeAggregationSpec(mode="mean"))` (pattern repeated with variations) | Exercises `feedbax/runtime/retained_observables.py`'s retention-plan derivation from a `TrainingSpec`/`LossTermSpec` tree — validates which selectors imply which `RetainedObservableSpec`s. |
| `tests/test_training_run_executor.py:68` | `loss=LossTermSpec(...)` (inside a `TrainingSpec` fixture) | Exercises the training-run executor's handling of a `TrainingSpec`-shaped payload (legacy container), separate from the `ObjectiveSlotSpec` path. |
| `tests/test_training_run_spec.py:58` | `loss=LossTermSpec(type="target_state", label="target", selector="output")` | Minimal `TrainingSpec` fixture for `TrainingRunSpec`-adjacent schema tests. |
| `tests/test_provider_contract.py:390` | `"LossTermSpec": LossTermSpec,` (used as a registry-snapshot type map entry, not a constructor call) | Asserts the provider's schema registry snapshot exposes `LossTermSpec` as a registered model type (see §5). |

**Non-test references** (imports / type-usage / schema-registration; no
constructor calls) confirming every surface that carries the type through the
system:

| File:line | Role |
|---|---|
| `feedbax/contracts/__init__.py:151,311` | Re-exports `LossTermSpec` from the package's public `__all__`. |
| `feedbax/contracts/training.py:62,76,91,192,585` | Definition (`:62`), self-referential `children` field (`:76`), `TrainingSpec.loss: LossTermSpec` (`:91`), `ObjectiveSlotSpec.loss: LossTermSpec | None` (`:192`), `LossTermSpec.model_rebuild()` forward-ref fixup (`:585`). |
| `feedbax/integrations/provider.py:95,245,918-922,1254` | Imports it; registers it in a type-map dict (`:245`); builds a `loss_registry_snapshot()` `RegistryEntry` describing it for the provider's schema catalog with `type_id="feedbax.objectives.loss.LossTermSpec"` (`:918-922`, note the type_id string names the wrong module — `feedbax.objectives.loss` rather than `feedbax.contracts.training`, a pre-existing naming inconsistency worth flagging for the migration doc); `_validate_loss_term(term: LossTermSpec, path: str)` recursive weight/time_agg validator (`:1254`, mirrors `service.py`'s own `validate_loss_spec` with an independent, slightly different implementation — a second validator surface). |
| `feedbax/objectives/service.py:12,337,351,425,516,522,573` | The primary lowering-service consumer (fully inventoried in §2–3). |
| `feedbax/runtime/retained_observables.py:28,218,233,277,473,948` | Consumes `LossTermSpec`/`TrainingSpec` to derive retained-observable/retention plans (`_validate_matrix_loss_term` at `:948` mirrors `service.py`'s own matrix validation independently — a third, separate validator). |
| `feedbax/web/api/training.py:11,187` | FastAPI `ValidateLossRequest.loss_spec: LossTermSpec` — the live HTTP endpoint (`POST /loss/validate`) that accepts raw `LossTermSpec` JSON from Studio's frontend. |
| `feedbax/web/models/__init__.py:29,71` | Re-exports `LossTermSpec` into the web package's public model surface. |
| `scripts/generate_studio_contracts.py:114,183` | Includes `LossTermSpec` in the list of Pydantic models whose JSON Schema is codegen'd into `web/src/generated/studioContracts.ts` (confirmed output at that path, §5). |

**Studio frontend (`web/`)** — this is where `LossTermSpec` is actually
*authored* (constructed as plain JS/TS object literals, not class
instantiation, since it's a generated `interface`, not a runtime
constructor). Key producer sites:

| File:line | Snippet | What it's doing |
|---|---|---|
| `web/src/components/canvas/PortContextMenu.tsx:132-138` | `const newTerm: LossTermSpec = { type: 'TargetStateLoss', label: ..., weight: 1, selector: ... }` (approx., full object literal at `:132`) | Builds a new loss term when a user adds a loss target from a canvas port's context menu. |
| `web/src/components/modals/AddLossTermModal.tsx:50` | `const newTerm: LossTermSpec = {...}` | Builds a new loss term from the "Add Loss Term" modal dialog. |
| `web/src/features/loss/operations.ts:11-233` (multiple functions: `findLossTerm`, `updateLossTermAtPath`, `insertLossTermAtPath`, `removeLossTermAtPath` (via `cloneLossTerm`), `collectLeafTerms`, `countLossTerms`, `generateUniqueKey`, `collectSelectors`) | The core client-side tree-manipulation library for editing a `LossTermSpec` tree in the Studio UI (add/remove/update/clone nodes by path). |
| `web/src/stores/trainingStore.ts:161-363` | Zustand store actions `updateLossTerm`, `addLossTerm`, plus private helpers `updateLossTermAtPath`/`insertLossTermAtPath`/`removeLossTermAtPath` (name-shadowing `operations.ts`'s versions — appears to be a **second, parallel implementation** of the same tree-editing logic, not a re-export) | Studio's `TrainingSpec.loss` state management — the actual mutation surface a user's edits flow through when editing the loss tree in the training panel. |
| `web/src/stores/workspaceStore.ts:216-259` (`objectiveSpecFromLossSpec`) | Converts a `LossTermSpec` tree into a `StudioObjectiveSpec` (walks children, flattens leaves into `StudioObjectiveTermSpec[]`, and stashes the original tree verbatim in `legacy_loss_spec`) | One direction of Studio's own legacy↔modern bridge — **already exists on this branch**, independent of the `feedbax.objectives` migration this report is scoping. |
| `web/src/features/scenario/objectives.ts:343-354` (`lossSpecFromObjectiveSpec`) | The reverse conversion: derives a `LossTermSpec` composite tree from a `StudioObjectiveSpec`'s terms, for whatever consumer still needs the legacy shape | The other direction of the same bridge. |
| `web/src/api/client.ts:262` | `lossSpec: LossTermSpec` (API client type for a request/response body) | Typed API client surface for the `/loss/validate` endpoint. |

**Studio-schema-level identity of `LossTermSpec` in TS**:
`web/src/generated/studioContracts.ts:564-587,1618-1635` — this is the
codegen'd output of `scripts/generate_studio_contracts.py`, containing both the
TS `interface LossTermSpec` and a matching Zod schema `LossTermSpecSchema`
(recursive via `z.lazy`). This file is fully derived (not hand-authored) from
the Python `LossTermSpec.model_json_schema()` — any field change to the Python
model must be re-synced through this codegen script, not hand-edited.

### 4.b rlrmp repo (current working-tree state, `/sessions/serene-jolly-ptolemy/mnt/rlrmp`, `src/` and `scripts/`)

**Zero producer call sites.** Grepped `src/` and `scripts/` for
`LossTermSpec`, `loss_term`, and `legacy_loss_spec` — no matches for the
Pydantic type or its constructor. The only string-adjacent hits are entirely
unrelated:

- `rlrmp.loss` module functions named `goal_hit_pos_loss_term_fn`,
  `goal_hit_vel_loss_term_fn`, `post_hit_pos_loss_term_fn`,
  `goal_hit_late_pos_loss_term_fn` (`src/rlrmp/loss.py:115-138`) — these are
  plain JAX callables passed into feedbax's `feedbax.objectives.loss`
  (`AbstractLoss`/`CompositeLoss`) machinery directly, with no `LossTermSpec`
  involved at all.
- `active_loss_term_labels(run_spec)` in
  `src/rlrmp/analysis/pipelines/{gru_pilot_figures,gru_checkpoint_selection}.py`
  — reads label strings back out of an already-materialized run-spec
  `Mapping`, not a `LossTermSpec` construction.

**Conclusion: rlrmp does not produce `LossTermSpec` payloads at all** — it
consumes feedbax's `objectives.loss` API at a lower level, bypassing the
Studio-facing spec layer entirely. This migration has **no rlrmp-side blast
radius** for producer call sites.

---

## 5. Schema identity status

**No, `LossTermSpec` carries no schema/version field today, at any level.**

- The Pydantic model itself (`feedbax/contracts/training.py:62-76`) has no
  `schema_version`, `kind`, `type` used as a schema discriminator (its `type:
  str` field is a free-form loss-kind tag, not a schema-version tag — see §1),
  or any other version marker. It is a plain `BaseModel`, not a
  `StrictModel`/`ObjectiveSpecModel`-family durable-spec base.
- Its *container* fields carry no version either: `TrainingSpec.loss:
  LossTermSpec` (`training.py:91`) and `ObjectiveSlotSpec.loss:
  Optional[LossTermSpec]` (`training.py:192`) are unversioned direct
  references. `ObjectiveSlotSpec` itself does carry `schema_id: str | None`
  and `schema_version: str | None` fields (`training.py:194-195`), but these
  are documented/used only for the `payload` branch (`kind="objective_spec"`
  or `"external"`) — the `_validate_payload` validator (`training.py:198-202`)
  only requires `loss` to be set when `kind="loss_term"`; it never requires or
  reads `schema_id`/`schema_version` in that branch, so a `loss_term`-kind
  slot is never schema-stamped in practice.
- The **only place `LossTermSpec` is named at all in the schema-migration
  registry** (`feedbax/contracts/migrations.py:1296-1303`) is a declarative
  `SpecSchemaFamily` entry:
  ```python
  _family(
      "LossTermSpec",
      "feedbax.spec.training.loss_term",
      "feedbax.spec.training.loss_term.v1",
      owner_module="feedbax.contracts.training",
      emitted_by=("TrainingSpec.loss", "provider_manifest.schemas"),
      consumed_by=("training loss lowering",),
      description="Legacy structured loss-term specification.",
  )
  ```
  This registers the **identity strings** `feedbax.spec.training.loss_term`
  (schema_id) / `feedbax.spec.training.loss_term.v1` (current_version) in the
  registry's bookkeeping, but:
  - No field on `LossTermSpec` itself actually carries either string at
    runtime — the family entry is documentation/governance metadata only,
    disconnected from the Pydantic model.
  - `_family`'s default `stance="reject"` applies (no `stance=` override was
    passed), and no `supported_old_versions` was supplied, so
    `rejected_old_versions` defaults via the `_old()` helper to
    `("feedbax.spec.training.loss_term.v1.v0",)` — i.e. the *only* version
    string this policy rejects is a `.v0`-suffixed variant of the *current*
    version string (a naming artifact of `_old()` appending `.v0` to whatever
    string it's given, here already-versioned `...loss_term.v1`, producing
    `...loss_term.v1.v0` rather than a sensible predecessor like
    `...loss_term.v0`). This looks like a pre-existing quirk in how `_family`
    is invoked for this entry (worth flagging verbatim, not fixing here).
  - Grepped `feedbax/contracts/migrations.py` for
    `register_migration("LossTermSpec", ...)` — **zero results**. No actual
    `SchemaMigration` (with a real `migrate` callable) has ever been
    registered for this family, unlike e.g. `GraphSpec` (two real migrations:
    `graph-spec-legacy-v1-to-v2`, `graph-spec-v2-to-v3-derived-dimensions`,
    `migrations.py:2125-2147`) or `StudioTaskBindingSpec`
    (`migrations.py:2148-2159`).
- On the **Studio/TypeScript side**, the only schema-version-bearing sibling
  is `StudioObjectiveSpec.schema_version: 'feedbax.studio.objective.v1' |
  string` (`web/src/types/workspace.ts:139`), which lives on the *modern*
  container, not on `LossTermSpec`. `StudioObjectiveSpec.legacy_loss_spec?:
  LossTermSpec | null` (`workspace.ts:142`) is an unversioned sibling field
  sitting alongside that versioned container — so a saved Studio scenario can
  carry a `LossTermSpec` payload with **no version tag of its own**, riding
  along inside a `StudioObjectiveSpec` envelope that is versioned but whose
  version string says nothing about the `legacy_loss_spec` payload's shape.

**Summary: No schema version field exists on `LossTermSpec` anywhere in the
in-memory model, its containers, or its serialized forms — only a
disconnected governance-registry `SpecSchemaFamily` declaration that names an
identity string but is never actually stamped onto emitted payloads, and has
no registered migration rule.**

**What a versioned migration rule would need to record**, based on §1–3
findings:

1. **A real schema-identity field must be added before any migration rule can
   fire.** Since `LossTermSpec` has zero version markers today, the migration
   cannot dispatch on a payload's own declared version — it must instead be
   keyed on *structural detection* (e.g. "payload matches `LossTermSpec`'s
   field set and lacks an `ObjectiveSpec`-style `schema_version`/`kind:
   "objective_spec"` marker") or on the container's `ObjectiveSlotSpec.kind ==
   "loss_term"` /  `StudioObjectiveSpec.legacy_loss_spec is not None` sentinel,
   consistent with how dispatch already works at runtime (§2).
2. **Trial-axis reduction must be pinned to `"mean"` unconditionally** for
   every migrated leaf term's `ReductionSpec.trial`, regardless of any other
   inferred settings — this is the one universal, unconditionally-true fact
   about every legacy term (§3), and getting it wrong silently changes the
   aggregate loss value for every migrated graph.
3. **Feature-axis reduction must be pinned to `"sum"`** (matching
   `_legacy_reduce_feature_metric`'s unconditional feature-sum) for
   non-matrix terms; for matrix-quadratic terms the feature axis is already
   collapsed by the einsum and `reduction.feature` is moot (§3).
4. **Time-axis mapping table**: `time_agg.mode` → `ReductionSpec.time` is
   direct for `"all"/"mean"→"mean"`, `"sum"→"sum"`, `"final"→"final"`, but
   `"range"` has no destination value and needs an explicit migration
   decision (fabricate a timeline epoch, or reject with a clear error) — and
   `"segment"`/`"custom"` should be **rejected outright**, matching their
   existing unimplemented status in the legacy executable path itself (no
   regression is introduced by refusing to migrate what could never execute
   anyway).
5. **Huber delta ambiguity must be flagged, not silently assumed.** Because
   the *modern* pipeline also currently hardcodes the Huber threshold to
   `1.0` (a latent bug, §3), a migration rule minted today should record
   `metric.huber_delta = 1.0` for migrated legacy huber terms to preserve
   current behavior — but this choice needs an explicit code comment/test
   tying it to the current hardcoded value, since fixing the modern-side bug
   later would otherwise silently break the migrated value's assumed
   semantics.
6. **`retention` field passthrough**: `LossTermSpec.retention:
   RetentionPolicySpec` has a same-named, same-typed field with no
   transformation needed — it is inert in both pipelines' lowering logic
   today (§1), so migrating it is a straight copy with no semantic risk, but
   should be preserved for forward-compatibility with whatever eventually
   reads it.
7. **Ambiguous-case flagging**: any legacy term with `matrix is not None` AND
   a non-default `norm` (legacy silently ignores `norm` once a matrix is
   present, §3, row "Matrix quadratic") should have the migration explicitly
   drop/ignore the stale `norm` value rather than mapping it into
   `MetricSpec.kind`, since `MatrixQuadraticLossSpec` requires `metric.kind ==
   "squared_l2"` (`spec.py:257-262`) and a straight copy of an inconsistent
   legacy `norm` would fail modern validation outright — this is a case where
   migration must actively normalize, not passthrough.

---

## 6. Proposed field mapping table

| `LossTermSpec` field | Proposed `ObjectiveSpec`/`ReductionSpec`/`SelectorAddressSpec` equivalent | Status | Note |
|---|---|---|---|
| `type` (leaf discriminator string) | `ObjectiveTermSpec`'s `type` discriminator (`"target_state"`/`"finite_difference"`/`"matrix_quadratic"`) | `equivalent-with-migration-rule` | Legacy accepts both PascalCase (`"TargetStateLoss"`, `"MatrixQuadraticLoss"`) and snake_case (`"target_state"`, `"matrix_quadratic"`) aliases (`service.py:585`); modern uses only the snake_case `Literal` tags (`spec.py:230,251`). Needs a small alias table, not a 1:1 copy. Any other `type` string is currently rejected by legacy lowering too (`service.py:588`), so no new rejection surface is introduced. |
| `type` (composite container, i.e. non-empty `children`) | No direct `ObjectiveTermSpec` equivalent — composites become a **flat `terms: list[ObjectiveTermSpec]`** with dotted/joined labels | `equivalent-with-migration-rule` | `ObjectiveSpec.terms` has no native nesting; the existing `objective.ts`-side conversion (`objectiveSpecFromLossSpec`, `web/src/stores/workspaceStore.ts:216-259`) already flattens by walking to leaves and joining path segments — the Python-side migration should mirror that flattening convention rather than invent a new one. |
| `label` | `ObjectiveTermBase.label` | `exact-equivalent` | Same string role, same uniqueness constraint (`spec.py:340-343` requires unique labels across the flat `terms` list — composite-tree labels were only unique *within a sibling group* in legacy, so flattening must generate collision-safe joined labels, e.g. the `dotted.path` scheme already used by `objectiveSpecFromLossSpec`). |
| `weight` | `ObjectiveTermBase.weight` | `equivalent-with-migration-rule` | Same field/semantics for leaves, but legacy's composite-node `weight` is not applied anywhere in `_lower_loss_term` (only child weights are used, §1) — migration must drop/ignore a composite node's own `weight` rather than trying to preserve it, matching current behavior. |
| `selector` | `SelectorAddressSpec.selector` (wrapped) | `equivalent-with-migration-rule` | Direct string copy into the `selector` sub-field, but `SelectorAddressSpec` additionally requires choosing a `kind`, and (for `matrix_quadratic`) a `feature_axis`, and (for `finite_difference`) a `temporal_axis` — these have no legacy source field and must be inferred/defaulted (e.g. `kind="state"` default, `temporal_axis="time"` default already matches `SelectorAddressSpec`'s own default, `spec.py:50`). `port:`/`probe:` selectors are **already broken in both pipelines** today (`service.py:722-727` raises unconditionally) — migrating them should preserve that failure, not paper over it. |
| `target_selector` | `TargetValueSpec(kind="selector", selector=SelectorAddressSpec(selector=...))` | `equivalent-with-migration-rule` | Needs wrapping into the tagged-union `TargetValueSpec`; mutually-exclusive-with-`target_value` constraint is already enforced identically on both sides (`service.py:579-583` legacy-side; `spec.py:59-62` `TargetValueSpec` model-side validator only allows one `kind`). |
| `target_value` | `TargetValueSpec(kind="constant", value=...)` | `exact-equivalent` | Direct value copy, just re-tagged into the union; for matrix-quadratic terms with `target_value=None`, legacy already defaults to `0.0` (`service.py:591-594`) — migration should materialize that default explicitly as `TargetValueSpec(kind="constant", value=0.0)` rather than leaving `target=None` (modern `MatrixQuadraticLossSpec.target` is `Optional`, `spec.py:250`, and `_objective_target_payload` already defaults `None`→`(None, 0.0)`, `service.py:682-685`, so leaving it `None` is also safe — either choice is behavior-preserving). |
| `retention` | No modern field — same-named `RetentionPolicySpec` type exists but is not a member of `ObjectiveTermBase`/`SelectorAddressSpec`/`ReductionSpec` | `no-equivalent-needs-design` | Inert in both pipelines' current lowering (§1, §5-point-6), but there is genuinely no slot to put it in on the modern spec today — needs either a new `metadata`-bag convention or a dedicated field addition to `ObjectiveTermBase` before this can round-trip losslessly. |
| `norm` | `MetricSpec.kind` | `equivalent-with-migration-rule` | Direct value copy for `squared_l2`/`l2`/`l1`/`huber` (all four legacy literals are valid modern `MetricKind` values, `spec.py:18`); default-when-unset differs in spelling only (`service.py:596`'s legacy default `"squared_l2"` matches `MetricSpec.kind`'s own Pydantic default, `spec.py:149`) so an absent `norm` maps to an absent/default `MetricSpec` with no rule needed. **Exception**: when `matrix is not None`, legacy silently ignores `norm` (§3) but `MatrixQuadraticLossSpec` requires `metric.kind == "squared_l2"` — migration must force `metric.kind = "squared_l2"` for any migrated matrix term regardless of the legacy `norm` value, per §5 point 7. |
| `matrix` | `MatrixPayloadSpec.value` (wrapped) | `exact-equivalent` | Direct array-payload copy into the wrapper object. |
| `matrix_kind` | `MatrixPayloadSpec.kind` | `exact-equivalent` | Direct copy; same two-value enum (`dense`/`diagonal`) on both sides, same default (`dense`) on both sides (`training.py` field has no Pydantic default but `service.py:600` defaults it to `"dense"` at lowering time; `MatrixPayloadSpec.kind` defaults to `"dense"` directly, `spec.py:163`). |
| `time_agg.mode = "all"`/`"mean"` | `ReductionSpec.time = "mean"` | `exact-equivalent` | Verified identical computation (§3). |
| `time_agg.mode = "sum"` | `ReductionSpec.time = "sum"` | `exact-equivalent` | Verified identical computation (§3). |
| `time_agg.mode = "final"` | `ReductionSpec.time = "final"` | `exact-equivalent` | Verified identical computation (§3, code paths are copy-identical). |
| `time_agg.mode = "range"` (+ `start`/`end`) | No `TimeReductionKind` value; would require an `EpochMaskSpec`/`TaskTimelineSpec` synthetic-epoch workaround | `no-equivalent-needs-design` | See §3/§5 point 4 — needs an explicit design decision (fabricate timeline epoch vs. reject), not a field-rename. |
| `time_agg.mode = "segment"` / `"custom"` (+ `segment_name`/`time_idxs`) | No equivalent; already non-executable in legacy itself | `no-equivalent-needs-design` (degenerate case: reject, since legacy cannot run these either) | See §3 — safe to reject during migration since these modes already error out of the current executable legacy path. |
| `time_agg.discount` / `discount_exp` | `ScheduleSpec` union (`PowerLawScheduleSpec` is the closest shape) | `no-equivalent-needs-design` | Legacy's `discount` is **never actually applied** by the executable lowering path today (§1.2) — it's dead data on any already-saved `LossTermSpec`. A migration could either drop it (data loss, but preserves current, already-inert, behavior) or attempt a best-effort `PowerLawScheduleSpec` reconstruction (new behavior, changes nothing observable today but changes what a *future* fix to the legacy path would have computed) — needs an explicit product decision, not a mechanical rule. |
| `children` (composite recursion) | Flattening into `ObjectiveSpec.terms` (see `type` composite row above) | `equivalent-with-migration-rule` | Structural transform, not a field copy — recursive walk producing joined labels, dropping the composite container's own `weight`/`type`/`selector`/etc. (which are unused by `_lower_loss_term` for composites anyway, §1). |
| *(no legacy field)* | `ReductionSpec.trial` | `equivalent-with-migration-rule` (forced constant) | Must be hardcoded to `"mean"` for every migrated term — see §5 point 2. Not a user-choice field during migration; a fixed constant is the only behavior-preserving value. |
| *(no legacy field)* | `ReductionSpec.feature` | `equivalent-with-migration-rule` (forced constant) | Must be hardcoded to `"sum"` for every migrated non-matrix term — see §5 point 3. |
| *(no legacy field)* | `ReductionSpec.tail_fraction`, `EpochMaskSpec`, `ScheduleSpec` (beyond discount, see above), `TargetValueSpec(kind="task_target")` | `no-equivalent-needs-design` | Purely additive modern capabilities with no legacy analog to migrate from — out of scope for a *migration* rule (nothing to map), but should be explicitly noted as "new capability, not available to migrated legacy specs" in the binding spec so nobody assumes migrated graphs silently gain mask/schedule/tail support. |

---

## Appendix: files read in full or in relevant part (for citation traceability)

- `feedbax/objectives/spec.py` (full, `d026038b`)
- `feedbax/objectives/service.py` (full, `d026038b`)
- `feedbax/contracts/training.py` (relevant sections: lines 1-250, 575-585, `d026038b`)
- `feedbax/contracts/graph.py` (lines 155-180, `RetentionPolicySpec`/`RetainedObservableTargetSpec`, `d026038b`)
- `feedbax/contracts/migrations.py` (lines 1090-1310, 2085-2170, `d026038b`)
- `feedbax/integrations/provider.py` (lines 890-930, 1240-1300, `d026038b`)
- `feedbax/runtime/retained_observables.py` (lines 1-40, `d026038b`; remaining `LossTermSpec` call sites located by grep, not fully re-read beyond confirming import/usage context)
- `feedbax/web/api/training.py` (lines 1-230, `d026038b`)
- `feedbax/objectives/loss.py` (class/def index only, dispatch-boundary scope per task instructions, `d026038b`)
- `tests/test_loss_service.py` (full, `d026038b`) — primary evidence source for §3
- `web/src/types/workspace.ts` (lines 120-150, `d026038b`)
- `web/src/types/training.ts` (lines 1-40, `d026038b`)
- `web/src/features/scenario/objectives.ts` (lines 1-60, 300-354, `d026038b`)
- `web/src/stores/workspaceStore.ts` (lines 200-260, `d026038b`)
- `scripts/generate_studio_contracts.py` (lines 100-190, `d026038b`)
- rlrmp repo: `src/`, `scripts/` full-tree grep for `LossTermSpec`/`loss_term`/`legacy_loss_spec` (current working-tree state, not branch-qualified per task instructions), plus targeted read of the four files the grep matched (`gru_pilot_figures.py`, `gru_checkpoint_selection.py`, `loss.py`, `cs_nominal_gru.py`) to confirm non-relevance.
