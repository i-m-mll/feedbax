# Execution provenance and prepared computation reuse

The kernel owns dependency comparison and receipt admission ([issue:8716737]). The scientific runtime owns preparation and executable specialization ([issue:f02fdbe], [issue:a1382d5]). The job service owns the admission transaction and the host owns job lifecycle ([issue:de9e2ac], [issue:f6f0bdd]). The recorder qualification is tracked in [issue:fca88cd]. This contract defines the shared boundary for candidate recorders, storage, and the job service.

## Dependency record

A versioned immutable record names the dependencies of one observed computation and the provenance records of prepared computations it consumed. Records can be shared by digest. The record schema and comparison-rule version determine how to interpret an observation.

| Observation | Comparison |
|---|---|
| First-party scientific source unit | Stable logical identity and normalized content. Normalization retains executable statements. If computation consumes normally omitted metadata such as a docstring, its value is a separate observation. |
| Binding or receiver | Logical lookup location and target actually used, including receiver class and contributing class definitions where dispatch depends on them. Definition-time callable, default, and closure values carry their own dependency observations. |
| Registry selection | Registry identity, key, and selected definition. An unrelated key is independent. Initialization effects that contribute to this selection are dependencies in their own right. |
| File or other scientific input | Exact resolved content digest of what the computation consumed, with a logical role or path for diagnostics. The kernel keeps the existing attempt binding as the authority for declared inputs. |
| External dependency | Relevant distribution/source/native identity and the producing environment's lock projection. Scientific precision and determinism settings retain their existing request identity semantics. |
| Coverage reason | A typed location and cause where dependency equivalence cannot be established, such as an unsupported dynamic capture, an untracked file read, or missing executable provenance. |

Scope begins before preparation that creates scientific runtime state or code and extends through production of scientific outputs. The current Studio worker constructs its numerical update before entering the native training loop. Pass 5 changed only the objective implementation on that path and detected the change only when preparation was included ([artifact-version:a875338c290e]). Log formatting, event transport, and paths for scratch outputs are attempt details. A job may import or execute code for both purposes. The observation follows its contribution to the result.

The recorder verifies that its observations describe the code and values actually used. Reading a mutable namespace or hashing an opened path after execution is insufficient. Source observations identify a verified source generation. File observations bind the consumed bytes. An unsupported or mixed observation records a coverage reason. Project-local source and binding metadata can be indexed and reused while a verified generation is current. Whole-worker imports, cache entries, and trace sets are context for maintaining the worker and cannot become the result's dependency record simply by being present.

## Prepared computation and cache

A prepared entry contains the callable or executable, its specialization, and the immutable dependency record for its preparation and the prepared computations it consumed. The specialization identifies the scientific program and the structural, numerical, backend, and compiler settings that select this executable. The dependency record identifies what produced it. Neither identifier becomes an authored project field.

One comparison operation compares a record against a verified current context and returns verified differences and coverage reasons. Both cache acquire and receipt admission use that comparison. Cache acquire also checks executable specialization and representation compatibility. Receipt admission additionally checks the request, its current inputs, and product integrity.

| Cache acquire condition | Outcome |
|---|---|
| Specialization compatible, dependency record present and supported, all dependencies match | Return the executable and its record together. |
| Entry absent, incompatible, or a verified dependency differs | Prepare under observation. Publish the executable and record together if reusable. |
| Entry provenance missing or unverifiable | Prepare under observation. If preparation remains unverifiable, the job may produce a historical result with a typed reason. |

A result includes records only for prepared work it used. When code executes using an existing compiled object, the record travels with the object even if Python performs no lowering during that call. Pass 5's A→B→A replay exercised this rule in one process. Persistent-cache restoration must either carry verified provenance for the restored preparation or take the ordinary observed preparation path. A JAX backend binary cache hit and reuse of an in-memory prepared callable are different cache operations. Qualify the path that actually occurs.

An explicit acceptance of a receipt's named code change applies to that receipt. Preparing a fresh result still requires the current scientific computation. A receipt acceptance never turns an old executable into a compatible cache entry.

## Receipt and host boundary

The worker resolves the actual attempt binding before scientific execution, commits it in the project store, and reports it through the existing binding event. A plan or an admission comparison may check whether a reference still resolves to the old input. A job that starts later resolves its own exact binding, including any progress product or selection rule, as defined in [issue:8716737]. Resume inherits the original binding.

Publish result objects and their provenance closure before publishing the immutable receipt. The receipt names the request, actual binding, scientific products, and dependency record. A product's digest excludes the attempt identity and publication timestamp. Prepared executable caches are disposable. A receipt's dependency records remain reachable from its closure after cache eviction, worker exit, and store transfer ([issue:92925eb]). Progress products that need reconstruction retain their source/model closure without waiting for a terminal receipt.

The signed M1 local integration at `39e6d096` establishes an existing host boundary. `feedbax/host/worker.py` invokes a preloaded entry using the job input, event, stop, and scratch paths. `feedbax/host/store.py` ingests publication keys and marks dependent edges ready. `feedbax/jobs.py` owns the current training receipt check. The recorded M1 walkthrough measured 0.844 s to first loss with its persistent compilation cache warm ([artifact-version:604e59870d36]). The provenance work uses these boundaries. The M2 service uses the kernel's comparison in place of the legacy training receipt check. The host can persist and route worker receipt and publication references without interpreting provenance, performing code comparisons, or gaining a second admission policy.

## Admission and latency

Reuse as current requires a successful receipt, intact referenced objects, matching request and current inputs, a supported comparison version, and a verified match of all dependencies after applying any finite accepted differences. A verified difference carries the changed dependency. Unverifiable coverage carries its reason and any verified differences. Historic products remain readable, and explicit digest-bound use is a choice of input. An unverified reason alone cannot schedule an automatic retry loop. The status projection consumes the kernel result through the existing job-service contract.

The admission decision names the verified store ref, source/binding index, and environment generation used in its comparison. Revalidate if that view changes before committing admission, following the service's existing stale-plan and stale-job checks. A later change is visible in the next plan. Project views can share computed comparisons for equal record digests and the same verified context. Index observations by logical dependency so a source edit affects only the records that could depend on it. Avoid re-reading receipts or rediscovering normalized bindings per run in a plan. A missing or not-yet-verified observation has a typed result and does not grant a cache hit.

The pre-consumer gate checks unchanged ordinary native training and downstream recipes, a changed preparation-time objective, cached A→B→A, unrelated edits, inherited receivers, definition-time captures, runtime rebinding, exact file consumption, and relevant environment comparison. It includes candidate CPU recording under 1%, warm full map construction under 50 ms, and measured admission costs without hiding work in background indexing. The plan and light-work budgets remain in [issue:4007bf2]. Persistent compiled-object restoration and a producer-compatible GPU environment require their own evidence before those paths authorize reuse.
