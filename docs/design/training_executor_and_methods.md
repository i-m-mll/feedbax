# Training executor and methods

This is the executor and method design for [issue:a1382d5], under [issue:30fae90] and [issue:4007bf2]. The [shared core and executor contract](shared_core_executor_contract.md) owns chained Train requests, exact checkpoint binding, restoration, live validation and the evaluation boundary. The source inventory below refers to Feedbax `6e3491e2` and rlrmp2 `50993e3`. Evidence and gates are stated at their exercised scope.

## Outcome

A method author supplies scientific settings, numerical transitions, and their state. Feedbax supplies launch, progress, stop, checkpoint, resume, diagnostics, and Studio controls. Gradient training, adversarial training, local learning rules, and evolutionary methods use this same lifecycle as they qualify. A notebook calls the same numerical executor, and a train request publishes a model that the built-in evaluator can run without invoking its training method.

The host owns process lifecycle and request ordering. The training executor owns one admitted training computation. The core runtime owns model execution, task materialization, and observable production. The objective owns reductions over those observables. The kernel owns input binding, content identities, products, receipts, and admission. These boundaries apply to both built-in and registered methods.

## Evidence from the current code

| Surface | Observed behavior | Design consequence |
|---|---|---|
| [`PhaseProgramExecutor`](../../feedbax/training/phase_executor.py), `run`, `preflight`, `resume_position` | Executes declared update kernels, phase transitions, checkpoint barriers, mapped state, cadence, update budgets, and resume coordinates. Numerical kernels have an ahead-of-time path. The Python loop still copies progress models and reads health and batch authority at update boundaries. | Retain and simplify this executor as the common driver. Preserve its externally exercised state and resume semantics while replacing duplicated declarations and storage. |
| [`execute_training_run_spec`](../../feedbax/training/executor.py) | Resolves methods and preparation, restores checkpoints, runs the phase executor, collects method observations, and emits diagnostics and a manifest. It also owns large amounts of old custody and manifest-binding machinery. | Split the single-request worker integration from the numerical driver. Kernel products and receipts replace the old custody projections at M2. |
| [`studio_graph_training_program`](../../feedbax/training/studio_executor.py) | Supplies a declared method, builds one compiled gradient update, restores model/optimizer/key/counter slots, and uses `training.retained_graph` for retained preparation. `reported_update` synchronizes and materializes scalar metrics on every call. | Move reusable preparation and rollout into core and expose the standard gradient method through the common registration. |
| [`compile_training_run`](../../feedbax/web/worker/execution.py) and `rollout_graph` | The current graph preparation explicitly rejects `batch_size != 1`. A rollout samples task data, initializes runtime state, and calls the executable graph. | Batched trials belong to the shared runtime contract. Batch support requires a behavioral proof before the M2 evaluator relies on it. |
| [`SimpleTrainer`](../../feedbax/training/trainer.py) | Has a separate whole-dataset Python gradient loop and returns a model. It partitions inexact array leaves and initializes Optax state internally. | Remove this convenience entry point and its export when the executor is replaced. Preserve supported in-process training through the common driver. |
| [`train_ppo_batched`](../../feedbax/training/rl/ppo.py) and `_run_ppo_updates` | PPO carries policy, optimizer, environment, episode totals, observation normalization, exploration noise, curriculum, and random state. The outer training horizon is one `lax.scan`, and final metrics are formatted after it returns. | Retire the legacy PPO implementation at executor replacement after resolving the bbooddyy callers and shared task imports. Record the last compatible revision. [issue:2dfe7f8] |
| [`DeclaredTrainingProgram`](../../feedbax/contracts/training.py) and `declare_training_program` | A declaration assembles runtime, preparation, authoring, row-lowering, and metadata/schedule/optimizer projection facets. Authored method payloads coexist with derived worker declarations. | Use one typed registration. Project lowering and schemas derive the surface data that these facets currently repeat. |
| rlrmp2 `src/rlrmp2/training/{programs,method_contracts,preparation,kernels}.py` | Adaptive-lambda and SISU methods use Feedbax preparation, slots, phase execution, and diagnostics. A mapped update combines sampled evidence, adversary optimization, controller optimization, clamp state, schedule observations, and health validation. | Preserve this scientific update as a cohesive method. Move generic lowering, clocks, state projection, and result publication into Feedbax. |
| [`compiled_trial_rollout`](../../feedbax/analysis/rollout.py) | Already runs trial batches through a stable compiled function with array operands. Its contract records a real retracing failure caused by fresh closures and captured arrays. | Reuse this numerical boundary in core. A separate evaluator-specific rollout implementation is unnecessary. |

The downstream source review establishes the method requirements. It does not establish execution parity for the proposed design. The SISU floor measurement in [issue:eb161f2] requires a concretely bound fixture before a slimming verdict can be made. Its 553-line static lower bound is neither a total method cost nor a before/after result.

## The train worker

The worker executes one admitted `train` request. Its spec contains effective model, task, objective, and method definitions, numerical policy, seed, work target, and any measurement policy that affects training. Each resolved row of an ordinary Train step supplies one request. Plan resolves the step's inherited definition and checkpoint input under the shared [chained Train contract](shared_core_executor_contract.md#chained-train-steps) before admission. Request identity follows [issue:d601693]. Job ID, output location, placement, issue annotation, event delivery and machine budget belong to the attempt.

An execution starts in this order.

1. The worker attests its loaded first-party packages and locked dependencies. It resolves the input references once, persists the binding under the job, and emits the binding event before numerical work. A resumed attempt inherits the original binding.
2. The method registration validates its typed settings and the compiled runtime's required capabilities. Preparation builds the model/task/objective runtime, materializes data, creates the state template, and initializes a fresh state or fills that template from the bound checkpoint.
3. The driver compiles the numerical transition for this request, reports its preparation, and advances at update boundaries.
4. It publishes checkpoint and diagnostic products as they become durable. A checkpoint publication can release an evaluation job through the host's publication edge.
5. It publishes one immutable outcome receipt and then one worker `result` event. The launcher supplies the sole host terminal event after the entry returns or dies. The receipt names this job, the request, exact bindings, observed execution trace, products, final cursor, and any resumed predecessor.

The kind-table entry for `train` binds this worker function. Studio and the CLI use `plan` and `admit`. The worker does not walk a workflow or launch another job. A supported direct Python call executes the same numerical driver in the caller's process, with an explicit in-memory or project-store execution context.

A rehearsal supplies an execution limit in a scratch context and uses the same preparation and update function. Exhausting a work budget before the request target yields a stopped outcome with a budget reason. Keep historical `update_budget` witnesses in scratch custody.

## Method registration and numerical contract

Generalize the current `DeclaredTrainingProgram` into one registration containing a stable method key, one settings model, a pure `compile` function, and a `prepare` function. `compile` validates authored settings and emits the effective training program and its requirements using schema and model metadata. It performs no model initialization, JAX compilation, or store writes during `plan`. `prepare` consumes that program and the bound runtime, and returns initial state, a restoration template, and its numerical transitions. The settings model supplies validation, defaults, migrations, and Studio forms. Generic project lowering resolves the same declaration.

The registration has no method-specific lifecycle callbacks. Progress, stopping, publication, receipt construction, and recovery remain driver behavior. The type and registration machinery replaces the current separate authoring, row-lowering, schedule, optimizer, and metadata projectors as their consumers move to the project and kernel model. A method's scientific parameters occur once in its settings. Runtime products carry the resolved values actually used.

The common representation retains the useful parts of `MethodContractSpec`, `PhaseProgramSpec`, and `NumericalUpdateKernel`.

| Value | Contents and authority |
|---|---|
| `TrainingProgram` | Validated transition and named clocks derived from this request and registration. |
| `TrainingState` | Model values and persistent model buffers, method state, optimizer states when used, sampler state, random state, and the program cursor. Every value that can affect a later update belongs here or in an immutable bound input. |
| `RuntimeOperands` | Immutable task data, frozen model arrays, effective objective and other numerical inputs. Large arrays are passed as dynamic operands. Callable structure is held by the prepared runtime. |
| `UpdateResult` | Candidate state, named clock advances, typed health outcome, scalar or labelled metric values, and the declared scientific diagnostic payload. |
| `Cursor` | Next transition, request-local completed work, resolved schedule origins, and named method clocks such as environment steps or optimizer updates. It specifies the next computation after restoration. |

The numerical transition is a pure function `update(state, operands) -> UpdateResult`. Its output structure and array shapes/dtypes remain fixed in the prepared transition. The driver atomically adopts a healthy candidate state and its cursor. An invalid candidate cannot advance published progress or enter a resumable checkpoint. Buffer donation is safe only when the compiled transition preserves the previous healthy state on its invalid branch. JAX exceptions are converted once at the execution boundary into typed failures.

One driver update is the smallest declared transaction whose state can be restored coherently. Standard gradient training uses one optimizer update. PPO uses one rollout-and-optimization update, including all epochs belonging to that rollout. Adversarial training uses one complete inner-adversary/outer-controller transaction. Local learning can use one trial or a resumable block of time steps. A method with a costly update may expose finer transitions only when it includes the intermediate state and cursor needed for exact restoration.

Numerical `scan` and `vmap` remain appropriate inside an update. The driver dispatches at most one update beyond its last checked boundary. This gives checkpoint stop the specified next-update behavior. Coalescing several driver updates into one uncancellable compiled block is an explicit future contract change, and requires both stop-latency and numerical-parity evidence.

State slots retain names for inspection and selective loading. Their persistence classification is declared once through the state schema. Required state is persisted by default. Disposable scratch values must be reconstructible from persisted state and bound inputs without consuming a fresh random sample. Metrics and diagnostic observations are outputs, unless the algorithm uses them in subsequent decisions, in which case their underlying statistics are state.

## Methods and composition

Methods compose through the data and state they actually share. Objective reductions, task sampling, normalizers, optimizer transformations, and schedules have independent typed definitions. The registration declares capabilities required by its compiled program. Planning compares these requirements with the runtime, such as differentiable rollout, state observability, supported precision, or forward-only execution. Unsupported combinations produce a path-bearing diagnostic before launch.

| Method | One update and persistent state | Shared facilities |
|---|---|---|
| Standard gradient | Sample a trial batch, differentiate the objective, update selected parameters, apply parameter constraints. Persist parameters, model buffers, Optax state, sampler and RNG state. | Shared rollout, objective, trainable selector and Optax transformations. |
| CVaR or another risk objective | Apply the authored risk reduction over labelled trial or realization losses within the chosen method. Persist any estimator state used by later updates. | Objective reduction and axis metadata. CVaR does not need a separate host or training loop. |
| Adversarial or minimax | Produce bounded adversary evidence, perform its declared optimizer work, then commit the controller and adaptive-state update together. | Multiple named optimizer states, objective terms, clocks and method diagnostics. Existing downstream kernels provide the concrete witness. |
| Future PPO registration | Collect bounded rollouts, calculate advantages, perform declared minibatch updates, then commit policy, optimizer, environment and enhancement state. | A future consumer would use the shared graph runtime and qualify its method registration. [issue:2dfe7f8] |
| Local update rules | A pure rollout transition returns updated parameters and local learning state as well as dynamical state. Commit at the method's declared boundary. | Runtime transition and explicit parameter ownership. Disjoint parameter owners compose directly. Overlapping writers require an authored order or combination rule. |
| Evolution strategies | Ask for a population from strategy state and a key, evaluate it, then tell the strategy the scores and commit its next distribution and selected model. | Shared batched rollout/objective, labelled population state and normal checkpointing. An installed strategy library can supply the numerical rule. |

Training configuration chooses the method and its parameters. The model graph describes the system that runs. An observation transformation needed during inference is part of the exported model and carries its learned statistics with it. Its training rule specifies when statistics update. Evaluation freezes those statistics by default. A deliberate evaluation adaptation policy is an authored computation with its own state and identity. A curriculum belongs to the task sampler and method state. Rewards and costs use the objective's declared observables and reductions. These choices settle [issue:32d0935] and the remaining placement question in [issue:c9ae44f] without a second normalization definition.

Trial-grouped and risk reductions consume labels and masks from the task's trial data. Their spec states the aggregation axis, group weights and empty-group outcome. This is the objective surface of [issue:63230c3], whose implementation remains deferred by the user's decision. The standard equal-per-trial objective remains the default for current consumers.

Independent project sweep rows remain separate requests and jobs. A population, coupled ensemble, environment bank, or realization axis used inside one method is an axis of that request's state. Mapping uses declared array-axis bindings. The existing one-level mapped execution remains the supported starting point. Multi-level mapping in [issue:4c59e7f] needs its concrete second-consumer proof, including order, RNG, reductions and memory growth. No per-method axis enum is needed in the driver.

## Update coordinates

The transition reads the scheduled value at its declared pre-update coordinate. Methods can supply clocks such as environment steps or optimizer updates. The shared [chained Train state carry contract](shared_core_executor_contract.md#chained-train-steps) owns origin, optimizer compatibility and successor initialization. A program receives the resolved schedule coordinates through its numerical operands.

## Stop, exact resume, and initialization

`checkpoint` stop is a durable intent of the host. Before dispatching an update, and after the outstanding update completes, the driver reads the intent. It finishes that update, validates the candidate, flushes required observations, writes a checkpoint at the resulting cursor, and writes a stopped receipt. The M2 checkpoint format supports the initialized state at update zero, so a stop received during preparation can publish that state once preparation completes. The M1 contract permits the first update to establish a checkpoint when the current format requires it. A preparation failure yields its typed failure. The row remains stopping while this work completes, with force stop available.

Force stop belongs to the host and is bounded by its five-second contract. It kills the launch process group and records interruption if the worker cannot publish an outcome. The last durable checkpoint remains available. The executor cannot promise a new checkpoint after an external kill.

An exact resume uses the original target and request-local cursor. It restores the full future-influencing state described in the [shared contract](shared_core_executor_contract.md#3-model-value-authority-and-reconstruction), including data-source position or sampled pool identity and curriculum state. A finished request has no remaining work to resume.

Exactness means the same next update inputs and decisions as uninterrupted execution. On a qualified deterministic backend with identical numerical settings and code, the acceptance comparison is bitwise equality of final scientific state and diagnostics. Cross-backend and nondeterministic algorithms must state a scientifically justified comparison separately. Merely loading arrays does not prove exact continuation.

The worker verifies the checkpoint's code/environment compatibility and state descriptor before advancing. A changed implementation that cannot supply the recorded transition semantics produces `incompatible_checkpoint` with the changed dependency or state path. The user can author a new request with an explicit initialization transform and new code. Request identities remain independent of code identity, as [issue:4007bf2] requires.

Continuation, warm start, and fork use the shared [checkpoint input and initialization contract](shared_core_executor_contract.md#chained-train-steps). A new Train step can inherit a definition with `base: {step: ...}` and initialize from the same step's `last` checkpoint input. The new request specifies retained, reset, transformed and discarded state and its schedule origin. A warm start selects model and inference values and initializes method state. A registered fork transform records its observed implementation and validated descriptor.

If an update is invalid, the driver retains the previous healthy state, publishes its typed failure and diagnostic attribution, and keeps the last durable checkpoint. Existing explicit restore-to-checkpoint policies are represented in the method settings and receipt. A policy may not silently skip a failed update or claim that an earlier checkpoint contains later progress.

## Checkpoint and diagnostic products

The checkpoint uses [issue:129ede5]'s safetensors objects and canonical descriptor. Its tree names model, method, optimizer, sampler and RNG slots by digest, plus the cursor. Model state includes all trained inference buffers. Evaluation resolves the model's closure without loading optimizer or training-only state.

Static callables and runtime structure are reconstructed from the recorded request and current compatible registration. The descriptor records their declared identity. Mutable scalar leaves are persisted according to their declared state schema. Compiled transitions preserve leaf structure and dtype. The registered Studio MLP has authenticated M1 custody evidence [issue:ff4f6cc]. A bare Equinox serialization probe is separate evidence. The new container still needs its own applicable qualification.

Publication writes immutable slot objects and the checkpoint tree before appending its product event. A crash between those acts leaves either an unpublished object or a replayable publication, never a reference to missing bytes. Publication keys and observation coordinates are stable within an attempt, so replay after host recovery is idempotent. The binding and published products remain retention roots through a stopped or interrupted attempt until success or explicit deletion, as [issue:92925eb] requires.

Publications distinguish exact request boundaries even when two requests share a displayed cumulative update. A successor initialization is a boundary of its own request. Host matching remains an opaque publication-key operation. M1 keeps its qualified key and schema; production M2 spellings are gated by the [shared contract](shared_core_executor_contract.md#production-gates).

Scientific diagnostics use the dataset contract in [issue:d0019f7]. A method declares its output variables, dimensions, units and measurement basis. Coordinates include request update and internal axes such as replica, with before-update measurements and after-update state distinguished by schema. The worker validates each batch of observations, stores immutable chunks, and links a complete index from the receipt. Exact resume joins the bound checkpoint's observation prefix with new points under the shared observation contract. The UI's downsampled metric stream never becomes the source of the scientific dataset.

This carries the actual requirements of [issue:0dff97e] and [issue:0da8b51] onto the common store and public dataset loader. Missing or truncated products, duplicate coordinates, schema disagreements and gaps in a promised dense trace are typed loading failures. Sparse observations carry their authored cadence. Consumers can analyze a stopped run's published prefix without rerunning training.

## Events and errors

The executor publishes the job event protocol agreed in [issue:de9e2ac] and the M1 contract on [issue:30fae90]. The host invokes the pre-imported entry as `function(argv[1:]) -> int`, with immutable input at `FEEDBAX_HOST_INPUT` and launcher-provided `FEEDBAX_HOST_JOB_ID`, `FEEDBAX_HOST_EVENT_LOG`, `FEEDBAX_HOST_STOP_FILE` and `FEEDBAX_HOST_SCRATCH`. The worker appends `{kind, data}` JSON records. The host adds the job and stream cursor envelope. Kinds are `binding`, `phase`, `progress`, `log`, `frame`, `publication` and `result`.

Phase values are importing, building, compiling, restoring, running and finalizing. Checkpoint work is publication detail. The sink exists before preparation or heavy imports, so failures there are observable. Worker bootstrap reports imports performed before assignment. A warm job reports the phases it actually performs and their elapsed timings. The host supplies waiting-for-slot and process lifecycle events.

A progress event includes the authoritative cursor, completed/target work with units, the current phase, timestamped named metric observations, and a checkpoint reference when available. Metrics include their dimensions and explicit validity. A mapped metric stays mapped unless its schema declares a display reduction. Missing values remain missing, which resolves the misleading `loss=nan` behavior in [issue:d8cc1a4]. Genuine nonfinite values produce the method's health outcome.

M1 progress uses cumulative `completed_updates` and the original `total_updates`. Checkpoint publications use `data.key = "checkpoint:<completed_updates>"` and exact transaction id, manifest/content digest and location. M2 carries the corresponding content-addressed checkpoint reference. Mandatory binding, publication and result writes fail loudly, and durable publication/result records are synced before dependents or resume can rely on them. Metrics may be coalesced before append. Once appended, they participate in cursor replay. A `result` states succeeded, stopped or failed, names the persisted receipt when available, and carries the final completed work and checkpoint. Success requires the original target and an admitted receipt.

Publish new observations at least every 500 ms while updates complete, and flush a terminal remainder. If an indivisible update exceeds 500 ms, publish liveness and the age of the last observation while that update is outstanding. A heartbeat cannot manufacture a new loss. Methods whose update latency prevents the product's 2 Hz metric target require a measured phase-level observation point or a corrected budget, recorded on [issue:4007bf2]. Method callbacks do not write browser events directly.

Frames are bounded preview data. The M1 worker performs a diagnostic rollout with a separate observation key at snapshot boundaries. Frames share the job stream and its cursor. Logs are structured records with severity and source. The inspector restores metrics, frames and logs from the durable stream after reload. Scientific live observation follows the [shared contract](shared_core_executor_contract.md#scientific-observations-and-live-validation).

Recoverable failure is a typed outcome with `code`, execution phase, authored or state path where relevant, message, and structured details. The common executor codes are `invalid_request`, `preparation_failed`, `compilation_failed`, `incompatible_checkpoint`, `numerical_failure`, `publication_failed` and `executor_failed`. Typed detail reasons distinguish invalid settings, unbound inputs, environment mismatch, unsupported runtime capability and resource exhaustion at the appropriate boundary. Library exceptions are translated at their boundary. Unexpected defects retain their traceback in diagnostic output and terminate the attempt. If storage failure prevents a receipt, the worker reports the failure through its independent host event log and the job store records it.

## Shared runtime and observations

The [shared core and executor contract](shared_core_executor_contract.md) owns model reconstruction, one prepared rollout, paired trials, one objective, physical time, and checkpoint-set readiness. The driver consumes that runtime with explicit dynamic operands. Rematerialization controls activation memory independently of durable checkpoints. The built-in evaluator uses bounded trial chunks when its production contract qualifies. Chunking and durable evaluation resume have separate evidence requirements.

Method diagnostics remain part of the update result. The Train driver runs approved bounded read-only validation at the post-commit boundary under the [shared observation contract](shared_core_executor_contract.md#scientific-observations-and-live-validation). Production wiring requires the behavioral, custody and workload evidence on [issue:4af71ac]. Jobs schedules ordinary dataset, analysis, or over-budget work. The driver runs a single request.

## Notebook entry

At executor replacement, remove `SimpleTrainer` and its export. Notebook users can run the prepared standard gradient method with eager model and task objects through the common in-memory driver, receive model, state, history and typed outcome, and use callbacks and the ordinary stop boundary.

Eager objects are accepted directly for in-process numerical work. Persisted project execution requires reconstructible authored definitions or registered factories and content-pinned data inputs. A notebook opting into project results uses the same plan/admit service. There is no implicit pickle of an arbitrary closure and no separate notebook trainer loop. Pure update and rollout functions remain callable for differentiation and scientific experiments.

## Libraries and execution cost

Keep Equinox for filtered pytrees and transforms, Optax for optimizers and their state, and JAX for compilation and vectorized numerical work. These are existing dependencies. Equinox's documented distinction between array leaves and static structure matches the proposed preparation boundary. Its template-based deserialization supports the reconstruction principle, while the kernel's indexed container supplies durable named access. [Equinox transformations](https://docs.kidger.site/equinox/api/transformations/), [Equinox serialization](https://docs.kidger.site/equinox/api/serialisation/).

Optax already supplies chained and parameter-partitioned transformations and stateful schedule injection. Use those numerical components behind explicit optimizer bindings and clocks. A full trainer framework would add another owner for orchestration that the host and executor already provide. [Optax optimizer composition](https://optax.readthedocs.io/en/latest/api/combining_optimizers.html), [Optax schedule state](https://optax.readthedocs.io/en/latest/api/generated/optax.schedules.inject_hyperparams.html).

Evosax provides an existing ask/evaluate/tell implementation for evolutionary strategies. A future concrete evolutionary method should first bind that API through the common method contract, then qualify the selected algorithm and version. This design adds no dependency or evolutionary algorithm implementation. [Evosax](https://github.com/RobertTLange/evosax).

JAX dispatch is asynchronous. Timing numerical work requires waiting for its result. `scan` requires a fixed carry structure, shape and dtype. The driver uses compiled updates with explicit array operands, caches compilation through JAX, and transfers one compact boundary observation containing health and authoritative work counters. It retains the exact per-update numerical-failure decision. Large model copies, Python leaf scans and diagnostic formatting stay off the update path. [JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html), [JAX scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html).

The local host owns worker preparation and the private persistent compilation cache, including the zero write threshold and size bound in [issue:a2c76b3]. The cache key belongs to JAX and includes compiled computation, backend and compilation settings. An edited numerical parameter is a dynamic operand whenever its shape and meaning permit this. A changed structure or dtype can require compilation. [JAX persistent compilation cache](https://docs.jax.dev/en/latest/501/compilation-cache.html).

The execution trace covers preparation and numerical tracing inside the executor's call, including method and runtime registry resolution. A compiled executable reused without Python retracing must carry its recorded first-party code dependencies into the current job's trace. Reusing an executable cannot make the component or objective disappear from admission's comparison. The trace proof in [issue:fca88cd] must cover a cold compile, a persistent-cache load and a warm executable hit, together with a change to a function used by the compiled computation. Warm workers follow the existing imported-code retirement rule before assignment.

Existing measurements on one Apple M4 Pro are the baseline, all from [issue:a2c76b3]. A small Studio model reached its first step in 2.2 s in a fresh process, 1.6 s with a working persistent cache, and 0.2–0.3 s on a second same-process run. Imports cost about 0.8 s. The old engine added about 0.2 s. A separate host skeleton reached its first event at p95 0.65 ms with an already imported worker. These observations used distinct paths and do not establish end-to-end M1 or M2 latency.

The M1 jobs bundle measures the actual shared walkthrough through admission, binding, preparation, first update, checkpoint publication and receipt completion. Report cold and warm cache cases, per-phase wall time with numerical synchronization, checkpoint bytes and write latency, peak memory, stop latency, event latency, and overhead per update. The targets remain [issue:4007bf2]'s 200 ms admission, 100 ms first event, 1 s cached first update, 2 Hz observations and 5 s force stop. Larger biomechanics models require their own evidence. No throughput gain is claimed from a rearrangement of host reads until that path is measured.

## Implementation sequence and acceptance

M1 establishes the ordinary Studio model's real stop/checkpoint/resume proof, typed progress and errors, and local host integration on today's request types. The M1 contract and executable fixture are owned by [issue:30fae90] and [issue:2d24abe]. Model initialization, training and observation keys derive from disjoint named branches of the authored seed. Resume restores the training key and observation never advances it. The registered MLP proof is recorded on [issue:ff4f6cc]. The qualified stream and frozen-value correction is [issue:2d84404]. The full design here supplies the consumer boundary for M2.

M2 first establishes the shared runtime result and checkpoint round trip. Then the standard method and train-request worker use the new kernel binding and receipt model. The built-in evaluator consumes the resulting model product and shared rollout. Method registration and in-process training consolidate as their callers move. Downstream method ports require their own scientific parity cases. Retire legacy PPO at executor replacement after the bbooddyy caller prerequisite in [issue:2dfe7f8]. The old envelope/engine path continues for its existing consumers through M5, as [issue:4007bf2] requires. Its removal is coordinated with downstream conversion and store migration.

The executable checks are behavior-based and run through supported entry points.

| Contract | Required evidence |
|---|---|
| Shared local walkthrough | Author the agreed small reaching model, run, observe loss, stop, resume, restart backend and host, and finish once. Reload reconstructs chart and log. Same-request duplicate admission never advances a second attempt concurrently. |
| Exact resume | Compare uninterrupted execution with stop/restart/resume at a committed update and around a chained request seam. Compare model, optimizer, RNG, sampler, method state, cursor and scientific diagnostics. Include the real walkthrough model and one bound adaptive downstream method. |
| Checkpoint structure | Round trip the callable-bearing model, typed keys and declared method state. Read the model without optimizer loading. Incompatible structure or code reports its concrete path. |
| Chained Train | Admit ordinary downstream steps through the checkpoint input. Verify freeze/unfreeze against an equivalent manually initialized method, compatible momentum, explicit optimizer reset, added state initialization, and exact successor restart. |
| Evaluation | Evaluate the trained model under two conditions using real task inputs, retain and read a chosen state variable, replay it, and demonstrate bounded execution and the qualified admission path. Compare the rollout with the training runtime for identical inputs. |
| Methods | Qualify a local-rule witness and a concrete evolutionary witness when scheduled. Verify their common lifecycle. Legacy PPO retirement follows the recorded caller prerequisite. |
| Diagnostics | Load exact scientific coordinates from the result through the public dataset entry. Truncation and coordinate/schema mismatch fail. Reloaded live charts and mapped metric display retain valid values. |
| Scheduled observations | Use the shared contract's first observer seam, and qualify independent cadences and ordinary checkpoint-consuming jobs with Studio disconnected. |
| Compilation and cost | Changing a dynamic value reuses its compiled structure. Changing a downstream Train definition preserves the unchanged upstream request. Measure compilation for the downstream executable. Measure real end-to-end timings, using the capacity manager for substantial workloads. |

Keep Python tests for supported behaviors that can regress. Rewrite coverage when a surviving behavior moves to the shared runtime or kernel. Retire tests that exclusively exercise a removed manifest format, duplicate loop, or old dispatch interface, recording their disposition in the replacing issue. Assertions that repeat production projections or UI copy do not establish coverage. Select affected Python tests through Pants and broaden only at the integration gate or for a new unresolved failure. Count debugging attempts from the first attempt. After a second failed iteration, audit adjacent failure paths before another run.

## Issue dispositions and consumer inputs

| Issue | Decision or remaining implementation |
|---|---|
| [issue:b77cbf4], [issue:fb58eff] | Bind one train request to the common driver with input attestation, products and an immutable receipt. The method cannot schedule requests. |
| [issue:dbbfb13], [issue:3f7916f] | Load the standard model product, use the shared runtime and objective, and publish a labelled dataset under the qualified product contract. |
| [issue:129ede5] | Checkpoint state schema and cursor here are the training consumer contract. Preserve mutable scalars and template-supplied callable identity. |
| [issue:0491f60], [issue:bfc9690] | One settings model and registration supplies Studio authoring and execution. Capability requirements derive from the compiled program and runtime. |
| [issue:5b81b7b] | Local learning is a first-class method using pure parameter/state transitions. Its implementation requires the stated ownership and resume proof. |
| [issue:ba00cef], [issue:f90e6e2] | Alternative algorithms use the same lifecycle. Risk is objective composition, and minimax is a cohesive numerical method. No separate Studio execution path. |
| [issue:32d0935], [issue:c9ae44f] | Inference transformations and learned buffers belong to the model product. Training policies update them through method state and evaluation freezes them by default. |
| [issue:1f46470] | Ordinary chained Train requests and explicit checkpoint initialization use the shared contract. |
| [issue:4af71ac] | Driver observations and ordinary scheduled work use the shared contract. Production wiring follows the evidence gates recorded here. |
| [issue:d8f6b09], [issue:2ea6e80] | Compile the numerical update and compact health observation, retain per-update decisions, and measure the real path before promising speedup. |
| [issue:4c59e7f] | General axis design remains, with multi-level execution gated on the concrete consumer and mapping proof. Project sweep rows retain distinct run keys. |
| [issue:0dff97e], [issue:0da8b51] | Carry the shipped method diagnostics into kernel datasets and the public loader. Preserve the owner's explicit open-state direction for 0dff97e. |
| [issue:eb161f2] | Registration consolidation removes generic per-method projections. A full bound second-family comparison remains necessary for the method-floor verdict. |
| [issue:d8cc1a4] | Typed metric validity and labelled axes eliminate the false scalar-NaN placeholder. |
| [issue:8088efc] | Reexpress remaining cadence, monotonicity and phase-boundary failure coverage against the new public executor path. |
| [issue:63230c3] | Group reductions are objective semantics. The user-deferred implementation remains deferred. |
| [issue:3fc3e8e] | The shared rollout carries valid-time masks, termination and bounded trial chunks. Adaptive-length implementation belongs to the core runtime. |
| [issue:3d4b1c9] | Issue provenance is attempt metadata on the common CLI request path, excluded from scientific request identity. |

The shared contract's evidence and production gates govern consumer changes. The core-runtime coordinator supplies prepared rollout and restoration diagnostics. Jobs supplies durable binding, publication and readiness.
