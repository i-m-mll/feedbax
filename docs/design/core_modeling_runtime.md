# Core modeling and runtime for project training and evaluation

The [shared core and executor contract](shared_core_executor_contract.md) owns scientific addresses, restoration, the root clock, paired trials, the objective and checkpoint selection. It records the qualification and production gates for those boundaries.

Design home: [issue:f02fdbe]. Product contract: [issue:4007bf2]. Coverage tracker: [issue:96a8f22]. Historical scientific compiler program: [issue:1eef16a].

Proposed M2 design, 2026-11-11. Source inspection used develop `6e3491e2e045a89770db3d3be8e7a1cf639a885d`. Numerical measurements come from the workloads cited below. The gates in §11 specify evidence still needed for delivery.

## 1. Outcome and scope

A researcher describes a model and a task once, trains that model, restores a checkpoint, evaluates it under explicit conditions, and obtains correctly named trajectories and objective measures. The model that runs has the same topology, parameters, physical geometry, trial timing, and interventions as the document the researcher inspected. A scientific change changes request identity or freshness with a source-mapped reason; a presentation change does neither.

The core provides this scientific computation. It does not own request admission, attempts, publication, event transport, durable status, retry, or remote resource lifecycle. Those remain in the kernel ([issue:2210d92]) and jobs ([issue:30fae90]). The project compiler ([issue:0c521d8]) expands authored steps into requests; the model compiler resolves the computation inside a request. These are different scopes with one dependency direction.

The design covers all existing model domains through the same runtime contract: causal graphs, nested neural components, scalar acausal mechanics, and articulated mechanics. It does not require M2 to introduce a general symbolic DAE compiler, a new mechanics engine, arbitrary nested rollout operators, an editable Penzai canvas, analytical controller synthesis, or spatial CAD. The existing Scene authoring and replay program ([issue:57a75dc]) remains its own product delivery.

## 2. Evidence and the decisions it supports

| Evidence | What it establishes | Limit |
|---|---|---|
| Existing graph probe results ([artifact-version:bad70b625000]), script ([artifact-version:70d9081a1982]) | One Simple Feedback graph: initial registry construction 512.4 ms; initial compile_graph 622.4 ms; subsequent identical compilation 20.1 ms; numeric mass edit 21.8 ms to rebuild and 267.7 ms for the next gradient call, with one retrace; an identical rebuild did not retrace | One graph on one machine. Does not establish plan latency, larger biomechanics latency, or the exact source of every cache miss |
| Existing eager probe results ([artifact-version:4c476776d702]), script ([artifact-version:e82d651d52ea]) | Eager point-mass construction took 1,252.8 ms; a mass edit retraced, a new-weight model did not; warm optimizer updates were 12.793 ms in that fixture | Different setup from the graph probe; their numerical timings are not a controlled graph-versus-eager benchmark |
| Scientific compiler program ([issue:1eef16a]) and current [graph compiler](../../feedbax/compiler/graph.py#L515) | Authored graph, resolved graph, source map, explicit component resolution, and executable realization already exist. Historical downstream continuation completed 1,000 versus 500+500 updates with exact terminal slots and traces | Historical evidence is valuable regression coverage, not proof of the new checkpoint or job path |
| Array-container investigation ([issue:129ede5]) | Safetensors has already been measured on real checkpoint slots and evaluation arrays; reconstructing structure from the request is already a preparation pattern | Actual method and Studio-model round trips, changing leaf types, callable leaves, and cross-version reconstruction are still owed |
| [MJX planar training evidence](brain://mjx-planar-limb-training-performance) | On Apple M4 Pro CPU the measured rigid full update was 26.82 ms analytical versus 118.40 ms MJX; mixed compliant was 32.713 versus 122.440 ms. All three MJX seeds in each workload met held-out targets | One planar workload and CPU. The compliant update exceeded the frozen 120 ms limit by 2.440 ms. Analytical end-to-end time to target was not measured |
| [MJX derivative evidence](brain://mjx-differentiation-biomechanical-modeling) and [backend qualification](brain://mjx-backend-interchangeability) | Particular constrained reverse-mode and inside-wrap paths fail; qualified constraint-free paths and fixed-guide muscle paths work. Warp has qualified forward execution and state transfer on the tested platforms | Forward correctness does not establish simulator gradients; a backend name is not a capability guarantee for every model |

Current code already centralizes the scan harness in [runtime/_rollout.py](../../feedbax/runtime/_rollout.py#L75), names StateIndex markers by occurrence in [state_indices.py](../../feedbax/runtime/state_indices.py#L57), and provides a common batched rollout utility in [analysis/rollout.py](../../feedbax/analysis/rollout.py#L386). The work is to make their contracts the only path for new project execution and correct the missing boundaries, not to assume none of this exists.

The existing libraries remain the implementation foundation: Pydantic for host-side schemas, JAX and Equinox for pure numerical execution, Optax for applicable optimizer updates, Diffrax and Optimistix for supported solver paths, and MuJoCo/MJX for articulated mechanics. [JAX documents static arguments and function identity as compilation-cache inputs](https://docs.jax.dev/en/latest/jit-compilation.html); [Equinox documents state separately from parameters](https://docs.kidger.site/equinox/api/nn/stateful/) and [restoration into a reconstructed template](https://docs.kidger.site/equinox/api/serialisation/). Those precedents support the separation below. They do not prove Feedbax meets its timing or restoration gates.

## 3. One authoring authority, three useful representations

### Authored scientific documents

The project owns named model, task, objective, and training/evaluation documents. A model is a root composite with one step period, its port list and its interior. A nested composite owns its port list and interior as fields. Its outside card and the boundary cards inside project that list. A leaf uses the ports declared by its registered type. Wires belong to one interior and join local component ports or that interior's boundary ports. Referenced imported assets are immutable, content-addressed inputs. Presentation and the path shown by a window are stored separately from the model. [issue:a1bdcee] [issue:6204d45] [issue:6a2760f]

A typed address contains the model document name, the composite IDs leading to the containing graph, and the element's local address. A port selects its owner and a declared port ID. An attachment selects its wire or port and its local ID. A field extends its owner's address. Python and TypeScript consume the generated contract. Parent and child may use the same local ID and have separate state, bindings, selections, positions and edits. [issue:a1bdcee] [issue:0d12b4f]

The identity encoder uses declared schema fields. Graph element IDs, topology, and the root port names used in task pairing enter request identity. Component names and interior port names leave it unchanged. Element IDs are assigned at creation and stay fixed across renames and retargeting. Grouping and ungrouping change addresses as a whole operation. An explicit schema migration may map old addresses to new ones when reading historical content, with that mapping recorded on the interpretation of the checkpoint.

Authored component IDs, types, configuration and explicit connections are authoritative. Inferred port widths, executable order, StateIndex markers, flattened state addresses and generated intervention component IDs are derived projections. Their descriptors carry enough origin and schema information to resolve checkpoint values back to the authored occurrence; they are not independently authored IDs. A change to their generation can require an explicit descriptor migration even when the scientific graph is unchanged. Kernel identity and aliases remain owned by [issue:d601693], not by a second runtime hash.

Templates are pure document factories. Instantiating a template produces a concrete graph; only that graph is executable authority. A template-generated interior is either derived from its template configuration or detached as an editable concrete graph. It cannot have both independently editable authorities. Detaching preserves the generated graph; regeneration is one explicit undoable replacement. No hidden wrapper class substitutes for the visible graph.

Python has two entry points with different promises. Registered portable components and graph factories produce these same documents and use the same compiler. An arbitrary eager GraphNode remains usable as a local numerical function, but cannot become a durable project job until its component declaration can reconstruct it. There is no attempt to infer a portable program from arbitrary closures.

### Resolved computation

Resolution produces one immutable scientific program, with:

- resolved component declarations and expanded templates;
- typed interfaces, inferred shapes, units where declared, parameter bindings and ownership;
- deterministic causal schedule and ordered wire attachments, including delay initializers;
- declared state and parameter addresses, task bindings, objective observations, and source maps;
- domain-lowering descriptions, asset references, and capability requirements.

This is the model compiler's IR. It is derived from authored documents and is not a second editable graph. The project plan may refer to its structural validation and source map, but neither a process-local executable nor a JIT cache becomes an authored document.

Keep the authored → resolved → executable separation. The host-side planning path resolves declarations, dimensions, source addresses and domain structure without allocating device training state or compiling XLA. Domain-specific resolution can produce reusable structural metadata. Worker preparation constructs the numerical runtime.

A compiler version can identify the format of compiler evidence. It is not a substitute for the kernel's observed dependency trace, and no manually maintained compiler revision decides freshness.

### Executable computation and numerical values

An executable is a process-local realization of the resolved program. It contains static topology and callable structure plus stable projections of numerical values and state. Its executable cache is disposable; checkpoints and results never require a serialized executable.

Distinguish three independent properties of every value:

1. **Structural or numerical:** topology, array shape, dtype, solver algorithm, and fixed loop structure specialize the executable; numerical arrays are arguments.
2. **Ownership/lifetime:** model values, persistent inference state, per-trial state, recurrent carry, method state, and executor cursor have different reset and checkpoint rules.
3. **Trainability:** a method selects eligible numerical leaves for update. Frozen mass is still numerical; trainability does not determine whether a value is static.

Do not place JAX arrays in static metadata because they are frozen during training. Explicitly type numerical scalars before tracing to avoid weak-type and dtype drift. Generated native mechanics buffers are derived from owned source parameters; they are neither independently editable values nor a second checkpoint authority.

A numerical edit can still require native engine preparation or a different executable when the engine's actual structural predicates change. The contract is correct binding and recorded preparation, not universal cache reuse. [issue:436cfef] retains the larger MJX reuse investigation; M2 proves the simple point-mass numerical boundary first.

## 4. Components and the compiler

### One declaration per component

Extend the existing ComponentOwner, typed configuration, Ports, ConfigBinding, and registry declarations rather than introducing a competing plugin model. A declaration owns:

| Facet | Meaning |
|---|---|
| Type and schema | Stable type ID, current authoring schema, permanent absent-value semantics, explicit migrations |
| Configuration | Typed authored fields, constraints, units, authoring defaults, and mapping to runtime-owned values |
| Ports | Signal input/output schemas or conserving physical ports; dynamic arity derived from configuration |
| Shape/state description | Abstract output and state shapes, reset inputs, initial-state ownership, inference-state ownership |
| Runtime | Construction and pure transition or domain-lowering implementation |
| Capabilities | Required inputs and supported execution/derivative modes for this resolved instance |
| Presentation | Labels and editor metadata projected from the declaration; specialized domain presentation may add geometry information |

A registry catalogs declarations; it does not restate their parameter semantics. Cache its immutable catalog per locked environment/plugin set. Forms, serialization, migration, validation, and constructor bindings are projections. Type/schema revision records authored meaning, while code freshness remains the kernel trace's job.

Signal ports and conserving ports are distinct typed variants. Conserving connections resolve inside a physical domain and have no causal schedule. Registered declarations also identify authoritative source and learned values, inference values and public component states by stable symbols. Each public state has its value schema, lifetime and reset semantics. Plan reports missing declarations on the owning component's address. Runtime state indices and private buffers are implementation details. [issue:6b6a44b] [issue:6204d45]

Unknown type, invalid parameter, missing shape input, mismatched ports, duplicate input drivers, and unsupported capabilities yield Problems with the typed model address and the authored source path when both are available. A Problem carries its phase, expected condition and observed condition. Expected refusal returns typed error data at the public boundary. Numerical solver failures carry the affected trial or condition where possible. [issue:7449df8]

### Compilation phases and ownership

| Phase | Owner | Work and output |
|---|---|---|
| Parse and normalize | Project/model schema | Parse one supported document schema, apply explicit migrations, resolve permanent absent values |
| Resolve declarations | Core compiler | Resolve type IDs, configuration, inputs, and immutable assets; produce source-mapped errors |
| Expand and lower | Core with domain implementations | Expand templates; lower interventions; assemble acausal interiors or articulated model descriptions |
| Solve interfaces | Core compiler | Infer port/state/parameter shapes; check units/frame constraints where declared; bind task inputs and selectors |
| Validate and schedule | Core compiler | Validate all reference dependencies, recurrence and initializers; establish deterministic execution |
| Realize and restore | Worker using core | Construct process-local model and state templates; restore selected checkpoint slots; apply authorized evaluation conditions |
| Specialize | Worker using JAX/domain backend | Compile numerical workload for actual shapes, dtypes, algorithms and derivative mode |
| Execute | Core numerical functions, called by method/evaluator | Return new state, observations, metrics, and typed numerical status |
| Publish | Executor/kernel/jobs | Write immutable products and receipts, append events, move refs |

The first five phases support plan and editor validation. They do not run the optimizer. Worker phases report importing, resolving/building, restoring, and compiling as bounded phase events. Plan can validate a future best-checkpoint reference against candidate training definitions without opening a checkpoint that does not exist.

Every reference is in one dependency inventory: wires, task bindings, state selectors, parameter bindings, recurrent initializers, nested histories, and generated intervention routes. Removing a referenced component either removes an explicitly selected dependent edit or rejects the edit with its remaining references. This resolves the recurrent-initializer deletion defect without another ad hoc scan.

Incremental compilation reuses resolution by document dependency and structural signature. Numeric values are rebound through the same declaration. A cache hit must attach the computation's recorded code/binding/environment dependencies to the current execution trace; an executable cache hit must never make those dependencies disappear. The trace proof owns the exact recorder mechanism.

## 5. Runtime semantics

### One transition and one carry

The common pure runtime is conceptually:

~~~
reset(program, model_values, trial, trial_key) -> Result<RunCarry, RuntimeDiagnostic>
step(program, model_values, carry, input_at_step) -> StepResult
rollout(program, model_values, carry, trial, observation_plan) -> RolloutResult
~~~

RunCarry contains component state, local and nested recurrent carry, the next root step index, and the trial random root needed for continuation. Persistent inference state is supplied and returned explicitly where a component has it. It is not confused with trial reset state. One resumed step must not require the caller to reconstruct a forgotten cycle buffer or guess the next timestep.

Keep Equinox State/StateIndex for component-owned state. Runtime checkpoint descriptors give slots stable scoped names; Python object identity and private StateIndex marker representation are not durable identity. The run carry is an ordinary typed pytree around the existing component state, not a replacement state library.

At step k, inputs refer to the interval [time(k), time(k+1)). A wire with no delay carries the current signal. Attachments act in order from source to target. A delay reads its initialized or previously written value and adds the incoming value to its carry for later steps. A mechanism's state-only output is available at the start of the interval. An ordinary nested graph advances once per parent step and inherits the root clock. [issue:b390808] [issue:6204d45]

A component needing time receives an explicit StepContext projection: root step index, interval start/end or dt, and declared execution mode. It does not discover time by reading a global or maintain a duplicate counter. A Clock component may expose this context as graph signals for ordinary wiring. Components can own a local clock only when it represents a distinct reset/elapsed-time state. Manual step and scanned rollout use the same carry and context; no implicit default back to step zero.

Physics substeps belong to the mechanics component's declared integrator, with their own physical substep time. They do not advance the controller graph repeatedly. An intentional inner iterative process must be an explicit component/operator with its own state and axis semantics; ordinary Subgraph does not acquire an n_steps switch in M2.

State history is a named observation projection, not a second runtime state system. SimpleFeedbackState may remain an ergonomic view during migration, but objectives and persisted results select declared paths. The scan does not reconstruct and store a whole privileged state view when only two arrays or a streaming objective are needed.

### Randomness and mapping

The [shared trial-key contract](shared_core_executor_contract.md#4-paired-trial-streams) defines stream purposes, pairing, and independence. The runtime derives random scopes from stable occurrence addresses and declared random-site purposes. State, parameter and delay checkpoint paths derive from the same addresses and declared member IDs. Renaming a component or interior port preserves those paths and its stream. Grouping moves occurrences to a new path.

Plan-level sweep rows are runs. Within one request, trial, condition, ensemble, and a method's population axes are declared mapped dimensions. The preparation layer presents their names, coordinates, leaf broadcast/mapped status and valid shapes once; methods do not infer them from tree position. Same-shape groups may be vmapped or mapped/chunked without changing scientific identity, provided their math and reduction order contract remain the same. A reduction or batch change that changes training mathematics belongs in the request.

## 6. Tasks, objectives, interventions, and evaluation

### Task and objective boundaries

The task interface is `AbstractTask`. The task document runtime supplies signals, variables, initialization and epoch bounds through `init` and `step`. Plan pairs its signals with the model's root inputs by compatible name and type, applies explicit exceptions, and resolves Links. An input with a default can remain unbound. Plan resolves time lengths at the model's root step period. [issue:b584e17] [issue:6a2760f]

Task-to-model bindings use typed model addresses and declared ports. Python selectors for portable projects resolve to these same declarations.

An in-rollout task composes its `step` into the lowered graph and supplies the recorded epoch bounds from its final state. A live model-side policy is ordinary graph computation over declared signals and the current step's mechanism outputs. [issue:b584e17] [issue:6204d45]

An objective term is a typed measure expression, window, reduction and weight. The measure names its targets. The compiler derives one observation plan for objective terms, requested retention and diagnostics. Train and Evaluate call the same term executor. Evaluate uses the trained objective and adds no authored terms. [issue:74f00a0] [issue:a1382d5]

### Evaluation variants and binding

The [shared evaluation contract](shared_core_executor_contract.md#1-evaluation-uses-the-existing-project-variation-model) defines the authorized model/task diff, owner compatibility and value precedence. Train and Evaluate compile the same stored document recursively. A compatible nested variant retains authoritative values at their declared addresses. The [finite set contract](shared_core_executor_contract.md#9-finite-checkpoint-sets-bind-by-coordinate) owns candidate coordinates, readiness and exact binding. Plan validates known interfaces, task binding, attachments and runtime capabilities with authored diagnostics. The worker authenticates the selected checkpoint and verifies its actual stored layout before realization.

### Intervention semantics

M2 provides registered Scale and Clamp interventions on wires and source ports. Each owns its typed fields and application window. A wire intervention affects its one reader. A source-port intervention affects every reader of the port. Train stores the attachment on the model document. Evaluate authors it as a diff at the same address. [issue:b390808]

An intervention before a delay changes the value carried to a later step. An intervention after it changes the value consumed on the current step. The list order establishes execution order. Delay and Initial fields define the step count and the reset values; a Link initializer may read a task variable or a local port at reset. [issue:b390808]

Between-condition parameter overrides follow the shared load and realization order. A component with within-trial adaptive weights declares them as state with initialization and reset semantics.

### Probe observation

A probe names the signal at its port or at its ordered point on a wire. It reads the value left by preceding attachments. Objective measures and analysis can select its observation by declaration. Retention is selected by the step, with M3 adding probe retention and folds. Public state without a port is available through the raw-state projection. [issue:b390808] [issue:e5adc22]

### Built-in evaluation and analysis boundary

The built-in evaluate executor ([issue:dbbfb13]) binds a model checkpoint, obtains validation trials or a seeded sample, applies conditions/interventions, invokes the shared rollout and objective functions, and returns datasets. It neither registers an old EvaluationRecipe nor creates its own cache or scheduler.

State samples contain the reset state and each post-transition state, with physical timestamps. Transition outputs/actions have their own step/time alignment, so an N-step action array is not falsely labeled as an N+1-state array. Store trial specifications, condition coordinates, units, objective terms, validity masks, and exact model/asset references alongside these arrays. Replay is a projection of this dataset and the original model closure, as required by [issue:57a75dc]; changing the draft cannot change old replay anatomy.

The container follows the existing dataset contract ([issue:d0019f7]): labeled coordinates and units remain host metadata, while JAX receives only the selected arrays and any coordinates it computes with. Per-member physical timestamps that are not one shared coordinate vector are a named data variable over condition/trial/time, with an integer sample coordinate and the valid-sample mask. Changing an unused label or metadata attribute must not specialize the numerical function.

M2 defaults to retaining all declared public runtime state variables, with size estimated from shape and dtype before running and actual bytes afterward. Backend scratch buffers are not public states. Observation subsets use the same projection machinery. M3 adds fold products and analysis operations. A computation that advances a model produces an evaluate request; analysis over already stored states produces analyze requests. Analysis can select observations but must not hide new trajectories inside a statistic without declaring the evaluation dependency.

## 7. The exact interface supplied to training and jobs

The following names describe the interface, not a demand for these exact class names.

| Contract | Core supplies | Executor/method responsibility |
|---|---|---|
| ResolvedComputation | Source documents; model/task/objective interfaces; typed paths; layout and capabilities; source map | Resolve method requirements against capabilities; reject unsupported combinations before updates |
| PreparedComputation | Stable executable structure; initial model values; slot codecs/layouts; trial sampler/reset/step/rollout/objective functions; mapped axes and observation plan | Construct method state and optimizer using the declared parameter selection; choose valid batching |
| Model slot | All model numerical values needed for inference, including frozen parameters, learned reset values and persistent inference state; reconstructible template reference | Publish the model slot at each checkpoint; evaluation reads this slot without method/optimizer slots |
| RunCarry | Component and solver state, recurrence, next trial step and RNG root | Retain only for methods that continue episodes across updates; otherwise reset trials each update |
| Objective result | Scalar loss plus named terms, reductions, relevant valid counts, numerical diagnostics | Differentiate/update or consume as fitness/reward as the registered method specifies |
| Core error | Structured source/path/capability/numerical refusal | Convert to the executor's terminal error and outcome; jobs records it and surfaces the same reason |
| Runtime cost facts | Preparation/compile/step timings and shapes, actual capability | Report phases and metrics; choose launch resource limits and rehearsal bounds |

A method owns its mathematical update. The executor owns request-local work, checkpoint cadence, publication, stop, and durable cursor. Core neither embeds an optimizer loop in Graph nor knows a job ID.

The method-facing transition is conceptually update(prepared, model_slot, method_state, executor_inputs) → update_result. The result contains a coherent new model slot, method state, committed progress coordinate, metrics, and any explicitly retained continuation state. Methods may implement gradient descent, RL, local update rules, or population methods without receiving a different job lifecycle. A method requiring reverse-mode simulator derivatives cannot run against a forward-only mechanics configuration. PPO can require controller derivatives without simulator derivatives; those capabilities must be distinct.

Method construction receives parameter/state descriptors and mapped axes, not an opaque model tree that it must rediscover. Named parameter selections are checked once against those descriptors. The jobs and training design owns the method API, schedule and optimizer carry policy, and executor replacement. Core supplies the descriptors and compatible restoration used by that executor. [issue:a1382d5]

Metrics travel from committed updates. Emit available metrics at 2 Hz or better when updates are fast enough; during a longer compiled update emit phase/heartbeat progress with the last completed metric, rather than fabricating new measurements. One logical update may contain multiple device calls, but only a complete method transition is checkpointable unless the method explicitly declares additional safe continuation boundaries.

Stop is observed before the next pending update. Finish an in-flight transition, commit its coherent state, and publish a checkpoint at that boundary regardless of normal cadence. A device call cannot be cooperatively interrupted halfway through. Forced termination returns to the last published checkpoint; it cannot claim that unsaved in-memory progress is resumable.

A rehearsal uses the exact preparation and method path on a separate bounded request/attempt with explicit reduced update/trial limits. Its checkpoint and receipt go to scratch. Core supplies bounded shapes and execution functions; jobs owns rehearsal admission, outcome and cleanup.

## 8. Checkpoint reconstruction and continuation

Core supplies the model descriptor, typed value destinations and effective-value-first realization under the [shared restoration contract](shared_core_executor_contract.md#3-model-value-authority-and-reconstruction). The model projection supports fresh trials. Each method supplies the layout of its own continuation state, including carried episodes when required. The kernel owns content custody and the executor owns publication. Prepared implementation structures come from compatible registrations.

Each Train request prepares its own effective model and method template. A downstream Train step resolves inherited settings through ordinary project definitions and obtains authoritative predecessor values through its bound `last` checkpoint input. Core validates retained value destinations and restores or initializes the resulting model under the [shared chained Train contract](shared_core_executor_contract.md#chained-train-steps). The executor applies the declared method and optimizer state policy. The production semantic address and slot format gates remain there.

## 9. Mechanics and acausal boundaries

Keep one physical authoring authority and the existing engine families. A mechanics component advances the plant for one controller interval, observes named physical quantities, resets, exports/restores continuation state, and reports concrete capabilities. Its graph-facing interface does not expose private engine buffers.

Scalar translational and rotational systems share the across/through algebra under [issue:45f9661], while retaining meaningful domain names. Resolve conserving connections and structural constraints once, then lower supported networks to the existing Diffrax/Optimistix-backed component. Acausal is an authoring domain with a causal boundary, not an alternative scheduler. Unsupported index, singular structure, incompatible domain, missing inertia, or failed solve must name the physical elements involved. Do not claim a general-purpose Modelica compiler from the existing restricted assembler.

Articulated bodies and attachments use the existing authored model → native compiled descriptor → backend path. Bodies, joint coordinates, muscle paths, frames and units retain stable identities. Exact numerical/asset identity governs results; structural signatures permit compatible executable grouping only. The analytical point-mass/two-link path remains a deliberate qualified specialization, not an automatic fallback from MJX failure. The existing generic engine choice remains dense MJX; the CPU measurements do not justify implementing a new N-link engine for M2.

A muscle law owns activation and any fiber/compliance state and computes its derivative and tension from excitation and path kinematics. The physical integrator couples those derivatives with skeletal dynamics at the same integration stage. Activation is not stepped in a separate controller-time loop. Rigid and compliant variants are declared laws sharing named attachments, not another plant inheritance tree. Current [MuscleLaw and MuscleLawBank](../../feedbax/mechanics/muscle_law.py#L19) already establish this useful boundary.

Backend support is checked for the resolved model, selected solver/integrator, derivative mode, precision, and required observables. Warp forward support cannot satisfy reverse-mode dynamics training: [upstream MJX documentation](https://mujoco.readthedocs.io/en/latest/mjx.html) explicitly distinguishes this. Imported models that trigger known constrained-solver or inside-wrap derivative failures are rejected for that mode until the exact path is qualified. Do not silently reduce solver iterations or switch to finite differences.

Restore common physical state through the backend adapter and reconstruct its derived buffers. Exact same-backend continuation also includes any required solver/warmstart/internal state. Cross-backend transfer is a separately qualified initialization, never presumed exact continuation. Sleeping/contact private-state cases outside the qualification remain unsupported for that transfer.

Use existing Diffrax adjoints appropriate to the requested derivative mode; [RecursiveCheckpointAdjoint](https://docs.kidger.site/diffrax/api/adjoints/) supports its documented reverse-mode path, not arbitrary forward-mode differentiation. [Optimistix](https://docs.kidger.site/optimistix/api/root_find/) exposes solver result status and explicit adjoint selection. These libraries remove a reason to invent solver-control protocols, while Feedbax still owns model-level derivative and continuation acceptance.

## 10. M1 handoff

M1 uses today's TrainingRunSpec, compile_graph/ExecutableGraph, task/objective preparation, method, and checkpoint path. The host treats payload and product keys as opaque. M1 does not wait for the M2 IR split, new array container, or full generic intervention work.

The proof fixture is the exact Studio-authored model/task used by [issue:2d24abe], agreed with its owner and stored once. The current coordinated choice is direct-force PointMass reaching with a GRU and linear readout, with no MLP; the exact authored file and normal execution entry still need to be recorded by the Studio/jobs owner. The currently registered checkpointable Studio graph method requires batch_size=1 and a positive horizon; the existing Studio default is 64. The fixture must explicitly use the supported settings, and the surface must either provide those settings or report the capability constraint rather than silently lower the requested batch size. This is an M1 limitation, not the full M2 batching contract. A separate MLP fixture answers [issue:ff4f6cc]. A bare leaf-codec failure is not yet an executor-path reproduction.

Use execute_training_run_spec through the actual preparation path. Compare uninterrupted and stopped/restored completion under the same locked environment, seeds and update schedule: model, optimizer/method state, cursor/schedule/RNG slots, loss/term sequence, and terminal outputs. Stop between ordinary checkpoint cadences, not only at a convenient pre-existing checkpoint. Rebuild the model in a fresh process on restore.

Adapt host Stop to [the existing stop_before_update seam](../../feedbax/training/executor.py#L709), which finalizes the last committed state independently of scheduled checkpoint cadence. In M1 the first update may complete to establish the first resumable checkpoint if the current format requires it. M2 supports publishing the initialized state at update zero. App/daemon survival and event replay remain jobs acceptance.

## 11. Delivery and realistic gates

All performance numbers below are targets unless linked to an existing measurement. A failed target is reported with the measured workload and corrected on its owning issue; it is not silently replaced with a smaller benchmark.

### Before M2 consumers

1. Publish this core contract in [issue:f02fdbe] and the executor-side agreement on [issue:30fae90]. Resolve slot ownership, inference state, method requirements, checkpoint initialization and stop semantics, mapped axes and reconstruction in both descriptions before the train and evaluate request consumers build.
2. Complete the request identity ([issue:d601693]), execution trace ([issue:fca88cd]), and project conversion ([issue:26cf5d3]) gates.
3. Freeze a shared smallest complete train→checkpoint→restore→evaluate fixture, plus the MLP and condition-order adversarial cases. Reuse historical downstream exact-resume evidence as an acceptance source; migration of that downstream corpus remains M5 and the user's choice.

### M2 core acceptance

| Gate | Required evidence |
|---|---|
| Authority and compilation | The same portable model authored as Python/spec/Studio resolves to equivalent topology, parameter ownership and observable paths. Source-mapped diagnostics name an invalid nested binding and a deleted recurrent initializer. Presentation-only edit leaves scientific identity unchanged |
| Runtime and continuation | Manual step and scan agree on a stochastic recurrent nested fixture; split rollout with complete carry agrees with uninterrupted execution, including the first step after restore. Initial state, recurrent read/write timing and schedule indexing are asserted numerically |
| Shared objective | Training and evaluate compute equal objective terms on identical model values and replayed trials. Streaming versus retained execution agrees only for declared streamable terms |
| Checkpoints | Fresh-process safetensors round trip through production preparation for the shared Studio model, MLP, and a registered method with changing scalar state; equal state yields equal bytes across two processes; model-only load reads no optimizer/method slot |
| Conditions | Restore trained checkpoint mass 1, evaluate mass 2, verify the executed mass and a physical response while learned weights stay equal; derived mechanics fields agree with a fresh build. Structural width change fails with its path |
| Binding and closure | Plan an evaluate request before best-checkpoint publication; start after publication and verify its exact binding. Model-only reconstruction works from a progress root without a terminal receipt. Resume retains binding after a newer upstream checkpoint appears |
| Dataset/time contract | Two conditions with known trajectories expose correct state and action timestamps, objective terms, masks, trial metadata and model reference. Unequal horizons have explicit valid lengths; incompatible feature shapes are refused |
| Capability refusal | Forward-only dynamics cannot enter a method requiring simulator reverse mode; missing model assets and incompatible descriptors fail before updates. Supported existing mechanics fixtures retain their qualified behavior |
| Real product | M2 portions of the real-backend walkthrough pass: train without sweep, stop/resume, evaluate two conditions, edit a parameter, see the precise stale reason, rerun only affected work, compare previous output, and evaluate an upstream subset |

Run appropriate Pants-selected tests for changed Python dependencies and use the shared acceptance walkthrough instead of building a second integration harness. Keep tests for surviving behavior, rewrite their entry points when replaced, retire tests only with deleted behavior/interfaces, and record those dispositions with each change.

### Performance and resource evidence

Measure cold and warm registry/resolution, native preparation, restore, JAX compilation/cache hit, first update, steady update, checkpoint publication, first evaluation, memory high-water mark and bytes read. Use the same model/task and synchronized device completion. Separate executable-cache reuse from result reuse.

The umbrella's first event target is 100 ms on an idle local host; cached first training update is 1 s for the small fixture, not for arbitrary biomechanics. Cold or structurally changed models show honest phases. Keep the existing whole-project plan targets (500 ms for 100 requests; 5 s for 10,000) and incremental plan/admit target under [issue:736786f]; measure them after the model-resolution split. The graph probe's warm 20.1 ms alone cannot establish the 30 ms cross-layer budget.

The simple numeric mass edit must avoid a new JAX specialization once its values are correctly dynamic; a hidden-width change should specialize. For native mechanics, report rebuild versus compile reuse honestly under [issue:436cfef]. Do not rerun the full biomechanics qualification simply because runtime wrappers were renamed; rerun affected physics/gradient cases when their inputs or semantics change.

A new finite-difference full-loss directional derivative check with replayed noise is required when modifying the derivative path, with tolerance tied to the matched model/precision/integration refinement. Existing held-out learning measurements are retained as evidence rather than claimed again. No M2 acceptance requires a paid GPU run.

### Sequencing

- **M1:** actual current executor stop/checkpoint/resume and MLP disposition.
- **M2 foundation:** component ownership/layout and compiler split; shared carry/time/observation contract; array-container reconstruction.
- **M2 integration:** project train/evaluate consumers, bind-time model restoration, condition/intervention lowering, datasets, source-linked replay inputs; product walkthrough.
- **M3:** folds and analysis projection; cross-surface notebook/CLI polish on the same runtime.
- **M5:** selected downstream migration, then delete protected old paths with their callers. No new feature goes onto those paths before deletion.

New project consumers must share this implementation; the M1 executor integration survives the M2 representation replacement. Existing envelope CLI and orchestration compatibility promised by [issue:4007bf2] remains intact until M5.

## 12. Disposition of the coverage tracker's core issues

These are proposed dispositions with explicit carriers. Do not close an issue as delivered because the design answers it; close as carried only after its remaining acceptance is copied to the carrier, or as delivered after evidence.

| Issue | Decision and acceptance carrier |
|---|---|
| [issue:84c3ec4] ChannelSpec | Channel shape comes from resolved connected-port prototypes; selector and delay are authored configuration. Remove any parallel ChannelSpec after caller inventory proves replacement. Core compiler/interface gate carries inferred shape and deterministic channel reset |
| [issue:e7b6e26] WhereDict/lambda keys | Durable selectors use typed scoped paths with explicit attribute/item/index semantics. Python lambdas may resolve to that form at authoring; no bytecode/lambda equality in persistence. Core binding/selector gate carries round-trip and nested identity |
| [issue:d999dea] Model needs task at construction | Model requires the task interface, not the task instance. Joint interface resolution supplies neural input shape; sampled task values remain runtime inputs. Shared fixture proves task value edits preserve topology |
| [issue:fb6fffc] Muscle composition | Typed law configuration and the existing MuscleLawBank compose heterogeneous laws over named paths. No new parallel muscle construction hierarchy. Mechanics carrier retains heterogeneous law and parameter reconstruction checks |
| [issue:138c2d3] Activation ownership | Activation dynamics belong to the law; integrate all physical states at one solver stage. Retain registered behavior and qualify any changed old-law implementation before deleting it |
| [issue:9b97528] SimpleFeedbackState | Named observation projection replaces privileged runtime/history class. Streaming retention and objectives share the projection. Core runtime gate carries equal histories/terms |
| [issue:b68e68c] Retire xabdeef | Replace supported factories with portable templates and objective/task documents; inventory docs/notebooks/tests at removal. Delete when all consumers move, with protected old downstream path removed no earlier than M5 |
| [issue:e919d5d] Parameter interventions/online learning | Between-condition numeric overrides use restore→patch→rebind. Within-trial adaptive weights are explicit component state; across-update learning belongs to method. Core gate carries override order; general online-learning component is not required to finish M2 |
| [issue:29c89d8] Static arrays | Numerical values are dynamic leaves regardless of trainability. Fix remaining array-bearing static declarations on touched/runtime-reachable components; numeric-edit specialization gate catches regression |
| [issue:c1ad986] Neural internal uniformity | Equinox remains the state system. Its actual module tree, declarations and typed paths supply inspection; no shadow nn_tree authority. Keep a focused selector/edit/inspection example; do not adopt Penzai state merely for display |
| [issue:e0686e7] Named neural axes | Named axes are required at trial/condition/dataset boundaries. No Haliax-wide numerical rewrite for M2 without a concrete component need. Close this broad adoption question as not scheduled or retain a specific future component proposal |
| [issue:12e94bc] Constant input variants | Move equivalent behavior into the generic operation plus explicit subset selector; preserve unit masking semantics before deleting a variant. Carried by [issue:b3ad156] |
| [issue:3fb7f56] Runtime timestep | RunCarry owns next step; StepContext exposes physical time and mode; nested graphs share parent tick. Runtime continuation gate carries manual/scan/reset/restore agreement |
| [issue:2f45ef8] Dynamic policies | A policy reads the current mechanism state and declared action, target and time signals before mechanics advance. Prove linear and affine live policy traces against a direct reference |
| [issue:3fae4f8] Analytical lowering | Keep separate from M2. Later registered lowering may expose LTI/LTV plus explicit assumptions; approximate linearization must not be labeled an exact representation. Runtime observation/linearization capabilities provide its seam |
| [issue:56dfd97] Nested inner iteration | Ordinary subgraphs execute once. Explicit iteration operators need named inner time, reset and carry semantics; deferred implementation, with a diagnostic preventing accidental nested rollout. Mechanics substepping remains supported |
| [issue:2f8dd61] Parametric composites | Pure template configuration generates the sole concrete interior. Derived versus detached is explicit and regeneration one undoable operation. Core owns generation semantics; Studio owns editing/history |
| [issue:b3ad156] Generic interventions | M2 qualifies ordered wire and source-port Scale and Clamp attachments, including application windows and delay-relative timing. The condition/intervention gate exercises the resulting execution and masking |
| [issue:3b2affb] Penzai canvas | Explicit user deferral remains in force. Do not schedule or close as absorbed. Existing foreign components retain their declared runtime capability boundary |
| [issue:c58dd36] Recurrent initializer deletion | All semantic references participate in compiler/edit dependency inventory. Deleting an initializer source is rejected with its dependency or an explicit dependent edit; include nested history routes |
| [issue:008332e] Subgraph editor isolation | Immediate task-binding case was fixed, broad invariant remains open. Scoped occurrence identity is core's contract; Studio [issue:ed09a0a] carries tests for equal local IDs across components, edges, bindings, selection, dynamic ports, undo and navigation |
| [issue:f3f8588] Publish jax-cookbook | Distribution work, not a core runtime design gate. Keep separate; the checkpoint descriptor/template boundary addresses persisted class relocation when publication happens |
| [issue:45f9661] Acausal algebra and fixtures | Core owns one domain-parameterized scalar algebra with domain-specific authoring names. Fixture consolidation follows actual surviving tests; the analysis JVP cleanup remains with its surviving operation owner |

The training-executor issues listed in [issue:96a8f22] remain the jobs/training planner's disposition responsibility. The built-in evaluation's previously undesigned condition/intervention surface and the evaluation-versus-analysis boundary are resolved by sections 6–8. The broader plugin-governance, macOS packaging, and provider programs are not absorbed into core.

## 13. Cross-layer agreement and remaining evidence

The core design is owned by [issue:f02fdbe]. The training and job contracts are owned by [issue:a1382d5] and [issue:30fae90]. The checkpoint container and built-in evaluation specifications are [issue:129ede5] and [issue:dbbfb13].

The engineering decisions are sufficiently specific to implement: single component declaration, source-mapped compiler split, explicit numerical ownership, one carry/clock, restore-before-condition order, model-only closure, shape refusal, typed interventions, existing mechanics engines, and distinct freshness/compatibility/exactness. The remaining coordination choices are the exact shared fixture and the executor's final API spelling, method-state layout and checkpoint compatibility declarations; these belong in the jobs agreement before dependent implementation.

The remaining evidence gates are real work, not design ambiguity: production restoration round trips, actual stop/resume, numerical intervention and timing equivalence, native derived-parameter rebuilding, bound-checkpoint closure, and measured plan/worker performance. A result remains a proposal until that evidence is recorded.
