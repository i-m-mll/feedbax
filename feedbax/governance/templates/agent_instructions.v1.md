## Working in a Feedbax project

This section is generated and maintained by Feedbax. Do not hand-edit it: run
`feedbax instructions install` to update it and `feedbax instructions check` to
see whether it is current. Everything outside the delimiting markers is yours.

### 1. The mental model

The spec is the model. What a compiled document says is what gets built, run,
and accounted for — there is no second place where architecture, parameters, or
topology are decided. Nothing in the pipeline may construct something the spec
does not describe, and nothing may substitute a stale or default value when a
declared one is missing. A missing piece is an error to raise, not a gap to fill
in silently. A workaround labelled "just for now" is a defect with a label on it.

### 2. Project map

This project's layout is declared, as data, in `feedbax.project.json` at the
repository root. That file is the only place the layout is stated:

- `project` — this project's name, used for diagnostics only.
- `envelope_directory` — where authored experiment envelopes live. An envelope
  alias resolves inside it and nowhere else.
- `output_directory` — where compiled documents and their locks are written. An
  authored base may never point into it.
- `authoring_budget` — the repo-relative authoring-budget document that bounds
  how large an authored envelope may be.

Read the declaration when you need a path. Do not infer paths from package
names, environment variables, or directory scans, and do not add a second
configuration source: commands take an explicit project root and read that one
file.

### 3. The experiment model

Science is authored as **envelopes**: small documents that vary a
content-pinned **base**, state why, and assert what must remain true. An
envelope is deliberately small — the authoring budget is what keeps it that way.

Feedbax owns everything downstream of authoring: compiling an envelope into a
document, writing the compile lock that pins exactly what went into it, planning
the work the document implies, fulfilling that plan, and holding custody of what
fulfillment produced. Those are not project concerns and must not be
reimplemented per project.

### 4. Naming: a name states a concept, never a value

A name is an identity, not a summary. Parameters — setpoints, grids, counts,
spans — live in the body of the document, where they can be read, diffed, and
migrated. A name that restates a parameter becomes false the moment that
parameter is revised, and a corpus named after values cannot be searched by
concept.

1. **A name encodes concept identity only.** Never a report-slot code (`k4`,
   `d1`), a model or cohort index (`m2`), or a parameter value or descriptor —
   a knot count, a grid span, a scalar setpoint. Those change; the concept does
   not. *`m2_robustness_report.v4` → `task_aware_robustness_report.v4`.*
2. **Never a campaign or coordination string.** Wave and stage ordinals
   (`wave1`, `stage2`), umbrella or issue handles, branch, worktree, and session
   names record when and how the work was organized, not what the thing is; that
   provenance belongs in the ledger and in commit history, where it stays
   accurate. *`wave1_stage.lock` → `controller_geometry.lock`, which is what the
   lock actually holds.*
3. **An adopted default versions on succession; a contrast variant may name its
   dimension.** Revising a recipe's parameters keeps the concept and bumps the
   version, so the succession is legible. Name a varied dimension only when that
   variation *is* the experiment — not when it restates the current settings.
   *`near_zero_knots.lock` → `sampling.lock.v2` (the span was a parameter of the
   adopted recipe); but `target2x.lock` → `sampling_target2x.lock` keeps
   `target2x`, because the doubled target is the contrast being run.*
4. **Study scoping is structural.** A study-specific document lives in a
   subdirectory named for the study, beneath its layer directory, and does not
   repeat the study in its filename; generic documents stay at the layer root
   under concept names. One prefix per corpus is not a hierarchy.
   *`post_run/<study>_induced_gain.base.json` →
   `post_run/<study>/induced_gain.base.json`.*
5. **Presentation never names substance.** A figure or analysis identity must
   not depend on its position or role in a report — reports bind figures, not
   the reverse — or reordering a report silently invalidates a name.
   *A figure named for its report slot is renamed `velocity_profiles`, and the
   report's slot binds it.*
6. **Schema and type identities are durable interface.** A registered schema id,
   component type id, or template id is renamed only through the migration path
   in §9, never as a side effect of renaming the file that carries it — its
   consumers bind the identity, not the path. *Moving a figure template into a
   study subdirectory preserves the `name` its consumers bind, byte for byte.*

The test for a candidate name: if a value in the body changed tomorrow, would
the name become wrong? Then it is describing a value. Name the concept.

### 5. The residence boundary, in both directions

**Belongs in this project:** modular science (components, recipes, evaluation
and analysis implementations, figure constructors), authored specs and
envelopes, data declarations, and generated custody.

**Belongs in Feedbax:** anything that emits or parses a structured record
format, any compiler, lowerer, migration, dispatch table, edge builder, or
fulfillment seam, and any mapping whose two sides are both framework-owned.
A format moves upstream; a parser for it never stays downstream.

If you find yourself writing a wrapper entry point around a Feedbax command,
that is evidence of a missing Feedbax API or CLI. File it upstream instead of
building the wrapper here.

### 6. What a science plugin may be

This project may register real scientific implementations through the
`feedbax.plugins` entry point: components, training methods, evaluation and
analysis recipes, figure constructors, report recipes. That is the whole
permitted use.

A plugin may not register a compiler, a project dialect, a declaration, an
orchestrator, a discovery mechanism, or any callable that decides how
compilation, planning, or fulfillment behaves. Project declarations are data and
are read from `feedbax.project.json`; they are not registered.

### 7. Command orientation

Run `feedbax --help` for the full inventory. The commands you will reach for:

- `feedbax preflight-experiment-envelope <envelope> --repo-root <root>` —
  compile one authored envelope into its document and compile lock.
- `feedbax fulfill-experiment-envelope <target> --receipt-root <root>` — fulfill
  one compiled target's whole dependency closure.
- `feedbax check-project-science-surface --root <root> --policy <path>
  --baseline-ref <ref>` — the deny-by-default production-Python gate.
- `feedbax instructions check` — whether this generated section is current.
- `feedbax init` — create or validate the project skeleton.

Exit codes are a contract, not decoration: **0** succeeded, **2** a stable typed
refusal with an actionable diagnostic on stderr, **1** an infrastructure failure.
A refusal (2) means the input was understood and rejected; do not retry it
unchanged, and do not work around it by editing generated output.

### 8. Science authorization

Which production Python this project is allowed to contain is decided by a
policy file, and that policy is read from a protected baseline ref — not from
the working tree. A branch cannot authorize itself by editing the file it is
judged against.

Adding a new production source file or a new top-level symbol therefore requires
the policy change to be ratified on the protected branch first. Until then the
gate fails closed, and that is the gate working correctly.

### 9. Durable formats migrate or reject

Every durable format carries explicit schema identity. When you change one, the
same change either preserves the existing semantics or adds a versioned
migration path plus focused tests for the transition. Unknown, removed, or
unsupported versions fail closed with an actionable error.

Never infer a version, never retry through a compatibility shim, and never leave
a schema-affecting change for someone downstream to reconstruct later.

### 10. Generated custody is hands-off

Compiled documents, compile locks, plans, receipts, manifests, and checkpoints
are produced artifacts. Do not hand-edit them, do not repair them by hand, and
do not delete them to make a check pass. If a generated artifact is wrong, the
authored input or the framework that produced it is wrong; fix that and
recompile.

### 11. When something does not fit: stop

If the science you need cannot be expressed through the existing generic
contracts, that is a finding, not an obstacle to route around.

Stop. Do not grow local machinery, do not add a project-local compiler or
parser, and do not introduce a transitional format "until the real one exists".
Report what could not be expressed, and add the smallest generic seam to Feedbax
that would express it.
