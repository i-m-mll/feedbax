# Feedbax external conformance fixture

This repo-owned package is the clean-installed downstream-author example for
`feedbax.testing`. It extends that kit instead of creating a second generic
harness: the fixture calls the public conformance helper for material
dependencies and keeps only fixture-specific builders and result handling in
this package.

Run it through the repository command:

```console
uv run --no-sync python scripts/run_external_conformance.py
```

The command builds both Feedbax and this package as wheels, installs them
non-editably into a fresh environment, changes away from the source checkout,
removes `PYTHONPATH`, denies runner-process outbound TCP connects before any
Feedbax execution-stack import, and verifies that all imported Feedbax modules
come from the installed wheel. This is not an OS-level DNS, UDP, or arbitrary
child-process sandbox. The only lifecycle child is asserted to remain the
fixed print-only Python command.

The fixture runs the public production `StageEngine` with
`LocalOrchestrationDriver` through a clean-wheel PREFLIGHT and LAUNCH. It stops
at the persisted PREFLIGHT boundary, reconstructs the engine, and finishes the
bounded local lifecycle from the stored public state. The exact revision gate
therefore authenticates installed-wheel provenance at both execution
boundaries. The deployment policy is explicitly local and cloud-unauthorized;
all custody, orchestration, and cache paths are unique temporary directories.

The result schema is `feedbax.external_conformance.result.v14`. Its evidence map
has exactly fourteen strict-boolean cases, in this order:

1. `ordered_registration`
2. `unified_plugin_bootstrap`
3. `external_driver_plugin`
4. `component_registration_and_migration`
5. `dynamic_component_ports`
6. `value_identity`
7. `component_param_array_values`
8. `material_dependencies`
9. `staged_exact_parent_migration`
10. `resolved_evaluation_row_projection`
11. `public_lifecycle_recovery`
12. `custody_persistence_recovery`
13. `figure_composition_public_contract`
14. `figure_role_reference_public_contract`

The `unified_plugin_bootstrap` case loads two typed `PluginRegistration` values
from the installed package through the single `feedbax.plugins` group and adds a
fixture-owned typed registry family without loader changes. It proves
deterministic dependency sorting, atomic failure and conflict non-publication,
isolated cached contexts, typed errors, sealed publication, provenance, and
fail-closed legacy registrar values. The `dynamic_component_ports` case proves an
external policy-bearing component through unified bootstrap, deterministic
materialization, GraphSpec build, runtime execution, and fail-closed mismatch.
The `custody_persistence_recovery` case proves a primary ENOSPC persistence
failure survives restart, publishes the bounded v1 emergency recovery record,
blocks destructive teardown until custody is complete, and permits exactly one
post-custody deletion. The `figure_composition_public_contract` case proves the
installed public figure composition and display contract, including the
`feedbax-figure resolve` CLI. The `figure_role_reference_public_contract` case
proves the installed public row index, row-set selector, and figure role
reference contract, including digest-pinned expansion and fail-closed selector
rejection.

Every earlier shipped version rejects rather than acquiring synthetic evidence.
V2 remains the protected exact six-case foundation and rejects rather than
pretending it measured later cases. Unshipped v3-v6 results reject explicitly;
shipped v7 lacks the array-value case, shipped v8 lacks the unified-bootstrap
case, shipped v9 lacks the dynamic-port case, and shipped v10 lacks the external
driver case. Shipped v11 lacks both `custody_persistence_recovery` and bound
numeric protocol roles, shipped v12 lacks
`figure_composition_public_contract`, and shipped v13 lacks
`figure_role_reference_public_contract`.

The versioned `policy_manifest.v2.json` maps each policy row's exact public API,
schema behavior, structural and command obligations, and externally exercised
cases to real case IDs. V1 lacked those authorities and is rejected rather than
completed from prose. V2 marks the three envelope-layer rows (`report-surface`,
`evaluation-surface`, `analysis-authoring`) as non-external-covered rather than
inventing a case for them. Nested terminal certification stays in the training
lifecycle contract with focused in-repo migration and rejection evidence.

The required `current` and `minimum` protocol role members are both the strict
numeric value `1`. Version 1 retains its v2 normalization-before-rejection
behavior. Unknown, ambiguous, versionless, and future versions reject.
