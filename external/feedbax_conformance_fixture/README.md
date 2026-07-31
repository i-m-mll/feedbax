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

The result schema is `feedbax.external_conformance.result.v12`. Its evidence map
has exactly twelve strict-boolean cases. The `unified_plugin_bootstrap` case loads
two typed `PluginRegistration` values from the installed package through the
single `feedbax.plugins` group and adds a fixture-owned typed registry family
without loader changes. It proves deterministic dependency sorting, atomic
failure and conflict non-publication, isolated cached contexts, typed errors,
sealed publication, provenance, and fail-closed legacy registrar values. V2
remains the protected exact six-case foundation and rejects rather than
pretending it measured later cases. Unshipped v3-v6 results reject explicitly;
shipped v7 lacks the array-value case, shipped v8 lacks the unified-bootstrap
case, shipped v9 lacks the dynamic-port case, and shipped v10 lacks the external
driver case, so all reject rather than acquiring synthetic evidence. Shipped
v11 lacks both `custody_persistence_recovery` and bound numeric protocol roles,
so it also rejects without synthetic migration. The
`dynamic_component_ports` case proves an
external policy-bearing component through unified bootstrap, deterministic
materialization, GraphSpec build, runtime execution, and fail-closed mismatch.
The `custody_persistence_recovery` case proves a primary ENOSPC persistence
failure survives restart, publishes the bounded v1 emergency recovery record,
blocks destructive teardown until custody is complete, and permits exactly one
post-custody deletion. The versioned `policy_manifest.v1.json` maps each
externally exercised policy row to real case IDs and marks terminal
certification as non-external-covered rather than inventing a case.

The required `current` and `minimum` protocol role members are both the strict
numeric value `1`. Version 1 retains its v2 normalization-before-rejection
behavior. Unknown, ambiguous, versionless, and future versions reject.
