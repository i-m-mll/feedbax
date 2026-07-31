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

The result schema is `feedbax.external_conformance.result.v8`. Its evidence map
has exactly eight strict-boolean cases, including the narrow resolver-handle
evaluation-row projection and public component-param array declarations. V2
remains the protected exact six-case foundation and rejects rather than
pretending it measured either later case. Unshipped v3-v6 results reject
explicitly; shipped v7 also rejects because it lacks the array-value case.
The `current` and `minimum` protocol role slots remain required and None-only
until owner ratification. Version 1 can normalize to v2 only when the later role
field is absent, then rejects for the same missing current evidence. Unknown,
ambiguous, versionless, and future versions reject.
