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
removes `PYTHONPATH`, denies network access during execution, and verifies that
all imported Feedbax modules come from the installed wheel.

The fixture runs the public production `StageEngine` with
`LocalOrchestrationDriver` through a clean-wheel PREFLIGHT and LAUNCH. It stops
at the persisted PREFLIGHT boundary, reconstructs the engine, and finishes the
bounded local lifecycle from the stored public state. The exact revision gate
therefore authenticates installed-wheel provenance at both execution
boundaries. The deployment policy is explicitly local and cloud-unauthorized;
all custody, orchestration, and cache paths are unique temporary directories.

The result schema is `feedbax.external_conformance.result.v2`. Version 1 results
migrate by adding explicit, unbound `current` and `minimum` protocol role slots.
Unknown, versionless, and future versions are rejected.
