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

The current result is deliberately `blocked`, not `pass`: Feedbax issue
`7e7dac8` owns the non-Git wheel-provenance seam required by orchestration
preflight and launch. The fixture asserts the current fail-closed behavior as a
negative canary. Once that issue integrates, the fixture can add the production
`StageEngine`/`LocalOrchestrationDriver` recovery run without changing its
package or result schema.

The result schema is `feedbax.external_conformance.result.v2`. Version 1 results
migrate by adding explicit, unbound `current` and `minimum` protocol role slots.
Unknown, versionless, and future versions are rejected.
