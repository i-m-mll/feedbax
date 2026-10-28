# Test fixtures

This directory holds saved inputs and expected results used by repository tests.
`canonical_json_v2.json` is the shared serialization and hashing corpus consumed
by both the Python and TypeScript suites.

`downstream/` is an independently packaged consumer of Feedbax's public APIs.
`scripts/run_external_conformance.py` builds both packages as wheels, installs
them into a fresh environment, and runs the fixture outside the source checkout.
Its package and schema identities are independent of this directory's location.

Reusable Python builders, fake projects, and test plugins live in
`tests/support/`. The shipped `feedbax/testing/` package provides the public
downstream testing API.
