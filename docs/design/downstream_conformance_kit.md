# Downstream conformance kit

`feedbax.testing` supplies policy-free engines for downstream repositories that
test their use of Feedbax. Repository roots, scan domains, suite manifests,
allowlists, and pin files are always supplied by the caller. The kit does not
discover downstream files relative to its own installation.

## Modules and downstream bindings

- `suite` loads the version-one family manifest, restricts marked pytest runs to
  live files, rejects skips and non-strict xfails, collects marked node IDs in
  an explicitly rooted subprocess, and checks family floors and declared
  negative canaries. A downstream `conftest.py` only needs a
  `ContractSuiteHooks` instance and two bound hook aliases.
- `ast_scan` supplies the scope-aware `SiteVisitor`, frozen `StructuralSite`
  convention, deterministic file/tree/domain scanners, and the single shared
  expression/target rendering path. Detection policy stays downstream.
- `allowlist` supplies caller-keyed findings, required owner and rationale,
  `python_scope`/`file`/`glob` confinement, two-sided unlisted/dead diagnostics,
  and set-based ratchets backed by JSON or TOML lists.
- `version_pin` compares an explicit TOML revision pin with an editable
  package's HEAD and verifies that revision is reachable from a supplied remote
  ref. Wheels and other non-Git installs return a documented skip result; a
  named environment escape hatch can allow an unpublished revision with a loud
  warning. The package path or checkout root is supplied explicitly.

The current rlrmp gate families map onto these primitives as follows:

| rlrmp family | Feedbax kit binding | Downstream-owned policy |
| --- | --- | --- |
| Marker/manifest plugin and meta-tests | `ContractSuiteHooks`, `load_suite_manifest`, `collect_contract_nodeids`, family/canary assertions | Manifest data and marker name |
| Write-surface custody and analysis-write custody | `SiteVisitor`, scan helpers, `StructuralSite`, `diff_allowlist` | Write calls, durable/ephemeral classification, structural key extras |
| Retired component-ID scan | Scan helpers plus scoped allowlist entries | Python and non-Python retired-ID detectors |
| Reaccretion mini-scanners | Scan-domain helpers and shrink-only baseline operations | Naming heuristics and ratchet keys |
| Pipeline-native contract scan | Visitors plus keyed, scoped allowlist comparison | Writer/evaluation facts and reason payloads |
| Feedbax import boundary | AST/domain scanning plus allowlist comparison | Public packages, canonical homes, private-token policy |
| Feedbax revision pin | `check_version_pin` | Pin path, package name, remote ref, escape-hatch name |
| `data_in_code` JSON baseline | `JsonBaseline`, `compare_ratchet`, `write_shrink_only` | Data detectors and key serialization |

The kit deliberately does not include rlrmp's classifiers, naming lexicons,
thresholds, or concrete allowlist data. Those remain a page-scale registration
layer per gate family. It also uses true set-inclusion checks when updating a
ratchet, closing the same-cardinality replacement hole in the reference JSON
writer. Generic TOML baselines are read-only because rewriting TOML without a
format-preserving editor would destroy downstream comments; downstream code
edits the list explicitly after inspecting the ratchet diff.

## Clean-installed external fixture

`external/feedbax_conformance_fixture` is the repo-owned downstream-author
example and vertical extension of this kit. It is separately packaged and
keeps fixture-specific builders private to that package. The fixture reuses
`check_material_dependency_contract` rather than copying its admission
canaries, and exercises public ordered registration, component migration,
value identity, material dependencies, and exact-parent migration.

`uv run --no-sync python scripts/run_external_conformance.py` builds Feedbax
and the fixture as wheels, installs both non-editably into a fresh environment,
and executes away from this checkout with `PYTHONPATH` removed. The installed
fixture rejects private Feedbax imports, verifies all loaded Feedbax modules
come from the wheel, uses unique runtime cache and custody roots, and denies
runner-process outbound TCP connects before importing the Feedbax execution
stack. This is not a general DNS, UDP, or child-process sandbox; the bounded
lifecycle child is separately fixed and asserted as a print-only Python
command.

The versioned machine result is
`feedbax.external_conformance.result.v7`. V2 remains the frozen exact six-case
foundation and rejects because it contains no
`resolved_evaluation_row_projection` evidence. V1 can be deterministically
normalized to v2 by adding separate unbound `current` and `minimum` protocol
role slots, then rejects for the same missing evidence. Intervening unshipped
v3-v6 results reject explicitly; they are not reinterpreted as proof of the
narrowed v7 contract. Only the protected v2 and current v7 case sets are
maintained. The role slots remain None-only until the owner-ratified policy
exists.

The evaluation-row case uses only public clean-wheel imports. It resolves
durable states through `resolve_analysis_inputs`, receives an exact
`ResolvedEvaluationStateHandle` with a private issuance sentinel and canonical
immutable source/authority facts, derives manifest, run-spec, metadata, and
producer provenance from authenticated raw bytes, and invokes one downstream
cross-field callback. A cross-authority splice rejects with a stable error
code. The handle records provenance and materialization at resolver issuance;
it does not claim durable content authentication for cache or recompute state
bytes, nor protection from hostile same-process mutation after resolution.
Coverage, duplicate handling, conditioning, geometry, replicate policy, and
mixed-authority scientific verdicts remain downstream.

The production lifecycle case uses the installed-wheel
`StageEngine`/`LocalOrchestrationDriver` path with one deterministic local row.
It persists through PREFLIGHT, reconstructs the engine over the same
`RunSetStateStore`, and completes LAUNCH and the remaining local lifecycle.
Assertions prove that ASSEMBLE and PREFLIGHT were consumed from persisted
state rather than rerun, and that the exact revision gate observed the
authenticated wheel commit. The fixture-owned compiler and builders remain
incubated here; lifecycle sequencing, state recovery, subprocess execution,
and revision enforcement are all Feedbax production implementations.
