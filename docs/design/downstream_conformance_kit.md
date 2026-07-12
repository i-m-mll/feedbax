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
