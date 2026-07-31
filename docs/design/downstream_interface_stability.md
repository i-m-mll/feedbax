# Downstream interface stability policy

| Field | Proposal |
|---|---|
| Policy identity | `feedbax.downstream-interface-stability.v1` |
| Status | Proposed; not owner-ratified |
| Adoption release | Feedbax `0.2.0` |
| Extension protocol | current `1`, minimum supported `1` |
| Decision owner | Feedbax owner |

This document is the ratification-ready proposal for issue `f4476ae`. It does
not itself make any API stable. Ratification is one yes/no decision on the
version window, the named guarantees, and the adoption delta below. Until that
decision and its follow-up change land, current source and tests are evidence,
not an owner guarantee.

## Proposed compatibility window

The extension protocol is an integer independent of individual durable schema
versions and the Python package version.

- At adoption, `current_protocol = 1`, `minimum_supported_protocol = 1`, and the
  minimum Feedbax release is `0.2.0`.
- After the first protocol increment, Feedbax supports exactly the current and
  immediately preceding protocol versions: for current version `N`,
  `minimum_supported_protocol = max(1, N - 1)`.
- A minimum-supported protocol remains accepted for at least 12 months after
  its successor is released. It may be removed only when both the two-version
  rule and the 12-month floor allow removal.
- Before Feedbax `1.0`, removing a supported protocol requires a minor release;
  at or after `1.0`, it requires a major release. The release notes must name
  the removed protocol, replacement, migration instructions, and first
  rejecting release.
- Removal additionally requires owner ratification, a deprecation issued in at
  least one earlier Feedbax release, a green external fixture for the remaining
  window, and an explicit rejection test for the removed version.

The initial window collapses to one version because no earlier protocol was
published. The fixture still records and exercises both the `current` and
`minimum` roles; they intentionally resolve to the same version until protocol
2 exists. Feedbax release compatibility follows the protocol declaration, not
an arbitrary range of package versions.

## What the guarantee means

For a symbol named below, Feedbax guarantees its import path and documented
behavior for every supported protocol version. It does **not** freeze its
source text, private helpers, implementation class hierarchy, complete Python
signature, or serialized representation unless those details are explicitly
part of a durable schema. Compatible changes may add optional parameters,
fields with defaults, methods, diagnostics, or accepted inputs.

A change is breaking when a clean external package that uses only the named
contract and declares a supported protocol version can no longer import,
register, validate, migrate, materialize, or execute as promised. Refactoring
behind the contract is not breaking.

Anything not listed is non-guaranteed. In particular, this policy does not
guarantee `feedbax.runtime.*` as a namespace. A separately documented public
runtime symbol can be promoted by a later policy version; an import merely
working today does not promote it.

## Proposed guaranteed surfaces

The following rows are proposed for protocol 1. “Current evidence” records what
exists at integration head `9efe9217`; “guarantee” is prospective and begins
only with ratification.

| Import namespace | Proposed public names | Guaranteed semantics | Current evidence |
|---|---|---|---|
| `feedbax.lowering` and the same names at `feedbax` | `LowererRegistration`, `LoweredContribution`, `LowererExecutionError`, `OrderedLowererRegistry` | Typed registration with stable `lowerer_id`, `owner`, and integer `order`; duplicate IDs reject; execution order is `(order, lowerer_id)`; inactive contributions return `None`; failures name lowerer and owner; the registry returns contributions without inventing merge policy. | `feedbax/lowering.py:LowererRegistration`, `feedbax/lowering.py:LoweredContribution`, `feedbax/lowering.py:LowererExecutionError`, `feedbax/lowering.py:OrderedLowererRegistry`; root aliases in `feedbax/__init__.py:_LAZY_EXPORTS` |
| `feedbax.component_registry` | `ComponentBuilder`, `ComponentMeta`, `ComponentResolution`, `ComponentRegistry`, `ComponentMigration`, `ComponentMigrationPack`, `register_component_type` | A downstream owner can register a namespaced component and parameter schema, contribute owner-matched deterministic migration edges, resolve an accepted or migrated component, and receive an actionable failure when an owner or migration is absent. Stable methods are `ComponentRegistry.register_component_type`, `register_migration`, `register_migration_pack`, and `resolve_component_spec`. | `feedbax/component_registry/registry.py:ComponentRegistry`, `ComponentResolution`, `register_component_type`; `feedbax/component_registry/meta.py:ComponentMeta`; `feedbax/contracts/migrations.py:ComponentMigration`, `ComponentMigrationPack`, `MissingComponentOwner`, `UnsupportedComponentMigration`; focused evidence in `tests/test_component_registration.py` |
| `feedbax.contracts.migrations` | `SchemaMigration`, `SpecSchemaFamily`, `SpecFamilyMigrationPolicy`, `SpecMigrationResult`, `SpecSchemaRegistry`, `UnknownSpecFamily`, `UnsupportedMigrationPath`, `UnsupportedSpecVersion`, `MissingComponentOwner`, `UnsupportedComponentMigration`, `default_spec_registry`, `migrate_structured_spec_payload` | A schema family has a stable identity and current version; old versions follow an explicit deterministic migration path or an explicit fail-closed rejection; versionless input rejects unless a caller deliberately requests `assume_current`; migration records preserve source and target versions. Component resolution failures name an absent owner or unsupported migration. | `feedbax/contracts/migrations.py:SpecSchemaRegistry.migrate`, `SchemaMigration.apply`, `migrate_structured_spec_payload`, `MissingComponentOwner`, `UnsupportedComponentMigration`; `tests/test_structured_spec_migrations.py:test_structured_spec_registry_accepts_current_version_without_migration`, `test_structured_spec_registry_applies_registered_family_migration`, `test_structured_spec_registry_rejects_explicit_unsupported_old_version` |
| `feedbax.contracts.graph` | `GRAPH_SPEC_SCHEMA_ID`, `GRAPH_SPEC_SCHEMA_VERSION`, `ComponentSpec`, `GraphProject`, `GraphSpec`, `ParamSchema`, `ParamValue`, `StudioValueSpec`, `WireSpec` | These are supported import locations for the durable GraphSpec family. Payload compatibility is governed by schema identity/version and migration policy, not by the in-memory model's source layout. Unknown versions fail with the current version and available migrations. | `feedbax/contracts/graph.py`; `tests/test_graphspec_schema_migrations.py:test_graph_spec_schema_identity_survives_json_round_trip`, `test_unknown_graph_spec_schema_version_reports_available_migrations` |

The policy deliberately does not yet guarantee the unified plugin bootstrap
symbols or registry-family protocols, provider registries, orchestration drivers, run kinds, custody-provider
factories, or Studio catalogs in protocol 1. Their lifecycle cells remain
partial or closed in `docs/design/extension_coverage.md`. A later seam may add
one only after its external conformance slice lands.

### Held dependent rows

These rows are mandatory inputs to the final ratification update, but their
names and schema versions are not invented here:

| Owner | Final row to insert | Required pins |
|---|---|---|
| `cd43b83` | Value encoding and identity contract | Exact public import namespace; authored, semantic, and realization identity types/functions; encoding declaration type; every affected schema ID; old/current schema versions; migrate/reject decision; canonical-byte semantics. |
| `43891d0` | Material-dependency admission and certification contract | Exact public import namespace; dependency declaration, factoring validator, admission result/error, and authored-waiver types/functions; terminal-status and certification schema IDs and versions; migration/rejection table. |

The parent must replace these placeholders from the integrated commits before
requesting owner ratification. If either child exposes no public symbol, the
row must say so and pin only its durable schema contract. A typed unified
registration context remains owned by `301dce2`; it is not implied by this
proposal or protocol 1.

## Durable state and the no-shim rule

“Backward compatibility is not a concern” continues to apply to internal and
unregistered Python helpers. They may move or disappear without aliases.
GraphSpec, Studio-persisted state, manifests, checkpoints, emitted specs, and
registered component/value identities are different: they are durable
Feedbax-owned formats.

For a durable format change, the owning change must do exactly one of:

1. preserve the existing schema semantics;
2. bump the schema version and provide a deterministic, tested migration; or
3. bump or retain the schema identity and explicitly reject the old version
   with an actionable error explaining why migration is intentionally absent.

Supported extension protocols may coexist through explicit version dispatch or
be converted at the single protocol boundary. Feedbax must not infer a version,
silently fall back to an old behavior, catch a rejection and retry another
path, or retain an unversioned compatibility shim. Deprecation warns on an
otherwise valid supported path; it never changes meaning silently.

## Pins: contract evidence versus accidents

Allowed semantic pins identify bytes or meaning that the contract deliberately
versions: schema IDs and versions, protocol versions, canonical authored or
semantic identities, artifact content hashes, exact-parent identities, and
golden hashes that test a documented canonicalization rule.

The following are not stability guarantees:

- hashes of source files, whole modules, generated source text, or incidental
  JSON formatting;
- string-keyed conventions hidden in `manifest.metadata`;
- imports from `feedbax.runtime.*` that are not separately listed here;
- downstream literals that encode Feedbax commit IDs or temporary blockers;
- golden hashes whose relationship to a documented semantic identity is
  unspecified.

Thus the ca09544 hazards are resolved by promotion or deletion, not prohibition:
a `manifest.metadata` key becomes a typed, versioned field or remains
non-contractual; a golden SHA-256 stays only when it pins documented canonical
identity; a runtime import moves to a named public namespace or remains
downstream risk; and a commit-ID literal is replaced by a protocol/schema
declaration. Implementation digests may remain non-authoritative diagnostics,
but do not stand in for semantic identity.

## Change, deprecation, and emergency procedure

A normal breaking proposal must:

1. open an issue naming affected protocol versions, imports, semantics, schemas,
   downstream fixtures, and migration or rejection behavior;
2. introduce the successor protocol without lowering the supported minimum;
3. update the policy identity, current/minimum constants, release floor,
   external fixture, per-seam coverage cells, and release notes in one governed
   integration;
4. emit a deprecation naming the replacement and removal eligibility date; and
5. remove the old protocol only after every removal gate above is green.

For a security, corruption, false-authentication, or unsafe-execution defect,
Feedbax may reject a nominally supported protocol immediately. The emergency
path must fail closed before side effects, name the rejected protocol and
reason, ship a focused negative test, publish a security/fix release, and
record owner ratification (retrospectively only when waiting would prolong the
unsafe condition). It may not silently reinterpret the input or route through
an older implementation.

## External conformance contract

Issue `380f897` owns a clean-wheel fixture package installed without the
Feedbax source tree on `PYTHONPATH`. Its stability-policy matrix must:

- build and install the candidate Feedbax wheel plus the external fixture wheel
  into a new environment;
- run one case declaring `current_protocol` and one declaring
  `minimum_supported_protocol`; at protocol 1 the two roles intentionally bind
  to the same numeric version but remain separate reported cases;
- import every name in the guaranteed-surface table only from its guaranteed
  namespace;
- register, migrate, materialize, validate, and execute the smallest external
  component and structured-spec examples;
- prove explicit rejection for `minimum_supported_protocol - 1` when that
  value is at least 1, and always prove rejection of an unknown future version;
- assert the policy identity, current/minimum values, candidate wheel version,
  and fixture version in the result; and
- contain no editable install, repo-relative import, source checkout, or
  private-module import.

Each seam child updates only its own lifecycle cells in
`docs/design/extension_coverage.md`, replaces provisional evidence with its
landed `file:symbol` and focused-test witnesses, and adds its case to the
external fixture in the same change. A cell becomes open only when the
clean-wheel case passes. The policy checker verifies that every guaranteed row
has a fixture case and every open stability/conformance cell cites one.

## Exact ratification delta

Ratification is a follow-up on this same issue after `cd43b83` and `43891d0`
integrate. It must make the following changes together; phase 1 makes none of
them.

### Mirrored instruction block

Replace the existing **Backward Compatibility** section in both `AGENTS.md` and
`CLAUDE.md` with this exact marked block:

```markdown
<!-- feedbax-downstream-stability:start -->
## Backward Compatibility and Downstream Stability

Internal and unregistered helpers are free to change; compatibility aliases and
silent legacy fallbacks are not maintained. The owner-ratified downstream
contract is `feedbax.downstream-interface-stability.v1` in
`docs/design/downstream_interface_stability.md`: extension protocol current
version `1`, minimum supported version `1`, minimum Feedbax release `0.2.0`.
Only the import paths, behavior, and durable schemas enumerated there are
guaranteed.

Supported protocol versions coexist or migrate at an explicit version boundary.
Unknown, removed, unsafe, or otherwise unsupported versions fail closed with an
actionable error; never infer a version or retry through a compatibility shim.
GraphSpec, Studio-persisted state, manifests, checkpoints, and emitted specs
remain governed by explicit schema identity plus tested migration or explicit
rejection. A breaking change follows the policy's deprecation, duration,
release, external-fixture, and owner-ratification gates.
<!-- feedbax-downstream-stability:end -->
```

### Checker, tests, and CI

1. Add `scripts/check_downstream_interface_policy.py`. It must compare the
   marked blocks byte-for-byte, parse this document's policy identity/current/
   minimum/release fields, verify the same values in both blocks, reject
   duplicate or missing markers, and require a fixture-case ID for every
   non-placeholder guaranteed row.
2. Add `tests/test_downstream_interface_policy.py` with import assertions for
   every named symbol, behavioral tests for current/minimum/future protocol
   admission, and a check that unlisted `feedbax.runtime.*` imports are not
   accidentally treated as policy entries. Do not snapshot signatures or
   source hashes.
3. Extend `scripts/check_instruction_policy.py` or invoke the new checker from
   it so local instruction validation covers both mirrored blocks.
4. In `.github/workflows/ci.yml`, add
   `uv run --no-sync python scripts/check_downstream_interface_policy.py` to
   the Python checks before contract tests. Add the `380f897` clean-wheel
   command as a required pull-request job; it must test both role labels and
   upload its machine-readable conformance result.
5. Pin `feedbax.downstream-interface-stability.v1`, protocol current/minimum,
   and the exact integrated `cd43b83`/`43891d0` schema versions in the external
   fixture manifest. Update only the stability/conformance cells owned by this
   issue and the integrated seam children.

The ratification change must also replace the two held rows above with exact
integrated names and versions. Any mismatch returns to proposal state; it is
not resolved with an alias, copied loader, or speculative compatibility layer.

## Ratification question

Approve or reject the proposal as one policy: identity
`feedbax.downstream-interface-stability.v1`; protocol 1 with the current-plus-
previous and 12-month window; Feedbax `0.2.0` adoption floor; only the named
public imports and semantics; explicit durable migrations or rejection; and the
held adoption delta completed after the identity and admission children land.
