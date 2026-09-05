# Canonical content store and legacy migration

## Status and authority

This document defines the candidate successor storage contract implemented for
issue 942abb3. It is not owner-ratified downstream policy until its protected
delivery records the required surface-admission decision.

`ImmutableArtifactBlobProvider` is the only content-store authority. Existing
`ArtifactRef` reads and writes and the successor content APIs share its one
SHA-256 object namespace. A content reference never contains a machine path,
binding name, root kind, or execution role.

## Durable schemas

| Concept | Current schema | Prior-version rule |
|---|---|---|
| Exact object reference | `feedbax.spec.content_object_ref.v1` | v0 and unknown reject |
| Role-free tree descriptor | `feedbax.spec.content_tree.v1` | v0 and unknown reject |
| Pinned tree reference | `feedbax.spec.content_tree_ref.v1` | v0 and unknown reject |
| Manifest kind/ID reference | `feedbax.spec.content_manifest_ref.v1` | v0 and unknown reject |
| Runtime role binding | `feedbax.spec.content_binding.v1` | staged-root custody v1 migrates offline |
| Migration receipt | `feedbax.spec.content_migration_receipt.v1` | v0 and unknown reject |
| Manifest packet | `feedbax.spec.manifest_packet.v2` | packet v1 migrates offline; runtime rejects it |
| Retention root | `feedbax.spec.content_root_record.v1` | v0 and unknown reject |
| Explicit lease | `feedbax.spec.content_lease_record.v1` | v0 and unknown reject |
| Scientific/owner hold | `feedbax.spec.content_hold_record.v1` | v0 and unknown reject |
| Retention index | `feedbax.spec.content_retention_index.v1` | v0 and unknown reject |
| GC receipt | `feedbax.spec.content_gc_receipt.v1` | v0 and unknown reject |

Tree identity is the SHA-256 of canonical `feedbax.spec.content_tree.v1` descriptor
bytes. The descriptor contains only ordered directories and ordered
path-to-object records. Binding names, root kinds, store domains, and source
paths are excluded. Empty directories are retained. Paths must be canonical
POSIX NFC strings and must not collide after case-folded NFC normalization.
Symlinks, hard links, traversal, and non-regular members reject.

`(manifest kind, manifest ID)` maps to one exact object digest. Repeating the
same bytes is idempotent. Different bytes raise `ContentManifestConflictError`;
the store never overwrites, searches another root, or chooses an original over
an import. Manifest bytes are stored and returned exactly as supplied.

## Packet v2

Packet v2 has one `objects` row and one physical member for each unique digest.
Manifest and artifact entries are logical references to those objects, so many
logical references may share one object. Import validates the complete object
table, canonical paths, object sizes and digests, logical-key uniqueness,
manifest schemas, target-path collisions, and resident conflicts before it
publishes any logical manifest reference. Import then holds the content-store
transaction while it stores objects, atomically projects every manifest file,
rebuilds the filesystem index, and publishes logical references last. An
interruption can therefore leave only exact unreferenced objects or exact
manifest files; retry validates and completes them, while no logical reference
can name an incomplete projection. The user must choose included
(self-contained) or external (references-only) artifacts explicitly; the
former is the default command behavior and `auto` rejects.

## Offline migration

The only supported command is:

```text
python -m feedbax migrate-content-store \
  (--packet-v1 PACKET | --staged-root-v1 CUSTODY_JSON | \
   --run-bundle-v12 BUNDLE_JSON | --path-backed-json MAP_JSON) \
  [--source-root SNAPSHOT | --source-roots-json ROOTS_JSON] \
  --destination-store STORE [--receipt RECEIPT_JSON]
```

Staged-root v1 migration authenticates the supplied snapshot against its exact
legacy custody record, ingests each file, builds the role-free tree, and emits
the role and root kind only in a separate `ContentBinding`. Packet-v1 migration
authenticates each self-contained member and exact manifest before publishing
pinned references. Path-backed migration requires an explicit absolute source
for every provenance ref. Every mode emits a total source-to-target mapping
with source digest, target digest, and byte count. RunBundle v12 migration
requires a nested root-kind/binding-name map with every exact staged snapshot;
it cannot synthesize content identity from JSON alone. It never rewrites source or
manifest bytes and never deletes old custody.

Successor runtime entry points call `require_content_binding` and
`import_manifest_packet`; these reject staged-root v1 and packet v1 with the
exact offline command and required source arguments. Legacy decoding exists
only inside the explicit migration functions. After a successful migration,
resolution uses only store refs, so source paths may be removed from runtime
configuration or made unavailable.

## Retention and manual garbage collection

Retention is part of the same `ImmutableArtifactBlobProvider` authority and the
same object namespace. Callers explicitly register canonical logical, run,
export, import, or migration roots. A tree root retains its descriptor and every
object in its validated tree closure. Nonterminal or restartable run roots
cannot be released. Scientific and owner holds are separate named records and
also require explicit release.

`acquire_lease`, `renew_lease`, and `release_lease` serialize against GC marking
through the store lock. Lease expiry is diagnostic evidence only: an elapsed
lease that has not been explicitly released makes collection a no-op. Remote
attempt caches may hold leases through this same contract, but cannot register
roots or holds and never become authoritative content custody.

`collect_garbage()` is a manual dry run by default and does not write receipts,
indexes, or object state. A destructive call must supply the exact generation
from a clean dry-run receipt. It records the candidate generation, moves exact
objects into a generation-specific quarantine, records that stage, verifies the
complete quarantine set, and only then unlinks it. `recover_garbage_collection`
resumes either a marked or quarantined generation idempotently. Unknown schema
versions, corrupt indexes, missing live objects, unsafe filesystem entries,
identity or generation drift, and expired-but-unreleased leases all return an
error receipt and delete nothing.

No automatic collector, schedule, or default time-to-live exists. A caller owns
the decision to register and release roots, choose lease times, request a dry
run, and authorize one exact destructive generation.

## Stable handoff APIs

Projection and RunPod work may depend on `ContentBinding`, `ContentTreeRef`,
`ContentObjectRef`, `ImmutableArtifactBlobProvider.get_tree`, `get_object`, and
`resolve_manifest`. It must not build another CAS or search another root.
Orchestration may additionally depend on the provider's `acquire_lease`,
`renew_lease`, and `release_lease` methods. Root, hold, dry-run, destructive GC,
and recovery methods are store-administration surfaces; timing stays caller
policy.

## Execution projection

`RunAssemblyRequest` v8 and `RunBundle` v13 carry canonically ordered
`ContentBinding` rows and do not contain a `staged_roots` field. Path-free
v7/v12 documents migrate by removing an empty legacy field and adding an empty
binding list. A v7 request or v12 bundle carrying non-empty `staged_roots` must use the
explicit offline migration because a registry migration has no source snapshot
from which to authenticate and ingest the tree.

Input preflight resolves every pinned descriptor and object before acquisition
or output publication. The derived projection is ephemeral: it deduplicates
trees and objects by digest, materializes one shared verified view per tree,
and projects distinct runtime roles onto that view. RunPod keeps the same
object/tree layout only as a non-authoritative cache, verifies resident digests,
uploads each missing object once, and hard-links one attempt-local runtime view
from the verified cache. Transfer evidence reports unique bytes, uploaded bytes,
and canonical plan metadata bytes independently of role alias count.

Nonterminal projections require the canonical store root and an explicit lease
duration. Drivers call the
provider's durable `ContentLeaseRecord` API directly; no narrower adapter
contract exists. The CLI exposes these runtime-only values as
`--content-store-root` and `--content-lease-seconds`. Acquisition occurs once
after input validation and before publication or remote transfer, renewals use
the same durable owner and targets, and every staging failure releases the
lease. Persistent lease records, retention decisions, and garbage collection
remain owned by issue cb5ae8e.

Object, tree, manifest-reference, retention, and garbage-collection publication
all share one re-entrant process lock backed by the same OS file lock. Compound
tree and packet operations hold it across their full publication boundary, so
a GC generation cannot mark a half-published closure or miss a concurrently
published object and then sweep it.

## Protected-merge surface admission

The candidate changes the five reviewed categories as follows:

- Durable schema families: adds the six content schemas and five retention
  schemas above, advances manifest packet v1 to v2 with offline-only migration,
  and advances the existing
  `RunAssemblyRequest` v7 to v8 and `RunBundle` v12 to v13 families to carry
  typed content bindings. No input-projection schema family is added.
- Authoring grammar: no addition.
- Guaranteed public surface: proposes the content and retention models,
  provider operations, migration functions, and
  `python -m feedbax migrate-content-store` command.
- Plugin or provider families: no new family; the fixed
  `ImmutableArtifactBlobProvider` is extended in place.
- Cloud lifecycle state machine: `STAGE_INPUTS` now negotiates a verified,
  non-authoritative digest cache and holds it through explicit leases. This adds
  no provider state and chooses no acquisition, renewal, release, or GC timing.

These additions cannot be represented truthfully by `ArtifactRef` or
staged-root v1: `ArtifactRef` couples object identity to logical presentation,
and staged-root v1 hashes binding name and root kind into content identity.
Owner ratification must approve the new durable schemas and public names for
the Feedbax 0.2.0 boundary, plus packet-v1 and staged-root-v1 runtime rejection.
It must also approve the remote attempt-cache lease lifecycle addition; ordinary
merge approval is not sufficient for either surface increase.

The repo-owned clean-install fixture is the relevant Feedbax compatibility
corpus. Its focused `check_content_store_candidate_contract` case exercises the
candidate public imports without changing the owner-ratified v14 result schema
or claiming a new guaranteed row before approval. The rlrmp2 inventory cited by
issue 942abb3 (5,182 paths across 3,735
observed digest groups) is not available here as a custody-pinned artifact, so
this lane cannot claim corpus compatibility. Protected delivery must name and
pin the exact refreshed rlrmp2 revision or Mandible artifact version, or state
that no relevant corpus was supplied and obtain the owner's decision on that
evidence gap.
