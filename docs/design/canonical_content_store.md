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
publishes any logical manifest reference. The user must choose included
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

## Stable handoff APIs

Projection and RunPod work may depend on `ContentBinding`, `ContentTreeRef`,
`ContentObjectRef`, `ImmutableArtifactBlobProvider.get_tree`, `get_object`, and
`resolve_manifest`. It must not build another CAS or search another root.
Retention work may depend on the same refs and on the provider's canonical
object closure; lease, hold, quarantine, and deletion policy remain outside
this contract.

## Protected-merge surface admission

The candidate changes the five reviewed categories as follows:

- Durable schema families: adds the six content schemas above and advances
  manifest packet v1 to v2 with offline-only migration.
- Authoring grammar: no addition.
- Guaranteed public surface: proposes the content models, provider operations,
  migration functions, and `python -m feedbax migrate-content-store` command.
- Plugin or provider families: no new family; the fixed
  `ImmutableArtifactBlobProvider` is extended in place.
- Cloud lifecycle state machine: no change.

These additions cannot be represented truthfully by `ArtifactRef` or
staged-root v1: `ArtifactRef` couples object identity to logical presentation,
and staged-root v1 hashes binding name and root kind into content identity.
Owner ratification must approve the new durable schemas and public names for
the Feedbax 0.2.0 boundary, plus packet-v1 and staged-root-v1 runtime rejection.

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
