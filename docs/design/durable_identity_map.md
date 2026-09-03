# Durable identity domains

Feedbax uses several durable hashes, but equal-looking digest strings do not
necessarily mean the same thing. A semantic identity answers whether two
declared meanings are equal under a named projection. A material identity
answers whether the stored bytes are exactly equal. Every durable record must
name or imply the byte domain that produced its digest.

The source implementations and their typed contract tests are authoritative.
This document records the distinctions that a refactor must preserve; it is
not a generated inventory of every field or source location.

## JSON domains

- `canonical_json_v1` is the established semantic domain used by authored
  composition, compile locks, manifests, graph identities, checkpoint
  structure, and several orchestration records. Existing stored digests remain
  v1 unless the owning schema provides an explicit migration.
- `canonical_json_v2` is the strict cross-language JSON domain. It rejects
  non-JSON values, cycles, non-string keys, non-finite numbers, unsafe
  integers, and lone surrogates. Python and TypeScript share the conformance
  vectors in `conformance/canonical_json_v2.json`.
- Training-spec storage and publication records retain their separately named
  byte contracts. Similar JSON rules do not make their digests interchangeable
  with either canonical JSON version.
- Evaluation-state metadata, expression hashes, resolved-semantics hashes, and
  permissive legacy report hashes retain their owning schema domains. They may
  converge only through an explicit versioned migration.

## Material and mixed domains

- Blob, archive, manifest, lockfile, patch, array, and repository-snapshot
  digests identify exact bytes. JSON content does not turn one of these into a
  semantic JSON identity.
- Checkpoint identities deliberately mix exact leaf bytes with versioned
  structural JSON. Slot roots, transaction roots, structural ABI fingerprints,
  and run-contract projections remain distinct from the archived blob hashes
  they quote.
- Evaluation-state containers similarly distinguish array bytes, metadata
  bytes, tree structure, and final container bytes.
- Value identity has three explicit tiers: authored declaration, normalized
  numeric meaning, and runtime realization. A consumer must not substitute one
  tier for another.
- Implementation digests identify producing code and declared dependencies.
  They do not identify authored scientific intent or emitted artifact bytes.
- Repository-realization digests identify source and environment material.
  They do not identify experiment meaning.

## Carrier decisions

- Experiment compile-lock v4 pins envelope, base, lineage, content, compiled
  document, and execution identities to their declared algorithms. Its
  canonical JSON content remains v1 so existing and newly emitted locks remain
  offline-verifiable.
- Consistency-predicate v2 records migrate to v3 by pinning the unchanged
  digest to `canonical_json_v1`. Newly derived v3 records use
  `canonical_json_v2`. Unknown pins reject.
- Graph documents, resolved graphs, compilation records, and Studio semantic
  anchors share the graph document's SHA-256 identity. These are not Studio's
  lightweight draft hashes.
- Publication records join semantic protocol identities to exact blob
  references. A `BlobRef` always identifies bytes.
- Studio draft hashes are presentation-revision markers shared by Python and
  TypeScript. They are not authenticity proofs and cannot satisfy a SHA-256
  assertion.
- Legacy MD5 fields keep their declared contract until their owning schema is
  migrated. They are never silently reinterpreted as SHA-256.

## Change rule

When changing a durable hash, follow the producing function to its typed
carrier and downstream assertion. Preserve exact bytes, or version the owning
schema and provide a migrate-or-reject path. Do not infer equivalence from a
field name, digest length, or shared hashing primitive.

The rlrmp2 downstream project consumes canonical JSON, training-spec, compile
lock, checkpoint, and material identities. Feedbax changes to those domains
therefore require downstream compilation and locked-experiment verification
before protected delivery.
