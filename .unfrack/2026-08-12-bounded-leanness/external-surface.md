# External surface and preservation boundary

Date: 2026-08-12

Baseline: `develop` at `9455af0a`

Accepted integration snapshot: `31d9628c1eb13c5314bb49f6a674e0a937b2df78`

Evidence owner: [issue:de0b456]

## What is formally guaranteed

**Verified current fact:** Feedbax's owner-ratified stable downstream contract is
`feedbax.downstream-interface-stability.v1`, current and minimum extension protocol
version 1, effective at Feedbax 0.2.0. Only the import paths, behavior, and durable
schemas enumerated in `docs/design/downstream_interface_stability.md` are guaranteed.
Unknown, removed, unsafe, or unsupported versions fail closed; durable format changes
require explicit migration or rejection evidence.

**Verified integrated fact:** the protected-tree snapshot contained stale ratification
and fixture-evidence wording even though its checker passed. Accepted [issue:c6d1181]
commits `1ecddedc` and `6fb88f3a` are integrated at `31d9628c`. They synchronize the
ratification record and separate policy-shape evidence from runtime-result evidence;
they do not change the guarantee rows, schema IDs, or public interfaces.

Formal guarantee and practical compatibility risk are different sets. A path outside
the table is not promised indefinitely, but known live production consumers make it
unsafe to delete or move that path in a bounded cleanup without coordinated migration.

## Current downstream evidence

The external survey inspected production source in rlrmp and rlrmp2 on 2026-08-12.
Its paths and counts remain survey-snapshot evidence; the accepted bounded code does
not turn those counts into current downstream guarantees. Re-run the inventory before
executing a later migration.

- **Verified current fact:** 250 of 396 rlrmp2 production `src` imports and 625 of
  882 rlrmp production `src` imports were outside the 28 formal policy namespaces.
- **Verified current fact:** frequently consumed non-guaranteed paths include
  `feedbax.contracts.worker`, `feedbax.contracts.training`,
  `feedbax.contracts.expressions`, `feedbax.contracts.checkpoints`, `analysis.*`,
  `training.*`, and `runtime.graph`.
- **Verified current fact:** rlrmp has a public-import manifest at
  `ci/feedbax-public-imports.toml`; the survey found no equivalent rlrmp2 ratchet.
- **Verified current fact:** downstream code consumes the orchestration environment
  names `FEEDBAX_ENV_FINGERPRINT`, `FEEDBAX_JAX_COMPILATION_CACHE_DIR`, `FEEDBAX_REF`,
  `FEEDBAX_ROW_DIR`, `FEEDBAX_ROW_ID`, `FEEDBAX_RUNS_DIR`,
  `FEEDBAX_RUN_EVENTS_DIR`, and `FEEDBAX_RUN_SET_ID`. Feedbax launchers emit or read
  the same names. [issue:77c5d59] owns the decision about registering them.
- **Verified current fact:** no rlrmp or rlrmp2 source, tests, scripts, or docs were
  found consuming Studio `/api/*` or `/ws/*` routes. This is absence of evidence in
  the surveyed repositories, not proof that no other external consumer exists.
- **Verified current fact:** downstream specs and instructions name
  `feedbax-analysis`, `feedbax-orchestrate`, and `python -m feedbax`; the seven console
  entry points are not dead aliases on the evidence reviewed.

## Invalidated cleanup scopes

### [issue:b853bc6] — deletion set

**Verified current fact:** the issue's zero-consumer premise is false for parts of the
proposed deletion set. Production consumers exist for `feedbax.analysis.eig`,
`feedbax.training.loss`, root lowering exports, and `intervention_state_indices` in
rlrmp or rlrmp2. The survey identified rlrmp2 fixed-point/report consumers and rlrmp
loss, science-lowering, and adaptive-epsilon-control consumers.

**Required disposition:** do not delete those surfaces under the present scope. The
isolated `feedbax/bin/qmd_convert.py`, `feedbax/bin/plotly_viewer.py`, `plotlyviewer`
extra, and associated base-dependency test were the only separately proposed bin
slice with no found downstream references or installed entry point. That smaller
slice still needs its own review and integration evidence.

### [issue:07c8428] — `contracts/graphs` package move

**Verified current fact:** the no-facade premise is false. rlrmp production imports
the current package from artifact migration and model construction paths.

**Required disposition:** retain the old import surface, or coordinate a downstream
migration with an explicit removal boundary. A package move without a facade or
synchronized consumer change is rejected for this bounded round.

### [issue:529c401] — `contracts/manifest.py` split

**Verified current fact:** rlrmp2 production has 64 import occurrences across 12
symbols and rlrmp has 66 across 22 symbols from the current manifest surface.

**Required disposition:** preserve current module paths through the split, or migrate
both downstreams in the same governed change. Internal elegance is not sufficient
evidence for removing the existing authority.

## Preservation rules for cleanup and refactoring

1. Search registered dynamic dispatch, rlrmp2, and rlrmp across Python, notebooks,
   and JSON/YAML specs before asserting that an import, component, schema, or entry
   point is unused.
2. Treat the formal stability table as the guaranteed minimum, not a complete live
   consumer inventory. Record known non-guaranteed consumers as migration inputs.
3. Preserve concrete schema classes, module locations, schema IDs, registry keys,
   JSON shapes, errors, and provenance when performing an internal extraction such as
   [issue:ce6f5b2], unless a versioned migration explicitly changes them.
4. Before a path-moving cleanup, add an rlrmp2 import-boundary manifest/test analogous
   to rlrmp's ratchet, then execute a coordinated migration or preserve a deliberate
   facade.
5. Do not infer API deletability from the absence of rlrmp/rlrmp2 route consumers.
   Studio routes remain backend/frontend product contracts and require their own
   consumer and version analysis.
6. Preserve command names and orchestration environment keys until their authority is
   either registered or deliberately migrated.
7. Re-run consumer evidence at the revision intended for integration; counts in this
   document are pinned to the 2026-08-12 survey.

## Identity domains

The external boundary must keep five identities separate:

| domain | authority | preservation question |
| --- | --- | --- |
| authored | user-authored document and authored hash | Did the authored bytes or declared schema change? |
| raw | stored/transferred file bytes and raw-file hash | Are the exact custody bytes unchanged? |
| canonical | algorithm-versioned canonical bytes and hash | Was canonicalization changed or merely re-run? |
| compiled | realized graph/model/program and compiled hash | Does the compiler produce the same realization? |
| runtime | execution receipt and observed environment | What revision, inputs, provider, and realization actually ran? |

A raw file may change while canonical meaning remains equal; a canonical document may
remain equal while a compiler changes; a compiled realization may match while the
runtime environment differs. None of those equivalences permits reusing another
domain's pin or receipt. In particular, compression bytes remain part of raw artifact
identity, so [issue:c41e12f]'s deliberate refusal to change gzip settings must not be
recast as unfinished cleanup.

## Recommended external gate

Before integrating a cleanup that touches any known consumer surface:

1. pin the Feedbax candidate revision and both downstream revisions;
2. generate import and durable-schema inventories for both downstreams;
3. classify every touched path as formally guaranteed, known practical dependency,
   dynamically registered, or proved unused;
4. provide old-version accept/migrate/reject evidence for durable formats;
5. compare authored, raw, canonical, compiled, and runtime identities independently;
6. run the applicable external conformance fixture and downstream focused tests; and
7. record whether compatibility is preserved, migrated, or intentionally broken at an
   owner-ratified boundary.

No external conformance fixture, downstream broad suite, protected auth, or protected
merge is claimed by this packet. The integration full suite and auth request remain
pending after the audit synthesis lands.

Co-Authored-By: Codex (GPT-5) <codex@openai.com>
