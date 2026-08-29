# Scientific declaration boundary

This document records the downstream boundary introduced by Slice 3. It is the
current contract, not a compatibility map for removed APIs.

## One declaration, selected facets

`feedbax.declarations` owns neutral identity (`Declaration`), atomic catalog
composition (`DeclarationCatalog`), and layer-local facets. A declaration has a
kind, type identity, schema identity and version, capabilities, runtime protocol,
and owner. Compiler, runtime, authoring, Studio, serialization, operation, and
backend facts are attached as independent facets. Composition rejects duplicate
identities, missing declarations, duplicate layer facets, and schema-version
disagreement before mutating the catalog.

Components use `DeclaredComponent`. Its compiler, serialization, runtime,
authoring, Studio, and training facets are immutable. `ComponentDefinition` and
the generated TypeScript client remain projections of this one registry source.
The mutable `ComponentMeta` predecessor no longer exists.

Training extensions use `declare_training_program` and register exactly one
`DeclaredTrainingProgram` through `feedbax.plugins.TRAINING_PROGRAMS`. The
application composition root derives row-lowering and execution-preparation
registries from that declaration. Those registries are private application
views, not independently composable plugin families. The predecessor
`training_methods`, `row_lowerers`, and `execution_preparations` families are
absent and fail closed.

## Scientific protocol ownership

Resolved trial production satisfies `TrialSourceProtocol`; resolved objective
computation satisfies `ObjectiveProtocol`. They are distinct contracts. A task
object may still provide both for authoring convenience, and
`resolve_task_contracts` projects that object into the two explicit runtime
authorities. `OperationProtocol` and `BackendProtocol` similarly keep scientific
operation execution separate from capability realization.

## Durable schema disposition

This change does not alter the serialized `GraphSpec`, `ComponentDefinition`,
training-run, row-lowering-reference, or plugin-declaration schemas. It replaces
Python registration APIs and their runtime composition only. Existing durable
schema migrations and explicit unsupported-version rejection therefore remain
the governing paths; no legacy inference or serialized compatibility shim is
introduced.

## Downstream adoption

The external conformance fixture is the executable downstream exemplar. It now
registers its component and training program through the declaration APIs and
proves authoring, row lowering, preparation, execution, and public policy
inventory without direct registration into derived private families.

The remaining rlrmp2 adoption surface is intentionally science-only:

- implement or reuse `TrialSourceProtocol` and `ObjectiveProtocol` separately;
- express SISU training through one `declare_training_program(...)` call;
- place its authoring hook, optional row compiler, preparation provider, and
  projections on that declaration;
- declare only `FamilyRequirement("training_programs")` and register through
  `TRAINING_PROGRAMS.register_program(...)`.

No rlrmp2 adapter registry, provider branch ladder, or Feedbax plumbing copy is
required.
