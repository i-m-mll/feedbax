# Feedbax Schema Namespace Taxonomy

Feedbax schema-bearing payloads use stable identities because manifests,
provider exports, Studio workspaces, and downstream analysis bundles need a
durable contract. New Feedbax-owned identities must not use ad hoc flat
`feedbax.*` names.

## Governed Namespaces

- `feedbax.spec.*`: request/spec payloads and reusable nested specs. This
  includes graph specs, training/task/objective specs, evaluation/analysis/report
  run specs, Studio-authored specs, provider input request specs, and reusable
  nested specs embedded in those payloads.
- `feedbax.manifest.*`: durable execution records, artifact records, manifest
  wrappers, provider capability records, registry snapshots, migration records,
  and manifest-owned payload containers.
- `feedbax.component.<component>.params`: globally named component parameter
  payload schemas when a component owner intentionally exports one.

Component parameter versions such as `"1"` remain component-local by default.
They are governed by `ComponentRegistry` and `ComponentMigration`, not by
`SpecSchemaRegistry`. A component parameter payload only receives a global
schema identity when it is exported as a reusable schema; at that point it must
use `feedbax.component.<component>.params`.

## Registry Boundary

`SpecSchemaRegistry` owns emitted schema families that Feedbax exposes through
provider, manifest, Studio, execution, or analysis surfaces. Each registered
family declares a stable schema identity, a current schema version, and an
explicit migration or rejection policy for old versions.

`MigrationRegistry` remains the low-level directed-edge graph for schema
version migrations. `ComponentRegistry` owns component type IDs and component
parameter schema-version migration. Component parameter migrations must not be
registered as generic structured spec families unless the component owner also
exports a reusable global parameter payload schema identity.

## Existing Family Audit

- Graph, training, task, objective, evaluation, analysis, report, Studio, and
  execution request families are aligned under `feedbax.spec.*`.
- `PopulationStructureSpec` is the first reusable nested-spec example. It is
  governed as `feedbax.spec.population_structure` because it is persisted inside
  GraphSpec node params and consumed by multiple network-lowering paths. Exact
  index arrays are the durable executable state; population counts, assignment
  policy, and seed remain authoring metadata unless lowered before persistence.
- Manifest models, `SpecPayload`, array-store payloads, produced training-run
  retention artifacts, provider records, registry snapshots, execution plans,
  execution results, and artifact records are aligned under
  `feedbax.manifest.*`.
- Legacy flat `feedbax.population_structure.v1` payloads are rejected with an
  explicit unsupported-version policy instead of migrated.
- Graph-authored retained-observable request specs remain
  `feedbax.spec.graph.retained_observable`; produced `retention_plan` and
  `retained_observables` artifacts use distinct `feedbax.manifest.training.*`
  identities because they are execution outputs, not request specs.
- Legacy `feedbax.studio.task_bindings.v1` remains a registered migration input
  to `feedbax.spec.studio.task_bindings.v2`.
- Legacy `feedbax.studio.task_bindings.v0` and `feedbax.objective.v0` remain
  explicit rejection inputs with clear unsupported-version diagnostics.
- Flat test-only or future identities such as `feedbax.demo` are rejected at
  `SpecSchemaRegistry.register_family`.

Downstream RLRMP analysis sidecar families should register their own scientific
sidecar identities after this taxonomy, rather than placing those semantics in
Feedbax.
