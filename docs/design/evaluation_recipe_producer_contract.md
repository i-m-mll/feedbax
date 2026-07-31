# Evaluation Recipe Producer Contract

Feedbax evaluation recipes are explicit producer contracts for downstream
packages that create evaluation states for later analysis and reporting.

## P1 Contract

1. Evaluation producers register lowercase dotted `evaluation_type` keys of the
   form `<package>.<name>`. Feedbax-owned recipes use `feedbax.*`; downstream
   packages use their import-name prefix, for example `rlrmp.*`.
2. A conforming producer registers a params schema family for
   `<package>.spec.evaluation.<name>`. Feedbax v1 conformance allows an
   explicit waiver for un-schema'd params; strict executor rejection is a later
   ratchet.
3. Recipes that return states declare
   `EvaluationRecipeResult.metadata["states_schema"]` as an opaque non-empty
   string identifying the states pytree shape.
4. Recipes are deterministic for the same `EvaluationRunSpec`, avoid durable
   writes inside the recipe, do not depend on ambient CWD or global config, and
   access model/artifact inputs only through resolved `ParentRef` inputs.

## Conformance

Downstream CI can import `feedbax.testing.evaluation_contract` and call
`check_evaluation_recipe(evaluation_type, spec_factory, evaluation_registry=registry)`
with the caller-owned `EvaluationRecipeRegistry` containing the recipe. The helper
validates the namespaced key and callable shape, runs a
completed manifest through the executor cache, confirms `states_schema`, checks
the params schema family or explicit waiver, and verifies that a recipe failure
writes a failed manifest.

The helper does not implement the P2 executor params-validation hook or
downstream schema namespace taxonomy. Those are separate contract ratchets.
