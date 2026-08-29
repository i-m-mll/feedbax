"""Contracts for neutral scientific declarations and layer-local facets."""

from __future__ import annotations

import pytest

from feedbax.declarations import (
    BackendProtocol,
    DeclarationCatalog,
    DeclarationCompositionError,
    Facet,
    ObjectiveProtocol,
    OperationProtocol,
    ResolvedBackend,
    ResolvedObjective,
    ResolvedOperation,
    ResolvedTrialSource,
    RuntimeFacet,
    TrialSourceProtocol,
    facet,
    scientific_declaration,
)
from feedbax.training.environment import resolve_task_contracts


class TrialSource:
    def sample_trial(self, key, batch_info):
        return (key, batch_info)

    def episode_length(self, trial_spec):
        return len(trial_spec)


class Objective:
    def compute_loss(self, states, trial_spec, model):
        return states, trial_spec, model


class Operation:
    def execute(self, **inputs: object) -> object:
        return inputs


class Backend:
    def realize(self, capability: str, request: object) -> object:
        return capability, request


def _declaration(kind: str, protocol: type[object]):
    return scientific_declaration(
        kind=kind,
        type_id=f"tests.{kind}",
        schema_id=f"tests.spec.{kind}",
        schema_version=f"tests.spec.{kind}.v1",
        capabilities=("runtime",),
        runtime_protocol=protocol,
        owner="tests",
    )


def test_catalog_composes_only_requested_layer_facets() -> None:
    declaration = _declaration("operation", OperationProtocol)
    catalog = DeclarationCatalog()
    catalog.register(declaration, (facet(declaration, "runtime", RuntimeFacet(Operation())),))

    resolved, facets = catalog.compose("operation", "tests.operation", required_layers=("runtime",))
    assert resolved is declaration
    assert tuple(facets) == ("runtime",)
    assert catalog.facet("operation", "tests.operation", "studio", required=False) is None


def test_invalid_facet_compositions_fail_without_partial_registration() -> None:
    declaration = _declaration("objective", ObjectiveProtocol)
    catalog = DeclarationCatalog()
    wrong_version = Facet(
        kind="objective",
        type_id="tests.objective",
        layer="runtime",
        schema_version="tests.spec.objective.v2",
        value=RuntimeFacet(Objective()),
    )

    with pytest.raises(DeclarationCompositionError, match="uses schema version"):
        catalog.register(declaration, (wrong_version,))
    assert catalog.identities() == ()

    catalog.register(declaration)
    with pytest.raises(DeclarationCompositionError, match="missing required facets"):
        catalog.compose("objective", "tests.objective", required_layers=("runtime",))


def test_trial_and_objective_resolution_are_distinct_contracts() -> None:
    trial_declaration = _declaration("trial_source", TrialSourceProtocol)
    objective_declaration = _declaration("objective", ObjectiveProtocol)
    trials = TrialSource()
    objective = Objective()

    assert ResolvedTrialSource(trial_declaration, trials).source is trials
    assert ResolvedObjective(objective_declaration, objective).objective is objective
    with pytest.raises(TypeError, match="TrialSourceProtocol"):
        ResolvedTrialSource(trial_declaration, objective)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ObjectiveProtocol"):
        ResolvedObjective(objective_declaration, trials)  # type: ignore[arg-type]


def test_authoring_convenience_does_not_merge_runtime_protocols() -> None:
    class Task(TrialSource, Objective):
        pass

    task = Task()
    resolved = resolve_task_contracts(task)
    assert resolved.trials is task
    assert resolved.objective is task
    with pytest.raises(TypeError, match="ObjectiveProtocol"):
        resolve_task_contracts(TrialSource())


def test_operations_and_backends_resolve_against_their_own_protocols() -> None:
    operation = Operation()
    backend = Backend()
    assert (
        ResolvedOperation(_declaration("operation", OperationProtocol), operation).operation
        is operation
    )
    assert ResolvedBackend(_declaration("backend", BackendProtocol), backend).backend is backend
    with pytest.raises(TypeError, match="kind='backend'"):
        ResolvedBackend(_declaration("operation", OperationProtocol), backend)
