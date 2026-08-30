"""Authoritative experiment and campaign root contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from feedbax.compiler import (
    CAMPAIGN_DOCUMENT_SCHEMA_VERSION,
    EXPERIMENT_DOCUMENT_SCHEMA_VERSION,
    RESOLVED_EXPERIMENT_SCHEMA_VERSION,
    BoundedBranch,
    CampaignDocument,
    CampaignVariant,
    DeclarationRef,
    DocumentRoot,
    ExperimentDocument,
    GraphDocument,
    RequestedProduct,
    ResolvedExperiment,
    ScientificSeedDomain,
    compile_graph,
    resolve_experiment,
)
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.authored_canonical import canonical_sha256
from feedbax.contracts.graph import ComponentSpec, GraphSpec


def _declaration(kind: str) -> DeclarationRef:
    return DeclarationRef(
        kind=kind,
        type_id=f"tests.{kind}",
        schema_id=f"tests.spec.{kind}",
        schema_version=f"tests.spec.{kind}.v1",
        owner="tests",
    )


def _experiment() -> tuple[ExperimentDocument, object]:
    compiled = compile_graph(
        GraphDocument(
            graph=GraphSpec(
                nodes={
                    "source": ComponentSpec(
                        type="Constant",
                        params={"value": 1.0},
                        output_ports=["output"],
                    )
                },
                output_ports=["output"],
                output_bindings={"output": ("source", "output")},
            )
        ),
        ComponentRegistry(load_user_components=False),
    )
    document = ExperimentDocument(
        document_id="tests.experiment",
        graph=DocumentRoot(
            schema_id="feedbax.graph_document",
            schema_version="1",
            content_sha256=compiled.resolved.document_sha256,
        ),
        trial_source=_declaration("trial_source"),
        objective=_declaration("objective"),
        training_program=_declaration("training_program"),
        observation_policy=DocumentRoot(
            schema_id="tests.observation_policy",
            schema_version="1",
            content_sha256="1" * 64,
        ),
        scientific_seeds=(ScientificSeedDomain(domain="training", root_seed=7),),
    )
    return document, compiled.resolved


def test_experiment_resolves_to_one_content_pinned_identity() -> None:
    document, graph = _experiment()

    resolved = resolve_experiment(document, graph)

    assert resolved.graph == graph
    assert resolved.trial_source.kind == "trial_source"
    assert ResolvedExperiment.model_validate_json(resolved.model_dump_json()) == resolved
    assert resolve_experiment(document, graph) == resolved


def test_campaign_is_finite_and_names_every_branch_outcome() -> None:
    document, _ = _experiment()
    experiment_sha256 = canonical_sha256(document.model_dump(mode="json"))
    campaign = CampaignDocument(
        document_id="tests.campaign",
        variants=(
            CampaignVariant(
                variant_id="base",
                experiment=DocumentRoot(
                    schema_id="feedbax.experiment_document",
                    schema_version="1",
                    content_sha256=experiment_sha256,
                ),
            ),
        ),
        requested_products=(
            RequestedProduct(product_id="trained-model", operation_type_id="feedbax.train"),
        ),
        branches=(
            BoundedBranch(
                branch_id="quality-gate",
                predicate_ref="metric:validation_loss",
                outcomes=("continue", "stop"),
            ),
        ),
    )

    assert campaign.branches[0].outcomes == ("continue", "stop")
    with pytest.raises(ValidationError, match="at least two named outcomes"):
        BoundedBranch(
            branch_id="unbounded",
            predicate_ref="metric:anything",
            outcomes=("continue",),
        )


@pytest.mark.parametrize(
    ("factory", "current_version"),
    [
        (lambda: _experiment()[0], EXPERIMENT_DOCUMENT_SCHEMA_VERSION),
        (
            lambda: CampaignDocument(
                document_id="tests.campaign",
                variants=(
                    CampaignVariant(
                        variant_id="base",
                        experiment=DocumentRoot(
                            schema_id="feedbax.experiment_document",
                            schema_version="1",
                            content_sha256="2" * 64,
                        ),
                    ),
                ),
                requested_products=(
                    RequestedProduct(product_id="model", operation_type_id="feedbax.train"),
                ),
            ),
            CAMPAIGN_DOCUMENT_SCHEMA_VERSION,
        ),
        (
            lambda: resolve_experiment(*_experiment()),
            RESOLVED_EXPERIMENT_SCHEMA_VERSION,
        ),
    ],
)
def test_authoritative_root_versions_fail_closed(factory, current_version: str) -> None:
    value = factory()
    payload = value.model_dump(mode="json")
    payload["schema_version"] = "0"

    assert current_version == "1"
    with pytest.raises(ValidationError, match="migration_intentionally_absent=yes"):
        type(value).model_validate(payload)
