"""Skeleton cold-start conformance fixture: one fake project, one declaration.

This package stands in for a real downstream project while the engine is being
upstreamed. Its vocabulary is deliberately invented — ``quillon`` names nothing
and belongs to nobody — so the fixture proves the contract rather than any
particular science.

It exists to prove exactly what lane A1 locks: a project declares what it
extends as typed data through the single ``feedbax.plugins`` entry point,
registers it inside the ordinary transactional bootstrap with its registry
injected, and gets fail-closed label resolution before its compiler runs. The
LOC and shape gates that a full cold-start fixture must enforce land in later
lanes; this is the skeleton they grow from.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from feedbax.plugins import (
    EXPERIMENT_ENVELOPE_COMPILERS,
    PROJECT_EXTENSIONS,
    ApplicabilityRuleBinding,
    AuthoredLabelSite,
    AuthoringBudgetResource,
    EnvelopeFamilyDeclaration,
    EnvelopeFamilyStatus,
    ExperimentEnvelopeCompileRequest,
    ExperimentEnvelopeCompileResult,
    ExperimentEnvelopeCompilerRegistration,
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
    FamilyRequirement,
    LayerBinding,
    PluginDeclaration,
    PluginRegistration,
    ProjectExtensionDeclaration,
    ProjectExtensionPlugPoint,
)

PROJECT = "quillon"
ENVELOPE_SCHEMA_ID = "quillon.study"
ENVELOPE_SCHEMA = "quillon.study.v1"
RETIRED_ENVELOPE_SCHEMA = "quillon.trial.v0"
DECLARATION_SOURCE = "tests.fake_project_extension:PROJECT_DECLARATION"


class SurveyBlock:
    """The authored model of the survey layer, reduced to what the skeleton needs."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload


class DigestBlock:
    """The authored model of the digest layer."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload


def lower_survey(block: SurveyBlock) -> dict[str, Any]:
    """Lower one authored survey block into its output document."""
    return {"schema": "quillon.survey_document.v1", "payload": block.payload}


def lower_digest(block: DigestBlock) -> dict[str, Any]:
    """Lower one authored digest block into its output document."""
    return {"schema": "quillon.digest_document.v1", "payload": block.payload}


def applies_always(_envelope: Any) -> bool:
    """The one applicability rule the skeleton declares."""
    return True


PROJECT_DECLARATION = ProjectExtensionDeclaration(
    project=PROJECT,
    declaration_source=DECLARATION_SOURCE,
    compiler_contract_id="quillon.compiler_contract",
    compiler_contract_version="quillon.compiler_contract.v1",
    authoring_budget=AuthoringBudgetResource(
        resource_id="quillon.authoring_budgets.v1",
        root=resources.files(__name__) / "budgets",
    ),
    envelope_families=(
        EnvelopeFamilyDeclaration(ENVELOPE_SCHEMA_ID, ENVELOPE_SCHEMA),
        EnvelopeFamilyDeclaration(
            "quillon.trial",
            RETIRED_ENVELOPE_SCHEMA,
            status=EnvelopeFamilyStatus.RETIRED,
            superseded_by=ENVELOPE_SCHEMA,
        ),
    ),
    layers=(
        LayerBinding("survey", SurveyBlock, lower_survey, "quillon.survey_document"),
        LayerBinding("digest", DigestBlock, lower_digest, "quillon.digest_document"),
    ),
    applicability_rules=(ApplicabilityRuleBinding("quillon.rule.always.v1", "v1", applies_always),),
    label_sites=(
        AuthoredLabelSite(ProjectExtensionPlugPoint.LAYER, "layer"),
        AuthoredLabelSite(ProjectExtensionPlugPoint.APPLICABILITY_RULE, "applicability.rule"),
    ),
)


def compile_quillon_envelope(
    request: ExperimentEnvelopeCompileRequest,
) -> ExperimentEnvelopeCompileResult:
    """Compile one authored envelope through the layer binding its label names."""
    envelope = request.envelope
    name = envelope.get("name")
    if not isinstance(name, str) or not name:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD,
            "a quillon study envelope requires a nonempty name",
            field="name",
            correct_home="the envelope's identity block",
        )
    binding = PROJECT_DECLARATION.binding(ProjectExtensionPlugPoint.LAYER, envelope["layer"])
    assert isinstance(binding, LayerBinding)
    document = binding.lowerer(binding.authored_model(envelope.get("body")))
    request.out_dir.mkdir(parents=True, exist_ok=True)
    lock_name = f"{name}.compile-lock.json"
    document_name = f"{name}.{binding.output_family}.json"
    (request.out_dir / lock_name).write_text(
        json.dumps(
            {
                "compiler_contract": PROJECT_DECLARATION.compiler_contract_version,
                "envelope_schema": ENVELOPE_SCHEMA,
                "layer": binding.label,
                "name": name,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (request.out_dir / document_name).write_text(
        json.dumps(document, sort_keys=True), encoding="utf-8"
    )
    return ExperimentEnvelopeCompileResult(
        envelope_schema=ENVELOPE_SCHEMA,
        name=name,
        family=binding.output_family,
        compile_lock_path=lock_name,
        document_path=document_name,
    )


def _register(context) -> None:
    context.registry(PROJECT_EXTENSIONS).register(PROJECT_DECLARATION)
    context.registry(EXPERIMENT_ENVELOPE_COMPILERS).register(
        ExperimentEnvelopeCompilerRegistration(
            envelope_schema=ENVELOPE_SCHEMA,
            owner=DECLARATION_SOURCE,
            compile=compile_quillon_envelope,
        )
    )


PLUGIN_REGISTRATION = PluginRegistration(
    PluginDeclaration(
        "tests.fake_project_extension",
        "1.0",
        1,
        families=(
            FamilyRequirement("experiment_envelope_compilers"),
            FamilyRequirement("project_extensions"),
        ),
    ),
    _register,
)
