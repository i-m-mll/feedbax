"""The typed project extension declaration and its label-resolution contract.

A downstream project declares what it extends as typed data behind the single
``feedbax.plugins`` entry point. Nothing here scans packages, imports module
paths, or reaches a second registry: the declaration is registered inside the
ordinary transactional bootstrap with its registry injected, and every failure
it can produce is one of exactly two documented outcomes — infrastructure
failure (exit 1) or an authored rejection (exit 2) that lands before the
compiler runs and therefore before any output file exists.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from feedbax.__main__ import main
from feedbax.contracts.experiment_envelope import ExperimentEnvelopeRejectionCategory
from feedbax.contracts.project_extension import (
    PROJECT_EXTENSION_DECLARATION_SCHEMA_VERSION,
    ApplicabilityRuleBinding,
    AuthoredLabelSite,
    AuthoringBudgetResource,
    EnvelopeFamilyDeclaration,
    EnvelopeFamilyStatus,
    LayerBinding,
    ProjectExtensionCollisionError,
    ProjectExtensionDeclaration,
    ProjectExtensionDeclarationError,
    ProjectExtensionPlugPoint,
    ProjectExtensionRegistry,
    UnresolvedExtensionLabelRejection,
    resolve_authored_extension_labels,
)
from feedbax.plugins.composition import compose_application

import tests.fake_project_extension as fixture

PLUGIN_MODULE = "tests.fake_project_extension"
ENVELOPE_SCHEMA = fixture.ENVELOPE_SCHEMA


def _budget(tmp_path: Path) -> AuthoringBudgetResource:
    root = tmp_path / "budgets"
    root.mkdir(exist_ok=True)
    return AuthoringBudgetResource(resource_id="probe.budgets.v1", root=root)


def _declaration(tmp_path: Path, **overrides: Any) -> ProjectExtensionDeclaration:
    fields: dict[str, Any] = {
        "project": "probe",
        "declaration_source": "tests:probe",
        "compiler_contract_id": "probe.compiler_contract",
        "compiler_contract_version": "probe.compiler_contract.v1",
        "authoring_budget": _budget(tmp_path),
        "envelope_families": (EnvelopeFamilyDeclaration("probe.study", "probe.study.v1"),),
        "layers": (LayerBinding("survey", dict, lambda block: block, "probe.document"),),
        "label_sites": (AuthoredLabelSite(ProjectExtensionPlugPoint.LAYER, "layer"),),
    }
    fields.update(overrides)
    return ProjectExtensionDeclaration(**fields)


def _run(tmp_path: Path, envelope: dict[str, Any], *extra: str) -> int:
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return main(
        [
            "preflight-experiment-envelope",
            str(envelope_path),
            "--repo-root",
            str(tmp_path),
            "--plugin",
            PLUGIN_MODULE,
            *extra,
        ]
    )


# --- the declaration itself -------------------------------------------------


def test_declaration_carries_exactly_the_declared_contract(tmp_path: Path) -> None:
    declaration = fixture.PROJECT_DECLARATION

    assert declaration.schema_version == PROJECT_EXTENSION_DECLARATION_SCHEMA_VERSION
    assert declaration.project == "quillon"
    assert declaration.compiler_contract_version == "quillon.compiler_contract.v1"
    assert [item.schema_version for item in declaration.accepted_families] == [ENVELOPE_SCHEMA]
    assert [item.schema_version for item in declaration.retired_families] == [
        fixture.RETIRED_ENVELOPE_SCHEMA
    ]
    assert declaration.retired_families[0].superseded_by == ENVELOPE_SCHEMA
    assert declaration.labels(ProjectExtensionPlugPoint.LAYER) == ("digest", "survey")
    assert declaration.labels(ProjectExtensionPlugPoint.APPLICABILITY_RULE) == (
        "quillon.rule.always.v1",
    )
    assert declaration.authoring_budget.root.is_dir()
    assert declaration.accepts_envelope_schema(ENVELOPE_SCHEMA)
    assert not declaration.accepts_envelope_schema(fixture.RETIRED_ENVELOPE_SCHEMA)


def test_layer_binding_names_authored_model_lowerer_and_output_family() -> None:
    binding = fixture.PROJECT_DECLARATION.binding(ProjectExtensionPlugPoint.LAYER, "survey")

    assert isinstance(binding, LayerBinding)
    assert binding.authored_model is fixture.SurveyBlock
    assert binding.lowerer is fixture.lower_survey
    assert binding.output_family == "quillon.survey_document"


def test_declaration_refuses_a_duplicate_layer_label(tmp_path: Path) -> None:
    with pytest.raises(ProjectExtensionDeclarationError, match="duplicate layer label"):
        _declaration(
            tmp_path,
            layers=(
                LayerBinding("survey", dict, lambda block: block, "probe.document"),
                LayerBinding("survey", list, lambda block: block, "probe.other"),
            ),
        )


def test_declaration_refuses_a_duplicate_applicability_rule_id(tmp_path: Path) -> None:
    rule = ApplicabilityRuleBinding("probe.rule.one.v1", "v1", lambda envelope: True)

    with pytest.raises(ProjectExtensionDeclarationError, match="duplicate applicability rule id"):
        _declaration(tmp_path, applicability_rules=(rule, rule))


def test_declaration_refuses_a_label_site_with_no_bindings(tmp_path: Path) -> None:
    with pytest.raises(ProjectExtensionDeclarationError, match="declares no applicability_rule"):
        _declaration(
            tmp_path,
            label_sites=(
                AuthoredLabelSite(ProjectExtensionPlugPoint.LAYER, "layer"),
                AuthoredLabelSite(
                    ProjectExtensionPlugPoint.APPLICABILITY_RULE, "applicability.rule"
                ),
            ),
        )


def test_applicability_rule_id_must_carry_its_declared_version() -> None:
    with pytest.raises(ProjectExtensionDeclarationError, match="does not end with its declared"):
        ApplicabilityRuleBinding("probe.rule.one", "v1", lambda envelope: True)


def test_retired_family_must_name_its_replacement() -> None:
    with pytest.raises(ProjectExtensionDeclarationError, match="must name what replaced it"):
        EnvelopeFamilyDeclaration(
            "probe.trial", "probe.trial.v0", status=EnvelopeFamilyStatus.RETIRED
        )


def test_declaration_rejects_an_unsupported_schema_version(tmp_path: Path) -> None:
    with pytest.raises(ProjectExtensionDeclarationError, match="unsupported"):
        _declaration(
            tmp_path,
            schema_version="feedbax.plugin.project_extension_declaration.v2",
        )


# --- registry -------------------------------------------------------------


def test_registry_refuses_two_declarations_of_one_project(tmp_path: Path) -> None:
    registry = ProjectExtensionRegistry()
    registry.register(_declaration(tmp_path))

    with pytest.raises(ProjectExtensionCollisionError, match="already declared"):
        registry.register(
            _declaration(
                tmp_path,
                envelope_families=(EnvelopeFamilyDeclaration("probe.other", "probe.other.v1"),),
            )
        )
    assert registry.available_keys() == ("probe",)


def test_registry_refuses_two_projects_claiming_one_envelope_family(tmp_path: Path) -> None:
    registry = ProjectExtensionRegistry()
    registry.register(_declaration(tmp_path))

    with pytest.raises(ProjectExtensionCollisionError, match="already accepted"):
        registry.register(_declaration(tmp_path, project="probe2"))


def test_sealed_registry_refuses_late_registration(tmp_path: Path) -> None:
    registry = ProjectExtensionRegistry()
    registry.seal()

    with pytest.raises(RuntimeError, match="sealed"):
        registry.register(_declaration(tmp_path))


def test_registry_reports_an_unknown_project_with_what_it_knows(tmp_path: Path) -> None:
    registry = ProjectExtensionRegistry()
    registry.register(_declaration(tmp_path))

    with pytest.raises(ProjectExtensionDeclarationError, match="registered projects: probe"):
        registry.get("absent")


# --- label resolution ------------------------------------------------------


def test_resolution_binds_every_authored_label(tmp_path: Path) -> None:
    registry = ProjectExtensionRegistry()
    registry.register(fixture.PROJECT_DECLARATION)

    resolved = resolve_authored_extension_labels(
        {
            "schema": ENVELOPE_SCHEMA,
            "name": "probe-run",
            "layer": "digest",
            "applicability": {"rule": "quillon.rule.always.v1"},
        },
        registry,
    )

    assert [(item.plug_point.value, item.label) for item in resolved] == [
        ("layer", "digest"),
        ("applicability_rule", "quillon.rule.always.v1"),
    ]
    assert resolved[0].binding is fixture.PROJECT_DECLARATION.layers[1]


def test_an_unclaimed_envelope_family_has_no_labels_to_resolve(tmp_path: Path) -> None:
    registry = ProjectExtensionRegistry()
    registry.register(fixture.PROJECT_DECLARATION)

    assert resolve_authored_extension_labels({"schema": "other.study.v1"}, registry) == ()


def test_missing_layer_label_rejects_with_the_full_stable_field_set() -> None:
    registry = ProjectExtensionRegistry()
    registry.register(fixture.PROJECT_DECLARATION)

    with pytest.raises(UnresolvedExtensionLabelRejection) as caught:
        resolve_authored_extension_labels(
            {"schema": ENVELOPE_SCHEMA, "name": "probe-run", "layer": "absent"}, registry
        )

    rejection = caught.value
    assert rejection.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_EXTENSION_LABEL
    assert rejection.project == "quillon"
    assert rejection.family == ENVELOPE_SCHEMA
    assert rejection.field == "layer"
    assert rejection.label == "absent"
    assert rejection.plug_point is ProjectExtensionPlugPoint.LAYER
    assert rejection.available_labels == ("digest", "survey")
    assert rejection.declaration_source == fixture.DECLARATION_SOURCE
    assert "add the binding to the declaration" in rejection.repair


def test_missing_applicability_rule_rejects_at_its_own_plug_point() -> None:
    registry = ProjectExtensionRegistry()
    registry.register(fixture.PROJECT_DECLARATION)

    with pytest.raises(UnresolvedExtensionLabelRejection) as caught:
        resolve_authored_extension_labels(
            {
                "schema": ENVELOPE_SCHEMA,
                "name": "probe-run",
                "layer": "survey",
                "applicability": {"rule": "quillon.rule.absent.v1"},
            },
            registry,
        )

    assert caught.value.plug_point is ProjectExtensionPlugPoint.APPLICABILITY_RULE
    assert caught.value.field == "applicability.rule"
    assert caught.value.available_labels == ("quillon.rule.always.v1",)


# --- entrypoint exit semantics ---------------------------------------------


def test_declared_project_compiles_through_its_resolved_layer_binding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(
        tmp_path,
        {
            "schema": ENVELOPE_SCHEMA,
            "name": "quill-run",
            "layer": "survey",
            "applicability": {"rule": "quillon.rule.always.v1"},
            "body": {"trials": 3},
        },
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["family"] == "quillon.survey_document"
    document = json.loads(
        (tmp_path / "generated" / "quill-run.quillon.survey_document.json").read_text()
    )
    assert document == {"schema": "quillon.survey_document.v1", "payload": {"trials": 3}}


def test_unresolved_label_exits_two_before_any_output_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(
        tmp_path,
        {"schema": ENVELOPE_SCHEMA, "name": "quill-run", "layer": "absent"},
    )

    assert code == 2
    stderr = capsys.readouterr().err
    for fragment in (
        "category=unresolved-extension-label",
        "project=quillon",
        f"family={ENVELOPE_SCHEMA}",
        "plug_point=layer",
        "field=layer",
        "label=absent",
        "available_labels=digest, survey",
        f"declaration={fixture.DECLARATION_SOURCE}",
        "repair:",
    ):
        assert fragment in stderr
    assert not (tmp_path / "generated").exists()


def test_duplicate_labels_are_an_infrastructure_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _duplicating_register(context) -> None:
        context.registry(fixture.PROJECT_EXTENSIONS).register(
            ProjectExtensionDeclaration(
                project="quillon",
                declaration_source=fixture.DECLARATION_SOURCE,
                compiler_contract_id="quillon.compiler_contract",
                compiler_contract_version="quillon.compiler_contract.v1",
                authoring_budget=_budget(tmp_path),
                envelope_families=(
                    EnvelopeFamilyDeclaration(fixture.ENVELOPE_SCHEMA_ID, ENVELOPE_SCHEMA),
                ),
                layers=(
                    LayerBinding("survey", dict, lambda block: block, "quillon.survey_document"),
                    LayerBinding("survey", list, lambda block: block, "quillon.digest_document"),
                ),
                label_sites=(AuthoredLabelSite(ProjectExtensionPlugPoint.LAYER, "layer"),),
            )
        )

    monkeypatch.setattr(
        fixture,
        "PLUGIN_REGISTRATION",
        fixture.PluginRegistration(
            fixture.PluginDeclaration(
                "tests.fake_project_extension",
                "1.0",
                1,
                families=(fixture.FamilyRequirement("project_extensions"),),
            ),
            _duplicating_register,
        ),
    )

    with pytest.raises(Exception) as caught:
        _run(tmp_path, {"schema": ENVELOPE_SCHEMA, "name": "quill-run", "layer": "survey"})

    assert "duplicate layer label" in str(caught.value)
    assert not (tmp_path / "generated").exists()


def test_a_project_that_fails_to_load_is_an_infrastructure_failure(tmp_path: Path) -> None:
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_text(json.dumps({"schema": ENVELOPE_SCHEMA}), encoding="utf-8")

    with pytest.raises(Exception, match="failed to import module"):
        main(
            [
                "preflight-experiment-envelope",
                str(envelope_path),
                "--repo-root",
                str(tmp_path),
                "--plugin",
                "tests.fake_project_extension_absent",
            ]
        )


# --- bootstrap isolation ----------------------------------------------------


def test_double_bootstrap_produces_isolated_sealed_registries(tmp_path: Path) -> None:
    async def _compose():
        return await compose_application(modules=(PLUGIN_MODULE,), local_component_source=None)

    first = asyncio.run(_compose())
    second = asyncio.run(_compose())

    assert first.bundle.project_extensions is not second.bundle.project_extensions
    assert first.bundle.project_extensions.available_keys() == ("quillon",)
    assert second.bundle.project_extensions.available_keys() == ("quillon",)
    assert first.bundle.project_extensions.get("quillon") is second.bundle.project_extensions.get(
        "quillon"
    )
    with pytest.raises(RuntimeError, match="sealed"):
        first.bundle.project_extensions.register(_declaration(tmp_path))
    assert second.bundle.project_extensions.available_keys() == ("quillon",)


def test_bootstrap_records_the_declared_project_as_provenance(tmp_path: Path) -> None:
    state = asyncio.run(compose_application(modules=(PLUGIN_MODULE,), local_component_source=None))

    provenance = {item.plugin_id: item for item in state.provenance}[PLUGIN_MODULE]
    assert provenance.registered_keys["project_extensions"] == ("quillon",)
    assert provenance.family_protocols["project_extensions"] == "1"
