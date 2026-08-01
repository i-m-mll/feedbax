"""The data-only project declaration and how an envelope finds the project.

A project declares where its files live and nothing else. These tests hold that
line in both directions: the declaration must carry no callable and no dialect
knowledge, and everything the engine used to get from a callable must now come
from Feedbax itself. Registration is still the ordinary transactional bootstrap
with an injected registry, and the two documented outcomes are unchanged —
infrastructure failure (exit 1) or an authored rejection (exit 2) that lands
before any output file exists.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from feedbax.__main__ import main
from feedbax.contracts.experiment_envelope_dialect import (
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
)
from feedbax.contracts.project_experiment import (
    PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID,
    PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION,
    AuthoringBudgetResource,
    ProjectExperimentCollisionError,
    ProjectExperimentDeclaration,
    ProjectExperimentDeclarationError,
    ProjectExperimentRegistry,
)
from feedbax.plugins import (
    BootstrapError,
    BootstrapErrorCode,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
    bootstrap_application,
    new_registration_context,
)
from feedbax.plugins.composition import compose_application

import tests.fake_project_experiment as fixture

PLUGIN_MODULE = "tests.fake_project_experiment"


def _budget(tmp_path: Path) -> AuthoringBudgetResource:
    root = tmp_path / "budgets"
    root.mkdir(exist_ok=True)
    return AuthoringBudgetResource(
        resource_id="probe.budgets.v1", root=root, document_name="probe.budgets.json"
    )


def _declaration(tmp_path: Path, **overrides: Any) -> ProjectExperimentDeclaration:
    fields: dict[str, Any] = {
        "project": "probe",
        "declaration_source": "tests:probe",
        "envelope_directory": "probe-studies",
        "output_directory": "probe-compiled",
        "authoring_budget": _budget(tmp_path),
    }
    fields.update(overrides)
    return ProjectExperimentDeclaration(**fields)


# --- the declaration is data ------------------------------------------------


def test_the_declaration_is_six_data_fields_and_no_callables() -> None:
    declaration = fixture.PROJECT_DECLARATION

    assert declaration.schema_id == PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID
    assert declaration.schema_version == PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION
    assert declaration.project == "quillon"
    assert declaration.envelope_directory == "studies"
    assert declaration.output_directory == "compiled"
    assert declaration.authoring_budget.root.is_dir()
    assert {field.name for field in dataclasses.fields(declaration)} == {
        "project",
        "declaration_source",
        "envelope_directory",
        "output_directory",
        "authoring_budget",
        "schema_id",
        "schema_version",
    }
    for field in dataclasses.fields(declaration):
        value = getattr(declaration, field.name)
        assert not callable(value), f"{field.name} is a callable slot"


def test_the_declaration_states_nothing_about_the_dialect_or_the_compiler() -> None:
    declaration = fixture.PROJECT_DECLARATION

    for absent in (
        "compiler_contract_id",
        "compiler_contract_version",
        "envelope_families",
        "layers",
        "applicability_rules",
        "label_sites",
    ):
        assert not hasattr(declaration, absent), absent


def test_a_project_cannot_move_the_compiler_contract() -> None:
    """The contract is global, so there is no field through which to differ."""
    fields = {field.name for field in dataclasses.fields(fixture.PROJECT_DECLARATION)}

    assert "compiler_contract_version" not in fields
    assert EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION.startswith(
        "feedbax.experiment_envelope.compiler."
    )


@pytest.mark.parametrize(
    "directory", ["", "/absolute/studies", "../escape", "studies/", "./studies"]
)
def test_a_directory_must_be_a_normalized_repo_relative_path(
    tmp_path: Path, directory: str
) -> None:
    with pytest.raises(ProjectExperimentDeclarationError):
        _declaration(tmp_path, envelope_directory=directory)


def test_envelopes_and_compiled_output_may_not_share_a_home(tmp_path: Path) -> None:
    with pytest.raises(ProjectExperimentDeclarationError, match="cannot share a home"):
        _declaration(tmp_path, envelope_directory="one", output_directory="one")


def test_declaration_rejects_an_unsupported_schema_version(tmp_path: Path) -> None:
    with pytest.raises(ProjectExperimentDeclarationError, match="unsupported"):
        _declaration(tmp_path, schema_version="feedbax.project_experiment.v2")


def test_a_budget_resource_names_one_document_by_bare_filename(tmp_path: Path) -> None:
    root = tmp_path / "budgets"
    root.mkdir()

    with pytest.raises(ProjectExperimentDeclarationError, match="bare filename"):
        AuthoringBudgetResource(
            resource_id="probe.budgets.v1", root=root, document_name="nested/budget.json"
        )


# --- registry ---------------------------------------------------------------


def test_registry_refuses_two_declarations_of_one_project(tmp_path: Path) -> None:
    registry = ProjectExperimentRegistry()
    registry.register(_declaration(tmp_path))

    with pytest.raises(ProjectExperimentCollisionError, match="already declared"):
        registry.register(_declaration(tmp_path, envelope_directory="other-studies"))
    assert registry.available_keys() == ("probe",)


def test_registry_refuses_two_projects_claiming_one_envelope_directory(tmp_path: Path) -> None:
    registry = ProjectExperimentRegistry()
    registry.register(_declaration(tmp_path))

    with pytest.raises(ProjectExperimentCollisionError, match="already claimed"):
        registry.register(_declaration(tmp_path, project="probe2"))


def test_sealed_registry_refuses_late_registration(tmp_path: Path) -> None:
    registry = ProjectExperimentRegistry()
    registry.seal()

    with pytest.raises(RuntimeError, match="sealed"):
        registry.register(_declaration(tmp_path))


def test_registry_reports_an_unknown_project_with_what_it_knows(tmp_path: Path) -> None:
    registry = ProjectExperimentRegistry()
    registry.register(_declaration(tmp_path))

    with pytest.raises(ProjectExperimentDeclarationError, match="registered projects: probe"):
        registry.get("absent")


# --- resolving which project owns an envelope --------------------------------


def test_an_envelope_resolves_to_the_project_whose_directory_holds_it(tmp_path: Path) -> None:
    registry = ProjectExperimentRegistry()
    registry.register(_declaration(tmp_path))
    registry.register(
        _declaration(tmp_path, project="other", envelope_directory="other-studies")
    )

    assert registry.for_envelope_ref("probe-studies/x.envelope.json").project == "probe"
    assert registry.for_envelope_ref("other-studies/x.envelope.json").project == "other"


def test_an_envelope_outside_every_declared_directory_resolves_to_nothing(
    tmp_path: Path,
) -> None:
    registry = ProjectExperimentRegistry()
    registry.register(_declaration(tmp_path))

    with pytest.raises(ProjectExperimentDeclarationError, match="no declared envelope directory"):
        registry.for_envelope_ref("elsewhere/x.envelope.json")


def test_resolution_is_by_directory_and_never_by_anything_in_the_envelope(
    tmp_path: Path,
) -> None:
    """A document cannot choose the budgets it is judged under.

    The only input to resolution is the envelope's repo-relative path, so an
    authored field naming a project would be inert. That is what makes budgets
    and layout non-negotiable from inside the document.
    """
    registry = ProjectExperimentRegistry()
    registry.register(_declaration(tmp_path))

    assert registry.for_envelope_ref("probe-studies/./x.envelope.json").project == "probe"
    with pytest.raises(ProjectExperimentDeclarationError):
        registry.for_envelope_ref("probe-studies-rival/x.envelope.json")


# --- entrypoint exit semantics ------------------------------------------------


def _run(repo: Path, alias: str, *extra: str) -> int:
    return main(
        [
            "preflight-experiment-envelope",
            str(fixture.envelope_path(repo, alias)),
            "--repo-root",
            str(repo),
            "--plugin",
            PLUGIN_MODULE,
            *extra,
        ]
    )


def test_a_declared_project_compiles_through_the_feedbax_compiler(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture.write_repo(tmp_path)

    code = _run(tmp_path, "widened", "--out-dir", fixture.OUTPUT_DIRECTORY)

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["envelope_schema"] == EXPERIMENT_ENVELOPE_SCHEMA_VERSION
    assert payload["family"] == "training_run_matrix"
    document = json.loads(
        (tmp_path / fixture.OUTPUT_DIRECTORY / "widened.training_run_matrix.json").read_text()
    )
    assert document["schema_id"] == "feedbax.spec.training_run_matrix"
    lock = json.loads(
        (tmp_path / fixture.OUTPUT_DIRECTORY / "widened.compile-lock.json").read_text()
    )
    assert lock["compiler_contract"]["contract_version"] == (
        EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION
    )


def test_a_rejected_envelope_exits_two_before_any_output_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    envelope = dict(fixture.TRAINING_ENVELOPE)
    envelope["training"] = {**envelope["training"], "invented_field": True}
    fixture.write_repo(tmp_path, envelopes={"widened": envelope})

    code = _run(tmp_path, "widened", "--out-dir", fixture.OUTPUT_DIRECTORY)

    assert code == 2
    assert "category=unknown-field" in capsys.readouterr().err
    assert not (tmp_path / fixture.OUTPUT_DIRECTORY).exists()


def test_an_envelope_in_no_declared_directory_is_an_infrastructure_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture.write_repo(tmp_path)
    stray = tmp_path / "elsewhere" / "widened.envelope.json"
    stray.parent.mkdir(parents=True)
    stray.write_text(
        json.dumps(fixture.TRAINING_ENVELOPE, indent=2) + "\n", encoding="utf-8"
    )

    code = main(
        [
            "preflight-experiment-envelope",
            str(stray),
            "--repo-root",
            str(tmp_path),
            "--plugin",
            PLUGIN_MODULE,
        ]
    )

    assert code == 1
    assert "declaration of the project" in capsys.readouterr().err


def test_two_projects_claiming_one_directory_collide_during_bootstrap(tmp_path: Path) -> None:
    rival = PluginRegistration(
        PluginDeclaration(
            "tests.fake_project_experiment_rival",
            "1.0",
            1,
            families=(FamilyRequirement("project_experiments"),),
        ),
        lambda context: context.registry(fixture.PROJECT_EXPERIMENTS).register(
            ProjectExperimentDeclaration(
                project="quillon-rival",
                declaration_source="tests:rival",
                envelope_directory=fixture.ENVELOPE_DIRECTORY,
                output_directory="rival-compiled",
                authoring_budget=_budget(tmp_path),
            )
        ),
    )

    async def _compose() -> Any:
        return await bootstrap_application(
            new_registration_context(local_component_source=None),
            registrations=(fixture.PLUGIN_REGISTRATION, rival),
        )

    with pytest.raises(BootstrapError) as caught:
        asyncio.run(_compose())

    assert caught.value.code is BootstrapErrorCode.NAMESPACE_COLLISION
    assert "already claimed by project" in str(caught.value)


def test_a_project_that_fails_to_load_is_an_infrastructure_failure(tmp_path: Path) -> None:
    fixture.write_repo(tmp_path)

    with pytest.raises(BootstrapError) as caught:
        main(
            [
                "preflight-experiment-envelope",
                str(fixture.envelope_path(tmp_path, "widened")),
                "--repo-root",
                str(tmp_path),
                "--plugin",
                "tests.fake_project_experiment_absent",
            ]
        )

    assert caught.value.code is BootstrapErrorCode.LOAD
    assert not (tmp_path / "generated").exists()


# --- bootstrap isolation ------------------------------------------------------


def test_double_bootstrap_produces_isolated_sealed_registries(tmp_path: Path) -> None:
    async def _compose() -> Any:
        return await compose_application(modules=(PLUGIN_MODULE,), local_component_source=None)

    first = asyncio.run(_compose())
    second = asyncio.run(_compose())

    assert first.bundle.project_experiments is not second.bundle.project_experiments
    assert first.bundle.project_experiments.available_keys() == ("quillon",)
    assert second.bundle.project_experiments.available_keys() == ("quillon",)
    assert first.bundle.project_experiments.get(
        "quillon"
    ) is second.bundle.project_experiments.get("quillon")
    with pytest.raises(RuntimeError, match="sealed"):
        first.bundle.project_experiments.register(_declaration(tmp_path))
    assert second.bundle.project_experiments.available_keys() == ("quillon",)


def test_bootstrap_records_the_declared_project_as_provenance() -> None:
    state = asyncio.run(compose_application(modules=(PLUGIN_MODULE,), local_component_source=None))

    provenance = {item.plugin_id: item for item in state.provenance}[PLUGIN_MODULE]
    assert provenance.registered_keys["project_experiments"] == ("quillon",)
    assert provenance.family_protocols["project_experiments"] == "1"


def test_a_project_registers_no_envelope_compiler_of_its_own() -> None:
    state = asyncio.run(compose_application(modules=(PLUGIN_MODULE,), local_component_source=None))

    provenance = {item.plugin_id: item for item in state.provenance}[PLUGIN_MODULE]
    assert "experiment_envelope_compilers" not in provenance.registered_keys
    assert state.bundle.experiment_envelope_compilers.available_keys() == (
        EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
    )
