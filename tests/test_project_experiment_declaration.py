"""The data-only project declaration and how a root is read as data.

A project declares where its files live and nothing else. These tests hold that
line in three directions. The declaration must carry no callable and no dialect
knowledge. Everything the engine used to get from a callable must now come from
Feedbax itself. And the declaration must arrive as *data* read from one stated
root, not as a Python object announced through the plugin bootstrap: there is no
``project_experiments`` family any more, so an empty project needs no importable
registration plumbing to be a project at all.

The two documented entrypoint outcomes are unchanged — infrastructure failure
(exit 1) or an authored rejection (exit 2) that lands before any output file
exists.
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
    EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS,
)
from feedbax.contracts.project_experiment import (
    PROJECT_DECLARATION_FILENAME,
    PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID,
    PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION,
    AuthoringBudgetResource,
    ProjectExperimentDeclaration,
    ProjectExperimentDeclarationError,
    load_project_declaration,
    parse_project_declaration,
    project_declaration_path,
)
import feedbax.plugins as plugins
from feedbax.plugins.composition import compose_application

import tests.fake_project_experiment as fixture


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


def _document(**overrides: Any) -> dict[str, Any]:
    document = dict(fixture.PROJECT_DECLARATION_DOCUMENT)
    for key, value in overrides.items():
        if value is None:
            document.pop(key, None)
        else:
            document[key] = value
    return document


def _write_declaration(root: Path, document: Any) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / PROJECT_DECLARATION_FILENAME
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


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


# --- a declaration is loaded from one stated root ---------------------------


def test_a_stated_root_loads_its_declaration_file(tmp_path: Path) -> None:
    fixture.write_declaration(tmp_path)

    declaration = load_project_declaration(tmp_path)

    assert declaration.project == fixture.PROJECT
    assert declaration.envelope_directory == fixture.ENVELOPE_DIRECTORY
    assert declaration.output_directory == fixture.OUTPUT_DIRECTORY
    assert declaration.declaration_source == str(project_declaration_path(tmp_path))
    assert declaration.authoring_budget.resource_id == fixture.BUDGET_REF
    budget = declaration.authoring_budget
    assert budget.root.joinpath(budget.document_name).read_bytes() == fixture.BUDGET_BYTES


def test_the_written_declaration_and_the_package_one_are_the_same_data(
    tmp_path: Path,
) -> None:
    """One document, two roots: the fixture cannot drift against itself."""
    fixture.write_declaration(tmp_path)

    loaded = load_project_declaration(tmp_path)

    for field in ("project", "envelope_directory", "output_directory", "schema_version"):
        assert getattr(loaded, field) == getattr(fixture.PROJECT_DECLARATION, field)
    assert (
        loaded.authoring_budget.resource_id
        == fixture.PROJECT_DECLARATION.authoring_budget.resource_id
    )


def test_the_declaration_file_name_is_fixed_and_never_searched_for(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    fixture.write_declaration(nested)

    assert project_declaration_path(nested).name == PROJECT_DECLARATION_FILENAME
    with pytest.raises(ProjectExperimentDeclarationError, match="no project declaration at"):
        load_project_declaration(tmp_path)


def test_a_root_that_declares_nothing_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ProjectExperimentDeclarationError, match=PROJECT_DECLARATION_FILENAME):
        load_project_declaration(tmp_path)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"{not json", "not a readable JSON"),
        (b"[]", "must be one JSON object"),
        (b"\xff\xfe", "not a readable JSON"),
    ],
)
def test_unreadable_declaration_bytes_fail_closed(tmp_path: Path, raw: bytes, match: str) -> None:
    (tmp_path / PROJECT_DECLARATION_FILENAME).write_bytes(raw)

    with pytest.raises(ProjectExperimentDeclarationError, match=match):
        load_project_declaration(tmp_path)


def test_an_unknown_schema_id_is_refused_rather_than_interpreted(tmp_path: Path) -> None:
    _write_declaration(tmp_path, _document(schema_id="someone_else.project"))

    with pytest.raises(ProjectExperimentDeclarationError, match="schema_id must be"):
        load_project_declaration(tmp_path)


@pytest.mark.parametrize(
    "version", ["feedbax.project_experiment.v0", "feedbax.project_experiment.v2", None]
)
def test_an_unsupported_schema_version_refuses_with_migration_absent(
    tmp_path: Path, version: str | None
) -> None:
    _write_declaration(tmp_path, _document(schema_version=version))

    with pytest.raises(
        ProjectExperimentDeclarationError, match="migration_intentionally_absent"
    ):
        load_project_declaration(tmp_path)


@pytest.mark.parametrize(
    "key", ["project", "envelope_directory", "output_directory", "authoring_budget"]
)
def test_an_omitted_key_is_incomplete_rather_than_defaulted(tmp_path: Path, key: str) -> None:
    _write_declaration(tmp_path, _document(**{key: None}))

    with pytest.raises(ProjectExperimentDeclarationError, match="omits required keys"):
        load_project_declaration(tmp_path)


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    _write_declaration(tmp_path, _document(base_directory="bases"))

    with pytest.raises(ProjectExperimentDeclarationError, match="does not define"):
        load_project_declaration(tmp_path)


@pytest.mark.parametrize("value", [42, "", "   ", None])
def test_a_non_string_field_is_refused(tmp_path: Path, value: Any) -> None:
    document = _document()
    document["project"] = value
    _write_declaration(tmp_path, document)

    with pytest.raises(ProjectExperimentDeclarationError, match="must be a nonempty string"):
        load_project_declaration(tmp_path)


@pytest.mark.parametrize(
    "budget_ref",
    ["/etc/budget.json", "../outside/budget.json", "budgets/", "budgets/./b.json"],
)
def test_a_budget_path_may_not_escape_or_wander(tmp_path: Path, budget_ref: str) -> None:
    _write_declaration(tmp_path, _document(authoring_budget=budget_ref))

    with pytest.raises(ProjectExperimentDeclarationError, match="authoring_budget"):
        load_project_declaration(tmp_path)


def test_a_budget_may_sit_at_the_project_root(tmp_path: Path) -> None:
    _write_declaration(tmp_path, _document(authoring_budget="budget.json"))

    declaration = load_project_declaration(tmp_path)

    assert declaration.authoring_budget.document_name == "budget.json"
    assert Path(str(declaration.authoring_budget.root)) == tmp_path


def test_parsing_records_the_stated_source_verbatim() -> None:
    declaration = parse_project_declaration(
        fixture.PROJECT_DECLARATION_BYTES,
        budget_root=Path("/nowhere"),
        source="an-explicit-source",
    )

    assert declaration.declaration_source == "an-explicit-source"


# --- declarations are no longer a plugin family -----------------------------


def test_the_bootstrap_has_no_project_declaration_family() -> None:
    state = asyncio.run(compose_application(modules=(), local_component_source=None))

    assert not hasattr(state.bundle, "project_experiments")
    families = {key.family for key in plugins.APPLICATION_REGISTRY_KEYS}
    assert "project_experiments" not in families


@pytest.mark.parametrize(
    "name",
    [
        "PROJECT_EXPERIMENTS",
        "ProjectExperimentRegistry",
        "ProjectExperimentCollisionError",
        "ProjectExperimentDeclaration",
    ],
)
def test_the_plugin_facade_no_longer_publishes_declaration_names(name: str) -> None:
    """A layout fact is not an implementation, so it is not part of the plugin API."""
    assert name not in plugins.__all__
    assert not hasattr(plugins, name)


def test_the_fixture_registers_no_plugin_for_its_declaration() -> None:
    assert not hasattr(fixture, "PLUGIN_REGISTRATION")
    assert "PLUGIN_REGISTRATION" not in fixture.__all__


# --- entrypoint exit semantics ------------------------------------------------


def _run(repo: Path, alias: str, *extra: str) -> int:
    return main(
        [
            "preflight-experiment-envelope",
            str(fixture.envelope_path(repo, alias)),
            "--repo-root",
            str(repo),
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
    # The version the compiled envelope declared: quillon's training envelope is
    # authored at v1 and is held to the v1 grammar, so the result reports v1.
    assert payload["envelope_schema"] == fixture.TRAINING_ENVELOPE["schema"]
    assert payload["envelope_schema"] in EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS
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


def test_an_envelope_outside_the_declared_directory_is_an_infrastructure_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture.write_repo(tmp_path)
    stray = tmp_path / "elsewhere" / "widened.envelope.json"
    stray.parent.mkdir(parents=True)
    stray.write_text(json.dumps(fixture.TRAINING_ENVELOPE, indent=2) + "\n", encoding="utf-8")

    code = main(
        ["preflight-experiment-envelope", str(stray), "--repo-root", str(tmp_path)]
    )

    assert code == 1
    assert "lies outside the envelope directory" in capsys.readouterr().err


def test_a_root_that_declares_nothing_reaches_the_compilers_own_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dispatch still happens, so a schema needing no declaration keeps working."""
    fixture.write_repo(tmp_path)
    project_declaration_path(tmp_path).unlink()

    code = _run(tmp_path, "widened", "--out-dir", fixture.OUTPUT_DIRECTORY)

    assert code == 1
    assert "needs the declaration of the project" in capsys.readouterr().err
    assert not (tmp_path / fixture.OUTPUT_DIRECTORY).exists()


def test_a_malformed_declaration_stops_before_any_compile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture.write_repo(tmp_path)
    _write_declaration(tmp_path, _document(schema_version="feedbax.project_experiment.v9"))

    code = _run(tmp_path, "widened", "--out-dir", fixture.OUTPUT_DIRECTORY)

    assert code == 1
    assert "cannot load the project declaration" in capsys.readouterr().err
    assert not (tmp_path / fixture.OUTPUT_DIRECTORY).exists()


# --- declared directories may have more than one segment --------------------


def test_a_nested_envelope_directory_owns_what_lies_under_it() -> None:
    """``_validate_directory`` accepts any normalized relative path, so ownership
    has to compare every segment. ``feedbax init`` writes ``specs/experiment`` by
    default, so a first-segment-only comparison would leave a freshly initialized
    project unable to compile any envelope it authored."""
    declaration = dataclasses.replace(
        fixture.PROJECT_DECLARATION,
        envelope_directory="specs/experiment",
        output_directory="generated",
    )

    assert declaration.owns_envelope_ref("specs/experiment/wide.envelope.json")
    assert declaration.owns_envelope_ref("specs/experiment/nested/wide.envelope.json")
    assert declaration.owns_envelope_ref("./specs/experiment/wide.envelope.json")
    assert not declaration.owns_envelope_ref("specs/base/tally.json")
    assert not declaration.owns_envelope_ref("specs.experiment/wide.envelope.json")
    assert not declaration.owns_envelope_ref("elsewhere/wide.envelope.json")


def test_a_nested_output_directory_still_refuses_a_compiled_base(tmp_path: Path) -> None:
    """The same segment-prefix rule guards the compiled-output base refusal."""
    from feedbax.contracts.experiment_envelope import ExperimentEnvelopeRejection
    from feedbax.envelope import kernel_for

    kernel = kernel_for(
        dataclasses.replace(
            fixture.PROJECT_DECLARATION,
            envelope_directory="specs/experiment",
            output_directory="build/generated",
        )
    )

    kernel.refuse_compiled_output_base("specs/base/tally.json", "base")
    kernel.refuse_compiled_output_base("build/other/tally.json", "base")
    with pytest.raises(ExperimentEnvelopeRejection, match="compiled output"):
        kernel.refuse_compiled_output_base("build/generated/wide.json", "base")
    with pytest.raises(ExperimentEnvelopeRejection, match="compiled output"):
        kernel.refuse_compiled_output_base("./build/generated/nested/wide.json", "base")
