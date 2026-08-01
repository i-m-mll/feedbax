"""The generic envelope engine kernel, exercised over an invented fake project.

Every case here is about *mechanism*: canonical hashing, budgets, the compile
lock's plan/receipt boundary, fail-closed loading, lineage resolution, compile
orchestration, and the choke point. The science is entirely ``quillon``'s, and
``quillon`` is made up, which is the point — a test that needed a real project's
vocabulary would be testing the wrong layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from feedbax.contracts.authoring_budget import (
    AUTHORING_BUDGET_MIGRATION_TABLE,
    AUTHORING_BUDGET_SCHEMA_ID,
    AuthoringBudgets,
    load_authoring_budget_document,
)
from feedbax.contracts.experiment_compile_lock import (
    EXPERIMENT_COMPILE_LOCK_MIGRATION_TABLE,
    EXPERIMENT_COMPILE_LOCK_SCHEMA_ID,
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION,
    RUN_RECEIPT_ONLY_FACTS,
    CompileLockInputs,
    CompilerContract,
    CompilerImplementation,
    PlanReceiptBoundaryError,
    build_compile_lock,
    check_plan_receipt_boundary,
    load_compile_lock,
)
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
)
from feedbax.contracts.migrations import default_spec_registry
from feedbax.envelope import (
    CANONICAL_PIN_ALGORITHM,
    ChokeFinding,
    Lineage,
    PinnedDocument,
    build_lineage,
    canonical_sha256,
    compare_tracked_outputs,
    emit_text,
    read_authored_document,
)
from feedbax.envelope.compile import check_no_co_created_protected_document

from tests.fake_project_extension import PROJECT_DECLARATION
from tests.fake_project_extension.kernel import (
    DIGEST_FAMILY,
    QUILLON_LAYOUT,
    SURVEY_FAMILY,
    load_quillon_budgets,
    quillon_kernel,
    write_envelope,
    write_json,
    write_repo,
)


@pytest.fixture
def budgets() -> AuthoringBudgets:
    return load_quillon_budgets()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    write_repo(tmp_path)
    return tmp_path


# -- canonical form ------------------------------------------------------


def test_canonical_hash_is_stable_across_key_order_and_instances() -> None:
    left = {"b": [1, {"y": 2, "x": 3}], "a": "one"}
    right = {"a": "one", "b": [1, {"x": 3, "y": 2}]}

    assert canonical_sha256(left) == canonical_sha256(right)
    assert canonical_sha256(left) == canonical_sha256(json.loads(json.dumps(left)))
    assert len(canonical_sha256(left)) == 64


def test_canonical_hash_separates_values_that_differ_only_in_type() -> None:
    assert canonical_sha256({"v": 1}) != canonical_sha256({"v": True})
    assert canonical_sha256({"v": 1}) != canonical_sha256({"v": "1"})


def test_emit_text_is_the_deterministic_tracked_form() -> None:
    document = {"b": 1, "a": 2}

    emitted = emit_text(document)

    assert emitted.endswith("\n")
    assert not emitted.endswith("\n\n")
    assert emit_text(document) == emitted
    assert json.loads(emitted) == document


# -- budgets --------------------------------------------------------------


def test_budget_states_one_section_per_declared_layer(budgets: AuthoringBudgets) -> None:
    assert set(budgets.layers) == set(PROJECT_DECLARATION.labels("layer"))
    assert budgets.budget_id == PROJECT_DECLARATION.authoring_budget.resource_id


def test_widest_caps_are_the_maximum_any_layer_states(budgets: AuthoringBudgets) -> None:
    widest = budgets.widest

    assert widest.max_scalar_bytes == max(
        budgets.for_layer(layer).max_scalar_bytes for layer in budgets.layers
    )
    assert widest.max_assertions == max(
        budgets.for_layer(layer).max_assertions for layer in budgets.layers
    )


def test_per_layer_cap_refuses_the_layer_that_states_the_tighter_bound(
    budgets: AuthoringBudgets,
) -> None:
    prose = "x" * (budgets.for_layer("survey").max_scalar_bytes + 1)
    document = {
        "schema": "quillon.study.v1",
        "name": "loud",
        "layer": "survey",
        "body": {"settings": {"note": prose}},
    }
    raw = (json.dumps(document, indent=2) + "\n").encode("utf-8")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        read_authored_document(
            raw, budgets, field="studies/loud.envelope.json", layer_of=_layer_of
        )

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED
    assert "survey layer's authored budget" in str(excinfo.value)


def test_the_same_content_is_admitted_under_the_layer_with_the_wider_cap(
    budgets: AuthoringBudgets,
) -> None:
    prose = "x" * (budgets.for_layer("survey").max_scalar_bytes + 1)
    document = {
        "schema": "quillon.study.v1",
        "name": "loud",
        "layer": "digest",
        "body": {"summary": prose},
    }
    raw = (json.dumps(document, indent=2) + "\n").encode("utf-8")

    parsed = read_authored_document(
        raw, budgets, field="studies/loud.envelope.json", layer_of=_layer_of
    )

    assert parsed["body"]["summary"] == prose


def test_project_caps_are_validated_but_left_to_the_project(budgets: AuthoringBudgets) -> None:
    assert budgets.for_layer("survey").project_cap("max_probes") == 6
    with pytest.raises(KeyError):
        budgets.for_layer("digest").project_cap("max_probes")


def test_budget_document_refuses_a_section_with_a_mistyped_cap() -> None:
    document = _budget_document()
    document["layers"]["survey"]["max_lnies"] = document["layers"]["survey"].pop("max_lines")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        AuthoringBudgets.from_document(
            document, field="budget.json", declared_layers=("survey", "digest")
        )

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE


def test_budget_document_refuses_a_nonpositive_cap() -> None:
    document = _budget_document()
    document["layers"]["survey"]["max_depth"] = 0

    with pytest.raises(ExperimentEnvelopeRejection):
        AuthoringBudgets.from_document(
            document, field="budget.json", declared_layers=("survey", "digest")
        )


def test_budget_document_refuses_a_layer_the_project_does_not_declare() -> None:
    document = _budget_document()
    document["layers"]["ghost"] = dict(document["layers"]["survey"])

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        AuthoringBudgets.from_document(
            document, field="budget.json", declared_layers=("survey", "digest")
        )

    assert "one section per declared layer" in str(excinfo.value)


def test_budget_loader_rejects_an_unsupported_version_naming_its_migration_table() -> None:
    document = _budget_document()
    document["schema_version"] = f"{AUTHORING_BUDGET_SCHEMA_ID}.v9"
    raw = json.dumps(document).encode("utf-8")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        load_authoring_budget_document(raw, field="budget.json")

    message = str(excinfo.value)
    assert excinfo.value.category is (
        ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION
    )
    assert str(AUTHORING_BUDGET_MIGRATION_TABLE) in message
    assert "migration_intentionally_absent=yes" in message


# -- pre-parse guards ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"\xef\xbb\xbf{}", "byte-order mark"),
        (b'{"a": 1}\t\n', "carriage returns and tabs"),
        (b'{"a": 1}\n\n', "at most one newline"),
        (b'{"a": 1}   \n', "trailing whitespace"),
        (b'{"a": NaN}\n', "non-finite"),
        (b'{"a": 1, "a": 2}\n', "authored twice"),
        (b'{"a": }\n', "invalid JSON"),
    ],
)
def test_noncanonical_bytes_are_refused_before_parsing(
    budgets: AuthoringBudgets, raw: bytes, reason: str
) -> None:
    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        read_authored_document(raw, budgets, field="doc.json")

    assert reason in str(excinfo.value)
    assert excinfo.value.category in {
        ExperimentEnvelopeRejectionCategory.NONCANONICAL_FORMAT,
        ExperimentEnvelopeRejectionCategory.DUPLICATE_KEY,
    }


# -- compile lock ----------------------------------------------------------


def test_lock_records_contract_and_implementation_provenance_apart() -> None:
    lock = _lock()

    assert lock["compiler_contract"] == {
        "contract_id": "quillon.compiler_contract",
        "contract_version": "quillon.compiler_contract.v1",
    }
    implementation = lock["compiler_implementation"]
    assert implementation["code_unit"] == "tests.test_envelope_engine_kernel"
    assert "feedbax" in implementation["package_versions"]
    assert "contract_version" not in implementation
    assert "code_unit" not in lock["compiler_contract"]


def test_contract_provenance_comes_from_the_project_declaration() -> None:
    contract = CompilerContract.from_declaration(PROJECT_DECLARATION)

    assert contract.contract_id == PROJECT_DECLARATION.compiler_contract_id
    assert contract.contract_version == PROJECT_DECLARATION.compiler_contract_version


def test_contract_version_must_extend_its_contract_id() -> None:
    with pytest.raises(Exception, match="does not extend"):
        CompilerContract("quillon.compiler_contract", "other.v1")


def test_an_uninstalled_package_records_none_rather_than_vanishing() -> None:
    record = CompilerImplementation(
        code_unit="probe", packages=("feedbax", "no-such-package-at-all")
    ).record()

    assert record["package_versions"]["no-such-package-at-all"] is None
    assert set(record["package_versions"]) == {"feedbax", "no-such-package-at-all"}


def test_lock_pins_the_envelope_and_the_compiled_document() -> None:
    envelope = {"schema": "quillon.study.v1", "name": "probe"}
    document = {"schema_id": SURVEY_FAMILY, "name": "probe"}

    lock = _lock(envelope=envelope, document=document)

    assert lock["envelope"]["envelope_hash"] == canonical_sha256(envelope)
    assert lock["compiled_document"]["content_hash"] == canonical_sha256(document)
    assert lock["envelope"]["pin_algorithm"] == CANONICAL_PIN_ALGORITHM
    assert lock["compiled_document"]["pin_algorithm"] == CANONICAL_PIN_ALGORITHM


def test_execution_identity_moves_with_an_identity_contribution() -> None:
    plain = _lock()
    contributed = _lock(identity_contributions={"subject": {"ref": "a"}})
    other = _lock(identity_contributions={"subject": {"ref": "b"}})

    assert plain["execution_identity"]["sha256"] != contributed["execution_identity"]["sha256"]
    assert contributed["execution_identity"]["sha256"] != other["execution_identity"]["sha256"]
    assert contributed["execution_identity"]["inputs"] == [
        "compiled_document.content_hash",
        "identity_contributions.subject",
    ]


def test_execution_identity_is_stable_for_identical_inputs() -> None:
    assert _lock()["execution_identity"] == _lock()["execution_identity"]


@pytest.mark.parametrize("fact", sorted(RUN_RECEIPT_ONLY_FACTS))
def test_plan_receipt_boundary_refuses_every_run_receipt_fact(fact: str) -> None:
    lock = _lock()
    lock[fact] = "produced by a run"

    with pytest.raises(PlanReceiptBoundaryError, match=fact):
        check_plan_receipt_boundary(lock)


def test_plan_receipt_boundary_refuses_a_receipt_fact_smuggled_into_identity() -> None:
    with pytest.raises(PlanReceiptBoundaryError, match="run_id"):
        _lock(identity_contributions={"run_id": "run-1"})


def test_a_built_lock_passes_its_own_boundary_check() -> None:
    check_plan_receipt_boundary(_lock(identity_contributions={"subject": {"ref": "a"}}))


def test_lock_loader_accepts_the_current_version_and_rechecks_the_boundary() -> None:
    lock = _lock()

    loaded = load_compile_lock(lock, field="compiled/probe.compile-lock.json")

    assert loaded["schema_version"] == EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION


def test_lock_loader_refuses_a_lock_edited_to_carry_a_receipt_fact() -> None:
    lock = _lock()
    lock["run_id"] = "run-1"

    with pytest.raises(PlanReceiptBoundaryError):
        load_compile_lock(lock, field="compiled/probe.compile-lock.json")


def test_lock_loader_rejects_an_unsupported_version_naming_its_migration_slot() -> None:
    lock = _lock()
    lock["schema_version"] = f"{EXPERIMENT_COMPILE_LOCK_SCHEMA_ID}.v0"

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        load_compile_lock(lock, field="compiled/probe.compile-lock.json")

    message = str(excinfo.value)
    assert excinfo.value.category is (
        ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION
    )
    assert str(EXPERIMENT_COMPILE_LOCK_MIGRATION_TABLE) in message
    assert "migration_intentionally_absent=yes" in message


def test_lock_loader_rejects_a_foreign_family() -> None:
    lock = _lock()
    lock["schema_id"] = "rival.compile_lock"

    with pytest.raises(ExperimentEnvelopeRejection, match="schema_id"):
        load_compile_lock(lock, field="compiled/probe.compile-lock.json")


def test_the_lock_migration_slot_exists_in_the_shared_spec_registry() -> None:
    family = default_spec_registry.resolve("ExperimentCompileLock")

    assert family.identity == EXPERIMENT_COMPILE_LOCK_SCHEMA_ID
    assert family.current_version == EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION
    assert family.policy is not None
    assert default_spec_registry.available_migrations("ExperimentCompileLock") == ()


# -- lineage resolution -----------------------------------------------------


def test_lineage_names_the_document_that_owns_an_inherited_value(repo: Path) -> None:
    write_json(
        repo / "bases" / "mid.survey.json",
        {
            "schema_id": SURVEY_FAMILY,
            "name": "mid",
            "base": {"ref": "bases/baseline.survey.json"},
            "settings": {"span": 6},
        },
    )
    pinned = PinnedDocument.of(
        "bases/mid.survey.json",
        json.loads((repo / "bases" / "mid.survey.json").read_text()),
    )

    lineage = build_lineage(repo, pinned)

    span = lineage.lookup("settings.span")
    cadence = lineage.lookup("settings.cadence")
    assert span is not None and span.value == 6
    assert span.owner_ref == "bases/mid.survey.json"
    assert cadence is not None and cadence.value == 2
    assert cadence.owner_ref == "bases/baseline.survey.json"


def test_lineage_pins_every_document_it_consulted(repo: Path) -> None:
    pinned = PinnedDocument.of(
        "bases/baseline.survey.json",
        json.loads((repo / "bases" / "baseline.survey.json").read_text()),
    )

    pins = build_lineage(repo, pinned).pins()

    assert [pin["ref"] for pin in pins] == ["bases/baseline.survey.json"]
    assert all(pin["pin_algorithm"] == CANONICAL_PIN_ALGORITHM for pin in pins)
    assert all(len(pin["content_hash"]) == 64 for pin in pins)


def test_lineage_resolves_a_value_bound_by_a_patch_list() -> None:
    pinned = PinnedDocument.of(
        "delta.json",
        {"patches": [{"op": "replace", "path": "settings.span", "value": 11}]},
    )

    found = Lineage((pinned,)).lookup("settings.span")

    assert found is not None and found.value == 11


def test_lineage_walk_is_cycle_safe(repo: Path) -> None:
    write_json(
        repo / "bases" / "loop_a.json",
        {"schema_id": SURVEY_FAMILY, "base": {"ref": "bases/loop_b.json"}},
    )
    write_json(
        repo / "bases" / "loop_b.json",
        {"schema_id": SURVEY_FAMILY, "base": {"ref": "bases/loop_a.json"}},
    )
    pinned = PinnedDocument.of(
        "bases/loop_a.json", json.loads((repo / "bases" / "loop_a.json").read_text())
    )

    lineage = build_lineage(repo, pinned)

    assert [document.ref for document in lineage.documents] == [
        "bases/loop_a.json",
        "bases/loop_b.json",
    ]


# -- compile orchestration ---------------------------------------------------


def test_a_two_layer_project_compiles_through_the_generic_kernel(repo: Path) -> None:
    kernel = quillon_kernel()

    survey = kernel.compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)
    digest = kernel.compile_envelope_file(_envelope(repo, "widened-digest"), repo_root=repo)

    assert survey.layer == "survey"
    assert survey.family == SURVEY_FAMILY
    assert survey.document["settings"] == {"span": 9, "cadence": 2, "damping": 0.5}
    assert survey.document["base"]["ref"] == "bases/baseline.survey.json"
    assert digest.layer == "digest"
    assert digest.family == DIGEST_FAMILY
    assert digest.document["subject"]["name"] == "widened"


def test_a_cross_layer_reference_pins_only_pre_run_facts(repo: Path) -> None:
    digest = quillon_kernel().compile_envelope_file(
        _envelope(repo, "widened-digest"), repo_root=repo
    )

    subject = digest.document["subject"]
    assert set(subject) == {"name", "envelope", "compiled_document"}
    assert subject["compiled_document"]["pin_algorithm"] == CANONICAL_PIN_ALGORITHM
    assert not RUN_RECEIPT_ONLY_FACTS & set(subject)
    check_plan_receipt_boundary(digest.compile_lock)


def test_compilation_is_deterministic(repo: Path) -> None:
    kernel = quillon_kernel()

    first = kernel.compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)
    second = kernel.compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)

    assert emit_text(first.document) == emit_text(second.document)
    assert first.compile_lock == second.compile_lock


def test_an_assertion_that_holds_is_recorded_with_its_owner(repo: Path) -> None:
    outcome = quillon_kernel().compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)

    assert outcome.compile_lock["assertions"] == [
        {
            "path": "settings.cadence",
            "expected": 2,
            "actual": 2,
            "owner_ref": "bases/baseline.survey.json",
        }
    ]


def test_an_assertion_that_fails_names_the_document_that_owns_the_value(repo: Path) -> None:
    _reauthor(repo, "widened", assert_=[{"path": "settings.cadence", "equals": 99}])

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        quillon_kernel().compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.ASSERTION_FAILED
    assert "bases/baseline.survey.json" in str(excinfo.value)


def test_an_assertion_may_not_guard_a_path_the_envelope_changes(repo: Path) -> None:
    _reauthor(
        repo,
        "widened",
        assert_=[{"path": "settings.span", "equals": 4}],
        body={"settings": {"span": 9}},
    )

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        quillon_kernel().compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is (
        ExperimentEnvelopeRejectionCategory.ILLEGAL_ASSERTION_PATH
    )


def test_an_assertion_on_an_uninherited_path_has_nothing_to_check(repo: Path) -> None:
    _reauthor(repo, "widened", assert_=[{"path": "settings.absent", "equals": 1}])

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        quillon_kernel().compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)

    assert "not inherited" in str(excinfo.value)


def test_an_echoed_inherited_value_is_refused(repo: Path) -> None:
    _reauthor(repo, "widened", assert_=[], body={"settings": {"cadence": 2}})

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        quillon_kernel().compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is (
        ExperimentEnvelopeRejectionCategory.ECHOED_INHERITED_VALUE
    )
    assert "bases/baseline.survey.json" in str(excinfo.value)


def test_an_echo_check_does_not_confuse_a_boolean_with_one(repo: Path) -> None:
    write_json(
        repo / "bases" / "flagged.survey.json",
        {"schema_id": SURVEY_FAMILY, "name": "flagged", "settings": {"span": 1}},
    )
    _reauthor(
        repo,
        "widened",
        assert_=[],
        base="bases/flagged.survey.json",
        body={"settings": {"span": True}},
    )

    outcome = quillon_kernel().compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)

    assert outcome.document["settings"]["span"] is True


def test_a_base_under_the_output_directory_is_refused(repo: Path) -> None:
    _reauthor(repo, "widened", assert_=[], base="compiled/widened.quillon.survey_document.json")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        quillon_kernel().compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE
    assert "compiled output" in str(excinfo.value)


def test_a_normalized_path_cannot_smuggle_in_a_compiled_base(repo: Path) -> None:
    _reauthor(
        repo, "widened", assert_=[], base="./compiled/../compiled/widened.survey_document.json"
    )

    with pytest.raises(ExperimentEnvelopeRejection, match="compiled output"):
        quillon_kernel().compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)


def test_a_cross_layer_base_is_refused(repo: Path) -> None:
    _reauthor(repo, "widened", assert_=[], base="bases/baseline.digest.json")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        quillon_kernel().compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.CROSS_FAMILY_BASE


def test_an_envelope_alias_parent_resolves_and_pins_its_compiled_bytes(repo: Path) -> None:
    write_envelope(
        repo / QUILLON_LAYOUT.envelope_directory / "narrowed.envelope.json",
        {
            "schema": "quillon.study.v1",
            "name": "narrowed",
            "layer": "survey",
            "base": "widened",
            "body": {"settings": {"span": 3}},
        },
    )
    kernel = quillon_kernel()

    outcome = kernel.compile_envelope_file(_envelope(repo, "narrowed"), repo_root=repo)
    parent = kernel.compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)

    base = outcome.compile_lock["base"]
    assert base["kind"] == "envelope_alias"
    assert base["ref"] == "studies/widened.envelope.json"
    assert base["content_hash"] == canonical_sha256(parent.document)


def test_an_envelope_alias_cycle_is_refused(repo: Path) -> None:
    for alias, base in (("ping", "pong"), ("pong", "ping")):
        write_envelope(
            repo / QUILLON_LAYOUT.envelope_directory / f"{alias}.envelope.json",
            {
                "schema": "quillon.study.v1",
                "name": alias,
                "layer": "survey",
                "base": base,
                "body": {"settings": {"span": 7}},
            },
        )

    with pytest.raises(ExperimentEnvelopeRejection, match="cycle"):
        quillon_kernel().compile_envelope_file(_envelope(repo, "ping"), repo_root=repo)


def test_a_retired_envelope_family_is_refused_naming_its_replacement(repo: Path) -> None:
    _reauthor(repo, "widened", assert_=[], schema="quillon.trial.v0")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        quillon_kernel().compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.RETIRED_BASE_FAMILY
    assert "quillon.study.v1" in str(excinfo.value)


def test_an_unclaimed_envelope_family_is_refused(repo: Path) -> None:
    _reauthor(repo, "widened", assert_=[], schema="rival.study.v1")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        quillon_kernel().compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is (
        ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION
    )


def test_a_compiled_document_that_lies_about_its_family_is_refused(repo: Path) -> None:
    write_json(
        repo / "bases" / "baseline.survey.json",
        {"schema_id": SURVEY_FAMILY, "name": "baseline", "settings": {"span": 4}},
    )
    kernel = quillon_kernel()
    kernel.hooks.validate_compiled(  # sanity: the hook accepts a truthful document
        {"schema_id": SURVEY_FAMILY}, SURVEY_FAMILY, "studies/widened.envelope.json"
    )

    with pytest.raises(ExperimentEnvelopeRejection, match="compiled as"):
        kernel.hooks.validate_compiled(
            {"schema_id": DIGEST_FAMILY}, SURVEY_FAMILY, "studies/widened.envelope.json"
        )


def test_an_over_budget_assertion_count_is_refused(repo: Path) -> None:
    budgets = load_quillon_budgets()
    _reauthor(
        repo,
        "widened",
        assert_=[
            {"path": "settings.cadence", "equals": 2}
            for _ in range(budgets.for_layer("survey").max_assertions + 1)
        ],
    )

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        quillon_kernel().compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED
    assert "assertions exceeds" in str(excinfo.value)


def test_write_outputs_is_byte_reproducible(repo: Path) -> None:
    kernel = quillon_kernel()
    outcome = kernel.compile_envelope_file(_envelope(repo, "widened"), repo_root=repo)
    out_dir = repo / QUILLON_LAYOUT.output_directory

    first = {
        path: path.read_bytes() for path in kernel.write_outputs(outcome, out_dir).values()
    }
    second = {
        path: path.read_bytes() for path in kernel.write_outputs(outcome, out_dir).values()
    }

    assert first == second
    assert set(first) == set(kernel.output_paths(outcome, out_dir).values())


# -- co-creation -------------------------------------------------------------


def test_a_co_created_protected_document_is_refused() -> None:
    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        check_no_co_created_protected_document(
            ["studies/widened.envelope.json", "bases/new.base.json"],
            "studies/widened.envelope.json",
            (".base.json",),
        )

    assert excinfo.value.category is (
        ExperimentEnvelopeRejectionCategory.CO_CREATED_PROTECTED_DOCUMENT
    )


def test_an_ordinary_authoring_change_is_admitted() -> None:
    check_no_co_created_protected_document(
        ["studies/widened.envelope.json"], "studies/widened.envelope.json", (".base.json",)
    )


# -- the choke point ----------------------------------------------------------


def test_identical_tracked_bytes_report_ok(repo: Path) -> None:
    kernel = quillon_kernel()
    _regenerate(kernel, repo)

    report = compare_tracked_outputs(kernel, repo)

    assert report.ok
    assert report.drift == ()
    assert len(report.by_finding(ChokeFinding.IDENTICAL)) == 4


def test_a_hand_edited_compiled_document_reports_structured_drift(repo: Path) -> None:
    kernel = quillon_kernel()
    _regenerate(kernel, repo)
    edited = repo / QUILLON_LAYOUT.output_directory / f"widened.{SURVEY_FAMILY}.json"
    edited.write_text(edited.read_text().replace('"span": 9', '"span": 10'), encoding="utf-8")

    report = compare_tracked_outputs(kernel, repo)

    assert not report.ok
    drift = report.by_finding(ChokeFinding.DIFFERS)
    assert [entry.path for entry in drift] == [
        f"{QUILLON_LAYOUT.output_directory}/widened.{SURVEY_FAMILY}.json"
    ]
    assert drift[0].envelope_ref == "studies/widened.envelope.json"


def test_an_untracked_output_reports_missing(repo: Path) -> None:
    kernel = quillon_kernel()
    _regenerate(kernel, repo)
    (repo / QUILLON_LAYOUT.output_directory / f"widened.{SURVEY_FAMILY}.json").unlink()

    report = compare_tracked_outputs(kernel, repo)

    assert [entry.finding for entry in report.drift] == [ChokeFinding.MISSING]


def test_a_compiled_document_no_envelope_produces_reports_orphaned(repo: Path) -> None:
    kernel = quillon_kernel()
    _regenerate(kernel, repo)
    write_json(repo / QUILLON_LAYOUT.output_directory / "stray.json", {"schema_id": "stray"})

    report = compare_tracked_outputs(kernel, repo)

    orphans = report.by_finding(ChokeFinding.ORPHANED)
    assert [entry.path for entry in orphans] == [
        f"{QUILLON_LAYOUT.output_directory}/stray.json"
    ]


def test_an_envelope_that_no_longer_compiles_is_a_finding_not_an_exception(repo: Path) -> None:
    """One broken envelope is reported, not raised, and takes its dependants with it.

    ``widened-digest`` names ``widened`` as an upstream reference, so breaking
    ``widened`` breaks both. Both arrive as findings on one report rather than as
    the first exception to escape, which is what lets a single pass state the
    whole tree's condition.
    """
    kernel = quillon_kernel()
    _regenerate(kernel, repo)
    _reauthor(repo, "widened", assert_=[{"path": "settings.cadence", "equals": 99}])

    report = compare_tracked_outputs(kernel, repo)

    rejected = report.by_finding(ChokeFinding.REJECTED)
    assert sorted(entry.envelope_ref or "" for entry in rejected) == [
        "studies/widened-digest.envelope.json",
        "studies/widened.envelope.json",
    ]
    assert all("no longer compiles" in (entry.detail or "") for entry in rejected)
    assert not report.ok
    assert report.describe()


# -- helpers -------------------------------------------------------------------


def _layer_of(document: Any) -> str | None:
    from tests.fake_project_extension.kernel import layer_of

    return layer_of(document)


def _budget_document() -> dict[str, Any]:
    resource = PROJECT_DECLARATION.authoring_budget
    return json.loads((resource.root / "quillon.authoring_budget.json").read_text())


def _lock(
    *,
    envelope: dict[str, Any] | None = None,
    document: dict[str, Any] | None = None,
    identity_contributions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_compile_lock(
        CompileLockInputs(
            envelope_ref="studies/probe.envelope.json",
            envelope_document=envelope or {"schema": "quillon.study.v1", "name": "probe"},
            envelope_schema="quillon.study.v1",
            name="probe",
            family=SURVEY_FAMILY,
            compiled_document=document or {"schema_id": SURVEY_FAMILY, "name": "probe"},
            contract=CompilerContract.from_declaration(PROJECT_DECLARATION),
            implementation=CompilerImplementation(code_unit="tests.test_envelope_engine_kernel"),
            identity_contributions=identity_contributions or {},
        )
    )


def _envelope(repo: Path, alias: str) -> Path:
    return repo / QUILLON_LAYOUT.envelope_directory / f"{alias}{QUILLON_LAYOUT.envelope_suffix}"


def _reauthor(repo: Path, alias: str, **changes: Any) -> None:
    path = _envelope(repo, alias)
    document = json.loads(path.read_text())
    if "assert_" in changes:
        document["assert"] = changes.pop("assert_")
    document.update(changes)
    write_envelope(path, document)


def _regenerate(kernel: Any, repo: Path) -> None:
    out_dir = repo / QUILLON_LAYOUT.output_directory
    for envelope_path in kernel.envelopes(repo):
        kernel.write_outputs(
            kernel.compile_envelope_file(envelope_path, repo_root=repo), out_dir
        )
