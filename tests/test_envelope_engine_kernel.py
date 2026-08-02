"""The envelope engine kernel, exercised over an invented fake project.

Every case here is about *mechanism*: canonical hashing, budgets, the compile
lock's plan/receipt boundary, fail-closed loading, lineage resolution, the
five-layer compile, and the choke point. The science is entirely ``quillon``'s,
and ``quillon`` is made up, which is the point — a test that needed a real
project's vocabulary would be testing the wrong layer.
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
from feedbax.contracts.experiment_envelope_dialect import (
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_ID,
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
    ExperimentEnvelopeLayer,
)
from feedbax.contracts.migrations import default_spec_registry
from feedbax.envelope import (
    CANONICAL_PIN_ALGORITHM,
    ChokeFinding,
    Lineage,
    PinnedDocument,
    authored_layer_of,
    build_lineage,
    canonical_sha256,
    compare_tracked_outputs,
    emit_text,
    kernel_for,
    load_project_budgets,
    read_authored_document,
)
from feedbax.envelope.compile import (
    REPORT_BINDING_STATE_FIELDS,
    check_no_co_created_protected_document,
)
from feedbax.envelope.entrypoint import DECLARED_LAYERS

from tests.fake_project_experiment import (
    ENVELOPE_DIRECTORY,
    OUTPUT_DIRECTORY,
    PROJECT_DECLARATION,
    TRAINING_BASE,
    envelope_path,
    write_envelope,
    write_json,
    write_repo,
)

TRAINING_FAMILY = "training_run_matrix"
TRAINING_SCHEMA_ID = "feedbax.spec.training_run_matrix"


@pytest.fixture
def budgets() -> AuthoringBudgets:
    return load_project_budgets(PROJECT_DECLARATION)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    write_repo(tmp_path)
    return tmp_path


def kernel() -> Any:
    """Return the one compiler bound to the fake project's data declaration."""
    return kernel_for(PROJECT_DECLARATION)


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


def test_budget_states_one_section_per_dialect_layer(budgets: AuthoringBudgets) -> None:
    assert set(budgets.layers) == set(DECLARED_LAYERS)
    assert set(DECLARED_LAYERS) == {layer.value for layer in ExperimentEnvelopeLayer}
    assert budgets.budget_id == _budget_document()["budget_id"]


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
    prose = "x" * (budgets.for_layer("training").max_scalar_bytes + 1)
    document = {
        "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
        "name": "loud",
        "reason": prose,
        "training": {"rows_mode": "append", "tags": {"add": ["loud"]}},
    }
    raw = (json.dumps(document, indent=2) + "\n").encode("utf-8")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        read_authored_document(
            raw, budgets, field="studies/loud.envelope.json", layer_of=authored_layer_of
        )

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED
    assert "training layer's authored budget" in str(excinfo.value)


def test_the_same_content_is_admitted_under_the_layer_with_the_wider_cap(
    budgets: AuthoringBudgets,
) -> None:
    prose = "x" * (budgets.for_layer("training").max_scalar_bytes + 1)
    document = {
        "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
        "name": "loud",
        "reason": prose,
        "report": {"bindings": []},
    }
    raw = (json.dumps(document, indent=2) + "\n").encode("utf-8")

    parsed = read_authored_document(
        raw, budgets, field="studies/loud.envelope.json", layer_of=authored_layer_of
    )

    assert parsed["reason"] == prose


def test_project_caps_are_validated_but_left_to_the_project(budgets: AuthoringBudgets) -> None:
    assert budgets.for_layer("training").project_cap("max_probes") == 6
    with pytest.raises(KeyError):
        budgets.for_layer("report").project_cap("max_probes")


def test_budget_document_refuses_a_section_with_a_mistyped_cap() -> None:
    document = _budget_document()
    document["layers"]["training"]["max_lnies"] = document["layers"]["training"].pop("max_lines")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        AuthoringBudgets.from_document(
            document, field="budget.json", declared_layers=DECLARED_LAYERS
        )

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE


def test_budget_document_refuses_a_nonpositive_cap() -> None:
    document = _budget_document()
    document["layers"]["training"]["max_depth"] = 0

    with pytest.raises(ExperimentEnvelopeRejection):
        AuthoringBudgets.from_document(
            document, field="budget.json", declared_layers=DECLARED_LAYERS
        )


def test_budget_document_refuses_a_layer_the_dialect_does_not_declare() -> None:
    document = _budget_document()
    document["layers"]["ghost"] = dict(document["layers"]["training"])

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        AuthoringBudgets.from_document(
            document, field="budget.json", declared_layers=DECLARED_LAYERS
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
        "contract_id": EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_ID,
        "contract_version": EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION,
    }
    implementation = lock["compiler_implementation"]
    assert implementation["code_unit"] == "tests.test_envelope_engine_kernel"
    assert "feedbax" in implementation["package_versions"]
    assert "contract_version" not in implementation
    assert "code_unit" not in lock["compiler_contract"]


def test_the_compiler_contract_is_global_rather_than_per_project(repo: Path) -> None:
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert outcome.compile_lock["compiler_contract"]["contract_version"] == (
        EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION
    )
    assert not hasattr(PROJECT_DECLARATION, "compiler_contract_id")
    assert not hasattr(PROJECT_DECLARATION, "compiler_contract_version")


def test_contract_version_must_extend_its_contract_id() -> None:
    with pytest.raises(Exception, match="does not extend"):
        CompilerContract("feedbax.experiment_envelope.compiler", "other.v1")


def test_an_uninstalled_package_records_none_rather_than_vanishing() -> None:
    record = CompilerImplementation(
        code_unit="probe", packages=("feedbax", "no-such-package-at-all")
    ).record()

    assert record["package_versions"]["no-such-package-at-all"] is None
    assert set(record["package_versions"]) == {"feedbax", "no-such-package-at-all"}


def test_lock_pins_the_envelope_and_the_compiled_document() -> None:
    envelope = {"schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION, "name": "probe"}
    document = {"schema_id": TRAINING_SCHEMA_ID, "name": "probe"}

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
    loaded = load_compile_lock(_lock(), field="compiled/probe.compile-lock.json")

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
        repo / "bases" / "mid.training_run_matrix.json",
        {
            "schema_id": TRAINING_SCHEMA_ID,
            "name": "mid",
            "base": {"ref": TRAINING_BASE},
            "settings": {"span": 6},
        },
    )
    pinned = PinnedDocument.of(
        "bases/mid.training_run_matrix.json",
        json.loads((repo / "bases" / "mid.training_run_matrix.json").read_text()),
    )

    lineage = build_lineage(repo, pinned)

    span = lineage.lookup("settings.span")
    cadence = lineage.lookup("base.inline.cadence")
    assert span is not None and span.value == 6
    assert span.owner_ref == "bases/mid.training_run_matrix.json"
    assert cadence is not None and cadence.value == 2
    assert cadence.owner_ref == TRAINING_BASE


def test_lineage_pins_every_document_it_consulted(repo: Path) -> None:
    pinned = PinnedDocument.of(
        TRAINING_BASE, json.loads((repo / TRAINING_BASE).read_text())
    )

    pins = build_lineage(repo, pinned).pins()

    assert [pin["ref"] for pin in pins] == [TRAINING_BASE]
    assert all(pin["pin_algorithm"] == CANONICAL_PIN_ALGORITHM for pin in pins)
    assert all(len(pin["content_hash"]) == 64 for pin in pins)


def test_lineage_resolves_a_value_bound_by_a_patch_list() -> None:
    pinned = PinnedDocument.of(
        "delta.json",
        {"patches": [{"op": "replace", "path": "settings.span", "value": 11}]},
    )

    found = Lineage((pinned,)).lookup("settings.span")

    assert found is not None and found.value == 11


def test_lineage_pins_the_documents_a_parent_reads_through_its_sources(repo: Path) -> None:
    write_json(
        repo / "bases" / "cadence.table.json",
        {"schema_id": "quillon.cadence_table", "cadence": 2},
    )
    write_json(
        repo / "bases" / "probe.table.json",
        {"schema_id": "quillon.probe_table", "depth": 1},
    )
    _training_sources(
        repo,
        [
            {
                "alias": "cadence",
                "kind": "quillon.cadence_table",
                "uri": "bases/cadence.table.json",
            },
            {
                "alias": "probe",
                "kind": "quillon.probe_table",
                "uri": "bases/probe.table.json",
            },
        ],
    )

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    lineage = outcome.compile_lock["lineage"]
    assert [pin["ref"] for pin in lineage] == [
        TRAINING_BASE,
        "bases/cadence.table.json",
        "bases/probe.table.json",
    ]
    assert all(pin["pin_algorithm"] == CANONICAL_PIN_ALGORITHM for pin in lineage)
    assert all(len(pin["content_hash"]) == 64 for pin in lineage)


def test_a_source_that_cannot_be_read_is_refused_rather_than_silently_unpinned(
    repo: Path,
) -> None:
    _training_sources(
        repo,
        [{"alias": "cadence", "kind": "quillon.cadence_table", "uri": "bases/absent.json"}],
    )

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE
    assert excinfo.value.field == f"{TRAINING_BASE}#sources[0].uri"


def test_an_optional_source_states_its_own_absence_rather_than_failing_closed(
    repo: Path,
) -> None:
    _training_sources(
        repo,
        [
            {
                "alias": "cadence",
                "kind": "quillon.cadence_table",
                "uri": "bases/absent.json",
                "optional": True,
                "missing_payload": {"cadence": 2},
            }
        ],
    )

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert [pin["ref"] for pin in outcome.compile_lock["lineage"]] == [TRAINING_BASE]


def test_a_source_binding_that_names_no_document_is_refused(repo: Path) -> None:
    _training_sources(repo, [{"alias": "cadence", "kind": "quillon.cadence_table"}])

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert excinfo.value.field == f"{TRAINING_BASE}#sources[0].uri"


def test_lineage_walk_is_cycle_safe(repo: Path) -> None:
    write_json(
        repo / "bases" / "loop_a.json",
        {"schema_id": TRAINING_SCHEMA_ID, "base": {"ref": "bases/loop_b.json"}},
    )
    write_json(
        repo / "bases" / "loop_b.json",
        {"schema_id": TRAINING_SCHEMA_ID, "base": {"ref": "bases/loop_a.json"}},
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


def test_every_dialect_layer_compiles_to_its_feedbax_output_family(repo: Path) -> None:
    compiler = kernel()

    compiled = {
        alias: compiler.compile_envelope_file(envelope_path(repo, alias), repo_root=repo)
        for alias in (
            "widened",
            "widened-probe",
            "widened-summary",
            "widened-plot",
            "widened-overlay",
            "widened-report",
        )
    }

    assert {alias: outcome.layer.value for alias, outcome in compiled.items()} == {
        "widened": "training",
        "widened-probe": "evaluation",
        "widened-summary": "analysis",
        "widened-plot": "figure",
        "widened-overlay": "figure",
        "widened-report": "report",
    }
    assert {alias: outcome.family for alias, outcome in compiled.items()} == {
        "widened": "training_run_matrix",
        "widened-probe": "evaluation_run_matrix",
        "widened-summary": "analysis_run",
        "widened-plot": "figure",
        "widened-overlay": "figure_composition",
        "widened-report": "report",
    }
    assert {outcome.document["schema_id"] for outcome in compiled.values()} == {
        "feedbax.spec.training_run_matrix",
        "feedbax.spec.evaluation_run_matrix",
        "feedbax.spec.analysis_run",
        "feedbax.spec.figure",
        "feedbax.spec.figure_composition",
        "feedbax.spec.report",
    }


def test_an_authored_training_row_inherits_names_seeds_and_records_replacement(
    repo: Path,
) -> None:
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    rows = {row["row_id"]: row for row in outcome.document["rows"]}
    assert set(rows) == {"baseline", "widened"}
    assert rows["widened"]["seed"] == 43
    assert rows["widened"]["metadata"]["replaces"] == {"row": "baseline", "seed": 42}
    assert outcome.document["name"] == "widened"
    assert outcome.document["tags"] == ["widened"]


def test_a_row_delta_lands_as_native_override_patches(repo: Path) -> None:
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    widened = next(row for row in outcome.document["rows"] if row["row_id"] == "widened")
    assert [(patch["path"], patch["op"]) for patch in widened["overrides"]] == [
        ("span", "replace"),
        ("probe.depth", "remove"),
        ("cadence_profile", "add"),
    ]


def test_the_lock_records_the_native_delta_it_resolved(repo: Path) -> None:
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert set(outcome.compile_lock["resolved_deltas"]) == {"widened.training"}
    patches = outcome.compile_lock["resolved_deltas"]["widened.training"]["patches"]
    assert [patch["path"] for patch in patches] == ["name", "rows.1", "tags.1", "tags.0"]


def test_a_cross_layer_reference_pins_only_pre_run_facts(repo: Path) -> None:
    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-probe"), repo_root=repo
    )

    reference = outcome.compile_lock["references"][0]
    assert reference["kind"] == "planned_product"
    assert set(reference) == {
        "kind",
        "envelope_ref",
        "envelope_hash",
        "product_name",
        "product_schema_id",
        "product_schema_version",
        "compiled_content_hash",
        "role_path",
        "consumer",
    }
    assert not RUN_RECEIPT_ONLY_FACTS & set(reference)
    check_plan_receipt_boundary(outcome.compile_lock)


def test_a_receipt_without_a_digest_is_a_locator_not_a_fabricated_authentication(
    repo: Path,
) -> None:
    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-summary"), repo_root=repo
    )

    reference = outcome.compile_lock["references"][0]
    assert reference["kind"] == "receipt_locator"
    assert "manifest_sha256" not in reference
    assert reference["consumer"] == {
        "consumer": "analysis_input",
        "alias": "probe",
        "role": "observations",
    }


def test_an_authored_receipt_with_a_digest_is_quoted_as_authenticated(repo: Path) -> None:
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    reference = next(
        item
        for item in outcome.compile_lock["references"]
        if item["kind"] == "authenticated_receipt"
    )
    assert reference["size_bytes"] == 4096
    assert reference["consumer"] == {
        "consumer": "checkpoint_initialization",
        "mode": "continue_from",
        "row_id": "widened",
    }


def test_authored_not_applicability_is_recorded_rather_than_left_silent(repo: Path) -> None:
    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-report"), repo_root=repo
    )

    absent = next(
        item
        for item in outcome.compile_lock["references"]
        if item["kind"] == "not_applicable"
    )
    assert absent["basis"] == "authored"
    assert absent["role_path"] == "params.sections.0.tables.0"
    assert "rule_id" not in absent


def test_a_payload_in_its_own_file_is_recorded_as_a_content_pin(repo: Path) -> None:
    write_json(repo / "bases" / "survey.payload.json", _survey_payload())
    _repoint_training_base_to_file(repo)
    _reauthor(repo, "widened", **{"assert": []})

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    pin = next(
        item for item in outcome.compile_lock["references"] if item["kind"] == "content_pin"
    )
    assert pin["ref"] == "bases/survey.payload.json"
    assert "consumer" not in pin


def test_compilation_is_deterministic(repo: Path) -> None:
    compiler = kernel()

    first = compiler.compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)
    second = compiler.compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert emit_text(first.document) == emit_text(second.document)
    assert first.compile_lock == second.compile_lock


def test_an_assertion_that_holds_is_recorded_with_its_owner(repo: Path) -> None:
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert outcome.compile_lock["assertions"] == [
        {
            "path": "base.inline.cadence",
            "expected": 2,
            "actual": 2,
            "owner_ref": TRAINING_BASE,
        }
    ]


def test_an_assertion_that_fails_names_the_document_that_owns_the_value(repo: Path) -> None:
    _reauthor(repo, "widened", **{"assert": [{"path": "base.inline.cadence", "equals": 99}]})

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.ASSERTION_FAILED
    assert TRAINING_BASE in str(excinfo.value)


def test_an_assertion_may_not_guard_a_path_the_envelope_changes(repo: Path) -> None:
    _reauthor(repo, "widened", **{"assert": [{"path": "name", "equals": "baseline"}]})

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is (
        ExperimentEnvelopeRejectionCategory.ILLEGAL_ASSERTION_PATH
    )


def test_an_assertion_on_an_uninherited_path_has_nothing_to_check(repo: Path) -> None:
    _reauthor(repo, "widened", **{"assert": [{"path": "base.inline.absent", "equals": 1}]})

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert "not inherited" in str(excinfo.value)


def test_an_echoed_inherited_name_is_refused(repo: Path) -> None:
    _reauthor(repo, "widened", name="baseline", **{"assert": []})

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is (
        ExperimentEnvelopeRejectionCategory.ECHOED_INHERITED_VALUE
    )
    assert TRAINING_BASE in str(excinfo.value)


def test_an_echoed_inherited_seed_is_refused(repo: Path) -> None:
    envelope = _read(repo, "widened")
    envelope["training"]["rows"][0]["seed"] = 42

    _write(repo, "widened", envelope)

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is (
        ExperimentEnvelopeRejectionCategory.ECHOED_INHERITED_VALUE
    )


def test_a_base_under_the_output_directory_is_refused(repo: Path) -> None:
    _reauthor(repo, "widened", base=f"{OUTPUT_DIRECTORY}/widened.training_run_matrix.json")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE
    assert "compiled output" in str(excinfo.value)


def test_a_normalized_path_cannot_smuggle_in_a_compiled_base(repo: Path) -> None:
    _reauthor(
        repo,
        "widened",
        base=f"./{OUTPUT_DIRECTORY}/../{OUTPUT_DIRECTORY}/widened.training_run_matrix.json",
    )

    with pytest.raises(ExperimentEnvelopeRejection, match="compiled output"):
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)


def test_a_cross_layer_base_is_refused(repo: Path) -> None:
    _reauthor(repo, "widened", base="bases/baseline.analysis_run.json", **{"assert": []})

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.CROSS_FAMILY_BASE


def test_a_base_of_no_feedbax_output_family_is_not_an_experiment_parent(repo: Path) -> None:
    write_json(repo / "bases" / "stray.json", {"schema_id": "quillon.notes", "name": "stray"})
    _reauthor(repo, "widened", base="bases/stray.json", **{"assert": []})

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE


def test_an_envelope_alias_parent_resolves_and_pins_its_compiled_bytes(repo: Path) -> None:
    write_envelope(
        envelope_path(repo, "narrowed"),
        {
            "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
            "name": "narrowed",
            "base": "widened",
            "training": {"rows_mode": "append", "tags": {"add": ["narrowed"]}},
        },
    )
    compiler = kernel()

    outcome = compiler.compile_envelope_file(envelope_path(repo, "narrowed"), repo_root=repo)
    parent = compiler.compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    base = outcome.compile_lock["base"]
    assert base["kind"] == "envelope_alias"
    assert base["ref"] == f"{ENVELOPE_DIRECTORY}/widened.envelope.json"
    assert base["content_hash"] == canonical_sha256(parent.document)


def test_an_envelope_alias_cycle_is_refused(repo: Path) -> None:
    for alias, base in (("ping", "pong"), ("pong", "ping")):
        write_envelope(
            envelope_path(repo, alias),
            {
                "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
                "name": alias,
                "base": base,
                "training": {"rows_mode": "append", "tags": {"add": [alias]}},
            },
        )

    with pytest.raises(ExperimentEnvelopeRejection, match="cycle"):
        kernel().compile_envelope_file(envelope_path(repo, "ping"), repo_root=repo)


def test_a_rootless_envelope_is_refused(repo: Path) -> None:
    envelope = _read(repo, "widened")
    envelope.pop("base")
    envelope.pop("assert")

    _write(repo, "widened", envelope)

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.MISSING_FIELD


def test_an_over_budget_assertion_count_is_refused(repo: Path) -> None:
    limit = load_project_budgets(PROJECT_DECLARATION).for_layer("training").max_assertions
    _reauthor(
        repo,
        "widened",
        **{
            "assert": [
                {"path": f"base.inline.probe.depth{index}", "equals": index}
                for index in range(limit + 1)
            ]
        },
    )

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED
    assert "assertions exceeds" in str(excinfo.value)


def test_write_outputs_is_byte_reproducible(repo: Path) -> None:
    compiler = kernel()
    outcome = compiler.compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)
    out_dir = repo / OUTPUT_DIRECTORY

    first = {
        path: path.read_bytes() for path in compiler.write_outputs(outcome, out_dir).values()
    }
    second = {
        path: path.read_bytes() for path in compiler.write_outputs(outcome, out_dir).values()
    }

    assert first == second
    assert set(first) == set(compiler.output_paths(outcome, out_dir).values())


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
    compiler = kernel()
    _regenerate(compiler, repo)

    report = compare_tracked_outputs(compiler, repo)

    assert report.ok
    assert report.drift == ()
    assert len(report.by_finding(ChokeFinding.IDENTICAL)) == 12


def test_a_hand_edited_compiled_document_reports_structured_drift(repo: Path) -> None:
    compiler = kernel()
    _regenerate(compiler, repo)
    edited = repo / OUTPUT_DIRECTORY / f"widened.{TRAINING_FAMILY}.json"
    edited.write_text(edited.read_text().replace('"seed": 43', '"seed": 44'), encoding="utf-8")

    report = compare_tracked_outputs(compiler, repo)

    assert not report.ok
    drift = report.by_finding(ChokeFinding.DIFFERS)
    assert [entry.path for entry in drift] == [
        f"{OUTPUT_DIRECTORY}/widened.{TRAINING_FAMILY}.json"
    ]
    assert drift[0].envelope_ref == f"{ENVELOPE_DIRECTORY}/widened.envelope.json"


def test_an_untracked_output_reports_missing(repo: Path) -> None:
    compiler = kernel()
    _regenerate(compiler, repo)
    (repo / OUTPUT_DIRECTORY / f"widened.{TRAINING_FAMILY}.json").unlink()

    report = compare_tracked_outputs(compiler, repo)

    assert [entry.finding for entry in report.drift] == [ChokeFinding.MISSING]


def test_a_compiled_document_no_envelope_produces_reports_orphaned(repo: Path) -> None:
    compiler = kernel()
    _regenerate(compiler, repo)
    write_json(repo / OUTPUT_DIRECTORY / "stray.json", {"schema_id": "stray"})

    report = compare_tracked_outputs(compiler, repo)

    orphans = report.by_finding(ChokeFinding.ORPHANED)
    assert [entry.path for entry in orphans] == [f"{OUTPUT_DIRECTORY}/stray.json"]


def test_an_envelope_that_no_longer_compiles_is_a_finding_not_an_exception(repo: Path) -> None:
    """One broken envelope is reported, not raised, and takes its dependants with it.

    ``widened-probe`` names ``widened`` as an upstream reference, so breaking
    ``widened`` breaks both. Both arrive as findings on one report rather than as
    the first exception to escape, which is what lets a single pass state the
    whole tree's condition.
    """
    compiler = kernel()
    _regenerate(compiler, repo)
    _reauthor(repo, "widened", **{"assert": [{"path": "base.inline.cadence", "equals": 99}]})

    report = compare_tracked_outputs(compiler, repo)

    rejected = report.by_finding(ChokeFinding.REJECTED)
    assert f"{ENVELOPE_DIRECTORY}/widened.envelope.json" in {
        entry.envelope_ref for entry in rejected
    }
    assert f"{ENVELOPE_DIRECTORY}/widened-probe.envelope.json" in {
        entry.envelope_ref for entry in rejected
    }
    assert all("no longer compiles" in (entry.detail or "") for entry in rejected)
    assert not report.ok
    assert report.describe()


# -- helpers -------------------------------------------------------------------


def _survey_payload() -> dict[str, Any]:
    from tests.fake_project_experiment import SURVEY_PAYLOAD

    return dict(SURVEY_PAYLOAD)


def _repoint_training_base_to_file(repo: Path) -> None:
    """Move the training payload out of the matrix and into its own tracked file."""
    from feedbax.contracts.authored_canonical import canonical_sha256 as _hash

    document = json.loads((repo / TRAINING_BASE).read_text())
    document["base"] = {
        "kind": "authored_intent",
        "ref": "bases/survey.payload.json",
        "content_hash": _hash(_survey_payload()),
    }
    write_json(repo / TRAINING_BASE, document)


def _training_sources(repo: Path, sources: list[dict[str, Any]]) -> None:
    """Give the training base a ``sources`` block naming further documents."""
    document = json.loads((repo / TRAINING_BASE).read_text())
    document["sources"] = sources
    write_json(repo / TRAINING_BASE, document)


def _budget_document() -> dict[str, Any]:
    resource = PROJECT_DECLARATION.authoring_budget
    return json.loads((resource.root / resource.document_name).read_text())


def _lock(
    *,
    envelope: dict[str, Any] | None = None,
    document: dict[str, Any] | None = None,
    identity_contributions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_compile_lock(
        CompileLockInputs(
            envelope_ref="studies/probe.envelope.json",
            envelope_document=envelope
            or {"schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION, "name": "probe"},
            envelope_schema=EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
            name="probe",
            family=TRAINING_FAMILY,
            compiled_document=document or {"schema_id": TRAINING_SCHEMA_ID, "name": "probe"},
            contract=CompilerContract(
                EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_ID,
                EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION,
            ),
            implementation=CompilerImplementation(code_unit="tests.test_envelope_engine_kernel"),
            identity_contributions=identity_contributions or {},
        )
    )


def _read(repo: Path, alias: str) -> dict[str, Any]:
    return json.loads(envelope_path(repo, alias).read_text())


def _write(repo: Path, alias: str, document: dict[str, Any]) -> None:
    write_envelope(envelope_path(repo, alias), document)


def _reauthor(repo: Path, alias: str, **changes: Any) -> None:
    document = _read(repo, alias)
    document.update(changes)
    _write(repo, alias, document)


def _regenerate(compiler: Any, repo: Path) -> None:
    out_dir = repo / OUTPUT_DIRECTORY
    for path in compiler.envelopes(repo):
        compiler.write_outputs(compiler.compile_envelope_file(path, repo_root=repo), out_dir)


# -- the ratified equivalence corrections --------------------------------------


def _training_layer(repo: Path, **changes: Any) -> None:
    """Reauthor the training envelope's layer body in place."""
    envelope = _read(repo, "widened")
    envelope["training"] = {**envelope["training"], **changes}
    _write(repo, "widened", envelope)


def test_authored_only_runs_exactly_the_rows_the_envelope_authors(repo: Path) -> None:
    _training_layer(repo, rows_mode="authored_only")

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert [row["row_id"] for row in outcome.document["rows"]] == ["widened"]
    patches = outcome.compile_lock["resolved_deltas"]["widened.training"]["patches"]
    assert ("rows", "replace") in [(patch["path"], patch["op"]) for patch in patches]


def test_append_keeps_the_inherited_rows_running_beside_the_authored_ones(
    repo: Path,
) -> None:
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert [row["row_id"] for row in outcome.document["rows"]] == ["baseline", "widened"]


@pytest.mark.parametrize("rows_mode", ["append", "authored_only"])
def test_a_derived_row_records_the_parent_row_it_was_resolved_from(
    repo: Path, rows_mode: str
) -> None:
    _training_layer(repo, rows_mode=rows_mode)

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert outcome.compile_lock["row_provenance"] == [
        {
            "row_id": "widened",
            "source_row_key": "baseline",
            "source_ref": TRAINING_BASE,
            "source_content_hash": canonical_sha256(
                json.loads((repo / TRAINING_BASE).read_text())
            ),
            "pin_algorithm": CANONICAL_PIN_ALGORITHM,
        }
    ]


def test_row_provenance_pins_the_same_parent_the_lock_records_as_its_base(
    repo: Path,
) -> None:
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    record = outcome.compile_lock["row_provenance"][0]
    base = outcome.compile_lock["base"]
    assert (record["source_ref"], record["source_content_hash"]) == (
        base["ref"],
        base["content_hash"],
    )


def test_a_layer_that_derives_no_row_records_no_row_provenance(repo: Path) -> None:
    _training_layer(repo, rows=[], checkpoint_initialization=[])

    training = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)
    evaluation = kernel().compile_envelope_file(
        envelope_path(repo, "widened-probe"), repo_root=repo
    )

    assert training.compile_lock["row_provenance"] == []
    assert evaluation.compile_lock["row_provenance"] == []


def test_a_new_row_is_labelled_by_its_own_id_rather_than_its_sources(repo: Path) -> None:
    write_json(
        repo / TRAINING_BASE,
        {
            **json.loads((repo / TRAINING_BASE).read_text()),
            "rows": [
                {
                    "row_id": "baseline",
                    "label": "the baseline survey",
                    "seed": 42,
                    "overrides": [],
                }
            ],
        },
    )

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    rows = {row["row_id"]: row for row in outcome.document["rows"]}
    assert rows["baseline"]["label"] == "the baseline survey"
    assert rows["widened"]["label"] == "widened"


def test_an_authored_row_label_is_carried_exactly(repo: Path) -> None:
    envelope = _read(repo, "widened")
    envelope["training"]["rows"][0]["label"] = "the widened survey"
    _write(repo, "widened", envelope)

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    widened = next(row for row in outcome.document["rows"] if row["row_id"] == "widened")
    assert widened["label"] == "the widened survey"


def test_a_changed_row_inherits_none_of_its_sources_opaque_metadata(repo: Path) -> None:
    """The source states facts about the experiment it was, not about this one."""
    write_json(
        repo / TRAINING_BASE,
        {
            **json.loads((repo / TRAINING_BASE).read_text()),
            "rows": [
                {
                    "row_id": "baseline",
                    "seed": 42,
                    "overrides": [],
                    "metadata": {"probe_delta": "none", "launch_set": "survey-1"},
                }
            ],
        },
    )

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    rows = {row["row_id"]: row for row in outcome.document["rows"]}
    assert rows["baseline"]["metadata"] == {"probe_delta": "none", "launch_set": "survey-1"}
    assert rows["widened"]["metadata"] == {"replaces": {"row": "baseline", "seed": 42}}


def test_a_row_without_authored_replacement_carries_no_metadata_at_all(
    repo: Path,
) -> None:
    envelope = _read(repo, "widened")
    envelope["training"]["rows"][0].pop("replaces")
    _write(repo, "widened", envelope)

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    widened = next(row for row in outcome.document["rows"] if row["row_id"] == "widened")
    assert "metadata" not in widened


def test_the_compiled_matrix_omits_the_parents_issue_and_opaque_metadata(
    repo: Path,
) -> None:
    """Provenance lives in the lock; the parent's ticket is not this matrix's."""
    write_json(
        repo / TRAINING_BASE,
        {
            **json.loads((repo / TRAINING_BASE).read_text()),
            "issue": "q0parent",
            "metadata": {"orchestration_root": "/runs/survey-1"},
        },
    )

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert "issue" not in outcome.document
    assert "metadata" not in outcome.document
    assert outcome.compile_lock["issue"] == "q1a2b3c"


def test_checkpoint_initialization_may_not_name_a_row_the_matrix_no_longer_runs(
    repo: Path,
) -> None:
    envelope = _read(repo, "widened")
    envelope["training"]["rows_mode"] = "authored_only"
    envelope["training"]["checkpoint_initialization"][0]["row"] = "baseline"
    _write(repo, "widened", envelope)

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_ROW_KEY
    assert "not a row this matrix runs" in str(excinfo.value)


# -- figure mode ---------------------------------------------------------------


def test_row_expansion_derives_the_multi_row_figure_from_the_row_index(
    repo: Path,
) -> None:
    from tests.fake_project_experiment import ROW_INDEX_BASE

    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-plot"), repo_root=repo
    )

    document = outcome.document
    assert document["schema_id"] == "feedbax.spec.figure"
    assert [panel["name"] for panel in document["panels"]] == [
        "row_1__span",
        "row_2__span",
    ]
    assert [panel["title"] for panel in document["panels"]] == [
        "near span — span",
        "far span — span",
    ]
    assert document["assembler_params"]["height"] == 600
    pins = [
        item for item in outcome.compile_lock["references"] if item["kind"] == "content_pin"
    ]
    assert [pin["ref"] for pin in pins] == [ROW_INDEX_BASE]


def test_row_expansion_names_no_produced_data_in_the_compiled_plan(repo: Path) -> None:
    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-plot"), repo_root=repo
    )

    assert not outcome.document.get("inputs")
    assert not outcome.document.get("input_authorities")
    kinds = {item["kind"] for item in outcome.compile_lock["references"]}
    assert "planned_product" not in kinds
    assert "not_applicable" in kinds
    check_plan_receipt_boundary(outcome.compile_lock)


def test_the_expansion_request_and_resolved_rows_carry_execution_identity(
    repo: Path,
) -> None:
    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-plot"), repo_root=repo
    )

    contributions = outcome.compile_lock["identity_contributions"]
    request = contributions["figure_row_expansion"]
    assert request["schema_id"] == "feedbax.spec.figure_row_expansion_request"
    assert request["inputs"] == {"observed": {"per_row": "observations"}}
    assert [contract["input_role"] for contract in request["role_contracts"]] == ["observed"]
    assert contributions["resolved_row_set"]["row_ids"] == ["near-span", "far-span"]
    assert outcome.compile_lock["execution_identity"]["inputs"] == [
        "compiled_document.content_hash",
        "identity_contributions.figure_row_expansion",
        "identity_contributions.resolved_row_set",
    ]


def test_composition_compiles_to_a_composition_document_pinning_its_parent(
    repo: Path,
) -> None:
    from tests.fake_project_experiment import FIGURE_BASE

    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-overlay"), repo_root=repo
    )

    assert outcome.document["schema_id"] == "feedbax.spec.figure_composition"
    assert outcome.document["parent"]["ref"] == FIGURE_BASE
    assert outcome.document["parent"]["sha256"] == canonical_sha256(
        json.loads((repo / FIGURE_BASE).read_text())
    )
    assert [delta["layer_id"] for delta in outcome.document["deltas"]] == [
        "widened-overlay"
    ]


def test_the_same_base_under_two_modes_produces_two_different_families(
    repo: Path,
) -> None:
    """A filename never selects semantics; the authored mode does."""
    compiler = kernel()

    expanded = compiler.compile_envelope_file(
        envelope_path(repo, "widened-plot"), repo_root=repo
    )
    composed = compiler.compile_envelope_file(
        envelope_path(repo, "widened-overlay"), repo_root=repo
    )

    assert expanded.compile_lock["base"]["ref"] == composed.compile_lock["base"]["ref"]
    assert (expanded.family, composed.family) == ("figure", "figure_composition")


def test_a_composition_document_is_not_an_experiment_parent(repo: Path) -> None:
    from tests.fake_project_experiment import FIGURE_BASE

    write_json(
        repo / "bases" / "composed.figure_composition.json",
        {
            "schema_id": "feedbax.spec.figure_composition",
            "schema_version": "feedbax.spec.figure_composition.v2",
            "parent": {"ref": FIGURE_BASE, "sha256": "a" * 64},
            "deltas": [{"layer_id": "prior", "patches": []}],
        },
    )
    _reauthor(repo, "widened-overlay", base="bases/composed.figure_composition.json")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened-overlay"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.CROSS_FAMILY_BASE


def test_a_composition_parent_is_pinned_by_path_so_an_alias_is_refused(
    repo: Path,
) -> None:
    _reauthor(repo, "widened-overlay", base="widened-plot")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened-overlay"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE
    assert "content-pins its parent" in str(excinfo.value)


def test_a_row_index_that_is_not_one_is_refused_by_identity(repo: Path) -> None:
    write_json(repo / "bases" / "notes.json", {"schema_id": "quillon.notes", "rows": []})
    envelope = _read(repo, "widened-plot")
    envelope["figure"]["rows"] = {"mode": "all", "index": "bases/notes.json"}
    _write(repo, "widened-plot", envelope)

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE
    assert "feedbax.spec.authenticated_row_index" in str(excinfo.value)


def test_a_selector_that_resolves_to_no_rows_is_an_empty_selection(repo: Path) -> None:
    from tests.fake_project_experiment import ROW_INDEX_BASE

    envelope = _read(repo, "widened-plot")
    envelope["figure"]["rows"] = {"mode": "tag", "tag": "absent", "index": ROW_INDEX_BASE}
    _write(repo, "widened-plot", envelope)

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.EMPTY_SELECTION


def test_an_input_role_without_a_declared_contract_is_refused(repo: Path) -> None:
    envelope = _read(repo, "widened-plot")
    envelope["figure"]["inputs"][0]["input_role"] = "unfilled"
    envelope["figure"]["inputs"][0]["contract"] = {
        "artifact_role": "span_observations",
        "artifact_provider": "quillon.custody",
    }
    _write(repo, "widened-plot", envelope)

    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-plot"), repo_root=repo
    )

    request = outcome.compile_lock["identity_contributions"]["figure_row_expansion"]
    assert list(request["inputs"]) == ["unfilled"]


# -- the top-level report ------------------------------------------------------


def test_the_report_layer_compiles_to_a_top_level_report_spec(repo: Path) -> None:
    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-report"), repo_root=repo
    )

    document = outcome.document
    assert document["schema_id"] == "feedbax.spec.report"
    assert document["schema_version"] == "feedbax.spec.report.v1"
    assert document["report_type"] == "feedbax.ordered_figure_report"
    assert document["params"]["title"] == "Quillon widened span"
    assert document["params"]["schema_id"] == "feedbax.spec.report.ordered_figure"
    assert not document.get("inputs")


def test_the_reports_params_are_validated_against_their_declared_content_type(
    repo: Path,
) -> None:
    envelope = _read(repo, "widened-report")
    envelope["report"]["delta"]["patches"] = [
        {"path": "params.output_name", "op": "add", "value": "not/a/name.md"}
    ]
    _write(repo, "widened-report", envelope)

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened-report"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert excinfo.value.field is not None and excinfo.value.field.endswith("#params")
    assert "output_name" in str(excinfo.value)


def test_a_params_only_report_document_is_not_a_report_parent(repo: Path) -> None:
    """The parent is the whole report; a bare ordered-figure block is not one."""
    write_json(
        repo / "bases" / "params_only.report.json",
        {
            "schema_id": "feedbax.spec.report.ordered_figure",
            "schema_version": "feedbax.spec.report.ordered_figure.v3",
            "title": "Quillon baseline span",
            "sections": [{"title": "Span", "figures": [], "tables": []}],
        },
    )
    _reauthor(repo, "widened-report", base="bases/params_only.report.json")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened-report"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE


def test_a_schemaless_report_parent_is_refused_rather_than_admitted(repo: Path) -> None:
    write_json(
        repo / "bases" / "schemaless.report.json",
        {"title": "Quillon baseline span", "sections": []},
    )
    _reauthor(repo, "widened-report", base="bases/schemaless.report.json")

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened-report"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE


def test_the_compiled_report_is_the_document_fulfillment_plans_against(
    repo: Path,
) -> None:
    from feedbax.analysis.fulfillment_derivation import COMPILED_PRODUCT_KINDS

    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-report"), repo_root=repo
    )

    kind = COMPILED_PRODUCT_KINDS[outcome.document["schema_id"]]
    assert kind.layer == "report"
    assert kind.executable


def test_a_row_slice_is_expressed_as_a_tag_over_the_same_row_index(repo: Path) -> None:
    """A slice is a selection over the index, not a list inside the envelope."""
    from tests.fake_project_experiment import ROW_INDEX_BASE

    index = json.loads((repo / ROW_INDEX_BASE).read_text())
    index["rows"][1]["tags"] = ["survey", "held-out"]
    write_json(repo / ROW_INDEX_BASE, index)
    envelope = _read(repo, "widened-plot")
    envelope["figure"]["rows"] = {
        "mode": "tag",
        "tag": "held-out",
        "index": ROW_INDEX_BASE,
    }
    _write(repo, "widened-plot", envelope)

    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-plot"), repo_root=repo
    )

    resolved = outcome.compile_lock["identity_contributions"]["resolved_row_set"]
    assert resolved["row_ids"] == ["far-span"]
    assert [panel["name"] for panel in outcome.document["panels"]] == ["row_1__span"]


def test_a_figure_runtime_input_that_has_not_run_is_a_locator_in_the_lock(
    repo: Path,
) -> None:
    envelope = _read(repo, "widened-plot")
    # A locator belongs to a role one manifest fills for every row; a per-row
    # role has no single locator and the dialect refuses one on it.
    envelope["figure"]["inputs"][0]["binding"] = "shared"
    envelope["figure"]["inputs"][0]["binding_key"] = "widened-plot-observed"
    envelope["figure"]["inputs"][0]["ref"] = {
        "kind": "receipt",
        "manifest_kind": "quillon.survey_run",
        "manifest_id": "widened-plot-observed",
    }
    _write(repo, "widened-plot", envelope)

    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-plot"), repo_root=repo
    )

    reference = next(
        item
        for item in outcome.compile_lock["references"]
        if item["kind"] == "receipt_locator"
    )
    assert "manifest_sha256" not in reference
    assert reference["role_path"] == "inputs.observed"
    assert reference["consumer"] == {
        "consumer": "figure_runtime_input",
        "input_role": "observed",
    }


# -- tag removal ----------------------------------------------------------------


def _training_tags(repo: Path, tags: list[str]) -> None:
    """Give the training base the inherited tag list a delta is stated over."""
    document = json.loads((repo / TRAINING_BASE).read_text())
    document["tags"] = tags
    write_json(repo / TRAINING_BASE, document)


def _generated_delta(outcome: Any) -> dict[str, Any]:
    """Return the layer the engine derived, as the lock records it."""
    resolved = outcome.compile_lock["resolved_deltas"]
    return resolved[f"{outcome.name}.training"]


def test_an_inherited_tag_can_be_removed(repo: Path) -> None:
    _training_tags(repo, ["baseline", "pilot"])
    _training_layer(repo, tags={"add": ["widened"], "remove": ["baseline", "pilot"]})

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert outcome.document["tags"] == ["widened"]


def test_the_generated_tag_layer_acknowledges_only_the_paths_it_rewrites(
    repo: Path,
) -> None:
    """Closing the list up behind one removal makes the next land on a written path."""
    _training_tags(repo, ["baseline", "pilot", "probe"])
    _training_layer(repo, tags={"remove": ["baseline", "pilot", "probe"]})

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    delta = _generated_delta(outcome)
    tag_paths = [
        patch["path"] for patch in delta["patches"] if patch["path"].startswith("tags.")
    ]
    assert tag_paths == ["tags.0", "tags.0", "tags.0"]
    assert delta["acknowledges_ancestor_paths"] == ["tags.0"]
    assert outcome.document["tags"] == []


def test_add_only_tag_authoring_acknowledges_nothing(repo: Path) -> None:
    _training_tags(repo, ["baseline"])
    _training_layer(repo, tags={"add": ["widened", "probe"]})

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert _generated_delta(outcome)["acknowledges_ancestor_paths"] == []
    assert outcome.document["tags"] == ["baseline", "widened", "probe"]


def test_removing_a_tag_the_base_does_not_state_is_still_refused(repo: Path) -> None:
    _training_tags(repo, ["baseline"])
    _training_layer(repo, tags={"remove": ["absent"]})

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert "nothing to remove" in str(excinfo.value)


# -- report binding state -------------------------------------------------------


def _report_base_figure(repo: Path, **figure: Any) -> None:
    """Give the report base one ordered-figure entry at the bound role."""
    from tests.fake_project_experiment import REPORT_BASE

    document = json.loads((repo / REPORT_BASE).read_text())
    document["params"]["sections"][0]["figures"] = [
        {"input_role": "span", "caption": "Widened span", **figure}
    ]
    write_json(repo / REPORT_BASE, document)


def _report_bindings(repo: Path, bindings: list[dict[str, Any]]) -> None:
    envelope = _read(repo, "widened-report")
    envelope["report"] = {**envelope["report"], "bindings": bindings}
    _write(repo, "widened-report", envelope)


def _compile_report(repo: Path) -> Any:
    return kernel().compile_envelope_file(
        envelope_path(repo, "widened-report"), repo_root=repo
    )


NOT_APPLICABLE_BINDING = {
    "role_path": "params.sections.0.figures.0",
    "ref": {
        "kind": "not_applicable",
        "reason": "the widened survey never produced this panel",
    },
}
FIGURE_BINDING = {
    "role_path": "params.sections.0.figures.0",
    "ref": {"kind": "envelope", "alias": "widened-plot"},
}


def test_a_not_applicable_binding_reconciles_the_inherited_applicability(
    repo: Path,
) -> None:
    _report_base_figure(repo, figure_spec_sha256="a" * 64, applicability="included")
    _report_bindings(repo, [NOT_APPLICABLE_BINDING])

    outcome = _compile_report(repo)

    entry = outcome.document["params"]["sections"][0]["figures"][0]
    reference = next(
        item
        for item in outcome.compile_lock["references"]
        if item["kind"] == "not_applicable"
    )
    assert entry["applicability"] == "not_applicable"
    assert entry["not_applicable_reason"] == reference["reason"]
    assert "figure_spec_sha256" not in entry
    assert "input_role" not in entry
    assert reference["role_path"] == "params.sections.0.figures.0"
    assert reference["basis"] == "authored"


def test_a_not_applicable_binding_states_the_applicability_the_base_left_default(
    repo: Path,
) -> None:
    """An absent descriptor still claims the contract's default, which is inclusion."""
    _report_base_figure(repo, figure_spec_sha256="a" * 64)
    _report_bindings(repo, [NOT_APPLICABLE_BINDING])

    outcome = _compile_report(repo)

    entry = outcome.document["params"]["sections"][0]["figures"][0]
    assert entry["applicability"] == "not_applicable"


def test_a_bound_role_carries_the_digest_of_the_figure_it_is_bound_to(
    repo: Path,
) -> None:
    _report_base_figure(repo, figure_spec_sha256="a" * 64)
    _report_bindings(repo, [FIGURE_BINDING])

    outcome = _compile_report(repo)

    entry = outcome.document["params"]["sections"][0]["figures"][0]
    planned = next(
        item
        for item in outcome.compile_lock["references"]
        if item["kind"] == "planned_product"
    )
    figure = kernel().compile_envelope_file(
        envelope_path(repo, "widened-plot"), repo_root=repo
    )
    assert entry["figure_spec_sha256"] == planned["compiled_content_hash"]
    assert entry["figure_spec_sha256"] == canonical_sha256(figure.document)
    assert entry["figure_spec_sha256"] != "a" * 64


def test_a_role_already_carrying_its_bound_digest_derives_no_patch(repo: Path) -> None:
    figure = kernel().compile_envelope_file(
        envelope_path(repo, "widened-plot"), repo_root=repo
    )
    _report_base_figure(repo, figure_spec_sha256=canonical_sha256(figure.document))
    _report_bindings(repo, [FIGURE_BINDING])

    outcome = _compile_report(repo)

    assert f"{outcome.name}.report" not in outcome.compile_lock["resolved_deltas"]


def test_a_receipt_bound_role_cannot_inherit_a_digest_it_cannot_replace(
    repo: Path,
) -> None:
    _report_base_figure(repo, figure_spec_sha256="a" * 64)
    _report_bindings(
        repo,
        [
            {
                "role_path": "params.sections.0.figures.0",
                "ref": {
                    "kind": "receipt",
                    "manifest_kind": "quillon.survey_run",
                    "manifest_id": "widened-plot-0",
                },
            }
        ],
    )

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        _compile_report(repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert "may never author one" in str(excinfo.value)


def test_a_role_the_base_states_as_not_applicable_may_not_be_bound_to_a_product(
    repo: Path,
) -> None:
    _report_base_figure(
        repo,
        applicability="not_applicable",
        not_applicable_reason="the baseline survey has no widened panel",
        input_role=None,
    )
    _report_bindings(repo, [FIGURE_BINDING])

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        _compile_report(repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert "cannot derive the input role" in str(excinfo.value)


def test_a_role_the_document_says_nothing_about_stays_a_lock_only_binding(
    repo: Path,
) -> None:
    """The ratified stance stands where there is no inherited state to contradict."""
    outcome = _compile_report(repo)

    assert outcome.document["params"]["sections"][0]["figures"] == []
    assert f"{outcome.name}.report" not in outcome.compile_lock["resolved_deltas"]


def test_binding_state_is_reconciled_only_inside_a_report_type_feedbax_owns(
    repo: Path,
) -> None:
    from tests.fake_project_experiment import REPORT_BASE

    _report_base_figure(repo, figure_spec_sha256="a" * 64)
    document = json.loads((repo / REPORT_BASE).read_text())
    document["report_type"] = "quillon.bulletin"
    write_json(repo / REPORT_BASE, document)
    _report_bindings(repo, [NOT_APPLICABLE_BINDING])

    outcome = _compile_report(repo)

    entry = outcome.document["params"]["sections"][0]["figures"][0]
    assert entry == {
        "input_role": "span",
        "caption": "Widened span",
        "figure_spec_sha256": "a" * 64,
    }


# -- a not-applicable section is reconciled by node type, not by inherited bytes --


SECTION_NOT_APPLICABLE_BINDING = {
    "role_path": "params.sections.0",
    "ref": {
        "kind": "not_applicable",
        "reason": "the widened survey has no comparison arm to report on",
    },
}


def _not_applicable_reference(outcome: Any) -> dict[str, Any]:
    return next(
        item
        for item in outcome.compile_lock["references"]
        if item["kind"] == "not_applicable"
    )


def test_a_not_applicable_section_states_the_applicability_its_bytes_never_carried(
    repo: Path,
) -> None:
    """The base section describes only content, which is the whole gap."""
    from tests.fake_project_experiment import REPORT_BASE

    _report_base_figure(repo, figure_spec_sha256="a" * 64, applicability="included")
    _report_bindings(repo, [SECTION_NOT_APPLICABLE_BINDING])
    inherited = json.loads((repo / REPORT_BASE).read_text())["params"]["sections"][0]
    assert not [name for name in REPORT_BINDING_STATE_FIELDS if name in inherited]

    outcome = _compile_report(repo)

    section = outcome.document["params"]["sections"][0]
    reference = _not_applicable_reference(outcome)
    assert section["applicability"] == "not_applicable"
    assert section["not_applicable_reason"] == reference["reason"]
    assert reference["role_path"] == "params.sections.0"
    assert reference["basis"] == "authored"


def test_a_not_applicable_section_no_longer_claims_the_figures_it_inherited(
    repo: Path,
) -> None:
    _report_base_figure(repo, figure_spec_sha256="a" * 64, applicability="included")
    _report_bindings(repo, [SECTION_NOT_APPLICABLE_BINDING])

    outcome = _compile_report(repo)

    section = outcome.document["params"]["sections"][0]
    assert "figures" not in section
    patches = outcome.compile_lock["resolved_deltas"][f"{outcome.name}.report"]["patches"]
    assert [(patch["path"], patch["op"]) for patch in patches] == [
        ("params.sections.0.applicability", "add"),
        ("params.sections.0.not_applicable_reason", "add"),
        ("params.sections.0.figures", "remove"),
    ]


def test_a_section_already_stating_its_bound_applicability_is_replaced_not_added(
    repo: Path,
) -> None:
    """A contradicting inherited descriptor is rewritten; an agreeing one is left."""
    from tests.fake_project_experiment import REPORT_BASE

    document = json.loads((repo / REPORT_BASE).read_text())
    document["params"]["sections"][0]["applicability"] = "included"
    write_json(repo / REPORT_BASE, document)
    _report_bindings(repo, [SECTION_NOT_APPLICABLE_BINDING])

    outcome = _compile_report(repo)

    patches = outcome.compile_lock["resolved_deltas"][f"{outcome.name}.report"]["patches"]
    assert [(patch["path"], patch["op"]) for patch in patches] == [
        ("params.sections.0.applicability", "replace"),
        ("params.sections.0.not_applicable_reason", "add"),
        ("params.sections.0.figures", "remove"),
    ]


def test_recompiling_a_reconciled_section_derives_no_further_patch(repo: Path) -> None:
    """A reconciled base is a fixed point: the second compile changes nothing."""
    from tests.fake_project_experiment import REPORT_BASE

    _report_base_figure(repo, figure_spec_sha256="a" * 64, applicability="included")
    _report_bindings(repo, [SECTION_NOT_APPLICABLE_BINDING])
    first = _compile_report(repo)
    write_json(repo / REPORT_BASE, first.document)

    second = _compile_report(repo)

    assert f"{second.name}.report" not in second.compile_lock["resolved_deltas"]
    assert second.document["params"]["sections"][0] == first.document["params"]["sections"][0]


def test_a_not_applicable_figure_entry_leaves_its_section_content_standing(
    repo: Path,
) -> None:
    """Only the bound node is reconciled; the section around it is not one."""
    _report_base_figure(repo, figure_spec_sha256="a" * 64, applicability="included")
    _report_bindings(repo, [NOT_APPLICABLE_BINDING])

    outcome = _compile_report(repo)

    section = outcome.document["params"]["sections"][0]
    assert "applicability" not in section
    assert [entry["applicability"] for entry in section["figures"]] == ["not_applicable"]


def test_a_role_path_the_report_content_model_does_not_know_is_refused(
    repo: Path,
) -> None:
    _report_bindings(
        repo,
        [{**SECTION_NOT_APPLICABLE_BINDING, "role_path": "params.chapters.0"}],
    )

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        _compile_report(repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert excinfo.value.field == "report.bindings[0].role_path"
    assert "OrderedFigureReportParams" in str(excinfo.value)


def test_a_role_path_below_a_known_node_is_refused_where_the_model_stops(
    repo: Path,
) -> None:
    _report_bindings(
        repo,
        [
            {
                **SECTION_NOT_APPLICABLE_BINDING,
                "role_path": "params.sections.0.figures.0.caption.text",
            }
        ],
    )

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        _compile_report(repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE


def test_a_role_path_is_unchecked_where_the_params_model_is_not_the_authority(
    repo: Path,
) -> None:
    """Nothing here reaches into params a recipe owns, not even to refuse a path."""
    from tests.fake_project_experiment import REPORT_BASE

    document = json.loads((repo / REPORT_BASE).read_text())
    document["report_type"] = "quillon.bulletin"
    write_json(repo / REPORT_BASE, document)
    _report_bindings(
        repo,
        [{**SECTION_NOT_APPLICABLE_BINDING, "role_path": "params.chapters.0"}],
    )

    outcome = _compile_report(repo)

    assert _not_applicable_reference(outcome)["role_path"] == "params.chapters.0"
    assert f"{outcome.name}.report" not in outcome.compile_lock["resolved_deltas"]


def test_a_not_applicable_section_that_still_tabulates_is_refused_not_emptied(
    repo: Path,
) -> None:
    """The engine removes the array it was ratified to remove and nothing else.

    An inherited scalar table is content the ordered-figure contract also forbids
    a not-applicable section, and this reconciliation does not remove it. The
    compiled params are therefore refused by the content model itself, which is
    the honest outcome: the engine never silently deletes authored tables.
    """
    from tests.fake_project_experiment import REPORT_BASE

    document = json.loads((repo / REPORT_BASE).read_text())
    document["params"]["sections"][0]["tables"] = [
        {"columns": ["arm"], "rows": [["near"]]}
    ]
    write_json(repo / REPORT_BASE, document)
    _report_bindings(repo, [SECTION_NOT_APPLICABLE_BINDING])

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        _compile_report(repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert "not-applicable section cannot declare figure or table content" in str(
        excinfo.value
    )


def test_a_params_node_the_contract_gives_no_applicability_stays_lock_only(
    repo: Path,
) -> None:
    """A scalar table is a node the model knows and states no applicability for."""
    from tests.fake_project_experiment import REPORT_BASE

    document = json.loads((repo / REPORT_BASE).read_text())
    document["params"]["sections"][0]["tables"] = [
        {"columns": ["arm"], "rows": [["near"]]}
    ]
    write_json(repo / REPORT_BASE, document)
    _report_bindings(
        repo,
        [{**SECTION_NOT_APPLICABLE_BINDING, "role_path": "params.sections.0.tables.0"}],
    )

    outcome = _compile_report(repo)

    assert outcome.document["params"]["sections"][0]["tables"][0]["columns"] == ["arm"]
    assert f"{outcome.name}.report" not in outcome.compile_lock["resolved_deltas"]


# -- a per-row figure input has no single locator --------------------------------


def test_a_per_row_input_without_a_reference_compiles(repo: Path) -> None:
    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-plot"), repo_root=repo
    )

    request = outcome.compile_lock["identity_contributions"]["figure_row_expansion"]
    assert request["inputs"]["observed"] == {"per_row": "observations"}
    assert request["role_contracts"][0]["input_role"] == "observed"


def test_the_lock_states_the_per_row_role_rather_than_a_false_locator(
    repo: Path,
) -> None:
    from feedbax.envelope.compile import PER_ROW_INPUT_RULE_ID

    outcome = kernel().compile_envelope_file(
        envelope_path(repo, "widened-plot"), repo_root=repo
    )

    references = outcome.compile_lock["references"]
    reference = next(item for item in references if item["kind"] == "not_applicable")
    assert reference["role_path"] == "inputs.observed"
    assert reference["basis"] == "compiler_rule"
    assert reference["rule_id"] == PER_ROW_INPUT_RULE_ID
    assert "per expanded row" in reference["reason"]
    assert not [item for item in references if item["kind"] == "planned_product"]
