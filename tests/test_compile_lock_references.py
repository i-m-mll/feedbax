"""The compile lock's closed typed reference union.

A lock's ``references`` block is the only thing the plan lane reads to decide
what has to run. These tests pin the two properties that makes it safe to read
mechanically: the union is closed, and every kind carries exactly the facts its
question needs — no digest on a locator, no fabricated receipt, no free-form
mapping smuggled through either the writer or the reader.

The vocabulary here is invented (``quillon``) for the same reason the engine
fixtures use it: the contract is what is under test, not anybody's science.
"""

from __future__ import annotations

from typing import Any

import pytest

from feedbax.contracts.authored_canonical import CANONICAL_PIN_ALGORITHM
from feedbax.contracts.experiment_compile_lock import (
    COMPILE_LOCK_PLAN_EDGE_KINDS,
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1,
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V2,
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V3,
    COMPILE_LOCK_REFERENCE_KINDS,
    AnalysisInputBinding,
    AnalysisReceiptSetBinding,
    AuthenticatedReceiptReference,
    CheckpointInitializationBinding,
    CompileLockInputs,
    CompilerContract,
    CompilerImplementation,
    ContentPinReference,
    EvaluationSubjectBinding,
    FigureRuntimeInputBinding,
    NotApplicableReference,
    PlannedProductReference,
    ReceiptLocatorReference,
    ReportParentBinding,
    RowProvenanceReference,
    build_compile_lock,
    compile_lock_plan_edges,
    load_compile_lock,
    parse_compile_lock_reference,
    parse_row_provenance_reference,
)
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def _content_pin() -> ContentPinReference:
    return ContentPinReference(ref="bases/baseline.survey.json", content_hash=DIGEST_A)


def _planned_product() -> PlannedProductReference:
    return PlannedProductReference(
        envelope_ref="studies/widened.envelope.json",
        envelope_hash=DIGEST_A,
        product_name="widened",
        product_schema_id="quillon.survey_document",
        product_schema_version="quillon.survey_document.v1",
        compiled_content_hash=DIGEST_B,
        role_path="subject.survey",
        consumer=EvaluationSubjectBinding(subject_id="widened"),
    )


def _receipt_locator() -> ReceiptLocatorReference:
    return ReceiptLocatorReference(
        manifest_kind="quillon.survey_run",
        manifest_id="widened-0",
        role_path="inputs.baseline",
        consumer=AnalysisInputBinding(alias="baseline", role="observations"),
    )


def _authenticated_receipt() -> AuthenticatedReceiptReference:
    return AuthenticatedReceiptReference(
        manifest_kind="quillon.survey_run",
        manifest_id="widened-0",
        manifest_sha256=DIGEST_C,
        size_bytes=2048,
        role_path="inputs.observed",
        consumer=FigureRuntimeInputBinding(input_role="observed"),
    )


def _not_applicable() -> NotApplicableReference:
    return NotApplicableReference(
        role_path="sections.summary.figure",
        basis="authored",
        reason="the widened survey has no comparison arm to plot",
    )


def _row_provenance() -> RowProvenanceReference:
    return RowProvenanceReference(
        row_id="widened",
        source_row_key="baseline",
        source_ref="bases/baseline.survey.json",
        source_content_hash=DIGEST_A,
    )


def _inputs(
    references: list[Any], row_provenance: list[Any] | None = None
) -> CompileLockInputs:
    return CompileLockInputs(
        envelope_ref="studies/widened.envelope.json",
        envelope_document={"schema": "quillon.study.v1", "name": "widened"},
        envelope_schema="quillon.study.v1",
        name="widened",
        family="quillon.survey_document",
        compiled_document={"schema_id": "quillon.survey_document", "name": "widened"},
        contract=CompilerContract("quillon.compiler_contract", "quillon.compiler_contract.v1"),
        implementation=CompilerImplementation(code_unit="tests.test_compile_lock_references"),
        references=references,
        row_provenance=row_provenance or [],
    )


# -- the union is closed and complete ------------------------------------


def test_reference_kinds_are_exactly_the_five_declared_members() -> None:
    assert COMPILE_LOCK_REFERENCE_KINDS == (
        "content_pin",
        "planned_product",
        "receipt_locator",
        "authenticated_receipt",
        "not_applicable",
    )


def test_every_member_round_trips_through_a_lock() -> None:
    references = [
        _content_pin(),
        _planned_product(),
        _receipt_locator(),
        _authenticated_receipt(),
        _not_applicable(),
    ]
    lock = build_compile_lock(_inputs(references))
    stored = lock["references"]
    assert [record["kind"] for record in stored] == list(COMPILE_LOCK_REFERENCE_KINDS)
    reloaded = load_compile_lock(lock, field="widened.compile-lock.json")
    assert reloaded["references"] == stored


def test_a_mapping_form_reference_is_normalized_and_still_validated() -> None:
    lock = build_compile_lock(_inputs([_content_pin().model_dump(mode="json")]))
    assert lock["references"] == [
        {
            "kind": "content_pin",
            "ref": "bases/baseline.survey.json",
            "content_hash": DIGEST_A,
            "pin_algorithm": CANONICAL_PIN_ALGORITHM,
        }
    ]


def test_an_unknown_reference_kind_is_refused_by_the_writer() -> None:
    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        build_compile_lock(_inputs([{"kind": "envelope", "alias": "widened"}]))
    assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert caught.value.field == "references[0]"


def test_a_free_form_reference_is_refused_by_the_reader() -> None:
    lock = build_compile_lock(_inputs([]))
    lock["references"] = [{"kind": "envelope", "alias": "widened"}]
    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        load_compile_lock(lock, field="widened.compile-lock.json")
    assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert caught.value.field == "widened.compile-lock.json#references[0]"


def test_a_references_block_that_is_not_a_list_is_refused_by_the_reader() -> None:
    lock = build_compile_lock(_inputs([]))
    lock["references"] = {"kind": "content_pin"}
    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        load_compile_lock(lock, field="widened.compile-lock.json")
    assert caught.value.field == "widened.compile-lock.json#references"


# -- content pins are inputs, not plan edges ------------------------------


def test_a_content_pin_is_not_a_plan_edge_and_every_other_kind_is() -> None:
    assert "content_pin" not in COMPILE_LOCK_PLAN_EDGE_KINDS
    assert COMPILE_LOCK_PLAN_EDGE_KINDS == frozenset(COMPILE_LOCK_REFERENCE_KINDS[1:])


def test_plan_edges_skip_content_pins_in_declaration_order() -> None:
    lock = build_compile_lock(
        _inputs([_content_pin(), _planned_product(), _content_pin(), _receipt_locator()])
    )
    edges = compile_lock_plan_edges(lock, field="widened.compile-lock.json")
    assert [edge.kind for edge in edges] == ["planned_product", "receipt_locator"]


def test_a_content_pin_states_no_consumer() -> None:
    with pytest.raises(ExperimentEnvelopeRejection):
        parse_compile_lock_reference(
            {
                "kind": "content_pin",
                "ref": "bases/baseline.survey.json",
                "content_hash": DIGEST_A,
                "consumer": {"consumer": "evaluation_subject", "subject_id": "widened"},
            },
            field="references[0]",
        )


# -- the locator carries no digest, and the receipt carries one -----------


def test_a_receipt_locator_may_not_carry_a_digest() -> None:
    with pytest.raises(ExperimentEnvelopeRejection):
        parse_compile_lock_reference(
            {
                **_receipt_locator().model_dump(mode="json"),
                "manifest_sha256": DIGEST_C,
            },
            field="references[0]",
        )


def test_a_receipt_locator_survives_with_no_digest_at_all() -> None:
    record = _receipt_locator().model_dump(mode="json", exclude_none=True)
    assert "manifest_sha256" not in record
    assert "size_bytes" not in record


def test_an_authenticated_receipt_requires_a_digest_and_a_size() -> None:
    partial = _authenticated_receipt().model_dump(mode="json", exclude_none=True)
    del partial["size_bytes"]
    with pytest.raises(ExperimentEnvelopeRejection):
        parse_compile_lock_reference(partial, field="references[0]")


def test_an_authenticated_receipt_refuses_a_non_sha256_digest() -> None:
    record = _authenticated_receipt().model_dump(mode="json", exclude_none=True)
    record["manifest_sha256"] = "DEADBEEF"
    with pytest.raises(ExperimentEnvelopeRejection):
        parse_compile_lock_reference(record, field="references[0]")


def test_an_authenticated_receipt_may_state_where_it_was_produced() -> None:
    reference = AuthenticatedReceiptReference(
        manifest_kind="quillon.survey_run",
        manifest_id="widened-0",
        manifest_sha256=DIGEST_C,
        size_bytes=0,
        role_path="inputs.observed",
        consumer=FigureRuntimeInputBinding(input_role="observed"),
        execution_uri="file:///custody/widened-0",
    )
    assert reference.execution_uri == "file:///custody/widened-0"


# -- authored not-applicability vs a versioned structural rule ------------


def test_authored_not_applicability_states_no_rule() -> None:
    with pytest.raises(ExperimentEnvelopeRejection):
        parse_compile_lock_reference(
            {
                "kind": "not_applicable",
                "role_path": "sections.summary.figure",
                "basis": "authored",
                "reason": "no comparison arm",
                "rule_id": "feedbax.experiment_envelope.rule.no_parent.v1",
            },
            field="references[0]",
        )


def test_compiler_rule_not_applicability_must_name_a_versioned_rule() -> None:
    with pytest.raises(ExperimentEnvelopeRejection):
        parse_compile_lock_reference(
            {
                "kind": "not_applicable",
                "role_path": "sections.summary.figure",
                "basis": "compiler_rule",
                "reason": "the layer has no parent",
            },
            field="references[0]",
        )
    with pytest.raises(ExperimentEnvelopeRejection):
        parse_compile_lock_reference(
            {
                "kind": "not_applicable",
                "role_path": "sections.summary.figure",
                "basis": "compiler_rule",
                "reason": "the layer has no parent",
                "rule_id": "feedbax.experiment_envelope.rule.no_parent",
            },
            field="references[0]",
        )
    accepted = parse_compile_lock_reference(
        {
            "kind": "not_applicable",
            "role_path": "sections.summary.figure",
            "basis": "compiler_rule",
            "reason": "the layer has no parent",
            "rule_id": "feedbax.experiment_envelope.rule.no_parent.v1",
        },
        field="references[0]",
    )
    assert accepted.rule_id == "feedbax.experiment_envelope.rule.no_parent.v1"


def test_a_third_applicability_basis_does_not_exist() -> None:
    with pytest.raises(ExperimentEnvelopeRejection):
        parse_compile_lock_reference(
            {
                "kind": "not_applicable",
                "role_path": "sections.summary.figure",
                "basis": "project_callback",
                "reason": "the project decided",
            },
            field="references[0]",
        )


# -- the consumer binding union ------------------------------------------


@pytest.mark.parametrize(
    "binding",
    [
        EvaluationSubjectBinding(subject_id="widened"),
        AnalysisInputBinding(alias="baseline", role="observations"),
        FigureRuntimeInputBinding(input_role="observed"),
        ReportParentBinding(parent_kind="figure", parent_id="widened-span"),
        CheckpointInitializationBinding(mode="continue_from", row_id="widened-0"),
    ],
)
def test_every_consumer_binding_is_accepted_on_a_planned_product(binding: Any) -> None:
    reference = _planned_product().model_copy(update={"consumer": binding})
    assert parse_compile_lock_reference(
        reference.model_dump(mode="json"), field="references[0]"
    ).consumer == binding


def test_an_unknown_consumer_is_refused() -> None:
    record = _planned_product().model_dump(mode="json")
    record["consumer"] = {"consumer": "project_specific_thing", "id": "x"}
    with pytest.raises(ExperimentEnvelopeRejection):
        parse_compile_lock_reference(record, field="references[0]")


def test_a_checkpoint_initialization_binding_uses_the_feedbax_mode_vocabulary() -> None:
    record = _planned_product().model_dump(mode="json")
    record["consumer"] = {"consumer": "checkpoint_initialization", "mode": "warm", "row_id": "r"}
    with pytest.raises(ExperimentEnvelopeRejection):
        parse_compile_lock_reference(record, field="references[0]")


# -- role paths -----------------------------------------------------------


@pytest.mark.parametrize("role_path", ["", "  ", "inputs..baseline", "inputs."])
def test_a_role_path_must_be_a_nonempty_dotted_path(role_path: str) -> None:
    record = _receipt_locator().model_dump(mode="json", exclude_none=True)
    record["role_path"] = role_path
    with pytest.raises(ExperimentEnvelopeRejection):
        parse_compile_lock_reference(record, field="references[0]")


def test_a_planned_product_schema_version_must_extend_its_schema_id() -> None:
    record = _planned_product().model_dump(mode="json")
    record["product_schema_version"] = "quillon.digest_document.v1"
    with pytest.raises(ExperimentEnvelopeRejection):
        parse_compile_lock_reference(record, field="references[0]")


# -- row provenance is its own block, beside the closed union ---------------


def test_row_provenance_states_the_row_key_and_the_pinned_document_it_resolved_in() -> None:
    lock = build_compile_lock(_inputs([], [_row_provenance()]))

    assert lock["row_provenance"] == [
        {
            "row_id": "widened",
            "source_row_key": "baseline",
            "source_ref": "bases/baseline.survey.json",
            "source_content_hash": DIGEST_A,
            "pin_algorithm": CANONICAL_PIN_ALGORITHM,
        }
    ]
    assert load_compile_lock(lock, field="widened.compile-lock.json")["row_provenance"] == (
        lock["row_provenance"]
    )


def test_a_lock_that_derived_no_row_states_an_empty_row_provenance_block() -> None:
    assert build_compile_lock(_inputs([]))["row_provenance"] == []


def test_row_provenance_accepts_the_mapping_form_a_tracked_lock_holds() -> None:
    record = _row_provenance().model_dump(mode="json")

    assert build_compile_lock(_inputs([], [record]))["row_provenance"] == [record]


def test_row_provenance_is_not_a_member_of_the_reference_union() -> None:
    assert "row_provenance" not in COMPILE_LOCK_REFERENCE_KINDS
    with pytest.raises(ExperimentEnvelopeRejection):
        parse_compile_lock_reference(
            _row_provenance().model_dump(mode="json"), field="references[0]"
        )


def test_row_provenance_is_not_a_plan_edge() -> None:
    lock = build_compile_lock(_inputs([_planned_product()], [_row_provenance()]))

    edges = compile_lock_plan_edges(lock, field="widened.compile-lock.json")

    assert [edge.kind for edge in edges] == ["planned_product"]


@pytest.mark.parametrize(
    "change",
    [
        {"row_id": ""},
        {"source_row_key": "  "},
        {"source_ref": ""},
        {"source_content_hash": "DEADBEEF"},
        {"pin_algorithm": "md5"},
        {"kind": "row_provenance"},
        {"consumer": {"consumer": "evaluation_subject", "subject_id": "widened"}},
    ],
)
def test_a_malformed_row_provenance_record_is_refused_by_the_writer(
    change: dict[str, Any],
) -> None:
    record = {**_row_provenance().model_dump(mode="json"), **change}

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        build_compile_lock(_inputs([], [record]))

    assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert caught.value.field == "row_provenance[0]"


def test_a_row_provenance_record_edited_after_the_fact_is_refused_by_the_reader() -> None:
    lock = build_compile_lock(_inputs([], [_row_provenance()]))
    lock["row_provenance"] = [{"row_id": "widened", "source_row_key": "baseline"}]

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        load_compile_lock(lock, field="widened.compile-lock.json")

    assert caught.value.field == "widened.compile-lock.json#row_provenance[0]"


def test_a_row_provenance_block_that_is_not_a_list_is_refused_by_the_reader() -> None:
    lock = build_compile_lock(_inputs([]))
    lock["row_provenance"] = {"row_id": "widened"}

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        load_compile_lock(lock, field="widened.compile-lock.json")

    assert caught.value.field == "widened.compile-lock.json#row_provenance"


def test_an_already_typed_row_provenance_record_passes_through_unchanged() -> None:
    record = _row_provenance()

    assert parse_row_provenance_reference(record, field="row_provenance[0]") is record


# -- the lock's own version boundary, at the figure input contract --------


def _figure_contract() -> dict[str, Any]:
    return {
        "input_role": "observed",
        "artifact_role": "result",
        "artifact_provider": "quillon.custody",
        "payload_name": "observed",
        "payload_schema_id": "quillon.span_result",
        "payload_schema_version": "quillon.span_result.v1",
    }


def _lock_with_figure_input(contract: dict[str, Any] | None) -> dict[str, Any]:
    consumer = FigureRuntimeInputBinding(input_role="observed", contract=contract)
    reference = _authenticated_receipt().model_copy(update={"consumer": consumer})
    return build_compile_lock(_inputs([reference]))


def test_a_current_lock_carries_the_typed_figure_input_contract() -> None:
    lock = _lock_with_figure_input(_figure_contract())

    assert lock["schema_version"] == EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V3
    loaded = load_compile_lock(lock, field="lock")
    consumer = parse_compile_lock_reference(
        loaded["references"][0], field="lock#references[0]"
    ).consumer
    assert isinstance(consumer, FigureRuntimeInputBinding)
    assert consumer.contract is not None
    assert consumer.contract.artifact_role == "result"
    assert consumer.contract.payload_name == "observed"
    assert consumer.contract.payload_schema_version == "quillon.span_result.v1"


def test_a_prior_lock_remains_readable_as_the_grammar_it_names() -> None:
    """v1 is read as v1: a lock without contracts still describes a real compile."""
    lock = {
        **_lock_with_figure_input(None),
        "schema_version": EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1,
    }

    loaded = load_compile_lock(lock, field="lock")

    assert loaded["schema_version"] == EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1


def test_a_prior_lock_stating_a_contract_is_refused_by_version() -> None:
    lock = {
        **_lock_with_figure_input(_figure_contract()),
        "schema_version": EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1,
    }

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        load_compile_lock(lock, field="lock")

    assert caught.value.category is (ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION)
    assert EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V2 in str(caught.value)


@pytest.mark.parametrize(
    "version",
    [EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1, EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V2],
)
def test_a_pre_v3_lock_cannot_be_relabelled_as_a_receipt_set(version: str) -> None:
    reference = _authenticated_receipt().model_copy(
        update={
            "consumer": AnalysisReceiptSetBinding(alias="evaluation", role="evaluation")
        }
    )
    lock = {**build_compile_lock(_inputs([reference])), "schema_version": version}
    with pytest.raises(ExperimentEnvelopeRejection, match="receipt-set binding"):
        load_compile_lock(lock, field="lock")


def test_a_figure_input_contract_names_the_role_its_binding_addresses() -> None:
    with pytest.raises(ValueError, match="one binding states one role"):
        FigureRuntimeInputBinding(
            input_role="observed",
            contract={**_figure_contract(), "input_role": "elsewhere"},
        )
