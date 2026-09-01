"""Plan derivation: typed lock references in, plan nodes and edges out.

Everything here is stated over ``quillon``'s compiled outputs, emitted through
the production compile-lock emitter into a ``tmp_path`` directory. Four claims
are under test:

* **an edge is decided by two types and nothing else** — the reference kind and
  the consumer binding it carries. Each of the five reference kinds lands where
  its own meaning says it does, and a content pin lands nowhere;
* **a node's layer is the compiled document's own identity**, never a label an
  envelope carried, and a schema this build does not plan against refuses;
* **campaign authoring lowers to a typed training operation**, represented in
  the same finite graph as its downstream evaluation;
* **every pin a reference carries is checked**, so a stale reference or a
  document edited after its compile refuses instead of binding the wrong bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feedbax.workflow.derivation import (
    COMPILED_PRODUCT_KINDS,
    CompiledOutputError,
    DuplicateReferenceRoleError,
    UnresolvedPlannedProductError,
    UnsupportedCompiledProductError,
    derive_workflow_plan,
    read_compiled_outputs,
)
from feedbax.workflow.plan import (
    WORKFLOW_PLAN_SCHEMA_VERSION,
    LogicalKey,
    workflow_plan_from_document,
)
from feedbax.contracts.experiment_compile_lock import (
    AnalysisInputBinding,
    AnalysisReceiptSetBinding,
    AuthenticatedReceiptReference,
    CheckpointInitializationBinding,
    ContentPinReference,
    EvaluationSubjectBinding,
    NotApplicableReference,
    ReceiptLocatorReference,
    ReportParentBinding,
)
from feedbax.contracts.experiment_envelope import ExperimentEnvelopeRejection
from feedbax.contracts.manifest import (
    ANALYSIS_RUN_SPEC_SCHEMA_ID,
    EVALUATION_RUN_SPEC_SCHEMA_ID,
    REPORT_SPEC_SCHEMA_ID,
)

from tests.fake_project_experiment.products import QuillonOutputs, planned

DIGEST = "a" * 64


@pytest.fixture
def outputs(tmp_path: Path) -> QuillonOutputs:
    return QuillonOutputs(tmp_path / "repo")


def _plan(outputs: QuillonOutputs, target: str):
    return derive_workflow_plan(read_compiled_outputs(outputs.output_directory), target=target)


# --------------------------------------------------------------------------
# Nodes: the compiled document's own identity decides the layer
# --------------------------------------------------------------------------


def test_a_planned_product_reference_is_one_edge_to_one_node(outputs: QuillonOutputs) -> None:
    probe = outputs.probe("baseline")
    outputs.bulletin(
        "baseline-bulletin",
        references=[
            planned(
                probe,
                role_path="body.of",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="baseline"),
            )
        ],
    )

    plan = _plan(outputs, "baseline-bulletin")

    assert [node.key.text for node in plan.nodes] == [
        "evaluation:baseline",
        "report:baseline-bulletin",
    ]
    assert plan.target == LogicalKey("report", "baseline-bulletin")
    assert [node.operation.parameters["compiled_schema_id"] for node in plan.nodes] == [
        EVALUATION_RUN_SPEC_SCHEMA_ID,
        REPORT_SPEC_SCHEMA_ID,
    ]
    (edge,) = plan.edges
    assert edge.consumer == LogicalKey("report", "baseline-bulletin")
    assert edge.producer == LogicalKey("evaluation", "baseline")
    assert edge.role_path == ("body", "of")
    assert edge.input_type == EVALUATION_RUN_SPEC_SCHEMA_ID
    assert edge.producer_output == "primary"
    assert (edge.status, edge.basis, edge.external, edge.rule) == (
        "required",
        "authored",
        None,
        None,
    )


def test_the_plan_records_the_compiler_that_produced_the_locks(
    outputs: QuillonOutputs,
) -> None:
    outputs.probe("solo")
    plan = _plan(outputs, "solo")
    assert plan.origin["target_envelope_ref"] == "studies/solo.envelope.json"
    assert plan.origin["compiler_contract"] == {
        "contract_id": "quillon.compiler_contract",
        "contract_version": "quillon.compiler_contract.v1",
    }
    assert plan.origin["compiler_implementation"]["code_unit"] == (
        "tests.fake_project_experiment.products"
    )


def test_a_target_is_addressable_by_envelope_path_or_compiled_name(
    outputs: QuillonOutputs,
) -> None:
    product = outputs.probe("addressed")
    by_name = _plan(outputs, "addressed")
    by_path = _plan(outputs, product.envelope_ref)
    assert by_name.document() == by_path.document()


def test_a_derived_plan_round_trips_through_its_emitted_document(
    outputs: QuillonOutputs,
) -> None:
    probe = outputs.probe("round-trip")
    outputs.condensate(
        "round-trip-analysis",
        references=[
            planned(
                probe,
                role_path="inputs.states",
                consumer=AnalysisInputBinding(alias="baseline", role="states"),
            )
        ],
    )
    plan = _plan(outputs, "round-trip-analysis")
    document = plan.document()
    assert document["schema_version"] == WORKFLOW_PLAN_SCHEMA_VERSION
    assert workflow_plan_from_document(document).document() == document


def test_an_unplannable_compiled_schema_refuses_with_the_supported_set(
    outputs: QuillonOutputs,
) -> None:
    outputs.emit(
        "stranger",
        {"schema_id": "quillon.survey_document", "schema_version": "quillon.survey.v1"},
    )
    with pytest.raises(UnsupportedCompiledProductError) as caught:
        _plan(outputs, "stranger")
    assert caught.value.schema_id == "quillon.survey_document"
    assert str(sorted(COMPILED_PRODUCT_KINDS)) in str(caught.value)


# --------------------------------------------------------------------------
# Every reference kind lands where its own meaning says it does
# --------------------------------------------------------------------------


def test_a_content_pin_is_an_input_and_never_an_edge(outputs: QuillonOutputs) -> None:
    outputs.probe(
        "pinned",
        references=[ContentPinReference(ref="bases/probe.json", content_hash=DIGEST)],
    )
    plan = _plan(outputs, "pinned")
    assert plan.edges == ()
    assert [node.key.text for node in plan.nodes] == ["evaluation:pinned"]


def test_a_receipt_locator_is_an_external_edge_with_no_digest(
    outputs: QuillonOutputs,
) -> None:
    outputs.condensate(
        "located",
        references=[
            ReceiptLocatorReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id="feedbax-evaluation-run:earlier",
                role_path="inputs.prior",
                consumer=AnalysisInputBinding(alias="prior", role="states"),
            )
        ],
    )
    (edge,) = _plan(outputs, "located").edges
    assert edge.producer is None
    assert edge.external == {
        "manifest_kind": "EvaluationRunManifest",
        "manifest_id": "feedbax-evaluation-run:earlier",
    }
    assert edge.status == "required"


def test_an_authenticated_receipt_carries_the_byte_profile_it_quoted(
    outputs: QuillonOutputs,
) -> None:
    outputs.condensate(
        "authenticated",
        references=[
            AuthenticatedReceiptReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id="feedbax-evaluation-run:earlier",
                manifest_sha256=DIGEST,
                size_bytes=1234,
                role_path="inputs.prior",
                consumer=AnalysisInputBinding(alias="prior", role="states"),
                execution_uri="manifests/earlier.json",
            )
        ],
    )
    (edge,) = _plan(outputs, "authenticated").edges
    assert edge.external == {
        "manifest_kind": "EvaluationRunManifest",
        "manifest_id": "feedbax-evaluation-run:earlier",
        "manifest_sha256": DIGEST,
        "size_bytes": 1234,
        "execution_uri": "manifests/earlier.json",
    }


@pytest.mark.parametrize(
    ("basis", "rule_id"),
    [("authored", None), ("compiler_rule", "feedbax.rule.unbound_role.v1")],
)
def test_a_not_applicable_reference_binds_nothing_and_quotes_its_basis(
    outputs: QuillonOutputs, basis: str, rule_id: str | None
) -> None:
    outputs.bulletin(
        f"omitting-{basis}",
        references=[
            NotApplicableReference(
                role_path="sections.appendix",
                basis=basis,  # type: ignore[arg-type]
                reason="this target has no appendix",
                rule_id=rule_id,
            )
        ],
    )
    (edge,) = _plan(outputs, f"omitting-{basis}").edges
    assert (edge.status, edge.basis, edge.rule) == ("not_applicable", basis, rule_id)
    assert edge.producer is None and edge.external is None
    assert edge.reason == "this target has no appendix"


def test_receipt_set_binding_is_derived_only_for_a_set_valued_product(
    outputs: QuillonOutputs,
) -> None:
    matrix = outputs.probe_matrix("matrix-set")
    outputs.condensate(
        "matrix-consumer",
        references=[
            planned(
                matrix,
                role_path="inputs.evaluation",
                consumer=AnalysisReceiptSetBinding(
                    alias="evaluation", role="evaluation"
                ),
            )
        ],
    )
    (edge,) = derive_workflow_plan(
        read_compiled_outputs(outputs.output_directory), target="matrix-consumer"
    ).required_edges(LogicalKey("analysis", "matrix-consumer"))
    assert edge.binding == "complete_receipt_set"

    single = outputs.probe("single-product")
    outputs.condensate(
        "invalid-set-consumer",
        references=[
            planned(
                single,
                role_path="inputs.evaluation",
                consumer=AnalysisReceiptSetBinding(
                    alias="evaluation", role="evaluation"
                ),
            )
        ],
    )
    with pytest.raises(CompiledOutputError, match="requires a set-valued product"):
        derive_workflow_plan(
            read_compiled_outputs(outputs.output_directory),
            target="invalid-set-consumer",
        )


def test_two_references_at_one_role_refuse(outputs: QuillonOutputs) -> None:
    probe = outputs.probe("shared-role")
    outputs.bulletin(
        "double-bound",
        references=[
            planned(
                probe,
                role_path="body.of",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="shared-role"),
            ),
            ReceiptLocatorReference(
                manifest_kind="EvaluationRunManifest",
                manifest_id="feedbax-evaluation-run:other",
                role_path="body.of",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="other"),
            ),
        ],
    )
    with pytest.raises(DuplicateReferenceRoleError) as caught:
        _plan(outputs, "double-bound")
    assert caught.value.role_path == "body.of"


# --------------------------------------------------------------------------
# Campaign authoring lowers to the same finite workflow
# --------------------------------------------------------------------------


def test_a_compiled_training_matrix_is_a_typed_campaign_operation(
    outputs: QuillonOutputs,
) -> None:
    cohort = outputs.cohort("sweep")
    outputs.probe(
        "sweep-probe",
        references=[
            planned(
                cohort,
                role_path="subject.cohort",
                consumer=EvaluationSubjectBinding(subject_id="sweep"),
            )
        ],
    )
    plan = _plan(outputs, "sweep-probe")
    campaign = plan.nodes[0]
    assert campaign.key == LogicalKey("campaign", "sweep")
    assert campaign.operation.type_id == "feedbax.operation.train"
    assert campaign.operation.effect == "external"
    assert campaign.operation.capabilities == ("training",)
    assert plan.descendants(campaign.key) == ("evaluation:sweep-probe",)


def test_a_checkpoint_initialization_binding_is_carried_as_an_ordinary_edge(
    outputs: QuillonOutputs,
) -> None:
    """Derivation never interprets a binding; only lowering does."""
    outputs.cohort(
        "continued",
        references=[
            AuthenticatedReceiptReference(
                manifest_kind="TrainingRunManifest",
                manifest_id="feedbax-training-run:earlier",
                manifest_sha256=DIGEST,
                size_bytes=17,
                role_path="rows.continued.checkpoint",
                consumer=CheckpointInitializationBinding(mode="continue_from", row_id="continued"),
            )
        ],
    )
    (edge,) = _plan(outputs, "continued").edges
    assert edge.role_path == ("rows", "continued", "checkpoint")
    assert edge.external is not None and edge.external["manifest_kind"] == "TrainingRunManifest"


# --------------------------------------------------------------------------
# Pins are checked, never trusted
# --------------------------------------------------------------------------


def test_an_upstream_no_compile_lock_reaches_refuses(outputs: QuillonOutputs) -> None:
    probe = outputs.probe("vanishing")
    outputs.bulletin(
        "orphaned",
        references=[
            planned(
                probe,
                role_path="body.of",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="vanishing"),
            )
        ],
    )
    probe.lock_path.unlink()
    probe.document_path.unlink()
    with pytest.raises(UnresolvedPlannedProductError) as caught:
        _plan(outputs, "orphaned")
    assert caught.value.reference.product_name == "vanishing"


def test_a_reference_whose_content_pin_went_stale_refuses(outputs: QuillonOutputs) -> None:
    probe = outputs.probe("moving")
    outputs.bulletin(
        "stale-pin",
        references=[
            planned(
                probe,
                role_path="body.of",
                consumer=ReportParentBinding(parent_kind="probe", parent_id="moving"),
            )
        ],
    )
    outputs.probe("moving", stage_note="the probe was re-authored")
    with pytest.raises(CompiledOutputError, match="compiled_content_hash"):
        _plan(outputs, "stale-pin")


def test_a_compiled_document_edited_after_its_compile_refuses_at_read(
    outputs: QuillonOutputs,
) -> None:
    product = outputs.probe("tampered")
    document = json.loads(product.document_path.read_text(encoding="utf-8"))
    document["params"]["stage"] = "moved"
    product.document_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(CompiledOutputError, match="changed after the compile"):
        read_compiled_outputs(outputs.output_directory)


def test_a_lock_whose_pinned_document_is_absent_refuses(outputs: QuillonOutputs) -> None:
    product = outputs.probe("half-emitted")
    product.document_path.unlink()
    with pytest.raises(CompiledOutputError, match="emitted together"):
        read_compiled_outputs(outputs.output_directory)


def test_an_unsupported_lock_version_fails_closed_with_no_migration(
    outputs: QuillonOutputs,
) -> None:
    product = outputs.probe("future")
    lock = json.loads(product.lock_path.read_text(encoding="utf-8"))
    lock["schema_version"] = "feedbax.spec.experiment_compile_lock.v99"
    product.lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    with pytest.raises(ExperimentEnvelopeRejection, match="migration_intentionally_absent"):
        read_compiled_outputs(outputs.output_directory)


def test_a_target_that_names_nothing_refuses_with_what_is_known(
    outputs: QuillonOutputs,
) -> None:
    outputs.probe("known")
    with pytest.raises(CompiledOutputError, match="known"):
        _plan(outputs, "unknown")


def test_a_diamond_contributes_one_node_and_two_edges(outputs: QuillonOutputs) -> None:
    probe = outputs.probe("shared")
    left = outputs.condensate(
        "left",
        references=[
            planned(
                probe,
                role_path="inputs.states",
                consumer=AnalysisInputBinding(alias="shared", role="states"),
            )
        ],
    )
    right = outputs.condensate(
        "right",
        references=[
            planned(
                probe,
                role_path="inputs.states",
                consumer=AnalysisInputBinding(alias="shared", role="states"),
            )
        ],
    )
    outputs.bulletin(
        "joined",
        references=[
            planned(
                left,
                role_path="sections.left",
                consumer=ReportParentBinding(
                    parent_kind=ANALYSIS_RUN_SPEC_SCHEMA_ID, parent_id="left"
                ),
            ),
            planned(
                right,
                role_path="sections.right",
                consumer=ReportParentBinding(
                    parent_kind=ANALYSIS_RUN_SPEC_SCHEMA_ID, parent_id="right"
                ),
            ),
        ],
    )
    plan = _plan(outputs, "joined")
    assert [node.key.text for node in plan.nodes] == [
        "evaluation:shared",
        "analysis:left",
        "analysis:right",
        "report:joined",
    ]
    assert len(plan.edges) == 4
