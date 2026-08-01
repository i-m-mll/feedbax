"""The closed ``feedbax.experiment_envelope.v1`` dialect.

Two properties are under test and nothing else. First, the dialect is *closed*:
one schema string, one layer per envelope, no unknown fields anywhere, and no
slot through which a project could widen it. Second, the vocabulary inside it is
*open*: dotted paths, values, recipe ids, and role strings are carried as data
and judged by the final Feedbax output model, not by the dialect.

The invented ``quillon`` vocabulary is used throughout for the same reason it is
used in the kernel tests: if a case needed a real project's words, it would be
testing the wrong thing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
)
from feedbax.contracts.experiment_envelope_dialect import (
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION,
    EXPERIMENT_ENVELOPE_FAMILY,
    EXPERIMENT_ENVELOPE_MIGRATION_TABLE,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
    EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS,
    LAYER_OUTPUT_CONTRACTS,
    ExperimentEnvelopeLayer,
    ReceiptReference,
    layer_of_document,
    output_contract_of_document,
    parse_experiment_envelope,
)
from feedbax.envelope import kernel_for
from feedbax.envelope.compile import authored_layer_of

from tests.fake_project_experiment import (
    ANALYSIS_ENVELOPE,
    EVALUATION_ENVELOPE,
    FIGURE_ENVELOPE,
    PROJECT_DECLARATION,
    REPORT_ENVELOPE,
    TRAINING_ENVELOPE,
    envelope_path,
    write_repo,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    write_repo(tmp_path)
    return tmp_path


def _parse(document: dict[str, Any]) -> Any:
    return parse_experiment_envelope(document, field="studies/probe.envelope.json")


def _minimal(**layer: Any) -> dict[str, Any]:
    return {
        "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
        "name": "probe",
        "base": "bases/baseline.training_run_matrix.json",
        **layer,
    }


# -- the family and its version boundary ----------------------------------


def test_the_dialect_is_one_family_at_one_version() -> None:
    assert EXPERIMENT_ENVELOPE_FAMILY == "feedbax.experiment_envelope"
    assert EXPERIMENT_ENVELOPE_SCHEMA_VERSION == "feedbax.experiment_envelope.v1"
    assert EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS == (
        EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
    )
    assert EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION == (
        "feedbax.experiment_envelope.compiler.v1"
    )


@pytest.mark.parametrize(
    "schema",
    ["feedbax.experiment_envelope.v0", "feedbax.experiment_envelope.v2", "quillon.study.v1", None],
)
def test_any_other_schema_fails_closed_naming_its_migration_slot(schema: Any) -> None:
    document = _minimal(training={"tags": {"add": ["probe"]}})
    if schema is None:
        document.pop("schema")
    else:
        document["schema"] = schema

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _parse(document)

    assert caught.value.category is (
        ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION
    )
    message = str(caught.value)
    assert str(EXPERIMENT_ENVELOPE_MIGRATION_TABLE) in message
    assert "migration_intentionally_absent=yes" in message


# -- exactly one layer ------------------------------------------------------


@pytest.mark.parametrize(
    "document",
    [TRAINING_ENVELOPE, EVALUATION_ENVELOPE, ANALYSIS_ENVELOPE, FIGURE_ENVELOPE, REPORT_ENVELOPE],
)
def test_each_corpus_envelope_authors_exactly_one_layer(document: dict[str, Any]) -> None:
    envelope = _parse(dict(document))

    authored = [
        layer for layer in ExperimentEnvelopeLayer if envelope.layer_of(layer) is not None
    ]
    assert authored == [envelope.layer]
    assert envelope.content is envelope.layer_of(envelope.layer)


def test_an_envelope_that_authors_no_layer_is_refused() -> None:
    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _parse(_minimal())

    assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert "exactly one layer" in str(caught.value)


def test_an_envelope_that_authors_two_layers_is_refused() -> None:
    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _parse(
            _minimal(
                training={"tags": {"add": ["probe"]}},
                report={"delta": {"layer_id": "one", "patches": []}},
            )
        )

    assert "exactly one layer" in str(caught.value)


def test_the_layer_set_is_the_five_feedbax_artifact_families() -> None:
    assert [layer.value for layer in ExperimentEnvelopeLayer] == [
        "training",
        "evaluation",
        "analysis",
        "figure",
        "report",
    ]
    assert {contract.layer for contract in LAYER_OUTPUT_CONTRACTS.values()} == set(
        ExperimentEnvelopeLayer
    )


def test_the_pre_parse_layer_probe_agrees_with_the_parsed_model() -> None:
    for document in (TRAINING_ENVELOPE, REPORT_ENVELOPE):
        assert authored_layer_of(document) == _parse(dict(document)).layer.value


def test_the_pre_parse_layer_probe_declines_an_ambiguous_document() -> None:
    assert authored_layer_of({"training": {}, "report": {}}) is None
    assert authored_layer_of({}) is None


# -- closed models ----------------------------------------------------------


@pytest.mark.parametrize(
    "document",
    [
        {**_minimal(training={"tags": {"add": ["p"]}}), "invented_top_level": 1},
        _minimal(training={"tags": {"add": ["p"]}, "invented": 1}),
        _minimal(training={"rows": [{"from": "baseline", "id": "x", "invented": 1}]}),
        _minimal(
            report={
                "bindings": [
                    {
                        "role_path": "a.b",
                        "ref": {"kind": "envelope", "alias": "x"},
                        "extra": 1,
                    }
                ]
            }
        ),
        _minimal(
            figure={
                "inputs": [
                    {
                        "input_role": "observed",
                        "ref": {"kind": "envelope", "alias": "x", "extra": 1},
                    }
                ]
            }
        ),
    ],
)
def test_an_unknown_field_is_refused_anywhere_in_the_document(document: dict[str, Any]) -> None:
    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _parse(document)

    assert caught.value.category is ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD


def test_a_missing_required_field_is_refused_as_missing() -> None:
    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _parse(_minimal(training={"rows": [{"id": "widened"}]}))

    assert caught.value.category is ExperimentEnvelopeRejectionCategory.MISSING_FIELD


def test_an_assertion_states_the_value_it_expects() -> None:
    with pytest.raises(ExperimentEnvelopeRejection, match="states the value it expects"):
        _parse(
            {
                **_minimal(training={"tags": {"add": ["probe"]}}),
                "assert": [{"path": "base.inline.cadence"}],
            }
        )


def test_an_assertion_may_expect_a_null_it_states_explicitly() -> None:
    envelope = _parse(
        {
            **_minimal(training={"tags": {"add": ["probe"]}}),
            "assert": [{"path": "base.inline.cadence", "equals": None}],
        }
    )

    assert envelope.assert_[0].equals is None


def test_a_tags_delta_must_change_something_and_may_not_contradict_itself() -> None:
    with pytest.raises(ExperimentEnvelopeRejection, match="adds or removes something"):
        _parse(_minimal(training={"tags": {"add": [], "remove": []}}))
    with pytest.raises(ExperimentEnvelopeRejection, match="both added and removed"):
        _parse(_minimal(training={"tags": {"add": ["p"], "remove": ["p"]}}))


def test_training_row_ids_and_delta_layer_ids_are_unique_within_one_envelope() -> None:
    row = {"from": "baseline", "id": "widened"}
    with pytest.raises(ExperimentEnvelopeRejection, match="row ids must be unique"):
        _parse(_minimal(training={"rows": [dict(row), dict(row)]}))
    with pytest.raises(ExperimentEnvelopeRejection, match="layer ids must be unique"):
        _parse(
            _minimal(
                training={
                    "rows": [
                        {**row, "delta": {"layer_id": "one", "patches": []}},
                        {**row, "id": "other", "delta": {"layer_id": "one", "patches": []}},
                    ]
                }
            )
        )


def test_a_report_binds_each_role_path_at_most_once() -> None:
    binding = {"role_path": "sections.0.figures.0", "ref": {"kind": "envelope", "alias": "x"}}
    with pytest.raises(ExperimentEnvelopeRejection, match="at most once"):
        _parse(_minimal(report={"bindings": [dict(binding), dict(binding)]}))


def test_an_analysis_bundle_does_not_take_run_shaped_fields() -> None:
    with pytest.raises(ExperimentEnvelopeRejection, match="belong to a single analysis run"):
        _parse(_minimal(analysis={"target": "bundle", "recipe": "quillon.span_summary"}))


# -- authored references are typed and closed --------------------------------


def test_a_receipt_states_a_digest_and_a_size_together_or_neither() -> None:
    ReceiptReference(manifest_kind="quillon.probe_run", manifest_id="p-0")
    ReceiptReference(
        manifest_kind="quillon.probe_run",
        manifest_id="p-0",
        manifest_sha256="a" * 64,
        size_bytes=1,
    )
    with pytest.raises(ValueError, match="or neither"):
        ReceiptReference(
            manifest_kind="quillon.probe_run", manifest_id="p-0", manifest_sha256="a" * 64
        )


def test_a_receipt_locator_has_no_execution_uri_because_nothing_produced_it() -> None:
    with pytest.raises(ValueError, match="has no execution uri"):
        ReceiptReference(
            manifest_kind="quillon.probe_run",
            manifest_id="p-0",
            execution_uri="file:///custody/p-0",
        )


def test_an_unknown_authored_reference_kind_is_refused() -> None:
    with pytest.raises(ExperimentEnvelopeRejection):
        _parse(
            _minimal(
                figure={
                    "inputs": [
                        {"input_role": "observed", "ref": {"kind": "http", "url": "x"}}
                    ]
                }
            )
        )


def test_checkpoint_initialization_that_is_not_applicable_is_simply_not_authored() -> None:
    with pytest.raises(ExperimentEnvelopeRejection, match="simply not authored"):
        _parse(
            _minimal(
                training={
                    "rows": [{"from": "baseline", "id": "widened"}],
                    "checkpoint_initialization": [
                        {
                            "row": "widened",
                            "mode": "continue_from",
                            "source": {"kind": "not_applicable", "reason": "nothing to continue"},
                        }
                    ],
                }
            )
        )


# -- project vocabulary is data, judged by the output model -------------------


def test_a_project_word_travels_as_data_and_reaches_the_compiled_document(
    repo: Path,
) -> None:
    outcome = kernel_for(PROJECT_DECLARATION).compile_envelope_file(
        envelope_path(repo, "widened-summary"), repo_root=repo
    )

    assert outcome.document["analysis_type"] == "quillon.span_summary"
    assert outcome.document["params"] == {"window": 3, "trim": 1}


def test_a_value_the_output_model_refuses_is_a_rejection_not_a_bad_document(
    repo: Path,
) -> None:
    from tests.fake_project_experiment import write_envelope

    envelope = dict(ANALYSIS_ENVELOPE)
    envelope["analysis"] = {
        **envelope["analysis"],
        "delta": {
            "layer_id": "broken",
            "patches": [{"path": "analysis_type", "op": "replace", "value": 5}],
        },
    }
    write_envelope(envelope_path(repo, "widened-summary"), envelope)

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        kernel_for(PROJECT_DECLARATION).compile_envelope_file(
            envelope_path(repo, "widened-summary"), repo_root=repo
        )

    assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert "feedbax.spec.analysis_run" in str(caught.value)


# -- documents announce their own layer ---------------------------------------


@pytest.mark.parametrize(
    ("schema_id", "layer", "family"),
    [
        ("feedbax.spec.training_run_matrix", "training", "training_run_matrix"),
        ("feedbax.spec.evaluation_run_matrix", "evaluation", "evaluation_run_matrix"),
        ("feedbax.spec.analysis_run", "analysis", "analysis_run"),
        ("feedbax.spec.analysis_bundle", "analysis", "analysis_bundle"),
        ("feedbax.spec.figure", "figure", "figure"),
        ("feedbax.spec.figure_composition", "figure", "figure_composition"),
        ("feedbax.spec.report.ordered_figure", "report", "report"),
    ],
)
def test_a_document_declares_which_layer_owns_it(
    schema_id: str, layer: str, family: str
) -> None:
    document = {"schema_id": schema_id}

    assert layer_of_document(document) is ExperimentEnvelopeLayer(layer)
    contract = output_contract_of_document(document)
    assert contract is not None
    assert contract.family == family
    assert contract.schema_version.startswith(f"{schema_id}.")


def test_a_document_of_no_feedbax_family_belongs_to_no_layer() -> None:
    assert layer_of_document({"schema_id": "quillon.notes"}) is None
    assert output_contract_of_document({"schema_id": "quillon.notes"}) is None
    assert layer_of_document("not a document") is None  # type: ignore[arg-type]


def test_every_declared_output_model_actually_imports() -> None:
    for contract in LAYER_OUTPUT_CONTRACTS.values():
        model = contract.model()
        assert hasattr(model, "model_validate")
