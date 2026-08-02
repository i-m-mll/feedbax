"""The closed ``feedbax.experiment_envelope`` dialect and its version boundary.

Three properties are under test and nothing else. First, the dialect is *closed*:
two enumerated schema strings, one layer per envelope, no unknown fields
anywhere, and no slot through which a project could widen it. Second, the
vocabulary inside it is *open*: dotted paths, values, recipe ids, and role
strings are carried as data and judged by the final Feedbax output model, not by
the dialect. Third, each version names exactly one grammar: a v1 document is
accepted as v1, migrates to v2 only through an explicit call, and is refused by
version if it states a v2 construct.

The invented ``quillon`` vocabulary is used throughout for the same reason it is
used in the kernel tests: if a case needed a real project's words, it would be
testing the wrong thing.
"""

from __future__ import annotations

import json
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
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2,
    EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS,
    LAYER_OUTPUT_CONTRACTS,
    ExperimentEnvelopeLayer,
    ReceiptReference,
    layer_of_document,
    migrate_experiment_envelope_payload,
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


#: The three authored fragments the version-boundary cases reuse. They are
#: deliberately the smallest thing each construct accepts, so a failure names the
#: version rule rather than an unrelated authoring mistake.
_RECEIPT: dict[str, Any] = {
    "kind": "receipt",
    "manifest_kind": "quillon.survey_run",
    "manifest_id": "baseline-0",
}
_ANALYSIS_DELTA: dict[str, Any] = {
    "layer_id": "probe-bundle",
    "patches": [{"path": "params_base.trim", "op": "add", "value": 1}],
}
_ROLE_CONTRACT: dict[str, Any] = {
    "kind": "quillon.survey_run",
    "artifact_role": "span_observations",
    "artifact_provider": "quillon.custody",
}


# -- the family and its version boundary ----------------------------------


def test_the_dialect_is_one_family_at_two_enumerated_versions() -> None:
    assert EXPERIMENT_ENVELOPE_FAMILY == "feedbax.experiment_envelope"
    assert EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1 == "feedbax.experiment_envelope.v1"
    assert EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2 == "feedbax.experiment_envelope.v2"
    assert EXPERIMENT_ENVELOPE_SCHEMA_VERSION == EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2
    assert EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS == (
        EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1,
        EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2,
    )
    assert EXPERIMENT_ENVELOPE_MIGRATION_TABLE == {
        EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1: EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2
    }
    assert EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION == (
        "feedbax.experiment_envelope.compiler.v1"
    )


@pytest.mark.parametrize(
    "schema",
    ["feedbax.experiment_envelope.v0", "feedbax.experiment_envelope.v3", "quillon.study.v1", None],
)
def test_any_other_schema_fails_closed_naming_its_migration_slot(schema: Any) -> None:
    document = _minimal(training={"rows_mode": "append", "tags": {"add": ["probe"]}})
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
    assert str(list(EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS)) in message


# -- the v1/v2 boundary: accept, migrate, reject ---------------------------


def _v1(**layer: Any) -> dict[str, Any]:
    return {**_minimal(**layer), "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1}


def test_a_v1_document_is_accepted_and_keeps_its_own_version() -> None:
    """A prior-version document compiles as itself, not as a wider current one."""
    envelope = _parse(_v1(training={"rows_mode": "append", "tags": {"add": ["probe"]}}))

    assert envelope.schema_ == EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1
    dumped = envelope.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert dumped["schema"] == EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1


@pytest.mark.parametrize(
    ("layer", "path"),
    [
        (
            {
                "evaluation": {
                    "subject_id": "trained",
                    "subject": _RECEIPT,
                    "prerequisites": {"bank": _RECEIPT},
                }
            },
            "evaluation.prerequisites",
        ),
        (
            {
                "analysis": {
                    "target": "bundle",
                    "roots": [{"alias": "one", "ref": _RECEIPT}],
                    "delta": _ANALYSIS_DELTA,
                }
            },
            "analysis.roots",
        ),
        (
            {
                "figure": {
                    "mode": "row_expansion",
                    "rows": {"mode": "all", "index": "bases/rows.row_index.json"},
                    "row_custody": "generated/rows.custody.json",
                    "inputs": [
                        {
                            "input_role": "per_row_states",
                            "binding": "per_row",
                            "binding_key": "states",
                            "contract": _ROLE_CONTRACT,
                        }
                    ],
                }
            },
            "figure.row_custody",
        ),
        (
            {
                "training": {
                    "rows_mode": "append",
                    "checkpoint_initialization": [
                        {"row": "row-a", "mode": "initialize_from", "source": _RECEIPT}
                    ],
                }
            },
            "training.checkpoint_initialization",
        ),
    ],
)
def test_a_v1_document_stating_v2_grammar_is_refused_by_version(
    layer: dict[str, Any], path: str
) -> None:
    """v1 names exactly one grammar; a v2 construct under it is a version refusal."""
    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _parse(_v1(**layer))

    assert caught.value.category is (
        ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION
    )
    message = str(caught.value)
    assert path in message
    assert EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2 in message
    # The same document at v2 is accepted, so the refusal is about the version
    # boundary and not about the construct being unauthorable.
    assert _parse(_minimal(**layer)) is not None


def test_the_v1_to_v2_migration_is_explicit_and_semantics_preserving() -> None:
    document = _v1(training={"rows_mode": "append", "tags": {"add": ["probe"]}})

    migrated = migrate_experiment_envelope_payload(document)

    assert migrated["schema"] == EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2
    assert {key: value for key, value in migrated.items() if key != "schema"} == {
        key: value for key, value in document.items() if key != "schema"
    }
    # The migration changes the authored bytes, which is exactly why a compile
    # never applies it: the original document still parses, as itself.
    assert _parse(document).schema_ == EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1
    assert _parse(migrated).schema_ == EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2


def test_migrating_an_unsupported_version_refuses_rather_than_guessing() -> None:
    document = _minimal(training={"rows_mode": "append", "tags": {"add": ["probe"]}})
    document["schema"] = "feedbax.experiment_envelope.v3"

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        migrate_experiment_envelope_payload(document)

    assert caught.value.category is (
        ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION
    )


def test_migrating_a_current_document_is_a_no_op() -> None:
    document = _minimal(training={"rows_mode": "append", "tags": {"add": ["probe"]}})

    assert migrate_experiment_envelope_payload(document) == document


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
                training={"rows_mode": "append", "tags": {"add": ["probe"]}},
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
        {
            **_minimal(training={"rows_mode": "append", "tags": {"add": ["p"]}}),
            "invented_top_level": 1,
        },
        _minimal(training={"rows_mode": "append", "tags": {"add": ["p"]}, "invented": 1}),
        _minimal(
            training={
                "rows_mode": "append",
                "rows": [{"from": "baseline", "id": "x", "invented": 1}],
            }
        ),
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
                "mode": "composition",
                "delta": {"layer_id": "one", "patches": []},
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
        _parse(_minimal(training={"rows_mode": "append", "rows": [{"id": "widened"}]}))

    assert caught.value.category is ExperimentEnvelopeRejectionCategory.MISSING_FIELD


def test_an_assertion_states_the_value_it_expects() -> None:
    with pytest.raises(ExperimentEnvelopeRejection, match="states the value it expects"):
        _parse(
            {
                **_minimal(training={"rows_mode": "append", "tags": {"add": ["probe"]}}),
                "assert": [{"path": "base.inline.cadence"}],
            }
        )


def test_an_assertion_may_expect_a_null_it_states_explicitly() -> None:
    envelope = _parse(
        {
            **_minimal(training={"rows_mode": "append", "tags": {"add": ["probe"]}}),
            "assert": [{"path": "base.inline.cadence", "equals": None}],
        }
    )

    assert envelope.assert_[0].equals is None


def test_a_tags_delta_must_change_something_and_may_not_contradict_itself() -> None:
    with pytest.raises(ExperimentEnvelopeRejection, match="adds or removes something"):
        _parse(_minimal(training={"rows_mode": "append", "tags": {"add": [], "remove": []}}))
    with pytest.raises(ExperimentEnvelopeRejection, match="both added and removed"):
        _parse(
            _minimal(training={"rows_mode": "append", "tags": {"add": ["p"], "remove": ["p"]}})
        )


def test_training_row_ids_and_delta_layer_ids_are_unique_within_one_envelope() -> None:
    row = {"from": "baseline", "id": "widened"}
    with pytest.raises(ExperimentEnvelopeRejection, match="row ids must be unique"):
        _parse(_minimal(training={"rows_mode": "append", "rows": [dict(row), dict(row)]}))
    with pytest.raises(ExperimentEnvelopeRejection, match="layer ids must be unique"):
        _parse(
            _minimal(
                training={
                    "rows_mode": "append",
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
                    "mode": "composition",
                    "delta": {"layer_id": "one", "patches": []},
                    "inputs": [
                        {"input_role": "observed", "ref": {"kind": "http", "url": "x"}}
                    ],
                }
            )
        )


def test_checkpoint_initialization_that_is_not_applicable_is_simply_not_authored() -> None:
    with pytest.raises(ExperimentEnvelopeRejection, match="simply not authored"):
        _parse(
            _minimal(
                training={
                    "rows_mode": "append",
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
        ("feedbax.spec.report", "report", "report"),
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


# -- the ratified equivalence corrections -------------------------------------


class TestTrainingRowsMode:
    """What the compiled row set is, is stated rather than inherited by default."""

    def test_an_absent_rows_mode_is_a_missing_field_rather_than_a_default(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection) as caught:
            _parse(_minimal(training={"tags": {"add": ["probe"]}}))

        assert caught.value.category is ExperimentEnvelopeRejectionCategory.MISSING_FIELD
        assert "rows_mode" in str(caught.value)

    @pytest.mark.parametrize("mode", ["authored_only", "append"])
    def test_the_mode_set_is_closed(self, mode: str) -> None:
        envelope = _parse(
            _minimal(
                training={"rows_mode": mode, "rows": [{"from": "baseline", "id": "x"}]}
            )
        )

        assert envelope.training is not None
        assert envelope.training.rows_mode.value == mode

    def test_an_invented_mode_is_refused(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection) as caught:
            _parse(
                _minimal(
                    training={"rows_mode": "inherit", "rows": [{"from": "b", "id": "x"}]}
                )
            )

        assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE

    def test_authored_only_with_no_authored_rows_would_run_nothing(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection, match="runs exactly"):
            _parse(
                _minimal(
                    training={"rows_mode": "authored_only", "tags": {"add": ["probe"]}}
                )
            )


class TestTrainingRowLabel:
    """A row's label follows its identity, never its source's."""

    def test_the_label_defaults_to_the_rows_own_id(self) -> None:
        envelope = _parse(
            _minimal(
                training={
                    "rows_mode": "append",
                    "rows": [{"from": "baseline", "id": "widened"}],
                }
            )
        )

        assert envelope.training is not None
        assert envelope.training.rows[0].effective_label == "widened"

    def test_an_authored_label_is_used_as_stated(self) -> None:
        envelope = _parse(
            _minimal(
                training={
                    "rows_mode": "append",
                    "rows": [
                        {"from": "baseline", "id": "widened", "label": "widened span"}
                    ],
                }
            )
        )

        assert envelope.training is not None
        assert envelope.training.rows[0].effective_label == "widened span"

    def test_an_empty_label_is_refused_rather_than_treated_as_absent(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection, match="nonempty one"):
            _parse(
                _minimal(
                    training={
                        "rows_mode": "append",
                        "rows": [{"from": "baseline", "id": "widened", "label": "  "}],
                    }
                )
            )


def test_a_top_level_training_delta_would_make_the_structured_layer_ornamental() -> None:
    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _parse(
            _minimal(
                training={
                    "rows_mode": "append",
                    "tags": {"add": ["probe"]},
                    "delta": {"layer_id": "raw", "patches": []},
                }
            )
        )

    assert caught.value.category is ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD


class TestFigureMode:
    """The operation a figure performs is authored, never read off a document."""

    def test_an_absent_mode_is_a_missing_field(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection) as caught:
            _parse(
                _minimal(figure={"delta": {"layer_id": "one", "patches": []}})
            )

        assert caught.value.category is ExperimentEnvelopeRejectionCategory.MISSING_FIELD
        assert "mode" in str(caught.value)

    def test_a_row_expansion_figure_authors_no_delta(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection, match="authors no delta"):
            _parse(
                _minimal(
                    figure={
                        "mode": "row_expansion",
                        "rows": {"mode": "all", "index": "bases/rows.json"},
                        "delta": {"layer_id": "one", "patches": []},
                        "inputs": [
                            {
                                "input_role": "observed",
                                "ref": {"kind": "envelope", "alias": "x"},
                                "binding": "shared",
                                "binding_key": "m-0",
                                "contract": {
                                    "artifact_role": "r",
                                    "artifact_provider": "p",
                                },
                            }
                        ],
                    }
                )
            )

    def test_a_row_expansion_figure_names_the_row_index_it_expands_against(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection, match="names the row index"):
            _parse(
                _minimal(
                    figure={
                        "mode": "row_expansion",
                        "rows": {"mode": "all"},
                        "inputs": [
                            {
                                "input_role": "observed",
                                "ref": {"kind": "envelope", "alias": "x"},
                                "binding": "shared",
                                "binding_key": "m-0",
                                "contract": {
                                    "artifact_role": "r",
                                    "artifact_provider": "p",
                                },
                            }
                        ],
                    }
                )
            )

    def test_a_row_expansion_input_states_a_whole_binding_profile_or_none(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection, match="partial profile"):
            _parse(
                _minimal(
                    figure={
                        "mode": "row_expansion",
                        "rows": {"mode": "all", "index": "bases/rows.json"},
                        "inputs": [
                            {
                                "input_role": "observed",
                                "ref": {"kind": "envelope", "alias": "x"},
                                "binding": "per_row",
                            }
                        ],
                    }
                )
            )

    def test_a_composition_figure_states_no_row_vocabulary(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection, match="row_expansion vocabulary"):
            _parse(
                _minimal(
                    figure={
                        "mode": "composition",
                        "delta": {"layer_id": "one", "patches": []},
                        "rows": {"mode": "all", "index": "bases/rows.json"},
                    }
                )
            )

    def test_a_composition_figure_states_the_deltas_it_composes(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection, match="ordered deltas"):
            _parse(_minimal(figure={"mode": "composition"}))

    def test_the_row_selector_is_the_closed_feedbax_selector(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection):
            _parse(
                _minimal(
                    figure={
                        "mode": "row_expansion",
                        "rows": {"mode": "regex", "pattern": ".*"},
                        "inputs": [],
                    }
                )
            )


def test_the_report_layer_output_is_the_top_level_report_spec() -> None:
    from feedbax.analysis.fulfillment_derivation import COMPILED_PRODUCT_KINDS
    from feedbax.contracts.experiment_envelope_dialect import REPORT_OUTPUT
    from feedbax.contracts.manifest import (
        REPORT_SPEC_SCHEMA_ID,
        REPORT_SPEC_SCHEMA_VERSION,
        ReportSpec,
    )

    assert REPORT_OUTPUT.schema_id == REPORT_SPEC_SCHEMA_ID
    assert REPORT_OUTPUT.schema_version == REPORT_SPEC_SCHEMA_VERSION
    assert REPORT_OUTPUT.model() is ReportSpec
    assert REPORT_OUTPUT.schema_id in COMPILED_PRODUCT_KINDS


def test_the_report_output_validates_its_ordered_figure_params() -> None:
    from feedbax.analysis.reports import (
        ORDERED_FIGURE_REPORT_TYPE,
        OrderedFigureReportParams,
    )
    from feedbax.contracts.experiment_envelope_dialect import REPORT_OUTPUT

    document = {"report_type": ORDERED_FIGURE_REPORT_TYPE}

    assert REPORT_OUTPUT.params_model(document) is OrderedFigureReportParams
    assert REPORT_OUTPUT.params_model({"report_type": "quillon.bulletin"}) is None


def test_an_ordered_figure_params_document_is_no_longer_a_layer_parent() -> None:
    """The parent is the whole report, not the params block inside one."""
    assert layer_of_document({"schema_id": "feedbax.spec.report.ordered_figure"}) is None


# -- a per-row role has no single reference to state ---------------------------


#: Where a row-expansion envelope says its per-row custody bindings will be.
CUSTODY_REF = "custody/quillon.row_custody.json"


class TestPerRowInputReference:
    """``ref`` is optional exactly where a single locator would be false."""

    @staticmethod
    def _figure(*, custody: str | None = CUSTODY_REF, **input_fields: Any) -> dict[str, Any]:
        figure: dict[str, Any] = {
            "mode": "row_expansion",
            "rows": {"mode": "all", "index": "bases/quillon.row_index.json"},
            "inputs": [
                {
                    "input_role": "observed",
                    "binding_key": "observations",
                    "contract": {
                        "kind": "quillon.survey_run",
                        "artifact_role": "span_observations",
                        "artifact_provider": "quillon.custody",
                    },
                    **input_fields,
                }
            ],
        }
        if custody is not None:
            figure["row_custody"] = custody
        return _minimal(figure=figure)

    def test_a_per_row_role_may_omit_the_reference_row_expansion_fills(self) -> None:
        envelope = _parse(self._figure(binding="per_row"))

        assert envelope.figure is not None
        item = envelope.figure.inputs[0]
        assert item.ref is None
        assert item.is_per_row
        assert item.is_row_expanded
        assert item.role_reference().model_dump() == {"per_row": "observations"}
        assert envelope.figure.row_custody == CUSTODY_REF

    def test_a_per_row_role_may_omit_the_custody_declaration_too(self) -> None:
        """An envelope authored before the declaration existed still parses.

        Its per-row roles cannot be *bound* — fulfillment refuses the figure by
        name — but the ratified corpus is not invalidated by a field that did not
        exist when it was written.
        """
        envelope = _parse(self._figure(binding="per_row", custody=None))

        assert envelope.figure is not None
        assert envelope.figure.row_custody is None
        assert envelope.figure.inputs[0].is_per_row

    def test_an_empty_custody_declaration_names_no_document(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection) as caught:
            _parse(self._figure(binding="per_row", custody="   "))

        assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
        assert "states a nonempty 'row_custody' path or states none at all" in str(caught.value)

    def test_a_shared_role_still_states_the_one_manifest_it_is_filled_from(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection) as caught:
            _parse(self._figure(binding="shared"))

        assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
        assert "no single locator addresses it" in str(caught.value)

    def test_a_figure_input_outside_row_expansion_still_states_its_reference(self) -> None:
        with pytest.raises(ExperimentEnvelopeRejection) as caught:
            _parse(
                _minimal(
                    figure={
                        "mode": "composition",
                        "delta": {"layer_id": "overlay", "patches": []},
                        "inputs": [{"input_role": "observed"}],
                    }
                )
            )

        assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
        assert "states no ref" in str(caught.value)

    def test_a_per_row_role_may_not_state_a_reference(self) -> None:
        """A locator on a per-row role is a false fact, not an extra one."""
        with pytest.raises(ExperimentEnvelopeRejection) as caught:
            _parse(
                self._figure(
                    binding="per_row",
                    ref={
                        "kind": "receipt",
                        "manifest_kind": "quillon.survey_run",
                        "manifest_id": "observed-0",
                    },
                )
            )

        assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
        assert "is a 'per_row' role and states a ref" in str(caught.value)
        assert "false for every other row" in str(caught.value)

    def test_a_per_row_role_may_not_state_an_envelope_reference_either(self) -> None:
        """The refusal is about the single-locator slot, not one locator kind."""
        with pytest.raises(ExperimentEnvelopeRejection, match="states a ref"):
            _parse(
                self._figure(
                    binding="per_row",
                    ref={"kind": "envelope", "alias": "widened-summary"},
                )
            )

    def test_custody_declared_where_nothing_is_filled_per_row_is_refused(self) -> None:
        """A shared role names its own manifest, so the declaration addresses nothing."""
        with pytest.raises(ExperimentEnvelopeRejection) as caught:
            _parse(
                self._figure(
                    binding="shared",
                    ref={
                        "kind": "receipt",
                        "manifest_kind": "quillon.survey_run",
                        "manifest_id": "observed-0",
                    },
                )
            )

        assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
        assert "only when a per-row role is filled from it" in str(caught.value)


# -- the corpus shape fits the ratified evaluation byte budget -------------


#: The paired-controller evaluation envelope, re-authored at v2 with the staged
#: prerequisite its compiled base names. Every scalar is the real corpus value,
#: because the point of the case is the *size* of the actual shape: a synthetic
#: stand-in with shorter ids would prove nothing about whether the corpus fits.
#:
#: The rlrmp2 evaluation layer caps an authored envelope at 2048 bytes
#: (``specs/experiment/experiment_envelope.budgets.v3.json``). That document is
#: ratified project policy and is not Feedbax's to widen, so the authoring form
#: has to fit it. The list-of-objects spelling this construct started with did
#: not: it spent about thirty bytes per prerequisite restating a key that JSON
#: already gives a mapping for free, and pushed this envelope to 2063 bytes.
PAIRED_CONTROLLER_BANK = (
    "feedbax-evaluation-run:3686909fa04735e7b802e444885ff71f"
)
PAIRED_CONTROLLER_ENVELOPE: dict[str, Any] = {
    "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2,
    "base": "specs/post_run/sisu_paired_controller_response.matrix.v1.json",
    "name": "sisu-paired-controller-response-continuous",
    "issue": "b7f3caa",
    "reason": (
        "Bind the wave-1 continuous trained run and its capture checkpoints to the "
        "paired controller response grid."
    ),
    "evaluation": {
        "subject_id": "trained",
        "subject": {
            "kind": "receipt",
            "manifest_kind": "TrainingRunManifest",
            "manifest_id": "feedbax-training-run:662f6e3d4f17c350bbdf9737b591b405",
            "manifest_sha256": (
                "7dbca684ee130475beac8261d5bdbbdb171a25c3f63b762b037c430954d28a33"
            ),
            "size_bytes": 262904,
        },
        "prerequisites": {
            "paired_trial_bank": {
                "kind": "receipt",
                "manifest_kind": "EvaluationRunManifest",
                "manifest_id": PAIRED_CONTROLLER_BANK,
                "manifest_sha256": (
                    "983beeff4164fd6b19616bc912f8e36f519fa225b45ecf8606e1e5813610f3d5"
                ),
                "size_bytes": 22582,
            }
        },
        "delta": {
            "layer_id": "sisu-paired-controller-response-continuous.subject",
            "acknowledges_ancestor_paths": [
                "axes.0.values.0.deltas.7.value",
                "axes.0.values.0.deltas.8.value",
                "axes.0.values.0.deltas.9.value",
                "axes.0.values.0.deltas.10.value",
            ],
            "patches": [
                {
                    "path": "axes.0.values.0.deltas.7.value",
                    "op": "replace",
                    "value": {
                        "kind": "TrainingCheckpointTransactionManifest",
                        "id": "tx-1003b37294bf4f9b83b074da256fa4a1",
                        "role": "training_checkpoint_custody",
                        "uri": (
                            "transactions/tx-1003b37294bf4f9b83b074da256fa4a1/manifest.json"
                        ),
                        "metadata": {
                            "manifest_sha256": (
                                "1f6b6dbbe0f18508a82db0e30951b0983ab213c79944f3ef9bf"
                                "738330a79acec"
                            )
                        },
                    },
                },
                {
                    "path": "axes.0.values.0.deltas.8.value",
                    "op": "replace",
                    "value": "capture-checkpoints",
                },
                {
                    "path": "axes.0.values.0.deltas.9.value",
                    "op": "replace",
                    "value": 12000,
                },
                {
                    "path": "axes.0.values.0.deltas.10.value",
                    "op": "replace",
                    "value": [
                        {
                            "kind": "TrainingRunManifest",
                            "id": "feedbax-training-run:662f6e3d4f17c350bbdf9737b591b405",
                            "role": "training_run",
                            "metadata": {
                                "ref_schema_id": "feedbax.ref.authenticated_manifest",
                                "ref_schema_version": (
                                    "feedbax.ref.authenticated_manifest.v1"
                                ),
                                "manifest_sha256": (
                                    "7dbca684ee130475beac8261d5bdbbdb171a25c3f63b762b03"
                                    "7c430954d28a33"
                                ),
                                "size_bytes": 262904,
                            },
                        }
                    ],
                },
            ],
        },
    },
}

#: The ratified rlrmp2 evaluation-layer cap. Restated as a literal on purpose:
#: reading it out of the project document would make this test pass whenever the
#: project widened the cap, which is the one outcome it exists to prevent.
RATIFIED_EVALUATION_MAX_BYTES = 2048


def test_the_corpus_paired_controller_shape_fits_its_ratified_byte_budget() -> None:
    """The real shape, at v2, inside the cap the project already ratified."""
    minimal = json.dumps(
        PAIRED_CONTROLLER_ENVELOPE, separators=(",", ":"), sort_keys=True
    ).encode()

    assert len(minimal) <= RATIFIED_EVALUATION_MAX_BYTES, (
        f"the corpus shape encodes to {len(minimal)} bytes against a ratified cap of "
        f"{RATIFIED_EVALUATION_MAX_BYTES}; the authoring form is Feedbax's to make lean, "
        "and the project's budget is not Feedbax's to widen"
    )


def test_the_corpus_paired_controller_shape_parses_as_the_v2_grammar() -> None:
    envelope = _parse(PAIRED_CONTROLLER_ENVELOPE)

    assert envelope.schema_ == EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2
    assert envelope.evaluation.subject_id == "trained"
    assert list(envelope.evaluation.prerequisites) == ["paired_trial_bank"]
    prerequisite = envelope.evaluation.prerequisites["paired_trial_bank"]
    assert isinstance(prerequisite, ReceiptReference)
    assert prerequisite.is_authenticated
    assert prerequisite.manifest_id == PAIRED_CONTROLLER_BANK


def test_the_list_spelling_of_prerequisites_is_gone_rather_than_also_accepted() -> None:
    """One spelling. The lean mapping is the form, not a second way to say it."""
    document = {
        **PAIRED_CONTROLLER_ENVELOPE,
        "evaluation": {
            **PAIRED_CONTROLLER_ENVELOPE["evaluation"],
            "prerequisites": [{"name": "paired_trial_bank", "ref": _RECEIPT}],
        },
    }

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _parse(document)

    assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
