"""The envelope engine kernel, exercised over an invented fake project.

Every case here is about *mechanism*: canonical hashing, budgets, the compile
lock's plan/receipt boundary, fail-closed loading, lineage resolution, the
closed-layer compile, and the choke point. The science is entirely ``quillon``'s,
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
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1,
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V2,
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
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION_V2,
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION_V3,
    EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_ID,
    EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_VERSION,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V3,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V4,
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V5,
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION_V1,
    REPORT_OUTPUT,
    TRAINING_OUTPUT_V6,
    ExperimentEnvelopeLayer,
)
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.run_composition import (
    CompositionNode,
    InlineIntentParent,
    authored_envelope_hash,
)
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
from feedbax.training.run_matrix import materialize_adapted_run_matrix

from tests.fake_project_experiment import (
    ENVELOPE_DIRECTORY,
    OUTPUT_DIRECTORY,
    PROJECT_DECLARATION,
    TRAINING_BASE,
    TRAINING_ENVELOPE,
    envelope_path,
    write_envelope,
    write_json,
    write_repo,
)
from tests.test_training_method_plugin_cli import _standard_run_spec_payload

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


def _root_envelope(root: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
        "name": "rooted-matrix",
        "issue": "generic-root-proof",
        "training": {"root": root},
    }


def _compile_root(repo: Path, root: dict[str, Any]) -> Any:
    path = envelope_path(repo, "rooted")
    write_envelope(path, _root_envelope(root))
    return kernel().compile_envelope_file(path, repo_root=repo)


def _layer_root_authority(kind: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_id": EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_ID,
        "schema_version": EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_VERSION,
        "kind": kind,
        **fields,
    }


def _comparison_policy_authority() -> dict[str, Any]:
    return _layer_root_authority(
        "comparison_policy",
        roles={
            "reference": {
                "source_class": "quillon.loss_trace",
                "label": "Reference",
                "training_policy": "fixed",
                "trace_schema_id": "quillon.trace",
                "trace_schema_version": "quillon.trace.v1",
                "retention_contract": "contracts/reference.json",
                "figure_template": "terminal",
            },
            "candidate": {
                "source_class": "quillon.loss_trace",
                "label": "Candidate",
                "training_policy": "adaptive",
                "figure_template": "terminal",
            },
        },
        figure_templates={
            "terminal": {
                "name": "terminal",
                "description": "Generic terminal comparison",
                "assembler": "quillon.comparison_grid",
            }
        },
        comparison_policy={
            "supported_source_class": "quillon.loss_trace",
            "required_cadence": "per_checkpoint",
            "required_equal_authority": ["training_data", "optimizer"],
            "mismatch_policy": "fail_closed",
        },
    )


def _compile_layer_root(
    repo: Path,
    alias: str,
    authority: dict[str, Any],
    layer: dict[str, Any],
    *,
    layer_name: str | None = None,
    payload_path: list[str] | None = None,
    whole_document: dict[str, Any] | None = None,
    schema: str = EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V4,
) -> Any:
    ref = f"authorities/{alias}.json"
    document = authority if whole_document is None else whole_document
    write_json(repo / ref, document)
    root: dict[str, Any] = {"ref": ref, "sha256": canonical_sha256(document)}
    if payload_path is not None:
        root["payload_path"] = payload_path
    content = dict(layer)
    content["root"] = root
    if layer_name is None:
        layer_name = "analysis" if "target" in content else "figure"
    envelope = {
        "schema": schema,
        "name": alias,
        layer_name: content,
    }
    write_envelope(envelope_path(repo, alias), envelope)
    return kernel().compile_envelope_file(envelope_path(repo, alias), repo_root=repo)


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


def test_widest_caps_are_the_maximum_any_layer_states(
    budgets: AuthoringBudgets,
) -> None:
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


def test_optional_caps_are_stated_only_where_the_dimension_exists(
    budgets: AuthoringBudgets,
) -> None:
    """``max_rows`` binds the one layer that authors rows, and nothing else."""
    assert budgets.for_layer("training").optional_cap("max_rows") == 2
    assert budgets.for_layer("report").optional_cap("max_rows") is None
    with pytest.raises(KeyError, match="not an optional budget cap"):
        budgets.for_layer("training").optional_cap("max_probes")


def test_budget_document_refuses_a_project_cap_nothing_enforces() -> None:
    """A declared cap nothing checks reads as a bound while permitting everything."""
    document = _budget_document()
    document["layers"]["training"]["project_caps"] = {"max_probes": 6}

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        AuthoringBudgets.from_document(
            document, field="budget.json", declared_layers=DECLARED_LAYERS
        )

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert "max_probes" in str(excinfo.value)
    assert "['max_rows']" in str(excinfo.value)


def test_budget_document_refuses_a_nonpositive_optional_cap() -> None:
    document = _budget_document()
    document["layers"]["training"]["project_caps"] = {"max_rows": 0}

    with pytest.raises(ExperimentEnvelopeRejection, match="positive integer"):
        AuthoringBudgets.from_document(
            document, field="budget.json", declared_layers=DECLARED_LAYERS
        )


def _authored_training_rows(repo: Path, count: int) -> None:
    """Re-author the training envelope so it states exactly *count* rows."""
    envelope = _read(repo, "widened")
    envelope["training"]["rows"] = [
        {"from": "baseline", "id": f"widened-{index}", "seed": 43 + index} for index in range(count)
    ]
    envelope["training"].pop("checkpoint_initialization", None)
    _write(repo, "widened", envelope)


def test_authoring_rows_up_to_the_row_cap_compiles(repo: Path) -> None:
    """The cap is a bound, not a target: authoring exactly it is admitted."""
    _authored_training_rows(repo, 2)

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert [row["row_id"] for row in outcome.document["rows"]] == [
        "baseline",
        "widened-0",
        "widened-1",
    ]


def test_authoring_more_rows_than_the_cap_is_refused(repo: Path) -> None:
    _authored_training_rows(repo, 3)

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.BUDGET_EXCEEDED
    assert "3 authored rows exceeds" in str(excinfo.value)
    assert str(excinfo.value.field).endswith("#training.rows")


def test_a_layer_that_authors_no_rows_is_bound_by_no_row_cap(repo: Path) -> None:
    """The figure layer's budget states no max_rows, so nothing here bounds it."""
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

    assert len(outcome.document["panels"]) == 2


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
        EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION_V1
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


# -- the whole v1 document is validated on read ---------------------------------


def _compiled_lock(repo: Path, alias: str = "widened") -> dict[str, Any]:
    """Return one real compiled lock as tracked JSON, ready to be damaged."""
    outcome = kernel().compile_envelope_file(envelope_path(repo, alias), repo_root=repo)
    return json.loads(json.dumps(outcome.compile_lock))


def _drop_key(lock: dict[str, Any]) -> None:
    del lock["base"]


def _foreign_key(lock: dict[str, Any]) -> None:
    lock["compiled_at"] = "2026-08-01"


def _blank_name(lock: dict[str, Any]) -> None:
    lock["name"] = "  "


def _blank_issue(lock: dict[str, Any]) -> None:
    lock["issue"] = ""


def _envelope_hash(lock: dict[str, Any]) -> None:
    lock["envelope"]["envelope_hash"] = "not-a-digest"


def _envelope_ref(lock: dict[str, Any]) -> None:
    del lock["envelope"]["ref"]


def _base_domain(lock: dict[str, Any]) -> None:
    lock["base"]["pin_algorithm"] = "md5"


def _base_digest(lock: dict[str, Any]) -> None:
    lock["base"]["content_hash"] = "0" * 63


def _lineage_shape(lock: dict[str, Any]) -> None:
    lock["lineage"] = {"ref": "bases/baseline.training_run_matrix.json"}


def _lineage_pin(lock: dict[str, Any]) -> None:
    del lock["lineage"][0]["content_hash"]


def _delta_key(lock: dict[str, Any]) -> None:
    key, value = next(iter(lock["resolved_deltas"].items()))
    lock["resolved_deltas"] = {f"{key}.renamed": value}


def _delta_shape(lock: dict[str, Any]) -> None:
    key = next(iter(lock["resolved_deltas"]))
    del lock["resolved_deltas"][key]["patches"]


def _delta_patch(lock: dict[str, Any]) -> None:
    key = next(iter(lock["resolved_deltas"]))
    del lock["resolved_deltas"][key]["patches"][0]["op"]


def _assertion_shape(lock: dict[str, Any]) -> None:
    del lock["assertions"][0]["owner_ref"]


def _document_digest(lock: dict[str, Any]) -> None:
    lock["compiled_document"]["content_hash"] = "zz" + "0" * 62


def _document_family(lock: dict[str, Any]) -> None:
    lock["compiled_document"]["family"] = ""


def _contract_version(lock: dict[str, Any]) -> None:
    lock["compiler_contract"]["contract_version"] = "rival.contract.v1"


def _implementation_shape(lock: dict[str, Any]) -> None:
    del lock["compiler_implementation"]["package_versions"]


def _implementation_code_unit(lock: dict[str, Any]) -> None:
    lock["compiler_implementation"]["code_unit"] = " "


def _identity_digest(lock: dict[str, Any]) -> None:
    lock["execution_identity"]["sha256"] = "a" * 64


def _identity_inputs(lock: dict[str, Any]) -> None:
    lock["execution_identity"]["inputs"] = []


@pytest.mark.parametrize(
    ("damage", "category", "match"),
    [
        (_drop_key, ExperimentEnvelopeRejectionCategory.MISSING_FIELD, "base"),
        (
            _foreign_key,
            ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD,
            "compiled_at",
        ),
        (_blank_name, ExperimentEnvelopeRejectionCategory.INVALID_VALUE, "name"),
        (_blank_issue, ExperimentEnvelopeRejectionCategory.INVALID_VALUE, "issue"),
        (
            _envelope_hash,
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "envelope_hash",
        ),
        (_envelope_ref, ExperimentEnvelopeRejectionCategory.MISSING_FIELD, "ref"),
        (
            _base_domain,
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "pin_algorithm",
        ),
        (
            _base_digest,
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "content_hash",
        ),
        (
            _lineage_shape,
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "lineage is a list",
        ),
        (
            _lineage_pin,
            ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
            "content_hash",
        ),
        (_delta_key, ExperimentEnvelopeRejectionCategory.INVALID_VALUE, "own layer id"),
        (_delta_shape, ExperimentEnvelopeRejectionCategory.MISSING_FIELD, "patches"),
        (_delta_patch, ExperimentEnvelopeRejectionCategory.MISSING_FIELD, "op"),
        (
            _assertion_shape,
            ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
            "owner_ref",
        ),
        (
            _document_digest,
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "content_hash",
        ),
        (_document_family, ExperimentEnvelopeRejectionCategory.INVALID_VALUE, "family"),
        (
            _contract_version,
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "does not extend contract id",
        ),
        (
            _implementation_shape,
            ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
            "package_versions",
        ),
        (
            _implementation_code_unit,
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "code unit",
        ),
        (
            _identity_digest,
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "re-derive",
        ),
        (
            _identity_inputs,
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "names the facts it was built from",
        ),
    ],
)
def test_the_loader_refuses_a_lock_damaged_anywhere_in_the_v1_document(
    repo: Path, damage: Any, category: Any, match: str
) -> None:
    """A consumer that trusts a lock trusts all of it, so all of it is checked."""
    lock = _compiled_lock(repo)
    damage(lock)

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        load_compile_lock(lock, field="compiled/widened.compile-lock.json")

    assert excinfo.value.category is category
    assert match in str(excinfo.value)


def test_the_loader_accepts_every_compiled_lock_the_fixture_produces(
    repo: Path,
) -> None:
    """The validation is derived from what the compiler emits, not guessed at."""
    for alias in ("widened", "widened-probe", "widened-summary", "widened-plot"):
        lock = _compiled_lock(repo, alias)
        assert load_compile_lock(lock, field=f"compiled/{alias}.compile-lock.json") == lock


def test_an_identity_contribution_dropped_after_emission_is_refused(repo: Path) -> None:
    """The contributions and the identity they were hashed into must agree."""
    lock = _compiled_lock(repo, "widened-plot")
    assert set(lock["identity_contributions"]) == {
        "figure_row_expansion",
        "resolved_row_set",
        "row_custody",
    }
    del lock["identity_contributions"]["resolved_row_set"]

    with pytest.raises(ExperimentEnvelopeRejection, match="names the facts it was built from"):
        load_compile_lock(lock, field="compiled/widened-plot.compile-lock.json")


def test_an_empty_identity_contributions_block_is_refused(repo: Path) -> None:
    """The emitter omits the block when there is nothing in it."""
    lock = _compiled_lock(repo, "widened-plot")
    lock["identity_contributions"] = {}
    lock["execution_identity"]["inputs"] = ["compiled_document.content_hash"]

    with pytest.raises(ExperimentEnvelopeRejection, match="empty block is omitted"):
        load_compile_lock(lock, field="compiled/widened-plot.compile-lock.json")


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
    pinned = PinnedDocument.of(TRAINING_BASE, json.loads((repo / TRAINING_BASE).read_text()))

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


def test_lineage_pins_the_documents_a_parent_reads_through_its_sources(
    repo: Path,
) -> None:
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
        [
            {
                "alias": "cadence",
                "kind": "quillon.cadence_table",
                "uri": "bases/absent.json",
            }
        ],
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
    assert [patch["path"] for patch in patches] == [
        "name",
        "rows.1",
        "tags.1",
        "tags.0",
    ]


def test_a_cross_layer_reference_pins_only_pre_run_facts(repo: Path) -> None:
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-probe"), repo_root=repo)

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
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-summary"), repo_root=repo)

    reference = outcome.compile_lock["references"][0]
    assert reference["kind"] == "receipt_locator"
    assert "manifest_sha256" not in reference
    assert reference["consumer"] == {
        "consumer": "analysis_input",
        "alias": "probe",
        "role": "observations",
    }


def test_an_authored_receipt_with_a_digest_is_quoted_as_authenticated(
    repo: Path,
) -> None:
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


def test_authored_not_applicability_is_recorded_rather_than_left_silent(
    repo: Path,
) -> None:
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-report"), repo_root=repo)

    absent = next(
        item for item in outcome.compile_lock["references"] if item["kind"] == "not_applicable"
    )
    assert absent["basis"] == "authored"
    assert absent["role_path"] == "params.sections.0.tables.0"
    assert "rule_id" not in absent


def test_a_payload_in_its_own_file_is_recorded_as_a_content_pin(repo: Path) -> None:
    write_json(repo / "bases" / "survey.payload.json", _survey_payload())
    _repoint_training_base_to_file(repo)
    _reauthor(repo, "widened", **{"assert": []})

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    pin = next(item for item in outcome.compile_lock["references"] if item["kind"] == "content_pin")
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


def test_an_assertion_that_fails_names_the_document_that_owns_the_value(
    repo: Path,
) -> None:
    _reauthor(repo, "widened", **{"assert": [{"path": "base.inline.cadence", "equals": 99}]})

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.ASSERTION_FAILED
    assert TRAINING_BASE in str(excinfo.value)


def test_an_assertion_may_not_guard_a_path_the_envelope_changes(repo: Path) -> None:
    _reauthor(repo, "widened", **{"assert": [{"path": "name", "equals": "baseline"}]})

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is (ExperimentEnvelopeRejectionCategory.ILLEGAL_ASSERTION_PATH)


def test_an_assertion_on_an_uninherited_path_has_nothing_to_check(repo: Path) -> None:
    _reauthor(repo, "widened", **{"assert": [{"path": "base.inline.absent", "equals": 1}]})

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert "not inherited" in str(excinfo.value)


def test_an_echoed_inherited_name_is_refused(repo: Path) -> None:
    _reauthor(repo, "widened", name="baseline", **{"assert": []})

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is (ExperimentEnvelopeRejectionCategory.ECHOED_INHERITED_VALUE)
    assert TRAINING_BASE in str(excinfo.value)


def test_an_echoed_inherited_seed_is_refused(repo: Path) -> None:
    envelope = _read(repo, "widened")
    envelope["training"]["rows"][0]["seed"] = 42

    _write(repo, "widened", envelope)

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is (ExperimentEnvelopeRejectionCategory.ECHOED_INHERITED_VALUE)


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


def test_a_base_of_no_feedbax_output_family_is_not_an_experiment_parent(
    repo: Path,
) -> None:
    write_json(repo / "bases" / "stray.json", {"schema_id": "quillon.notes", "name": "stray"})
    _reauthor(repo, "widened", base="bases/stray.json", **{"assert": []})

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE


def test_an_envelope_alias_parent_resolves_and_pins_its_compiled_bytes(
    repo: Path,
) -> None:
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

    first = {path: path.read_bytes() for path in compiler.write_outputs(outcome, out_dir).values()}
    second = {path: path.read_bytes() for path in compiler.write_outputs(outcome, out_dir).values()}

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
        ["studies/widened.envelope.json"],
        "studies/widened.envelope.json",
        (".base.json",),
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
    assert [entry.path for entry in drift] == [f"{OUTPUT_DIRECTORY}/widened.{TRAINING_FAMILY}.json"]
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


def test_an_envelope_that_no_longer_compiles_is_a_finding_not_an_exception(
    repo: Path,
) -> None:
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


# -- v3 root training --------------------------------------------------------


def _lock_at_version_v1(lock: dict[str, Any]) -> dict[str, Any]:
    """Return one lock with only its own schema version restated as v1.

    The signed bases below were recorded when the compile lock family was at v1.
    v2 adds the optional typed artifact contract a figure runtime input binding
    carries, and nothing else: no other byte of any lock moved, and a prior
    envelope grammar still compiles to exactly the bytes it always did. Restating
    the one version field and comparing against the original base is what proves
    that, where re-signing the bases would only record whatever the code now
    emits.
    """
    return {**lock, "schema_version": EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V1}


def test_prior_and_authority_free_root_document_lock_bytes_match_signed_base(
    repo: Path,
) -> None:
    expected = {
        EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1: (
            "1d918888b54f08e5a15449e97621260332d931521b28cb2e3f6bc6f5db5b2af7"
        ),
        EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2: (
            "d6db3543cf5bdea2e4ab59f0a5934aa7e51629b2f0d304a318c7d31df257bf03"
        ),
        EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V3: (
            "a6edabd2ae82b750daf2b13c318620de76d98ae847c3139c6a3b73be9ef3235a"
        ),
    }
    path = envelope_path(repo, "training")
    for schema in (
        EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1,
        EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V2,
    ):
        document = json.loads(json.dumps(TRAINING_ENVELOPE))
        document["schema"] = schema
        write_envelope(path, document)
        outcome = kernel().compile_envelope_file(path, repo_root=repo)
        assert outcome.compile_lock["schema_version"] == (
            EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION_V2
        )
        assert (
            canonical_sha256(
                {"document": outcome.document, "lock": _lock_at_version_v1(outcome.compile_lock)}
            )
            == expected[schema]
        )

    write_envelope(
        path,
        {
            **_root_envelope(
                {
                    "kind": "composition",
                    "parent": {
                        "kind": "resolved_output",
                        "ref": "artifact-blob:generic-source",
                        "resolved_root_hash": "3" * 64,
                        "row_id": "source-row",
                        "checkpoint_transaction_id": "checkpoint-1",
                    },
                    "rows": [{"id": "condition-a"}],
                }
            ),
            "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V3,
        },
    )
    outcome = kernel().compile_envelope_file(path, repo_root=repo)
    assert (
        canonical_sha256(
            {"document": outcome.document, "lock": _lock_at_version_v1(outcome.compile_lock)}
        )
        == expected[EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V3]
    )


def test_authored_composition_root_preserves_both_hash_domains_and_root_identity(
    repo: Path,
) -> None:
    node = CompositionNode(
        name="generic-task-composition",
        parent=InlineIntentParent(
            payload={
                "schema_id": "quillon.training_intent",
                "schema_version": "quillon.training_intent.v1",
                "gain": 1,
            },
            schema_id="quillon.training_intent",
            schema_version="quillon.training_intent.v1",
        ),
    )
    document = node.model_dump(mode="json", exclude_none=True)
    ref = "intent/generic.composition.json"
    write_json(repo / ref, document)
    semantic_hash = authored_envelope_hash(node)
    canonical_hash = canonical_sha256(document)
    assert semantic_hash != canonical_hash

    root = {
        "kind": "composition",
        "parent": {
            "kind": "authored_intent",
            "ref": ref,
            "content_hash": semantic_hash,
            "symbolic_name": "generic-task-composition",
        },
        "deltas": [{"layer_id": "root-layer"}],
        "rows": [
            {
                "id": "condition-a",
                "label": "Condition A",
                "seed": 11,
                "delta": {"layer_id": "condition-a-layer"},
            }
        ],
        "checkpoint_initialization": [
            {
                "row": "condition-a",
                "mode": "initialize_from",
                "source": {
                    "kind": "receipt",
                    "manifest_kind": "quillon.training",
                    "manifest_id": "generic-parent",
                },
            }
        ],
        "tags": ["generic"],
    }

    outcome = _compile_root(repo, root)

    assert outcome.document["schema_version"] == "feedbax.spec.training_run_matrix.v6"
    assert outcome.document["base"] == {
        "kind": "authored_intent",
        "ref": ref,
        "content_hash": canonical_hash,
        "pin_algorithm": "canonical_json_v1",
        "symbolic_name": "generic-task-composition",
    }
    assert outcome.document["rows"] == [
        {
            "row_id": "condition-a",
            "label": "Condition A",
            "overrides": [],
            "seed": 11,
            "metadata": {},
        }
    ]
    lock = outcome.compile_lock
    assert lock["base"] is None
    assert lock["lineage"] == []
    assert lock["row_provenance"] == []
    assert set(lock["resolved_deltas"]) == {"root-layer", "condition-a-layer"}
    assert (
        lock["identity_contributions"]["training_root"]["parent"]["content_hash"] == semantic_hash
    )
    content_pin = next(item for item in lock["references"] if item["kind"] == "content_pin")
    assert content_pin["content_hash"] == canonical_hash
    assert any(
        item.get("consumer", {}).get("consumer") == "checkpoint_initialization"
        for item in lock["references"]
    )


def test_authored_composition_root_refuses_a_semantic_parent_hash_mismatch(
    repo: Path,
) -> None:
    node = CompositionNode(
        name="generic-composition",
        parent=InlineIntentParent(
            payload={
                "schema_id": "quillon.intent",
                "schema_version": "quillon.intent.v1",
            },
            schema_id="quillon.intent",
            schema_version="quillon.intent.v1",
        ),
    )
    ref = "intent/generic.composition.json"
    write_json(repo / ref, node.model_dump(mode="json", exclude_none=True))

    with pytest.raises(ExperimentEnvelopeRejection, match="authored composition hash mismatch"):
        _compile_root(
            repo,
            {
                "kind": "composition",
                "parent": {
                    "kind": "authored_intent",
                    "ref": ref,
                    "content_hash": "0" * 64,
                },
                "rows": [{"id": "condition-a"}],
            },
        )


def test_generated_schema_boundary_delta_round_trips_the_compile_lock(repo: Path) -> None:
    outcome = _compile_root(
        repo,
        {
            "kind": "composition",
            "parent": {
                "kind": "resolved_output",
                "ref": "artifact-blob:generic-source",
                "resolved_root_hash": "3" * 64,
                "row_id": "source-row",
                "checkpoint_transaction_id": "source-transaction",
            },
            "deltas": [
                {
                    "layer_id": "schema-boundary",
                    "schema_id": "quillon.training_intent",
                    "schema_version": "quillon.training_intent.v2",
                    "patches": [
                        {
                            "op": "replace",
                            "path": "schema_version",
                            "value": "quillon.training_intent.v2",
                        }
                    ],
                }
            ],
            "rows": [{"id": "condition-a"}],
        },
    )

    assert (
        load_compile_lock(
            outcome.compile_lock,
            field="generated/schema-boundary.compile-lock.json",
        )
        == outcome.compile_lock
    )

    missing = json.loads(json.dumps(outcome.compile_lock))
    del missing["resolved_deltas"]["schema-boundary"]["schema_version"]
    with pytest.raises(ExperimentEnvelopeRejection) as missing_error:
        load_compile_lock(missing, field="generated/schema-boundary.compile-lock.json")
    assert missing_error.value.category is ExperimentEnvelopeRejectionCategory.MISSING_FIELD

    blank = json.loads(json.dumps(outcome.compile_lock))
    blank["resolved_deltas"]["schema-boundary"]["schema_id"] = " "
    with pytest.raises(ExperimentEnvelopeRejection) as blank_error:
        load_compile_lock(blank, field="generated/schema-boundary.compile-lock.json")
    assert blank_error.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE


def test_resolved_output_composition_root_is_not_materialized_at_compile_time(
    repo: Path,
) -> None:
    immutable_ref = "artifact-blob:56756c94"
    root = {
        "kind": "composition",
        "parent": {
            "kind": "resolved_output",
            "ref": immutable_ref,
            "resolved_root_hash": "3" * 64,
            "row_id": "source-row",
            "checkpoint_transaction_id": "checkpoint-1",
        },
        "rows": [{"id": "condition-a"}],
    }

    outcome = _compile_root(repo, root)

    assert outcome.document["base"] == {
        "kind": "resolved_output",
        "ref": immutable_ref,
        "resolved_root_hash": "3" * 64,
        "row_id": "source-row",
        "checkpoint_transaction_id": "checkpoint-1",
    }
    assert outcome.compile_lock["references"] == []
    identity = outcome.compile_lock["identity_contributions"]["training_root"]["parent"]
    assert identity["row_id"] == "source-row"
    assert identity["checkpoint_transaction_id"] == "checkpoint-1"


def test_root_selected_checkpoint_lowers_exact_resolved_authority_and_barrier(
    repo: Path,
) -> None:
    root = {
        "kind": "composition",
        "parent": {
            "kind": "resolved_output",
            "ref": "artifact-blob:generic-source",
            "resolved_root_hash": "3" * 64,
            "row_id": "source-row",
            "checkpoint_transaction_id": "checkpoint-1",
        },
        "selected_checkpoint": {
            "source_run_id": "source-run",
            "checkpoint_root_hash": "4" * 64,
            "source_barrier": "after_segment",
        },
        "rows": [{"id": "condition-a"}],
        "fork": {
            "lr_continuation": "continue",
            "parity": "require",
            "absolute_lr_tolerance": 1e-12,
        },
    }

    outcome = _compile_root(repo, root)
    dependency = outcome.document["execution_dependencies"][0]
    assert dependency == {
        "kind": "fork_from_selected_checkpoint",
        "source_authority": {
            "kind": "resolved_output_root",
            "source_run_id": "source-run",
            "resolved_root_hash": "3" * 64,
        },
        "source_row_id": "source-row",
        "checkpoint_transaction_id": "checkpoint-1",
        "checkpoint_root_hash": "4" * 64,
        "source_barrier": "after_segment",
        "slot_transforms": [],
    }
    assert "execution_hash" not in str(dependency)
    first_hash = outcome.compile_lock["compiled_document"]["content_hash"]

    root["selected_checkpoint"]["source_barrier"] = "after_validation"
    second = _compile_root(repo, root)
    assert second.compile_lock["compiled_document"]["content_hash"] != first_hash


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_row",
        "missing_transaction",
        "missing_source_run",
        "checkpoint_initialization",
        "authored_parent",
    ],
)
def test_root_selected_checkpoint_refuses_incomplete_or_competing_authority(
    repo: Path, mutation: str
) -> None:
    parent = {
        "kind": "resolved_output",
        "ref": "artifact-blob:generic-source",
        "resolved_root_hash": "3" * 64,
        "row_id": "source-row",
        "checkpoint_transaction_id": "checkpoint-1",
    }
    root: dict[str, Any] = {
        "kind": "composition",
        "parent": parent,
        "selected_checkpoint": {
            "source_run_id": "source-run",
            "checkpoint_root_hash": "4" * 64,
            "source_barrier": "after_segment",
        },
        "rows": [{"id": "condition-a"}],
        "fork": {"lr_continuation": "continue", "parity": "require"},
    }
    if mutation == "missing_row":
        parent.pop("row_id")
    elif mutation == "missing_transaction":
        parent.pop("checkpoint_transaction_id")
    elif mutation == "missing_source_run":
        root["selected_checkpoint"].pop("source_run_id")
    elif mutation == "checkpoint_initialization":
        root["checkpoint_initialization"] = [
            {
                "row": "condition-a",
                "mode": "continue_from",
                "source": {
                    "kind": "receipt",
                    "manifest_kind": "generic.training",
                    "manifest_id": "source",
                },
            }
        ]
    else:
        root["parent"] = {
            "kind": "authored_intent",
            "ref": "intent/source.json",
            "content_hash": "3" * 64,
        }
    with pytest.raises(ExperimentEnvelopeRejection):
        _compile_root(repo, root)


def test_training_run_v4_root_is_canonical_pinned_and_emits_no_matrix_deltas(
    repo: Path,
) -> None:
    training_run = _standard_run_spec_payload()
    ref = "intent/generic.training_run.json"
    write_json(repo / ref, training_run)
    digest = canonical_sha256(training_run)

    outcome = _compile_root(
        repo,
        {
            "kind": "training_run",
            "ref": ref,
            "content_hash": digest,
            "rows": [{"id": "condition-a", "seed": 5}],
        },
    )

    assert outcome.document["base"]["content_hash"] == digest
    assert outcome.document["base"]["pin_algorithm"] == "canonical_json_v1"
    assert outcome.document["rows"][0]["label"] == "condition-a"
    assert outcome.document["rows"][0]["metadata"] == {}
    assert outcome.document["deltas"] == []
    assert outcome.compile_lock["resolved_deltas"] == {}
    assert outcome.compile_lock["references"] == [
        {
            "kind": "content_pin",
            "ref": ref,
            "content_hash": digest,
            "pin_algorithm": "canonical_json_v1",
        }
    ]


@pytest.mark.parametrize(
    "ref",
    [
        "/absolute/training.json",
        "../escaped.training.json",
        "intent/./noncanonical.training.json",
        "intent/not-json.txt",
    ],
)
def test_training_run_root_refuses_noncanonical_or_uncontained_refs(repo: Path, ref: str) -> None:
    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _compile_root(
            repo,
            {
                "kind": "training_run",
                "ref": ref,
                "content_hash": "4" * 64,
                "rows": [{"id": "condition-a"}],
            },
        )
    assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE


def test_training_run_root_refuses_missing_invalid_schema_and_pin_drift(
    repo: Path,
) -> None:
    with pytest.raises(ExperimentEnvelopeRejection) as missing:
        _compile_root(
            repo,
            {
                "kind": "training_run",
                "ref": "intent/missing.training.json",
                "content_hash": "4" * 64,
                "rows": [{"id": "condition-a"}],
            },
        )
    assert missing.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE
    assert missing.value.field == "training.root.ref"

    invalid = _standard_run_spec_payload()
    invalid["schema_version"] = "feedbax.spec.training_run.v3"
    ref = "intent/invalid.training.json"
    write_json(repo / ref, invalid)
    with pytest.raises(ExperimentEnvelopeRejection, match="training_run.v4") as schema:
        _compile_root(
            repo,
            {
                "kind": "training_run",
                "ref": ref,
                "content_hash": canonical_sha256(invalid),
                "rows": [{"id": "condition-a"}],
            },
        )
    assert schema.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert schema.value.field == "training.root.ref"

    valid = _standard_run_spec_payload()
    write_json(repo / ref, valid)
    with pytest.raises(ExperimentEnvelopeRejection, match="content hash mismatch") as drift:
        _compile_root(
            repo,
            {
                "kind": "training_run",
                "ref": ref,
                "content_hash": "4" * 64,
                "rows": [{"id": "condition-a"}],
            },
        )
    assert drift.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE
    assert drift.value.field == "training.root.content_hash"

    with pytest.raises(ExperimentEnvelopeRejection) as payload:
        _compile_root(
            repo,
            {
                "kind": "training_run",
                "ref": ref,
                "content_hash": canonical_sha256(valid),
                "payload_path": "missing.payload",
                "rows": [{"id": "condition-a"}],
            },
        )
    assert payload.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE
    assert payload.value.field == "training.root.payload_path"


def test_invalid_composition_document_is_a_closed_root_rejection(repo: Path) -> None:
    ref = "intent/invalid.composition.json"
    invalid = {
        "schema_id": "feedbax.spec.training_run_composition",
        "schema_version": "feedbax.spec.training_run_composition.v3",
        "name": "invalid",
        "parent": {
            "kind": "resolved_output",
            "ref": "artifact-blob:terminal",
            "resolved_root_hash": "6" * 64,
        },
    }
    write_json(repo / ref, invalid)
    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _compile_root(
            repo,
            {
                "kind": "composition",
                "parent": {
                    "kind": "authored_intent",
                    "ref": ref,
                    "content_hash": "7" * 64,
                },
                "rows": [{"id": "condition-a"}],
            },
        )
    assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert caught.value.field == "training.root.parent"


def test_validation_runtime_errors_are_not_reclassified_as_authoring_rejections(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = CompositionNode(
        name="generic",
        parent=InlineIntentParent(
            payload={
                "schema_id": "quillon.intent",
                "schema_version": "quillon.intent.v1",
            },
            schema_id="quillon.intent",
            schema_version="quillon.intent.v1",
        ),
    )
    ref = "intent/generic.composition.json"
    write_json(repo / ref, node.model_dump(mode="json", exclude_none=True))

    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("validator implementation failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(CompositionNode, "model_validate", classmethod(explode))
        with pytest.raises(RuntimeError, match="validator implementation failed"):
            _compile_root(
                repo,
                {
                    "kind": "composition",
                    "parent": {
                        "kind": "authored_intent",
                        "ref": ref,
                        "content_hash": authored_envelope_hash(node),
                    },
                    "rows": [{"id": "condition-a"}],
                },
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(TRAINING_OUTPUT_V6.model(), "model_validate", classmethod(explode))
        with pytest.raises(RuntimeError, match="validator implementation failed"):
            _compile_root(
                repo,
                {
                    "kind": "composition",
                    "parent": {
                        "kind": "resolved_output",
                        "ref": "artifact-blob:terminal",
                        "resolved_root_hash": "8" * 64,
                        "row_id": "source-row",
                        "checkpoint_transaction_id": "checkpoint-1",
                    },
                    "rows": [{"id": "condition-a"}],
                },
            )

    report = kernel().compile_envelope_file(envelope_path(repo, "widened-report"), repo_root=repo)
    params_model = REPORT_OUTPUT.params_model(report.document)
    assert params_model is not None
    with monkeypatch.context() as scoped:
        scoped.setattr(params_model, "model_validate", classmethod(explode))
        with pytest.raises(RuntimeError, match="validator implementation failed"):
            kernel().compile_envelope_file(envelope_path(repo, "widened-report"), repo_root=repo)


def test_root_sources_are_closed_unique_contained_and_pinned_when_present(
    repo: Path,
) -> None:
    training_run = _standard_run_spec_payload()
    ref = "intent/generic.training_run.json"
    write_json(repo / ref, training_run)
    digest = canonical_sha256(training_run)
    source_ref = "inputs/generic.json"
    write_json(repo / source_ref, {"value": 1})
    base_root = {
        "kind": "training_run",
        "ref": ref,
        "content_hash": digest,
        "rows": [{"id": "condition-a"}],
    }

    outcome = _compile_root(
        repo,
        {
            **base_root,
            "sources": [
                {"alias": "present", "kind": "json", "uri": source_ref},
                {
                    "alias": "optional",
                    "kind": "json",
                    "uri": "inputs/optional.json",
                    "optional": True,
                    "missing_payload": {},
                },
            ],
        },
    )
    assert [item["ref"] for item in outcome.compile_lock["references"]] == [
        ref,
        source_ref,
    ]

    with pytest.raises(ExperimentEnvelopeRejection, match="cannot be pinned"):
        _compile_root(
            repo,
            {
                **base_root,
                "sources": [{"alias": "required", "kind": "json", "uri": "inputs/missing.json"}],
            },
        )
    with pytest.raises(ExperimentEnvelopeRejection, match="canonical repository-relative"):
        _compile_root(
            repo,
            {
                **base_root,
                "sources": [
                    {
                        "alias": "optional",
                        "kind": "json",
                        "uri": "../optional.json",
                        "optional": True,
                        "missing_payload": {},
                    }
                ],
            },
        )
    with pytest.raises(ExperimentEnvelopeRejection) as duplicate:
        _compile_root(
            repo,
            {
                **base_root,
                "sources": [
                    {"alias": "same", "kind": "json", "uri": source_ref},
                    {"alias": "same", "kind": "json", "uri": source_ref},
                ],
            },
        )
    assert duplicate.value.category is ExperimentEnvelopeRejectionCategory.DUPLICATE_KEY
    assert duplicate.value.field == "training.root.sources[1].alias"


def test_root_derivation_validation_maps_to_exact_authored_fields(repo: Path) -> None:
    training_run = _standard_run_spec_payload()
    ref = "intent/generic.training_run.json"
    write_json(repo / ref, training_run)
    base_root = {
        "kind": "training_run",
        "ref": ref,
        "content_hash": canonical_sha256(training_run),
        "rows": [{"id": "condition-a"}],
    }
    duplicate_derivations = [
        {"output_path": "derived.value", "query": {"item": "source"}},
        {"output_path": "derived.value", "query": {"item": "source"}},
    ]
    with pytest.raises(ExperimentEnvelopeRejection) as duplicate:
        _compile_root(repo, {**base_root, "derivations": duplicate_derivations})
    assert duplicate.value.category is ExperimentEnvelopeRejectionCategory.DUPLICATE_KEY
    assert duplicate.value.field == "training.root.derivations[1].output_path"

    with pytest.raises(ExperimentEnvelopeRejection) as no_source:
        _compile_root(
            repo,
            {
                **base_root,
                "derivations": [duplicate_derivations[0]],
            },
        )
    assert no_source.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert no_source.value.field == "training.root.derivations"


def test_root_authority_is_selected_flattened_and_shared_by_two_consumers(repo: Path) -> None:
    authority_source_ref = "inputs/shared-items.json"
    local_source_ref = "inputs/local-items.json"
    write_json(repo / authority_source_ref, {"items": [1, 2]})
    write_json(repo / local_source_ref, {"items": [3]})
    authority = {
        "schema_id": "feedbax.spec.root_training_authority",
        "schema_version": "feedbax.spec.root_training_authority.v1",
        "sources": [{"alias": "shared", "kind": "json", "uri": authority_source_ref}],
        "derivations": [
            {
                "output_path": "method_payload.payload.metadata.shared",
                "query": {
                    "kind": "map_object_list",
                    "items": {"item": "shared", "path": "items"},
                    "template": {"fixed": True},
                    "item_output_path": "value",
                },
            }
        ],
    }
    authority_ref = "authorities/shared-root.json"
    whole_document = {"selected": authority, "unselected": {"sentinel": True}}
    write_json(repo / authority_ref, whole_document)

    training_run = _standard_run_spec_payload()
    run_ref = "intent/generic.training_run.json"
    write_json(repo / run_ref, training_run)
    root = {
        "kind": "training_run",
        "ref": run_ref,
        "content_hash": canonical_sha256(training_run),
        "rows": [{"id": "condition-a"}],
        "authority": {
            "ref": authority_ref,
            "sha256": canonical_sha256(whole_document),
            "payload_path": ["selected"],
        },
        "sources": [{"alias": "local", "kind": "json", "uri": local_source_ref}],
        "derivations": [
            {
                "output_path": "method_payload.payload.metadata.local",
                "query": {"item": "local", "path": "items"},
            }
        ],
    }

    first = _compile_root(repo, root)
    second = _compile_root(repo, {**root, "rows": [{"id": "condition-b"}]})
    assert (
        first.document["sources"]
        == second.document["sources"]
        == [
            {**authority["sources"][0], "optional": False},
            {**root["sources"][0], "optional": False},
        ]
    )
    assert (
        first.document["derivations"]
        == second.document["derivations"]
        == [
            authority["derivations"][0],
            root["derivations"][0],
        ]
    )
    assert "authority" not in first.document
    assert (
        first.compile_lock["identity_contributions"]["training_root"]["authority"]
        == (root["authority"])
    )
    pins = {
        (item["ref"], item["content_hash"])
        for item in first.compile_lock["references"]
        if item["kind"] == "content_pin"
    }
    assert (authority_ref, canonical_sha256(whole_document)) in pins
    assert (authority_source_ref, canonical_sha256({"items": [1, 2]})) in pins
    assert (local_source_ref, canonical_sha256({"items": [3]})) in pins
    assert "map_object_list" not in json.dumps(root)


@pytest.mark.parametrize(
    "mutation,category,field,match",
    [
        (
            "wrong_hash",
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
            "training.root.authority",
            "hash mismatch",
        ),
        (
            "escape",
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "training.root.authority.ref",
            "canonical repository-relative",
        ),
        (
            "missing_file",
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
            "training.root.authority",
            "cannot load",
        ),
        (
            "missing_path",
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
            "training.root.authority.payload_path",
            "missing object key",
        ),
        (
            "nonobject_path",
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "training.root.authority.payload_path",
            "must select a JSON object",
        ),
        (
            "missing_schema",
            ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
            "training.root.authority",
            "root_training_authority.v1",
        ),
        (
            "wrong_schema",
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "training.root.authority",
            "root_training_authority",
        ),
        (
            "wrong_version",
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "training.root.authority",
            "root_training_authority.v1",
        ),
        (
            "malformed_source",
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "training.root.authority",
            "uri",
        ),
        (
            "unknown_field",
            ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD,
            "training.root.authority",
            "Extra inputs are not permitted",
        ),
    ],
)
def test_root_authority_refuses_invalid_pin_selection_or_closed_schema(
    repo: Path,
    mutation: str,
    category: ExperimentEnvelopeRejectionCategory,
    field: str,
    match: str,
) -> None:
    authority = {
        "schema_id": "feedbax.spec.root_training_authority",
        "schema_version": "feedbax.spec.root_training_authority.v1",
        "sources": [],
        "derivations": [],
    }
    whole_document: dict[str, Any] = {"selected": authority, "scalar": 1}
    authority_ref = "authorities/shared-root.json"
    write_json(repo / authority_ref, whole_document)
    authority_authoring: dict[str, Any] = {
        "ref": authority_ref,
        "sha256": canonical_sha256(whole_document),
        "payload_path": ["selected"],
    }
    if mutation == "wrong_hash":
        authority_authoring["sha256"] = "0" * 64
    elif mutation == "escape":
        authority_authoring["ref"] = "../shared-root.json"
    elif mutation == "missing_file":
        authority_authoring["ref"] = "authorities/missing.json"
    elif mutation == "missing_path":
        authority_authoring["payload_path"] = ["missing"]
    elif mutation == "nonobject_path":
        authority_authoring["payload_path"] = ["scalar"]
    elif mutation == "missing_schema":
        authority.pop("schema_id")
    elif mutation == "wrong_schema":
        authority["schema_id"] = "feedbax.spec.unknown"
    elif mutation == "wrong_version":
        authority["schema_version"] = "feedbax.spec.root_training_authority.v2"
    elif mutation == "malformed_source":
        authority["sources"] = [{"alias": "broken", "kind": "json"}]
    else:
        authority["invented"] = True
    if mutation in {
        "missing_schema",
        "wrong_schema",
        "wrong_version",
        "malformed_source",
        "unknown_field",
    }:
        authority_authoring["sha256"] = canonical_sha256(whole_document)
        write_json(repo / authority_ref, whole_document)

    with pytest.raises(ExperimentEnvelopeRejection, match=match) as caught:
        _compile_root(
            repo,
            {
                "kind": "composition",
                "parent": {
                    "kind": "resolved_output",
                    "ref": "artifact-blob:generic-source",
                    "resolved_root_hash": "3" * 64,
                    "row_id": "source-row",
                    "checkpoint_transaction_id": "checkpoint-1",
                },
                "rows": [{"id": "condition-a"}],
                "authority": authority_authoring,
            },
        )
    assert caught.value.category is category
    assert caught.value.field == field


def test_root_authority_refuses_absolute_ref_and_unpinnable_import(repo: Path) -> None:
    base_root = {
        "kind": "composition",
        "parent": {
            "kind": "resolved_output",
            "ref": "artifact-blob:generic-source",
            "resolved_root_hash": "3" * 64,
            "row_id": "source-row",
            "checkpoint_transaction_id": "checkpoint-1",
        },
        "rows": [{"id": "condition-a"}],
    }
    with pytest.raises(ExperimentEnvelopeRejection, match="relative path") as absolute:
        _compile_root(
            repo,
            {
                **base_root,
                "authority": {"ref": "/authority.json", "sha256": "0" * 64},
            },
        )
    assert absolute.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE

    authority = {
        "schema_id": "feedbax.spec.root_training_authority",
        "schema_version": "feedbax.spec.root_training_authority.v1",
        "sources": [{"alias": "missing", "kind": "json", "uri": "inputs/missing.json"}],
        "derivations": [],
    }
    authority_ref = "authorities/shared-root.json"
    write_json(repo / authority_ref, authority)
    with pytest.raises(ExperimentEnvelopeRejection, match="cannot be pinned"):
        _compile_root(
            repo,
            {
                **base_root,
                "authority": {"ref": authority_ref, "sha256": canonical_sha256(authority)},
            },
        )


@pytest.mark.parametrize(
    "kind,local,expected_field",
    [
        (
            "alias",
            False,
            "training.root.authority.sources[1].alias",
        ),
        (
            "alias",
            True,
            "training.root.sources[0].alias",
        ),
        (
            "output",
            False,
            "training.root.authority.derivations[1].output_path",
        ),
        (
            "output",
            True,
            "training.root.derivations[0].output_path",
        ),
    ],
)
def test_root_authority_refuses_internal_and_cross_boundary_collisions_before_pinning(
    repo: Path,
    kind: str,
    local: bool,
    expected_field: str,
) -> None:
    authority = {
        "schema_id": "feedbax.spec.root_training_authority",
        "schema_version": "feedbax.spec.root_training_authority.v1",
        "sources": [{"alias": "shared", "kind": "json", "uri": "inputs/missing.json"}],
        "derivations": [
            {"output_path": "payload.shared", "query": {"item": "shared", "path": "items"}}
        ],
    }
    if not local and kind == "alias":
        authority["sources"].append(
            {"alias": "shared", "kind": "json", "uri": "inputs/also-missing.json"}
        )
    if not local and kind == "output":
        authority["derivations"].append(
            {"output_path": "payload.shared", "query": {"item": "other", "path": "items"}}
        )
    authority_ref = "authorities/shared-root.json"
    write_json(repo / authority_ref, authority)
    root: dict[str, Any] = {
        "kind": "composition",
        "parent": {
            "kind": "resolved_output",
            "ref": "artifact-blob:generic-source",
            "resolved_root_hash": "3" * 64,
            "row_id": "source-row",
            "checkpoint_transaction_id": "checkpoint-1",
        },
        "rows": [{"id": "condition-a"}],
        "authority": {"ref": authority_ref, "sha256": canonical_sha256(authority)},
    }
    if local and kind == "alias":
        root["sources"] = [{"alias": "shared", "kind": "json", "uri": "inputs/local-missing.json"}]
    if local and kind == "output":
        root["derivations"] = [
            {"output_path": "payload.shared", "query": {"item": "local", "path": "items"}}
        ]

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _compile_root(repo, root)
    assert caught.value.category is ExperimentEnvelopeRejectionCategory.DUPLICATE_KEY
    assert caught.value.field == expected_field


def test_root_list_projection_is_exact_and_binds_both_compact_source_pins(
    repo: Path,
) -> None:
    primary_points = [
        [0.15, 0.0],
        [0.149429204714, 0.013073361412],
        [0.147721162952, 0.02604722665],
        [0.144888873943, 0.038822856765],
        [0.140953893118, 0.051303021499],
        [0.135946168055, 0.063392739261],
        [0.129903810568, 0.075],
        [0.086036465453, 0.122872806643],
        [0.075, 0.129903810568],
        [0.063392739261, 0.135946168055],
        [0.051303021499, 0.140953893118],
        [0.038822856765, 0.144888873943],
        [0.02604722665, 0.147721162952],
        [0.013073361412, 0.149429204714],
        [0.0, 0.15],
        [-0.013073361412, 0.149429204714],
        [-0.02604722665, 0.147721162952],
        [-0.038822856765, 0.144888873943],
        [-0.051303021499, 0.140953893118],
        [-0.063392739261, 0.135946168055],
        [-0.075, 0.129903810568],
        [-0.122872806643, 0.086036465453],
        [-0.129903810568, 0.075],
        [-0.135946168055, 0.063392739261],
        [-0.140953893118, 0.051303021499],
        [-0.144888873943, 0.038822856765],
        [-0.147721162952, 0.02604722665],
        [-0.149429204714, 0.013073361412],
        [-0.15, 0.0],
        [-0.149429204714, -0.013073361412],
        [-0.147721162952, -0.02604722665],
        [-0.144888873943, -0.038822856765],
        [-0.140953893118, -0.051303021499],
        [-0.135946168055, -0.063392739261],
        [-0.129903810568, -0.075],
        [-0.086036465453, -0.122872806643],
        [-0.075, -0.129903810568],
        [-0.063392739261, -0.135946168055],
        [-0.051303021499, -0.140953893118],
        [-0.038822856765, -0.144888873943],
        [-0.02604722665, -0.147721162952],
        [-0.013073361412, -0.149429204714],
        [-0.0, -0.15],
        [0.013073361412, -0.149429204714],
        [0.02604722665, -0.147721162952],
        [0.038822856765, -0.144888873943],
        [0.051303021499, -0.140953893118],
        [0.063392739261, -0.135946168055],
        [0.075, -0.129903810568],
        [0.122872806643, -0.086036465453],
        [0.129903810568, -0.075],
        [0.135946168055, -0.063392739261],
        [0.140953893118, -0.051303021499],
        [0.144888873943, -0.038822856765],
        [0.147721162952, -0.02604722665],
        [0.149429204714, -0.013073361412],
    ]
    reserved_points = [
        [0.122872806643, 0.086036465453],
        [0.114906666468, 0.096418141453],
        [0.106066017178, 0.106066017178],
        [0.096418141453, 0.114906666468],
        [-0.086036465453, 0.122872806643],
        [-0.096418141453, 0.114906666468],
        [-0.106066017178, 0.106066017178],
        [-0.114906666468, 0.096418141453],
        [-0.122872806643, -0.086036465453],
        [-0.114906666468, -0.096418141453],
        [-0.106066017178, -0.106066017178],
        [-0.096418141453, -0.114906666468],
        [0.086036465453, -0.122872806643],
        [0.096418141453, -0.114906666468],
        [0.106066017178, -0.106066017178],
        [0.114906666468, -0.096418141453],
    ]
    source_documents = {
        "inputs/primary-points.json": {"points": primary_points},
        "inputs/reserved-points.json": {"points": reserved_points},
    }
    for ref, document in source_documents.items():
        write_json(repo / ref, document)

    training_run = _standard_run_spec_payload()
    ref = "intent/generic.training_run.json"
    write_json(repo / ref, training_run)
    template = {
        "initial": {"encoding": "constant", "shape": [36], "value": 0.0},
        "signal": {"encoding": "constant", "shape": [60, 1], "value": 1.0},
    }
    authority = {
        "schema_id": "feedbax.spec.root_training_authority",
        "schema_version": "feedbax.spec.root_training_authority.v1",
        "sources": [
            {"alias": "primary", "kind": "json", "uri": "inputs/primary-points.json"},
            {"alias": "reserved", "kind": "json", "uri": "inputs/reserved-points.json"},
        ],
        "derivations": [
            {
                "output_path": "method_payload.payload.metadata.primary_records",
                "query": {
                    "kind": "map_object_list",
                    "items": {"item": "primary", "path": "points"},
                    "template": template,
                    "item_output_path": "target",
                },
            },
            {
                "output_path": "method_payload.payload.metadata.reserved_records",
                "query": {
                    "kind": "map_object_list",
                    "items": {"item": "reserved", "path": "points"},
                    "template": template,
                    "item_output_path": "target",
                },
            },
        ],
    }
    authority_ref = "authorities/record-geometry.json"
    write_json(repo / authority_ref, authority)
    outcome = _compile_root(
        repo,
        {
            "kind": "training_run",
            "ref": ref,
            "content_hash": canonical_sha256(training_run),
            "rows": [{"id": "condition-a"}],
            "authority": {"ref": authority_ref, "sha256": canonical_sha256(authority)},
        },
    )

    pins = {
        item["ref"]: item["content_hash"]
        for item in outcome.compile_lock["references"]
        if item["kind"] == "content_pin" and item["ref"] in source_documents
    }
    assert pins == {
        "inputs/primary-points.json": (
            "ab3ae3941afb7594964d805787ace3d4647fb52d0856156ea037e5e49a251f0f"
        ),
        "inputs/reserved-points.json": (
            "c251b5f6eb96220a8f18dc2e80d5f726528d3814e7a5635476ed4758d531230c"
        ),
    }

    seen_by_lowerer: list[dict[str, Any]] = []

    def lower(authored_row: Any, _context: Any) -> None:
        seen_by_lowerer.append(authored_row.payload)
        return None

    materialize_adapted_run_matrix(
        outcome.document,
        repo_root=repo,
        row_lowerer=lower,
        row_validator=lambda _payload, _row_id: None,
    )
    metadata = seen_by_lowerer[0]["method_payload"]["payload"]["metadata"]

    def expected(points: list[list[float]]) -> list[dict[str, Any]]:
        return [{**template, "target": point} for point in points]

    assert len(metadata["primary_records"]) == 56
    assert len(metadata["reserved_records"]) == 16
    assert metadata["primary_records"] == expected(primary_points)
    assert metadata["reserved_records"] == expected(reserved_points)

    def reject_mapped_records(_authored_row: Any, _context: Any) -> None:
        raise ValueError("row lowerer rejected mapped records")

    with pytest.raises(ValueError, match="row lowerer rejected mapped records"):
        materialize_adapted_run_matrix(
            outcome.document,
            repo_root=repo,
            row_lowerer=reject_mapped_records,
            row_validator=lambda _payload, _row_id: None,
        )

    source_documents["inputs/primary-points.json"]["points"][0] = [9.0, 9.0]
    write_json(
        repo / "inputs/primary-points.json",
        source_documents["inputs/primary-points.json"],
    )
    changed = _compile_root(
        repo,
        json.loads(envelope_path(repo, "rooted").read_text())["training"]["root"],
    )
    changed_pins = {
        item["ref"]: item["content_hash"]
        for item in changed.compile_lock["references"]
        if item["kind"] == "content_pin" and item["ref"] in source_documents
    }
    assert changed_pins["inputs/primary-points.json"] != pins["inputs/primary-points.json"]
    assert changed_pins["inputs/reserved-points.json"] == pins["inputs/reserved-points.json"]


def test_root_output_validation_maps_invalid_row_and_payload_syntax(repo: Path) -> None:
    training_run = _standard_run_spec_payload()
    ref = "intent/generic.training_run.json"
    write_json(repo / ref, training_run)
    digest = canonical_sha256(training_run)
    with pytest.raises(ExperimentEnvelopeRejection) as row:
        _compile_root(
            repo,
            {
                "kind": "training_run",
                "ref": ref,
                "content_hash": digest,
                "rows": [{"id": "not/path-safe"}],
            },
        )
    assert row.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert row.value.field == "training.root.rows[0].id"

    with pytest.raises(ExperimentEnvelopeRejection) as payload:
        _compile_root(
            repo,
            {
                "kind": "training_run",
                "ref": ref,
                "content_hash": digest,
                "payload_path": "graph..inline",
                "rows": [{"id": "condition-a"}],
            },
        )
    assert payload.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert payload.value.field == "training.root.payload_path"


def test_root_checkpoint_envelope_binding_is_lock_only_and_rows_are_exact(
    repo: Path,
) -> None:
    root = {
        "kind": "composition",
        "parent": {
            "kind": "resolved_output",
            "ref": "artifact-blob:terminal",
            "resolved_root_hash": "5" * 64,
            "row_id": "source-row",
            "checkpoint_transaction_id": "checkpoint-1",
        },
        "rows": [{"id": "condition-a"}],
        "checkpoint_initialization": [
            {
                "row": "condition-a",
                "mode": "continue_from",
                "source": {"kind": "envelope", "alias": "widened-summary"},
            }
        ],
    }
    outcome = _compile_root(repo, root)
    assert "checkpoint_initialization" not in outcome.document
    planned = outcome.compile_lock["references"][0]
    assert planned["kind"] == "planned_product"
    assert planned["consumer"]["consumer"] == "checkpoint_initialization"

    root["checkpoint_initialization"][0]["row"] = "missing"
    with pytest.raises(ExperimentEnvelopeRejection, match="absent from this root"):
        _compile_root(repo, root)
    root["checkpoint_initialization"] = [
        {
            "row": "condition-a",
            "mode": "continue_from",
            "source": {"kind": "envelope", "alias": "widened-summary"},
        },
        {
            "row": "condition-a",
            "mode": "initialize_from",
            "source": {
                "kind": "receipt",
                "manifest_kind": "quillon.training",
                "manifest_id": "other",
            },
        },
    ]
    with pytest.raises(ExperimentEnvelopeRejection, match="at most one"):
        _compile_root(repo, root)


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

    assert [row["row_id"] for row in outcome.document["rows"]] == [
        "baseline",
        "widened",
    ]


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
            "source_content_hash": canonical_sha256(json.loads((repo / TRAINING_BASE).read_text())),
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


def test_a_new_row_is_labelled_by_its_own_id_rather_than_its_sources(
    repo: Path,
) -> None:
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
    assert rows["baseline"]["metadata"] == {
        "probe_delta": "none",
        "launch_set": "survey-1",
    }
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


# -- v4 analysis/figure layer roots -------------------------------------------


def test_all_four_layer_root_kinds_compile_with_one_pin_and_no_parent(repo: Path) -> None:
    run = _compile_layer_root(
        repo,
        "root-analysis-run",
        _layer_root_authority(
            "analysis_run",
            input_requirements=[],
            evaluation_states_policy="require_durable",
            params={"window": 4},
        ),
        {
            "target": "run",
            "recipe": "quillon.span_summary",
            "params": {"stride": 2},
        },
    )
    bundle = _compile_layer_root(
        repo,
        "root-analysis-bundle",
        _layer_root_authority(
            "analysis_bundle",
            description="generic bundle",
            predicate={"manifest_kind": "EvaluationRunManifest"},
            templates=[],
            params_base={"params": {"trim": 1}},
            stages=[],
            metadata={},
        ),
        {"target": "bundle"},
    )
    figure = _compile_layer_root(
        repo,
        "root-figure",
        _layer_root_authority(
            "figure",
            assembler="quillon.span_assembler",
            assembler_params={"height": 300},
            panels=[{"name": "span", "title": "span", "row": 1, "col": 1}],
        ),
        {
            "mode": "root",
            "inputs": [
                {
                    "input_role": "summary",
                    "ref": {
                        "kind": "receipt",
                        "manifest_kind": "quillon.analysis_run",
                        "manifest_id": "summary-1",
                        "manifest_sha256": "7" * 64,
                        "size_bytes": 4096,
                    },
                }
            ],
            "delta": {
                "layer_id": "figure-height",
                "patches": [{"path": "assembler_params.height", "op": "replace", "value": 450}],
            },
        },
    )
    comparison = _compile_layer_root(
        repo,
        "root-comparison-policy",
        _comparison_policy_authority(),
        {},
        layer_name="comparison",
    )

    assert run.document == {
        "schema_id": "feedbax.spec.analysis_run",
        "schema_version": "feedbax.spec.analysis_run.v2",
        "analysis_type": "quillon.span_summary",
        "inputs": [],
        "input_requirements": [],
        "evaluation_states_policy": "require_durable",
        "params": {"window": 4, "stride": 2},
    }
    assert bundle.document["schema_version"] == "feedbax.spec.analysis_bundle.v6"
    assert bundle.document["name"] == "root-analysis-bundle"
    assert bundle.document["params_base"]["params"] == {"trim": 1}
    assert figure.document["schema_version"] == "feedbax.spec.figure.v2"
    assert figure.document["name"] == "root-figure"
    assert figure.document["inputs"] == []
    assert figure.document["input_authorities"] == []
    assert figure.document["assembler_params"]["height"] == 450
    assert comparison.document["schema_id"] == "feedbax.spec.comparison_policy"
    assert comparison.document["schema_version"] == "feedbax.spec.comparison_policy.v1"
    assert comparison.document["name"] == "root-comparison-policy"
    assert list(comparison.document["roles"]) == ["reference", "candidate"]
    assert comparison.document["comparison_policy"]["mismatch_policy"] == "fail_closed"
    assert canonical_sha256(comparison.document) == (
        "06f6eb8fb69efdbe29f089aecf4fde289b8fa14f31f3fc38b8f81a5943d35bf8"
    )
    assert canonical_sha256(_lock_at_version_v1(comparison.compile_lock)) == (
        "d5091f4194cb7c0030becce6ef7f7f9a016bbfe0d174311dfa373b62249d629e"
    )

    for outcome, kind in (
        (run, "analysis_run"),
        (bundle, "analysis_bundle"),
        (figure, "figure"),
        (comparison, "comparison_policy"),
    ):
        lock = outcome.compile_lock
        assert lock["base"] is None
        assert lock["lineage"] == []
        pins = [item for item in lock["references"] if item["kind"] == "content_pin"]
        assert len(pins) == 1
        identity = lock["identity_contributions"]["layer_root"]
        assert identity["kind"] == kind
        assert identity["sha256"] == pins[0]["content_hash"]
        assert len(identity["selected_authority_sha256"]) == 64
        assert lock["compiler_contract"]["contract_version"] == (
            EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION_V3
        )
    assert any(
        reference["kind"] == "authenticated_receipt"
        for reference in figure.compile_lock["references"]
    )


def test_v3_compilation_keeps_compiler_v2_bytes(repo: Path) -> None:
    envelope = _read(repo, "widened")
    envelope["schema"] = EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V3
    _write(repo, "widened", envelope)

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert outcome.compile_lock["compiler_contract"]["contract_version"] == (
        EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION_V2
    )


def test_v4_compilation_keeps_compiler_v3_bytes(repo: Path) -> None:
    """A v4 root figure still compiles under the compiler its version owns."""
    envelope = _read(repo, "widened")
    envelope["schema"] = EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V4
    _write(repo, "widened", envelope)

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened"), repo_root=repo)

    assert outcome.compile_lock["compiler_contract"]["contract_version"] == (
        EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION_V3
    )


def test_a_current_root_figure_carries_its_input_contracts_into_the_lock(repo: Path) -> None:
    """The contract is a plan fact, so it lands in the lock and nowhere else."""
    outcome = _compile_layer_root(
        repo,
        "contracted-root-figure",
        _layer_root_authority(
            "figure",
            assembler="quillon.span_assembler",
            assembler_params={"height": 300},
            panels=[{"name": "span", "title": "span", "row": 1, "col": 1}],
        ),
        {
            "mode": "root",
            "inputs": [
                {
                    "input_role": "summary",
                    "ref": {
                        "kind": "receipt",
                        "manifest_kind": "AnalysisRunManifest",
                        "manifest_id": "summary-1",
                        "manifest_sha256": "7" * 64,
                        "size_bytes": 4096,
                    },
                    "contract": {
                        "artifact_role": "result",
                        "artifact_provider": "quillon.custody",
                        "payload_name": "summary",
                        "payload_schema_id": "quillon.span_result",
                        "payload_schema_version": "quillon.span_result.v1",
                    },
                }
            ],
        },
        schema=EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V5,
    )

    assert outcome.compile_lock["compiler_contract"]["contract_version"] == (
        EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION
    )
    assert outcome.document["inputs"] == []
    assert outcome.document["input_authorities"] == []
    (binding,) = [
        reference["consumer"]
        for reference in outcome.compile_lock["references"]
        if reference.get("consumer", {}).get("consumer") == "figure_runtime_input"
    ]
    assert binding["contract"]["input_role"] == "summary"
    assert binding["contract"]["payload_name"] == "summary"
    assert binding["contract"]["payload_schema_id"] == "quillon.span_result"


def test_comparison_root_selector_changes_semantic_identity_after_whole_file_pin(
    repo: Path,
) -> None:
    first = _comparison_policy_authority()
    second = json.loads(json.dumps(first))
    second["roles"]["candidate"]["label"] = "Alternate"
    whole = {"members": {"first": first, "second": second}, "padding": "x" * 6_000}
    outcome = _compile_layer_root(
        repo,
        "selected-comparison",
        first,
        {},
        layer_name="comparison",
        payload_path=["members", "first"],
        whole_document=whole,
    )

    identity = outcome.compile_lock["identity_contributions"]["layer_root"]
    assert identity == {
        "kind": "comparison_policy",
        "ref": "authorities/selected-comparison.json",
        "sha256": canonical_sha256(whole),
        "payload_path": ["members", "first"],
        "selected_authority_sha256": canonical_sha256(first),
    }
    assert len(envelope_path(repo, "selected-comparison").read_bytes()) < 2_048
    assert len((repo / "authorities/selected-comparison.json").read_bytes()) > 6_000

    envelope = _read(repo, "selected-comparison")
    envelope["comparison"]["root"]["payload_path"] = ["members", "second"]
    _write(repo, "selected-comparison", envelope)
    selected_second = kernel().compile_envelope_file(
        envelope_path(repo, "selected-comparison"), repo_root=repo
    )
    assert selected_second.document["roles"]["candidate"]["label"] == "Alternate"
    assert selected_second.compile_lock["identity_contributions"]["layer_root"][
        "selected_authority_sha256"
    ] == canonical_sha256(second)

    envelope["comparison"]["root"]["sha256"] = "0" * 64
    _write(repo, "selected-comparison", envelope)
    with pytest.raises(ExperimentEnvelopeRejection) as wrong_pin:
        kernel().compile_envelope_file(envelope_path(repo, "selected-comparison"), repo_root=repo)
    assert wrong_pin.value.category is ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE


def test_comparison_root_kind_and_typed_shape_fail_closed(repo: Path) -> None:
    with pytest.raises(ExperimentEnvelopeRejection) as mismatch:
        _compile_layer_root(
            repo,
            "comparison-wrong-kind",
            _layer_root_authority("figure", assembler="quillon.panel_assembler"),
            {},
            layer_name="comparison",
        )
    assert mismatch.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert "requires authority kind 'comparison_policy'" in str(mismatch.value)

    malformed = _comparison_policy_authority()
    malformed["roles"].pop("candidate")
    with pytest.raises(ExperimentEnvelopeRejection) as invalid:
        _compile_layer_root(
            repo,
            "comparison-invalid-shape",
            malformed,
            {},
            layer_name="comparison",
        )
    assert invalid.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert "at least two source roles" in str(invalid.value)


def test_selector_is_verified_after_whole_file_and_changes_selected_identity(repo: Path) -> None:
    first = _layer_root_authority("analysis_run", params={"window": 3})
    second = _layer_root_authority("analysis_run", params={"window": 5})
    whole = {"members": {"first": first, "second": second}, "padding": "x" * 20_000}
    outcome = _compile_layer_root(
        repo,
        "selected-analysis",
        first,
        {"target": "run", "recipe": "quillon.span_summary"},
        payload_path=["members", "first"],
        whole_document=whole,
    )
    envelope_bytes = envelope_path(repo, "selected-analysis").read_bytes()

    assert len(envelope_bytes) < 2_048
    assert len((repo / "authorities/selected-analysis.json").read_bytes()) > 20_000
    assert outcome.document["params"] == {"window": 3}
    identity = outcome.compile_lock["identity_contributions"]["layer_root"]
    assert identity["payload_path"] == ["members", "first"]
    assert identity["selected_authority_sha256"] == canonical_sha256(first)

    envelope = _read(repo, "selected-analysis")
    envelope["analysis"]["root"]["payload_path"] = ["members", "second"]
    _write(repo, "selected-analysis", envelope)
    selected_second = kernel().compile_envelope_file(
        envelope_path(repo, "selected-analysis"), repo_root=repo
    )
    assert selected_second.document["params"] == {"window": 5}
    assert (
        selected_second.compile_lock["identity_contributions"]["layer_root"][
            "selected_authority_sha256"
        ]
        != identity["selected_authority_sha256"]
    )


@pytest.mark.parametrize(
    ("mutation", "category", "message"),
    [
        ("missing_ref", ExperimentEnvelopeRejectionCategory.MISSING_FIELD, "Field required"),
        ("missing_pin", ExperimentEnvelopeRejectionCategory.MISSING_FIELD, "Field required"),
        ("missing_file", ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE, "cannot load"),
        ("wrong_pin", ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE, "hash mismatch"),
        (
            "missing_selector",
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
            "missing object key",
        ),
        (
            "non_object",
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "must select a JSON object",
        ),
        (
            "wrong_version",
            ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
            "unsupported layer root authority",
        ),
        (
            "missing_schema_id",
            ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
            "selected payload",
        ),
        (
            "missing_schema_version",
            ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
            "selected payload",
        ),
        ("missing_kind", ExperimentEnvelopeRejectionCategory.MISSING_FIELD, "selected payload"),
        ("unknown_field", ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD, "selected payload"),
    ],
)
def test_layer_root_failures_use_the_closed_rejection_vocabulary(
    repo: Path,
    mutation: str,
    category: ExperimentEnvelopeRejectionCategory,
    message: str,
) -> None:
    authority = _layer_root_authority("analysis_run", params={})
    whole: dict[str, Any] = {"selected": authority, "scalar": 3}
    ref = "authorities/refusal.json"
    write_json(repo / ref, whole)
    root: dict[str, Any] = {
        "ref": ref,
        "sha256": canonical_sha256(whole),
        "payload_path": ["selected"],
    }
    if mutation == "missing_ref":
        root.pop("ref")
    elif mutation == "missing_pin":
        root.pop("sha256")
    elif mutation == "missing_file":
        root["ref"] = "authorities/missing.json"
    elif mutation == "wrong_pin":
        root["sha256"] = "0" * 64
    elif mutation == "missing_selector":
        root["payload_path"] = ["missing"]
    elif mutation == "non_object":
        root["payload_path"] = ["scalar"]
    else:
        selected = dict(authority)
        if mutation == "wrong_version":
            selected["schema_version"] = f"{EXPERIMENT_LAYER_ROOT_AUTHORITY_SCHEMA_ID}.v0"
        elif mutation == "missing_schema_id":
            selected.pop("schema_id")
        elif mutation == "missing_schema_version":
            selected.pop("schema_version")
        elif mutation == "missing_kind":
            selected.pop("kind")
        else:
            selected["payload"] = {}
        whole["selected"] = selected
        write_json(repo / ref, whole)
        root["sha256"] = canonical_sha256(whole)
    write_envelope(
        envelope_path(repo, "root-refusal"),
        {
            "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V4,
            "name": "root-refusal",
            "analysis": {
                "target": "run",
                "recipe": "quillon.span_summary",
                "root": root,
            },
        },
    )

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        kernel().compile_envelope_file(envelope_path(repo, "root-refusal"), repo_root=repo)
    assert caught.value.category is category
    assert message in str(caught.value)
    if mutation == "wrong_version":
        assert caught.value.field == "analysis.root.schema_version"


def test_layer_root_kind_mismatch_and_nested_identity_delta_fail_closed(repo: Path) -> None:
    with pytest.raises(ExperimentEnvelopeRejection) as mismatch:
        _compile_layer_root(
            repo,
            "wrong-kind",
            _layer_root_authority("figure", assembler="quillon.span_assembler"),
            {"target": "run", "recipe": "quillon.span_summary"},
        )
    assert mismatch.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert "requires authority kind 'analysis_run'" in str(mismatch.value)

    rooted = _compile_layer_root(
        repo,
        "nested-result",
        _layer_root_authority(
            "analysis_run",
            params={
                "result": {
                    "schema_id": "quillon.result",
                    "schema_version": "quillon.result.v1",
                    "fields": [],
                }
            },
        ),
        {"target": "run", "recipe": "quillon.span_summary"},
    )
    assert rooted.document["params"]["result"]["schema_version"] == "quillon.result.v1"

    with pytest.raises(ExperimentEnvelopeRejection, match="without a declared"):
        _compile_layer_root(
            repo,
            "nested-result-delta",
            _layer_root_authority("analysis_run", params={}),
            {
                "target": "run",
                "recipe": "quillon.span_summary",
                "delta": {
                    "layer_id": "typed-result",
                    "patches": [
                        {
                            "path": "params.result",
                            "op": "add",
                            "value": {
                                "schema_id": "quillon.result",
                                "schema_version": "quillon.result.v1",
                            },
                        }
                    ],
                },
            },
        )


def test_malformed_nested_bundle_authority_uses_output_model_and_closed_rejection(
    repo: Path,
) -> None:
    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _compile_layer_root(
            repo,
            "malformed-bundle",
            _layer_root_authority(
                "analysis_bundle",
                predicate={"manifest_kind": "EvaluationRunManifest"},
                stages=[{"name": "missing-required-kind"}],
            ),
            {"target": "bundle"},
        )

    assert caught.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert "analysis_bundle.stages.0.kind" in str(caught.value)
    assert "kind" in str(caught.value)


def test_rooted_figure_feeds_existing_row_expansion(repo: Path) -> None:
    from tests.fake_project_experiment import FIGURE_BASE, ROW_INDEX_BASE

    base = json.loads((repo / FIGURE_BASE).read_text())
    authority = _layer_root_authority(
        "figure",
        **{
            key: value
            for key, value in base.items()
            if key not in {"schema_id", "schema_version", "name", "inputs", "input_authorities"}
        },
    )
    _compile_layer_root(repo, "rooted-figure", authority, {"mode": "root"})
    expansion = _read(repo, "widened-plot")
    expansion["base"] = "rooted-figure"
    _write(repo, "widened-plot", expansion)

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

    assert [panel["name"] for panel in outcome.document["panels"]] == [
        "row_1__span",
        "row_2__span",
    ]
    pins = [item for item in outcome.compile_lock["references"] if item["kind"] == "content_pin"]
    assert [pin["ref"] for pin in pins] == [ROW_INDEX_BASE]


# -- figure mode ---------------------------------------------------------------


def test_row_expansion_derives_the_multi_row_figure_from_the_row_index(
    repo: Path,
) -> None:
    from tests.fake_project_experiment import ROW_INDEX_BASE

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

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
    pins = [item for item in outcome.compile_lock["references"] if item["kind"] == "content_pin"]
    assert [pin["ref"] for pin in pins] == [ROW_INDEX_BASE]


def test_row_expansion_names_no_produced_data_in_the_compiled_plan(repo: Path) -> None:
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

    assert not outcome.document.get("inputs")
    assert not outcome.document.get("input_authorities")
    kinds = {item["kind"] for item in outcome.compile_lock["references"]}
    assert "planned_product" not in kinds
    assert "not_applicable" in kinds
    check_plan_receipt_boundary(outcome.compile_lock)


def test_the_expansion_request_and_resolved_rows_carry_execution_identity(
    repo: Path,
) -> None:
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

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
        "identity_contributions.row_custody",
    ]


def test_a_row_expansion_without_a_custody_declaration_records_none(repo: Path) -> None:
    """An envelope authored before ``row_custody`` existed compiles unchanged.

    A compile states what the envelope said. An undeclared locator is therefore
    an absent contribution rather than an invented one, and the lock a ratified
    row-expansion envelope compiles to is byte-identical to the one it always
    compiled to: the same two contributions, the same identity inputs, and an
    execution identity that re-derives from exactly those facts.
    """
    envelope = _read(repo, "widened-plot")
    del envelope["figure"]["row_custody"]
    _write(repo, "widened-plot", envelope)

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

    contributions = outcome.compile_lock["identity_contributions"]
    assert set(contributions) == {"figure_row_expansion", "resolved_row_set"}
    assert outcome.compile_lock["execution_identity"]["inputs"] == [
        "compiled_document.content_hash",
        "identity_contributions.figure_row_expansion",
        "identity_contributions.resolved_row_set",
    ]
    assert outcome.compile_lock["execution_identity"]["sha256"] == canonical_sha256(
        {
            "compiled_document": outcome.compile_lock["compiled_document"]["content_hash"],
            "figure_row_expansion": canonical_sha256(contributions["figure_row_expansion"]),
            "resolved_row_set": canonical_sha256(contributions["resolved_row_set"]),
        }
    )


def test_the_custody_declaration_leaves_the_compiled_figure_untouched(
    repo: Path,
) -> None:
    """Custody is a locator for fulfillment, never part of the figure's identity."""
    declared = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)
    envelope = _read(repo, "widened-plot")
    del envelope["figure"]["row_custody"]
    _write(repo, "widened-plot", envelope)

    undeclared = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

    assert declared.document == undeclared.document
    assert (
        declared.compile_lock["compiled_document"]["content_hash"]
        == undeclared.compile_lock["compiled_document"]["content_hash"]
    )


def test_composition_compiles_to_a_composition_document_pinning_its_parent(
    repo: Path,
) -> None:
    from tests.fake_project_experiment import FIGURE_BASE

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-overlay"), repo_root=repo)

    assert outcome.document["schema_id"] == "feedbax.spec.figure_composition"
    assert outcome.document["parent"]["ref"] == FIGURE_BASE
    assert outcome.document["parent"]["sha256"] == canonical_sha256(
        json.loads((repo / FIGURE_BASE).read_text())
    )
    assert [delta["layer_id"] for delta in outcome.document["deltas"]] == ["widened-overlay"]


def test_the_same_base_under_two_modes_produces_two_different_families(
    repo: Path,
) -> None:
    """A filename never selects semantics; the authored mode does."""
    compiler = kernel()

    expanded = compiler.compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)
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
    envelope["figure"]["rows"] = {
        "mode": "tag",
        "tag": "absent",
        "index": ROW_INDEX_BASE,
    }
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

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

    request = outcome.compile_lock["identity_contributions"]["figure_row_expansion"]
    assert list(request["inputs"]) == ["unfilled"]


# -- the top-level report ------------------------------------------------------


def test_the_report_layer_compiles_to_a_top_level_report_spec(repo: Path) -> None:
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-report"), repo_root=repo)

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

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-report"), repo_root=repo)

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

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

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
    # Nothing is filled per row any more, so there is no row custody to name.
    envelope["figure"].pop("row_custody")
    _write(repo, "widened-plot", envelope)

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

    reference = next(
        item for item in outcome.compile_lock["references"] if item["kind"] == "receipt_locator"
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
    tag_paths = [patch["path"] for patch in delta["patches"] if patch["path"].startswith("tags.")]
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
    return kernel().compile_envelope_file(envelope_path(repo, "widened-report"), repo_root=repo)


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
        item for item in outcome.compile_lock["references"] if item["kind"] == "not_applicable"
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
        item for item in outcome.compile_lock["references"] if item["kind"] == "planned_product"
    )
    figure = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)
    assert entry["figure_spec_sha256"] == planned["compiled_content_hash"]
    assert entry["figure_spec_sha256"] == canonical_sha256(figure.document)
    assert entry["figure_spec_sha256"] != "a" * 64


def test_a_role_already_carrying_its_bound_digest_derives_no_patch(repo: Path) -> None:
    figure = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)
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
        item for item in outcome.compile_lock["references"] if item["kind"] == "not_applicable"
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


# -- an authored delta may not restate what a binding decided --------------------


def _report_delta(repo: Path, patches: list[dict[str, Any]]) -> None:
    """Give the report envelope one authored composition delta."""
    envelope = _read(repo, "widened-report")
    envelope["report"] = {
        **envelope["report"],
        "delta": {"layer_id": "widened-report", "patches": patches},
    }
    _write(repo, "widened-report", envelope)


def _reconciled_section(repo: Path) -> None:
    """Set up the compile whose binding derives state over section 0."""
    _report_base_figure(repo, figure_spec_sha256="a" * 64, applicability="included")
    _report_bindings(repo, [SECTION_NOT_APPLICABLE_BINDING])


@pytest.mark.parametrize(
    ("path", "op", "value"),
    [
        # the derived path itself
        ("params.sections.0.applicability", "replace", "included"),
        # a path under one the derivation removes
        ("params.sections.0.figures.0.caption", "replace", "Widened span"),
        # the node that contains every derived path
        ("params.sections.0", "replace", {"title": "Span", "figures": []}),
    ],
)
def test_an_authored_patch_over_binding_derived_state_is_refused(
    repo: Path, path: str, op: str, value: Any
) -> None:
    """The delta is applied after the derivation, so it would simply win."""
    _reconciled_section(repo)
    _report_delta(repo, [{"path": path, "op": op, "value": value}])

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        _compile_report(repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert path in str(excinfo.value)
    assert "params.sections.0" in str(excinfo.value)
    assert str(excinfo.value.field).startswith("report.delta.patches[0]")


def test_an_authored_patch_on_a_path_no_binding_decides_still_compiles(
    repo: Path,
) -> None:
    """Only the derived nodes are closed to the delta; the rest of the report is not."""
    _reconciled_section(repo)
    _report_delta(
        repo,
        [
            {"path": "params.title", "op": "replace", "value": "Quillon widened span"},
            {"path": "params.sections.1", "op": "add", "value": {"title": "Appendix"}},
        ],
    )

    outcome = _compile_report(repo)

    assert outcome.document["params"]["title"] == "Quillon widened span"
    assert outcome.document["params"]["sections"][1]["title"] == "Appendix"
    assert outcome.document["params"]["sections"][0]["applicability"] == "not_applicable"


def test_an_authored_delta_is_free_when_the_bindings_derive_nothing(repo: Path) -> None:
    """No derivation, no owned paths: the ordinary authored delta is untouched."""
    _report_bindings(
        repo,
        [
            {
                "role_path": "params.sections.0.figures.0",
                "ref": {"kind": "not_applicable", "reason": "no panel was produced"},
            }
        ],
    )
    _report_delta(
        repo,
        [{"path": "params.sections.0.title", "op": "replace", "value": "Span"}],
    )

    outcome = _compile_report(repo)

    assert outcome.document["params"]["sections"][0]["title"] == "Span"


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
    document["params"]["sections"][0]["tables"] = [{"columns": ["arm"], "rows": [["near"]]}]
    write_json(repo / REPORT_BASE, document)
    _report_bindings(repo, [SECTION_NOT_APPLICABLE_BINDING])

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        _compile_report(repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert "not-applicable section cannot declare figure or table content" in str(excinfo.value)


def test_a_params_node_the_contract_gives_no_applicability_stays_lock_only(
    repo: Path,
) -> None:
    """A scalar table is a node the model knows and states no applicability for."""
    from tests.fake_project_experiment import REPORT_BASE

    document = json.loads((repo / REPORT_BASE).read_text())
    document["params"]["sections"][0]["tables"] = [{"columns": ["arm"], "rows": [["near"]]}]
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
    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

    request = outcome.compile_lock["identity_contributions"]["figure_row_expansion"]
    assert request["inputs"]["observed"] == {"per_row": "observations"}
    assert request["role_contracts"][0]["input_role"] == "observed"


def test_the_lock_states_the_per_row_role_rather_than_a_false_locator(
    repo: Path,
) -> None:
    from feedbax.envelope.compile import PER_ROW_INPUT_RULE_ID

    outcome = kernel().compile_envelope_file(envelope_path(repo, "widened-plot"), repo_root=repo)

    references = outcome.compile_lock["references"]
    reference = next(item for item in references if item["kind"] == "not_applicable")
    assert reference["role_path"] == "inputs.observed"
    assert reference["basis"] == "compiler_rule"
    assert reference["rule_id"] == PER_ROW_INPUT_RULE_ID
    assert "per expanded row" in reference["reason"]
    assert not [item for item in references if item["kind"] == "planned_product"]


# -- what the v2 grammar makes authenticable -------------------------------
#
# Three constructs exist because the corpus needs facts the lock could not
# otherwise carry. Each case asserts the *lock*, because the lock is the only
# thing downstream treats as authority: a compiled document states plans, and a
# construct that produced no reference would have authored nothing at all.


def _authored(repo: Path, alias: str, document: dict[str, Any]) -> Any:
    """Write one envelope under *alias* and compile it."""
    _write(repo, alias, document)
    return kernel().compile_envelope_file(envelope_path(repo, alias), repo_root=repo)


def _receipt(manifest_id: str) -> dict[str, Any]:
    return {
        "kind": "receipt",
        "manifest_kind": "quillon.survey_run",
        "manifest_id": manifest_id,
        "manifest_sha256": "a" * 64,
        "size_bytes": 512,
    }


def test_an_evaluation_authors_further_staged_prerequisites_into_its_lock(
    repo: Path,
) -> None:
    """A second named prerequisite reaches the lock, which is what authenticates it.

    An evaluation matrix's rows consume named staged parents, and a compiled
    document cannot authenticate one. Before this the dialect could author
    exactly one reference, so a matrix whose base named a second prerequisite had
    no way to be fulfilled at all.
    """
    from tests.fake_project_experiment import EVALUATION_BASE

    outcome = _authored(
        repo,
        "paired-probe",
        {
            "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
            "name": "paired-probe",
            "base": EVALUATION_BASE,
            "evaluation": {
                "subject": {"kind": "envelope", "alias": "widened"},
                "subject_id": "trained",
                "prerequisites": {
                    "trial_bank": _receipt("bank-0"),
                    "reference_run": _receipt("reference-0"),
                },
            },
        },
    )

    references = [
        item for item in outcome.compile_lock["references"] if item["kind"] != "content_pin"
    ]
    assert [item["role_path"] for item in references] == [
        "subjects.trained",
        "subjects.trial_bank",
        "subjects.reference_run",
    ]
    assert [item["consumer"]["subject_id"] for item in references] == [
        "trained",
        "trial_bank",
        "reference_run",
    ]
    # The subject is a planned product of another envelope; the two prerequisites
    # quote receipts that already exist, so they are authenticated references and
    # not locators.
    assert [item["kind"] for item in references] == [
        "planned_product",
        "authenticated_receipt",
        "authenticated_receipt",
    ]
    assert references[1]["manifest_sha256"] == "a" * 64


def test_a_prerequisite_may_not_take_the_subjects_own_binding_name(repo: Path) -> None:
    from tests.fake_project_experiment import EVALUATION_BASE

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _authored(
            repo,
            "colliding-probe",
            {
                "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
                "name": "colliding-probe",
                "base": EVALUATION_BASE,
                "evaluation": {
                    "subject": {"kind": "envelope", "alias": "widened"},
                    "subject_id": "trained",
                    "prerequisites": {"trained": _receipt("bank-0")},
                },
            },
        )

    assert "one binding name addresses exactly one authenticated parent" in str(caught.value)


ANALYSIS_BUNDLE_BASE = "bases/baseline.analysis_bundle.json"

ANALYSIS_BUNDLE_BASE_DOCUMENT: dict[str, Any] = {
    "schema_id": "feedbax.spec.analysis_bundle",
    "schema_version": "feedbax.spec.analysis_bundle.v6",
    "name": "baseline-bundle",
    "predicate": {"manifest_kind": "EvaluationRunManifest"},
    "templates": [{"name": "span", "analysis_type": "quillon.span_summary", "params": {}}],
    "params_base": {"params": {}},
}


def _bundle_envelope(roots: list[dict[str, Any]] | None) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "target": "bundle",
        "delta": {
            "layer_id": "widened-bundle",
            "patches": [{"path": "params_base.params.trim", "op": "add", "value": 1}],
        },
    }
    if roots is not None:
        analysis["roots"] = roots
    return {
        "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
        "name": "widened-bundle",
        "base": ANALYSIS_BUNDLE_BASE,
        "analysis": analysis,
    }


def test_a_bundle_authors_its_exact_root_set_into_its_lock(repo: Path) -> None:
    """The only way a converted bundle can stop widening with its repository.

    A bundle's authored predicate re-selects whatever manifests exist when it
    runs. Authoring the roots puts each one in the lock, and fulfillment then
    binds by exact manifest identity instead of by ambient selection.
    """
    write_json(repo / ANALYSIS_BUNDLE_BASE, ANALYSIS_BUNDLE_BASE_DOCUMENT)

    outcome = _authored(
        repo,
        "widened-bundle",
        _bundle_envelope(
            [
                {"alias": "near", "ref": _receipt("near-run")},
                {"alias": "far", "ref": _receipt("far-run")},
            ]
        ),
    )

    references = [
        item for item in outcome.compile_lock["references"] if item["kind"] != "content_pin"
    ]
    assert [item["role_path"] for item in references] == ["roots.near", "roots.far"]
    assert [item["manifest_id"] for item in references] == ["near-run", "far-run"]
    assert [item["consumer"]["alias"] for item in references] == ["near", "far"]


def test_a_bundle_that_declares_no_roots_records_no_root_reference(repo: Path) -> None:
    """Absent roots is the honest record of a bundle that really selects ambiently."""
    write_json(repo / ANALYSIS_BUNDLE_BASE, ANALYSIS_BUNDLE_BASE_DOCUMENT)

    outcome = _authored(repo, "widened-bundle", _bundle_envelope(None))

    assert not [
        item for item in outcome.compile_lock["references"] if item["kind"] != "content_pin"
    ]


def test_a_bundle_root_set_with_no_members_is_refused(repo: Path) -> None:
    write_json(repo / ANALYSIS_BUNDLE_BASE, ANALYSIS_BUNDLE_BASE_DOCUMENT)

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _authored(repo, "widened-bundle", _bundle_envelope([]))

    assert "states at least one root" in str(caught.value)


def test_a_checkpoint_only_training_layer_compiles(repo: Path) -> None:
    """A fork that inherits its rows and only attaches checkpoint sources.

    The layer authors no row and no tag: what it changes is what the inherited
    rows are initialized from, which is a change to the runs themselves and is
    recorded where every other authenticated input is, in the lock.
    """
    from tests.fake_project_experiment import TRAINING_BASE

    outcome = _authored(
        repo,
        "forked",
        {
            "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
            "name": "forked",
            "base": TRAINING_BASE,
            "training": {
                "rows_mode": "append",
                "checkpoint_initialization": [
                    {
                        "row": "baseline",
                        "mode": "continue_from",
                        "source": _receipt("baseline-0"),
                    }
                ],
            },
        },
    )

    assert [row["row_id"] for row in outcome.document["rows"]] == ["baseline"]
    assert outcome.document["tags"] == ["baseline"]
    reference = next(
        item
        for item in outcome.compile_lock["references"]
        if item["kind"] == "authenticated_receipt"
    )
    assert reference["role_path"] == "rows.baseline.checkpoint_initialization"
    assert reference["consumer"] == {
        "consumer": "checkpoint_initialization",
        "mode": "continue_from",
        "row_id": "baseline",
    }


def test_a_training_layer_contributing_nothing_at_all_is_still_refused(
    repo: Path,
) -> None:
    """The relaxation is exactly one construct wide, not an empty-layer licence."""
    from tests.fake_project_experiment import TRAINING_BASE

    with pytest.raises(ExperimentEnvelopeRejection) as caught:
        _authored(
            repo,
            "empty",
            {
                "schema": EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
                "name": "empty",
                "base": TRAINING_BASE,
                "training": {"rows_mode": "append"},
            },
        )

    assert "rows, tags, checkpoint initialization" in str(caught.value)
