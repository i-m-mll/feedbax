"""Figure role references, authenticated row index, and row-set selectors.

The mechanical gates below are the preregistered figure benchmark recorded on
umbrella `cfdadc7` before implementation started (benchmark 2, "k2 figure as a
row-set change"). They are grounded in the real corpus pair:

* base figure — `specs/post_run/sisu_m2_velocity_profiles.figure.v1.json`,
  copied verbatim into `tests/fixtures/figures/`;
* authored envelope — the preregistered 16-line variant, copied verbatim into
  `tests/fixtures/figure_roles/`;
* row index and custody — projected from
  `results/4e08c8d/report-production.v1.json#reports.target2x_target4x_floor`
  and the preregistered manifest identity/digest table. The one `k1` custody
  binding is synthetic (digest `cc..cc`, size 1); it exists only to prove that
  binding-key resolution picks the right artifact when several are present.

Mapping the authored envelope onto a `FigureRowExpansionRequest` is the
downstream compiler's job. The small mapping performed here stands in for it
and is itself part of what the benchmark checks: the authored envelope must be
sufficient, carrying zero realization fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.contracts.figure_roles import (
    FigureRoleBindingContract,
    FigureRoleReferenceError,
    FigureRowExpansionRequest,
    PerRowInputReference,
    SharedInputReference,
    expand_figure_rows,
    resolve_figure_input_roles,
    row_namespace,
)
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.row_index import (
    AuthenticatedRowIndex,
    RowIndexCustodyBindings,
    RowSelectionError,
    RowSelectionErrorCode,
    expand_row_selector,
    normalize_row_tags,
)

FIXTURES = Path(__file__).parent / "fixtures"
BASE_FIGURE_PATH = FIXTURES / "figures" / "sisu_m2_velocity_profiles.figure.v1.json"
ENVELOPE_PATH = FIXTURES / "figure_roles" / "benchmark2_figure.envelope.json"
ROW_INDEX_PATH = FIXTURES / "figure_roles" / "target2x_target4x_floor.row_index.json"
ROW_CUSTODY_PATH = FIXTURES / "figure_roles" / "target2x_target4x_floor.row_custody.json"

#: Gate 2A — the literal `row_order` of the grounding production record.
EXPECTED_ROW_IDS = ["target2x-floor", "target4x-floor"]

#: Gate 2B — resolved input identities, in resolved order.
EXPECTED_INPUT_IDENTITIES = [
    ("row_1__trained", "feedbax-analysis-run:3d916232abe4ae8eccf16b676c890e7d"),
    ("row_1__comparators", "feedbax-analysis-run:d6464fe0ac4c2e2a14e36c476b15297e"),
    ("row_1__hinf_1p05_reference", "feedbax-analysis-run:6d1befa7d02075a2b58fe8787fda2296"),
    ("row_2__trained", "feedbax-analysis-run:11b2be35e7a3118ef25d6822c992e79a"),
    ("row_2__comparators", "feedbax-analysis-run:d6464fe0ac4c2e2a14e36c476b15297e"),
    ("row_2__hinf_1p05_reference", "feedbax-analysis-run:6d1befa7d02075a2b58fe8787fda2296"),
]

#: Gate 2B — compile-lock custody for the per-row roles.
EXPECTED_PER_ROW_CUSTODY = {
    "row_1__trained": (
        "710b5922eabb11180d83f3abf35b0dc1022d7d15585ab825bfd188cc37e90f0f",
        10658924,
    ),
    "row_2__trained": (
        "7170a4d1db43586d826b2fc94c90a2eff838f2faa20738616bf7e8c8e444cd94",
        10658924,
    ),
}

#: Gate 2D — compiler-derived restatements that must never be emitted.
DERIVED_RESTATEMENTS = (
    "composition",
    "row_count",
    "rows",
    "legend_policy",
    "colorbar_policy",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def base_payload() -> dict[str, Any]:
    return _load(BASE_FIGURE_PATH)


@pytest.fixture(scope="module")
def envelope() -> dict[str, Any]:
    return _load(ENVELOPE_PATH)


@pytest.fixture(scope="module")
def row_index() -> AuthenticatedRowIndex:
    return AuthenticatedRowIndex.model_validate(_load(ROW_INDEX_PATH))


@pytest.fixture(scope="module")
def custody() -> RowIndexCustodyBindings:
    return RowIndexCustodyBindings.model_validate(_load(ROW_CUSTODY_PATH))


def _role_contracts(base: dict[str, Any]) -> list[FigureRoleBindingContract]:
    """Project the base figure's declared input binding contracts.

    ``trained`` is the one role delivered as a typed analysis data product;
    the two reference roles are ordinary artifact-provider inputs. Both facts
    are declared by the base, not decided per row.
    """
    contracts: list[FigureRoleBindingContract] = []
    for declared in base["metadata"]["input_binding_contracts"]:
        payload = {
            "input_role": declared["input_role"],
            "kind": declared["kind"],
            "artifact_role": declared["artifact_role"],
            "artifact_provider": declared["artifact_provider"],
            "manifest_status": declared["manifest_status"],
            "analysis_type": declared.get("analysis_type"),
        }
        if declared["input_role"] == "trained":
            payload.update(
                {
                    "authority": "analysis_data_product",
                    "product_role": "velocity_profile_result",
                    "product_schema_id": "rlrmp2.analysis.velocity_profile_result",
                    "product_schema_version": "rlrmp2.analysis.velocity_profile_result.v1",
                }
            )
        contracts.append(FigureRoleBindingContract.model_validate(payload))
    return contracts


def _request(envelope: dict[str, Any], base: dict[str, Any]) -> FigureRowExpansionRequest:
    figure = envelope["figure"]
    return FigureRowExpansionRequest(
        figure_name=envelope["name"],
        rows=figure["rows"],
        inputs=figure["inputs"],
        role_contracts=_role_contracts(base),
        assembler_title=figure["title"],
    )


@pytest.fixture(scope="module")
def request_model(
    envelope: dict[str, Any], base_payload: dict[str, Any]
) -> FigureRowExpansionRequest:
    return _request(envelope, base_payload)


@pytest.fixture(scope="module")
def expanded(
    base_payload: dict[str, Any],
    request_model: FigureRowExpansionRequest,
    row_index: AuthenticatedRowIndex,
    custody: RowIndexCustodyBindings,
) -> FigureSpec:
    resolved_rows = expand_row_selector(request_model.rows, row_index)
    resolved_inputs = resolve_figure_input_roles(request_model, resolved_rows, custody)
    return expand_figure_rows(base_payload, request_model, resolved_rows, resolved_inputs)


def _substituted(value: Any, source: str, target: str) -> Any:
    if isinstance(value, str):
        return value.replace(source, target)
    if isinstance(value, dict):
        return {key: _substituted(item, source, target) for key, item in value.items()}
    if isinstance(value, list):
        return [_substituted(item, source, target) for item in value]
    return value


def _dump(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=True)


def _typed_cause(error: ValidationError) -> Any:
    """Return the typed exception a model validator raised.

    Pydantic wraps validator exceptions, but keeps the original in the error
    context, so callers still get the actionable code and field.
    """
    return error.errors()[0]["ctx"]["error"]


# --- Gate 2A: selector expansion ------------------------------------------------


def test_gate_2a_selector_expands_to_the_ordered_floor_rows(
    request_model: FigureRowExpansionRequest, row_index: AuthenticatedRowIndex
) -> None:
    resolved = expand_row_selector(request_model.rows, row_index)

    assert resolved.row_ids == EXPECTED_ROW_IDS
    assert resolved.index_id == "target2x_target4x_floor"
    assert resolved.index_sha256 == row_index.canonical_sha256()
    assert resolved.index_ref == (
        "results/4e08c8d/report-production.v1.json#reports.target2x_target4x_floor"
    )


def test_selector_union_is_closed(row_index: AuthenticatedRowIndex) -> None:
    for rejected in (
        {"mode": "any"},
        {"mode": "all", "tag": "sisu"},
        {"mode": "tag"},
        {"mode": "tag", "tag": "sisu", "not": "target4x"},
        {"mode": "all", "order": "reverse"},
    ):
        with pytest.raises(ValueError):
            expand_row_selector(rejected, row_index)


def test_tag_mode_selects_the_declared_subset(row_index: AuthenticatedRowIndex) -> None:
    resolved = expand_row_selector({"mode": "tag", "tag": "target4x"}, row_index)

    assert resolved.row_ids == ["target4x-floor"]
    assert expand_row_selector({"mode": "tag", "tag": "floor-saturated"}, row_index).row_ids == (
        EXPECTED_ROW_IDS
    )


def test_empty_selection_fails_closed(row_index: AuthenticatedRowIndex) -> None:
    with pytest.raises(RowSelectionError) as excinfo:
        expand_row_selector({"mode": "tag", "tag": "no-such-tag"}, row_index)

    assert excinfo.value.code is RowSelectionErrorCode.EMPTY_SELECTION


def test_duplicate_row_ids_fail_closed(row_index: AuthenticatedRowIndex) -> None:
    payload = _dump(row_index)
    payload["rows"].append(dict(payload["rows"][0]))

    with pytest.raises(ValidationError) as excinfo:
        AuthenticatedRowIndex.model_validate(payload)

    cause = _typed_cause(excinfo.value)
    assert isinstance(cause, RowSelectionError)
    assert cause.code is RowSelectionErrorCode.DUPLICATE_ROW_ID


def test_ambiguous_custody_binding_fails_closed(custody: RowIndexCustodyBindings) -> None:
    payload = _dump(custody)
    payload["bindings"].append(dict(payload["bindings"][0]))

    with pytest.raises(ValidationError) as excinfo:
        RowIndexCustodyBindings.model_validate(payload)

    cause = _typed_cause(excinfo.value)
    assert isinstance(cause, RowSelectionError)
    assert cause.code is RowSelectionErrorCode.AMBIGUOUS_ROW_BINDING


def test_row_tags_are_bounded_and_normalized() -> None:
    assert normalize_row_tags(["sisu", "floor-saturated"]) == ("floor-saturated", "sisu")
    for rejected in (["SISU"], ["a b"], ["x" * 41], ["sisu", "sisu"], ["t"] * 9):
        with pytest.raises(ValueError):
            normalize_row_tags(rejected)


# --- Gate 2B: resolved figure inputs --------------------------------------------


def test_gate_2b_inputs_resolve_to_the_named_manifest_identities(
    request_model: FigureRowExpansionRequest,
    row_index: AuthenticatedRowIndex,
    custody: RowIndexCustodyBindings,
) -> None:
    resolved_rows = expand_row_selector(request_model.rows, row_index)
    resolved = resolve_figure_input_roles(request_model, resolved_rows, custody)

    assert len(resolved.inputs) == 6
    assert [
        (item.role, item.parent.id) for item in resolved.inputs
    ] == EXPECTED_INPUT_IDENTITIES
    assert resolved.fully_bound


def test_gate_2b_compile_lock_records_custody_the_figure_never_carries(
    request_model: FigureRowExpansionRequest,
    row_index: AuthenticatedRowIndex,
    custody: RowIndexCustodyBindings,
    expanded: FigureSpec,
) -> None:
    resolved_rows = expand_row_selector(request_model.rows, row_index)
    resolved = resolve_figure_input_roles(request_model, resolved_rows, custody)

    observed = {
        item.role: (item.manifest_sha256, item.size_bytes)
        for item in resolved.inputs
        if item.binding == "per_row"
    }
    assert observed == EXPECTED_PER_ROW_CUSTODY

    payload = _dump(expanded)
    assert all(not parent.get("metadata") for parent in payload["inputs"])
    assert "manifest_sha256" not in json.dumps(payload)
    assert "size_bytes" not in json.dumps(payload)


def test_gate_2b_role_names_follow_the_row_namespace(
    request_model: FigureRowExpansionRequest,
    row_index: AuthenticatedRowIndex,
    custody: RowIndexCustodyBindings,
) -> None:
    resolved_rows = expand_row_selector(request_model.rows, row_index)
    resolved = resolve_figure_input_roles(request_model, resolved_rows, custody)

    for item in resolved.inputs:
        assert item.role == f"{row_namespace(item.row_ordinal)}{item.input_role}"
        assert item.input_role in {"trained", "comparators", "hinf_1p05_reference"}


def test_undeclared_input_role_is_rejected(base_payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError) as excinfo:
        FigureRowExpansionRequest(
            figure_name="x",
            rows={"mode": "all"},
            inputs={"unknown_role": PerRowInputReference(per_row="k2")},
            role_contracts=_role_contracts(base_payload),
        )

    cause = _typed_cause(excinfo.value)
    assert isinstance(cause, FigureRoleReferenceError)
    assert cause.input_role == "unknown_role"


def test_authored_authority_block_is_illegal(base_payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError) as excinfo:
        FigureRowExpansionRequest(
            figure_name="x",
            rows={"mode": "all"},
            inputs={
                "trained": {
                    "shared": "feedbax-analysis-run:3d916232abe4ae8eccf16b676c890e7d",
                    "manifest_sha256": (
                        "710b5922eabb11180d83f3abf35b0dc1022d7d15585ab825bfd188cc37e90f0f"
                    ),
                    "size_bytes": 10658924,
                }
            },
            role_contracts=_role_contracts(base_payload),
        )

    cause = _typed_cause(excinfo.value)
    assert isinstance(cause, FigureRoleReferenceError)
    assert cause.field == "manifest_sha256"
    assert "compile lock" in str(cause)


def test_role_reference_admits_exactly_one_of_per_row_or_shared(
    base_payload: dict[str, Any],
) -> None:
    for rejected in ({}, {"per_row": "k2", "shared": "x"}, {"role": "k2"}):
        with pytest.raises(ValidationError) as excinfo:
            FigureRowExpansionRequest(
                figure_name="x",
                rows={"mode": "all"},
                inputs={"trained": rejected},
                role_contracts=_role_contracts(base_payload),
            )
        assert isinstance(_typed_cause(excinfo.value), FigureRoleReferenceError)


def test_authored_envelope_carries_no_realization_fields(envelope: dict[str, Any]) -> None:
    text = json.dumps(envelope)

    for forbidden in (
        "manifest_sha256",
        "size_bytes",
        "input_authorities",
        "authenticated_manifest",
        "patches",
        "assembler_params",
    ):
        assert forbidden not in text
    assert len(ENVELOPE_PATH.read_text(encoding="utf-8").splitlines()) == 16


# --- Gate 2C: derived structure --------------------------------------------------


def test_gate_2c_expansion_reproduces_the_k2_figure_shape(
    base_payload: dict[str, Any], expanded: FigureSpec
) -> None:
    assert expanded.name == "sisu-floor-velocity-profiles"
    assert len(expanded.trace_families or []) == 16
    assert len(expanded.panels) == 4
    assert expanded.assembler_params["height"] == (
        base_payload["assembler_params"]["height"] * len(EXPECTED_ROW_IDS)
    )
    assert expanded.assembler_params["height"] == 1080
    assert expanded.assembler_params["title"] == "Floor rows: velocity profiles"
    assert expanded.colorbar is not None
    assert expanded.colorbar.family == "row_1__trained-seen"
    assert expanded.colorbar.placement is not None
    assert expanded.colorbar.placement.panel == "row_1__held_out"


def test_gate_2c_later_families_are_prefix_substitutions_plus_legend_leaves(
    expanded: FigureSpec,
) -> None:
    families = [_dump(family) for family in expanded.trace_families or []]
    exact = 0
    legend_only = 0
    for index in range(8):
        first, second = families[index], families[index + 8]
        substituted = _substituted(first, "row_1__", "row_2__")
        if substituted == second:
            exact += 1
            continue
        legend_only += 1
        assert "legend_index" in substituted and "legend_index" not in second
        assert second["trace"]["params"]["showlegend"] is False
        stripped = {key: value for key, value in substituted.items() if key != "legend_index"}
        stripped["trace"] = dict(stripped["trace"])
        stripped["trace"]["params"] = {
            key: value
            for key, value in second["trace"]["params"].items()
            if key != "showlegend"
        }
        candidate = {key: value for key, value in second.items() if key != "legend_index"}
        candidate["trace"] = dict(candidate["trace"])
        candidate["trace"]["params"] = stripped["trace"]["params"]
        assert stripped == candidate

    # The grounding k2 file: families 8, 9, 11, 13, 15 are exact prefix
    # substitutions; 10, 12, 14 differ in exactly the two legend leaves.
    assert (exact, legend_only) == (5, 3)


def test_gate_2c_later_panels_are_prefix_substitutions(expanded: FigureSpec) -> None:
    panels = [_dump(panel) for panel in expanded.panels]
    for index in range(2):
        first, second = panels[index], panels[index + 2]
        substituted = _substituted(first, "row_1__", "row_2__")
        substituted["row"] = first["row"] + 1
        substituted["title"] = second["title"]
        assert substituted == second
    assert [panel["row"] for panel in panels] == [1, 1, 2, 2]
    assert [panel["col"] for panel in panels] == [1, 2, 1, 2]


def test_gate_2c_panel_titles_are_derived_from_row_labels(
    base_payload: dict[str, Any], expanded: FigureSpec
) -> None:
    base_titles = [panel["title"] for panel in base_payload["panels"]]

    assert [panel.title for panel in expanded.panels] == [
        f"target2x floor — {base_titles[0]}",
        f"target2x floor — {base_titles[1]}",
        f"target4x floor — {base_titles[0]}",
        f"target4x floor — {base_titles[1]}",
    ]
    assert [panel.name for panel in expanded.panels] == [
        "row_1__seen",
        "row_1__held_out",
        "row_2__seen",
        "row_2__held_out",
    ]


def test_gate_2c_input_authorities_are_fully_derived_from_the_role(
    expanded: FigureSpec,
) -> None:
    payload = _dump(expanded)
    authorities = payload["input_authorities"]

    assert len(authorities) == 6
    for index, authority in enumerate(authorities):
        assert authority["parent"] == payload["inputs"][index]
        assert len(authority["artifact_payloads"]) == 1
        artifact = authority["artifact_payloads"][0]
        role = payload["inputs"][index]["role"]
        input_role = role.split("__", 1)[1]
        assert artifact["name"] == input_role
        assert artifact["manifest_role"] == role
        assert artifact["artifact_role"] == "velocity_profile_result"
        assert artifact["artifact_provider"] == "results"
        assert artifact["media_type"] == "application/json"
        assert artifact["manifest_status"] == "completed"
        if input_role == "trained":
            assert artifact["authority"] == "analysis_data_product"
            assert artifact["product_role"] == "velocity_profile_result"
            assert artifact["product_schema_id"] == "rlrmp2.analysis.velocity_profile_result"
            assert (
                artifact["product_schema_version"]
                == "rlrmp2.analysis.velocity_profile_result.v1"
            )
        else:
            assert artifact["authority"] == "artifact_provider"
            assert "product_role" not in artifact


def test_expansion_produces_ordinary_current_figure_semantics(expanded: FigureSpec) -> None:
    assert expanded.schema_version == "feedbax.spec.figure.v2"
    assert FigureSpec.model_validate(_dump(expanded)) == expanded


# --- Gate 2D: exclusions ---------------------------------------------------------


def test_gate_2d_derived_restatements_are_absent_and_contracts_preserved(
    base_payload: dict[str, Any], expanded: FigureSpec
) -> None:
    for key in DERIVED_RESTATEMENTS:
        assert key not in expanded.metadata
    assert expanded.metadata["input_binding_contracts"] == (
        base_payload["metadata"]["input_binding_contracts"]
    )


# --- R6: the compile/run custody boundary ----------------------------------------


def test_first_time_figure_resolves_roles_without_any_production_record(
    request_model: FigureRowExpansionRequest, row_index: AuthenticatedRowIndex
) -> None:
    resolved_rows = expand_row_selector(request_model.rows, row_index)
    resolved = resolve_figure_input_roles(request_model, resolved_rows)

    assert resolved.row_ids == EXPECTED_ROW_IDS
    assert not resolved.fully_bound
    assert resolved.pending_roles == ("row_1__trained", "row_2__trained")
    shared = [item for item in resolved.inputs if item.binding == "shared"]
    assert len(shared) == 4
    assert all(item.parent is not None for item in shared)


def test_expansion_refuses_to_name_data_that_has_not_been_produced(
    base_payload: dict[str, Any],
    request_model: FigureRowExpansionRequest,
    row_index: AuthenticatedRowIndex,
) -> None:
    resolved_rows = expand_row_selector(request_model.rows, row_index)
    resolved = resolve_figure_input_roles(request_model, resolved_rows)

    with pytest.raises(RowSelectionError) as excinfo:
        expand_figure_rows(base_payload, request_model, resolved_rows, resolved)

    assert excinfo.value.code is RowSelectionErrorCode.UNRESOLVED_ROW_KEY


def test_unbound_per_row_key_fails_closed(
    request_model: FigureRowExpansionRequest,
    row_index: AuthenticatedRowIndex,
    custody: RowIndexCustodyBindings,
) -> None:
    resolved_rows = expand_row_selector(request_model.rows, row_index)
    missing = RowIndexCustodyBindings.model_validate(
        {
            **_dump(custody),
            "bindings": [
                item for item in _dump(custody)["bindings"] if item["binding_key"] != "k2"
            ],
        }
    )

    with pytest.raises(RowSelectionError) as excinfo:
        resolve_figure_input_roles(request_model, resolved_rows, missing)

    assert excinfo.value.code is RowSelectionErrorCode.UNRESOLVED_ROW_KEY
    assert excinfo.value.binding_key == "k2"


def test_custody_from_another_index_is_refused(
    request_model: FigureRowExpansionRequest,
    row_index: AuthenticatedRowIndex,
    custody: RowIndexCustodyBindings,
) -> None:
    resolved_rows = expand_row_selector(request_model.rows, row_index)
    foreign = RowIndexCustodyBindings.model_validate(
        {**_dump(custody), "index_id": "some_other_report"}
    )

    with pytest.raises(RowSelectionError) as excinfo:
        resolve_figure_input_roles(request_model, resolved_rows, foreign)

    assert excinfo.value.code is RowSelectionErrorCode.INDEX_MISMATCH


def test_custody_bindings_require_authenticated_manifests(
    custody: RowIndexCustodyBindings,
) -> None:
    payload = _dump(custody)
    payload["bindings"][0]["parent"].pop("metadata")

    with pytest.raises(ValueError, match="authenticated manifest"):
        RowIndexCustodyBindings.model_validate(payload)


# --- Versioned schema identity ----------------------------------------------------


@pytest.mark.parametrize(
    ("model", "path"),
    [
        (AuthenticatedRowIndex, ROW_INDEX_PATH),
        (RowIndexCustodyBindings, ROW_CUSTODY_PATH),
    ],
)
def test_unknown_schema_versions_reject_rather_than_being_inferred(
    model: type, path: Path
) -> None:
    payload = _load(path)
    # ``.v9`` stands in for "a version this build has never heard of"; the
    # concrete neighbouring versions differ per family and are covered by each
    # family's own version-boundary test.
    for bad in (f"{payload['schema_id']}.v0", f"{payload['schema_id']}.v9", "other.v1"):
        with pytest.raises(ValueError, match="unsupported"):
            model.model_validate({**payload, "schema_version": bad})
    with pytest.raises(ValueError, match="unsupported"):
        model.model_validate({**payload, "schema_id": "feedbax.spec.something_else"})


def test_registered_row_and_figure_role_families_reject_old_versions() -> None:
    from feedbax.contracts.migrations import default_spec_registry

    for kind in (
        "AuthenticatedRowIndex",
        "ResolvedRowSet",
        "FigureRowExpansionRequest",
        "ResolvedFigureInputs",
    ):
        family = default_spec_registry.resolve(kind)
        assert family.policy is not None
        assert family.policy.stance == "reject"
        assert family.policy.rejected_old_versions == (f"{family.identity}.v0",)
        with pytest.raises(ValueError):
            default_spec_registry.migrate(kind, {"schema_version": f"{family.identity}.v0"})

    # Row custody has a real predecessor, and it is rejected rather than
    # migrated in place: a v1 document states no index digest, so any digest a
    # reader supplied would be one nothing observed.
    custody = default_spec_registry.resolve("RowIndexCustodyBindings")
    assert custody.policy is not None
    assert custody.policy.stance == "reject"
    assert custody.policy.rejected_old_versions == (
        f"{custody.identity}.v0",
        f"{custody.identity}.v1",
    )
    for absent in (f"{custody.identity}.v0", f"{custody.identity}.v1"):
        with pytest.raises(ValueError):
            default_spec_registry.migrate(
                "RowIndexCustodyBindings", {"schema_version": absent}
            )


def test_shared_reference_is_reused_across_rows_without_row_local_authority(
    request_model: FigureRowExpansionRequest,
    row_index: AuthenticatedRowIndex,
    custody: RowIndexCustodyBindings,
) -> None:
    resolved_rows = expand_row_selector(request_model.rows, row_index)
    resolved = resolve_figure_input_roles(request_model, resolved_rows, custody)

    shared: dict[str, set[str]] = {}
    for item in resolved.inputs:
        if item.binding == "shared":
            shared.setdefault(item.input_role, set()).add(item.parent.id)
    assert shared == {
        "comparators": {"feedbax-analysis-run:d6464fe0ac4c2e2a14e36c476b15297e"},
        "hinf_1p05_reference": {"feedbax-analysis-run:6d1befa7d02075a2b58fe8787fda2296"},
    }
    assert isinstance(request_model.inputs["comparators"], SharedInputReference)
    assert isinstance(request_model.inputs["trained"], PerRowInputReference)
