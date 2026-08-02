"""Binding a row-expanded figure's per-row roles from produced custody.

Before this surface existed, a row-expanded figure compiled fine and then
executed with its per-row roles silently unfilled: the compile recorded the
expansion and the rows, the lock stated the single-locator slot not-applicable,
and nothing ever read a custody bindings document. The claims under test are:

* the compile *names* the custody document — data, stated by the envelope, never
  derived by convention from the index id — and pins the row-index identity the
  located document must match;
* fulfillment *locates and believes* it: a row-expanded figure's node request
  carries one authenticated parent per expanded row per per-row role, with the
  matching input authority, as a runtime overlay outside figure identity;
* every way that can fail is a named refusal — undeclared, absent, foreign to the
  index, foreign to the index *cut*, incomplete for the rows, or unreadable — and
  none of them is ever a silent fallback to the pending state the compile
  recorded.

The row set, the roles, and the custody are ``quillon``'s invented vocabulary,
because none of this is about any particular science.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from feedbax.analysis.fulfillment_adapters import FulfillmentEnvironment
from feedbax.analysis.fulfillment_derivation import (
    derive_fulfillment_plan,
    read_compiled_outputs,
)
from feedbax.analysis.fulfillment_driver import closure_requests, preflight
from feedbax.analysis.fulfillment_row_custody import (
    RowCustodyFulfillmentError,
    read_row_expansion_record,
    resolve_row_custody_overlay,
)
from feedbax.analysis.fulfillment_plan import LogicalKey
from feedbax.contracts.applicability_rules import (
    PER_ROW_FIGURE_INPUT_RULE,
    certify_not_applicable,
)
from feedbax.contracts.figure_roles import FigureRowCustodyLocator
from feedbax.contracts.row_index import (
    AuthenticatedRowIndex,
    build_row_index_custody_bindings,
    expand_row_selector,
    write_row_index_custody_bindings,
)
from feedbax.envelope import kernel_for

from tests.fake_project_experiment import (
    PROJECT_DECLARATION,
    ROW_CUSTODY_REF,
    ROW_INDEX_BASE,
    envelope_path,
    write_repo,
)
from tests.fake_project_experiment.products import QuillonOutputs

#: One authenticated survey manifest per row, as a receipt layer would record.
ROW_MANIFESTS = {
    "near-span": ("near-span-0", "a" * 64, 2048),
    "far-span": ("far-span-0", "b" * 64, 4096),
}


# -- one compiled row-expanded figure, and the custody it names ------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    write_repo(tmp_path)
    return tmp_path


def _compile_plot(repo: Path) -> Path:
    """Compile the row-expansion figure and write its two emitted files."""
    outcome = kernel_for(PROJECT_DECLARATION).compile_envelope_file(
        envelope_path(repo, "widened-plot"), repo_root=repo
    )
    directory = repo / "compiled"
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / f"{outcome.name}.compile-lock.json"
    lock_path.write_text(
        json.dumps(outcome.compile_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / f"{outcome.name}.{outcome.compile_lock['compiled_document']['family']}.json").write_text(
        json.dumps(outcome.document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return lock_path


def _row_index(repo: Path) -> AuthenticatedRowIndex:
    return AuthenticatedRowIndex.model_validate(
        json.loads((repo / ROW_INDEX_BASE).read_text(encoding="utf-8"))
    )


def _parent(manifest_id: str, digest: str, size: int) -> dict[str, Any]:
    return {
        "kind": "quillon.survey_run",
        "id": manifest_id,
        "role": "observations",
        "metadata": {
            "ref_schema_id": "feedbax.ref.authenticated_manifest",
            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
            "manifest_sha256": digest,
            "size_bytes": size,
        },
    }


def _write_custody(
    repo: Path,
    *,
    rows: dict[str, tuple[str, str, int]] | None = None,
    index: AuthenticatedRowIndex | None = None,
    ref: str = ROW_CUSTODY_REF,
) -> Path:
    """Write the custody bindings a run receipt layer would have produced."""
    resolved = _row_index(repo) if index is None else index
    document = build_row_index_custody_bindings(
        resolved,
        {
            row_id: {"observations": _parent(*record)}
            for row_id, record in (ROW_MANIFESTS if rows is None else rows).items()
        },
    )
    return write_row_index_custody_bindings(document, repo / ref)


def _closure_request(repo: Path, environment: FulfillmentEnvironment):
    index = read_compiled_outputs(repo / "compiled")
    plan = derive_fulfillment_plan(index, target="widened-plot")
    closure = preflight(plan, index)
    key = LogicalKey("figure", "widened-plot")
    return closure_requests(closure, environment=environment, stop_at=key)[-1]


@pytest.fixture
def environment(tmp_path: Path, repo: Path, application_registry_bundle):
    return FulfillmentEnvironment(
        root=tmp_path / "receipts", registries=application_registry_bundle, repo_root=repo
    )


# -- the compile names the document, and pins the index identity -----------


def test_the_compile_records_where_the_per_row_custody_will_be(repo: Path) -> None:
    lock_path = _compile_plot(repo)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    locator = FigureRowCustodyLocator.model_validate(
        lock["identity_contributions"]["row_custody"]
    )
    assert locator.ref == ROW_CUSTODY_REF
    assert locator.index_id == "quillon-span-survey"
    assert locator.binding_keys == ["observations"]
    assert locator.index_sha256 == _row_index(repo).canonical_sha256()


def test_the_declaration_is_named_rather_than_derived_from_the_index_id(repo: Path) -> None:
    """A convention is a rule nothing states, so the locator quotes the envelope."""
    envelope = json.loads(
        envelope_path(repo, "widened-plot").read_text(encoding="utf-8")
    )
    envelope["figure"]["row_custody"] = "elsewhere/named-by-hand.json"
    envelope_path(repo, "widened-plot").write_text(
        json.dumps(envelope, indent=2) + "\n", encoding="utf-8"
    )

    lock = json.loads(_compile_plot(repo).read_text(encoding="utf-8"))

    assert lock["identity_contributions"]["row_custody"]["ref"] == (
        "elsewhere/named-by-hand.json"
    )


def test_a_first_time_figure_compiles_before_the_custody_document_exists(repo: Path) -> None:
    """The locator is a declaration, not a production record."""
    lock_path = _compile_plot(repo)

    assert not (repo / ROW_CUSTODY_REF).exists()
    assert "row_custody" in json.loads(lock_path.read_text(encoding="utf-8"))[
        "identity_contributions"
    ]


def test_a_custody_ref_that_escapes_the_repository_is_refused() -> None:
    with pytest.raises(ValueError, match="escapes the repository root"):
        FigureRowCustodyLocator(
            index_id="quillon-span-survey",
            index_sha256="c" * 64,
            ref="../outside/custody.json",
            binding_keys=["observations"],
        )


def test_an_absolute_custody_ref_is_refused() -> None:
    with pytest.raises(ValueError, match="is absolute"):
        FigureRowCustodyLocator(
            index_id="quillon-span-survey",
            index_sha256="c" * 64,
            ref="/custody/quillon.row_custody.json",
            binding_keys=["observations"],
        )


# -- fulfillment locates it, and binds every row ---------------------------


def test_the_figure_request_binds_one_parent_per_row_from_custody(
    repo: Path, environment: FulfillmentEnvironment
) -> None:
    _compile_plot(repo)
    _write_custody(repo)

    request = _closure_request(repo, environment)

    assert request.node_kind == "figure"
    assert [parent.role for parent in request.runtime_inputs] == [
        "row_1__observed",
        "row_2__observed",
    ]
    assert [parent.id for parent in request.runtime_inputs] == ["near-span-0", "far-span-0"]
    assert [authority.parent.role for authority in request.runtime_input_authorities] == [
        "row_1__observed",
        "row_2__observed",
    ]


def test_the_bound_custody_stays_out_of_the_compiled_figure(
    repo: Path, environment: FulfillmentEnvironment
) -> None:
    """Digests belong to the runtime overlay; the document is the figure's identity."""
    _compile_plot(repo)
    _write_custody(repo)

    request = _closure_request(repo, environment)

    assert "inputs" not in request.spec
    assert "input_authorities" not in request.spec
    assert all(parent.metadata in (None, {}) for parent in request.runtime_inputs)


def test_the_authority_carries_the_declared_artifact_contract(
    repo: Path, environment: FulfillmentEnvironment
) -> None:
    _compile_plot(repo)
    _write_custody(repo)

    request = _closure_request(repo, environment)

    payload = request.runtime_input_authorities[0].artifact_payloads[0]
    assert payload.manifest_role == "row_1__observed"
    assert payload.artifact_role == "span_observations"
    assert payload.artifact_provider == "quillon.custody"


def test_a_figure_that_is_not_row_expanded_binds_no_custody(tmp_path: Path) -> None:
    outputs = QuillonOutputs(tmp_path / "repo")
    outputs.plate("plain-plate")
    index = read_compiled_outputs(outputs.output_directory)

    compiled = index.require(index.envelopes[0].envelope_ref)

    assert read_row_expansion_record(compiled) is None
    assert resolve_row_custody_overlay(compiled, repo_root=tmp_path) is None


# -- every failure is a refusal, never a pending-role fallback -------------


def test_an_absent_custody_document_is_a_refusal(
    repo: Path, environment: FulfillmentEnvironment
) -> None:
    _compile_plot(repo)

    with pytest.raises(RowCustodyFulfillmentError) as caught:
        _closure_request(repo, environment)

    assert "not a file" in str(caught.value)
    assert "never that the per-row roles do not apply" in str(caught.value)


def test_an_undeclared_custody_document_is_a_refusal(repo: Path, tmp_path: Path) -> None:
    """A lock whose per-row roles name no custody cannot be fulfilled."""
    lock_path = _compile_plot(repo)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    del lock["identity_contributions"]["row_custody"]
    index = read_compiled_outputs(repo / "compiled")
    compiled = index.require(index.envelopes[0].envelope_ref)
    stripped = type(compiled)(
        lock=lock,
        document=compiled.document,
        lock_path=compiled.lock_path,
        document_path=compiled.document_path,
    )

    with pytest.raises(RowCustodyFulfillmentError) as caught:
        resolve_row_custody_overlay(stripped, repo_root=repo)

    assert "names no custody bindings document" in str(caught.value)
    assert "'row_custody'" in str(caught.value)


def test_a_declared_repo_root_is_required(repo: Path) -> None:
    _compile_plot(repo)
    _write_custody(repo)
    index = read_compiled_outputs(repo / "compiled")

    with pytest.raises(RowCustodyFulfillmentError, match="declares no repo_root"):
        resolve_row_custody_overlay(index.envelopes[0], repo_root=None)


def test_custody_for_another_row_index_is_refused(repo: Path) -> None:
    _compile_plot(repo)
    foreign = AuthenticatedRowIndex(
        index_id="quillon-other-survey",
        rows=[
            {"row_id": "near-span", "label": "near span"},
            {"row_id": "far-span", "label": "far span"},
        ],
    )
    _write_custody(repo, index=foreign)
    index = read_compiled_outputs(repo / "compiled")

    with pytest.raises(RowCustodyFulfillmentError) as caught:
        resolve_row_custody_overlay(index.envelopes[0], repo_root=repo)

    assert "quillon-other-survey" in str(caught.value)


def test_custody_for_another_cut_of_the_same_index_is_refused(repo: Path) -> None:
    """The locator pins the index digest, so a re-cut index cannot bind here."""
    lock_path = _compile_plot(repo)
    _write_custody(repo)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["identity_contributions"]["row_custody"]["index_sha256"] = "d" * 64
    index = read_compiled_outputs(repo / "compiled")
    compiled = index.envelopes[0]
    drifted = type(compiled)(
        lock=lock,
        document=compiled.document,
        lock_path=compiled.lock_path,
        document_path=compiled.document_path,
    )

    with pytest.raises(RowCustodyFulfillmentError) as caught:
        resolve_row_custody_overlay(drifted, repo_root=repo)

    assert "two cuts of one index" in str(caught.value)


def test_custody_missing_a_row_is_refused_rather_than_partially_bound(repo: Path) -> None:
    _compile_plot(repo)
    _write_custody(repo, rows={"near-span": ROW_MANIFESTS["near-span"]})
    index = read_compiled_outputs(repo / "compiled")

    with pytest.raises(RowCustodyFulfillmentError) as caught:
        resolve_row_custody_overlay(index.envelopes[0], repo_root=repo)

    assert "does not satisfy this figure's declared roles" in str(caught.value)
    assert "far-span" in str(caught.value)


def test_an_unreadable_custody_document_is_refused(repo: Path) -> None:
    _compile_plot(repo)
    target = repo / ROW_CUSTODY_REF
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{not json", encoding="utf-8")
    index = read_compiled_outputs(repo / "compiled")

    with pytest.raises(RowCustodyFulfillmentError) as caught:
        resolve_row_custody_overlay(index.envelopes[0], repo_root=repo)

    assert "does not load as a RowIndexCustodyBindings document" in str(caught.value)


def test_a_foreign_schema_version_in_the_custody_document_is_refused(repo: Path) -> None:
    _compile_plot(repo)
    path = _write_custody(repo)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = "feedbax.spec.row_index_custody_bindings.v2"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    index = read_compiled_outputs(repo / "compiled")

    with pytest.raises(RowCustodyFulfillmentError) as caught:
        resolve_row_custody_overlay(index.envelopes[0], repo_root=repo)

    assert "unsupported RowIndexCustodyBindings schema_version" in str(caught.value)


def test_a_half_recorded_expansion_is_a_broken_lock(repo: Path) -> None:
    lock_path = _compile_plot(repo)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    del lock["identity_contributions"]["resolved_row_set"]
    index = read_compiled_outputs(repo / "compiled")
    compiled = index.envelopes[0]
    broken = type(compiled)(
        lock=lock,
        document=compiled.document,
        lock_path=compiled.lock_path,
        document_path=compiled.document_path,
    )

    with pytest.raises(RowCustodyFulfillmentError, match="one without the other"):
        read_row_expansion_record(broken)


def test_the_expanded_rows_and_the_custody_agree_on_order(repo: Path) -> None:
    """Row-index order alone decides the namespaces, so binding follows it."""
    _compile_plot(repo)
    _write_custody(repo)
    index = read_compiled_outputs(repo / "compiled")

    overlay = resolve_row_custody_overlay(index.envelopes[0], repo_root=repo)
    record = read_row_expansion_record(index.envelopes[0])

    assert record is not None and overlay is not None
    rows = list(record.resolved_rows.row_ids)
    assert rows == ["near-span", "far-span"]
    assert [parent.id for parent in overlay.inputs] == [
        ROW_MANIFESTS[row][0] for row in rows
    ]


def test_the_selector_the_compile_expanded_is_what_custody_binds(repo: Path) -> None:
    """Expansion happens once, at compile; fulfillment never re-expands a selector."""
    _compile_plot(repo)
    _write_custody(repo)
    index = read_compiled_outputs(repo / "compiled")
    record = read_row_expansion_record(index.envelopes[0])

    assert record is not None
    expanded = expand_row_selector(record.request.rows, _row_index(repo))
    assert list(expanded.row_ids) == list(record.resolved_rows.row_ids)


def test_a_per_row_omission_still_reaches_the_lock_through_the_closed_rule(
    repo: Path,
) -> None:
    """The two halves are one story: nothing binds the slot, custody binds the role."""
    lock = json.loads(_compile_plot(repo).read_text(encoding="utf-8"))

    reference = next(
        item for item in lock["references"] if item["kind"] == "not_applicable"
    )
    assert reference == certify_not_applicable(
        reference["role_path"], PER_ROW_FIGURE_INPUT_RULE
    ).model_dump(mode="json", exclude_none=True)
