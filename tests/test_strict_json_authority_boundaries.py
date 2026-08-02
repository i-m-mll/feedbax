"""Duplicated JSON object keys refuse at every authority boundary.

Python's :func:`json.loads` keeps the *last* value for a repeated object member
name and says nothing. Everywhere Feedbax reads an authority document — a
manifest, a compile lock, a custody sidecar, a row index, a spec, a report or
figure payload, a checkpoint pointer — a repeated member name means the document
states two authorities for one fact, and last-value-wins silently picks one.

These tests state the boundary in three parts:

* the strict loader parses a well-formed document to *exactly* what
  :func:`json.loads` parses it to, so nothing about byte identity changes;
* a repeated member name refuses, at the top level and nested at any depth
  inside objects and arrays, naming the JSON path it found; and
* the refusal really is reached through the real loaders, not only through the
  helper in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from feedbax.analysis.evaluation_inputs import (
    EvaluationInputResolutionError,
    resolve_evaluation_inputs,
)
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ParentRef,
    EvaluationRunSpec,
    Provenance,
    TrainingRunManifest,
    load_manifest_bytes,
    spec_payload,
)
from feedbax.contracts.row_index import (
    AuthenticatedRowIndex,
    build_row_index_custody_bindings,
    load_row_index_custody_bindings,
    write_row_index_custody_bindings,
)
from feedbax.contracts.strict_json import (
    DuplicateJsonKeyError,
    StrictJsonError,
    strict_json_loads,
)

pytestmark = [pytest.mark.feedbax_contract]


DIGEST = "c" * 64


# -- well-formed documents parse exactly as they did -------------------------

VALID_DOCUMENTS: tuple[str, ...] = (
    "{}",
    "[]",
    "null",
    "true",
    "false",
    "0",
    "-12",
    "1.5",
    "1e3",
    '"text"',
    '{"a": 1}',
    '{"a": 1, "b": 2, "c": 3}',
    '{"a": [1, 2, {"b": null}], "c": {"d": {"e": []}}}',
    '[[[{"a": [1, {"b": 2}]}]]]',
    '{"unicode": "\\u00e9\\u4e2d", "esc": "a\\"b", "empty key": {"": 1}}',
    '{"outer": {"same": 1}, "other": {"same": 2}, "list": [{"same": 3}, {"same": 4}]}',
    '{"a": 1e400, "b": -0.0, "c": 12345678901234567890}',
)


@pytest.mark.parametrize("document", VALID_DOCUMENTS)
def test_strict_loader_parses_valid_json_identically_to_json_loads(document: str) -> None:
    """No well-formed document changes meaning: this is a refusal, not a codec."""
    strict = strict_json_loads(document)
    plain = json.loads(document)
    assert strict == plain
    # Equality alone would not catch a reordered mapping, and Feedbax hashes
    # canonical re-serializations of parsed documents in several places.
    assert json.dumps(strict, sort_keys=False) == json.dumps(plain, sort_keys=False)
    assert repr(strict) == repr(plain)


def test_strict_loader_accepts_bytes_and_text_alike() -> None:
    document = '{"a": {"b": [1, 2, 3]}}'
    assert strict_json_loads(document) == strict_json_loads(document.encode("utf-8"))
    assert strict_json_loads(bytearray(document.encode("utf-8"))) == json.loads(document)


def test_strict_loader_still_raises_the_ordinary_decode_error() -> None:
    with pytest.raises(json.JSONDecodeError):
        strict_json_loads("{not json")


def test_a_key_repeated_at_different_paths_is_not_a_duplicate() -> None:
    """One name used by two different objects states two different facts."""
    document = '{"left": {"id": "a"}, "right": {"id": "b"}, "rows": [{"id": "c"}, {"id": "d"}]}'
    assert strict_json_loads(document) == json.loads(document)


# -- a repeated member name refuses, and names where ------------------------


@pytest.mark.parametrize(
    ("document", "json_path", "key"),
    [
        ('{"id": "a", "id": "b"}', "$.id", "id"),
        ('{"a": 1, "b": 2, "a": 3}', "$.a", "a"),
        ('{"outer": {"id": "a", "id": "b"}}', "$.outer.id", "id"),
        ('{"rows": [{"ok": 1}, {"id": "a", "id": "b"}]}', "$.rows[1].id", "id"),
        ('[[{"deep": {"k": 1, "k": 2}}]]', "$[0][0].deep.k", "k"),
        ('{"a b": 1, "a b": 2}', '$["a b"]', "a b"),
        ('{"": 1, "": 2}', '$[""]', ""),
        ('{"n": {"m": [0, {"p": [{"q": 1, "q": 2}]}]}}', "$.n.m[1].p[0].q", "q"),
    ],
)
def test_duplicate_object_keys_refuse_with_their_json_path(
    document: str, json_path: str, key: str
) -> None:
    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        strict_json_loads(document)
    assert excinfo.value.json_path == json_path
    assert excinfo.value.key == key
    assert json_path in str(excinfo.value)


def test_the_refusal_names_the_document_when_the_caller_does() -> None:
    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        strict_json_loads('{"id": 1, "id": 2}', ref="some-manifest.json")
    assert excinfo.value.ref == "some-manifest.json"
    assert "some-manifest.json" in str(excinfo.value)


def test_the_refusal_is_a_value_error_so_existing_handlers_fail_closed() -> None:
    assert issubclass(DuplicateJsonKeyError, StrictJsonError)
    assert issubclass(StrictJsonError, ValueError)


def test_the_earliest_duplicate_in_document_order_is_reported() -> None:
    document = '{"first": {"a": 1, "a": 2}, "second": {"b": 1, "b": 2}}'
    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        strict_json_loads(document)
    assert excinfo.value.json_path == "$.first.a"


def test_a_duplicate_the_standard_parser_would_silently_collapse_refuses() -> None:
    """The whole point: ``json.loads`` answers, and answers one of the two."""
    document = '{"sha256": "' + "a" * 64 + '", "sha256": "' + "b" * 64 + '"}'
    assert json.loads(document)["sha256"] == "b" * 64
    with pytest.raises(DuplicateJsonKeyError):
        strict_json_loads(document)


# -- the real loaders reach the refusal -------------------------------------


def _analysis_manifest(manifest_id: str = "feedbax-analysis:strict-json") -> AnalysisRunManifest:
    spec = AnalysisRunSpec(analysis_type="feedbax.test.strict_json", inputs=[], params={})
    return AnalysisRunManifest(
        id=manifest_id,
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec", spec.model_dump(mode="json", exclude_none=True)
        ),
        provenance=Provenance(parents=[]),
    )


def _with_duplicated_member(raw: bytes, member: str, value: Any) -> bytes:
    """Return ``raw`` with ``member`` stated once more, ahead of its real value."""
    assert raw.startswith(b"{")
    prefix = json.dumps({member: value})[1:-1].encode("utf-8")
    return b"{" + prefix + b"," + raw[1:]


def test_load_manifest_bytes_refuses_a_manifest_that_states_its_id_twice() -> None:
    manifest = _analysis_manifest()
    raw = manifest.model_dump_json(exclude_none=True).encode("utf-8")
    assert load_manifest_bytes(raw).id == manifest.id

    duplicated = _with_duplicated_member(raw, "id", "feedbax-analysis:decoy")
    # Standard parsing answers, and the decoy simply disappears.
    assert json.loads(duplicated)["id"] == manifest.id
    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        load_manifest_bytes(duplicated)
    assert excinfo.value.json_path == "$.id"


def test_load_manifest_bytes_refuses_a_duplicate_nested_in_provenance() -> None:
    manifest = _analysis_manifest("feedbax-analysis:strict-json-nested")
    raw = manifest.model_dump_json(exclude_none=True).encode("utf-8")
    document = json.loads(raw)
    text = json.dumps(document)
    nested = text.replace('"provenance": {', '"provenance": {"parents": [], ', 1)
    assert nested != text
    assert json.loads(nested)["provenance"]["parents"] == []
    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        load_manifest_bytes(nested.encode("utf-8"))
    assert excinfo.value.json_path == "$.provenance.parents"


def _row_index() -> AuthenticatedRowIndex:
    return AuthenticatedRowIndex.model_validate(
        {
            "index_id": "strict-json-index",
            "rows": [{"row_id": row, "label": row} for row in ("r0", "r1")],
        }
    )


def _row_parent(manifest_id: str) -> dict[str, Any]:
    return {
        "kind": "AnalysisRunManifest",
        "id": manifest_id,
        "role": "observations",
        "metadata": {
            "ref_schema_id": "feedbax.ref.authenticated_manifest",
            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
            "manifest_sha256": DIGEST,
            "size_bytes": 128,
        },
    }


def test_row_index_custody_sidecar_refuses_a_duplicated_binding_key(tmp_path: Path) -> None:
    index = _row_index()
    custody = build_row_index_custody_bindings(
        index,
        {row_id: {"observations": _row_parent(f"run-{row_id}")} for row_id in index.row_ids},
    )
    path = write_row_index_custody_bindings(custody, tmp_path / "custody.json", index=index)
    assert load_row_index_custody_bindings(path).index_id == index.index_id

    text = path.read_text(encoding="utf-8")
    duplicated = "{" + json.dumps({"index_id": "some-other-index"})[1:-1] + "," + text[1:]
    assert json.loads(duplicated)["index_id"] == index.index_id
    duplicated_path = tmp_path / "custody-duplicated.json"
    duplicated_path.write_text(duplicated, encoding="utf-8")
    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        load_row_index_custody_bindings(duplicated_path)
    assert excinfo.value.json_path == "$.index_id"


STRICT_JSON_TRAINING_ID = "feedbax-training-run:strict-json"
STRICT_JSON_TRAINING_URI = "manifests/training_runs/strict-json.json"


def test_evaluation_input_resolution_refuses_a_manifest_that_states_a_member_twice(
    tmp_path: Path,
) -> None:
    manifest = TrainingRunManifest(id=STRICT_JSON_TRAINING_ID, status="completed")
    path = tmp_path / STRICT_JSON_TRAINING_URI
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (manifest.model_dump_json(indent=2, exclude_none=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    run_spec = EvaluationRunSpec(
        evaluation_type="tests.strict_json_boundary",
        inputs=[
            ParentRef(
                kind="TrainingRunManifest",
                id=STRICT_JSON_TRAINING_ID,
                role="training_run",
            )
        ],
    )
    (resolved,) = resolve_evaluation_inputs(run_spec, manifest_root=tmp_path)
    assert resolved.id == STRICT_JSON_TRAINING_ID

    duplicated = _with_duplicated_member(raw, "status", "failed")
    assert json.loads(duplicated)["status"] == "completed"
    path.write_bytes(duplicated)
    # The census cannot decide a candidate it cannot parse, so the refusal
    # surfaces as the uniqueness refusal that quotes it. Either way the bytes
    # never resolve.
    with pytest.raises(EvaluationInputResolutionError) as excinfo:
        resolve_evaluation_inputs(run_spec, manifest_root=tmp_path)
    assert "states a member twice" in str(excinfo.value)
    assert "$.status" in str(excinfo.value)
