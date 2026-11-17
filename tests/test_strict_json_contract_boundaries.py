"""Every authority boundary in ``feedbax/contracts`` refuses a two-authority document.

A JSON object that states one member name twice states two authorities for one
fact. ``json.loads`` answers anyway, keeping the last value, so a document can
satisfy every digest, pin, and schema check this layer performs while presenting
a different first value to whatever else reads the bytes. These tests hold the
boundaries where that document could previously enter: the packet importer, the
packet index, the target manifest root, repo-ref composition parents, extraction
sources, content-pinned bases, evaluation-state containers, and NPZ array stores.

Each boundary is stated twice: a well-formed document still loads exactly as it
did, and the two-authority document is refused with a message naming what and
where. The hostile documents here are internally consistent — their digests and
pins are recomputed after the duplicate is introduced — because that is the
document a packet author or corpus author actually controls.

The measured semantic difference between ``strict_json_loads`` and ``json.loads``
is the duplicate refusal and nothing else: every other input the permissive
parser accepts is still accepted, which
``test_the_strict_loader_tightens_only_the_duplicate_case`` states directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from feedbax.contracts.artifact_schema import (
    METADATA_KEY,
    ArrayRecord,
    ArrayStorePayload,
    read_npz_array_store,
    write_npz_array_store,
)
from feedbax.contracts.extraction import SourceBinding, load_expression_context
from feedbax.contracts.base import (
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.matrix_core import (
    ContentPinnedJsonBase,
    load_content_pinned_json_base,
)
from feedbax.contracts.strict_json import DuplicateJsonKeyError, strict_json_loads

pytestmark = [pytest.mark.feedbax_contract]

# -- helpers -----------------------------------------------------------------


def _state_member_twice(text: str, member: str, decoy: Any) -> str:
    """Return ``text`` with ``member`` stated once more, ahead of its real value.

    ``json.loads`` keeps the *last* value, so the returned document parses to
    exactly what the original did while its bytes lead with ``decoy``.
    """
    stripped = text.lstrip()
    assert stripped.startswith("{"), "only an object document can state a member twice"
    prefix = json.dumps({member: decoy})[1:-1]
    return "{" + prefix + "," + stripped[1:]


def _assert_decoy_is_invisible_to_json_loads(text: str, member: str, expected: Any) -> None:
    """The premise of every test below: the permissive parser sees nothing wrong."""
    assert json.loads(text)[member] == expected


# -- strict parsing -----------------------------------------------------------


PERMISSIVE_INPUTS_THAT_MUST_STILL_BE_ACCEPTED: tuple[str, ...] = (
    # Non-finite literals: json.loads accepts these extensions, and so must the
    # strict loader, or durable metric payloads carrying NaN would stop loading.
    '{"loss": NaN}',
    '{"loss": Infinity, "other": -Infinity}',
    # Deep nesting, large integers, and float extremes.
    '{"a": [[[[[{"b": 12345678901234567890}]]]]], "c": 1e400, "d": -0.0}',
    # Escapes, lone surrogates in escaped form, empty and non-identifier keys.
    '{"": 1, "a b": 2, "\\ud83d\\ude00": 3, "esc": "a\\"b\\\\c\\u0000"}',
    # Leading/trailing whitespace, which json.loads tolerates.
    '  \n\t{"a": 1}\n  ',
    # A repeated name used by two *different* objects is not a duplicate.
    '{"left": {"id": "a"}, "right": {"id": "b"}}',
)

PERMISSIVE_INPUTS_ALREADY_REFUSED: tuple[str, ...] = (
    '{"a": 1} trailing',
    "{'a': 1}",
    '{"a": 1,}',
    "",
)


@pytest.mark.parametrize("document", PERMISSIVE_INPUTS_THAT_MUST_STILL_BE_ACCEPTED)
def test_the_strict_loader_tightens_only_the_duplicate_case(document: str) -> None:
    """Everything ``json.loads`` accepts, minus duplicates, is still accepted.

    This is the empirical statement that routing a boundary through the strict
    loader cannot newly refuse a document for size, encoding, non-finite
    literals, nesting depth, integer magnitude, or whitespace: the strict loader
    delegates all of that to ``json.loads`` unchanged.
    """
    strict = strict_json_loads(document)
    plain = json.loads(document)
    assert repr(strict) == repr(plain)
    assert json.dumps(strict, sort_keys=False) == json.dumps(plain, sort_keys=False)


@pytest.mark.parametrize("document", PERMISSIVE_INPUTS_ALREADY_REFUSED)
def test_documents_the_permissive_parser_already_refuses_are_unchanged(document: str) -> None:
    with pytest.raises(json.JSONDecodeError):
        json.loads(document)
    with pytest.raises(json.JSONDecodeError):
        strict_json_loads(document)


# -- extraction sources ------------------------------------------------------


def test_extraction_source_that_states_a_member_twice_is_refused(tmp_path: Path) -> None:
    payload = {"anchors": {"epsilon": 0.1}, "count": 3}
    uri = "results/source.json"
    path = tmp_path / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    binding = SourceBinding(alias="source", kind="tests.strict_json", uri=uri)

    context = load_expression_context([binding], tmp_path)
    assert context.items["source"].payload == payload

    duplicated = _state_member_twice(path.read_text(encoding="utf-8"), "count", 99)
    _assert_decoy_is_invisible_to_json_loads(duplicated, "count", 3)
    path.write_text(duplicated, encoding="utf-8")

    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        load_expression_context([binding], tmp_path)
    assert excinfo.value.key == "count"
    assert excinfo.value.ref == uri


# -- content-pinned bases ----------------------------------------------------


def test_content_pinned_base_that_states_a_member_twice_is_refused(tmp_path: Path) -> None:
    """The pin is computed over the parse, so it cannot detect the duplicate."""
    document = {"aggregation": "mean", "radius": 3}
    path = tmp_path / "base.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    pin = sha256_bytes(canonical_json_bytes(document))
    base = ContentPinnedJsonBase(ref="base.json", sha256=pin)

    assert load_content_pinned_json_base(base, repo_root=tmp_path) == document

    duplicated = _state_member_twice(path.read_text(encoding="utf-8"), "aggregation", "max")
    _assert_decoy_is_invisible_to_json_loads(duplicated, "aggregation", "mean")
    path.write_text(duplicated, encoding="utf-8")
    assert sha256_bytes(canonical_json_bytes(json.loads(duplicated))) == pin

    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        load_content_pinned_json_base(base, repo_root=tmp_path)
    assert excinfo.value.key == "aggregation"
    assert excinfo.value.ref == "base.json"


# -- NPZ array stores --------------------------------------------------------


def test_npz_array_store_metadata_stating_a_member_twice_is_refused(tmp_path: Path) -> None:
    """Pydantic's own JSON reader keeps the last value exactly as ``json.loads`` does."""
    path = tmp_path / "store.npz"
    arrays = {"policy.weights": np.arange(6, dtype=np.float32).reshape(2, 3)}
    write_npz_array_store(path, arrays, store_role="params")

    store = read_npz_array_store(path)
    assert store.payload.roles == ["policy.weights"]
    assert store.payload.arrays[0].shape == (2, 3)
    np.testing.assert_array_equal(store.arrays["policy.weights"], arrays["policy.weights"])

    with np.load(path, allow_pickle=False) as npz:
        members = {name: npz[name] for name in npz.files}
    text = members[METADATA_KEY].tobytes().decode("utf-8")
    duplicated = _state_member_twice(text, "store_role", "optimizer")
    _assert_decoy_is_invisible_to_json_loads(duplicated, "store_role", "params")
    members[METADATA_KEY] = np.asarray(duplicated.encode("utf-8"))
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **members)

    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        read_npz_array_store(tampered)
    assert excinfo.value.key == "store_role"
    assert str(tampered) in str(excinfo.value)


def test_npz_array_store_payload_still_validates_through_the_python_path(tmp_path: Path) -> None:
    """Routing the parse must not change how the payload model coerces its fields."""
    payload = ArrayStorePayload(
        store_role="params",
        arrays=[
            ArrayRecord(
                role="policy.weights",
                storage_key="array_000000",
                dtype="float32",
                shape=(2, 3),
                sha256="0" * 64,
            )
        ],
    )
    document = payload.model_dump(mode="json", exclude_none=True)
    from_json = ArrayStorePayload.model_validate_json(json.dumps(document))
    from_python = ArrayStorePayload.model_validate(strict_json_loads(json.dumps(document)))
    assert from_json == from_python
    assert from_python.arrays[0].shape == (2, 3)
