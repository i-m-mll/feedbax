"""The row-custody version boundary: which index *cut* a custody document names.

``RowIndexCustodyBindings`` v1 recorded ``index_id`` and nothing else. An index
id is stable across cuts, so a custody document written for an earlier cut of an
index carries the same id, names rows that still exist, and passes every check
there was. v2 adds ``index_sha256``, the digest of the exact index the bindings
were produced against, which is what turns "belongs to this index" into "belongs
to this cut of it".

The version decision is stated here in full, because the alternative — accepting
v1 and filling the digest in from whatever index is at hand — would manufacture
the very authentication the field exists to carry:

* **accept** — a v2 document validates and binds;
* **migrate** — :func:`migrate_row_index_custody_payload` upgrades a v1 payload,
  but only when the caller hands over the index it is asserting the bindings came
  from, and only when that index plausibly is the one they named;
* **reject** — a v1 document loaded on its own is refused by version, and so is
  any other version. Neither is reinterpreted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from feedbax.contracts.row_index import (
    ROW_INDEX_CUSTODY_SCHEMA_ID,
    ROW_INDEX_CUSTODY_SCHEMA_VERSION,
    ROW_INDEX_CUSTODY_SCHEMA_VERSION_V1,
    ROW_INDEX_CUSTODY_SCHEMA_VERSION_V2,
    AuthenticatedRowIndex,
    RowIndexCustodyBindings,
    RowSelectionError,
    RowSelectionErrorCode,
    build_row_index_custody_bindings,
    load_row_index_custody_bindings,
    migrate_row_index_custody_payload,
    write_row_index_custody_bindings,
)

DIGEST = "c" * 64


def _index(*row_ids: str, index_id: str = "span-survey") -> AuthenticatedRowIndex:
    return AuthenticatedRowIndex.model_validate(
        {
            "index_id": index_id,
            "rows": [{"row_id": row_id, "label": row_id} for row_id in row_ids],
        }
    )


def _parent(manifest_id: str) -> dict[str, Any]:
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


def _custody(index: AuthenticatedRowIndex) -> RowIndexCustodyBindings:
    return build_row_index_custody_bindings(
        index,
        {row_id: {"observations": _parent(f"run-{row_id}")} for row_id in index.row_ids},
    )


# -- the current version pins the cut ---------------------------------------


def test_the_current_version_is_v2_and_pins_the_index_digest() -> None:
    assert ROW_INDEX_CUSTODY_SCHEMA_VERSION == ROW_INDEX_CUSTODY_SCHEMA_VERSION_V2
    assert ROW_INDEX_CUSTODY_SCHEMA_VERSION_V1 == f"{ROW_INDEX_CUSTODY_SCHEMA_ID}.v1"

    index = _index("near", "far")
    document = _custody(index)

    assert document.schema_version == ROW_INDEX_CUSTODY_SCHEMA_VERSION_V2
    assert document.index_sha256 == index.canonical_sha256()


def test_a_custody_document_from_another_cut_of_one_index_is_refused() -> None:
    """The failure v1 could not detect: same id, same rows, different cut."""
    produced_against = _index("near", "far")
    document = _custody(produced_against)
    # The index gains a row. Its id is unchanged and every bound row is still
    # declared, so every v1-era check still passes.
    recut = _index("near", "far", "middle")
    assert recut.index_id == produced_against.index_id
    assert set(binding.row_id for binding in document.bindings) <= set(recut.row_ids)

    with pytest.raises(RowSelectionError) as caught:
        document.require_index(recut)

    assert caught.value.code is RowSelectionErrorCode.INDEX_MISMATCH
    assert document.index_sha256 in str(caught.value)
    assert recut.canonical_sha256() in str(caught.value)


def test_an_index_sha256_that_is_not_a_digest_is_refused() -> None:
    with pytest.raises(ValueError, match="lowercase sha256"):
        RowIndexCustodyBindings.model_validate(
            {"index_id": "span-survey", "index_sha256": "not-a-digest", "bindings": []}
        )


# -- accept, migrate, reject ------------------------------------------------


def _v1_payload(index: AuthenticatedRowIndex) -> dict[str, Any]:
    """One v1 document: the current one with its digest and version rolled back."""
    payload = json.loads(_custody(index).model_dump_json(exclude_none=True))
    payload.pop("index_sha256")
    payload["schema_version"] = ROW_INDEX_CUSTODY_SCHEMA_VERSION_V1
    return payload


def test_a_v1_document_is_rejected_by_version_rather_than_read_as_current() -> None:
    index = _index("near", "far")

    with pytest.raises(ValueError) as caught:
        RowIndexCustodyBindings.model_validate(_v1_payload(index))

    message = str(caught.value)
    assert ROW_INDEX_CUSTODY_SCHEMA_VERSION_V1 in message
    assert "migrate_row_index_custody_payload" in message


def test_loading_a_v1_sidecar_from_disk_refuses_too(tmp_path: Path) -> None:
    index = _index("near", "far")
    path = tmp_path / "stale.row_custody.json"
    path.write_text(json.dumps(_v1_payload(index), indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=ROW_INDEX_CUSTODY_SCHEMA_VERSION_V1):
        load_row_index_custody_bindings(path)


def test_a_v1_document_migrates_when_the_caller_supplies_its_index() -> None:
    index = _index("near", "far")
    payload = _v1_payload(index)

    migrated = migrate_row_index_custody_payload(payload, index=index)

    assert migrated["schema_version"] == ROW_INDEX_CUSTODY_SCHEMA_VERSION_V2
    assert migrated["index_sha256"] == index.canonical_sha256()
    assert migrated["bindings"] == payload["bindings"]
    document = RowIndexCustodyBindings.model_validate(migrated)
    document.require_index(index)


def test_migration_refuses_an_index_the_bindings_never_named() -> None:
    payload = _v1_payload(_index("near", "far"))
    other = _index("near", "far", index_id="a-different-survey")

    with pytest.raises(RowSelectionError) as caught:
        migrate_row_index_custody_payload(payload, index=other)

    assert caught.value.code is RowSelectionErrorCode.INDEX_MISMATCH


def test_migration_refuses_an_index_that_does_not_declare_the_bound_rows() -> None:
    """A digest is never minted over an index the bindings could not have come from."""
    payload = _v1_payload(_index("near", "far"))
    narrowed = _index("near")

    with pytest.raises(RowSelectionError) as caught:
        migrate_row_index_custody_payload(payload, index=narrowed)

    assert caught.value.code is RowSelectionErrorCode.UNRESOLVED_ROW_KEY


def test_migrating_a_current_document_is_a_no_op_that_still_checks_the_cut() -> None:
    index = _index("near", "far")
    payload = json.loads(_custody(index).model_dump_json(exclude_none=True))

    assert migrate_row_index_custody_payload(payload, index=index) == payload

    with pytest.raises(RowSelectionError):
        migrate_row_index_custody_payload(payload, index=_index("near", "far", "middle"))


def test_an_unknown_version_has_nothing_to_migrate_from() -> None:
    index = _index("near", "far")
    payload = {**_v1_payload(index), "schema_version": f"{ROW_INDEX_CUSTODY_SCHEMA_ID}.v9"}

    with pytest.raises(ValueError, match="unsupported RowIndexCustodyBindings schema_version"):
        migrate_row_index_custody_payload(payload, index=index)


# -- the writer keeps the sidecar and the index in agreement ----------------


def test_the_written_sidecar_carries_the_digest_it_was_built_against(
    tmp_path: Path,
) -> None:
    index = _index("near", "far")
    path = write_row_index_custody_bindings(
        _custody(index), tmp_path / "span.row_custody.json", index=index
    )

    reloaded = load_row_index_custody_bindings(path)

    assert reloaded.index_sha256 == index.canonical_sha256()
    reloaded.require_index(index)


def test_writing_custody_for_another_cut_is_refused_before_any_bytes_land(
    tmp_path: Path,
) -> None:
    """The cut comparison is the write's job, not the eventual reader's.

    This document is well formed and internally consistent: it states the cut it
    was built against, and it would load cleanly from anywhere. It is simply
    custody for a different cut of the index the caller is writing custody for,
    which is the one thing serialization cannot notice. Refusing here leaves the
    sidecar untouched, so the wrong cut never becomes the durable answer at the
    path this cut's custody is read from.
    """
    produced_against = _index("near", "far")
    document = _custody(produced_against)
    recut = _index("near", "far", "middle")
    target = tmp_path / "span.row_custody.json"

    with pytest.raises(RowSelectionError) as caught:
        write_row_index_custody_bindings(document, target, index=recut)

    assert caught.value.code is RowSelectionErrorCode.INDEX_MISMATCH
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_writing_custody_without_the_index_cannot_be_expressed() -> None:
    """The index is not an optional check the caller may decline to run.

    While it was omissible, the omission was the failure: a call that passed no
    index wrote whatever document it was handed, and the only thing that had ever
    compared that document with an index was the caller that chose it.
    """
    index = _index("near", "far")

    with pytest.raises(TypeError, match="index"):
        write_row_index_custody_bindings(  # type: ignore[call-arg]
            _custody(index), "unwritten.row_custody.json"
        )


def test_an_explicitly_absent_index_is_a_typed_refusal(tmp_path: Path) -> None:
    """A type hint is not a runtime gate, so ``None`` refuses in the taxonomy."""
    index = _index("near", "far")
    target = tmp_path / "span.row_custody.json"

    with pytest.raises(RowSelectionError) as caught:
        write_row_index_custody_bindings(
            _custody(index),
            target,
            index=None,  # type: ignore[arg-type]
        )

    assert caught.value.code is RowSelectionErrorCode.INDEX_MISMATCH
    assert "cannot be written without the row index" in str(caught.value)
    assert not target.exists()
