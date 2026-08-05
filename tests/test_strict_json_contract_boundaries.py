"""Every authority boundary in ``feedbax/contracts`` refuses a two-authority document.

A JSON object that states one member name twice states two authorities for one
fact. ``json.loads`` answers anyway, keeping the last value, so a document can
satisfy every digest, pin, and schema check this layer performs while presenting
a different first value to whatever else reads the bytes. These tests hold the
boundaries where that document could previously enter: the packet importer, the
packet index, the target manifest root, repo-ref composition parents, extraction
sources, content-pinned bases, authoring budgets, project declarations,
evaluation-state containers, and NPZ array stores.

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

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.contracts.artifact_schema import (
    METADATA_KEY,
    ArrayRecord,
    ArrayStorePayload,
    read_npz_array_store,
    write_npz_array_store,
)
from feedbax.contracts.authoring_budget import load_authoring_budget_document
from feedbax.contracts.evaluation_states import (
    EVALUATION_STATES_METADATA_KEY,
    EVALUATION_STATES_METADATA_VALUES_KEY,
    evaluation_states_container_bytes,
    load_evaluation_states_container_bytes,
)
from feedbax.contracts.experiment_envelope import ExperimentEnvelopeRejection
from feedbax.contracts.extraction import SourceBinding, load_expression_context
from feedbax.contracts.manifest import (
    TrainingRunManifest,
    canonical_json_bytes,
    sha256_bytes,
    write_manifest,
)
from feedbax.contracts.manifest_packet import (
    export_manifest_packet,
    import_manifest_packet,
)
from feedbax.contracts.matrix_core import (
    ContentPinnedJsonBase,
    load_content_pinned_json_base,
)
from feedbax.contracts.project_experiment import (
    PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID,
    PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION,
    AuthoringBudgetResource,
    ProjectExperimentDeclarationError,
    parse_project_declaration,
)
from feedbax.contracts.run_composition import (
    CompositionNode,
    ResolvedOutputParent,
    flatten_repo_composition,
)
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.strict_json import DuplicateJsonKeyError, strict_json_loads

pytestmark = [pytest.mark.feedbax_contract]

BUDGET_FIXTURE = (
    Path(__file__).parent / "fake_project_experiment" / "budgets" / "quillon.envelope_budgets.json"
)


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


def _rewrite_zip_member(data: bytes, name: str, replacement: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data), mode="r") as source:
        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                payload = replacement if info.filename == name else source.read(info.filename)
                target.writestr(info, payload)
    return output.getvalue()


# -- the one semantic difference, measured -----------------------------------


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


# -- manifest packet import: the externally supplied packet -------------------


def _packet_source(root: Path) -> str:
    manifest = TrainingRunManifest(
        id="feedbax-training-run:strict-json-packet",
        status="completed",
        summary_metrics={"loss": 0.25},
    )
    write_manifest(manifest, root=root)
    return manifest.id


def _packet_index(packet: Path) -> dict[str, Any]:
    return json.loads((packet / "packet.json").read_text(encoding="utf-8"))


def test_packet_manifest_that_states_its_status_twice_is_refused(tmp_path: Path) -> None:
    """The packet author controls both the manifest bytes and their recorded digest."""
    source = tmp_path / "source"
    manifest_id = _packet_source(source)
    packet = tmp_path / "packet"
    export_manifest_packet([manifest_id], root=source, dest=packet)

    # A well-formed packet still imports exactly as it did.
    good_target = tmp_path / "good_target"
    imported = import_manifest_packet(packet, root=good_target)
    assert imported.imported_manifest_ids == [manifest_id]

    index = _packet_index(packet)
    entry = next(item for item in index["manifests"] if item["id"] == manifest_id)
    manifest_path = packet / entry["path"]
    duplicated = _state_member_twice(
        manifest_path.read_text(encoding="utf-8"), "status", "failed"
    )
    _assert_decoy_is_invisible_to_json_loads(duplicated, "status", "completed")
    manifest_path.write_text(duplicated, encoding="utf-8")
    # Recompute the packet's own digest so the sha256 gate cannot mask the point.
    entry["sha256"] = sha256_bytes(manifest_path.read_bytes())
    (packet / "packet.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    target = tmp_path / "target"
    with pytest.raises(Exception) as excinfo:
        import_manifest_packet(packet, root=target)
    message = str(excinfo.value)
    assert "duplicate JSON object key 'status'" in message
    assert "$.status" in message
    assert not (target / "manifests").exists()


def test_packet_index_that_states_its_schema_id_twice_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest_id = _packet_source(source)
    packet = tmp_path / "packet"
    export_manifest_packet([manifest_id], root=source, dest=packet)

    index_path = packet / "packet.json"
    text = index_path.read_text(encoding="utf-8")
    duplicated = _state_member_twice(text, "schema_id", "attacker.spec.packet")
    _assert_decoy_is_invisible_to_json_loads(
        duplicated, "schema_id", json.loads(text)["schema_id"]
    )
    index_path.write_text(duplicated, encoding="utf-8")

    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        import_manifest_packet(packet, root=tmp_path / "target")
    assert excinfo.value.key == "schema_id"
    assert excinfo.value.json_path == "$.schema_id"
    assert str(index_path) in str(excinfo.value)


def test_target_manifest_root_document_that_states_a_member_twice_is_refused(
    tmp_path: Path,
) -> None:
    """The import scan of the destination root reads it as authority too.

    ``load_manifest`` on the same path already refuses this document, so routing
    the scan's own parse adds no new refusal; it makes the two parses of one
    file agree instead of disagreeing about which value is authoritative.
    """
    source = tmp_path / "source"
    manifest_id = _packet_source(source)
    packet = tmp_path / "packet"
    export_manifest_packet([manifest_id], root=source, dest=packet)

    target = tmp_path / "target"
    existing = TrainingRunManifest(
        id="feedbax-training-run:strict-json-resident", status="completed"
    )
    resident_path = write_manifest(existing, root=target)
    duplicated = _state_member_twice(
        resident_path.read_text(encoding="utf-8"), "status", "failed"
    )
    _assert_decoy_is_invisible_to_json_loads(duplicated, "status", "completed")
    resident_path.write_text(duplicated, encoding="utf-8")

    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        import_manifest_packet(packet, root=target)
    assert excinfo.value.json_path == "$.status"


# -- repo-ref composition parents --------------------------------------------


def _composition_over(ref: str, resolved_root_hash: str) -> CompositionNode:
    return CompositionNode(
        name="strict-json-composition",
        parent=ResolvedOutputParent(ref=ref, resolved_root_hash=resolved_root_hash),
    )


def test_repo_composition_parent_that_states_a_member_twice_is_refused(tmp_path: Path) -> None:
    """The pin covers the *parsed* document, so the duplicate satisfies it."""
    resolved = {"learning_rate": 0.001, "steps": 10}
    ref = "intents/parent.json"
    path = tmp_path / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    node = _composition_over(ref, training_spec_sha256(resolved))

    flattened = flatten_repo_composition(node, repo_root=tmp_path)
    assert flattened.payload == resolved

    duplicated = _state_member_twice(
        path.read_text(encoding="utf-8"), "learning_rate", 10.0
    )
    _assert_decoy_is_invisible_to_json_loads(duplicated, "learning_rate", 0.001)
    path.write_text(duplicated, encoding="utf-8")
    # The pin still matches, because the pin is computed from the parse.
    assert training_spec_sha256(json.loads(duplicated)) == node.parent.resolved_root_hash

    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        flatten_repo_composition(node, repo_root=tmp_path)
    assert excinfo.value.key == "learning_rate"
    assert excinfo.value.ref == ref


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


# -- authoring budgets -------------------------------------------------------


def test_authoring_budget_that_states_a_cap_twice_is_refused() -> None:
    """Two caps for one dimension is exactly the fact a budget must not restate."""
    raw = BUDGET_FIXTURE.read_bytes()
    document = load_authoring_budget_document(raw, field="budget.json")
    assert document["schema_id"] == "feedbax.spec.authoring_budget"

    text = raw.decode("utf-8")
    duplicated = text.replace(
        '"training": {', '"training": {"max_bytes": 1048576, ', 1
    )
    assert duplicated != text
    assert json.loads(duplicated)["layers"]["training"]["max_bytes"] == 8192

    with pytest.raises(ExperimentEnvelopeRejection) as excinfo:
        load_authoring_budget_document(duplicated.encode("utf-8"), field="budget.json")
    message = str(excinfo.value)
    assert "duplicate JSON object key 'max_bytes'" in message
    assert "$.layers.training.max_bytes" in message
    assert "budget.json" in message


# -- project declarations ----------------------------------------------------


def _declaration_document() -> dict[str, Any]:
    return {
        "schema_id": PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID,
        "schema_version": PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION,
        "project": "probe",
        "envelope_directory": "envelopes",
        "output_directory": "results",
        "authoring_budget": "probe.budgets.json",
    }


def test_project_declaration_that_states_its_project_twice_is_refused(tmp_path: Path) -> None:
    budget_root = tmp_path / "budgets"
    budget_root.mkdir()
    (budget_root / "probe.budgets.json").write_bytes(BUDGET_FIXTURE.read_bytes())

    raw = json.dumps(_declaration_document()).encode("utf-8")
    declaration = parse_project_declaration(
        raw, budget_root=budget_root, source="tests:probe"
    )
    assert declaration.project == "probe"
    assert isinstance(declaration.authoring_budget, AuthoringBudgetResource)

    duplicated = _state_member_twice(raw.decode("utf-8"), "project", "attacker")
    _assert_decoy_is_invisible_to_json_loads(duplicated, "project", "probe")
    with pytest.raises(ProjectExperimentDeclarationError) as excinfo:
        parse_project_declaration(
            duplicated.encode("utf-8"), budget_root=budget_root, source="tests:probe"
        )
    message = str(excinfo.value)
    assert "duplicate JSON object key 'project'" in message
    assert "tests:probe" in message


# -- evaluation-state containers ---------------------------------------------


def _states() -> dict[str, Any]:
    return {"trajectory": jnp.arange(6.0).reshape(2, 3), "note": {"units": "cm", "trials": 2}}


def test_evaluation_states_container_metadata_stating_a_member_twice_is_refused() -> None:
    data, _payload = evaluation_states_container_bytes(_states())
    loaded = load_evaluation_states_container_bytes(data)
    assert np.allclose(np.asarray(loaded["trajectory"]), np.asarray(_states()["trajectory"]))

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        text = archive.read(EVALUATION_STATES_METADATA_KEY).decode("utf-8")
    duplicated = _state_member_twice(text, "schema_version", "attacker.v0")
    _assert_decoy_is_invisible_to_json_loads(
        duplicated, "schema_version", json.loads(text)["schema_version"]
    )
    tampered = _rewrite_zip_member(
        data, EVALUATION_STATES_METADATA_KEY, duplicated.encode("utf-8")
    )

    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        load_evaluation_states_container_bytes(tampered)
    assert excinfo.value.key == "schema_version"


def test_evaluation_states_container_metadata_values_stating_a_member_twice_is_refused() -> None:
    data, _payload = evaluation_states_container_bytes(_states())
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        values = archive.read(EVALUATION_STATES_METADATA_VALUES_KEY).decode("utf-8")
        header = json.loads(archive.read(EVALUATION_STATES_METADATA_KEY).decode("utf-8"))

    records = json.loads(values)
    assert isinstance(records, list) and records
    entry = json.dumps(records[0])
    duplicated_entry = _state_member_twice(entry, "index", 99)
    duplicated_values = json.dumps(records[1:])[1:] if len(records) > 1 else "]"
    duplicated = "[" + duplicated_entry + ("," + duplicated_values if len(records) > 1 else "]")
    assert json.loads(duplicated)[0]["index"] == records[0]["index"]

    # The section digest is recomputed, as a container author would.
    header["metadata_sha256"] = sha256_bytes(duplicated.encode("utf-8"))
    tampered = _rewrite_zip_member(
        data, EVALUATION_STATES_METADATA_VALUES_KEY, duplicated.encode("utf-8")
    )
    tampered = _rewrite_zip_member(
        tampered,
        EVALUATION_STATES_METADATA_KEY,
        (json.dumps(header, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

    with pytest.raises(DuplicateJsonKeyError) as excinfo:
        load_evaluation_states_container_bytes(tampered)
    assert excinfo.value.key == "index"


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
