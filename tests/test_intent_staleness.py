import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from feedbax.bin.provider import ProviderStalenessResolutionError
from feedbax.bin.provider import main as provider_main
from feedbax.contracts.graph import (
    GRAPH_SPEC_SCHEMA_ID,
    GRAPH_SPEC_SCHEMA_VERSION,
    GRAPH_SPEC_SCHEMA_VERSION_V3,
)
from feedbax.contracts.intent_diff import (
    collect_contract_identities,
    staleness_report,
)
from feedbax.contracts.run_composition import (
    AuthoredIntentParent,
    CompositionDelta,
    CompositionNode,
    FlattenedIntent,
    InlineIntentParent,
    ResolvedOutputParent,
    authored_envelope_hash,
    flatten_composition,
)
from feedbax.contracts.spec_storage import training_spec_sha256


def _root(payload: dict[str, object]) -> CompositionNode:
    return CompositionNode(
        name="root",
        parent=InlineIntentParent(
            payload=payload,
            schema_id=str(payload["schema_id"]),
            schema_version=str(payload["schema_version"]),
        ),
    )


def _stale_child(root: CompositionNode) -> CompositionNode:
    return CompositionNode(
        name="child",
        parent=AuthoredIntentParent(ref="root.json", content_hash=authored_envelope_hash(root)),
        deltas=[
            CompositionDelta(
                layer_id="child-pinned-contract",
                schema_id=GRAPH_SPEC_SCHEMA_ID,
                schema_version=GRAPH_SPEC_SCHEMA_VERSION_V3,
                patches=[
                    {
                        "op": "replace",
                        "path": "schema_version",
                        "value": GRAPH_SPEC_SCHEMA_VERSION_V3,
                    }
                ],
            )
        ],
    )


def test_staleness_report_attributes_old_version_to_child_layer() -> None:
    root = _root({"schema_id": GRAPH_SPEC_SCHEMA_ID, "schema_version": GRAPH_SPEC_SCHEMA_VERSION})
    flattened = flatten_composition(_stale_child(root), lambda _parent: root)

    report = staleness_report(flattened)

    assert report.model_dump(mode="json") == {
        "stale": [
            {
                "schema_id": GRAPH_SPEC_SCHEMA_ID,
                "found_version": GRAPH_SPEC_SCHEMA_VERSION_V3,
                "current_version": GRAPH_SPEC_SCHEMA_VERSION,
                "layer": "child-pinned-contract",
            }
        ],
        "unresolved": [],
    }


def test_current_contracts_produce_empty_report() -> None:
    flattened = flatten_composition(
        _root({"schema_id": GRAPH_SPEC_SCHEMA_ID, "schema_version": GRAPH_SPEC_SCHEMA_VERSION}),
        lambda _parent: {},
    )
    assert staleness_report(flattened).model_dump(mode="json") == {
        "stale": [],
        "unresolved": [],
    }


def test_unknown_family_is_unresolved() -> None:
    flattened = FlattenedIntent(
        payload={"schema_id": "example.spec.unknown", "schema_version": "example.spec.unknown.v1"},
        attribution={"schema_version": "unknown-layer"},
        layer_hashes=[],
    )
    report = staleness_report(flattened)
    assert report.stale == []
    assert report.unresolved[0].model_dump(mode="json") == {
        "schema_id": "example.spec.unknown",
        "schema_version": "example.spec.unknown.v1",
        "layer": "unknown-layer",
    }


def test_contract_identity_collection_is_deterministic_and_keeps_duplicates() -> None:
    flattened = FlattenedIntent(
        payload={
            "z": {"schema_id": "example.spec.z", "schema_version": "example.spec.z.v1"},
            "a": [
                {"schema_id": "example.spec.a", "schema_version": "example.spec.a.v2"},
                {"schema_id": "example.spec.a", "schema_version": "example.spec.a.v2"},
            ],
        },
        attribution={"z": "z-layer", "a.0": "a-layer", "a.1.schema_id": "a-layer"},
        layer_hashes=[],
    )

    first = collect_contract_identities(flattened)
    second = collect_contract_identities(flattened)

    assert first == second
    assert [identity.schema_id for identity in first] == [
        "example.spec.a",
        "example.spec.a",
        "example.spec.z",
    ]
    assert [identity.layer for identity in first] == ["a-layer", "a-layer", "z-layer"]


def test_provider_staleness_cli_json_smoke_and_strict_exit(tmp_path: Path, capsys) -> None:
    root = _root({"schema_id": GRAPH_SPEC_SCHEMA_ID, "schema_version": GRAPH_SPEC_SCHEMA_VERSION})
    child = _stale_child(root)
    (tmp_path / "root.json").write_text(root.model_dump_json(), encoding="utf-8")
    child_path = tmp_path / "child.json"
    child_path.write_text(child.model_dump_json(), encoding="utf-8")

    assert (
        provider_main(["staleness", str(child_path), "--repo-root", str(tmp_path), "--json"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["stale"][0]["layer"] == "child-pinned-contract"
    assert (
        provider_main(["staleness", str(child_path), "--repo-root", str(tmp_path), "--strict"]) == 1
    )
    output = capsys.readouterr().out
    assert GRAPH_SPEC_SCHEMA_VERSION_V3 in output
    assert GRAPH_SPEC_SCHEMA_VERSION in output


def test_provider_staleness_cli_current_is_strict_clean(tmp_path: Path, capsys) -> None:
    root = _root({"schema_id": GRAPH_SPEC_SCHEMA_ID, "schema_version": GRAPH_SPEC_SCHEMA_VERSION})
    root_path = tmp_path / "root.json"
    root_path.write_text(root.model_dump_json(), encoding="utf-8")

    assert (
        provider_main(["staleness", str(root_path), "--repo-root", str(tmp_path), "--strict"]) == 0
    )
    assert capsys.readouterr().out == "No stale or unresolved contract identities.\n"


def test_provider_staleness_cli_unresolved_only_is_strict_clean(tmp_path: Path, capsys) -> None:
    root = _root({"schema_id": "example.spec.unknown", "schema_version": "example.spec.unknown.v1"})
    root_path = tmp_path / "root.json"
    root_path.write_text(root.model_dump_json(), encoding="utf-8")

    assert (
        provider_main(["staleness", str(root_path), "--repo-root", str(tmp_path), "--strict"]) == 0
    )
    output = capsys.readouterr().out
    assert "Unresolved contract identities:" in output
    assert "example.spec.unknown: example.spec.unknown.v1" in output


def test_provider_staleness_resolves_repo_root_parents_with_file_stdin_parity(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root({"schema_id": GRAPH_SPEC_SCHEMA_ID, "schema_version": GRAPH_SPEC_SCHEMA_VERSION})
    child = _stale_child(root).model_copy(
        update={
            "parent": AuthoredIntentParent(
                ref="specs/training/root.json",
                content_hash=authored_envelope_hash(root),
            )
        }
    )
    parent_path = tmp_path / "specs" / "training" / "root.json"
    parent_path.parent.mkdir(parents=True)
    parent_path.write_text(root.model_dump_json(), encoding="utf-8")
    child_path = tmp_path / "specs" / "training" / "nested" / "child.json"
    child_path.parent.mkdir()
    child_path.write_text(child.model_dump_json(), encoding="utf-8")

    arguments = ["staleness", str(child_path), "--repo-root", str(tmp_path), "--json"]
    assert provider_main(arguments) == 0
    from_path = json.loads(capsys.readouterr().out)

    monkeypatch.setattr(sys, "stdin", StringIO(child.model_dump_json()))
    assert provider_main(["staleness", "-", "--repo-root", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == from_path


def test_provider_staleness_resolves_exact_custodied_parent_and_reports_project_schema(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "schema_id": "rlrmp2.spec.training_method.nominal_cs_gru",
        "schema_version": "rlrmp2.spec.training_method.nominal_cs_gru.v2",
    }
    parent = ResolvedOutputParent(
        ref=f"artifact-blob:{training_spec_sha256(payload)}",
        resolved_root_hash=training_spec_sha256(payload),
        row_id="native-baseline",
        checkpoint_transaction_id="tx-baseline",
    )
    node = CompositionNode(name="nominal-stochastic-hold", parent=parent)
    path = tmp_path / "specs" / "training" / "nominal.json"
    path.parent.mkdir(parents=True)
    path.write_text(node.model_dump_json(), encoding="utf-8")

    def resolve(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:4] == ["mandible", "artifact", "content", parent.ref]
        assert kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
            "timeout": 30,
        }
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr("feedbax.bin.provider.subprocess.run", resolve)
    assert provider_main(["staleness", str(path), "--repo-root", str(tmp_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["stale"] == []
    assert report["unresolved"] == [
        {
            "schema_id": "rlrmp2.spec.training_method.nominal_cs_gru",
            "schema_version": "rlrmp2.spec.training_method.nominal_cs_gru.v2",
        }
    ]


def test_provider_staleness_refuses_resolved_output_without_exact_custody_authority(
    tmp_path: Path,
) -> None:
    node = CompositionNode(
        name="unsupported-parent",
        parent=ResolvedOutputParent(ref="local-parent.json", resolved_root_hash="0" * 64),
    )
    path = tmp_path / "child.json"
    path.write_text(node.model_dump_json(), encoding="utf-8")

    with pytest.raises(
        ProviderStalenessResolutionError,
        match="requires an exact artifact-blob:<sha256> custody ref",
    ):
        provider_main(["staleness", str(path), "--repo-root", str(tmp_path)])
