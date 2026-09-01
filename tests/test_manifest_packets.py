from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from feedbax.analysis.context import AnalysisRunContext
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    EvaluationRunManifest,
    ParentRef,
    ReportManifest,
    TrainingRunManifest,
    load_manifest,
    spec_payload,
    store_bytes_artifact,
    write_manifest,
)
from feedbax.contracts.manifest_packet import (
    MANIFEST_PACKET_SCHEMA_VERSION,
    ManifestPacketValidationError,
    export_manifest_packet,
    import_manifest_packet,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.persistence.manifest_index import iter_manifest_files


def _parent(kind: str, manifest_id: str, role: str) -> ParentRef:
    return ParentRef(kind=kind, id=manifest_id, role=role)


def _write_run_tree(root: Path) -> dict[str, str]:
    training = TrainingRunManifest(
        id="feedbax-training-run:packet-source",
        status="completed",
        summary_metrics={"loss": 0.1},
    )
    write_manifest(training, root=root)

    training_parent = _parent("TrainingRunManifest", training.id, "training_run")
    states = store_bytes_artifact(
        b"states",
        root=root,
        role="evaluation_states",
        logical_name="states.npz",
        metadata={
            "schema_id": "feedbax.manifest.evaluation_states_container",
            "schema_version": "feedbax.manifest.evaluation_states_container.v1",
        },
    )
    evaluation = EvaluationRunManifest(
        id="feedbax-evaluation-run:packet-source",
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            {
                "evaluation_type": "testpkg.packet_eval",
                "inputs": [training_parent.model_dump(mode="json", exclude_none=True)],
                "params": {"n_trials": 2},
            },
        ),
        input_training_runs=[training_parent],
        artifacts=[states],
    )
    write_manifest(evaluation, root=root)

    evaluation_parent = _parent("EvaluationRunManifest", evaluation.id, "evaluation_run")
    analysis_artifact = store_bytes_artifact(
        b"analysis",
        root=root,
        role="data_product",
        logical_name="analysis.json",
        media_type="application/json",
    )
    analysis = AnalysisRunManifest(
        id="feedbax-analysis-run:packet-source",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {
                "analysis_type": "testpkg.packet_analysis",
                "inputs": [evaluation_parent.model_dump(mode="json", exclude_none=True)],
            },
        ),
        inputs=[evaluation_parent],
        artifacts=[analysis_artifact],
    )
    write_manifest(analysis, root=root)

    analysis_parent = _parent("AnalysisRunManifest", analysis.id, "analysis_run")
    report_artifact = store_bytes_artifact(
        b"# report\n",
        root=root,
        role="report_render",
        logical_name="report.md",
        media_type="text/markdown",
    )
    report = ReportManifest(
        id="feedbax-report:packet-source",
        status="completed",
        report_spec=spec_payload(
            "ReportSpec",
            {
                "report_type": "testpkg.packet_report",
                "inputs": [analysis_parent.model_dump(mode="json", exclude_none=True)],
            },
        ),
        inputs=[analysis_parent],
        artifacts=[report_artifact],
    )
    write_manifest(report, root=root)

    return {
        "training": training.id,
        "evaluation": evaluation.id,
        "analysis": analysis.id,
        "report": report.id,
    }


def _packet_index(packet: Path) -> dict[str, object]:
    return json.loads((packet / "packet.json").read_text(encoding="utf-8"))


def _write_packet_index(packet: Path, index: dict[str, object]) -> None:
    (packet / "packet.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def test_manifest_packet_exports_descendant_closure_and_imports_fresh_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    packet = tmp_path / "packet"
    ids = _write_run_tree(source)

    export_result = export_manifest_packet(
        [ids["training"]],
        root=source,
        dest=packet,
        direction="descendants",
        include_artifacts="include",
        producer={"tool": "pytest"},
    )

    assert export_result.manifest_count == 4
    assert export_result.included_artifact_count == 3
    index = _packet_index(packet)
    assert index["schema_version"] == MANIFEST_PACKET_SCHEMA_VERSION
    assert {entry["id"] for entry in index["manifests"]} == set(ids.values())

    import_result = import_manifest_packet(packet, root=target)

    assert set(import_result.imported_manifest_ids) == set(ids.values())
    loaded = {load_manifest(path).id for path in iter_manifest_files(target)}
    assert loaded == set(ids.values())
    for path in iter_manifest_files(target):
        manifest = load_manifest(path)
        assert manifest.metadata["imported_from"]["schema_version"] == (
            MANIFEST_PACKET_SCHEMA_VERSION
        )
        assert manifest.provenance.parents == load_manifest(path).provenance.parents

    conn = sqlite3.connect(target / "index" / "feedbax.sqlite")
    try:
        row_count = conn.execute("SELECT COUNT(*) FROM manifests").fetchone()[0]
    finally:
        conn.close()
    assert row_count == 4


def test_manifest_packet_includes_canonical_analysis_artifact_explicitly_and_automatically(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    payload = {"labels": ["canonical", "packet"]}
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(analysis_type="testpkg.canonical_packet"),
        root=source,
        index_manifest=False,
    )
    artifact = context.record_json_artifact(
        payload,
        role="data_product",
        logical_name="canonical.json",
    )
    manifest, _path = context.finalize(status="completed")
    expected_bytes = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"

    explicit_packet = tmp_path / "explicit-packet"
    explicit = export_manifest_packet(
        [manifest.id],
        root=source,
        dest=explicit_packet,
        include_artifacts=True,
    )
    explicit_entry = _packet_index(explicit_packet)["artifacts"][0]

    assert artifact.uri == artifact.artifact_id
    assert explicit.included_artifact_count == 1
    assert explicit.external_artifact_count == 0
    assert (explicit_packet / explicit_entry["path"]).read_bytes() == expected_bytes

    auto_packet = tmp_path / "auto-packet"
    automatic = export_manifest_packet(
        [manifest.id],
        root=source,
        dest=auto_packet,
        include_artifacts="auto",
    )
    automatic_entry = _packet_index(auto_packet)["artifacts"][0]

    assert automatic.included_artifact_count == 1
    assert automatic.external_artifact_count == 0
    assert automatic_entry["mode"] == "included"
    assert (auto_packet / automatic_entry["path"]).read_bytes() == expected_bytes


def test_manifest_packet_import_is_idempotent_and_rejects_divergent_id(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    packet = tmp_path / "packet"
    ids = _write_run_tree(source)
    export_manifest_packet([ids["training"]], root=source, dest=packet)

    first = import_manifest_packet(packet, root=target)
    second = import_manifest_packet(packet, root=target)

    assert len(first.imported_manifest_ids) == 4
    assert second.imported_manifest_ids == []
    assert set(second.skipped_manifest_ids) == set(ids.values())

    training_path = next(
        path for path in iter_manifest_files(target) if load_manifest(path).id == ids["training"]
    )
    payload = json.loads(training_path.read_text(encoding="utf-8"))
    payload["summary_metrics"] = {"loss": 9.9}
    training_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ManifestPacketValidationError, match="Divergent manifest id"):
        import_manifest_packet(packet, root=target)


def test_manifest_packet_import_fails_closed_for_tampered_manifest_and_artifact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    ids = _write_run_tree(source)

    manifest_packet = tmp_path / "manifest_packet"
    export_manifest_packet([ids["training"]], root=source, dest=manifest_packet)
    first_manifest = next(
        entry["path"]
        for entry in _packet_index(manifest_packet)["manifests"]
        if entry["id"] == ids["training"]
    )
    path = manifest_packet / first_manifest
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ManifestPacketValidationError, match="manifest sha256 mismatch"):
        import_manifest_packet(manifest_packet, root=tmp_path / "manifest_target")
    assert not (tmp_path / "manifest_target" / "manifests").exists()

    artifact_packet = tmp_path / "artifact_packet"
    export_manifest_packet([ids["training"]], root=source, dest=artifact_packet)
    first_artifact = next(
        entry["path"]
        for entry in _packet_index(artifact_packet)["artifacts"]
        if entry["mode"] == "included"
    )
    artifact_path = artifact_packet / first_artifact
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tamper")
    with pytest.raises(ManifestPacketValidationError, match="artifact sha256 mismatch"):
        import_manifest_packet(artifact_packet, root=tmp_path / "artifact_target")
    assert not (tmp_path / "artifact_target" / "manifests").exists()


def test_manifest_packet_import_rejects_manifest_path_traversal_without_writes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    packet = tmp_path / "packet_parent" / "packet"
    target = tmp_path / "target_parent" / "target"
    ids = _write_run_tree(source)
    export_manifest_packet([ids["training"]], root=source, dest=packet)

    index = _packet_index(packet)
    entry = next(
        entry
        for entry in index["manifests"]
        if entry["id"] == ids["training"]
    )
    outside_packet = packet.parent / "escape.json"
    outside_packet.write_bytes((packet / entry["path"]).read_bytes())
    entry["path"] = "../escape.json"
    entry["sha256"] = hashlib.sha256(outside_packet.read_bytes()).hexdigest()
    _write_packet_index(packet, index)

    outside_target = target.parent / "escape.json"
    assert not outside_target.exists()

    with pytest.raises(ManifestPacketValidationError, match="Unsafe packet manifest path"):
        import_manifest_packet(packet, root=target)

    assert not outside_target.exists()
    assert not (target / "manifests").exists()


def test_manifest_packet_import_rejects_absolute_manifest_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    packet = tmp_path / "packet"
    target = tmp_path / "target"
    ids = _write_run_tree(source)
    export_manifest_packet([ids["training"]], root=source, dest=packet)

    index = _packet_index(packet)
    entry = next(
        entry
        for entry in index["manifests"]
        if entry["id"] == ids["training"]
    )
    outside_packet = tmp_path / "absolute_escape.json"
    outside_packet.write_bytes((packet / entry["path"]).read_bytes())
    entry["path"] = str(outside_packet)
    entry["sha256"] = hashlib.sha256(outside_packet.read_bytes()).hexdigest()
    _write_packet_index(packet, index)

    with pytest.raises(ManifestPacketValidationError, match="Unsafe packet manifest path"):
        import_manifest_packet(packet, root=target)

    assert not (target / "manifests").exists()


def test_manifest_packet_import_rejects_artifact_path_traversal_without_writes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    packet = tmp_path / "packet_parent" / "packet"
    target = tmp_path / "target"
    ids = _write_run_tree(source)
    export_manifest_packet([ids["training"]], root=source, dest=packet)

    index = _packet_index(packet)
    entry = next(entry for entry in index["artifacts"] if entry["mode"] == "included")
    outside_packet = packet.parent / "artifact.bin"
    outside_packet.write_bytes((packet / entry["path"]).read_bytes())
    entry["path"] = "../artifact.bin"
    entry["sha256"] = hashlib.sha256(outside_packet.read_bytes()).hexdigest()
    entry["size_bytes"] = outside_packet.stat().st_size
    _write_packet_index(packet, index)

    with pytest.raises(ManifestPacketValidationError, match="Unsafe packet artifact path"):
        import_manifest_packet(packet, root=target)

    assert not (target / "artifacts").exists()
    assert not (target / "manifests").exists()


def test_manifest_packet_import_rejects_packet_version_and_unknown_spec_family(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    packet = tmp_path / "packet"
    ids = _write_run_tree(source)
    export_manifest_packet([ids["training"]], root=source, dest=packet)

    index_path = packet / "packet.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["schema_version"] = "feedbax.spec.manifest_packet.v0"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(UnsupportedSpecVersion, match="ManifestPacket"):
        import_manifest_packet(packet, root=tmp_path / "target_version")

    packet_unknown = tmp_path / "packet_unknown"
    export_manifest_packet([ids["training"]], root=source, dest=packet_unknown)
    eval_entry = next(
        entry
        for entry in _packet_index(packet_unknown)["manifests"]
        if entry["id"] == ids["evaluation"]
    )
    eval_path = packet_unknown / eval_entry["path"]
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    payload["evaluation_spec"]["kind"] = "DownstreamEvaluationSpec"
    payload["evaluation_spec"]["schema_id"] = "rlrmp.spec.evaluation"
    payload["evaluation_spec"]["schema_version"] = "rlrmp.spec.evaluation.v1"
    eval_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    packet_index = _packet_index(packet_unknown)
    for entry in packet_index["manifests"]:
        if entry["id"] == ids["evaluation"]:
            entry["sha256"] = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    _write_packet_index(packet_unknown, packet_index)

    with pytest.raises(ManifestPacketValidationError, match="DownstreamEvaluationSpec"):
        import_manifest_packet(packet_unknown, root=tmp_path / "target_unknown")


def test_manifest_packet_external_artifacts_are_indexed_as_external(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    packet = tmp_path / "packet"
    ids = _write_run_tree(source)

    export_result = export_manifest_packet(
        [ids["training"]],
        root=source,
        dest=packet,
        include_artifacts="external",
    )
    assert export_result.included_artifact_count == 0
    assert export_result.external_artifact_count == 3

    import_manifest_packet(packet, root=target)
    conn = sqlite3.connect(target / "index" / "feedbax.sqlite")
    try:
        backends = {
            row[0] for row in conn.execute("SELECT DISTINCT storage_backend FROM artifacts")
        }
    finally:
        conn.close()

    assert backends == {"external"}


def test_default_registry_registers_manifest_packet_family() -> None:
    family = default_spec_registry.resolve("ManifestPacket")

    assert family.identity == "feedbax.spec.manifest_packet"
    assert family.current_version == MANIFEST_PACKET_SCHEMA_VERSION
    assert family.policy is not None
    assert family.policy.stance == "reject"
