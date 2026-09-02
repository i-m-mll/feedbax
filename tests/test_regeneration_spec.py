from __future__ import annotations

from pathlib import Path

import pytest

from feedbax.contracts.base import (
    ArtifactRef,
    EntrypointRef,
    ParentRef,
    Provenance,
    canonical_json_bytes,
    file_hash_ref,
    sha256_bytes,
    sha256_file,
    tree_hash_ref,
)
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    REGENERATION_SPEC_SCHEMA_ID,
    REGENERATION_SPEC_SCHEMA_VERSION,
    RegenerationCommand,
    RegenerationSpec,
    ReportManifest,
    load_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion


def test_file_and_tree_hash_refs_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "src" / "analysis.py"
    source.parent.mkdir()
    source.write_text("print('analysis')\n", encoding="utf-8")
    data = tmp_path / "src" / "data.json"
    data.write_text('{"value": 1}\n', encoding="utf-8")

    file_ref = file_hash_ref(source, root=tmp_path, role="source")
    tree_ref = tree_hash_ref(source.parent, root=tmp_path, role="source_tree")

    assert file_ref.path == "src/analysis.py"
    assert file_ref.sha256 == sha256_file(source)
    assert file_ref.size_bytes == source.stat().st_size
    assert tree_ref.path == "src"
    assert tree_ref.file_count == 2
    assert tree_ref.total_size_bytes == source.stat().st_size + data.stat().st_size
    assert [entry.path for entry in tree_ref.files] == ["analysis.py", "data.json"]
    assert tree_ref.sha256 == sha256_bytes(
        canonical_json_bytes(
            [entry.model_dump(mode="json", exclude_none=True) for entry in tree_ref.files]
        )
    )


def test_regeneration_spec_serializes_command_params_and_source_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.py"
    source.write_text("REPORT = True\n", encoding="utf-8")
    source_ref = file_hash_ref(source, root=tmp_path)
    spec = RegenerationSpec(
        command=RegenerationCommand(argv=["python", "report.py"], cwd="."),
        parameters={"format": "pdf", "threshold": 0.1},
        inputs=[ParentRef(kind="AnalysisRunManifest", id="analysis-1")],
        outputs=[
            ArtifactRef(
                role="report",
                logical_name="report.pdf",
                artifact_id="artifact://sha256/report",
                sha256="f" * 64,
            )
        ],
        source_files=[source_ref],
        provenance=Provenance(
            source_repo="https://example.invalid/feedbax.git",
            source_branch="feature/replay",
            source_commit="abc123",
            dirty=True,
            entrypoint=EntrypointRef(kind="python", command="python report.py"),
        ),
    )

    payload = spec_payload("RegenerationSpec", spec.model_dump(mode="json"))

    assert payload.schema_id == REGENERATION_SPEC_SCHEMA_ID
    assert payload.schema_version == REGENERATION_SPEC_SCHEMA_VERSION
    assert payload.inline["command"]["argv"] == ["python", "report.py"]
    assert payload.inline["parameters"] == {"format": "pdf", "threshold": 0.1}
    assert payload.inline["source_files"][0]["sha256"] == source_ref.sha256
    assert payload.inline["provenance"]["dirty"] is True
    assert payload.sha256 == sha256_bytes(canonical_json_bytes(payload.inline))


def test_analysis_and_report_manifests_round_trip_regeneration_spec_refs(
    tmp_path: Path,
) -> None:
    regeneration_payload = spec_payload(
        "RegenerationSpec",
        RegenerationSpec(
            command=RegenerationCommand(shell_command="python make_figures.py"),
            parameters={"figure": "loss"},
            inputs=[ParentRef(kind="EvaluationRunManifest", id="eval-1")],
            outputs=[ArtifactRef(role="figure", logical_name="loss.png")],
        ).model_dump(mode="json"),
    )
    analysis = AnalysisRunManifest(
        id="feedbax-analysis-run:test",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "toy", "inputs": [], "params": {}},
        ),
        regeneration_specs=[regeneration_payload],
    )
    report = ReportManifest(
        id="feedbax-report:test",
        report_spec=spec_payload(
            "ReportSpec",
            {"report_type": "toy", "inputs": [], "params": {}},
        ),
        regeneration_specs=[
            ParentRef(
                kind="RegenerationSpec",
                id="feedbax-regeneration:test",
                uri="artifact://sha256/regeneration",
            )
        ],
    )

    analysis_path = write_manifest(analysis, root=tmp_path, index=False)
    report_path = write_manifest(report, root=tmp_path, index=False)
    loaded_analysis = load_manifest(analysis_path)
    loaded_report = load_manifest(report_path)

    assert isinstance(loaded_analysis, AnalysisRunManifest)
    assert loaded_analysis.regeneration_specs[0].kind == "RegenerationSpec"
    assert loaded_analysis.regeneration_specs[0].schema_id == REGENERATION_SPEC_SCHEMA_ID
    assert isinstance(loaded_report, ReportManifest)
    assert loaded_report.regeneration_specs[0].kind == "RegenerationSpec"
    assert loaded_report.regeneration_specs[0].uri == "artifact://sha256/regeneration"


def test_regeneration_spec_rejects_unsupported_old_version() -> None:
    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        spec_payload(
            "RegenerationSpec",
            {
                "schema_id": REGENERATION_SPEC_SCHEMA_ID,
                "schema_version": "feedbax.spec.regeneration.v0",
                "command": {"argv": ["python", "old.py"]},
            },
        )

    message = str(excinfo.value)
    assert "family='RegenerationSpec'" in message
    assert f"schema_id='{REGENERATION_SPEC_SCHEMA_ID}'" in message
    assert "source_version='feedbax.spec.regeneration.v0'" in message
    assert "migration_intentionally_absent=yes" in message
