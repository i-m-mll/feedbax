from __future__ import annotations

import os
import hashlib
from pathlib import Path
import shutil

import pytest

from feedbax.analysis.figures import execute_figure_spec
from feedbax.analysis.manifest_inputs import (
    AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
    AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
    authenticated_manifest_ref,
    is_authenticated_manifest_ref,
    resolve_manifest_input,
)
from feedbax.analysis.reports import (
    execute_report_spec,
    register_report_recipe,
    unregister_report_recipe,
)
from feedbax.analysis.specs import (
    execute_analysis_run_spec,
    register_analysis_recipe,
    unregister_analysis_recipe,
)
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    EvaluationRunManifest,
    ParentRef,
    ReportSpec,
    spec_payload,
    write_manifest,
)


def _manifest(root: Path, *, manifest_id: str = "feedbax-evaluation-run:test"):
    manifest = EvaluationRunManifest(
        id=manifest_id,
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            {"evaluation_type": "test.authenticated", "inputs": [], "params": {}},
        ),
    )
    path = write_manifest(manifest, root=root)
    return manifest, path, authenticated_manifest_ref(manifest, path, "evaluation_run")


def _analysis_manifest(root: Path):
    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:authenticated-input",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "test.authenticated.source", "inputs": [], "params": {}},
        ),
    )
    path = write_manifest(manifest, root=root)
    return manifest, path, authenticated_manifest_ref(manifest, path, "analysis_run")


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_authenticated_manifest_ref_is_portable_and_relocation_invariant(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest, path, ref = _manifest(source)
    assert ref.uri is None
    assert ref.metadata["ref_schema_id"] == AUTHENTICATED_MANIFEST_REF_SCHEMA_ID
    assert ref.metadata["ref_schema_version"] == AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION
    assert ref.metadata["size_bytes"] == path.stat().st_size
    assert is_authenticated_manifest_ref(ref)
    assert resolve_manifest_input(ref, source).manifest == manifest

    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)
    resolved = resolve_manifest_input(ref, relocated)
    assert resolved.manifest == manifest
    assert resolved.raw_bytes == path.read_bytes()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("manifest_sha256", "0" * 64, "SHA-256 mismatch"),
        ("size_bytes", 0, "size mismatch"),
        ("size_bytes", True, "invalid byte size"),
        ("ref_schema_id", "unknown", "schema_id"),
        ("ref_schema_version", "unknown", "schema_version"),
    ],
)
def test_authenticated_manifest_ref_rejects_bad_profile(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    _manifest_obj, _path, ref = _manifest(tmp_path)
    bad = ref.model_copy(update={"metadata": {**ref.metadata, field: value}})
    with pytest.raises(ValueError, match=match):
        resolve_manifest_input(bad, tmp_path)


def test_authenticated_manifest_ref_rejects_partial_profile(tmp_path: Path) -> None:
    _manifest_obj, _path, ref = _manifest(tmp_path)
    partial = ref.model_copy(update={"metadata": {"ref_schema_id": ref.metadata["ref_schema_id"]}})
    with pytest.raises(ValueError, match="incomplete"):
        is_authenticated_manifest_ref(partial)


def test_exact_training_parent_hash_metadata_is_not_staged_profile() -> None:
    exact_parent = ParentRef(
        kind="TrainingRunManifest",
        id="feedbax-training-run:exact",
        role="training_run",
        uri=f"artifact://sha256/{'a' * 64}",
        metadata={
            "manifest_sha256": "a" * 64,
            "certificate_sha256": "b" * 64,
            "size_bytes": 1234,
            "run_set_id": "run-set",
            "row_id": "row-1",
            "planned_run_id": "feedbax-training-run:exact",
            "manifest_status": "completed",
            "registration_status": "completed",
            "conformance_overall": "pass",
        },
    )
    assert not is_authenticated_manifest_ref(exact_parent)


def test_authenticated_manifest_ref_rejects_wrong_typed_bytes(tmp_path: Path) -> None:
    raw = b'{"kind":"EvaluationRunManifest","id":"typed-but-incomplete"}\n'
    path = tmp_path / "typed.json"
    path.write_bytes(raw)
    ref = ParentRef(
        kind="EvaluationRunManifest",
        id="typed-but-incomplete",
        metadata={
            "ref_schema_id": AUTHENTICATED_MANIFEST_REF_SCHEMA_ID,
            "ref_schema_version": AUTHENTICATED_MANIFEST_REF_SCHEMA_VERSION,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
    )
    with pytest.raises(ValueError):
        resolve_manifest_input(ref, tmp_path, runtime_locator=path.name)


@pytest.mark.parametrize(
    "locator",
    [
        "",
        "/tmp/manifest.json",
        "file:///tmp/manifest.json",
        "../manifest.json",
        "%2e%2e/manifest.json",
        "%2Ftmp/manifest.json",
        "manifests\\evaluation_runs\\manifest.json",
        "manifest.json?query=1",
        "manifest.json#fragment",
    ],
)
def test_authenticated_manifest_ref_rejects_unsafe_locator(tmp_path: Path, locator: str) -> None:
    _manifest_obj, _path, ref = _manifest(tmp_path)
    with pytest.raises(ValueError):
        resolve_manifest_input(ref, tmp_path, runtime_locator=locator)


def test_authenticated_manifest_ref_rejects_tamper_kind_and_id(tmp_path: Path) -> None:
    _manifest_obj, path, ref = _manifest(tmp_path)
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace('"completed"', '"pending"'), encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        resolve_manifest_input(ref, tmp_path)

    other, other_path, _other_ref = _manifest(tmp_path / "other", manifest_id="other")
    wrong_id_ref = authenticated_manifest_ref(other, other_path, "evaluation_run").model_copy(
        update={"id": "declared-id"}
    )
    with pytest.raises(ValueError, match="id mismatch"):
        resolve_manifest_input(
            wrong_id_ref,
            tmp_path / "other",
            runtime_locator=other_path.relative_to(tmp_path / "other"),
        )

    wrong_kind_ref = authenticated_manifest_ref(other, other_path, "evaluation_run").model_copy(
        update={"kind": "AnalysisRunManifest"}
    )
    with pytest.raises(ValueError, match="kind mismatch"):
        resolve_manifest_input(
            wrong_kind_ref,
            tmp_path / "other",
            runtime_locator=other_path.relative_to(tmp_path / "other"),
        )


def test_authenticated_manifest_ref_rejects_missing_symlink_directory_and_fifo(
    tmp_path: Path,
) -> None:
    _manifest_obj, path, ref = _manifest(tmp_path)
    relative = path.relative_to(tmp_path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        resolve_manifest_input(ref, tmp_path)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    with pytest.raises(ValueError):
        resolve_manifest_input(ref, tmp_path)
    path.rmdir()

    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    path.symlink_to(target)
    with pytest.raises(ValueError):
        resolve_manifest_input(ref, tmp_path)
    path.unlink()

    real_dir = tmp_path / "real-evaluation-runs"
    real_dir.mkdir()
    path.parent.rmdir()
    path.parent.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ValueError):
        resolve_manifest_input(ref, tmp_path)
    path.parent.unlink()
    path.parent.mkdir()

    fifo = tmp_path.joinpath(*relative.parts[:-1], "manifest.fifo")
    os.mkfifo(fifo)
    with pytest.raises(ValueError):
        resolve_manifest_input(ref, tmp_path, runtime_locator=fifo.relative_to(tmp_path))


def test_legacy_manifest_ref_makes_no_authenticated_claim(tmp_path: Path) -> None:
    legacy = ParentRef(
        kind="EvaluationRunManifest",
        id="legacy",
        uri=str(tmp_path / "legacy.json"),
    )
    assert not is_authenticated_manifest_ref(legacy)
    with pytest.raises(ValueError, match="makes no authenticated claim"):
        resolve_manifest_input(legacy, tmp_path)


def test_analysis_integrity_failure_precedes_recipe_cache_and_filesystem_effects(
    tmp_path: Path,
) -> None:
    _manifest_obj, _path, ref = _analysis_manifest(tmp_path)
    bad_ref = ref.model_copy(
        update={"metadata": {**ref.metadata, "size_bytes": ref.metadata["size_bytes"] + 1}}
    )
    spec = AnalysisRunSpec(
        analysis_type="test.authenticated.no_effect_analysis",
        inputs=[bad_ref],
    )
    calls: list[object] = []

    def recipe(run_spec, root, inputs):
        calls.append((run_spec, root, inputs))
        raise AssertionError("analysis recipe must not run")

    register_analysis_recipe(spec.analysis_type, recipe, replace=True)
    before = _file_snapshot(tmp_path)
    try:
        with pytest.raises(ValueError, match="size mismatch"):
            execute_analysis_run_spec(spec, root=tmp_path)
    finally:
        unregister_analysis_recipe(spec.analysis_type)
    assert calls == []
    assert _file_snapshot(tmp_path) == before
    assert not (tmp_path / "cache").exists()


def test_figure_integrity_failure_precedes_render_and_filesystem_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_obj, _path, ref = _analysis_manifest(tmp_path)
    bad_ref = ref.model_copy(update={"metadata": {**ref.metadata, "manifest_sha256": "0" * 64}})
    calls: list[object] = []

    def build_figures(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("figure rendering must not run")

    monkeypatch.setattr("feedbax.analysis.figures._build_figures", build_figures)
    before = _file_snapshot(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        execute_figure_spec(
            FigureSpec(
                name="authenticated-no-effect",
                assembler="feedbax.grid_figure",
                inputs=[bad_ref],
            ),
            root=tmp_path,
        )
    assert calls == []
    assert _file_snapshot(tmp_path) == before


def test_report_integrity_failure_precedes_recipe_and_filesystem_effects(
    tmp_path: Path,
) -> None:
    _manifest_obj, _path, ref = _analysis_manifest(tmp_path)
    metadata = dict(ref.metadata)
    metadata.pop("size_bytes")
    bad_ref = ref.model_copy(update={"metadata": metadata})
    spec = ReportSpec(
        report_type="test.authenticated.no_effect_report",
        inputs=[bad_ref],
    )
    calls: list[object] = []

    def recipe(report_spec, root, inputs):
        calls.append((report_spec, root, inputs))
        raise AssertionError("report recipe must not run")

    register_report_recipe(spec.report_type, recipe, replace=True)
    before = _file_snapshot(tmp_path)
    try:
        with pytest.raises(ValueError, match="incomplete"):
            execute_report_spec(spec, root=tmp_path)
    finally:
        unregister_report_recipe(spec.report_type)
    assert calls == []
    assert _file_snapshot(tmp_path) == before
