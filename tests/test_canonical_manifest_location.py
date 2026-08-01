"""Canonical `(kind, id, root)` manifest location is authoritative over the index."""

from __future__ import annotations

from pathlib import Path

import pytest

from feedbax.analysis.manifest_inputs import authenticated_manifest_ref, resolve_manifest_input
from feedbax.analysis.specs import find_manifest_by_id
from feedbax.contracts.manifest import (
    MANIFEST_KIND_DIRECTORIES,
    MANIFEST_MODELS,
    AnalysisRunManifest,
    EvaluationRunManifest,
    FigureManifest,
    ReportManifest,
    canonical_manifest_candidate_paths,
    canonical_manifest_path,
    canonical_manifest_relative_path,
    manifest_kind_directory,
    spec_payload,
    write_manifest,
)
from feedbax.persistence.manifest_index import (
    connect_index,
    default_index_path,
    index_manifest,
    find_manifest_paths_by_id,
)


def _evaluation_manifest(manifest_id: str = "feedbax-evaluation-run:canonical") -> (
    EvaluationRunManifest
):
    return EvaluationRunManifest(
        id=manifest_id,
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            {"evaluation_type": "test.canonical", "inputs": [], "params": {}},
        ),
    )


def _analysis_manifest(manifest_id: str = "feedbax-analysis-run:canonical") -> AnalysisRunManifest:
    return AnalysisRunManifest(
        id=manifest_id,
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "test.canonical", "inputs": [], "params": {}},
        ),
    )


def _report_manifest(manifest_id: str = "feedbax-report:canonical") -> ReportManifest:
    return ReportManifest(
        id=manifest_id,
        status="completed",
        report_spec=spec_payload(
            "ReportSpec",
            {"report_type": "test.canonical", "inputs": [], "params": {}},
        ),
    )


def _figure_manifest(manifest_id: str = "feedbax-figure:canonical") -> FigureManifest:
    return FigureManifest(
        id=manifest_id,
        status="completed",
        figure_spec=spec_payload("FigureSpec", {"name": "canonical"}),
    )


def test_declared_directories_cover_every_writable_manifest_kind() -> None:
    assert set(MANIFEST_KIND_DIRECTORIES) == set(MANIFEST_MODELS)


def test_unknown_manifest_kind_fails_closed() -> None:
    with pytest.raises(ValueError, match="no canonical storage location"):
        manifest_kind_directory("NotAManifestKind")
    with pytest.raises(ValueError, match="no canonical storage location"):
        canonical_manifest_path("NotAManifestKind", "feedbax-x:1", root=Path("/tmp"))


@pytest.mark.parametrize(
    "manifest",
    [
        _evaluation_manifest(),
        _analysis_manifest(),
        _report_manifest(),
        _figure_manifest(),
    ],
    ids=["evaluation", "analysis", "report", "figure"],
)
def test_write_manifest_uses_the_canonical_location(tmp_path: Path, manifest) -> None:
    written = write_manifest(manifest, root=tmp_path)
    assert written == canonical_manifest_path(manifest.kind, manifest.id, root=tmp_path)
    relative = canonical_manifest_relative_path(manifest.kind, manifest.id)
    assert written == tmp_path.joinpath(*relative.split("/"))


def test_canonical_candidate_paths_are_deterministic_and_complete(tmp_path: Path) -> None:
    first = canonical_manifest_candidate_paths("feedbax-report:x", root=tmp_path)
    second = canonical_manifest_candidate_paths("feedbax-report:x", root=tmp_path)
    assert first == second
    assert len(first) == len(MANIFEST_KIND_DIRECTORIES)


def test_authenticated_staged_locator_uses_the_canonical_layout(tmp_path: Path) -> None:
    manifest = _evaluation_manifest()
    path = write_manifest(manifest, root=tmp_path)
    ref = authenticated_manifest_ref(manifest, path, "evaluation_run")
    resolved = resolve_manifest_input(ref, tmp_path)
    assert resolved.path == canonical_manifest_path(manifest.kind, manifest.id, root=tmp_path)


def test_absent_index_resolves_through_the_canonical_path(tmp_path: Path) -> None:
    manifest = _analysis_manifest()
    path = write_manifest(manifest, root=tmp_path, index=False)
    assert not default_index_path(tmp_path).exists()
    assert find_manifest_paths_by_id(manifest.id, root=tmp_path) == []

    found, found_path = find_manifest_by_id(manifest.id, root=tmp_path)
    assert found_path == path
    assert found.id == manifest.id


def test_stale_index_entry_falls_back_to_the_canonical_path(tmp_path: Path) -> None:
    manifest = _analysis_manifest()
    path = write_manifest(manifest, root=tmp_path)
    stale = tmp_path / "manifests" / "analysis_runs" / "moved-away.json"

    conn = connect_index(default_index_path(tmp_path))
    try:
        index_manifest(conn, manifest, path=stale)
    finally:
        conn.close()
    assert find_manifest_paths_by_id(manifest.id, root=tmp_path) == [stale]
    assert not stale.exists()

    found, found_path = find_manifest_by_id(manifest.id, root=tmp_path)
    assert found_path == path
    assert found.id == manifest.id


def test_canonical_resolution_does_not_pay_for_a_full_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _analysis_manifest()
    path = write_manifest(manifest, root=tmp_path, index=False)

    def _refuse(*_args, **_kwargs):
        raise AssertionError("canonical resolution must not fall through to a full scan")

    monkeypatch.setattr("feedbax.analysis.specs.iter_manifest_files", _refuse)
    assert find_manifest_by_id(manifest.id, root=tmp_path)[1] == path


def test_stale_index_entry_naming_other_bytes_is_ignored(tmp_path: Path) -> None:
    wanted = _analysis_manifest()
    other = _evaluation_manifest("feedbax-evaluation-run:decoy")
    wanted_path = write_manifest(wanted, root=tmp_path)
    other_path = write_manifest(other, root=tmp_path)

    conn = connect_index(default_index_path(tmp_path))
    try:
        index_manifest(conn, wanted, path=other_path)
    finally:
        conn.close()
    assert find_manifest_paths_by_id(wanted.id, root=tmp_path) == [other_path]

    found, found_path = find_manifest_by_id(wanted.id, root=tmp_path)
    assert found_path == wanted_path
    assert found.id == wanted.id


def test_missing_manifest_still_raises_after_every_tier(tmp_path: Path) -> None:
    write_manifest(_analysis_manifest(), root=tmp_path)
    with pytest.raises(FileNotFoundError, match="feedbax-report:absent"):
        find_manifest_by_id("feedbax-report:absent", root=tmp_path)


def test_duplicate_non_canonical_records_refuse(tmp_path: Path) -> None:
    manifest = _analysis_manifest()
    first = tmp_path / "manifests" / "analysis_runs" / "copy-a.json"
    second = tmp_path / "manifests" / "analysis_runs" / "copy-b.json"
    first.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump_json(indent=2, exclude_none=True) + "\n"
    first.write_text(payload, encoding="utf-8")
    second.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="is duplicated under"):
        find_manifest_by_id(manifest.id, root=tmp_path)
