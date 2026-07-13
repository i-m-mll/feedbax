from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.execution import run_analyses_with_context
from feedbax.analysis.materialization import (
    ContextMaterializer,
    MaterializationResult,
    directory_artifact_group,
    existing_file_artifact_group,
    manifest_artifact_group,
    read_json_payload,
)
from feedbax.contracts.manifest import AnalysisRunSpec, load_manifest
from tests.analysis_fixtures import build_toy_analysis_data


def _array_closing_materializer(value: float):
    array = jnp.asarray([value, value + 1])

    def materialize(_context):
        return {"value": array[0]}

    return materialize


def test_context_materializer_identity_does_not_traverse_callable_array_closures() -> None:
    first = ContextMaterializer(
        materializer=_array_closing_materializer(1.0),
        artifact_role="toy",
        logical_name="toy.json",
    )
    second = ContextMaterializer(
        materializer=_array_closing_materializer(9.0),
        artifact_role="toy",
        logical_name="toy.json",
    )

    assert first._field_params == {
        "artifact_role": "toy",
        "logical_name": "toy.json",
        "materializer_input": "context",
        "schema_boundary": None,
        "metadata": {},
    }
    assert first.md5_str == second.md5_str


@pytest.mark.parametrize(
    ("changed", "value"),
    [
        ("artifact_role", "other-role"),
        ("logical_name", "other.json"),
        ("materializer_input", "context_and_data"),
        ("schema_boundary", "toy.v2"),
        ("metadata", {"schema": "toy.v2"}),
    ],
)
def test_context_materializer_declared_identity_fields_affect_hash(
    changed: str,
    value,
) -> None:
    defaults = {
        "materializer": _array_closing_materializer(1.0),
        "artifact_role": "toy",
        "logical_name": "toy.json",
        "materializer_input": "context",
        "schema_boundary": "toy.v1",
        "metadata": {"schema": "toy.v1"},
    }
    baseline = ContextMaterializer(**defaults)
    changed_fields = {**defaults, changed: value}
    candidate = ContextMaterializer(**changed_fields)

    assert baseline.md5_str != candidate.md5_str


def test_context_materializer_explicitly_dispatches_analysis_input_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    data = build_toy_analysis_data()
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(analysis_type="toy_context_and_data_materializer"),
        root=tmp_path,
    )

    def materialize(run_context, analysis_data):
        assert analysis_data.states == data.states
        return {
            "manifest_id": run_context.manifest_id,
            "state_value": analysis_data.states["value"],
        }

    analysis = ContextMaterializer(
        materializer=materialize,
        materializer_input="context_and_data",
        artifact_role="toy_context_and_data",
        logical_name="toy/context-and-data.json",
    )
    _analyses, results, _figures = run_analyses_with_context(
        {"materializer": analysis},
        data,
        context,
    )

    assert results["materializer"] == {
        "manifest_id": context.manifest_id,
        "state_value": 2,
    }
    manifest = load_manifest(context.manifest_path)
    assert manifest.artifacts[0].role == "toy_context_and_data"


def test_context_materializer_rejects_unknown_input_mode() -> None:
    with pytest.raises(ValueError, match="materializer_input"):
        ContextMaterializer(
            materializer=lambda context: {},
            materializer_input="data",
            artifact_role="toy",
            logical_name="toy.json",
        )


def test_existing_file_artifact_group_skips_missing_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    assert existing_file_artifact_group(
        tmp_path / "missing.json",
        group_id="optional",
        role="optional",
    ) == ()

    payload_path = tmp_path / "payload.json"
    payload_path.write_text("{}\n", encoding="utf-8")
    (group,) = existing_file_artifact_group(
        payload_path,
        group_id="payloads",
        role="payload",
        logical_name="nested/payload.json",
        group_role="summary",
        metadata={"source": "test"},
        group_metadata={"schema": "toy.v1"},
    )

    assert group.group_id == "payloads"
    assert group.metadata == {"schema": "toy.v1"}
    assert len(group.members) == 1
    assert group.members[0].logical_name == "nested/payload.json"
    assert group.members[0].metadata == {"source": "test"}


def test_artifact_group_helper_flows_through_context_custody(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(analysis_type="toy_helper_custody"),
        root=tmp_path,
    )

    def materialize(run_context):
        source_path = run_context.results_cache_dir / "source.json"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text('{"value": 5}\n', encoding="utf-8")
        return MaterializationResult(
            payload={"status": "complete"},
            artifact_groups=existing_file_artifact_group(
                source_path,
                group_id="toy-group",
                role="toy-source",
                logical_name="toy/source.json",
                group_role="source",
                group_metadata={"schema": "toy.v1"},
            ),
        )

    analysis = ContextMaterializer(
        materializer=materialize,
        artifact_role="toy-payload",
        logical_name="toy/payload.json",
    )
    run_analyses_with_context(
        {"materializer": analysis},
        build_toy_analysis_data(),
        context,
    )

    manifest = load_manifest(context.manifest_path)
    artifacts_by_role = {artifact.role: artifact for artifact in manifest.artifacts}
    source_ref = artifacts_by_role["toy-source"]
    assert source_ref.metadata["artifact_group"] == {
        "id": "toy-group",
        "member_role": "source",
        "metadata": {"schema": "toy.v1"},
    }
    assert read_json_payload(source_ref.uri) == {"value": 5}


def test_directory_artifact_group_is_recursive_sorted_and_rooted(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "nested").mkdir(parents=True)
    (output_dir / "z.txt").write_text("z", encoding="utf-8")
    (output_dir / "nested" / "a.txt").write_text("a", encoding="utf-8")

    (group,) = directory_artifact_group(
        output_dir,
        group_id="outputs",
        role="output",
        group_role="member",
        logical_name_root=tmp_path,
        metadata_for=lambda path: {"size": path.stat().st_size},
    )

    assert [member.logical_name for member in group.members] == [
        "outputs/nested/a.txt",
        "outputs/z.txt",
    ]
    assert [member.metadata for member in group.members] == [{"size": 1}, {"size": 1}]
    assert directory_artifact_group(
        tmp_path / "missing",
        group_id="missing",
        role="missing",
    ) == ()


def test_directory_artifact_group_rejects_files_outside_logical_root(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "payload.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        directory_artifact_group(
            output_dir,
            group_id="outputs",
            role="output",
            logical_name_root=tmp_path / "elsewhere",
        )


def test_manifest_artifact_group_skips_invalid_entries_and_resolves_paths(
    tmp_path: Path,
) -> None:
    bulk_dir = tmp_path / "bulk"
    bulk_dir.mkdir()
    (bulk_dir / "run_b.npz").write_bytes(b"b")
    (bulk_dir / "run_a.npz").write_bytes(b"a")
    manifest = {
        "runs": {
            "run_b": {"bulk_arrays": {"path": "bulk/run_b.npz", "count": 2}},
            "invalid": {"bulk_arrays": "not-a-mapping"},
            "invalid_path": {"bulk_arrays": {"path": {"nested": "path"}}},
            "missing": {"bulk_arrays": {"path": "bulk/missing.npz"}},
            "run_a": {"bulk_arrays": {"path": "bulk/run_a.npz", "count": 1}},
        }
    }

    (group,) = manifest_artifact_group(
        manifest,
        entries_key="runs",
        artifact_key="bulk_arrays",
        group_id="bulk",
        role="bulk-array",
        path_root=tmp_path,
        group_role="arrays",
        logical_name_for=lambda entry_id, _artifact, _path: f"bulk/{entry_id}.npz",
        metadata_for=lambda entry_id, artifact, _path: {
            "entry_id": entry_id,
            "count": artifact["count"],
        },
    )

    assert [member.logical_name for member in group.members] == [
        "bulk/run_b.npz",
        "bulk/run_a.npz",
    ]
    assert [member.metadata["count"] for member in group.members] == [2, 1]
    assert manifest_artifact_group(
        {"runs": []},
        entries_key="runs",
        artifact_key="bulk_arrays",
        group_id="bulk",
        role="bulk-array",
    ) == ()


def test_read_json_payload_accepts_non_mapping_json(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text('["café", 3]\n', encoding="utf-8")

    assert read_json_payload(payload_path) == ["café", 3]
