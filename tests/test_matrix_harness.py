from pathlib import Path

import pytest

from feedbax.analysis.harness import (
    MatrixMaterializerHarness,
    diff_regenerated_archived,
    diff_resolved_rows,
)


def test_harness_owns_row_paths_custody_replay_and_note(tmp_path: Path) -> None:
    seen: list[Path] = []

    def execute(row_id, resolved, root):
        seen.append(root)
        return {"row_id": row_id, **resolved}, root / "manifest.json"

    result = MatrixMaterializerHarness(root=tmp_path).materialize(
        [("control", {"gain": 1}), ("treatment", {"gain": 2})],
        execute=execute,
        command=["python", "pipeline.py"],
        title="Evaluation conditions",
        source="test.matrix",
    )

    assert seen == [tmp_path / "control", tmp_path / "treatment"]
    assert [row.row_id for row in result.rows] == ["control", "treatment"]
    assert all(row.regeneration is not None for row in result.rows)
    assert all(
        {artifact.role for artifact in row.artifacts}
        == {"resolved_row_spec", "shared_regeneration_context", "regeneration_spec"}
        for row in result.rows
    )
    assert len(
        {
            artifact.artifact_id
            for row in result.rows
            for artifact in row.artifacts
            if artifact.role == "shared_regeneration_context"
        }
    ) == 1
    assert "| Conditions | 2 |" in result.note


def test_escape_hatch_requires_and_records_reason(tmp_path: Path) -> None:
    harness = MatrixMaterializerHarness(root=tmp_path, custody="manifest")
    with pytest.raises(ValueError, match="stated non-empty reason"):
        harness.materialize(
            [("flat", {})],
            execute=lambda *_: ({}, None),
            command=["tool"],
            title="Flat",
            source="test.flat",
            escape_hatch_reason=" ",
        )

    result = harness.materialize(
        [("flat", {})],
        execute=lambda *_: ({}, None),
        command=["tool"],
        title="Flat",
        source="test.flat",
        escape_hatch_reason="legacy analysis cannot be expressed as rows",
    )
    assert result.escape_hatch_reason == "legacy analysis cannot be expressed as rows"
    assert "Flat-spec escape hatch" in result.note


def test_semantic_diff_uses_structural_paths() -> None:
    row_changes = diff_resolved_rows(
        {"params": {"gain": 1}, "items": ["a"]},
        {"params": {"gain": 2, "bias": 3}, "items": ["a", "b"]},
    )
    assert [(change.path, change.kind) for change in row_changes] == [
        ("$.items[1]", "added"),
        ("$.params.bias", "added"),
        ("$.params.gain", "changed"),
    ]
    archive_changes = diff_regenerated_archived({"value": 2}, {"value": 1})
    assert archive_changes[0].before == 1
    assert archive_changes[0].after == 2
