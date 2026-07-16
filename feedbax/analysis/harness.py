"""Generic row materialization harness for evaluation and analysis pipelines."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from feedbax.analysis.rendering import render_markdown_note
from feedbax.contracts.manifest import (
    ArtifactRef,
    RegenerationCommand,
    RegenerationSpec,
    store_bytes_artifact,
    store_json_artifact,
    write_manifest,
)

CustodyMode = Literal["manifest", "content-addressed"]


@dataclass(frozen=True)
class SemanticChange:
    """One path-addressed structural difference."""

    path: str
    before: Any
    after: Any
    kind: Literal["added", "removed", "changed"]


@dataclass(frozen=True)
class MaterializedRow:
    """Artifacts and execution result for one resolved condition row."""

    row_id: str
    resolved: dict[str, Any]
    result: Any
    manifest_path: Path | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    regeneration: RegenerationSpec | None = None


@dataclass(frozen=True)
class HarnessResult:
    """Complete output of one harness invocation."""

    rows: tuple[MaterializedRow, ...]
    note: str
    escape_hatch_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def semantic_diff(before: Any, after: Any, *, path: str = "$") -> list[SemanticChange]:
    """Return a deterministic, path-tracked structural diff."""
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        changes: list[SemanticChange] = []
        before_keys = set(before)
        after_keys = set(after)
        for key in sorted(before_keys | after_keys, key=str):
            child = f"{path}.{key}"
            if key not in before:
                changes.append(SemanticChange(child, None, after[key], "added"))
            elif key not in after:
                changes.append(SemanticChange(child, before[key], None, "removed"))
            else:
                changes.extend(semantic_diff(before[key], after[key], path=child))
        return changes
    if isinstance(before, Sequence) and not isinstance(before, (str, bytes)) and isinstance(
        after, Sequence
    ) and not isinstance(after, (str, bytes)):
        changes = []
        for index in range(max(len(before), len(after))):
            child = f"{path}[{index}]"
            if index >= len(before):
                changes.append(SemanticChange(child, None, after[index], "added"))
            elif index >= len(after):
                changes.append(SemanticChange(child, before[index], None, "removed"))
            else:
                changes.extend(semantic_diff(before[index], after[index], path=child))
        return changes
    if before != after:
        return [SemanticChange(path, before, after, "changed")]
    return []


def diff_resolved_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[SemanticChange]:
    """Compare two fully resolved rows semantically."""
    return semantic_diff(left, right)


def diff_regenerated_archived(
    regenerated: Mapping[str, Any], archived: Mapping[str, Any]
) -> list[SemanticChange]:
    """Compare regenerated output with its archived representation."""
    return semantic_diff(archived, regenerated)


class MatrixMaterializerHarness:
    """Own row expansion, execution, custody, manifests, replay data, and notes."""

    def __init__(self, *, root: Path | str, custody: CustodyMode = "content-addressed"):
        if custody not in ("manifest", "content-addressed"):
            raise ValueError(
                "unknown output custody route; registered routes are: "
                "content-addressed, manifest"
            )
        self.root = Path(root)
        self.custody = custody

    def materialize(
        self,
        rows: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        execute: Callable[[str, Mapping[str, Any], Path], tuple[Any, Path | None]],
        command: Sequence[str],
        title: str,
        source: str,
        escape_hatch_reason: str | None = None,
        matrix_metadata: Mapping[str, Any] | None = None,
        regeneration_parameters: Mapping[str, Any] | None = None,
    ) -> HarnessResult:
        """Execute resolved rows and emit standard custody and regeneration records."""
        if not rows:
            raise ValueError("materializer requires at least one row or condition")
        if escape_hatch_reason is not None and not escape_hatch_reason.strip():
            raise ValueError("flat-spec escape hatch requires a stated non-empty reason")
        materialized: list[MaterializedRow] = []
        shared_metadata = dict(matrix_metadata or {})
        replay_parameters = dict(regeneration_parameters or {})
        for row_id, resolved in rows:
            if not row_id:
                raise ValueError("materializer row_id must be non-empty")
            row_root = self.root / row_id
            row_root.mkdir(parents=True, exist_ok=True)
            result, manifest_path = execute(row_id, resolved, row_root)
            regeneration = RegenerationSpec(
                command=RegenerationCommand(argv=list(command)),
                parameters={
                    **replay_parameters,
                    "row_id": row_id,
                    "resolved": dict(resolved),
                },
                metadata={
                    "source": source,
                    "custody": self.custody,
                    "escape_hatch_reason": escape_hatch_reason,
                    **shared_metadata,
                },
            )
            artifacts: list[ArtifactRef] = []
            if self.custody == "content-addressed":
                artifacts.append(
                    store_json_artifact(
                        dict(resolved),
                        root=self.root,
                        role="resolved_row_spec",
                        logical_name=f"{row_id}.resolved.json",
                        metadata={"row_id": row_id, "source": source},
                    )
                )
                artifacts.append(
                    store_json_artifact(
                        regeneration.model_dump(mode="json", exclude_none=True),
                        root=self.root,
                        role="regeneration_spec",
                        logical_name=f"{row_id}.regeneration.json",
                        metadata={"row_id": row_id, "source": source},
                    )
                )
            if manifest_path is not None and hasattr(result, "metadata") and hasattr(
                result, "artifacts"
            ):
                result.metadata.setdefault("matrix_harness", {}).update(
                    {
                        "row_id": row_id,
                        "source": source,
                        "custody": self.custody,
                        "escape_hatch_reason": escape_hatch_reason,
                        **shared_metadata,
                        "regeneration_spec": regeneration.model_dump(
                            mode="json", exclude_none=True
                        ),
                    }
                )
                known_artifact_ids = {artifact.artifact_id for artifact in result.artifacts}
                result.artifacts.extend(
                    artifact
                    for artifact in artifacts
                    if artifact.artifact_id not in known_artifact_ids
                )
                manifest_path = write_manifest(result, root=row_root)
            materialized.append(
                MaterializedRow(
                    row_id=row_id,
                    resolved=dict(resolved),
                    result=result,
                    manifest_path=manifest_path,
                    artifacts=tuple(artifacts),
                    regeneration=regeneration,
                )
            )
        note = render_markdown_note(
            title=title,
            narrative=(
                f"Flat-spec escape hatch: {escape_hatch_reason}"
                if escape_hatch_reason is not None
                else None
            ),
            rows=[("Source", source), ("Conditions", len(materialized))]
            + [("Row", row.row_id) for row in materialized],
        )
        if self.custody == "content-addressed":
            store_bytes_artifact(
                note.encode("utf-8"),
                root=self.root,
                role="materialization_note",
                logical_name="materialization.md",
                media_type="text/markdown",
                suffix=".md",
                metadata={"source": source, "escape_hatch_reason": escape_hatch_reason},
            )
        return HarnessResult(
            tuple(materialized),
            note,
            escape_hatch_reason,
            metadata={
                **shared_metadata,
                "regeneration_parameters": replay_parameters,
            },
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Materialize a serialized evaluation matrix through the standard harness."""
    parser = argparse.ArgumentParser(prog="feedbax matrix-harness")
    parser.add_argument("spec", help="EvaluationRunMatrixSpec JSON path")
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--repo-root")
    parser.add_argument("--plugin", action="append")
    parser.add_argument("--escape-hatch-reason")
    parser.add_argument("--parent-manifest-root")
    parser.add_argument("--execution-descriptor")
    parser.add_argument("--artifact-provider", action="append", default=[])
    parser.add_argument("--checkpoint-custody", action="append", default=[])
    args = parser.parse_args(argv)
    from feedbax.plugins import load_training_method_plugins

    load_training_method_plugins(modules=args.plugin)
    from feedbax.analysis.evaluation import execute_evaluation_run_matrix
    from feedbax.analysis.execution_context import (
        StagedArtifactProviderRootBinding,
        StagedCheckpointCustodyRootBinding,
    )

    def binding_parts(value: str, *, option: str) -> tuple[str, str]:
        name, separator, root = value.partition("=")
        if not separator or not name or not root:
            raise ValueError(f"{option} entries must use NAME=ROOT")
        return name, root

    payload = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    execution_descriptor = (
        json.loads(Path(args.execution_descriptor).read_text(encoding="utf-8"))
        if args.execution_descriptor is not None
        else None
    )
    execute_evaluation_run_matrix(
        payload,
        root=args.manifest_root,
        repo_root=args.repo_root,
        escape_hatch_reason=args.escape_hatch_reason,
        parent_manifest_root=args.parent_manifest_root,
        execution_descriptor=execution_descriptor,
        artifact_provider_bindings=[
            StagedArtifactProviderRootBinding(
                *binding_parts(value, option="--artifact-provider")
            )
            for value in args.artifact_provider
        ],
        checkpoint_custody_bindings=[
            StagedCheckpointCustodyRootBinding(
                *binding_parts(value, option="--checkpoint-custody")
            )
            for value in args.checkpoint_custody
        ],
    )
    return 0


__all__ = [
    "HarnessResult",
    "MaterializedRow",
    "MatrixMaterializerHarness",
    "SemanticChange",
    "diff_regenerated_archived",
    "diff_resolved_rows",
    "semantic_diff",
]
