"""Generic row materialization harness for evaluation and analysis pipelines."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from feedbax.contracts.strict_json import (
    StrictJsonError,
    strict_json_loads,
)

from feedbax.plugins.bootstrap import BootstrapState

from feedbax.analysis.rendering import render_markdown_note
from feedbax.contracts.artifact_custody import CONTENT_OBJECT_REF_PREFIX, ContentObjectRef
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.contracts.base import (
    ArtifactMigrationRecord,
    ArtifactRef,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.artifact_store import (
    store_json_artifact,
)
from feedbax.contracts.manifest import (
    EVALUATION_RUN_MANIFEST_SCHEMA_VERSION,
    EvaluationRunManifest,
    RegenerationCommand,
    RegenerationSpec,
    write_manifest,
)

CustodyMode = Literal["manifest", "content-addressed"]
SUPPORTED_EXECUTION_POLICY_OVERRIDE_FLAG = "--allow-large-per-row"
SHARED_REGENERATION_ROLE = "shared_regeneration_context"
ROW_REGENERATION_ROLE = "regeneration_spec"


def _store_compact_json_artifact(
    value: Any,
    *,
    root: Path,
    role: str,
    logical_name: str,
    metadata: Mapping[str, Any],
) -> ArtifactRef:
    return ImmutableArtifactBlobProvider(root.absolute()).store_bytes(
        canonical_json_bytes(value),
        role=role,
        logical_name=logical_name,
        media_type="application/json",
        metadata=dict(metadata),
    )


def build_shared_regeneration_context(
    *,
    command: Sequence[str],
    source: str,
    custody: CustodyMode,
    escape_hatch_reason: str | None,
    matrix_metadata: Mapping[str, Any],
    regeneration_parameters: Mapping[str, Any],
) -> RegenerationSpec:
    return RegenerationSpec(
        command=RegenerationCommand(argv=list(command)),
        parameters=dict(regeneration_parameters),
        metadata={
            "source": source,
            "custody": custody,
            **(
                {"escape_hatch_reason": escape_hatch_reason}
                if escape_hatch_reason is not None
                else {}
            ),
            **dict(matrix_metadata),
        },
    )


def publish_shared_regeneration_context(
    *,
    root: Path,
    command: Sequence[str],
    source: str,
    custody: CustodyMode,
    escape_hatch_reason: str | None,
    matrix_metadata: Mapping[str, Any],
    regeneration_parameters: Mapping[str, Any],
) -> ArtifactRef:
    """Publish the batch-wide replay context once in one custody domain."""
    shared = build_shared_regeneration_context(
        command=command,
        source=source,
        custody=custody,
        escape_hatch_reason=escape_hatch_reason,
        matrix_metadata=matrix_metadata,
        regeneration_parameters=regeneration_parameters,
    )
    return _store_compact_json_artifact(
        shared.model_dump(mode="json", exclude_none=True),
        root=root,
        role=SHARED_REGENERATION_ROLE,
        logical_name="matrix.regeneration.json",
        metadata={"source": source},
    )


def build_row_regeneration_spec(
    *,
    command: Sequence[str],
    row_id: str,
    resolved: Mapping[str, Any],
    shared_context: ArtifactRef,
) -> RegenerationSpec:
    return RegenerationSpec(
        command=RegenerationCommand(argv=list(command)),
        parameters={"row_id": row_id, "resolved": dict(resolved)},
        inputs=[shared_context],
    )


def publish_row_regeneration_spec(
    *,
    root: Path,
    command: Sequence[str],
    row_id: str,
    resolved: Mapping[str, Any],
    shared_context: ArtifactRef,
) -> tuple[RegenerationSpec, ArtifactRef]:
    """Publish one row-local replay record referencing exact shared context."""
    row_spec = build_row_regeneration_spec(
        command=command,
        row_id=row_id,
        resolved=resolved,
        shared_context=shared_context,
    )
    ref = _store_compact_json_artifact(
        row_spec.model_dump(mode="json", exclude_none=True),
        root=root,
        role=ROW_REGENERATION_ROLE,
        logical_name=f"{row_id}.regeneration.json",
        metadata={"row_id": row_id},
    )
    return row_spec, ref


def _load_regeneration_artifact(artifact: ArtifactRef, *, root: Path) -> RegenerationSpec:
    raw = ImmutableArtifactBlobProvider(root.absolute()).get_bytes(artifact)
    payload = strict_json_loads(raw, ref=f"regeneration artifact {artifact.artifact_id!r}")
    return RegenerationSpec.model_validate(payload)


def resolve_manifest_regeneration(
    manifest: EvaluationRunManifest,
    *,
    root: Path | str,
) -> RegenerationSpec:
    """Resolve one compact evaluation manifest's full replay description."""
    row_refs = [
        artifact for artifact in manifest.artifacts if artifact.role == ROW_REGENERATION_ROLE
    ]
    if len(row_refs) != 1:
        raise ValueError(
            f"evaluation manifest requires exactly one row regeneration artifact; found {len(row_refs)}"
        )
    row = _load_regeneration_artifact(row_refs[0], root=Path(root))
    shared_refs = [
        item
        for item in row.inputs
        if isinstance(item, ArtifactRef) and item.role == SHARED_REGENERATION_ROLE
    ]
    if len(shared_refs) != 1:
        raise ValueError(
            f"row regeneration requires exactly one shared context artifact; found {len(shared_refs)}"
        )
    declared_shared = [
        artifact for artifact in manifest.artifacts if artifact.role == SHARED_REGENERATION_ROLE
    ]
    if declared_shared != shared_refs:
        raise ValueError("manifest and row regeneration disagree on shared context identity")
    shared = _load_regeneration_artifact(shared_refs[0], root=Path(root))
    if row.command != shared.command:
        raise ValueError("row and shared regeneration commands disagree")

    def merge(label: str, common: Mapping[str, Any], local: Mapping[str, Any]) -> dict[str, Any]:
        conflicts = sorted(key for key in common.keys() & local.keys() if common[key] != local[key])
        if conflicts:
            raise ValueError(f"row and shared regeneration {label} conflict: {conflicts}")
        return {**common, **local}

    return row.model_copy(
        update={
            "parameters": merge("parameters", shared.parameters, row.parameters),
            "metadata": merge("metadata", shared.metadata, row.metadata),
            "inputs": [*shared.inputs, *(item for item in row.inputs if item != shared_refs[0])],
            "outputs": [*shared.outputs, *row.outputs],
            "source_files": [*shared.source_files, *row.source_files],
            "source_trees": [*shared.source_trees, *row.source_trees],
            "provenance": shared.provenance,
        }
    )


@dataclass(frozen=True, slots=True)
class EvaluationManifestStorageConversion:
    """Exact identity mapping for one byte-preserving historical conversion input."""

    manifest_id: str
    source_schema_version: str
    target_schema_version: str
    source_sha256: str
    source_size_bytes: int
    target_sha256: str
    target_size_bytes: int
    target_path: Path


def externalize_evaluation_manifest_storage(
    manifest: EvaluationRunManifest,
    *,
    root: Path | str,
) -> EvaluationRunManifest:
    """Convert inline matrix replay data through the normal shared-content publisher."""
    harness = manifest.metadata.get("matrix_harness")
    if not isinstance(harness, Mapping):
        raise ValueError("evaluation manifest has no matrix_harness metadata to normalize")
    inline = harness.get("regeneration_spec")
    if not isinstance(inline, Mapping):
        raise ValueError("evaluation manifest has no inline regeneration_spec to normalize")
    regeneration = RegenerationSpec.model_validate(inline)
    row_id = regeneration.parameters.get("row_id")
    resolved = regeneration.parameters.get("resolved")
    if not isinstance(row_id, str) or not row_id or not isinstance(resolved, Mapping):
        raise ValueError("inline regeneration_spec lacks one row_id and resolved row mapping")
    declared_row_id = harness.get("row_id")
    if declared_row_id is not None and declared_row_id != row_id:
        raise ValueError("inline regeneration and manifest row_id conflict")

    shared_parameters = {
        key: value
        for key, value in regeneration.parameters.items()
        if key not in {"row_id", "resolved"}
    }
    shared_metadata = dict(regeneration.metadata)
    for key, value in harness.items():
        if key in {"row_id", "regeneration_spec"} or value is None:
            continue
        existing = shared_metadata.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"inline regeneration and manifest {key} conflict")
        shared_metadata[key] = value

    command = regeneration.command.argv
    if not command or regeneration.command.shell_command is not None:
        raise ValueError("matrix regeneration normalization requires one argv command")
    root_path = Path(root)
    source = shared_metadata.pop("source", None)
    custody = shared_metadata.pop("custody", None)
    if not isinstance(source, str) or not source:
        raise ValueError("historical regeneration context requires a non-empty source")
    if custody not in {"manifest", "content-addressed"}:
        raise ValueError(f"historical regeneration context has unsupported custody: {custody!r}")
    shared_ref = publish_shared_regeneration_context(
        root=root_path,
        command=command,
        source=source,
        custody=custody,
        escape_hatch_reason=shared_metadata.pop("escape_hatch_reason", None),
        matrix_metadata=shared_metadata,
        regeneration_parameters=shared_parameters,
    )
    _row, row_ref = publish_row_regeneration_spec(
        root=root_path,
        command=command,
        row_id=row_id,
        resolved=resolved,
        shared_context=shared_ref,
    )
    compact_harness = {
        key: value
        for key in ("row_id", "source", "custody", "escape_hatch_reason")
        if (value := harness.get(key)) is not None
    }
    metadata = {**manifest.metadata, "matrix_harness": compact_harness}
    artifacts = [
        artifact
        for artifact in manifest.artifacts
        if artifact.role not in {SHARED_REGENERATION_ROLE, ROW_REGENERATION_ROLE}
    ]
    artifacts.extend((shared_ref, row_ref))
    return manifest.model_copy(
        update={
            "schema_version": EVALUATION_RUN_MANIFEST_SCHEMA_VERSION,
            "artifacts": artifacts,
            "metadata": metadata,
        }
    )


def _validate_historical_regeneration_artifact(
    manifest: EvaluationRunManifest,
    *,
    source_path: Path,
    regeneration: RegenerationSpec,
) -> None:
    """Verify any legacy replay artifact before replacing its repeated representation."""
    legacy_refs = [
        artifact for artifact in manifest.artifacts if artifact.role == ROW_REGENERATION_ROLE
    ]
    if not legacy_refs:
        return
    if len(legacy_refs) != 1:
        raise ValueError(
            "historical evaluation manifest requires at most one regeneration artifact; "
            f"found {len(legacy_refs)}"
        )
    legacy_ref = legacy_refs[0]
    if legacy_ref.sha256 is None or legacy_ref.size_bytes is None:
        raise ValueError("historical regeneration artifact lacks exact sha256 and size identity")
    relative = legacy_ref.metadata.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("historical regeneration artifact lacks a relative custody path")
    if (
        source_path.parent.name != "evaluation_runs"
        or source_path.parent.parent.name != "manifests"
    ):
        raise ValueError(
            "historical regeneration artifact requires a canonical manifests/evaluation_runs source path"
        )
    source_root = source_path.parent.parent.parent.resolve()
    artifact_path = (source_root / relative).resolve(strict=True)
    if not artifact_path.is_relative_to(source_root):
        raise ValueError("historical regeneration artifact path escapes its custody root")
    artifact_bytes = artifact_path.read_bytes()
    if len(artifact_bytes) != legacy_ref.size_bytes:
        raise ValueError(
            "historical regeneration artifact size disagrees with its manifest identity"
        )
    if sha256_bytes(artifact_bytes) != legacy_ref.sha256:
        raise ValueError(
            "historical regeneration artifact sha256 disagrees with its manifest identity"
        )
    artifact_payload = strict_json_loads(
        artifact_bytes,
        ref=f"historical regeneration artifact {legacy_ref.artifact_id!r}",
    )
    if RegenerationSpec.model_validate(artifact_payload) != regeneration:
        raise ValueError("historical inline and referenced regeneration specs conflict")


def convert_evaluation_manifest_storage(
    source_path: Path | str,
    *,
    destination_root: Path | str,
) -> EvaluationManifestStorageConversion:
    """Normalize one historical file without rebinding its old byte identity."""
    source = Path(source_path)
    source_bytes = source.read_bytes()
    source_payload = strict_json_loads(source_bytes, ref=f"historical manifest {source}")
    if not isinstance(source_payload, Mapping):
        raise ValueError("historical evaluation manifest must be a JSON object")
    source_version = source_payload.get("schema_version")
    if not isinstance(source_version, str):
        raise ValueError("historical evaluation manifest lacks schema_version")
    from feedbax.contracts.manifest import load_manifest_bytes

    loaded = load_manifest_bytes(source_bytes)
    if not isinstance(loaded, EvaluationRunManifest):
        raise TypeError("historical storage conversion requires EvaluationRunManifest bytes")
    harness = loaded.metadata.get("matrix_harness")
    inline = harness.get("regeneration_spec") if isinstance(harness, Mapping) else None
    if not isinstance(inline, Mapping):
        raise ValueError("historical evaluation manifest has no inline regeneration_spec")
    _validate_historical_regeneration_artifact(
        loaded,
        source_path=source,
        regeneration=RegenerationSpec.model_validate(inline),
    )
    source_digest = sha256_bytes(source_bytes)
    migration = ArtifactMigrationRecord(
        migration_id="evaluation-run-manifest-v1-to-v2-shared-content",
        source_schema_version=source_version,
        target_schema_version=EVALUATION_RUN_MANIFEST_SCHEMA_VERSION,
        applied_at=loaded.created_at,
        metadata={
            "source_sha256": source_digest,
            "source_size_bytes": len(source_bytes),
        },
    )
    normalized = externalize_evaluation_manifest_storage(loaded, root=destination_root)
    normalized = normalized.model_copy(
        update={"migration_records": [*normalized.migration_records, migration]}
    )
    from feedbax.contracts.manifest import canonical_manifest_path

    target_path = canonical_manifest_path(normalized.kind, normalized.id, root=destination_root)
    expected_target_bytes = normalized.model_dump_json(exclude_none=True).encode() + b"\n"
    if target_path.exists():
        existing = target_path.read_bytes()
        if existing != expected_target_bytes:
            raise ValueError(
                "historical manifest versions with one logical id require separate explicit "
                f"custody domains: id={normalized.id!r}, existing_sha256={sha256_bytes(existing)}, "
                f"proposed_sha256={sha256_bytes(expected_target_bytes)}"
            )
        target_bytes = existing
    else:
        target_path = write_manifest(normalized, root=destination_root, index=False)
        target_bytes = target_path.read_bytes()
    return EvaluationManifestStorageConversion(
        manifest_id=normalized.id,
        source_schema_version=source_version,
        target_schema_version=normalized.schema_version,
        source_sha256=source_digest,
        source_size_bytes=len(source_bytes),
        target_sha256=sha256_bytes(target_bytes),
        target_size_bytes=len(target_bytes),
        target_path=target_path,
    )


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


@dataclass(frozen=True)
class _ExecutionPolicy:
    content_sha256: str
    per_row_max_rows: int
    threshold_source: str
    override_flag: str


def _read_json_object(path: str, *, description: str) -> dict[str, Any]:
    try:
        payload = strict_json_loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, StrictJsonError) as exc:
        raise ValueError(f"could not load {description} {path!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} {path!r} must contain a JSON object")
    return payload


def _load_execution_policy(path: str) -> _ExecutionPolicy:
    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"could not load execution policy {path!r}: {exc}") from exc
    try:
        payload = strict_json_loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError, StrictJsonError) as exc:
        raise ValueError(f"could not load execution policy {path!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"execution policy {path!r} must contain a JSON object")
    per_row_max_rows = payload.get("per_row_max_rows")
    threshold_source = payload.get("threshold_source")
    override_flag = payload.get("override_flag")
    if type(per_row_max_rows) is not int or per_row_max_rows <= 0:
        raise ValueError("execution policy per_row_max_rows must be a positive integer")
    if not isinstance(threshold_source, str) or not threshold_source.strip():
        raise ValueError("execution policy threshold_source must be a non-blank string")
    if not isinstance(override_flag, str) or not override_flag.strip():
        raise ValueError("execution policy override_flag must be a non-blank string")
    if override_flag != SUPPORTED_EXECUTION_POLICY_OVERRIDE_FLAG:
        raise ValueError(
            f"execution policy override_flag {override_flag!r} does not match the supported "
            f"override surface {SUPPORTED_EXECUTION_POLICY_OVERRIDE_FLAG!r}"
        )
    return _ExecutionPolicy(
        content_sha256=sha256_bytes(content),
        per_row_max_rows=per_row_max_rows,
        threshold_source=threshold_source,
        override_flag=override_flag,
    )


_ROW_CONTEXT_ROLES = frozenset(
    ("resolved_row_spec", SHARED_REGENERATION_ROLE, ROW_REGENERATION_ROLE)
)


@dataclass(frozen=True, slots=True)
class _MatrixRowPublication:
    """The harness-owned portion of a canonical row receipt."""

    metadata: dict[str, Any]
    artifacts: tuple[ArtifactRef, ...]

    @classmethod
    def from_manifest(cls, manifest: EvaluationRunManifest) -> _MatrixRowPublication | None:
        metadata = manifest.metadata.get("matrix_harness")
        if not isinstance(metadata, dict) or not {"row_id", "source", "custody"} <= metadata.keys():
            return None
        return cls(
            metadata=metadata,
            artifacts=tuple(a for a in manifest.artifacts if a.role in _ROW_CONTEXT_ROLES),
        )

    def ensure_row_custody(self, row_root: Path, *, matrix_root: Path) -> None:
        """Keep references verifiable from both the matrix and its standalone row."""
        if not self.artifacts or row_root == matrix_root:
            return
        row_store = ImmutableArtifactBlobProvider(row_root.absolute())
        matrix_store = ImmutableArtifactBlobProvider(matrix_root.absolute())
        for artifact in self.artifacts:
            ref = ContentObjectRef(
                object_id=f"{CONTENT_OBJECT_REF_PREFIX}{artifact.sha256}",
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
            )
            try:
                row_store.get_object(ref)
            except FileNotFoundError:
                row_store.store_object(matrix_store.get_object(ref))

    def attach(self, manifest: EvaluationRunManifest) -> EvaluationRunManifest:
        artifacts = [a for a in manifest.artifacts if a.role not in _ROW_CONTEXT_ROLES]
        return manifest.model_copy(
            update={
                "metadata": {**manifest.metadata, "matrix_harness": self.metadata},
                "artifacts": [*artifacts, *self.artifacts],
            }
        )


class MatrixMaterializerHarness:
    """Own row expansion, execution, custody, manifests, replay data, and notes."""

    def __init__(
        self,
        *,
        root: Path | str,
        custody: CustodyMode = "content-addressed",
        shared_store: bool = False,
    ):
        if custody not in ("manifest", "content-addressed"):
            raise ValueError(
                "unknown output custody route; registered routes are: content-addressed, manifest"
            )
        self.root = Path(root)
        self.custody = custody
        # Shared-store mode writes every row into one physical store at ``root``
        # (one manifest tree, one cache, one content-addressed artifact store)
        # instead of a self-contained per-row ``root/row_id`` store, and defers
        # index construction to the caller so the batch owner can build a single
        # store-wide index. Rows stay logically distinct through their globally
        # unique manifest identifiers.
        self.shared_store = shared_store

    def materialize(
        self,
        rows: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        execute: Callable[[str, Mapping[str, Any], Path], tuple[Any, Path | None]] | None = None,
        publish_row: Callable[
            [Mapping[str, Any], Path, _MatrixRowPublication], tuple[EvaluationRunManifest, Path]
        ]
        | None = None,
        command: Sequence[str],
        title: str,
        source: str,
        escape_hatch_reason: str | None = None,
        matrix_metadata: Mapping[str, Any] | None = None,
        regeneration_parameters: Mapping[str, Any] | None = None,
    ) -> HarnessResult:
        """Materialize rows through a publisher or decorate a legacy executor's result.

        A row publisher owns the canonical receipt and includes the supplied context
        in that publication. The harness only writes receipts for legacy executors.
        """
        if (execute is None) == (publish_row is None):
            raise ValueError("materializer requires exactly one row executor or publisher")
        if not rows:
            raise ValueError("materializer requires at least one row or condition")
        if escape_hatch_reason is not None and not escape_hatch_reason.strip():
            raise ValueError("flat-spec escape hatch requires a stated non-empty reason")
        materialized: list[MaterializedRow] = []
        shared_metadata = dict(matrix_metadata or {})
        replay_parameters = dict(regeneration_parameters or {})
        for row_id, _resolved in rows:
            if not row_id:
                raise ValueError("materializer row_id must be non-empty")
        shared_context = (
            publish_shared_regeneration_context(
                root=self.root,
                command=command,
                source=source,
                custody=self.custody,
                escape_hatch_reason=escape_hatch_reason,
                matrix_metadata=shared_metadata,
                regeneration_parameters=replay_parameters,
            )
            if self.custody == "content-addressed"
            else None
        )
        for row_id, resolved in rows:
            row_root = self.root if self.shared_store else self.root / row_id
            row_root.mkdir(parents=True, exist_ok=True)
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
                    **(
                        {"escape_hatch_reason": escape_hatch_reason}
                        if escape_hatch_reason is not None
                        else {}
                    ),
                    **shared_metadata,
                },
            )
            artifacts: list[ArtifactRef] = []
            if self.custody == "content-addressed":
                assert shared_context is not None
                artifacts.append(
                    store_json_artifact(
                        dict(resolved),
                        root=self.root,
                        role="resolved_row_spec",
                        logical_name=f"{row_id}.resolved.json",
                        metadata={"row_id": row_id, "source": source},
                    )
                )
                _row_regeneration, row_regeneration_ref = publish_row_regeneration_spec(
                    root=self.root,
                    command=command,
                    row_id=row_id,
                    resolved=resolved,
                    shared_context=shared_context,
                )
                artifacts.extend((shared_context, row_regeneration_ref))
            harness_metadata = {
                "row_id": row_id,
                "source": source,
                "custody": self.custody,
                **(
                    {"escape_hatch_reason": escape_hatch_reason}
                    if escape_hatch_reason is not None
                    else {}
                ),
            }
            if self.custody == "manifest":
                harness_metadata.update(shared_metadata)
                harness_metadata["regeneration_spec"] = regeneration.model_dump(
                    mode="json", exclude_none=True
                )
            row_publication = _MatrixRowPublication(harness_metadata, tuple(artifacts))
            if publish_row is not None:
                row_publication.ensure_row_custody(row_root, matrix_root=self.root)
                result, manifest_path = publish_row(resolved, row_root, row_publication)
            else:
                assert execute is not None
                result, manifest_path = execute(row_id, resolved, row_root)
                if (
                    manifest_path is not None
                    and hasattr(result, "metadata")
                    and hasattr(result, "artifacts")
                ):
                    result.metadata.setdefault("matrix_harness", {}).update(
                        row_publication.metadata
                    )
                    known_artifact_ids = {artifact.artifact_id for artifact in result.artifacts}
                    result.artifacts.extend(
                        artifact
                        for artifact in row_publication.artifacts
                        if artifact.artifact_id not in known_artifact_ids
                    )
                    manifest_path = write_manifest(
                        result, root=row_root, index=not self.shared_store
                    )
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
        return HarnessResult(
            tuple(materialized),
            note,
            escape_hatch_reason,
            metadata={
                **shared_metadata,
                "regeneration_parameters": replay_parameters,
            },
        )


def main(
    argv: Sequence[str] | None = None, *, bootstrap_state: BootstrapState | None = None
) -> int:
    """Materialize a serialized evaluation matrix through the standard harness."""
    parser = argparse.ArgumentParser(prog="feedbax matrix-harness")
    parser.add_argument(
        "spec", help="EvaluationRunMatrixSpec or EvaluationRunMatrixDeltaSpec JSON path"
    )
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--repo-root")
    parser.add_argument("--plugin", action="append")
    parser.add_argument("--escape-hatch-reason")
    parser.add_argument("--parent-manifest-root")
    parser.add_argument("--execution-descriptor")
    parser.add_argument("--artifact-provider", action="append", default=[])
    parser.add_argument("--checkpoint-custody", action="append", default=[])
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Execute the whole matrix through its registered batch recipe; this bypasses "
            "the execution policy's per-row row-count refusal."
        ),
    )
    parser.add_argument(
        "--locked-spec",
        help=(
            "Require the runtime matrix's exact ordered row_id sequence to equal the locked "
            "evaluation matrix sequence; the locked document may be either authoring kind."
        ),
    )
    parser.add_argument(
        "--execution-policy",
        help=(
            "JSON policy requiring positive-integer per_row_max_rows, non-blank "
            "threshold_source, and override_flag set to --allow-large-per-row."
        ),
    )
    parser.add_argument(
        "--allow-large-per-row",
        action="store_true",
        help=(
            "Explicitly bypass an execution policy's oversized per-row refusal; the decision "
            "is recorded in matrix-harness evidence."
        ),
    )
    args = parser.parse_args(argv)
    if bootstrap_state is None:
        from feedbax.plugins.composition import compose_application

        bootstrap_state = asyncio.run(compose_application(modules=tuple(args.plugin or ())))
    from feedbax.analysis.evaluation import (
        EvaluationBatchExecution,
        execute_evaluation_run_matrix,
        materialize_evaluation_run_matrix,
    )
    from feedbax.analysis.execution_context import (
        StagedArtifactProviderRootBinding,
        StagedCheckpointCustodyRootBinding,
    )
    def binding_parts(value: str, *, option: str) -> tuple[str, str]:
        name, separator, root = value.partition("=")
        if not separator or not name or not root:
            raise ValueError(f"{option} entries must use NAME=ROOT")
        return name, root

    try:
        payload = _read_json_object(args.spec, description="evaluation matrix spec")
        policy = (
            _load_execution_policy(args.execution_policy)
            if args.execution_policy is not None
            else None
        )
        runtime_rows = None
        if args.locked_spec is not None:
            locked_document = _read_json_object(
                args.locked_spec,
                description="locked evaluation matrix spec",
            )
            locked_payload = locked_document.get("evaluation_matrix", locked_document)
            locked_rows = materialize_evaluation_run_matrix(
                locked_payload,
                registry=bootstrap_state.bundle.evaluation_recipes,
                repo_root=args.repo_root,
            )
            runtime_rows = materialize_evaluation_run_matrix(
                payload,
                registry=bootstrap_state.bundle.evaluation_recipes,
                repo_root=args.repo_root,
            )
            locked_row_ids = tuple(row.row_id for row in locked_rows)
            runtime_row_ids = tuple(row.row_id for row in runtime_rows)
            if runtime_row_ids != locked_row_ids:
                raise ValueError(
                    "runtime evaluation matrix row_id sequence does not match locked spec: "
                    f"runtime={runtime_row_ids!r}, locked={locked_row_ids!r}"
                )
        if policy is not None:
            if runtime_rows is None:
                runtime_rows = materialize_evaluation_run_matrix(
                    payload,
                    registry=bootstrap_state.bundle.evaluation_recipes,
                    repo_root=args.repo_root,
                )
            row_count = len(runtime_rows)
            if (
                not args.batch
                and not args.allow_large_per_row
                and row_count > policy.per_row_max_rows
            ):
                raise ValueError(
                    f"per-row execution refused for {row_count} rows; authored threshold "
                    f"is {policy.per_row_max_rows} rows from {policy.threshold_source!r}; "
                    f"pass authored override {policy.override_flag!r} to proceed"
                )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    policy_metadata = (
        {
            "execution_policy": {
                "content_sha256": policy.content_sha256,
                "per_row_max_rows": policy.per_row_max_rows,
                "threshold_source": policy.threshold_source,
                "override_flag": policy.override_flag,
                "execution_mode": "batch" if args.batch else "per-row",
                "explicit_override_used": args.allow_large_per_row and not args.batch,
            }
        }
        if policy is not None
        else None
    )
    execution_descriptor = (
        strict_json_loads(Path(args.execution_descriptor).read_text(encoding="utf-8"))
        if args.execution_descriptor is not None
        else None
    )
    execute_evaluation_run_matrix(
        payload,
        registry=bootstrap_state.bundle.evaluation_recipes,
        root=args.manifest_root,
        repo_root=args.repo_root,
        escape_hatch_reason=args.escape_hatch_reason,
        parent_manifest_root=args.parent_manifest_root,
        execution_descriptor=execution_descriptor,
        artifact_provider_bindings=[
            StagedArtifactProviderRootBinding(*binding_parts(value, option="--artifact-provider"))
            for value in args.artifact_provider
        ],
        checkpoint_custody_bindings=[
            StagedCheckpointCustodyRootBinding(*binding_parts(value, option="--checkpoint-custody"))
            for value in args.checkpoint_custody
        ],
        batch=EvaluationBatchExecution() if args.batch else None,
        matrix_metadata=policy_metadata,
    )
    return 0


__all__ = [
    "HarnessResult",
    "MaterializedRow",
    "MatrixMaterializerHarness",
]
