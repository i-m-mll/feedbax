"""Static TrainingRunManifest spec-payload preflight helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from feedbax.contracts.manifest import (
    ArtifactMigrationRecord,
    SpecPayload,
    TrainingRunManifest,
    canonical_json_bytes,
    normalize_manifest_spec_payloads,
    sha256_bytes,
    spec_payload,
    training_run_manifest_id,
)
from feedbax.contracts.training import TrainingRunSpec


class TrainingRunManifestPreflightError(ValueError):
    """Raised when final TrainingRunManifest payload normalization would fail."""


@dataclass(frozen=True)
class TrainingRunManifestSpecPayloads:
    """Normalized spec-payload fields embedded in a TrainingRunManifest."""

    graph_spec: SpecPayload | None
    training_spec: SpecPayload
    task_spec: SpecPayload
    task_binding_spec: SpecPayload | None

    def model_dump(self) -> dict[str, Any]:
        """Return JSON-serializable normalized payloads."""
        return {
            key: value.model_dump(mode="json", exclude_none=True)
            for key, value in {
                "graph_spec": self.graph_spec,
                "training_spec": self.training_spec,
                "task_spec": self.task_spec,
                "task_binding_spec": self.task_binding_spec,
            }.items()
            if value is not None
        }


def resolve_training_spec_payload_ref(
    explicit_ref: str | None,
    *,
    authoritative_ref: str | None,
) -> str | None:
    """Resolve the embedded training-spec ref without permitting custody drift."""
    if explicit_ref is not None and not explicit_ref:
        raise ValueError("/training_spec_payload_ref must not be empty")
    if authoritative_ref is not None and not authoritative_ref:
        raise ValueError("authoritative training payload ref must not be empty")
    if (
        explicit_ref is not None
        and authoritative_ref is not None
        and explicit_ref != authoritative_ref
    ):
        raise ValueError(
            "/training_spec_payload_ref disagrees with authoritative execution payload; "
            f"explicit={explicit_ref!r}, authoritative={authoritative_ref!r}"
        )
    return authoritative_ref or explicit_ref


def validate_training_run_spec(spec: TrainingRunSpec | Mapping[str, Any]) -> TrainingRunSpec:
    """Accept or migrate a TrainingRunSpec mapping for manifest preflight."""
    return _validate_training_run_spec_with_evidence(spec)[0]


def _validate_training_run_spec_with_evidence(
    spec: TrainingRunSpec | Mapping[str, Any],
) -> tuple[TrainingRunSpec, list[ArtifactMigrationRecord]]:
    if isinstance(spec, TrainingRunSpec):
        return spec, []

    from feedbax.contracts.migrations import migrate_structured_spec_payload

    migrated = migrate_structured_spec_payload(
        "TrainingRunSpec",
        spec,
        path="training_run_spec",
    )
    return TrainingRunSpec.model_validate(migrated.payload), migrated.migration_records


def build_training_run_manifest_spec_payloads(
    spec: TrainingRunSpec,
    *,
    training_spec_payload: Mapping[str, Any] | None = None,
    training_spec_payload_kind: str = "TrainingRunSpec",
    training_spec_payload_schema_id: str | None = None,
    training_spec_payload_schema_version: str | None = None,
    training_spec_payload_ref: str | None = None,
    task_binding_spec: Mapping[str, Any] | None = None,
    migration_records: list[ArtifactMigrationRecord] | None = None,
) -> TrainingRunManifestSpecPayloads:
    """Build the exact spec payload fields embedded in a TrainingRunManifest."""
    graph_inline = dict(spec.graph.inline) if spec.graph.inline is not None else None
    normalized = _normalize_payload_fields(
        graph_spec=(
            spec_payload("GraphSpec", graph_inline)
            if graph_inline is not None
            else None
        ),
        training_spec=_training_spec_payload(
            training_spec_payload_kind,
            dict(training_spec_payload or spec.model_dump(mode="json")),
            schema_id=training_spec_payload_schema_id,
            schema_version=training_spec_payload_schema_version,
            ref=training_spec_payload_ref,
        ).model_copy(update={"migration_records": migration_records or []}),
        task_spec=spec_payload("TaskSpec", spec.task.model_dump(mode="json")),
        task_binding_spec=(
            spec_payload("StudioTaskBindingSpec", dict(task_binding_spec))
            if task_binding_spec is not None
            else None
        ),
    )
    return normalized


def preflight_training_run_manifest_payloads(
    spec: TrainingRunSpec | Mapping[str, Any],
    *,
    training_spec_payload: Mapping[str, Any] | None = None,
    training_spec_payload_kind: str = "TrainingRunSpec",
    training_spec_payload_schema_id: str | None = None,
    training_spec_payload_schema_version: str | None = None,
    training_spec_payload_ref: str | None = None,
    authoritative_training_spec_payload_ref: str | None = None,
    task_binding_spec: Mapping[str, Any] | None = None,
    row_id: str | None = None,
    spec_path: str | Path | None = None,
) -> TrainingRunManifestSpecPayloads:
    """Validate final TrainingRunManifest spec payloads without running training."""
    try:
        run_spec, migration_records = _validate_training_run_spec_with_evidence(spec)
        effective_ref = resolve_training_spec_payload_ref(
            training_spec_payload_ref,
            authoritative_ref=authoritative_training_spec_payload_ref,
        )
        return build_training_run_manifest_spec_payloads(
            run_spec,
            training_spec_payload=training_spec_payload,
            training_spec_payload_kind=training_spec_payload_kind,
            training_spec_payload_schema_id=training_spec_payload_schema_id,
            training_spec_payload_schema_version=training_spec_payload_schema_version,
            training_spec_payload_ref=effective_ref,
            task_binding_spec=task_binding_spec,
            migration_records=migration_records,
        )
    except Exception as exc:
        raise TrainingRunManifestPreflightError(
            _preflight_error_message(exc, row_id=row_id, spec_path=spec_path)
        ) from exc


def _normalize_payload_fields(
    *,
    graph_spec: SpecPayload | None,
    training_spec: SpecPayload,
    task_spec: SpecPayload,
    task_binding_spec: SpecPayload | None,
) -> TrainingRunManifestSpecPayloads:
    manifest = TrainingRunManifest(
        id=training_run_manifest_id("preflight"),
        job_id="preflight",
        graph_spec=graph_spec,
        training_spec=training_spec,
        task_spec=task_spec,
        task_binding_spec=task_binding_spec,
    )
    normalized = normalize_manifest_spec_payloads(manifest)
    return TrainingRunManifestSpecPayloads(
        graph_spec=normalized.graph_spec,
        training_spec=normalized.training_spec,
        task_spec=normalized.task_spec,
        task_binding_spec=normalized.task_binding_spec,
    )


def _training_spec_payload(
    kind: str,
    inline: dict[str, Any],
    *,
    schema_id: str | None,
    schema_version: str | None,
    ref: str | None,
) -> SpecPayload:
    if kind == "TrainingRunSpec":
        return spec_payload(kind, inline, ref=ref)
    if schema_id is None or schema_version is None:
        raise ValueError(
            "/training_spec_payload requires schema_id and schema_version for external kinds"
        )
    return SpecPayload(
        kind=kind,
        inline=inline,
        schema_id=schema_id,
        schema_version=schema_version,
        ref=ref,
        sha256=sha256_bytes(canonical_json_bytes(inline)),
        metadata={"external": True},
    )


def _preflight_error_message(
    exc: Exception,
    *,
    row_id: str | None,
    spec_path: str | Path | None,
) -> str:
    parts = ["TrainingRunManifest preflight failed"]
    if row_id:
        parts.append(f"row_id={row_id!r}")
    if spec_path is not None:
        parts.append(f"spec_path={str(spec_path)!r}")
    parts.append(str(exc))
    return ": ".join(parts)
