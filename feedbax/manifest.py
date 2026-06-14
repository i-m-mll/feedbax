"""Durable manifest and artifact-reference models for Feedbax runs.

The database remains useful as an index, but these models are the portable
records that describe specs, executions, lineage, and large output artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from feedbax.contracts.graph import AnalysisInputRequirement

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - Python 3.12 always has importlib.metadata.
    PackageNotFoundError = Exception  # type: ignore[assignment]
    version = None  # type: ignore[assignment]


SCHEMA_VERSION = "feedbax.manifest.v1"
PROVIDER_VERSION = "feedbax-provider.v1"
DEFAULT_MANIFEST_ROOT_ENV = "FEEDBAX_RUNS_DIR"

ManifestStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


def feedbax_version() -> str:
    """Return the installed Feedbax package version, or a useful local fallback."""
    if version is None:
        return "unknown"
    try:
        return version("feedbax")
    except PackageNotFoundError:
        return "unknown"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp with stable second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def default_manifest_root() -> Path:
    """Return the root directory for local manifests and artifacts."""
    configured = os.environ.get(DEFAULT_MANIFEST_ROOT_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / "feedbax_runs"


class StrictModel(BaseModel):
    """Base model for provider-contract records."""

    model_config = ConfigDict(extra="forbid")


class ArtifactRef(StrictModel):
    """Reference to a large output artifact stored outside a manifest."""

    role: str
    logical_name: str
    artifact_id: Optional[str] = None
    sha256: Optional[str] = None
    media_type: str = "application/octet-stream"
    size_bytes: Optional[int] = None
    storage_backend: str = "feedbax-local"
    uri: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArrayStoreRef(StrictModel):
    """Reference to a role-addressed parameter/state array store."""

    role: Literal["params", "state", "optimizer", "history"]
    schema_version: str
    storage_backend: str
    logical_name: str
    artifact_id: Optional[str] = None
    sha256: Optional[str] = None
    uri: Optional[str] = None
    array_count: int
    roles: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactValidationRecord(StrictModel):
    """Validation outcome for a durable artifact or migration step."""

    name: str
    status: Literal["passed", "failed", "warning"]
    checked_at: datetime = Field(default_factory=utc_now)
    schema_version: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class ArtifactMigrationRecord(StrictModel):
    """Provenance for a schema-to-schema artifact migration."""

    migration_id: str
    source_schema_version: str
    target_schema_version: str
    applied_at: datetime = Field(default_factory=utc_now)
    tool: str = "feedbax"
    deterministic: bool = True
    validation: list[ArtifactValidationRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntrypointRef(StrictModel):
    """How a manifest-producing operation was invoked."""

    kind: str
    command: Optional[str] = None
    name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParentRef(StrictModel):
    """Reference to an input spec, parent manifest, or parent artifact."""

    kind: str
    id: str
    role: Optional[str] = None
    uri: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Provenance(StrictModel):
    """Shared provenance fields recorded on durable manifests."""

    source_repo: Optional[str] = None
    source_branch: Optional[str] = None
    source_commit: Optional[str] = None
    dirty: Optional[bool] = None
    entrypoint: Optional[EntrypointRef] = None
    issues: list[str] = Field(default_factory=list)
    parents: list[ParentRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpecPayload(StrictModel):
    """Inline spec payload plus optional stable reference metadata."""

    kind: str
    inline: dict[str, Any]
    schema_id: Optional[str] = None
    schema_version: Optional[str] = None
    ref: Optional[str] = None
    sha256: Optional[str] = None
    source_sha256: Optional[str] = None
    migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OverridePatch(StrictModel):
    """Machine-readable override applied relative to a base spec."""

    path: str
    value: Any
    op: Literal["add", "replace", "remove"] = "replace"


class BaseManifest(StrictModel):
    """Common manifest fields."""

    kind: str
    schema_version: str = SCHEMA_VERSION
    id: str
    created_at: datetime = Field(default_factory=utc_now)
    feedbax_version: str = Field(default_factory=feedbax_version)
    provider_version: str = PROVIDER_VERSION
    status: Optional[ManifestStatus] = None
    provenance: Provenance = Field(default_factory=Provenance)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphSpecManifest(BaseManifest):
    kind: Literal["GraphSpecManifest"] = "GraphSpecManifest"
    graph_spec: SpecPayload
    migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)


class ModelArtifactManifest(BaseManifest):
    """Manifest binding a graph spec to role-addressed params/state stores."""

    kind: Literal["ModelArtifactManifest"] = "ModelArtifactManifest"
    graph_spec: ParentRef | SpecPayload
    parameter_store: Optional[ArrayStoreRef] = None
    state_store: Optional[ArrayStoreRef] = None
    optimizer_store: Optional[ArrayStoreRef] = None
    validation_records: list[ArtifactValidationRecord] = Field(default_factory=list)
    migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)


class TrainingRunSetManifest(BaseManifest):
    kind: Literal["TrainingRunSetManifest"] = "TrainingRunSetManifest"
    name: str
    run_ids: list[str] = Field(default_factory=list)
    graph_spec: Optional[ParentRef | SpecPayload] = None
    tags: list[str] = Field(default_factory=list)


class TrainingRunManifest(BaseManifest):
    kind: Literal["TrainingRunManifest"] = "TrainingRunManifest"
    run_set_id: Optional[str] = None
    job_id: Optional[str] = None
    graph_spec: Optional[SpecPayload | ParentRef] = None
    training_spec: Optional[SpecPayload] = None
    task_spec: Optional[SpecPayload] = None
    task_binding_spec: Optional[SpecPayload] = None
    overrides: list[OverridePatch] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    summary_metrics: dict[str, Any] = Field(default_factory=dict)


class EvaluationRunSpec(StrictModel):
    """Declarative request for an evaluation run."""

    evaluation_type: str
    training_run_ids: list[str] = Field(default_factory=list)
    inputs: list[ParentRef] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluationRunManifest(BaseManifest):
    kind: Literal["EvaluationRunManifest"] = "EvaluationRunManifest"
    evaluation_spec: SpecPayload
    input_training_runs: list[ParentRef] = Field(default_factory=list)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)


class AnalysisRunSpec(StrictModel):
    """Declarative request for an analysis run."""

    analysis_type: str
    inputs: list[ParentRef] = Field(default_factory=list)
    input_requirements: list[AnalysisInputRequirement] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class AnalysisRunManifest(BaseManifest):
    kind: Literal["AnalysisRunManifest"] = "AnalysisRunManifest"
    analysis_spec: SpecPayload
    inputs: list[ParentRef] = Field(default_factory=list)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)


class ReportSpec(StrictModel):
    """Declarative request for a report product."""

    report_type: str
    inputs: list[ParentRef] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    narrative: Optional[str] = None


class ReportManifest(BaseManifest):
    kind: Literal["ReportManifest"] = "ReportManifest"
    report_spec: SpecPayload
    inputs: list[ParentRef] = Field(default_factory=list)


AnyManifest = (
    GraphSpecManifest
    | ModelArtifactManifest
    | TrainingRunSetManifest
    | TrainingRunManifest
    | EvaluationRunManifest
    | AnalysisRunManifest
    | ReportManifest
)

class GraphSpecLoadResult(StrictModel):
    """Migrated GraphSpec payload plus the manifest that owns its migration records."""

    payload: dict[str, Any]
    manifest: GraphSpecManifest | ModelArtifactManifest
    custody_manifest_kind: Literal["GraphSpecManifest", "ModelArtifactManifest"]
    custody_manifest_id: str
    applied_migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)
    migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)
    downstream_migration_records: list[ArtifactMigrationRecord] = Field(default_factory=list)

MANIFEST_MODELS: dict[str, type[BaseManifest]] = {
    "GraphSpecManifest": GraphSpecManifest,
    "ModelArtifactManifest": ModelArtifactManifest,
    "TrainingRunSetManifest": TrainingRunSetManifest,
    "TrainingRunManifest": TrainingRunManifest,
    "EvaluationRunManifest": EvaluationRunManifest,
    "AnalysisRunManifest": AnalysisRunManifest,
    "ReportManifest": ReportManifest,
}

SPEC_PAYLOAD_FIELDS_BY_MANIFEST_KIND: dict[str, tuple[str, ...]] = {
    "GraphSpecManifest": ("graph_spec",),
    "ModelArtifactManifest": ("graph_spec",),
    "TrainingRunSetManifest": ("graph_spec",),
    "TrainingRunManifest": (
        "graph_spec",
        "training_spec",
        "task_spec",
        "task_binding_spec",
    ),
    "EvaluationRunManifest": ("evaluation_spec",),
    "AnalysisRunManifest": ("analysis_spec",),
    "ReportManifest": ("report_spec",),
}


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value using stable JSON for hashing."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spec_payload_record_metadata(
    record: ArtifactMigrationRecord,
    *,
    kind: str,
    path: str,
) -> ArtifactMigrationRecord:
    metadata = {
        **record.metadata,
        "spec_payload_kind": kind,
        "spec_payload_path": path,
    }
    return record.model_copy(update={"metadata": metadata})


def _ensure_spec_payload_hash(payload: SpecPayload, *, path: str) -> str:
    inline_sha256 = sha256_bytes(canonical_json_bytes(payload.inline))
    if payload.sha256 is not None and payload.sha256 != inline_sha256:
        raise ValueError(
            "Embedded SpecPayload sha256 does not match canonical inline payload: "
            f"path={path!r}, kind={payload.kind!r}, sha256={payload.sha256!r}, "
            f"computed_sha256={inline_sha256!r}"
        )
    return inline_sha256


def migrate_spec_payload(
    payload: SpecPayload | dict[str, Any],
    *,
    path: str = "spec",
    registry: Any | None = None,
) -> SpecPayload:
    """Accept or migrate one manifest-embedded structured spec payload.

    ``sha256`` is the content hash of the canonical, post-migration inline
    payload. If migration changes the inline payload, ``source_sha256`` retains
    the original inline hash for provenance.
    """
    from feedbax.contracts.graph import GRAPH_SPEC_SCHEMA_ID
    from feedbax.migrations import (
        UnknownSpecFamily,
        UnsupportedSpecVersion,
        default_spec_registry,
        migrate_graph_spec,
    )

    spec_payload_obj = (
        payload if isinstance(payload, SpecPayload) else SpecPayload.model_validate(payload)
    )
    active_registry = registry or default_spec_registry
    try:
        family = active_registry.resolve(spec_payload_obj.kind)
    except UnknownSpecFamily as exc:
        raise UnknownSpecFamily(
            "Unknown embedded SpecPayload family: "
            f"path={path!r}, kind={spec_payload_obj.kind!r}; {exc}"
        ) from exc
    expected_schema_id = family.identity
    if spec_payload_obj.schema_id is not None and spec_payload_obj.schema_id != expected_schema_id:
        raise UnsupportedSpecVersion(
            "Unsupported embedded SpecPayload schema identity: "
            f"path={path!r}, kind={spec_payload_obj.kind!r}, "
            f"schema_id={spec_payload_obj.schema_id!r}, expected={expected_schema_id!r}"
        )

    source_sha256 = _ensure_spec_payload_hash(spec_payload_obj, path=path)
    source_version = spec_payload_obj.schema_version
    inline_schema_version = spec_payload_obj.inline.get("schema_version")
    if (
        source_version is not None
        and isinstance(inline_schema_version, str)
        and inline_schema_version
        and inline_schema_version != source_version
    ):
        raise UnsupportedSpecVersion(
            "Embedded SpecPayload schema version disagrees with inline payload: "
            f"path={path!r}, kind={spec_payload_obj.kind!r}, "
            f"schema_version={source_version!r}, "
            f"inline_schema_version={inline_schema_version!r}"
        )

    try:
        if spec_payload_obj.kind == "GraphSpec" and expected_schema_id == GRAPH_SPEC_SCHEMA_ID:
            result = migrate_graph_spec(
                spec_payload_obj.inline,
                source_version=source_version,
                path=path,
                registry=active_registry,
            )
        else:
            result = active_registry.migrate(
                spec_payload_obj.kind,
                spec_payload_obj.inline,
                source_version=source_version,
            )
    except UnsupportedSpecVersion as exc:
        raise UnsupportedSpecVersion(
            "Unsupported embedded SpecPayload version: "
            f"path={path!r}, kind={spec_payload_obj.kind!r}; {exc}"
        ) from exc

    migrated_sha256 = sha256_bytes(canonical_json_bytes(result.payload))
    migration_records = [
        *spec_payload_obj.migration_records,
        *(
            _spec_payload_record_metadata(record, kind=spec_payload_obj.kind, path=path)
            for record in result.migration_records
        ),
    ]
    update: dict[str, Any] = {
        "inline": result.payload,
        "schema_id": result.schema_id,
        "schema_version": result.target_version,
        "sha256": migrated_sha256,
        "migration_records": migration_records,
    }
    if source_sha256 != migrated_sha256:
        update["source_sha256"] = spec_payload_obj.source_sha256 or source_sha256
    elif spec_payload_obj.source_sha256 is not None:
        update["source_sha256"] = spec_payload_obj.source_sha256
    return spec_payload_obj.model_copy(update=update)


def spec_payload(kind: str, inline: dict[str, Any], ref: Optional[str] = None) -> SpecPayload:
    """Build a registry-stamped spec payload hashed after inline migration."""
    payload = SpecPayload(kind=kind, inline=inline, ref=ref)
    return migrate_spec_payload(payload, path=kind)


def collect_git_provenance(cwd: Path | str | None = None) -> Provenance:
    """Collect best-effort local Git provenance without mutating repository state."""
    repo_cwd = Path(cwd) if cwd is not None else Path.cwd()

    def _git(*args: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=repo_cwd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return proc.stdout.strip() or None

    status = _git("status", "--porcelain")
    return Provenance(
        source_repo=_git("config", "--get", "remote.origin.url"),
        source_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        source_commit=_git("rev-parse", "HEAD"),
        dirty=(bool(status) if status is not None else None),
    )


def _artifact_path(root: Path, digest: str, suffix: str = "") -> Path:
    return root / "artifacts" / "sha256" / digest[:2] / f"{digest}{suffix}"


def store_artifact(
    source_path: Path | str,
    *,
    root: Path | str | None = None,
    role: str,
    logical_name: Optional[str] = None,
    media_type: str = "application/octet-stream",
    metadata: Optional[dict[str, Any]] = None,
) -> ArtifactRef:
    """Copy an artifact into the local content-addressed store and return its ref."""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)
    root_path = Path(root) if root is not None else default_manifest_root()
    digest = sha256_file(source)
    dest = _artifact_path(root_path, digest, source.suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(source, dest)
    stat = dest.stat()
    artifact_metadata = dict(metadata or {})
    artifact_metadata.setdefault("original_uri", str(source))
    artifact_metadata.setdefault("relative_path", str(dest.relative_to(root_path)))
    return ArtifactRef(
        role=role,
        logical_name=logical_name or source.name,
        artifact_id=f"artifact://sha256/{digest}",
        sha256=digest,
        media_type=media_type,
        size_bytes=stat.st_size,
        uri=str(dest),
        metadata=artifact_metadata,
    )


def store_json_artifact(
    value: Any,
    *,
    root: Path | str | None = None,
    role: str,
    logical_name: str,
    metadata: Optional[dict[str, Any]] = None,
) -> ArtifactRef:
    """Write stable JSON into the local content-addressed store."""
    root_path = Path(root) if root is not None else default_manifest_root()
    data = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    digest = sha256_bytes(data)
    dest = _artifact_path(root_path, digest, ".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)
    stat = dest.stat()
    artifact_metadata = dict(metadata or {})
    artifact_metadata.setdefault("relative_path", str(dest.relative_to(root_path)))
    return ArtifactRef(
        role=role,
        logical_name=logical_name,
        artifact_id=f"artifact://sha256/{digest}",
        sha256=digest,
        media_type="application/json",
        size_bytes=stat.st_size,
        uri=str(dest),
        metadata=artifact_metadata,
    )


def _manifest_dir(root: Path, kind: str) -> Path:
    names = {
        "GraphSpecManifest": "graph_specs",
        "ModelArtifactManifest": "model_artifacts",
        "TrainingRunSetManifest": "training_run_sets",
        "TrainingRunManifest": "training_runs",
        "EvaluationRunManifest": "evaluation_runs",
        "AnalysisRunManifest": "analysis_runs",
        "ReportManifest": "reports",
    }
    return root / "manifests" / names.get(kind, kind)


def _safe_manifest_filename(manifest_id: str) -> str:
    safe = manifest_id.replace(":", "_").replace("/", "_")
    return f"{safe}.json"


def safe_manifest_key(manifest_id: str) -> str:
    """Return a filesystem-safe key derived from a manifest identifier."""
    return manifest_id.replace(":", "_").replace("/", "_")


def _is_spec_payload_data(value: Any) -> bool:
    return isinstance(value, dict) and "kind" in value and "inline" in value


def _normalize_spec_payload_field(value: Any, *, path: str) -> Any:
    if value is None:
        return None
    if isinstance(value, SpecPayload) or _is_spec_payload_data(value):
        return migrate_spec_payload(value, path=path)
    return value


def _normalize_manifest_data_spec_payloads(data: dict[str, Any]) -> dict[str, Any]:
    raw_kind = data.get("kind")
    if not isinstance(raw_kind, str):
        return data
    kind = raw_kind
    fields = SPEC_PAYLOAD_FIELDS_BY_MANIFEST_KIND.get(kind)
    if not fields:
        return data
    normalized = dict(data)
    for field_name in fields:
        if field_name not in normalized:
            continue
        normalized[field_name] = _normalize_spec_payload_field(
            normalized[field_name],
            path=field_name,
        )
    return normalized


def normalize_manifest_spec_payloads(manifest: AnyManifest) -> AnyManifest:
    """Return a copy with embedded spec payloads registry-stamped and migrated."""
    fields = SPEC_PAYLOAD_FIELDS_BY_MANIFEST_KIND.get(manifest.kind)
    if not fields:
        return manifest
    updates: dict[str, Any] = {}
    for field_name in fields:
        value = getattr(manifest, field_name, None)
        normalized = _normalize_spec_payload_field(value, path=field_name)
        if normalized is not value:
            updates[field_name] = normalized
    if not updates:
        return manifest
    return manifest.model_copy(update=updates)  # type: ignore[return-value]


def evaluation_run_manifest_id(spec: EvaluationRunSpec) -> str:
    """Return deterministic run identity for an evaluation spec."""
    digest = sha256_bytes(canonical_json_bytes(spec))
    return f"feedbax-evaluation-run:{digest[:32]}"


def analysis_run_manifest_id(spec: AnalysisRunSpec) -> str:
    """Return deterministic run identity for an analysis spec."""
    digest = sha256_bytes(canonical_json_bytes(spec))
    return f"feedbax-analysis-run:{digest[:32]}"


def evaluation_states_cache_path(
    manifest_id: str,
    *,
    root: Path | str | None = None,
) -> Path:
    """Return the manifest-root cache path for evaluated state trajectories."""
    root_path = Path(root) if root is not None else default_manifest_root()
    return root_path / "cache" / "states" / f"{safe_manifest_key(manifest_id)}.pkl"


def analysis_results_cache_dir(
    manifest_id: str,
    *,
    root: Path | str | None = None,
) -> Path:
    """Return the manifest-root cache directory for computed analysis results."""
    root_path = Path(root) if root is not None else default_manifest_root()
    return root_path / "cache" / "analysis_results" / safe_manifest_key(manifest_id)


def write_manifest(
    manifest: AnyManifest,
    *,
    root: Path | str | None = None,
    index: bool = True,
) -> Path:
    """Write a manifest to the local manifest layout and optionally index it."""
    manifest = normalize_manifest_spec_payloads(manifest)
    root_path = Path(root) if root is not None else default_manifest_root()
    manifest_dir = _manifest_dir(root_path, manifest.kind)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / _safe_manifest_filename(manifest.id)
    path.write_text(
        manifest.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )
    if index:
        from feedbax.manifest_index import index_manifest_file

        index_manifest_file(path, root=root_path)
    return path


def load_manifest(path: Path | str) -> AnyManifest:
    """Load a known Feedbax manifest from disk."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data = _normalize_manifest_data_spec_payloads(data)
    raw_kind = data.get("kind")
    if not isinstance(raw_kind, str):
        raise ValueError(f"Unknown Feedbax manifest kind: {raw_kind!r}")
    kind = raw_kind
    model = MANIFEST_MODELS.get(kind)
    if model is None:
        raise ValueError(f"Unknown Feedbax manifest kind: {kind!r}")
    return model.model_validate(data)  # type: ignore[return-value]


def load_graph_spec_from_manifest(
    manifest: GraphSpecManifest | ModelArtifactManifest | Path | str,
    *,
    root: Path | str | None = None,
) -> GraphSpecLoadResult:
    """Load a manifest-owned GraphSpec while preserving migration-record custody.

    Feedbax GraphSpec migration records stay on the manifest that directly owns
    the inline GraphSpec payload. Downstream legacy-conversion records already
    present on a model artifact remain distinct and are surfaced separately.
    """
    manifest_obj, manifest_path = _load_graph_spec_manifest_source(manifest)
    base = Path(root) if root is not None else (
        manifest_path.parent if manifest_path is not None else Path(".")
    )

    if isinstance(manifest_obj, GraphSpecManifest):
        return _migrate_manifest_graph_spec_payload(
            manifest_obj,
            manifest_obj.graph_spec,
            existing_records=manifest_obj.migration_records,
            custody_manifest_kind="GraphSpecManifest",
        )

    graph_spec = manifest_obj.graph_spec
    if isinstance(graph_spec, ParentRef):
        referenced = _load_parent_graph_spec_manifest(graph_spec, base=base)
        referenced_result = load_graph_spec_from_manifest(referenced, root=base)
        downstream_records = _append_unique_migration_records(
            referenced_result.downstream_migration_records,
            _downstream_migration_records(manifest_obj.migration_records),
        )
        return referenced_result.model_copy(
            update={"downstream_migration_records": downstream_records}
        )

    return _migrate_manifest_graph_spec_payload(
        manifest_obj,
        graph_spec,
        existing_records=manifest_obj.migration_records,
        custody_manifest_kind="ModelArtifactManifest",
    )


def _load_graph_spec_manifest_source(
    manifest: GraphSpecManifest | ModelArtifactManifest | Path | str,
) -> tuple[GraphSpecManifest | ModelArtifactManifest, Path | None]:
    if isinstance(manifest, GraphSpecManifest | ModelArtifactManifest):
        return manifest, None
    path = Path(manifest)
    loaded = load_manifest(path)
    if not isinstance(loaded, GraphSpecManifest | ModelArtifactManifest):
        raise TypeError(
            "Expected GraphSpecManifest or ModelArtifactManifest, "
            f"got {type(loaded).__name__}."
        )
    return loaded, path


def _load_parent_graph_spec_manifest(parent: ParentRef, *, base: Path) -> GraphSpecManifest:
    if parent.uri is None:
        raise ValueError(
            f"Model artifact graph_spec parent {parent.id!r} has no URI; "
            "GraphSpec migration custody is not discoverable."
        )
    path = Path(parent.uri)
    if not path.is_absolute():
        path = base / path
    loaded = load_manifest(path)
    if not isinstance(loaded, GraphSpecManifest):
        raise TypeError(
            f"Model artifact graph_spec parent {parent.id!r} resolved to "
            f"{type(loaded).__name__}, expected GraphSpecManifest."
        )
    return loaded


def _migrate_manifest_graph_spec_payload(
    manifest: GraphSpecManifest | ModelArtifactManifest,
    payload: SpecPayload,
    *,
    existing_records: list[ArtifactMigrationRecord],
    custody_manifest_kind: Literal["GraphSpecManifest", "ModelArtifactManifest"],
) -> GraphSpecLoadResult:
    if payload.kind != "GraphSpec":
        raise TypeError(f"Expected GraphSpec payload, got {payload.kind!r}.")

    migrated_payload = migrate_spec_payload(payload, path="graph_spec")
    applied_records = [
        record for record in migrated_payload.migration_records if record.tool == "feedbax"
    ]
    migration_records = _append_unique_migration_records(
        existing_records,
        applied_records,
    )
    updated_manifest = manifest.model_copy(
        update={
            "graph_spec": migrated_payload,
            "migration_records": migration_records,
        }
    )
    return GraphSpecLoadResult(
        payload=migrated_payload.inline,
        manifest=updated_manifest,
        custody_manifest_kind=custody_manifest_kind,
        custody_manifest_id=manifest.id,
        applied_migration_records=applied_records,
        migration_records=migration_records,
        downstream_migration_records=_downstream_migration_records(migration_records),
    )


def _append_unique_migration_records(
    existing: list[ArtifactMigrationRecord],
    added: list[ArtifactMigrationRecord],
) -> list[ArtifactMigrationRecord]:
    records = list(existing)
    seen = {_migration_record_key(record) for record in records}
    for record in added:
        key = _migration_record_key(record)
        if key not in seen:
            records.append(record)
            seen.add(key)
    return records


def _migration_record_key(record: ArtifactMigrationRecord) -> tuple[str, str, str, str, str]:
    metadata = json.dumps(record.metadata, sort_keys=True, separators=(",", ":"))
    return (
        record.tool,
        record.migration_id,
        record.source_schema_version,
        record.target_schema_version,
        metadata,
    )


def _downstream_migration_records(
    records: list[ArtifactMigrationRecord],
) -> list[ArtifactMigrationRecord]:
    return [record for record in records if record.tool != "feedbax"]


def training_run_manifest_id(job_id: Optional[str] = None) -> str:
    key = job_id or str(uuid.uuid4())
    return f"feedbax-training-run:{key}"


def write_training_run_manifest(
    *,
    job_id: Optional[str],
    total_batches: int,
    training_spec: Optional[dict[str, Any]] = None,
    task_spec: Optional[dict[str, Any]] = None,
    task_binding_spec: Optional[dict[str, Any]] = None,
    graph_spec: Optional[dict[str, Any]] = None,
    checkpoint_path: Optional[Path | str] = None,
    history_events: Optional[list[dict[str, Any]]] = None,
    retention_plan: Optional[dict[str, Any]] = None,
    retained_observables: Optional[dict[str, Any] | list[dict[str, Any]]] = None,
    status: ManifestStatus = "completed",
    final_loss: Optional[float] = None,
    root: Path | str | None = None,
    provenance: Optional[Provenance] = None,
    issues: Optional[list[str]] = None,
) -> tuple[TrainingRunManifest, Path]:
    """Build, store, and index a local training-run manifest."""
    root_path = Path(root) if root is not None else default_manifest_root()
    artifacts: list[ArtifactRef] = []
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        artifacts.append(
            store_artifact(
                checkpoint_path,
                root=root_path,
                role="training_checkpoint",
                logical_name=f"feedbax_checkpoint_{job_id}.eqx" if job_id else None,
                media_type="application/x-equinox",
            )
        )
    if history_events is not None:
        artifacts.append(
            store_json_artifact(
                history_events,
                root=root_path,
                role="training_history",
                logical_name=f"feedbax_training_history_{job_id}.json"
                if job_id
                else "feedbax_training_history.json",
            )
        )
    if retention_plan is not None:
        artifacts.append(
            store_json_artifact(
                retention_plan,
                root=root_path,
                role="retention_plan",
                logical_name=f"feedbax_retention_plan_{job_id}.json"
                if job_id
                else "feedbax_retention_plan.json",
            )
        )
    if retained_observables is not None:
        artifacts.append(
            store_json_artifact(
                retained_observables,
                root=root_path,
                role="retained_observables",
                logical_name=f"feedbax_retained_observables_{job_id}.json"
                if job_id
                else "feedbax_retained_observables.json",
            )
        )

    prov = provenance or collect_git_provenance()
    if issues:
        prov.issues.extend(issue for issue in issues if issue not in prov.issues)
    if prov.entrypoint is None:
        prov.entrypoint = EntrypointRef(kind="feedbax-worker", name="training")

    manifest = TrainingRunManifest(
        id=training_run_manifest_id(job_id),
        job_id=job_id,
        status=status,
        completed_at=utc_now() if status in {"completed", "failed", "cancelled"} else None,
        graph_spec=spec_payload("GraphSpec", graph_spec) if graph_spec is not None else None,
        training_spec=spec_payload("TrainingSpec", training_spec)
        if training_spec is not None
        else None,
        task_spec=spec_payload("TaskSpec", task_spec) if task_spec is not None else None,
        task_binding_spec=spec_payload("StudioTaskBindingSpec", task_binding_spec)
        if task_binding_spec is not None
        else None,
        summary_metrics={
            key: value
            for key, value in {
                "final_loss": final_loss,
                "total_batches": total_batches,
            }.items()
            if value is not None
        },
        provenance=prov,
        artifacts=artifacts,
    )
    return manifest, write_manifest(manifest, root=root_path)
