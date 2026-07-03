"""Directory packet import/export for Feedbax run manifests and artifacts."""

from __future__ import annotations

import json
import shutil
from collections import deque
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from feedbax.contracts.manifest import (
    BaseManifest,
    MANIFEST_MODELS,
    StrictModel,
    default_manifest_root,
    feedbax_version,
    load_manifest,
    safe_manifest_key,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from feedbax.contracts.migrations import (
    SpecSchemaRegistry,
    UnknownSpecFamily,
    UnsupportedSpecVersion,
    default_spec_registry,
)


MANIFEST_PACKET_SCHEMA_ID = "feedbax.spec.manifest_packet"
MANIFEST_PACKET_SCHEMA_VERSION = "feedbax.spec.manifest_packet.v1"
DEFAULT_PACKET_ARTIFACT_TOTAL_LIMIT_BYTES = 512 * 1024 * 1024

ClosureDirection = Literal["ancestors", "descendants", "both"]
ArtifactInclusionPolicy = Literal["auto", "include", "external"]
ArtifactMode = Literal["included", "external"]


class ManifestPacketError(ValueError):
    """Base class for manifest-packet failures."""


class ManifestPacketValidationError(ManifestPacketError):
    """Raised when packet validation fails before import writes begin."""


class ManifestPacketConflictError(ManifestPacketError):
    """Raised when an import would overwrite a divergent manifest id."""


class ManifestPacketManifestEntry(StrictModel):
    """One manifest file carried by a manifest packet."""

    kind: str
    id: str
    path: str
    sha256: str
    schema_version: str | None = None


class ManifestPacketArtifactEntry(StrictModel):
    """One artifact byte object or external declaration carried by a packet."""

    artifact_id: str
    logical_name: str
    manifest_id: str
    manifest_kind: str
    ref_path: str
    mode: ArtifactMode
    sha256: str
    size_bytes: int | None = None
    path: str | None = None
    external_uri: str | None = None
    media_type: str = "application/octet-stream"
    role: str | None = None

    @model_validator(mode="after")
    def _validate_location(self) -> "ManifestPacketArtifactEntry":
        if self.mode == "included" and not self.path:
            raise ValueError("included packet artifacts require path")
        if self.mode == "external" and not self.external_uri:
            raise ValueError("external packet artifacts require external_uri")
        return self


class ManifestPacketIndex(StrictModel):
    """Schema-governed index for a Feedbax manifest packet directory."""

    schema_id: Literal[MANIFEST_PACKET_SCHEMA_ID] = MANIFEST_PACKET_SCHEMA_ID
    schema_version: Literal[MANIFEST_PACKET_SCHEMA_VERSION] = MANIFEST_PACKET_SCHEMA_VERSION
    producer: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    feedbax_version: str = Field(default_factory=feedbax_version)
    closure_root_ids: list[str]
    closure_direction: ClosureDirection = "descendants"
    manifests: list[ManifestPacketManifestEntry] = Field(default_factory=list)
    artifacts: list[ManifestPacketArtifactEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ManifestPacketExportResult(StrictModel):
    """Result returned by ``export_manifest_packet``."""

    packet_path: str
    index_path: str
    manifest_count: int
    artifact_count: int
    included_artifact_count: int
    external_artifact_count: int


class ManifestPacketImportResult(StrictModel):
    """Result returned by ``import_manifest_packet``."""

    root: str
    packet_path: str
    imported_manifest_ids: list[str] = Field(default_factory=list)
    skipped_manifest_ids: list[str] = Field(default_factory=list)
    artifact_count: int = 0
    included_artifact_count: int = 0
    external_artifact_count: int = 0
    index_path: str | None = None


def export_manifest_packet(
    root_ids: Iterable[str],
    *,
    root: Path | str | None = None,
    dest: Path | str,
    direction: ClosureDirection = "descendants",
    include_artifacts: ArtifactInclusionPolicy | bool = "auto",
    max_included_artifact_bytes: int = DEFAULT_PACKET_ARTIFACT_TOTAL_LIMIT_BYTES,
    producer: Mapping[str, Any] | None = None,
) -> ManifestPacketExportResult:
    """Export a lineage-closed directory packet from a manifest root."""
    root_path = Path(root) if root is not None else default_manifest_root()
    dest_path = Path(dest)
    if dest_path.exists() and any(dest_path.iterdir()):
        raise FileExistsError(f"Manifest packet destination is not empty: {dest_path}")
    dest_path.mkdir(parents=True, exist_ok=True)

    ids = [str(root_id) for root_id in root_ids]
    if not ids:
        raise ManifestPacketValidationError("manifest packet export requires at least one root id")

    records = _scan_manifest_root(root_path)
    missing_roots = [root_id for root_id in ids if root_id not in records]
    if missing_roots:
        raise ManifestPacketValidationError(
            f"Cannot export unknown manifest root ids: {missing_roots}"
        )

    selected_ids = _lineage_closure(ids, records, direction=direction)
    selected = [records[manifest_id] for manifest_id in selected_ids]
    artifact_refs = [
        artifact
        for record in selected
        for artifact in _artifact_refs_from_manifest_data(record.data)
        if artifact.ref.get("sha256")
    ]
    policy = _normalize_artifact_policy(include_artifacts)
    include_all = _should_include_artifacts(
        artifact_refs,
        root=root_path,
        policy=policy,
        max_total_bytes=max_included_artifact_bytes,
    )

    manifest_entries: list[ManifestPacketManifestEntry] = []
    artifact_entries: list[ManifestPacketArtifactEntry] = []
    for record in selected:
        packet_manifest_path = Path("manifests") / record.relative_path.relative_to("manifests")
        target = dest_path / packet_manifest_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.path, target)
        manifest_entries.append(
            ManifestPacketManifestEntry(
                kind=record.manifest.kind,
                id=record.manifest.id,
                path=packet_manifest_path.as_posix(),
                sha256=sha256_bytes(record.raw_bytes),
                schema_version=record.manifest.schema_version,
            )
        )

        for artifact in _artifact_refs_from_manifest_data(record.data):
            entry = _packet_artifact_entry(
                artifact,
                manifest=record.manifest,
                root=root_path,
                packet_root=dest_path,
                include=include_all,
                policy=policy,
            )
            if entry is not None:
                artifact_entries.append(entry)

    index = ManifestPacketIndex(
        producer=dict(producer or {}),
        closure_root_ids=ids,
        closure_direction=direction,
        manifests=manifest_entries,
        artifacts=artifact_entries,
    )
    index_path = dest_path / "packet.json"
    index_path.write_text(
        index.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )
    return ManifestPacketExportResult(
        packet_path=str(dest_path),
        index_path=str(index_path),
        manifest_count=len(manifest_entries),
        artifact_count=len(artifact_entries),
        included_artifact_count=sum(1 for entry in artifact_entries if entry.mode == "included"),
        external_artifact_count=sum(1 for entry in artifact_entries if entry.mode == "external"),
    )


def import_manifest_packet(
    packet: Path | str,
    *,
    root: Path | str | None = None,
    on_conflict: Literal["reject"] = "reject",
    registry: SpecSchemaRegistry | None = None,
) -> ManifestPacketImportResult:
    """Validate and import a directory packet into a manifest root all at once."""
    if on_conflict != "reject":
        raise ValueError("manifest packet v1 only supports on_conflict='reject'")

    packet_path = Path(packet)
    root_path = Path(root) if root is not None else default_manifest_root()
    active_registry = registry or default_spec_registry
    index, packet_hash = _load_packet_index(packet_path, registry=active_registry)
    target_records = _scan_manifest_root(root_path)

    manifest_writes: list[tuple[ManifestPacketManifestEntry, Path, bytes]] = []
    artifact_copies: list[tuple[ManifestPacketArtifactEntry, Path, bytes]] = []
    errors: list[str] = []

    artifact_entries_by_manifest: dict[str, list[ManifestPacketArtifactEntry]] = {}
    for artifact in index.artifacts:
        artifact_entries_by_manifest.setdefault(artifact.manifest_id, []).append(artifact)
        try:
            source_bytes = _validate_packet_artifact(packet_path, artifact)
        except ManifestPacketValidationError as exc:
            errors.append(str(exc))
            continue
        if artifact.mode == "included":
            artifact_copies.append(
                (artifact, _imported_artifact_path(root_path, artifact), source_bytes)
            )

    for entry in index.manifests:
        source_path = packet_path / entry.path
        if not source_path.is_file():
            errors.append(f"Missing packet manifest entry: id={entry.id!r}, path={entry.path!r}")
            continue
        raw_bytes = source_path.read_bytes()
        actual_sha = sha256_bytes(raw_bytes)
        if actual_sha != entry.sha256:
            errors.append(
                "Packet manifest sha256 mismatch: "
                f"id={entry.id!r}, path={entry.path!r}, expected={entry.sha256!r}, "
                f"actual={actual_sha!r}"
            )
            continue
        try:
            raw_data = json.loads(raw_bytes.decode("utf-8"))
            _validate_manifest_schema_families(
                raw_data,
                registry=active_registry,
                manifest_id=entry.id,
            )
            imported_data = _manifest_data_for_import(
                raw_data,
                packet_index=index,
                packet_hash=packet_hash,
                manifest_entry=entry,
                artifact_entries=artifact_entries_by_manifest.get(entry.id, []),
                root=root_path,
            )
            expected_bytes = _manifest_import_bytes(imported_data)
        except Exception as exc:
            errors.append(f"Invalid packet manifest id={entry.id!r}: {exc}")
            continue

        existing = target_records.get(entry.id)
        if existing is not None:
            expected_sha = sha256_bytes(expected_bytes)
            existing_sha = sha256_bytes(existing.raw_bytes)
            if existing_sha not in {entry.sha256, expected_sha}:
                errors.append(
                    "Divergent manifest id already exists: "
                    f"id={entry.id!r}, existing_sha256={existing_sha!r}, "
                    f"packet_sha256={entry.sha256!r}, expected_import_sha256={expected_sha!r}"
                )
            continue

        manifest_writes.append((entry, root_path / entry.path, expected_bytes))

    if errors:
        raise ManifestPacketValidationError(
            "Manifest packet import failed before writing any files:\n- " + "\n- ".join(errors)
        )

    for _artifact, target, data in artifact_copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)

    for _entry, target, data in manifest_writes:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    from feedbax.persistence.manifest_index import rebuild_manifest_index

    db_path = rebuild_manifest_index(root_path)
    skipped = [
        entry.id
        for entry in index.manifests
        if entry.id in target_records and all(write[0].id != entry.id for write in manifest_writes)
    ]
    return ManifestPacketImportResult(
        root=str(root_path),
        packet_path=str(packet_path),
        imported_manifest_ids=[entry.id for entry, _target, _data in manifest_writes],
        skipped_manifest_ids=skipped,
        artifact_count=len(index.artifacts),
        included_artifact_count=sum(1 for entry in index.artifacts if entry.mode == "included"),
        external_artifact_count=sum(1 for entry in index.artifacts if entry.mode == "external"),
        index_path=str(db_path),
    )


class _ManifestRecord(StrictModel):
    path: Path
    relative_path: Path
    raw_bytes: bytes
    data: dict[str, Any]
    manifest: BaseManifest
    parent_ids: set[str]

    model_config = {"arbitrary_types_allowed": True}


class _ArtifactRefRecord(StrictModel):
    ref_path: str
    ref: dict[str, Any]


def _scan_manifest_root(root: Path) -> dict[str, _ManifestRecord]:
    manifests_dir = root / "manifests"
    if not manifests_dir.exists():
        return {}
    records: dict[str, _ManifestRecord] = {}
    for path in sorted(manifests_dir.glob("**/*.json")):
        raw_bytes = path.read_bytes()
        data = json.loads(raw_bytes.decode("utf-8"))
        manifest = load_manifest(path)
        if manifest.id in records:
            raise ManifestPacketValidationError(
                f"Duplicate manifest id in root: id={manifest.id!r}"
            )
        records[manifest.id] = _ManifestRecord(
            path=path,
            relative_path=path.relative_to(root),
            raw_bytes=raw_bytes,
            data=data,
            manifest=manifest,
            parent_ids=_manifest_parent_ids(data, self_id=manifest.id),
        )
    return records


def _lineage_closure(
    root_ids: list[str],
    records: Mapping[str, _ManifestRecord],
    *,
    direction: ClosureDirection,
) -> list[str]:
    selected: set[str] = set(root_ids)
    queue: deque[str] = deque(root_ids)
    child_ids_by_parent: dict[str, set[str]] = {}
    for manifest_id, record in records.items():
        for parent_id in record.parent_ids:
            child_ids_by_parent.setdefault(parent_id, set()).add(manifest_id)

    while queue:
        manifest_id = queue.popleft()
        next_ids: set[str] = set()
        if direction in {"ancestors", "both"}:
            next_ids.update(records[manifest_id].parent_ids & records.keys())
        if direction in {"descendants", "both"}:
            next_ids.update(child_ids_by_parent.get(manifest_id, set()))
        for next_id in sorted(next_ids):
            if next_id not in selected:
                selected.add(next_id)
                queue.append(next_id)
    return [manifest_id for manifest_id in records if manifest_id in selected]


def _manifest_parent_ids(value: Any, *, self_id: str) -> set[str]:
    parent_ids: set[str] = set()
    manifest_kinds = set(MANIFEST_MODELS)

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            kind = item.get("kind")
            ref_id = item.get("id")
            if (
                isinstance(kind, str)
                and kind in manifest_kinds
                and isinstance(ref_id, str)
                and ref_id != self_id
            ):
                parent_ids.add(ref_id)
            for child in item.values():
                visit(child)
            return
        if isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return parent_ids


def _artifact_refs_from_manifest_data(data: dict[str, Any]) -> list[_ArtifactRefRecord]:
    artifacts: list[_ArtifactRefRecord] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            if _is_artifact_ref_data(item):
                artifacts.append(_ArtifactRefRecord(ref_path=path, ref=dict(item)))
            for key, child in item.items():
                visit(child, f"{path}/{key}" if path else f"/{key}")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")

    visit(data, "")
    return artifacts


def _is_artifact_ref_data(value: Mapping[str, Any]) -> bool:
    return (
        isinstance(value.get("role"), str)
        and isinstance(value.get("logical_name"), str)
        and (
            "artifact_id" in value
            or "sha256" in value
            or "uri" in value
            or value.get("storage_backend") is not None
        )
        and "kind" not in value
    )


def _normalize_artifact_policy(
    include_artifacts: ArtifactInclusionPolicy | bool,
) -> ArtifactInclusionPolicy:
    if include_artifacts is True:
        return "include"
    if include_artifacts is False:
        return "external"
    return include_artifacts


def _should_include_artifacts(
    artifacts: list[_ArtifactRefRecord],
    *,
    root: Path,
    policy: ArtifactInclusionPolicy,
    max_total_bytes: int,
) -> bool:
    if policy == "include":
        return True
    if policy == "external":
        return False
    total = 0
    for artifact in artifacts:
        path = _resolve_artifact_source_path(artifact.ref, root=root)
        if path is None or not path.is_file():
            return False
        total += path.stat().st_size
        if total > max_total_bytes:
            return False
    return True


def _packet_artifact_entry(
    artifact: _ArtifactRefRecord,
    *,
    manifest: BaseManifest,
    root: Path,
    packet_root: Path,
    include: bool,
    policy: ArtifactInclusionPolicy,
) -> ManifestPacketArtifactEntry | None:
    ref = artifact.ref
    source_path = _resolve_artifact_source_path(ref, root=root)
    ref_sha = ref.get("sha256")
    if source_path is not None and source_path.is_file():
        actual_sha = sha256_file(source_path)
        if ref_sha is not None and ref_sha != actual_sha:
            raise ManifestPacketValidationError(
                "Artifact sha256 mismatch during packet export: "
                f"manifest_id={manifest.id!r}, ref_path={artifact.ref_path!r}, "
                f"expected={ref_sha!r}, actual={actual_sha!r}"
            )
        ref_sha = actual_sha
    if not isinstance(ref_sha, str) or not ref_sha:
        return None

    fallback_name = source_path.name if source_path is not None else ref_sha
    logical_name = str(ref.get("logical_name") or fallback_name)
    artifact_id = str(ref.get("artifact_id") or f"artifact://sha256/{ref_sha}")
    media_type = str(ref.get("media_type") or "application/octet-stream")
    role = ref.get("role") if isinstance(ref.get("role"), str) else None
    size_bytes = (
        source_path.stat().st_size
        if source_path is not None and source_path.is_file()
        else _optional_int(ref.get("size_bytes"))
    )

    if include:
        if source_path is None or not source_path.is_file():
            if policy == "include":
                raise ManifestPacketValidationError(
                    "Requested included artifact is not locally readable: "
                    f"manifest_id={manifest.id!r}, ref_path={artifact.ref_path!r}"
                )
        else:
            rel_path = (
                Path("artifacts") / safe_manifest_key(artifact_id) / safe_manifest_key(logical_name)
            )
            target = packet_root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            return ManifestPacketArtifactEntry(
                artifact_id=artifact_id,
                logical_name=logical_name,
                manifest_id=manifest.id,
                manifest_kind=manifest.kind,
                ref_path=artifact.ref_path,
                mode="included",
                sha256=ref_sha,
                size_bytes=size_bytes,
                path=rel_path.as_posix(),
                media_type=media_type,
                role=role,
            )

    external_uri = str(ref.get("uri") or ref.get("artifact_id") or artifact_id)
    return ManifestPacketArtifactEntry(
        artifact_id=artifact_id,
        logical_name=logical_name,
        manifest_id=manifest.id,
        manifest_kind=manifest.kind,
        ref_path=artifact.ref_path,
        mode="external",
        sha256=ref_sha,
        size_bytes=size_bytes,
        external_uri=external_uri,
        media_type=media_type,
        role=role,
    )


def _resolve_artifact_source_path(ref: Mapping[str, Any], *, root: Path) -> Path | None:
    candidates: list[Path] = []
    uri = ref.get("uri")
    if isinstance(uri, str) and uri:
        path = Path(uri).expanduser()
        candidates.append(path if path.is_absolute() else root / path)
    metadata = ref.get("metadata")
    if isinstance(metadata, Mapping):
        relative_path = metadata.get("relative_path")
        if isinstance(relative_path, str) and relative_path:
            candidates.append(root / relative_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if candidates else None


def _load_packet_index(
    packet_path: Path,
    *,
    registry: SpecSchemaRegistry,
) -> tuple[ManifestPacketIndex, str]:
    index_path = packet_path / "packet.json"
    if not index_path.is_file():
        raise ManifestPacketValidationError(f"Missing manifest packet index: {index_path}")
    raw_bytes = index_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    schema_id = payload.get("schema_id")
    if schema_id != MANIFEST_PACKET_SCHEMA_ID:
        raise UnsupportedSpecVersion(
            "Unsupported manifest packet schema identity: "
            f"schema_id={schema_id!r}, expected={MANIFEST_PACKET_SCHEMA_ID!r}"
        )
    registry.migrate(
        "ManifestPacket",
        payload,
        source_version=payload.get("schema_version"),
    )
    return ManifestPacketIndex.model_validate(payload), sha256_bytes(raw_bytes)


def _validate_packet_artifact(
    packet_path: Path,
    artifact: ManifestPacketArtifactEntry,
) -> bytes:
    if artifact.mode == "external":
        return b""
    source = packet_path / str(artifact.path)
    if not source.is_file():
        raise ManifestPacketValidationError(
            "Missing packet artifact entry: "
            f"artifact_id={artifact.artifact_id!r}, path={artifact.path!r}"
        )
    data = source.read_bytes()
    actual_sha = sha256_bytes(data)
    if actual_sha != artifact.sha256:
        raise ManifestPacketValidationError(
            "Packet artifact sha256 mismatch: "
            f"artifact_id={artifact.artifact_id!r}, path={artifact.path!r}, "
            f"expected={artifact.sha256!r}, actual={actual_sha!r}"
        )
    if artifact.size_bytes is not None and len(data) != artifact.size_bytes:
        raise ManifestPacketValidationError(
            "Packet artifact size mismatch: "
            f"artifact_id={artifact.artifact_id!r}, expected={artifact.size_bytes}, "
            f"actual={len(data)}"
        )
    return data


def _validate_manifest_schema_families(
    value: Any,
    *,
    registry: SpecSchemaRegistry,
    manifest_id: str,
) -> None:
    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            if "kind" in item and "inline" in item:
                kind = item.get("kind")
                if not isinstance(kind, str):
                    raise UnsupportedSpecVersion(
                        f"SpecPayload kind must be a string: manifest_id={manifest_id!r}, path={path!r}"
                    )
                try:
                    family = registry.resolve(kind)
                except UnknownSpecFamily as exc:
                    schema_id = item.get("schema_id")
                    schema_version = item.get("schema_version")
                    raise UnknownSpecFamily(
                        "Unknown embedded SpecPayload family in manifest packet: "
                        f"manifest_id={manifest_id!r}, path={path!r}, kind={kind!r}, "
                        f"schema_id={schema_id!r}, schema_version={schema_version!r}; {exc}"
                    ) from exc
                schema_id = item.get("schema_id")
                if schema_id is not None and schema_id != family.identity:
                    raise UnsupportedSpecVersion(
                        "Unsupported embedded SpecPayload schema identity in manifest packet: "
                        f"manifest_id={manifest_id!r}, path={path!r}, kind={kind!r}, "
                        f"schema_id={schema_id!r}, expected={family.identity!r}"
                    )
                inline = item.get("inline")
                if not isinstance(inline, Mapping):
                    raise UnsupportedSpecVersion(
                        "Embedded SpecPayload inline value must be a mapping in manifest packet: "
                        f"manifest_id={manifest_id!r}, path={path!r}, kind={kind!r}"
                    )
                registry.migrate(kind, inline, source_version=item.get("schema_version"))
            for key, child in item.items():
                visit(child, f"{path}/{key}" if path else f"/{key}")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")

    visit(value, "")


def _manifest_data_for_import(
    data: dict[str, Any],
    *,
    packet_index: ManifestPacketIndex,
    packet_hash: str,
    manifest_entry: ManifestPacketManifestEntry,
    artifact_entries: list[ManifestPacketArtifactEntry],
    root: Path,
) -> dict[str, Any]:
    imported = json.loads(json.dumps(data))
    metadata = dict(imported.get("metadata") or {})
    metadata["imported_from"] = {
        "schema_id": packet_index.schema_id,
        "schema_version": packet_index.schema_version,
        "packet_sha256": packet_hash,
        "producer": packet_index.producer,
        "manifest_sha256": manifest_entry.sha256,
        "manifest_path": manifest_entry.path,
    }
    imported["metadata"] = metadata
    for artifact in artifact_entries:
        _update_artifact_ref(imported, artifact, root=root)
    return imported


def _update_artifact_ref(
    data: dict[str, Any],
    artifact: ManifestPacketArtifactEntry,
    *,
    root: Path,
) -> None:
    target = _path_get(data, artifact.ref_path)
    if not isinstance(target, dict):
        return
    metadata = dict(target.get("metadata") or {})
    metadata["manifest_packet"] = {
        "mode": artifact.mode,
        "packet_artifact_id": artifact.artifact_id,
        "sha256": artifact.sha256,
    }
    target["metadata"] = metadata
    target["sha256"] = artifact.sha256
    target["size_bytes"] = artifact.size_bytes
    target["artifact_id"] = artifact.artifact_id
    if artifact.mode == "included":
        target["storage_backend"] = "feedbax-local"
        target["uri"] = str(_imported_artifact_path(root, artifact))
    else:
        target["storage_backend"] = "external"
        target["uri"] = artifact.external_uri


def _path_get(data: Any, path: str) -> Any:
    current = data
    for part in [part for part in path.split("/") if part]:
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _manifest_import_bytes(data: dict[str, Any]) -> bytes:
    kind = data.get("kind")
    if not isinstance(kind, str) or kind not in MANIFEST_MODELS:
        raise ValueError(f"Unknown Feedbax manifest kind: {kind!r}")
    model = MANIFEST_MODELS[kind].model_validate(data)
    return (model.model_dump_json(indent=2, exclude_none=True) + "\n").encode("utf-8")


def _imported_artifact_path(root: Path, artifact: ManifestPacketArtifactEntry) -> Path:
    logical_name = safe_manifest_key(artifact.logical_name)
    return (
        root
        / "artifacts"
        / "manifest_packet"
        / safe_manifest_key(artifact.artifact_id)
        / logical_name
    )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None
