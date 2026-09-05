"""Directory packet import/export for Feedbax run manifests and artifacts."""

from __future__ import annotations

import stat
import unicodedata
from collections import deque
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from feedbax.contracts.base import (
    ArtifactRef,
    StrictModel,
    default_manifest_root,
    feedbax_version,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from feedbax.contracts.artifact_custody import (
    LEGACY_CONTENT_MIGRATION_COMMAND,
    ContentManifestConflictError,
    ContentMigrationMapping,
    ContentMigrationReceipt,
    LegacyContentFormatError,
)
from feedbax.contracts.manifest import (
    BaseManifest,
    MANIFEST_MODELS,
    load_manifest,
)
from feedbax.contracts.migrations import (
    SpecSchemaRegistry,
    UnknownSpecFamily,
    UnsupportedSpecVersion,
    default_spec_registry,
)
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


MANIFEST_PACKET_SCHEMA_ID = "feedbax.spec.manifest_packet"
MANIFEST_PACKET_SCHEMA_VERSION_V1 = "feedbax.spec.manifest_packet.v1"
MANIFEST_PACKET_SCHEMA_VERSION = "feedbax.spec.manifest_packet.v2"

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
    target_path: str | None = None
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


class ManifestPacketObjectEntry(StrictModel):
    """One unique exact object carried once by a self-contained packet."""

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    path: str


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
    objects: list[ManifestPacketObjectEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_unique_entries(self) -> "ManifestPacketIndex":
        object_digests = [entry.sha256 for entry in self.objects]
        object_paths = [entry.path for entry in self.objects]
        manifest_keys = [(entry.kind, entry.id) for entry in self.manifests]
        target_paths = [entry.target_path for entry in self.manifests]
        artifact_keys = [
            (entry.manifest_kind, entry.manifest_id, entry.ref_path) for entry in self.artifacts
        ]
        for name, values in (
            ("object digest", object_digests),
            ("object path", object_paths),
            ("manifest kind/id", manifest_keys),
            ("artifact logical ref", artifact_keys),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"manifest packet contains duplicate {name} entries")
        if any(path is None for path in target_paths):
            raise ValueError("manifest packet v2 requires every manifest target_path")
        collision_keys = [str(path).casefold() for path in target_paths]
        if len(collision_keys) != len(set(collision_keys)):
            raise ValueError("manifest packet target paths collide on a portable filesystem")
        return self


class _ManifestPacketIndexV1(StrictModel):
    """Offline-decoder shape for the retired path-backed packet format."""

    schema_id: Literal[MANIFEST_PACKET_SCHEMA_ID] = MANIFEST_PACKET_SCHEMA_ID
    schema_version: Literal[MANIFEST_PACKET_SCHEMA_VERSION_V1] = MANIFEST_PACKET_SCHEMA_VERSION_V1
    producer: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    feedbax_version: str
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
    include_artifacts: ArtifactInclusionPolicy | bool = "include",
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
    policy = _normalize_artifact_policy(include_artifacts)
    include_all = policy == "include"

    manifest_entries: list[ManifestPacketManifestEntry] = []
    artifact_entries: list[ManifestPacketArtifactEntry] = []
    object_entries: dict[str, ManifestPacketObjectEntry] = {}
    for record in selected:
        packet_manifest_path = _packet_object_path(sha256_bytes(record.raw_bytes))
        target = dest_path / packet_manifest_path
        _publish_packet_object(record.raw_bytes, target)
        object_entries.setdefault(
            sha256_bytes(record.raw_bytes),
            ManifestPacketObjectEntry(
                sha256=sha256_bytes(record.raw_bytes),
                size_bytes=len(record.raw_bytes),
                path=packet_manifest_path.as_posix(),
            ),
        )
        manifest_entries.append(
            ManifestPacketManifestEntry(
                kind=record.manifest.kind,
                id=record.manifest.id,
                path=packet_manifest_path.as_posix(),
                target_path=record.relative_path.as_posix(),
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
                if entry.mode == "included":
                    assert entry.path is not None and entry.size_bytes is not None
                    object_entries.setdefault(
                        entry.sha256,
                        ManifestPacketObjectEntry(
                            sha256=entry.sha256,
                            size_bytes=entry.size_bytes,
                            path=entry.path,
                        ),
                    )

    index = ManifestPacketIndex(
        producer=dict(producer or {}),
        closure_root_ids=ids,
        closure_direction=direction,
        manifests=manifest_entries,
        artifacts=artifact_entries,
        objects=[object_entries[digest] for digest in sorted(object_entries)],
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
    index = _load_packet_index(packet_path, registry=active_registry)
    target_records = _scan_manifest_root(root_path)
    packet_objects = _validate_packet_objects(packet_path, index)
    provider = ImmutableArtifactBlobProvider(root_path.absolute())

    manifest_writes: list[tuple[ManifestPacketManifestEntry, Path, bytes]] = []
    errors: list[str] = []

    for artifact in index.artifacts:
        try:
            source_bytes = _validate_packet_artifact(packet_path, artifact)
        except ManifestPacketValidationError as exc:
            errors.append(str(exc))
            continue
        if artifact.mode == "included" and packet_objects.get(artifact.sha256) != source_bytes:
            errors.append(
                f"packet artifact does not resolve its unique object: {artifact.artifact_id!r}"
            )

    for entry in index.manifests:
        try:
            packet_relative_path = _safe_packet_entry_path(
                entry.path,
                entry_type="manifest",
                entry_id=entry.id,
            )
            source_path = _contained_path(
                packet_path,
                packet_relative_path,
                container_name="packet directory",
                entry_type="manifest",
                entry_id=entry.id,
                raw_path=entry.path,
            )
            if entry.target_path is None:
                raise ManifestPacketValidationError(
                    f"Manifest packet v2 entry lacks target_path: id={entry.id!r}"
                )
            target_relative_path = _safe_packet_entry_path(
                entry.target_path,
                entry_type="manifest target",
                entry_id=entry.id,
            )
            target_path = _contained_path(
                root_path,
                target_relative_path,
                container_name="target root",
                entry_type="manifest target",
                entry_id=entry.id,
                raw_path=entry.target_path,
            )
        except ManifestPacketValidationError as exc:
            errors.append(str(exc))
            continue
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
        if packet_objects.get(entry.sha256) != raw_bytes:
            errors.append(f"packet manifest does not resolve its unique object: id={entry.id!r}")
            continue
        try:
            raw_data = strict_json_loads(
                raw_bytes.decode("utf-8"),
                ref=f"packet manifest {entry.path!r}",
            )
            _validate_manifest_schema_families(
                raw_data,
                registry=active_registry,
                manifest_id=entry.id,
            )
        except Exception as exc:
            errors.append(f"Invalid packet manifest id={entry.id!r}: {exc}")
            continue

        existing = target_records.get(entry.id)
        if existing is not None:
            existing_sha = sha256_bytes(existing.raw_bytes)
            if existing_sha != entry.sha256:
                errors.append(
                    "Divergent manifest id already exists: "
                    f"id={entry.id!r}, existing_sha256={existing_sha!r}, "
                    f"packet_sha256={entry.sha256!r}"
                )
            continue

        manifest_writes.append((entry, target_path, raw_bytes))

    if errors:
        raise ManifestPacketValidationError(
            "Manifest packet import failed before writing any files:\n- " + "\n- ".join(errors)
        )

    for digest in sorted(packet_objects):
        provider.store_object(packet_objects[digest])

    try:
        provider.bind_manifests(
            tuple((entry.kind, entry.id, data) for entry, _target, data in manifest_writes)
            + tuple(
                (entry.kind, entry.id, packet_objects[entry.sha256])
                for entry in index.manifests
                if entry.id in target_records
            )
        )
    except ContentManifestConflictError as exc:
        raise ManifestPacketConflictError(str(exc)) from exc

    for _entry, target, data in manifest_writes:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != data:
                raise ManifestPacketConflictError(
                    f"manifest target changed after validation: {target}"
                )
        else:
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


def migrate_manifest_packet_v1(
    packet: Path | str,
    *,
    destination_root: Path | str,
) -> ContentMigrationReceipt:
    """Offline-only conversion of one authenticated packet-v1 directory into the store."""
    packet_path = Path(packet)
    index_path = packet_path / "packet.json"
    try:
        raw_index = index_path.read_bytes()
        payload = strict_json_loads(raw_index.decode("utf-8"), ref=str(index_path))
        index = _ManifestPacketIndexV1.model_validate(payload)
    except Exception as exc:
        raise ManifestPacketValidationError(f"invalid manifest packet v1 index: {exc}") from exc
    provider = ImmutableArtifactBlobProvider(Path(destination_root).absolute())
    mappings: list[ContentMigrationMapping] = []
    manifests: list[tuple[str, str, bytes]] = []

    for entry in index.manifests:
        source = _contained_path(
            packet_path,
            _safe_packet_entry_path(entry.path, entry_type="manifest", entry_id=entry.id),
            container_name="packet directory",
            entry_type="manifest",
            entry_id=entry.id,
            raw_path=entry.path,
        )
        data = _read_legacy_packet_file(source, expected_sha256=entry.sha256)
        manifests.append((entry.kind, entry.id, data))
        object_ref = provider.store_object(data)
        mappings.append(
            ContentMigrationMapping(
                source_type="manifest-packet-v1",
                source_ref=f"{index_path}:{entry.path}",
                source_sha256=entry.sha256,
                target_ref=f"content-manifest://{entry.kind}/{entry.id}",
                target_sha256=object_ref.sha256,
                size_bytes=object_ref.size_bytes,
            )
        )

    for entry in index.artifacts:
        if entry.mode != "included" or entry.path is None:
            raise ManifestPacketValidationError(
                "packet-v1 external/path-backed artifacts require their exact source paths; "
                f"rerun {LEGACY_CONTENT_MIGRATION_COMMAND} with a self-contained packet"
            )
        source = _contained_path(
            packet_path,
            _safe_packet_entry_path(
                entry.path,
                entry_type="artifact",
                entry_id=entry.artifact_id,
            ),
            container_name="packet directory",
            entry_type="artifact",
            entry_id=entry.artifact_id,
            raw_path=entry.path,
        )
        data = _read_legacy_packet_file(
            source,
            expected_sha256=entry.sha256,
            expected_size=entry.size_bytes,
        )
        object_ref = provider.store_object(data)
        mappings.append(
            ContentMigrationMapping(
                source_type="manifest-packet-v1",
                source_ref=(
                    f"{index_path}:{entry.path}"
                    f"#{entry.manifest_kind}/{entry.manifest_id}{entry.ref_path}"
                ),
                source_sha256=entry.sha256,
                target_ref=object_ref.object_id,
                target_sha256=object_ref.sha256,
                size_bytes=object_ref.size_bytes,
            )
        )
    provider.bind_manifests(tuple(manifests))
    return ContentMigrationReceipt(
        source_schema_version=MANIFEST_PACKET_SCHEMA_VERSION_V1,
        destination_store=str(provider.root),
        mappings=tuple(sorted(mappings, key=lambda item: (item.source_ref, item.target_ref))),
        metadata={
            "packet_index_sha256": sha256_bytes(raw_index),
            "migration_mode": "offline-explicit",
        },
    )


def _read_legacy_packet_file(
    source: Path,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
) -> bytes:
    try:
        source_stat = source.lstat()
    except OSError as exc:
        raise ManifestPacketValidationError(
            f"legacy packet member is unavailable: {source}"
        ) from exc
    if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_nlink != 1:
        raise ManifestPacketValidationError(
            f"legacy packet member must be one regular non-link file: {source}"
        )
    data = source.read_bytes()
    if sha256_bytes(data) != expected_sha256 or (
        expected_size is not None and len(data) != expected_size
    ):
        raise ManifestPacketValidationError(f"legacy packet member identity mismatch: {source}")
    return data


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
    if manifests_dir.is_symlink() or not manifests_dir.is_dir():
        raise ManifestPacketValidationError(
            f"Manifest packet source root must be a non-symlink directory: {manifests_dir}"
        )
    records: dict[str, _ManifestRecord] = {}
    portable_paths: set[str] = set()
    for path in sorted(manifests_dir.glob("**/*.json")):
        path_stat = path.lstat()
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
            raise ManifestPacketValidationError(
                f"Manifest packet source must be one regular non-link file: {path}"
            )
        portable_key = unicodedata.normalize("NFC", path.relative_to(root).as_posix()).casefold()
        if portable_key in portable_paths:
            raise ManifestPacketValidationError(
                f"Manifest packet source paths collide on a portable filesystem: {path}"
            )
        portable_paths.add(portable_key)
        raw_bytes = path.read_bytes()
        data = strict_json_loads(raw_bytes.decode("utf-8"), ref=str(path))
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
    if include_artifacts == "auto":
        raise ManifestPacketValidationError(
            "manifest packet v2 requires explicit included or external artifact mode; "
            "use include_artifacts='include' for a self-contained packet"
        )
    return include_artifacts


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
        source_stat = source_path.lstat()
        if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_nlink != 1:
            raise ManifestPacketValidationError(
                "Packet artifacts must be one regular non-link file: "
                f"manifest_id={manifest.id!r}, ref_path={artifact.ref_path!r}"
            )
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
            rel_path = _packet_object_path(ref_sha)
            target = packet_root / rel_path
            _publish_packet_object(source_path.read_bytes(), target)
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


def _packet_object_path(digest: str) -> Path:
    return Path("objects") / "sha256" / digest[:2] / digest


def _publish_packet_object(data: bytes, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_bytes() != data:
            raise ManifestPacketValidationError(
                f"packet object path already contains conflicting bytes: {target}"
            )
        return
    target.write_bytes(data)


def _resolve_artifact_source_path(ref: Mapping[str, Any], *, root: Path) -> Path | None:
    candidates: list[Path] = []
    uri = ref.get("uri")
    if isinstance(uri, str) and uri:
        if uri.startswith("artifact://sha256/"):
            artifact = ArtifactRef.model_validate(ref)
            provider = ImmutableArtifactBlobProvider(root.absolute())
            candidates.append(provider.root / provider.canonical_relative_path(artifact))
        else:
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
) -> ManifestPacketIndex:
    index_path = packet_path / "packet.json"
    if not index_path.is_file():
        raise ManifestPacketValidationError(f"Missing manifest packet index: {index_path}")
    index_stat = index_path.lstat()
    if not stat.S_ISREG(index_stat.st_mode) or index_stat.st_nlink != 1:
        raise ManifestPacketValidationError(
            f"Manifest packet index must be one regular non-link file: {index_path}"
        )
    raw_bytes = index_path.read_bytes()
    payload = strict_json_loads(raw_bytes.decode("utf-8"), ref=str(index_path))
    schema_id = payload.get("schema_id")
    if schema_id != MANIFEST_PACKET_SCHEMA_ID:
        raise UnsupportedSpecVersion(
            "Unsupported manifest packet schema identity: "
            f"schema_id={schema_id!r}, expected={MANIFEST_PACKET_SCHEMA_ID!r}"
        )
    if payload.get("schema_version") == MANIFEST_PACKET_SCHEMA_VERSION_V1:
        raise LegacyContentFormatError(
            "manifest packet v1 is offline-migration input, not runtime input; run "
            f"{LEGACY_CONTENT_MIGRATION_COMMAND} --packet-v1 <packet> "
            "--destination-store <root>"
        )
    registry.migrate(
        "ManifestPacket",
        payload,
        source_version=payload.get("schema_version"),
    )
    return ManifestPacketIndex.model_validate(payload)


def _validate_packet_objects(
    packet_path: Path,
    index: ManifestPacketIndex,
) -> dict[str, bytes]:
    objects: dict[str, bytes] = {}
    paths: set[str] = set()
    errors: list[str] = []
    for entry in index.objects:
        try:
            relative = _safe_packet_entry_path(
                entry.path,
                entry_type="object",
                entry_id=entry.sha256,
            )
            expected_path = _packet_object_path(entry.sha256)
            if relative != expected_path:
                raise ManifestPacketValidationError(
                    f"Packet object path is not canonical: digest={entry.sha256!r}, "
                    f"path={entry.path!r}, expected={expected_path.as_posix()!r}"
                )
            if entry.sha256 in objects:
                raise ManifestPacketValidationError(
                    f"Duplicate packet object digest: {entry.sha256!r}"
                )
            if entry.path in paths:
                raise ManifestPacketValidationError(f"Duplicate packet object path: {entry.path!r}")
            source = _contained_path(
                packet_path,
                relative,
                container_name="packet directory",
                entry_type="object",
                entry_id=entry.sha256,
                raw_path=entry.path,
            )
            source_stat = source.lstat()
            if not source.is_file() or source.is_symlink() or source_stat.st_nlink != 1:
                raise ManifestPacketValidationError(
                    f"Packet object must be one regular non-link file: {entry.path!r}"
                )
            data = source.read_bytes()
            if len(data) != entry.size_bytes or sha256_bytes(data) != entry.sha256:
                raise ManifestPacketValidationError(
                    f"Packet object identity mismatch: digest={entry.sha256!r}, path={entry.path!r}"
                )
            objects[entry.sha256] = data
            paths.add(entry.path)
        except (OSError, ManifestPacketValidationError) as exc:
            errors.append(str(exc))
    referenced = {entry.sha256 for entry in index.manifests}
    referenced.update(entry.sha256 for entry in index.artifacts if entry.mode == "included")
    missing = sorted(referenced - objects.keys())
    if missing:
        errors.append(f"Packet references missing object digests: {missing}")
    if errors:
        raise ManifestPacketValidationError(
            "Manifest packet object validation failed before publishing references:\n- "
            + "\n- ".join(errors)
        )
    return objects


def _validate_packet_artifact(
    packet_path: Path,
    artifact: ManifestPacketArtifactEntry,
) -> bytes:
    if artifact.mode == "external":
        return b""
    relative_path = _safe_packet_entry_path(
        artifact.path,
        entry_type="artifact",
        entry_id=artifact.artifact_id,
    )
    source = _contained_path(
        packet_path,
        relative_path,
        container_name="packet directory",
        entry_type="artifact",
        entry_id=artifact.artifact_id,
        raw_path=artifact.path,
    )
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


def _safe_packet_entry_path(
    path: str | None,
    *,
    entry_type: str,
    entry_id: str,
) -> Path:
    if not isinstance(path, str) or not path:
        raise ManifestPacketValidationError(
            f"Unsafe packet {entry_type} path: id={entry_id!r}, path={path!r}; "
            "packet entries must use a non-empty relative path"
        )
    candidate = Path(path)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or "\\" in path
        or "\x00" in path
        or unicodedata.normalize("NFC", path) != path
    ):
        raise ManifestPacketValidationError(
            f"Unsafe packet {entry_type} path: id={entry_id!r}, path={path!r}; "
            "packet entries must stay within the packet directory and target root"
        )
    if candidate == Path(".") or any(part == "" for part in candidate.parts):
        raise ManifestPacketValidationError(
            f"Unsafe packet {entry_type} path: id={entry_id!r}, path={path!r}; "
            "packet entries must name a file"
        )
    return candidate


def _contained_path(
    root: Path,
    relative_path: Path,
    *,
    container_name: str,
    entry_type: str,
    entry_id: str,
    raw_path: str | None,
) -> Path:
    root_path = root.resolve(strict=False)
    candidate = (root_path / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ManifestPacketValidationError(
            f"Unsafe packet {entry_type} path: id={entry_id!r}, path={raw_path!r}; "
            f"resolved path escapes the {container_name}"
        ) from exc
    current = root_path
    for part in relative_path.parts:
        current = current / part
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(current_stat.st_mode):
            raise ManifestPacketValidationError(
                f"Unsafe packet {entry_type} path: id={entry_id!r}, path={raw_path!r}; "
                f"symlinks are forbidden in the {container_name}"
            )
    return candidate


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None
