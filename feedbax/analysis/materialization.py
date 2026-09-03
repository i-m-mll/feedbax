"""Context-bound analysis materialization helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import equinox as eqx
from pydantic import BaseModel

from feedbax.contracts.strict_json import strict_json_loads

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisArtifactFile, AnalysisRunContext
from feedbax.contracts.base import (
    ArtifactRef,
    ParentRef,
)
from feedbax.contracts.manifest import (
    RegenerationSpec,
    SpecPayload,
)
from feedbax.analysis.types import AnalysisInputData


RegenerationSpecRef = RegenerationSpec | SpecPayload | ParentRef | ArtifactRef


@dataclass(frozen=True)
class ExistingAnalysisArtifact:
    """Existing file that a context-bound materializer wants Feedbax to own."""

    path: Path | str
    role: str
    logical_name: str | None = None
    media_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    group_id: str | None = None
    group_role: str | None = None
    group_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class AnalysisArtifactGroup:
    """Logical multi-file artifact group emitted by a materializer."""

    group_id: str
    members: Sequence[AnalysisArtifactFile]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterializationResult:
    """Opaque downstream payload plus Feedbax-owned artifact custody hints."""

    payload: Any
    payload_metadata: dict[str, Any] = field(default_factory=dict)
    existing_artifacts: Sequence[ExistingAnalysisArtifact] = field(default_factory=tuple)
    artifact_groups: Sequence[AnalysisArtifactGroup] = field(default_factory=tuple)
    artifact_refs: Sequence[ArtifactRef] = field(default_factory=tuple)
    regeneration_specs: Sequence[RegenerationSpecRef] = field(default_factory=tuple)


@dataclass(frozen=True)
class ContextMaterializationPending:
    """Explicit compute result for work that requires ``AnalysisRunContext``."""

    artifact_role: str
    logical_name: str
    schema_boundary: str | None = None
    status: str = "pending_context_artifact_emission"


ContextMaterializerFn = Callable[[AnalysisRunContext], Any | MaterializationResult]
DataContextMaterializerFn = Callable[
    [AnalysisRunContext, AnalysisInputData],
    Any | MaterializationResult,
]
MaterializerInputMode = Literal["context", "context_and_data"]


def _materializer_input_mode(value: str) -> MaterializerInputMode:
    if value not in {"context", "context_and_data"}:
        raise ValueError(
            f"materializer_input must be 'context' or 'context_and_data', got {value!r}"
        )
    return cast(MaterializerInputMode, value)


def materialization_metadata(
    context: AnalysisRunContext,
    *,
    schema_owner: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return generic ownership metadata for opaque downstream materializations."""
    metadata = {
        "analysis_type": context.spec.analysis_type,
        "analysis_manifest_id": context.manifest_id,
        "artifact_owner": "feedbax.AnalysisRunManifest",
    }
    if schema_owner is not None:
        metadata["schema_owner"] = schema_owner
    if extra:
        metadata.update(extra)
    return metadata


class ContextMaterializer(AbstractAnalysis):
    """Analysis node for context-bound JSON artifact emission.

    ``compute()`` intentionally performs no durable work and returns only a
    transparent sentinel. The materializer runs from ``emit_artifacts()`` because
    it needs ``AnalysisRunContext`` for manifest-root paths, artifact custody, and
    regeneration-spec recording. The value returned from ``emit_artifacts()``
    becomes the downstream analysis result.

    Feedbax records artifacts and embedded refs but treats the JSON payload as an
    opaque downstream-owned schema. Materializers receive only the run context by
    default. Set ``materializer_input="context_and_data"`` to opt in to receiving
    the resolved :class:`AnalysisInputData` as a second argument.
    """

    materializer: ContextMaterializerFn | DataContextMaterializerFn = eqx.field(
        kw_only=True,
        static=True,
    )
    artifact_role: str = eqx.field(kw_only=True, static=True)
    logical_name: str = eqx.field(kw_only=True, static=True)
    materializer_input: MaterializerInputMode = eqx.field(
        default="context",
        kw_only=True,
        static=True,
        converter=_materializer_input_mode,
    )
    schema_boundary: str | None = eqx.field(default=None, static=True)
    metadata: dict[str, Any] = eqx.field(default_factory=dict, static=True)

    @property
    def _field_params(self) -> dict[str, Any]:
        """Return stable declarative identity fields without traversing the callable.

        Materializer callbacks may close over resolved analysis inputs, including JAX
        arrays that are neither stable nor safe to inspect while constructing the
        analysis dependency graph. The surrounding recipe/spec owns the callback's
        implementation identity; this node hashes only its declared artifact and
        dispatch contract.
        """
        return {
            "artifact_role": self.artifact_role,
            "logical_name": self.logical_name,
            "materializer_input": self.materializer_input,
            "schema_boundary": self.schema_boundary,
            "metadata": self.metadata,
        }

    def compute(self, data: AnalysisInputData, **kwargs: Any) -> ContextMaterializationPending:
        """Return an explicit sentinel; context-bound materialization is in the hook."""
        del data, kwargs
        return ContextMaterializationPending(
            artifact_role=self.artifact_role,
            logical_name=self.logical_name,
            schema_boundary=self.schema_boundary,
        )

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result: ContextMaterializationPending,
        **kwargs: Any,
    ) -> Any:
        """Materialize the opaque payload and record all declared artifact refs."""
        del result, kwargs
        if self.materializer_input == "context_and_data":
            materializer = cast(DataContextMaterializerFn, self.materializer)
            value = materializer(context, data)
        else:
            materializer = cast(ContextMaterializerFn, self.materializer)
            value = materializer(context)
        materialized = self._coerce_materialization_result(value)
        context.record_artifact_refs_from_value(materialized.payload)
        payload = _json_payload(materialized.payload)
        metadata = {
            **self.metadata,
            **materialized.payload_metadata,
        }
        if self.schema_boundary is not None:
            metadata.setdefault("schema_boundary", self.schema_boundary)

        payload_ref = context.record_json_artifact(
            payload,
            role=self.artifact_role,
            logical_name=self.logical_name,
            metadata=metadata,
        )
        context.record_artifact_refs([payload_ref, *materialized.artifact_refs])

        for artifact in materialized.existing_artifacts:
            context.record_artifact(
                artifact.path,
                role=artifact.role,
                logical_name=artifact.logical_name,
                media_type=artifact.media_type,
                metadata=artifact.metadata,
                group_id=artifact.group_id,
                group_role=artifact.group_role,
                group_metadata=artifact.group_metadata,
            )
        for group in materialized.artifact_groups:
            context.record_artifact_group(
                group_id=group.group_id,
                members=group.members,
                metadata=group.metadata,
            )
        context.record_regeneration_specs(materialized.regeneration_specs)
        return payload

    def _coerce_materialization_result(self, value: Any | MaterializationResult):
        if isinstance(value, MaterializationResult):
            return value
        return MaterializationResult(payload=value)


def existing_file_artifact_group(
    path: Path | str,
    *,
    group_id: str,
    role: str,
    logical_name: str | None = None,
    group_role: str | None = None,
    media_type: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    group_metadata: Mapping[str, Any] | None = None,
) -> tuple[AnalysisArtifactGroup, ...]:
    """Describe an existing file as a one-member artifact group.

    Missing paths and paths that are not regular files produce no group. This
    makes optional materializations composable without deferring the existence
    check to artifact custody.
    """
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_file():
        return ()
    return (
        AnalysisArtifactGroup(
            group_id=group_id,
            members=(
                AnalysisArtifactFile(
                    path=artifact_path,
                    role=role,
                    logical_name=logical_name,
                    media_type=media_type,
                    metadata=dict(metadata or {}),
                    group_role=group_role,
                ),
            ),
            metadata=dict(group_metadata or {}),
        ),
    )


def directory_artifact_group(
    path: Path | str,
    *,
    group_id: str,
    role: str,
    group_role: str | None = None,
    logical_name_root: Path | str | None = None,
    media_type: str | None = None,
    metadata_for: Callable[[Path], Mapping[str, Any]] | None = None,
    group_metadata: Mapping[str, Any] | None = None,
) -> tuple[AnalysisArtifactGroup, ...]:
    """Describe all regular files below a directory as one artifact group.

    Members are sorted by path for deterministic manifests. Logical names are
    relative POSIX paths rooted at ``logical_name_root`` when supplied, or at
    the input directory otherwise. A file outside an explicit logical-name root
    raises :class:`ValueError` instead of silently emitting a host-specific name.
    """
    directory = Path(path).expanduser()
    if not directory.is_dir():
        return ()
    files = tuple(sorted(item for item in directory.rglob("*") if item.is_file()))
    if not files:
        return ()
    name_root = directory if logical_name_root is None else Path(logical_name_root).expanduser()
    members = tuple(
        AnalysisArtifactFile(
            path=item,
            role=role,
            logical_name=item.relative_to(name_root).as_posix(),
            media_type=media_type,
            metadata=dict(metadata_for(item)) if metadata_for is not None else {},
            group_role=group_role,
        )
        for item in files
    )
    return (
        AnalysisArtifactGroup(
            group_id=group_id,
            members=members,
            metadata=dict(group_metadata or {}),
        ),
    )


def manifest_artifact_group(
    manifest: Mapping[str, Any],
    *,
    entries_key: str,
    artifact_key: str,
    group_id: str,
    role: str,
    path_key: str = "path",
    path_root: Path | str | None = None,
    group_role: str | None = None,
    media_type: str | None = None,
    logical_name_for: Callable[[str, Mapping[str, Any], Path], str | None] | None = None,
    metadata_for: Callable[[str, Mapping[str, Any], Path], Mapping[str, Any]] | None = None,
    group_metadata: Mapping[str, Any] | None = None,
) -> tuple[AnalysisArtifactGroup, ...]:
    """Build one artifact group from path-bearing entries in a manifest mapping.

    ``manifest[entries_key]`` is expected to map entry IDs to mappings, each of
    which may contain a nested ``artifact_key`` mapping with ``path_key``. Invalid,
    absent, and non-file entries are skipped. Relative paths are resolved against
    ``path_root`` when supplied. Callbacks receive the entry ID, nested artifact
    mapping, and resolved path so downstream packages can own logical names and
    metadata without Feedbax interpreting their schemas.
    """
    entries = manifest.get(entries_key)
    if not isinstance(entries, Mapping):
        return ()
    root = None if path_root is None else Path(path_root).expanduser()
    members: list[AnalysisArtifactFile] = []
    for raw_entry_id, entry in entries.items():
        if not isinstance(entry, Mapping):
            continue
        artifact = entry.get(artifact_key)
        if not isinstance(artifact, Mapping):
            continue
        raw_path = artifact.get(path_key)
        if raw_path is None:
            continue
        try:
            artifact_path = Path(raw_path).expanduser()
        except TypeError:
            continue
        if not artifact_path.is_absolute() and root is not None:
            artifact_path = root / artifact_path
        if not artifact_path.is_file():
            continue
        entry_id = str(raw_entry_id)
        members.append(
            AnalysisArtifactFile(
                path=artifact_path,
                role=role,
                logical_name=(
                    logical_name_for(entry_id, artifact, artifact_path)
                    if logical_name_for is not None
                    else artifact_path.name
                ),
                media_type=media_type,
                metadata=(
                    dict(metadata_for(entry_id, artifact, artifact_path))
                    if metadata_for is not None
                    else {}
                ),
                group_role=group_role,
            )
        )
    if not members:
        return ()
    return (
        AnalysisArtifactGroup(
            group_id=group_id,
            members=tuple(members),
            metadata=dict(group_metadata or {}),
        ),
    )


def read_json_payload(path: Path | str) -> Any:
    """Read one UTF-8 JSON payload from ``path``."""
    return strict_json_loads(Path(path).read_text(encoding="utf-8"))


def _json_payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {key: _json_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_payload(item) for item in value]
    return value


__all__ = [
    "AnalysisArtifactGroup",
    "ContextMaterializationPending",
    "ContextMaterializer",
    "ContextMaterializerFn",
    "DataContextMaterializerFn",
    "ExistingAnalysisArtifact",
    "MaterializerInputMode",
    "MaterializationResult",
    "RegenerationSpecRef",
    "directory_artifact_group",
    "existing_file_artifact_group",
    "manifest_artifact_group",
    "materialization_metadata",
    "read_json_payload",
]
