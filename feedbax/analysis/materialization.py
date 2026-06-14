"""Context-bound analysis materialization helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import equinox as eqx
from pydantic import BaseModel

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisArtifactFile, AnalysisRunContext
from feedbax.manifest import ArtifactRef, ParentRef, RegenerationSpec, SpecPayload
from feedbax.types import AnalysisInputData


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
    opaque downstream-owned schema.
    """

    materializer: ContextMaterializerFn = eqx.field(kw_only=True, static=True)
    artifact_role: str = eqx.field(kw_only=True, static=True)
    logical_name: str = eqx.field(kw_only=True, static=True)
    schema_boundary: str | None = eqx.field(default=None, static=True)
    metadata: dict[str, Any] = eqx.field(default_factory=dict, static=True)

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
        del data, result, kwargs
        materialized = self._coerce_materialization_result(self.materializer(context))
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
    "ExistingAnalysisArtifact",
    "MaterializationResult",
    "RegenerationSpecRef",
    "materialization_metadata",
]
