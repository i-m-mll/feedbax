"""Workspace replay artifact contracts and retention compilation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from feedbax.contracts.graph import (
    RetainedObservableSpec,
    RetainedObservableTargetSpec,
    RetentionPolicySpec,
    StudioArtifactRef,
    StudioManifestRef,
    StudioSelectorRef,
)


WORKSPACE_REPLAY_SCHEMA_ID = "feedbax.manifest.studio.workspace_replay"
WORKSPACE_REPLAY_SCHEMA_VERSION = "feedbax.manifest.studio.workspace_replay.v1"
WORKSPACE_REPLAY_SCHEMA_VERSION_V0 = "feedbax.manifest.studio.workspace_replay.v0"

WorkspaceReplaySourceMode = Literal["resolved_scene", "imported_artifact"]
WorkspaceReplayTrialIdentitySource = Literal["stable_id", "index_fallback"]
WorkspaceReplayWarningCode = Literal[
    "missing_trial_metadata",
    "missing_manifest_refs",
    "missing_anchor_resolution",
    "missing_overlay_metadata",
    "npz_browser_downgrade",
]


class WorkspaceReplayModel(BaseModel):
    """Base model for workspace replay contracts."""

    model_config = ConfigDict(extra="forbid")


class WorkspaceReplayWarning(WorkspaceReplayModel):
    """Human-readable downgrade or metadata warning attached to replay products."""

    code: WorkspaceReplayWarningCode
    message: str
    path: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceReplayTrialIdentity(WorkspaceReplayModel):
    """Stable trial identity with explicit deterministic-index fallback."""

    index: int = Field(ge=0)
    stable_id: Optional[str] = None
    source: WorkspaceReplayTrialIdentitySource = "index_fallback"
    label: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity_source(self) -> "WorkspaceReplayTrialIdentity":
        if self.stable_id:
            self.source = "stable_id"
        elif self.source != "index_fallback":
            raise ValueError("trial identity source requires stable_id unless index_fallback")
        return self


class WorkspaceReplaySampleAxis(WorkspaceReplayModel):
    """Time/sample axis for replay tracks."""

    length: int = Field(ge=0)
    units: Optional[str] = None
    dt: Optional[float] = Field(default=None, gt=0)
    values: Optional[list[float]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_values_length(self) -> "WorkspaceReplaySampleAxis":
        if self.values is not None and len(self.values) != self.length:
            raise ValueError("time axis values length must equal length")
        return self


class WorkspaceReplayTrack(WorkspaceReplayModel):
    """One server-resolved Cartesian anchor track for a trial."""

    anchor_id: str
    selector: StudioSelectorRef
    samples: list[list[float]]
    dim: int = Field(gt=0)
    dtype: str = "float32"
    units: Optional[str] = None
    frame: Optional[str] = None
    label: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_samples_shape(self) -> "WorkspaceReplayTrack":
        for index, sample in enumerate(self.samples):
            if len(sample) != self.dim:
                raise ValueError(
                    f"track {self.anchor_id!r} sample {index} has dim {len(sample)}, "
                    f"expected {self.dim}"
                )
        return self


class WorkspaceReplayOverlayChannel(WorkspaceReplayModel):
    """Bound overlay channel such as muscle activation for one replay trial."""

    id: str
    selector: StudioSelectorRef
    samples: list[Any]
    dtype: str = "float32"
    shape: Optional[list[Any]] = None
    units: Optional[str] = None
    label: Optional[str] = None
    binding: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceReplayTrialSpecSnapshot(WorkspaceReplayModel):
    """Trial-spec snapshot sufficient to label and interpret replay tracks."""

    summary: dict[str, Any] = Field(default_factory=dict)
    targets: dict[str, Any] = Field(default_factory=dict)
    timeline: Optional[dict[str, Any]] = None
    snapshot: Optional[dict[str, Any]] = None
    schema_id: Optional[str] = None
    schema_version: Optional[str] = None
    sha256: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceReplayManifestRefs(WorkspaceReplayModel):
    """Manifest and artifact refs needed to reproduce replay provenance."""

    spec_snapshot: Optional[StudioArtifactRef] = None
    checkpoint: Optional[StudioArtifactRef | StudioManifestRef] = None
    task_variant: Optional[StudioManifestRef | StudioArtifactRef] = None
    seed: Optional[int | str] = None
    environment: Optional[StudioArtifactRef | StudioManifestRef | dict[str, Any]] = None
    producer_manifest: Optional[StudioManifestRef] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceReplayTrial(WorkspaceReplayModel):
    """Per-trial replay payload with selector-keyed geometry and overlays."""

    identity: WorkspaceReplayTrialIdentity
    time: WorkspaceReplaySampleAxis
    tracks: list[WorkspaceReplayTrack] = Field(default_factory=list)
    overlays: list[WorkspaceReplayOverlayChannel] = Field(default_factory=list)
    trial_spec: WorkspaceReplayTrialSpecSnapshot = Field(
        default_factory=WorkspaceReplayTrialSpecSnapshot
    )
    manifest_refs: WorkspaceReplayManifestRefs = Field(default_factory=WorkspaceReplayManifestRefs)
    warnings: list[WorkspaceReplayWarning] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_trial_lengths(self) -> "WorkspaceReplayTrial":
        for track in self.tracks:
            if len(track.samples) != self.time.length:
                raise ValueError(
                    f"track {track.anchor_id!r} sample count {len(track.samples)} "
                    f"does not match trial time length {self.time.length}"
                )
        for channel in self.overlays:
            if len(channel.samples) != self.time.length:
                raise ValueError(
                    f"overlay {channel.id!r} sample count {len(channel.samples)} "
                    f"does not match trial time length {self.time.length}"
                )
        return self


class WorkspaceReplayImportedArtifact(WorkspaceReplayModel):
    """Explicit downgrade reference for legacy/offline trajectory browser data."""

    artifact: StudioArtifactRef
    source_format: Literal["npz", "unknown"] = "npz"
    missing_metadata: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceReplayProduct(WorkspaceReplayModel):
    """Durable replay product for Studio workspace playback."""

    schema_id: Literal[WORKSPACE_REPLAY_SCHEMA_ID] = WORKSPACE_REPLAY_SCHEMA_ID
    schema_version: Literal[WORKSPACE_REPLAY_SCHEMA_VERSION] = WORKSPACE_REPLAY_SCHEMA_VERSION
    product_kind: Literal["workspace_replay"] = "workspace_replay"
    source_mode: WorkspaceReplaySourceMode = "resolved_scene"
    trials: list[WorkspaceReplayTrial] = Field(default_factory=list)
    imported_artifact: Optional[WorkspaceReplayImportedArtifact] = None
    manifest_refs: WorkspaceReplayManifestRefs = Field(default_factory=WorkspaceReplayManifestRefs)
    warnings: list[WorkspaceReplayWarning] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source_mode(self) -> "WorkspaceReplayProduct":
        if self.source_mode == "resolved_scene" and self.imported_artifact is not None:
            raise ValueError("resolved_scene workspace replay must not set imported_artifact")
        if self.source_mode == "imported_artifact" and self.imported_artifact is None:
            raise ValueError("imported_artifact workspace replay requires imported_artifact")
        if self.source_mode == "imported_artifact" and not self.warnings:
            raise ValueError("imported_artifact workspace replay requires explicit warnings")
        return self


class WorkspaceReplayRequiredSelector(WorkspaceReplayModel):
    """Server-resolved selector required by authored workspace geometry."""

    anchor_id: str
    selector: StudioSelectorRef
    label: Optional[str] = None
    retention_id: Optional[str] = None
    value_schema: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResolvedWorkspaceReplayScene(WorkspaceReplayModel):
    """Narrow scene adapter shape consumed by workspace replay retention compile."""

    required_selectors: list[WorkspaceReplayRequiredSelector] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def compile_workspace_replay_retention(
    scene: ResolvedWorkspaceReplayScene | Mapping[str, Any] | Iterable[Any],
) -> list[RetainedObservableSpec]:
    """Compile resolved workspace replay selectors into trajectory retention specs."""

    required_selectors = _required_selectors(scene)
    observables: list[RetainedObservableSpec] = []
    for index, required in enumerate(required_selectors):
        target = _selector_ref_to_retention_target(
            required.selector,
            path=f"/required_selectors/{index}/selector",
        )
        observables.append(
            RetainedObservableSpec(
                id=required.retention_id or f"workspace_replay:{required.anchor_id}",
                label=required.label or required.anchor_id,
                target=target,
                retention=RetentionPolicySpec(
                    mode="trajectory",
                    reason="workspace_replay",
                    metadata={
                        "anchor_id": required.anchor_id,
                        **required.metadata,
                    },
                ),
                value_schema=required.value_schema,
                metadata={
                    "workspace_replay": True,
                    "anchor_id": required.anchor_id,
                    "selector": required.selector.model_dump(mode="json", exclude_none=True),
                },
            )
        )
    return observables


def workspace_replay_metadata() -> dict[str, str]:
    """Return schema metadata for a workspace replay artifact ref."""

    return {
        "schema_id": WORKSPACE_REPLAY_SCHEMA_ID,
        "schema_version": WORKSPACE_REPLAY_SCHEMA_VERSION,
        "product_kind": "workspace_replay",
    }


def imported_npz_workspace_replay_product(
    artifact: StudioArtifactRef,
    *,
    missing_metadata: Iterable[str],
    metadata: Mapping[str, Any] | None = None,
) -> WorkspaceReplayProduct:
    """Create the explicit imported-artifact downgrade product for legacy NPZ data."""

    missing = tuple(missing_metadata)
    warnings = [
        WorkspaceReplayWarning(
            code="npz_browser_downgrade",
            message=(
                "Imported NPZ trajectory data is usable for browsing but is not "
                "authoritative workspace replay data."
            ),
            path="/imported_artifact",
        )
    ]
    warnings.extend(
        WorkspaceReplayWarning(
            code=_warning_code_for_missing_metadata(item),
            message=f"Imported NPZ trajectory data is missing {item}.",
            path=f"/imported_artifact/missing_metadata/{index}",
            metadata={"field": item},
        )
        for index, item in enumerate(missing)
    )
    return WorkspaceReplayProduct(
        source_mode="imported_artifact",
        imported_artifact=WorkspaceReplayImportedArtifact(
            artifact=artifact,
            missing_metadata=list(missing),
        ),
        warnings=warnings,
        metadata=dict(metadata or {}),
    )


def _required_selectors(
    scene: ResolvedWorkspaceReplayScene | Mapping[str, Any] | Iterable[Any],
) -> list[WorkspaceReplayRequiredSelector]:
    if isinstance(scene, ResolvedWorkspaceReplayScene):
        return list(scene.required_selectors)
    if isinstance(scene, Mapping):
        raw = scene.get("required_selectors")
        if raw is None:
            raise ValueError("workspace replay scene is missing required_selectors")
        values = raw
    else:
        values = scene
    return [
        item
        if isinstance(item, WorkspaceReplayRequiredSelector)
        else WorkspaceReplayRequiredSelector.model_validate(item)
        for item in values
    ]


def _selector_ref_to_retention_target(
    selector: StudioSelectorRef,
    *,
    path: str,
) -> RetainedObservableTargetSpec:
    kind = _retention_kind_for_selector(selector.namespace, path=path)
    return RetainedObservableTargetSpec(
        kind=kind,
        selector=selector.compact,
        node_id=selector.target_id if kind == "port" else None,
        edge_id=selector.target_id if kind in {"edge", "recurrent_carry"} else None,
        port=_port_from_compact(selector.compact) if kind == "port" else None,
        path=selector.path,
        metadata={
            "studio_selector": selector.model_dump(mode="json", exclude_none=True),
            "source": "workspace_replay",
        },
    )


def _retention_kind_for_selector(namespace: str, *, path: str) -> str:
    direct = {
        "graph_port": "port",
        "graph_edge": "edge",
        "graph_output": "graph_output",
        "recurrent_carry": "recurrent_carry",
        "state_path": "state_path",
        "task_data": "task_data",
    }
    if namespace in direct:
        return direct[namespace]
    raise ValueError(
        "workspace replay required selector must be server-resolved to a retained "
        f"observable namespace at {path}; got unresolved namespace {namespace!r}"
    )


def _port_from_compact(compact: str) -> Optional[str]:
    if ":" not in compact:
        return None
    suffix = compact.split(":", 1)[1]
    if "." not in suffix:
        return None
    return suffix.rsplit(".", 1)[1] or None


def _warning_code_for_missing_metadata(field: str) -> WorkspaceReplayWarningCode:
    if "trial" in field:
        return "missing_trial_metadata"
    if "manifest" in field or "checkpoint" in field or "environment" in field:
        return "missing_manifest_refs"
    if "anchor" in field or "selector" in field:
        return "missing_anchor_resolution"
    if "overlay" in field:
        return "missing_overlay_metadata"
    return "missing_trial_metadata"
