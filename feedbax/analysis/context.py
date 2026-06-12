"""Context objects for manifest-canonical analysis execution."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from jaxtyping import PyTree
from sqlalchemy.orm import Session

from feedbax.database import EvaluationRecord, ModelRecord, add_evaluation_figure
from feedbax.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ArtifactRef,
    EntrypointRef,
    ManifestStatus,
    ParentRef,
    Provenance,
    analysis_results_cache_dir,
    analysis_run_manifest_id,
    collect_git_provenance,
    default_manifest_root,
    safe_manifest_key,
    spec_payload,
    store_artifact,
    store_json_artifact,
    write_manifest,
)
from feedbax.plot_utils import savefig
from jax_cookbook import arrays_to_lists


_MEDIA_TYPES = {
    "html": "text/html",
    "json": "application/json",
    "svg": "image/svg+xml",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "pdf": "application/pdf",
    "npz": "application/x-npz",
}

_FIGURE_ROUTING_KEY = "figure_routing"
_FIGURE_ROUTING_REQUIRED_KEYS = {"package", "experiment", "topic"}


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "figure"


@dataclass(frozen=True)
class AnalysisArtifactFile:
    """Description of one file member in a logical analysis artifact group."""

    path: Path | str
    role: str
    logical_name: str | None = None
    media_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    group_role: str | None = None


@dataclass
class AnalysisRunContext:
    """Execution context for writing analysis outputs without requiring Studio DB state."""

    spec: AnalysisRunSpec
    root: Path | str | None = None
    db_session: Session | None = None
    eval_info: EvaluationRecord | None = None
    model_info: PyTree[ModelRecord] | None = None
    fig_dump_path: Path | str | None = None
    fig_dump_formats: Sequence[str] = ("html",)
    provenance: Provenance | None = None
    issues: list[str] | None = None
    metadata: dict[str, Any] | None = None
    index_manifest: bool = True
    _artifacts: list[ArtifactRef] = field(default_factory=list, init=False)
    _manifest_path: Path | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root) if self.root is not None else default_manifest_root()
        if self.fig_dump_path is not None:
            self.fig_dump_path = Path(self.fig_dump_path)
        if (self.db_session is None) != (self.eval_info is None):
            raise ValueError("db_session and eval_info must be provided together")

    @property
    def manifest_id(self) -> str:
        """Return the deterministic analysis-run manifest identifier."""
        return analysis_run_manifest_id(self.spec)

    @property
    def root_path(self) -> Path:
        return Path(self.root)

    @property
    def results_cache_dir(self) -> Path:
        """Return this run's manifest-root result-cache directory."""
        return analysis_results_cache_dir(self.manifest_id, root=self.root_path)

    @property
    def manifest_path(self) -> Path | None:
        """Return the most recent manifest path written by :meth:`finalize`."""
        return self._manifest_path

    @property
    def artifacts(self) -> tuple[ArtifactRef, ...]:
        """Return artifacts recorded so far."""
        return tuple(self._artifacts)

    def record_artifact(
        self,
        path: Path | str,
        *,
        role: str,
        logical_name: str | None = None,
        media_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        group_id: str | None = None,
        group_role: str | None = None,
        group_metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Persist a non-figure artifact file and record it on this analysis run."""
        artifact_path = Path(path)
        artifact = store_artifact(
            artifact_path,
            root=self.root_path,
            role=role,
            logical_name=logical_name or artifact_path.name,
            media_type=media_type or _MEDIA_TYPES.get(
                artifact_path.suffix.lstrip(".").lower(),
                "application/octet-stream",
            ),
            metadata=self._artifact_metadata(
                metadata,
                group_id=group_id,
                group_role=group_role,
                group_metadata=group_metadata,
            ),
        )
        self.record_artifact_refs([artifact])
        return artifact

    def record_json_artifact(
        self,
        value: Any,
        *,
        role: str,
        logical_name: str,
        metadata: dict[str, Any] | None = None,
        group_id: str | None = None,
        group_role: str | None = None,
        group_metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Persist stable JSON payload and record it on this analysis run."""
        artifact = store_json_artifact(
            arrays_to_lists(value),
            root=self.root_path,
            role=role,
            logical_name=logical_name,
            metadata=self._artifact_metadata(
                metadata,
                group_id=group_id,
                group_role=group_role,
                group_metadata=group_metadata,
            ),
        )
        self.record_artifact_refs([artifact])
        return artifact

    def record_artifact_group(
        self,
        *,
        group_id: str,
        members: Sequence[AnalysisArtifactFile],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[ArtifactRef, ...]:
        """Record a logical multi-file analysis artifact group.

        Feedbax records group/member identity and opaque per-file metadata only; it does
        not interpret downstream domain schemas.
        """
        artifacts = [
            self.record_artifact(
                member.path,
                role=member.role,
                logical_name=member.logical_name,
                media_type=member.media_type,
                metadata=member.metadata,
                group_id=group_id,
                group_role=member.group_role,
                group_metadata=metadata,
            )
            for member in members
        ]
        return tuple(artifacts)

    def record_artifact_refs(self, artifacts: Sequence[ArtifactRef]) -> tuple[ArtifactRef, ...]:
        """Record already-materialized artifact refs, preserving insertion order."""
        existing = {self._artifact_key(artifact) for artifact in self._artifacts}
        added = []
        for artifact in artifacts:
            key = self._artifact_key(artifact)
            if key in existing:
                continue
            self._artifacts.append(artifact)
            existing.add(key)
            added.append(artifact)
        return tuple(added)

    def record_artifact_refs_from_value(self, value: Any) -> tuple[ArtifactRef, ...]:
        """Find and record ``ArtifactRef`` objects embedded in a result payload."""
        return self.record_artifact_refs(tuple(self._iter_artifact_refs(value)))

    def record_figure(
        self,
        *,
        fig: go.Figure,
        analysis_name: str,
        analysis_label: str | None,
        ordinal: int,
        params: dict[str, Any],
        dump_path: Path | str | None = None,
        dump_formats: Sequence[str] | None = None,
    ) -> list[ArtifactRef]:
        """Persist one figure and record it as an ``AnalysisRunManifest`` artifact."""
        if self.db_session is not None and self.eval_info is not None:
            add_evaluation_figure(
                self.db_session,
                self.eval_info,
                fig,
                analysis_name,
                model_records=self.model_info,
                **params,
            )

        formats = tuple(dump_formats or self.fig_dump_formats)
        figure_dir = self._figure_dir(dump_path)
        figure_dir.mkdir(parents=True, exist_ok=True)
        filename = self._figure_filename(analysis_name, analysis_label, ordinal)
        savefig(fig, filename, figure_dir, formats, metadata=params)
        routing_projection = self._route_figure_projection(
            fig=fig,
            analysis_name=analysis_name,
            analysis_label=analysis_label,
            ordinal=ordinal,
            params=params,
            filename=filename,
        )

        artifacts = []
        safe_label = _safe_name(analysis_label or analysis_name)
        for ext in formats:
            path = figure_dir / f"{filename}.{ext}"
            if not path.exists():
                continue
            metadata = {
                "analysis_name": analysis_name,
                "analysis_label": analysis_label,
                "format": ext,
                "ordinal": ordinal,
                "params": arrays_to_lists(params),
            }
            if routing_projection is not None:
                metadata[_FIGURE_ROUTING_KEY] = routing_projection
            artifact = store_artifact(
                path,
                root=self.root_path,
                role="figure",
                logical_name=f"{safe_label}/{path.name}",
                media_type=_MEDIA_TYPES.get(ext, "application/octet-stream"),
                metadata=metadata,
            )
            artifacts.append(artifact)
        return list(self.record_artifact_refs(artifacts))

    def finalize(
        self,
        *,
        status: ManifestStatus = "completed",
        summary_metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[AnalysisRunManifest, Path]:
        """Write the completed analysis manifest and optional SQLite manifest index row."""
        provenance = self._provenance()
        manifest = AnalysisRunManifest(
            id=self.manifest_id,
            status=status,
            analysis_spec=spec_payload(
                "AnalysisRunSpec",
                self.spec.model_dump(mode="json", exclude_none=True),
            ),
            inputs=list(self.spec.inputs),
            summary_metrics={
                "artifact_count": len(self._artifacts),
                "figure_count": sum(artifact.role == "figure" for artifact in self._artifacts),
                **(summary_metrics or {}),
            },
            provenance=provenance,
            artifacts=list(self._artifacts),
            metadata={
                **(self.metadata or {}),
                **(metadata or {}),
            },
        )
        path = write_manifest(manifest, root=self.root_path, index=self.index_manifest)
        self._manifest_path = path
        return manifest, path

    def _figure_dir(self, dump_path: Path | str | None) -> Path:
        if dump_path is not None:
            return Path(dump_path)
        if self.fig_dump_path is not None:
            return Path(self.fig_dump_path)
        return self.root_path / "outputs" / "analysis_runs" / safe_manifest_key(self.manifest_id)

    def _figure_filename(
        self,
        analysis_name: str,
        analysis_label: str | None,
        ordinal: int,
    ) -> str:
        label = _safe_name(analysis_label) if analysis_label is not None else _safe_name(analysis_name)
        return f"{label}_{_safe_name(analysis_name)}_{ordinal}"

    def _route_figure_projection(
        self,
        *,
        fig: go.Figure,
        analysis_name: str,
        analysis_label: str | None,
        ordinal: int,
        params: dict[str, Any],
        filename: str,
    ) -> dict[str, Any] | None:
        routing = self._figure_routing(params)
        if routing is None:
            return None

        from feedbax.plot.io import save_figure as save_routed_figure  # noqa: PLC0415

        figure_spec = self._figure_routing_spec(
            routing,
            analysis_name=analysis_name,
            analysis_label=analysis_label,
            ordinal=ordinal,
            params=params,
            filename=filename,
        )
        result = save_routed_figure(
            fig,
            figure_spec,
            package=str(routing["package"]),
            experiment=str(routing["experiment"]),
            topic=str(routing["topic"]),
            extra_packages=self._figure_routing_extra_packages(routing),
        )
        return {
            key: str(value) if value is not None else None
            for key, value in result.items()
        }

    def _figure_routing(self, params: dict[str, Any]) -> dict[str, Any] | None:
        for candidate in self._figure_routing_candidates(params):
            if candidate is None:
                continue
            if not isinstance(candidate, Mapping):
                raise TypeError(f"{_FIGURE_ROUTING_KEY} must be a mapping")
            routing = dict(candidate)
            missing = _FIGURE_ROUTING_REQUIRED_KEYS - routing.keys()
            if missing:
                raise ValueError(
                    f"{_FIGURE_ROUTING_KEY} is missing required key(s): {sorted(missing)!r}"
                )
            return routing
        return None

    def _figure_routing_candidates(self, params: dict[str, Any]):
        yield params.get(_FIGURE_ROUTING_KEY)
        presentation = params.get("presentation")
        if isinstance(presentation, Mapping):
            yield presentation.get(_FIGURE_ROUTING_KEY)

        metadata = self.metadata or {}
        yield metadata.get(_FIGURE_ROUTING_KEY)
        bundle = metadata.get("bundle")
        if isinstance(bundle, Mapping):
            yield bundle.get(_FIGURE_ROUTING_KEY)
            bundle_metadata = bundle.get("metadata")
            if isinstance(bundle_metadata, Mapping):
                yield bundle_metadata.get(_FIGURE_ROUTING_KEY)

    def _figure_routing_spec(
        self,
        routing: Mapping[str, Any],
        *,
        analysis_name: str,
        analysis_label: str | None,
        ordinal: int,
        params: dict[str, Any],
        filename: str,
    ) -> dict[str, Any]:
        figure_spec = dict(routing.get("spec") or {})
        figure_spec.setdefault("analysis", {})
        if isinstance(figure_spec["analysis"], Mapping):
            figure_spec["analysis"] = {
                **dict(figure_spec["analysis"]),
                "manifest_id": self.manifest_id,
                "analysis_type": self.spec.analysis_type,
                "analysis_name": analysis_name,
                "analysis_label": analysis_label,
                "ordinal": ordinal,
                "filename": filename,
            }
        figure_spec.setdefault("plot_kwargs", {})
        if isinstance(figure_spec["plot_kwargs"], Mapping):
            figure_spec["plot_kwargs"] = {
                **dict(figure_spec["plot_kwargs"]),
                "params": arrays_to_lists(params),
            }
        return arrays_to_lists(figure_spec)

    def _figure_routing_extra_packages(self, routing: Mapping[str, Any]) -> list[str] | None:
        extra_packages = routing.get("extra_packages")
        if extra_packages is None:
            return None
        if not isinstance(extra_packages, Sequence) or isinstance(extra_packages, (str, bytes)):
            raise TypeError(f"{_FIGURE_ROUTING_KEY}.extra_packages must be a sequence")
        return [str(package) for package in extra_packages]

    def _provenance(self) -> Provenance:
        provenance = (
            self.provenance.model_copy(deep=True)
            if self.provenance is not None
            else collect_git_provenance()
        )
        provenance.parents = list(self.spec.inputs)
        if self.issues:
            provenance.issues.extend(issue for issue in self.issues if issue not in provenance.issues)
        if provenance.entrypoint is None:
            provenance.entrypoint = EntrypointRef(
                kind="feedbax-analysis-context",
                name=self.spec.analysis_type,
            )
        return provenance

    def _artifact_metadata(
        self,
        metadata: dict[str, Any] | None,
        *,
        group_id: str | None = None,
        group_role: str | None = None,
        group_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact_metadata = arrays_to_lists(dict(metadata or {}))
        if group_id is not None:
            artifact_group: dict[str, Any] = {"id": group_id}
            if group_role is not None:
                artifact_group["member_role"] = group_role
            if group_metadata is not None:
                artifact_group["metadata"] = arrays_to_lists(group_metadata)
            artifact_metadata["artifact_group"] = artifact_group
        return artifact_metadata

    def _artifact_key(self, artifact: ArtifactRef) -> tuple[Any, ...]:
        return (
            artifact.role,
            artifact.logical_name,
            artifact.artifact_id,
            artifact.sha256,
            artifact.uri,
        )

    def _iter_artifact_refs(self, value: Any):
        if isinstance(value, ArtifactRef):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from self._iter_artifact_refs(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                yield from self._iter_artifact_refs(item)


def parent_ref_from_evaluation_manifest(
    manifest_id: str,
    *,
    uri: str | None = None,
    role: str = "evaluation_run",
    metadata: dict[str, Any] | None = None,
) -> ParentRef:
    """Build the canonical parent ref for analysis stages consuming an eval run."""
    return ParentRef(
        kind="EvaluationRunManifest",
        id=manifest_id,
        role=role,
        uri=uri,
        metadata=metadata or {},
    )
