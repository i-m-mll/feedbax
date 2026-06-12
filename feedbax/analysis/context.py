"""Context objects for manifest-canonical analysis execution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

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
}


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "figure"


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

        artifacts = []
        safe_label = _safe_name(analysis_label or analysis_name)
        for ext in formats:
            path = figure_dir / f"{filename}.{ext}"
            if not path.exists():
                continue
            artifact = store_artifact(
                path,
                root=self.root_path,
                role="figure",
                logical_name=f"{safe_label}/{path.name}",
                media_type=_MEDIA_TYPES.get(ext, "application/octet-stream"),
                metadata={
                    "analysis_name": analysis_name,
                    "analysis_label": analysis_label,
                    "format": ext,
                    "ordinal": ordinal,
                    "params": arrays_to_lists(params),
                },
            )
            artifacts.append(artifact)
        self._artifacts.extend(artifacts)
        return artifacts

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
                "figure_count": len(self._artifacts),
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

    def _provenance(self) -> Provenance:
        provenance = self.provenance or collect_git_provenance()
        provenance.parents = list(self.spec.inputs)
        if self.issues:
            provenance.issues.extend(issue for issue in self.issues if issue not in provenance.issues)
        if provenance.entrypoint is None:
            provenance.entrypoint = EntrypointRef(
                kind="feedbax-analysis-context",
                name=self.spec.analysis_type,
            )
        return provenance


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
