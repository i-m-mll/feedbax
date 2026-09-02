"""Context objects for manifest-canonical analysis execution."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
from jaxtyping import PyTree
from sqlalchemy.orm import Session

from feedbax.contracts.base import (
    ArtifactRef,
    EntrypointRef,
    ParentRef,
    Provenance,
    authenticated_manifest_ref_profile,
    collect_git_provenance,
    default_manifest_root,
)
from feedbax.contracts.artifact_store import (
    media_type_for_extension,
    store_artifact,
)
from feedbax.contracts.manifest import (
    AnalysisDataProduct,
    AnalysisEvaluationStateResolutionDiagnostic,
    AnalysisEvaluationStateSource,
    AnalysisRunManifest,
    AnalysisRunSpec,
    DataProductParentRef,
    ManifestStatus,
    RegenerationSpec,
    SpecPayload,
    analysis_results_cache_dir,
    analysis_run_manifest_id,
    safe_manifest_key,
    spec_payload,
    write_manifest,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from jax_cookbook import arrays_to_lists

if TYPE_CHECKING:
    from feedbax.persistence.database import EvaluationRecord, ModelRecord
    from feedbax.plugins.registry import ExperimentRegistry


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
    experiment_registry: ExperimentRegistry | None = None
    index_manifest: bool = True
    _artifacts: list[ArtifactRef] = field(default_factory=list, init=False)
    _regeneration_specs: list[SpecPayload | ParentRef | ArtifactRef] = field(
        default_factory=list,
        init=False,
    )
    _manifest_path: Path | None = field(default=None, init=False)
    _evaluation_state_sources: list[AnalysisEvaluationStateSource] = field(
        default_factory=list,
        init=False,
    )
    _evaluation_state_resolution_diagnostics: list[AnalysisEvaluationStateResolutionDiagnostic] = (
        field(default_factory=list, init=False)
    )
    _produced_data: list[AnalysisDataProduct] = field(default_factory=list, init=False)

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

    @property
    def regeneration_specs(self) -> tuple[SpecPayload | ParentRef | ArtifactRef, ...]:
        """Return regeneration specs recorded so far."""
        return tuple(self._regeneration_specs)

    @property
    def produced_data(self) -> tuple[AnalysisDataProduct, ...]:
        """Return typed data products recorded so far."""
        return tuple(self._produced_data)

    def record_evaluation_state_sources(
        self,
        sources: Iterable[AnalysisEvaluationStateSource],
    ) -> None:
        """Record truthful suppliers for evaluation states consumed by this run."""
        self._evaluation_state_sources.extend(sources)

    def record_evaluation_state_resolution_diagnostic(
        self,
        diagnostic: AnalysisEvaluationStateResolutionDiagnostic,
    ) -> None:
        """Record one fail-closed evaluation-state resolution diagnostic."""
        self._evaluation_state_resolution_diagnostics.append(diagnostic)

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
            media_type=media_type or media_type_for_extension(artifact_path.suffix),
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
        artifact = self._store_json_artifact(
            value,
            role=role,
            logical_name=logical_name,
            metadata=metadata,
            group_id=group_id,
            group_role=group_role,
            group_metadata=group_metadata,
        )
        self.record_artifact_refs([artifact])
        return artifact

    def record_data_product(
        self,
        payload: Any,
        *,
        product_schema_id: str,
        product_schema_version: str,
        role: str,
        logical_name: str,
        label: str | None = None,
        artifact_role: str | None = None,
        artifact_logical_name: str | None = None,
        artifact_metadata: dict[str, Any] | None = None,
        producer_manifest_hash: str | None = None,
        parent_manifests: Sequence[DataProductParentRef] | None = None,
        checkpoint_policy: dict[str, Any] | None = None,
        rollout_policy: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        descriptor_basis_hash: str | None = None,
        materialization: dict[str, Any] | None = None,
        regeneration: Sequence[RegenerationSpec | ParentRef | ArtifactRef] = (),
        metadata: dict[str, Any] | None = None,
    ) -> AnalysisDataProduct:
        """Store and record one typed, content-addressed JSON data product.

        The product payload is retained by the analysis root's immutable artifact
        provider. Product identity and output-binding collisions fail before a
        second product is recorded.
        """
        parents = (
            list(parent_manifests)
            if parent_manifests is not None
            else [self._data_product_parent(parent) for parent in self.spec.inputs]
        )
        product_fields = {
            "product_schema_id": product_schema_id,
            "product_schema_version": product_schema_version,
            "role": role,
            "logical_name": logical_name,
            "label": label,
            "producer_manifest_id": self.manifest_id,
            "producer_manifest_hash": producer_manifest_hash,
            "parent_manifests": parents,
            "checkpoint_policy": dict(checkpoint_policy or {}),
            "rollout_policy": dict(rollout_policy or {}),
            "parameters": arrays_to_lists(dict(parameters or {})),
            "descriptor_basis_hash": descriptor_basis_hash,
            "materialization": arrays_to_lists(dict(materialization or {})),
            "regeneration": list(regeneration),
            "metadata": arrays_to_lists(dict(metadata or {})),
        }
        candidate = AnalysisDataProduct.model_validate(
            {
                **product_fields,
                "artifacts": [],
            }
        )
        self._reject_data_product_collision(candidate)

        artifact = self._store_json_artifact(
            payload,
            role=artifact_role or role,
            logical_name=artifact_logical_name or f"{logical_name}.json",
            metadata=artifact_metadata,
        )
        product = AnalysisDataProduct.model_validate(
            {
                **product_fields,
                "artifacts": [artifact],
            }
        )
        self._reject_data_product_collision(product)
        self._artifacts.append(artifact)
        self._produced_data.append(product)
        return product

    def _store_json_artifact(
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
        """Store stable JSON bytes without mutating the context record."""
        data = json.dumps(arrays_to_lists(value), indent=2, sort_keys=True).encode() + b"\n"
        provider = ImmutableArtifactBlobProvider(self.root_path.absolute())
        artifact = provider.store_bytes(
            data,
            role=role,
            logical_name=logical_name,
            media_type="application/json",
            metadata=self._artifact_metadata(
                metadata,
                group_id=group_id,
                group_role=group_role,
                group_metadata=group_metadata,
            ),
        )
        artifact = artifact.model_copy(
            update={
                "metadata": {
                    **artifact.metadata,
                    "local_relative_path": str(provider.canonical_relative_path(artifact)),
                }
            }
        )
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

    def record_regeneration_specs(
        self,
        specs: Sequence[RegenerationSpec | SpecPayload | ParentRef | ArtifactRef],
    ) -> tuple[SpecPayload | ParentRef | ArtifactRef, ...]:
        """Record replay/provenance specs for this analysis run.

        Feedbax owns only the manifest custody shape here. Downstream materializers
        may provide inline ``RegenerationSpec`` values, already-packaged
        ``SpecPayload`` objects, or external refs to durable regeneration records.
        """
        existing = {self._regeneration_spec_key(spec) for spec in self._regeneration_specs}
        added = []
        for raw_spec in specs:
            spec = self._coerce_regeneration_spec(raw_spec)
            key = self._regeneration_spec_key(spec)
            if key in existing:
                continue
            self._regeneration_specs.append(spec)
            existing.add(key)
            added.append(spec)
        return tuple(added)

    def record_regeneration_spec(
        self,
        spec: RegenerationSpec | SpecPayload | ParentRef | ArtifactRef,
    ) -> SpecPayload | ParentRef | ArtifactRef:
        """Record one replay/provenance spec for this analysis run."""
        recorded = self.record_regeneration_specs([spec])
        return recorded[0] if recorded else self._coerce_regeneration_spec(spec)

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
        from feedbax.plot.lifecycle import close_figure
        from feedbax.plot.utils import savefig

        if self.db_session is not None and self.eval_info is not None:
            from feedbax.persistence.database import add_evaluation_figure

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
        try:
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
                    media_type=media_type_for_extension(ext),
                    metadata=metadata,
                )
                artifacts.append(artifact)
            return list(self.record_artifact_refs(artifacts))
        finally:
            close_figure(fig)

    def finalize(
        self,
        *,
        status: ManifestStatus = "completed",
        summary_metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[AnalysisRunManifest, Path]:
        """Write the completed analysis manifest and optional SQLite manifest index row."""
        self._validate_produced_data_custody()
        provenance = self._provenance()
        manifest = AnalysisRunManifest(
            id=self.manifest_id,
            status=status,
            analysis_spec=spec_payload(
                "AnalysisRunSpec",
                self.spec.model_dump(mode="json", exclude_none=True),
            ),
            inputs=list(self.spec.inputs),
            evaluation_state_sources=list(self._evaluation_state_sources),
            evaluation_state_resolution_diagnostics=list(
                self._evaluation_state_resolution_diagnostics
            ),
            summary_metrics={
                "artifact_count": len(self._artifacts),
                "figure_count": sum(artifact.role == "figure" for artifact in self._artifacts),
                **(summary_metrics or {}),
            },
            provenance=provenance,
            artifacts=list(self._artifacts),
            regeneration_specs=list(self._regeneration_specs),
            produced_data=list(self._produced_data),
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
        label = (
            _safe_name(analysis_label) if analysis_label is not None else _safe_name(analysis_name)
        )
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
        if self.experiment_registry is None:
            raise ValueError("routed analysis figures require an injected experiment registry")

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
            registry=self.experiment_registry,
            extra_packages=self._figure_routing_extra_packages(routing),
        )
        return {key: str(value) if value is not None else None for key, value in result.items()}

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
            provenance.issues.extend(
                issue for issue in self.issues if issue not in provenance.issues
            )
        if provenance.entrypoint is None:
            provenance.entrypoint = EntrypointRef(
                kind="feedbax-analysis-context",
                name=self.spec.analysis_type,
            )
        return provenance

    def _data_product_parent(self, parent: ParentRef) -> DataProductParentRef:
        profile = authenticated_manifest_ref_profile(parent)
        return DataProductParentRef(
            kind=parent.kind,
            id=parent.id,
            role=parent.role,
            manifest_hash=profile[0] if profile is not None else None,
            uri=parent.uri,
            metadata=dict(parent.metadata),
        )

    def _reject_data_product_collision(self, product: AnalysisDataProduct) -> None:
        for existing in self._produced_data:
            if existing.product_identity_hash == product.product_identity_hash:
                raise ValueError(
                    f"duplicate AnalysisDataProduct identity: {product.product_identity_hash}"
                )
            if (
                existing.role,
                existing.product_schema_id,
                existing.product_schema_version,
            ) == (
                product.role,
                product.product_schema_id,
                product.product_schema_version,
            ):
                raise ValueError(
                    "duplicate AnalysisDataProduct role/schema binding: "
                    f"role={product.role!r}, "
                    f"product_schema_id={product.product_schema_id!r}, "
                    f"product_schema_version={product.product_schema_version!r}"
                )

    def _validate_produced_data_custody(self) -> None:
        provider = ImmutableArtifactBlobProvider(self.root_path.absolute())
        for product in self._produced_data:
            validated = AnalysisDataProduct.model_validate(
                product.model_dump(mode="python", exclude_none=True)
            )
            if validated.product_identity_hash != product.product_identity_hash:
                raise ValueError(
                    "recorded AnalysisDataProduct identity changed before finalization: "
                    f"role={product.role!r}"
                )
            if not product.artifacts:
                raise ValueError(
                    "recorded AnalysisDataProduct has no materialized payload artifact: "
                    f"role={product.role!r}"
                )
            for artifact in product.artifacts:
                provider.get_bytes(artifact)

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

    def _coerce_regeneration_spec(
        self,
        spec: RegenerationSpec | SpecPayload | ParentRef | ArtifactRef,
    ) -> SpecPayload | ParentRef | ArtifactRef:
        if isinstance(spec, RegenerationSpec):
            return spec_payload("RegenerationSpec", spec.model_dump(mode="json"))
        return spec

    def _regeneration_spec_key(
        self,
        spec: SpecPayload | ParentRef | ArtifactRef,
    ) -> tuple[Any, ...]:
        if isinstance(spec, SpecPayload):
            return ("SpecPayload", spec.kind, spec.sha256, spec.ref)
        if isinstance(spec, ParentRef):
            return ("ParentRef", spec.kind, spec.id, spec.uri, spec.role)
        return (
            "ArtifactRef",
            spec.role,
            spec.logical_name,
            spec.artifact_id,
            spec.sha256,
            spec.uri,
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
