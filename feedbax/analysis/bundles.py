"""Human-authored analysis bundles over manifest-canonical run records."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from feedbax.config.yaml import get_yaml_loader
from feedbax.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    AnyManifest,
    EvaluationRunManifest,
    ParentRef,
    StrictModel,
    default_manifest_root,
    load_manifest,
)
from feedbax.manifest_index import iter_manifest_files
from feedbax.plugins import EXPERIMENT_REGISTRY
from feedbax.plugins.registry import ExperimentRegistry
from feedbax.analysis.specs import execute_analysis_run_spec

AnalysisBundleMode = Literal["per-run", "grouped"]


class ManifestPredicate(StrictModel):
    """Predicate selecting upstream run manifests for an analysis bundle."""

    manifest_kind: str = "EvaluationRunManifest"
    run_ids: list[str] = Field(default_factory=list)
    metadata_equals: dict[str, Any] = Field(default_factory=dict)
    params_equals: dict[str, Any] = Field(default_factory=dict)


class AnalysisSpecTemplate(StrictModel):
    """Template expanded into executable ``AnalysisRunSpec`` instances."""

    name: str
    mode: AnalysisBundleMode = "per-run"
    analysis_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    requested_outputs: list[str] = Field(default_factory=list)
    input_requirements: list[Any] = Field(default_factory=list)
    inputs: list[ParentRef] = Field(default_factory=list)


class AnalysisBundleSpec(StrictModel):
    """A named set of analysis spec templates and manifest applicability rules."""

    name: str
    description: str | None = None
    predicate: ManifestPredicate = Field(default_factory=ManifestPredicate)
    templates: list[AnalysisSpecTemplate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class BundleExpansion:
    """One executable spec generated from a bundle template."""

    bundle_name: str
    template_name: str
    mode: AnalysisBundleMode
    matched_run_ids: tuple[str, ...]
    spec: AnalysisRunSpec


def _split_bundle_key(key: str, registry: ExperimentRegistry) -> tuple[str, str]:
    if "/" in key:
        package_name, bundle_name = key.split("/", 1)
        if not bundle_name:
            raise ValueError(f"Empty bundle name after package prefix {package_name!r}")
        registry.get_package_metadata(package_name)
        return package_name, bundle_name

    single = registry.single_package_name()
    if single is not None:
        return single, key

    matches: list[str] = []
    for package_name, metadata in registry._packages.items():
        resource_root = (
            f"{metadata.package_module.__name__}.{metadata.config_resource_root}.analysis_bundles"
        )
        try:
            if resources.files(resource_root).joinpath(f"{key}.yml").is_file():
                matches.append(package_name)
        except (FileNotFoundError, ModuleNotFoundError):
            continue

    if not matches:
        raise FileNotFoundError(
            f"Analysis bundle {key!r} not found under any registered package config resource root"
        )
    if len(matches) > 1:
        options = "', '".join(f"{package_name}/{key}" for package_name in matches)
        raise ValueError(f"Analysis bundle {key!r} is ambiguous. Use one of '{options}'.")
    return matches[0], key


def load_analysis_bundle(
    key: str,
    *,
    registry: ExperimentRegistry | None = None,
) -> AnalysisBundleSpec:
    """Load and pydantic-validate an analysis bundle YAML resource."""
    active_registry = registry or EXPERIMENT_REGISTRY
    package_name, bundle_name = _split_bundle_key(key, active_registry)
    metadata = active_registry.get_package_metadata(package_name)
    resource_root = (
        f"{metadata.package_module.__name__}.{metadata.config_resource_root}.analysis_bundles"
    )
    yaml = get_yaml_loader(typ="safe")
    try:
        resource = resources.files(resource_root).joinpath(f"{bundle_name}.yml")
        with resource.open("r", encoding="utf-8") as stream:
            data = yaml.load(stream) or {}
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise FileNotFoundError(
            f"Analysis bundle {bundle_name!r} not found under {resource_root}"
        ) from exc
    return AnalysisBundleSpec.model_validate(data)


def iter_candidate_manifests(
    root: Path | str | None = None,
    *,
    manifest_kind: str = "EvaluationRunManifest",
) -> list[AnyManifest]:
    """Load candidate run manifests of one kind from a manifest root."""
    root_path = Path(root) if root is not None else default_manifest_root()
    manifests: list[AnyManifest] = []
    for manifest_path in iter_manifest_files(root_path):
        manifest = load_manifest(manifest_path)
        if manifest.kind == manifest_kind:
            manifests.append(manifest)
    return manifests


def _get_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def _params_payload(manifest: AnyManifest) -> dict[str, Any]:
    if isinstance(manifest, EvaluationRunManifest):
        return dict(manifest.evaluation_spec.inline.get("params", {}))
    if isinstance(manifest, AnalysisRunManifest):
        return dict(manifest.analysis_spec.inline.get("params", {}))
    spec_payload = getattr(manifest, "training_spec", None)
    if spec_payload is not None:
        return dict(spec_payload.inline)
    return {}


def _equals_all(actual: Any, expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        try:
            actual_value = _get_path(actual, key)
        except (AttributeError, KeyError, TypeError):
            return False
        if actual_value != expected_value:
            return False
    return True


def predicate_matches_manifest(
    predicate: ManifestPredicate,
    manifest: AnyManifest,
    *,
    run_ids: set[str] | None = None,
) -> bool:
    """Return whether a manifest satisfies explicit IDs and equality predicates."""
    if manifest.kind != predicate.manifest_kind:
        return False
    allowed_ids = run_ids if run_ids is not None else set(predicate.run_ids)
    if allowed_ids and manifest.id not in allowed_ids:
        return False
    if not _equals_all(manifest.metadata, predicate.metadata_equals):
        return False
    if not _equals_all(_params_payload(manifest), predicate.params_equals):
        return False
    return True


def select_bundle_manifests(
    bundle: AnalysisBundleSpec,
    root: Path | str | None = None,
    *,
    run_ids: Iterable[str] | None = None,
) -> list[AnyManifest]:
    """Select manifests in a root that match a bundle predicate."""
    allowed_ids = set(run_ids) if run_ids is not None else None
    candidates = iter_candidate_manifests(root, manifest_kind=bundle.predicate.manifest_kind)
    return [
        manifest
        for manifest in candidates
        if predicate_matches_manifest(bundle.predicate, manifest, run_ids=allowed_ids)
    ]


def _parent_ref_for_manifest(manifest: AnyManifest) -> ParentRef:
    return ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role="evaluation_run" if manifest.kind == "EvaluationRunManifest" else "run_manifest",
    )


def _params_for_template(template: AnalysisSpecTemplate) -> dict[str, Any]:
    params = dict(template.params)
    if template.requested_outputs and "requested_outputs" not in params and "outputs" not in params:
        params["requested_outputs"] = list(template.requested_outputs)
    return params


def expand_analysis_bundle(
    bundle: AnalysisBundleSpec,
    matched_manifests: Sequence[AnyManifest],
) -> list[BundleExpansion]:
    """Expand bundle templates into executable analysis run specs."""
    if not bundle.templates:
        raise ValueError(f"Analysis bundle {bundle.name!r} has no templates")
    if not matched_manifests:
        return []

    expansions: list[BundleExpansion] = []
    for template in bundle.templates:
        if template.mode == "per-run":
            for manifest in matched_manifests:
                inputs = [*template.inputs, _parent_ref_for_manifest(manifest)]
                spec = AnalysisRunSpec(
                    analysis_type=template.analysis_type,
                    inputs=inputs,
                    input_requirements=template.input_requirements,
                    params=_params_for_template(template),
                )
                expansions.append(
                    BundleExpansion(
                        bundle_name=bundle.name,
                        template_name=template.name,
                        mode=template.mode,
                        matched_run_ids=(manifest.id,),
                        spec=spec,
                    )
                )
        else:
            inputs = [
                *template.inputs,
                *[_parent_ref_for_manifest(manifest) for manifest in matched_manifests],
            ]
            spec = AnalysisRunSpec(
                analysis_type=template.analysis_type,
                inputs=inputs,
                input_requirements=template.input_requirements,
                params=_params_for_template(template),
            )
            expansions.append(
                BundleExpansion(
                    bundle_name=bundle.name,
                    template_name=template.name,
                    mode=template.mode,
                    matched_run_ids=tuple(manifest.id for manifest in matched_manifests),
                    spec=spec,
                )
            )
    return expansions


def execute_analysis_bundle(
    bundle: AnalysisBundleSpec,
    *,
    root: Path | str | None = None,
    run_ids: Iterable[str] | None = None,
    issues: list[str] | None = None,
    fig_dump_path: Path | str | None = None,
    fig_dump_formats: Sequence[str] = ("html",),
) -> list[tuple[BundleExpansion, AnalysisRunManifest, Path]]:
    """Apply a bundle to a manifest root and execute all generated specs."""
    root_path = Path(root) if root is not None else default_manifest_root()
    matched_manifests = select_bundle_manifests(bundle, root_path, run_ids=run_ids)
    expansions = expand_analysis_bundle(bundle, matched_manifests)
    outputs: list[tuple[BundleExpansion, AnalysisRunManifest, Path]] = []
    for expansion in expansions:
        manifest, path = execute_analysis_run_spec(
            expansion.spec,
            root=root_path,
            issues=issues,
            metadata={
                "bundle": {
                    "name": expansion.bundle_name,
                    "template": expansion.template_name,
                    "mode": expansion.mode,
                    "matched_run_ids": list(expansion.matched_run_ids),
                }
            },
            fig_dump_path=fig_dump_path,
            fig_dump_formats=fig_dump_formats,
        )
        outputs.append((expansion, manifest, path))
    return outputs
