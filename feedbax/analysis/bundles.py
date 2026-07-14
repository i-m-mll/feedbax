"""Human-authored analysis bundles over manifest-canonical run records."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
import json
from pathlib import Path
import re
from typing import Any, Literal, TypeVar
from urllib.parse import unquote, urlsplit

from pydantic import Field, field_validator, model_validator

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.evaluation import execute_evaluation_run_spec
from feedbax.analysis.evaluation_inputs import resolve_evaluation_inputs
from feedbax.analysis.execution_context import (
    StagedArtifactProviderRootBinding,
    StagedCheckpointCustodyRootBinding,
    StagedExecutionContext,
    StagedParentExecutionLocation,
    resolve_staged_execution_context,
    with_staged_parent_execution_locations,
)
from feedbax.analysis.exact_parents import StagedExactParents
from feedbax.analysis.figures import FIGURE_RENDER_ROLE, execute_figure_spec
from feedbax.analysis.materialization import ContextMaterializer
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.analysis.reports import BUNDLE_SUMMARY_REPORT_TYPE, execute_report_spec
from feedbax.analysis.specs import AnalysisRecipeResult, execute_analysis_run_spec
from feedbax.config.yaml import get_yaml_loader
from feedbax.contracts.expressions import (
    ContextItem,
    Expr,
    ExpressionContext,
    canonical_expression_json,
    evaluate_expr,
)
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ArtifactRef,
    AnyManifest,
    EvaluationRunManifest,
    EvaluationRunSpec,
    FigureManifest,
    ParentRef,
    Provenance,
    RegenerationCommand,
    RegenerationSpec,
    ReportManifest,
    ReportSpec,
    SpecPayload,
    StrictModel,
    TrainingRunManifest,
    OverridePatch,
    canonical_json_bytes,
    collect_git_provenance,
    default_manifest_root,
    evaluation_run_manifest_id,
    load_manifest,
    safe_manifest_key,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.staged_execution import StagedExecutionDescriptor
from feedbax.contracts.run_matrix import apply_override_patches
from feedbax.contracts.selection import (
    ManifestIndexRow,
    ManifestPredicate,
    SelectionPreview,
    SelectionSpec,
    predicate_matches_row,
    preview_selection_spec,
    select_parent_refs,
)
from feedbax.persistence.manifest_index import (
    iter_indexed_manifest_paths_by_kind,
    iter_manifest_files,
)
from feedbax.plugins import EXPERIMENT_REGISTRY
from feedbax.plugins.registry import ExperimentRegistry

ANALYSIS_BUNDLE_SCHEMA_ID = "feedbax.spec.analysis_bundle"
ANALYSIS_BUNDLE_SCHEMA_VERSION_V2 = "feedbax.spec.analysis_bundle.v2"
ANALYSIS_BUNDLE_SCHEMA_VERSION = "feedbax.spec.analysis_bundle.v3"
ANALYSIS_BUNDLE_EXECUTION_SCHEMA_ID = "feedbax.manifest.analysis_bundle_execution"
ANALYSIS_BUNDLE_EXECUTION_SCHEMA_VERSION = "feedbax.manifest.analysis_bundle_execution.v1"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_IMMUTABLE_MANIFEST_URI_PATTERN = re.compile(r"artifact://sha256/([0-9a-f]{64})")
_EXACT_PARENT_REQUIRED_TEXT_METADATA = (
    "run_set_id",
    "row_id",
    "planned_run_id",
)
_EXACT_PARENT_REQUIRED_STATUS_METADATA = {
    "manifest_status": "completed",
    "registration_status": "completed",
    "conformance_overall": "pass",
}

AnalysisBundleMode = Literal["per-run", "grouped"]
BundleStageKind = Literal["evaluation", "analysis", "materialization", "figure", "report"]
BundleOutputStatus = Literal["materialized", "skipped", "missing", "not_applicable"]
BundleDryRunStageStatus = Literal["would_run", "would_skip", "missing", "not_applicable"]

_StageSpecT = TypeVar("_StageSpecT")


class AnalysisSpecTemplate(StrictModel):
    """Template expanded into executable ``AnalysisRunSpec`` instances."""

    name: str
    mode: AnalysisBundleMode = "per-run"
    analysis_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    requested_outputs: list[str] = Field(default_factory=list)
    input_requirements: list[Any] = Field(default_factory=list)
    inputs: list[ParentRef] = Field(default_factory=list)


class BundleStageOutputSpec(StrictModel):
    """Declared output role for one bundle stage."""

    role: str
    required: bool = True


class StageArtifactDependency(StrictModel):
    """Role-addressed artifact dependency on an earlier bundle stage."""

    stage: str
    role: str
    required: bool = True
    bind_as: str | None = None


class BundleParamsBase(StrictModel):
    """Shared, typed parameter envelope resolved by every staged bundle stage."""

    params: dict[str, Any] = Field(default_factory=dict)


class BundleStageSpec(StrictModel):
    """One ordered stage in a schema-bearing analysis bundle plan.

    Runtime ``run_condition`` gates execution: skipped means a predicate chose not
    to run this otherwise-valid stage; not_applicable remains reserved for
    structural inapplicability in this bundle configuration.
    """

    name: str
    kind: BundleStageKind
    mode: AnalysisBundleMode = "grouped"
    depends_on: list[str] = Field(default_factory=list)
    depends_on_roles: list[StageArtifactDependency] = Field(default_factory=list)
    include_bundle_inputs: bool = False
    evaluation_type: str | None = None
    analysis_type: str | None = None
    figure: FigureSpec | None = None
    report_type: str | None = None
    params_patches: list[OverridePatch] = Field(default_factory=list)
    local_params: dict[str, Any] | None = None
    states_custody: Literal["cache", "durable"] | None = None
    requested_outputs: list[str] = Field(default_factory=list)
    input_requirements: list[Any] = Field(default_factory=list)
    outputs: list[BundleStageOutputSpec] = Field(default_factory=list)
    skip_reason: str | None = None
    not_applicable_reason: str | None = None
    run_condition: Expr | None = None

    @model_validator(mode="after")
    def _validate_stage_payload(self) -> "BundleStageSpec":
        if self.local_params is not None and self.params_patches:
            raise ValueError(
                f"bundle stage {self.name!r} cannot combine local_params with params_patches"
            )
        no_static_status = self.skip_reason is None and self.not_applicable_reason is None
        if self.kind == "evaluation" and not self.evaluation_type and no_static_status:
            raise ValueError(f"evaluation bundle stage {self.name!r} requires evaluation_type")
        if self.kind == "analysis" and not self.analysis_type and no_static_status:
            raise ValueError(f"analysis bundle stage {self.name!r} requires analysis_type")
        if self.kind == "materialization" and not self.analysis_type and no_static_status:
            raise ValueError(f"materialization bundle stage {self.name!r} requires analysis_type")
        if self.kind == "figure" and self.figure is None and no_static_status:
            raise ValueError(f"figure bundle stage {self.name!r} requires figure")
        if self.kind == "report" and not self.report_type and no_static_status:
            raise ValueError(f"report bundle stage {self.name!r} requires report_type")
        if self.skip_reason and any(output.required for output in self.outputs):
            raise ValueError(f"bundle stage {self.name!r} cannot skip required outputs")
        if self.run_condition is not None and self.skip_reason is not None:
            raise ValueError(
                f"bundle stage {self.name!r} cannot combine static skip_reason with run_condition"
            )
        return self


class AnalysisBundleSpec(StrictModel):
    """A named set of analysis spec templates and manifest applicability rules."""

    schema_id: str = ANALYSIS_BUNDLE_SCHEMA_ID
    schema_version: str = ANALYSIS_BUNDLE_SCHEMA_VERSION
    name: str
    description: str | None = None
    predicate: ManifestPredicate = Field(
        default_factory=lambda: ManifestPredicate(manifest_kind="EvaluationRunManifest")
    )
    templates: list[AnalysisSpecTemplate] = Field(default_factory=list)
    params_base: BundleParamsBase = Field(default_factory=BundleParamsBase)
    stages: list[BundleStageSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schema_and_payload(self) -> "AnalysisBundleSpec":
        if self.schema_id != ANALYSIS_BUNDLE_SCHEMA_ID:
            raise ValueError(
                "unsupported AnalysisBundleSpec schema_id: "
                f"{self.schema_id!r}, expected {ANALYSIS_BUNDLE_SCHEMA_ID!r}"
            )
        if self.schema_version != ANALYSIS_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported AnalysisBundleSpec schema_version: "
                f"{self.schema_version!r}, expected {ANALYSIS_BUNDLE_SCHEMA_VERSION!r}"
            )
        if self.templates and self.stages:
            raise ValueError("AnalysisBundleSpec cannot mix simple templates and staged plan")
        seen: set[str] = set()
        stage_kinds: dict[str, BundleStageKind] = {}
        for stage in self.stages:
            if stage.name in seen:
                raise ValueError(f"AnalysisBundleSpec has duplicate stage name {stage.name!r}")
            for dependency in stage.depends_on:
                if dependency not in stage_kinds:
                    raise ValueError(
                        f"bundle stage {stage.name!r} depends_on entry for "
                        f"stage {dependency!r} must refer to an earlier stage"
                    )
                if stage.kind != "report" and stage_kinds[dependency] == "figure":
                    raise ValueError(
                        f"bundle stage {stage.name!r} of kind {stage.kind!r} cannot "
                        f"depend on figure stage {dependency!r}; figure stages are leaves"
                    )
            for dependency in stage.depends_on_roles:
                if dependency.stage not in seen:
                    raise ValueError(
                        f"bundle stage {stage.name!r} depends_on_roles entry for "
                        f"stage {dependency.stage!r} must refer to an earlier stage"
                    )
                if stage.kind != "report" and stage_kinds.get(dependency.stage) == "figure":
                    raise ValueError(
                        f"bundle stage {stage.name!r} of kind {stage.kind!r} cannot "
                        f"depend_on_roles from figure stage {dependency.stage!r}; "
                        "figure stages are leaves"
                    )
            seen.add(stage.name)
            stage_kinds[stage.name] = stage.kind
        return self

    @field_validator("predicate", mode="before")
    @classmethod
    def _default_bundle_predicate_manifest_kind(cls, value: Any) -> Any:
        if isinstance(value, dict) and "manifest_kind" not in value:
            return {"manifest_kind": "EvaluationRunManifest", **value}
        return value


@dataclass(frozen=True)
class BundleExpansion:
    """One executable spec generated from a bundle template."""

    bundle_name: str
    template_name: str
    mode: AnalysisBundleMode
    matched_run_ids: tuple[str, ...]
    spec: AnalysisRunSpec


@dataclass(frozen=True)
class StageMaterialization:
    """Manifest and artifact products emitted by one stage invocation."""

    manifest_ref: ParentRef | None
    artifacts: tuple[ArtifactRef, ...] = ()
    manifest_path: Path | None = None
    regeneration_spec: SpecPayload | None = None


@dataclass(frozen=True)
class ResolvedStageInputs:
    """Inputs bound for one stage before execution and condition evaluation."""

    parent_refs: tuple[ParentRef, ...] = ()
    artifact_refs_by_alias: dict[str, tuple[ArtifactRef, ...]] = field(default_factory=dict)


class BundleStageOutputRecord(StrictModel):
    """Observed status for one declared stage output role."""

    role: str
    required: bool = True
    status: BundleOutputStatus
    reason: str | None = None
    manifest_refs: list[ParentRef] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class BundleStageExecutionRecord(StrictModel):
    """Durable provenance for one executed bundle stage."""

    name: str
    kind: BundleStageKind
    status: BundleOutputStatus
    depends_on: list[str] = Field(default_factory=list)
    inputs: list[ParentRef] = Field(default_factory=list)
    manifest_refs: list[ParentRef] = Field(default_factory=list)
    artifact_groups: dict[str, list[ArtifactRef]] = Field(default_factory=dict)
    outputs: list[BundleStageOutputRecord] = Field(default_factory=list)
    regeneration_specs: list[SpecPayload] = Field(default_factory=list)
    reason: str | None = None


class StagedAnalysisBundleExecution(StrictModel):
    """Durable execution provenance for a staged analysis bundle plan."""

    schema_id: str = ANALYSIS_BUNDLE_EXECUTION_SCHEMA_ID
    schema_version: str = ANALYSIS_BUNDLE_EXECUTION_SCHEMA_VERSION
    bundle_name: str
    matched_run_ids: list[str] = Field(default_factory=list)
    stages: list[BundleStageExecutionRecord] = Field(default_factory=list)
    report_outputs: list[BundleStageOutputRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BundleMissingRoleRecord(StrictModel):
    """Required role dependency that is unavailable in a dry-run stage plan."""

    stage: str
    role: str
    required: bool = True
    bind_as: str | None = None
    reason: str


class BundleStageDryRunOutputRecord(StrictModel):
    """Predicted output-role status for one dry-run stage."""

    role: str
    required: bool = True
    status: BundleDryRunStageStatus
    reason: str | None = None


class BundleStageDryRunRecord(StrictModel):
    """Side-effect-free stage plan for analysis bundle preflight."""

    name: str
    kind: BundleStageKind
    status: BundleDryRunStageStatus
    depends_on: list[str] = Field(default_factory=list)
    inputs: list[ParentRef] = Field(default_factory=list)
    outputs: list[BundleStageDryRunOutputRecord] = Field(default_factory=list)
    missing_roles: list[BundleMissingRoleRecord] = Field(default_factory=list)
    reason: str | None = None


class AnalysisBundleDryRunResult(StrictModel):
    """Side-effect-free analysis bundle preflight over a matched selection."""

    bundle_name: str
    match_preview: SelectionPreview
    matched_run_ids: list[str] = Field(default_factory=list)
    stages: list[BundleStageDryRunRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    for package_name, metadata in registry.iter_package_metadata():
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
    from feedbax.contracts.migrations import migrate_structured_spec_payload

    migrated = migrate_structured_spec_payload("AnalysisBundleSpec", data)
    return AnalysisBundleSpec.model_validate(migrated.payload)


def iter_candidate_manifests(
    root: Path | str | None = None,
    *,
    manifest_kind: str = "EvaluationRunManifest",
) -> list[AnyManifest]:
    """Load candidate run manifests of one kind from a manifest root."""
    root_path = Path(root) if root is not None else default_manifest_root()
    manifests: list[AnyManifest] = []
    manifest_paths = iter_indexed_manifest_paths_by_kind(manifest_kind, root=root_path)
    if not manifest_paths:
        manifest_paths = iter_manifest_files(root_path)
    for manifest_path in manifest_paths:
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
    allowed_ids = run_ids if run_ids is not None else set(predicate.run_ids)
    if allowed_ids and manifest.id not in allowed_ids:
        return False
    return predicate_matches_row(predicate, _manifest_index_row_for_manifest(manifest))


def select_bundle_manifests(
    bundle: AnalysisBundleSpec,
    root: Path | str | None = None,
    *,
    run_ids: Iterable[str] | None = None,
) -> list[AnyManifest]:
    """Select manifests in a root that match a bundle predicate."""
    allowed_ids = set(run_ids) if run_ids is not None else None
    candidates = iter_candidate_manifests(root, manifest_kind=bundle.predicate.manifest_kind)
    if bundle.predicate.top_k_by_metric_per_group is not None:
        effective_predicate = bundle.predicate
        if allowed_ids is not None:
            effective_predicate = effective_predicate.model_copy(
                update={"run_ids": sorted(allowed_ids)}
            )
        by_id = {manifest.id: manifest for manifest in candidates}
        selected = select_parent_refs(
            effective_predicate,
            [_manifest_index_row_for_manifest(manifest) for manifest in candidates],
        )
        return [by_id[ref.id] for ref in selected if ref.id in by_id]
    return [
        manifest
        for manifest in candidates
        if predicate_matches_manifest(bundle.predicate, manifest, run_ids=allowed_ids)
    ]


def _manifest_index_row_for_manifest(manifest: AnyManifest) -> ManifestIndexRow:
    return ManifestIndexRow(
        id=manifest.id,
        kind=manifest.kind,
        schema_version=manifest.schema_version,
        created_at=manifest.created_at.isoformat(),
        status=manifest.status,
        payload=manifest.model_dump(mode="json"),
    )


def _parent_ref_for_manifest(manifest: AnyManifest) -> ParentRef:
    role_by_kind = {
        "TrainingRunManifest": "training_run",
        "EvaluationRunManifest": "evaluation_run",
        "AnalysisRunManifest": "analysis_run",
        "ReportManifest": "report",
    }
    return ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role=role_by_kind.get(manifest.kind, "run_manifest"),
    )


def _params_for_template(template: AnalysisSpecTemplate) -> dict[str, Any]:
    params = dict(template.params)
    if template.requested_outputs and "requested_outputs" not in params and "outputs" not in params:
        params["requested_outputs"] = list(template.requested_outputs)
    return params


def _params_for_stage(
    stage: BundleStageSpec,
    params_base: BundleParamsBase | None = None,
) -> dict[str, Any]:
    """Resolve one stage's parameters from the shared base or explicit local escape."""
    if stage.local_params is not None:
        params = dict(stage.local_params)
    else:
        params = apply_override_patches(
            (params_base or BundleParamsBase()).params,
            stage.params_patches,
        )
    if (
        stage.kind == "evaluation"
        and stage.states_custody is not None
        and "states_custody" not in params
    ):
        params["states_custody"] = stage.states_custody
    if stage.requested_outputs and "requested_outputs" not in params and "outputs" not in params:
        params["requested_outputs"] = list(stage.requested_outputs)
    return params


def _manifest_ref(
    manifest: EvaluationRunManifest | AnalysisRunManifest | FigureManifest | ReportManifest,
    path: Path,
    role: str,
) -> ParentRef:
    return authenticated_manifest_ref(manifest, path, role)


def _stage_regeneration_payload(
    stage: BundleStageSpec,
    *,
    inputs: Sequence[ParentRef],
    outputs: Sequence[ParentRef | ArtifactRef],
    issues: Sequence[str] = (),
) -> SpecPayload:
    stage_payload = stage.model_dump(mode="json", exclude_none=True)
    if not stage.depends_on_roles:
        stage_payload.pop("depends_on_roles", None)
    command = RegenerationCommand(
        argv=["feedbax", "analysis", "bundle", "--stage", stage.name],
        metadata={"bundle_stage_kind": stage.kind},
    )
    regeneration = RegenerationSpec(
        command=command,
        parameters={
            "stage": stage_payload,
        },
        inputs=list(inputs),
        outputs=list(outputs),
        provenance=Provenance(
            parents=list(inputs),
            issues=list(issues),
            metadata={"bundle_stage": stage.name},
        ),
    )
    return spec_payload(
        "RegenerationSpec",
        regeneration.model_dump(mode="json", exclude_none=True),
    )


def _with_regeneration_spec(
    manifest: AnalysisRunManifest | FigureManifest | ReportManifest,
    regeneration_payload: SpecPayload,
    *,
    root: Path,
) -> tuple[AnalysisRunManifest | FigureManifest | ReportManifest, Path]:
    if regeneration_payload in manifest.regeneration_specs:
        path = (
            root
            / "manifests"
            / {
                "AnalysisRunManifest": "analysis_runs",
                "FigureManifest": "FigureManifest",
                "ReportManifest": "reports",
            }[manifest.kind]
            / f"{safe_manifest_key(manifest.id)}.json"
        )
        return manifest, path
    updated = manifest.model_copy(
        update={"regeneration_specs": [*manifest.regeneration_specs, regeneration_payload]}
    )
    path = write_manifest(updated, root=root)
    return updated, path


def _artifact_parent_ref(artifact: ArtifactRef, *, role: str) -> ParentRef:
    artifact_id = artifact.artifact_id or artifact.sha256 or artifact.uri or artifact.logical_name
    return ParentRef(
        kind="ArtifactRef",
        id=artifact_id,
        role=role,
        uri=artifact.uri,
        metadata={
            "source_role": artifact.role,
            "logical_name": artifact.logical_name,
            "artifact": artifact.model_dump(mode="json", exclude_none=True),
        },
    )


def _resolve_stage_inputs(
    stage: BundleStageSpec,
    matched_manifests: Sequence[AnyManifest],
    stage_products: dict[str, list[StageMaterialization]],
    *,
    bundle_parent_refs: Sequence[ParentRef] | None = None,
) -> ResolvedStageInputs:
    inputs: list[ParentRef] = []
    artifacts_by_alias: dict[str, tuple[ArtifactRef, ...]] = {}
    if stage.include_bundle_inputs or (not stage.depends_on and not stage.depends_on_roles):
        if bundle_parent_refs is None:
            inputs.extend(_parent_ref_for_manifest(manifest) for manifest in matched_manifests)
        else:
            inputs.extend(bundle_parent_refs)
    for dependency in stage.depends_on:
        products = stage_products.get(dependency)
        if products is None:
            raise ValueError(f"Bundle stage {stage.name!r} depends on unknown stage {dependency!r}")
        inputs.extend(product.manifest_ref for product in products if product.manifest_ref)
    for dependency in stage.depends_on_roles:
        products = stage_products.get(dependency.stage)
        if products is None:
            raise ValueError(
                f"Bundle stage {stage.name!r} depends on unknown stage {dependency.stage!r}"
            )
        artifacts = tuple(
            artifact
            for product in products
            for artifact in product.artifacts
            if artifact.role == dependency.role
        )
        if not artifacts:
            if dependency.required:
                raise ValueError(
                    f"Bundle stage {stage.name!r} requires artifact role "
                    f"{dependency.role!r} from stage {dependency.stage!r}, "
                    "but no matching artifact was materialized"
                )
            continue
        alias = dependency.bind_as or dependency.role
        artifacts_by_alias[alias] = (*artifacts_by_alias.get(alias, ()), *artifacts)
        inputs.extend(_artifact_parent_ref(artifact, role=alias) for artifact in artifacts)
    return ResolvedStageInputs(parent_refs=tuple(inputs), artifact_refs_by_alias=artifacts_by_alias)


def _exact_execution_location_key(execution_uri: str) -> str:
    """Validate and normalize one root-relative execution URI for uniqueness."""
    if not execution_uri.strip():
        raise ValueError("exact parent execution_uri must not be empty")
    parsed = urlsplit(execution_uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(
            "exact parent execution_uri must be a relative local path without "
            f"scheme, query, or fragment: {execution_uri!r}"
        )
    decoded = unquote(parsed.path)
    if "\\" in decoded:
        raise ValueError(
            f"exact parent execution_uri contains an unsupported path separator: {execution_uri!r}"
        )
    relative = Path(decoded)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(
            f"exact parent execution_uri escapes the explicit manifest root: {execution_uri!r}"
        )
    return relative.as_posix()


def _require_exact_parent_metadata(parent: ParentRef) -> tuple[str, int]:
    """Validate exact immutable-parent identity and return digest and size."""
    if parent.kind != "TrainingRunManifest":
        raise ValueError(f"exact parent kind must be 'TrainingRunManifest'; got {parent.kind!r}")
    if parent.role != "training_run":
        raise ValueError(f"exact parent role must be 'training_run'; got {parent.role!r}")
    if not parent.id.strip():
        raise ValueError("exact parent id must not be empty")

    manifest_digest = parent.metadata.get("manifest_sha256")
    if not isinstance(manifest_digest, str) or _SHA256_PATTERN.fullmatch(manifest_digest) is None:
        raise ValueError(
            "exact parent metadata.manifest_sha256 must be exactly 64 lowercase hex characters"
        )
    certificate_digest = parent.metadata.get("certificate_sha256")
    if (
        not isinstance(certificate_digest, str)
        or _SHA256_PATTERN.fullmatch(certificate_digest) is None
    ):
        raise ValueError(
            "exact parent metadata.certificate_sha256 must be exactly 64 lowercase hex characters"
        )
    size_bytes = parent.metadata.get("size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError("exact parent metadata.size_bytes must be a nonnegative integer")
    for field_name in _EXACT_PARENT_REQUIRED_TEXT_METADATA:
        value = parent.metadata.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"exact parent metadata.{field_name} must be a nonempty string")
    for field_name, expected in _EXACT_PARENT_REQUIRED_STATUS_METADATA.items():
        observed = parent.metadata.get(field_name)
        if observed != expected:
            raise ValueError(
                f"exact parent metadata.{field_name} must be {expected!r}; got {observed!r}"
            )
    if parent.metadata["planned_run_id"] != parent.id:
        raise ValueError("exact parent metadata.planned_run_id must equal ParentRef.id")

    uri = parent.uri
    match = _IMMUTABLE_MANIFEST_URI_PATTERN.fullmatch(uri or "")
    if match is None:
        raise ValueError("exact parent uri must be immutable artifact://sha256/<digest>")
    uri_digest = match.group(1)
    if uri_digest != manifest_digest:
        raise ValueError("exact parent artifact URI digest must equal metadata.manifest_sha256")
    return manifest_digest, size_bytes


def _available_training_identity_values(
    manifest: TrainingRunManifest,
    field_name: str,
) -> list[tuple[str, Any]]:
    """Return declared manifest identity facts that are available for comparison."""
    values: list[tuple[str, Any]] = []
    root_field_names = (
        ("status", "manifest_status") if field_name == "manifest_status" else (field_name,)
    )
    for root_field_name in root_field_names:
        root_value = getattr(manifest, root_field_name, None)
        if root_value is not None:
            values.append((root_field_name, root_value))
    if field_name in manifest.metadata:
        values.append((f"metadata.{field_name}", manifest.metadata[field_name]))
    if field_name in manifest.provenance.metadata:
        values.append(
            (f"provenance.metadata.{field_name}", manifest.provenance.metadata[field_name])
        )
    if manifest.training_spec is not None and field_name in manifest.training_spec.inline:
        values.append(
            (f"training_spec.inline.{field_name}", manifest.training_spec.inline[field_name])
        )
    return values


def _validate_exact_manifest_identity(
    manifest: TrainingRunManifest,
    parent: ParentRef,
) -> None:
    if manifest.status != "completed":
        raise ValueError(
            f"exact parent TrainingRunManifest status must be 'completed'; got {manifest.status!r}"
        )
    governed_facts = (
        *_EXACT_PARENT_REQUIRED_TEXT_METADATA,
        *_EXACT_PARENT_REQUIRED_STATUS_METADATA,
        "certificate_sha256",
    )
    for field_name in governed_facts:
        expected = parent.metadata[field_name]
        for source, observed in _available_training_identity_values(manifest, field_name):
            if observed != expected:
                raise ValueError(
                    f"exact parent {source} disagrees with ParentRef metadata.{field_name}: "
                    f"expected={expected!r}, observed={observed!r}"
                )


def _preflight_staged_exact_parents(
    bundle: AnalysisBundleSpec,
    exact_parents: StagedExactParents,
    *,
    root: Path,
) -> tuple[
    list[TrainingRunManifest],
    tuple[ParentRef, ...],
    tuple[StagedParentExecutionLocation, ...],
]:
    """Resolve and validate authoritative staged parents before any recipe work."""
    entries = list(exact_parents.parents)
    if not entries:
        raise ValueError("StagedExactParents.parents must be nonempty")

    parent_refs = tuple(entry.parent for entry in entries)
    serialized_refs = [
        json.dumps(
            parent.model_dump(mode="json", exclude_none=False),
            sort_keys=True,
            separators=(",", ":"),
        )
        for parent in parent_refs
    ]
    if len(set(serialized_refs)) != len(serialized_refs):
        raise ValueError("StagedExactParents contains a duplicate ParentRef")
    parent_ids = [parent.id for parent in parent_refs]
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("StagedExactParents contains a duplicate ParentRef id")
    location_keys = [_exact_execution_location_key(entry.execution_uri) for entry in entries]
    if len(set(location_keys)) != len(location_keys):
        raise ValueError("StagedExactParents contains a duplicate execution location")

    for parent in parent_refs:
        _require_exact_parent_metadata(parent)

    predicate = bundle.predicate
    exact_id_set = set(parent_ids)
    if predicate.top_k_by_metric_per_group is not None:
        raise ValueError(
            "exact-parent staged execution rejects top_k_by_metric_per_group; "
            "frozen membership cannot be narrowed"
        )
    if predicate.run_ids and set(predicate.run_ids) != exact_id_set:
        raise ValueError(
            "bundle predicate.run_ids must equal the exact parent ID set; "
            "predicates cannot add, remove, or narrow frozen parents"
        )
    if len(set(predicate.run_ids)) != len(predicate.run_ids):
        raise ValueError("bundle predicate.run_ids contains duplicate parent IDs")
    for stage in bundle.stages:
        is_root_evaluation = (
            stage.kind == "evaluation" and not stage.depends_on and not stage.depends_on_roles
        )
        if is_root_evaluation and stage.mode != "per-run":
            raise ValueError(
                f"exact-parent root evaluation stage {stage.name!r} must use per-run mode; "
                "parent grouping is only valid downstream"
            )

    manifests: list[TrainingRunManifest] = []
    locations: list[StagedParentExecutionLocation] = []
    for entry in entries:
        parent = entry.parent
        declared_digest, declared_size = _require_exact_parent_metadata(parent)
        execution_ref = parent.model_copy(update={"uri": entry.execution_uri})
        resolved = resolve_evaluation_inputs(
            EvaluationRunSpec(
                evaluation_type="feedbax.internal.staged_exact_parent_preflight",
                inputs=[execution_ref],
            ),
            manifest_root=root,
            require_unique_manifest_id=False,
        )[0]
        if resolved.sha256 != declared_digest:
            raise ValueError(
                "exact parent manifest bytes do not match the immutable artifact digest"
            )
        observed_size = resolved.size_bytes
        if observed_size != declared_size:
            raise ValueError(
                "exact parent manifest byte size does not match metadata.size_bytes: "
                f"declared={declared_size}, observed={observed_size}"
            )
        _validate_exact_manifest_identity(resolved.manifest, parent)
        if not predicate_matches_manifest(predicate, resolved.manifest):
            raise ValueError(f"exact parent {parent.id!r} does not satisfy the bundle predicate")
        manifests.append(resolved.manifest)
        locations.append(
            StagedParentExecutionLocation(
                parent=parent,
                root=root.resolve(strict=True),
                execution_uri=entry.execution_uri,
            )
        )

    if len(manifests) != len(parent_refs):
        raise ValueError("exact parent resolution did not preserve one-to-one cardinality")
    return manifests, parent_refs, tuple(locations)


def _manifest_condition_payload(manifest: AnyManifest) -> dict[str, Any]:
    return {
        "kind": manifest.kind,
        "id": manifest.id,
        "metadata": dict(manifest.metadata),
        "params": _params_payload(manifest),
    }


def _stage_expression_context(
    stage: BundleStageSpec,
    matched_manifests: Sequence[AnyManifest],
    resolved_inputs: ResolvedStageInputs,
    params_base: BundleParamsBase,
) -> ExpressionContext:
    items: dict[str, ContextItem] = {
        "params": ContextItem(kind="params", payload=_params_for_stage(stage, params_base)),
        "manifests": ContextItem(
            kind="manifests",
            payload=[_manifest_condition_payload(manifest) for manifest in matched_manifests],
        ),
    }
    if len(matched_manifests) == 1:
        items["manifest"] = ContextItem(
            kind="manifest",
            payload=_manifest_condition_payload(matched_manifests[0]),
        )
    for manifest in matched_manifests:
        items[f"manifest:{manifest.id}"] = ContextItem(
            kind="manifest",
            payload=_manifest_condition_payload(manifest),
        )
    for input_ref in resolved_inputs.parent_refs:
        if input_ref.role is None or input_ref.kind == "ArtifactRef":
            continue
        payload = input_ref.model_dump(mode="json", exclude_none=True)
        existing = items.get(input_ref.role)
        if existing is None:
            items[input_ref.role] = ContextItem(kind="parent_role", payload=[payload])
        else:
            existing.payload.append(payload)
    for alias, artifacts in resolved_inputs.artifact_refs_by_alias.items():
        items[alias] = ContextItem(
            kind="artifact_role",
            payload=[artifact.model_dump(mode="json", exclude_none=True) for artifact in artifacts],
        )
    return ExpressionContext(items=items)


def _run_condition_skip_reason(
    stage: BundleStageSpec,
    matched_manifests: Sequence[AnyManifest],
    resolved_inputs: ResolvedStageInputs,
    params_base: BundleParamsBase,
) -> str | None:
    if stage.run_condition is None:
        return None
    ctx = _stage_expression_context(stage, matched_manifests, resolved_inputs, params_base)
    result = evaluate_expr(stage.run_condition, ctx)
    if result:
        return None
    return (
        f"run_condition evaluated false: {canonical_expression_json(stage.run_condition)} -> false"
    )


def _stage_input_groups(
    stage: BundleStageSpec,
    inputs: Sequence[ParentRef],
) -> list[list[ParentRef]]:
    if stage.mode == "per-run":
        return [[input_ref] for input_ref in inputs]
    return [list(inputs)]


def _artifact_groups(products: Sequence[StageMaterialization]) -> dict[str, list[ArtifactRef]]:
    groups: dict[str, list[ArtifactRef]] = {}
    for product in products:
        for artifact in product.artifacts:
            groups.setdefault(artifact.role, []).append(artifact)
    return groups


def _default_outputs_for_stage(stage: BundleStageSpec) -> list[BundleStageOutputSpec]:
    if stage.outputs:
        return list(stage.outputs)
    if stage.kind == "figure":
        return [
            BundleStageOutputSpec(role=FIGURE_RENDER_ROLE),
            BundleStageOutputSpec(role="figure_spec", required=False),
            BundleStageOutputSpec(role="manifest", required=False),
        ]
    return [BundleStageOutputSpec(role="manifest")]


def _output_records(
    stage: BundleStageSpec,
    *,
    products: Sequence[StageMaterialization],
    skipped_reason: str | None = None,
    not_applicable_reason: str | None = None,
) -> list[BundleStageOutputRecord]:
    output_specs = _default_outputs_for_stage(stage)
    manifest_refs = [product.manifest_ref for product in products if product.manifest_ref]
    artifact_groups = _artifact_groups(products)
    records: list[BundleStageOutputRecord] = []
    for output_spec in output_specs:
        if skipped_reason is not None:
            records.append(
                BundleStageOutputRecord(
                    role=output_spec.role,
                    required=output_spec.required,
                    status="skipped",
                    reason=skipped_reason,
                )
            )
            continue
        if not_applicable_reason is not None:
            records.append(
                BundleStageOutputRecord(
                    role=output_spec.role,
                    required=output_spec.required,
                    status="not_applicable",
                    reason=not_applicable_reason,
                )
            )
            continue

        artifacts = artifact_groups.get(output_spec.role, [])
        role_has_manifest = output_spec.role == "manifest" and bool(manifest_refs)
        if role_has_manifest or artifacts:
            records.append(
                BundleStageOutputRecord(
                    role=output_spec.role,
                    required=output_spec.required,
                    status="materialized",
                    manifest_refs=list(manifest_refs) if role_has_manifest else [],
                    artifacts=list(artifacts),
                )
            )
        elif output_spec.required:
            raise ValueError(
                f"Bundle stage {stage.name!r} required output role "
                f"{output_spec.role!r} was not materialized"
            )
        else:
            records.append(
                BundleStageOutputRecord(
                    role=output_spec.role,
                    required=False,
                    status="missing",
                    reason="optional output role was not materialized",
                )
            )
    return records


def _record_status(records: Sequence[BundleStageOutputRecord]) -> BundleOutputStatus:
    if any(record.status == "materialized" for record in records):
        return "materialized"
    if records:
        return records[0].status
    return "missing"


def _validate_completed_evaluation_cache(
    manifest: EvaluationRunManifest,
    *,
    requested_spec: EvaluationRunSpec,
    requested_manifest_id: str,
) -> None:
    if manifest.id != requested_manifest_id:
        raise ValueError(
            "cached EvaluationRunManifest id does not match the requested evaluation spec: "
            f"expected {requested_manifest_id!r}, got {manifest.id!r}"
        )
    if manifest.evaluation_spec.kind != "EvaluationRunSpec":
        raise ValueError(
            "cached EvaluationRunManifest evaluation_spec kind must be "
            f"'EvaluationRunSpec'; got {manifest.evaluation_spec.kind!r}"
        )
    try:
        cached_spec = EvaluationRunSpec.model_validate(manifest.evaluation_spec.inline)
    except ValueError as exc:
        raise ValueError(
            "cached EvaluationRunManifest evaluation_spec is not a valid EvaluationRunSpec"
        ) from exc
    if canonical_json_bytes(cached_spec) != canonical_json_bytes(requested_spec):
        raise ValueError(
            "cached EvaluationRunManifest evaluation_spec does not match the requested "
            "EvaluationRunSpec"
        )


def _execute_evaluation_stage(
    stage: BundleStageSpec,
    input_groups: Sequence[Sequence[ParentRef]],
    *,
    root: Path,
    issues: Sequence[str],
    bundle: AnalysisBundleSpec,
    execution_context: StagedExecutionContext,
) -> list[StageMaterialization]:
    products: list[StageMaterialization] = []
    for inputs in input_groups:
        spec = EvaluationRunSpec(
            evaluation_type=str(stage.evaluation_type),
            inputs=list(inputs),
            params=_params_for_stage(stage, bundle.params_base),
        )
        manifest_id = evaluation_run_manifest_id(spec)
        path = root / "manifests" / "evaluation_runs" / f"{safe_manifest_key(manifest_id)}.json"
        existing = load_manifest(path) if path.is_file() else None
        if isinstance(existing, EvaluationRunManifest) and existing.status == "completed":
            _validate_completed_evaluation_cache(
                existing,
                requested_spec=spec,
                requested_manifest_id=manifest_id,
            )
            manifest = existing
        else:
            manifest, path = execute_evaluation_run_spec(
                spec,
                root=root,
                issues=list(issues),
                execution_context=execution_context,
                metadata={
                    "bundle": {
                        "name": bundle.name,
                        "stage": stage.name,
                        "schema_id": bundle.schema_id,
                        "schema_version": bundle.schema_version,
                    }
                },
            )
        products.append(
            StageMaterialization(
                manifest_ref=_manifest_ref(manifest, path, "evaluation_run"),
                artifacts=tuple(manifest.artifacts),
                manifest_path=path,
            )
        )
    return products


def _execute_analysis_stage(
    stage: BundleStageSpec,
    input_groups: Sequence[Sequence[ParentRef]],
    *,
    root: Path,
    issues: Sequence[str],
    bundle: AnalysisBundleSpec,
    fig_dump_path: Path | str | None,
    fig_dump_formats: Sequence[str],
    execution_context: StagedExecutionContext,
) -> list[StageMaterialization]:
    def build_spec(inputs: Sequence[ParentRef], _index: int) -> AnalysisRunSpec:
        return AnalysisRunSpec(
            analysis_type=str(stage.analysis_type),
            inputs=list(inputs),
            input_requirements=stage.input_requirements,
            params=_params_for_stage(stage, bundle.params_base),
        )

    def execute_spec(
        spec: AnalysisRunSpec,
        inputs: Sequence[ParentRef],
        _index: int,
    ) -> tuple[AnalysisRunManifest, Path]:
        base_provenance = collect_git_provenance()
        base_provenance.parents = list(inputs)
        return execute_analysis_run_spec(
            spec,
            root=root,
            issues=list(issues),
            provenance=base_provenance,
            execution_context=execution_context,
            metadata={
                "bundle": {
                    "name": bundle.name,
                    "stage": stage.name,
                    "schema_id": bundle.schema_id,
                    "schema_version": bundle.schema_version,
                }
            },
            fig_dump_path=fig_dump_path,
            fig_dump_formats=fig_dump_formats,
        )

    return _execute_stage_common(
        stage,
        input_groups,
        root=root,
        issues=issues,
        manifest_role="analysis_run",
        build_spec=build_spec,
        execute_spec=execute_spec,
    )


def _has_materializer_capability(analysis: AbstractAnalysis) -> bool:
    return (
        isinstance(analysis, ContextMaterializer)
        or analysis.__class__.emit_artifacts is not AbstractAnalysis.emit_artifacts
    )


def _validate_materialization_recipe_result(
    stage: BundleStageSpec,
    analysis_type: str,
    result: AnalysisRecipeResult,
) -> None:
    for name, analysis in result.analyses.items():
        if not _has_materializer_capability(analysis):
            raise ValueError(
                f"Bundle materialization stage {stage.name!r} analysis_type "
                f"{analysis_type!r} resolved analysis {name!r} of type "
                f"{type(analysis).__name__}, but materialization stages require "
                "ContextMaterializer or an analysis overriding emit_artifacts()"
            )


def _execute_materialization_stage(
    stage: BundleStageSpec,
    input_groups: Sequence[Sequence[ParentRef]],
    *,
    root: Path,
    issues: Sequence[str],
    bundle: AnalysisBundleSpec,
    fig_dump_path: Path | str | None,
    fig_dump_formats: Sequence[str],
    execution_context: StagedExecutionContext,
) -> list[StageMaterialization]:
    def build_spec(inputs: Sequence[ParentRef], _index: int) -> AnalysisRunSpec:
        return AnalysisRunSpec(
            analysis_type=str(stage.analysis_type),
            inputs=list(inputs),
            input_requirements=stage.input_requirements,
            params=_params_for_stage(stage, bundle.params_base),
        )

    def execute_spec(
        spec: AnalysisRunSpec,
        inputs: Sequence[ParentRef],
        _index: int,
    ) -> tuple[AnalysisRunManifest, Path]:
        base_provenance = collect_git_provenance()
        base_provenance.parents = list(inputs)
        return execute_analysis_run_spec(
            spec,
            root=root,
            issues=list(issues),
            provenance=base_provenance,
            execution_context=execution_context,
            metadata={
                "bundle": {
                    "name": bundle.name,
                    "stage": stage.name,
                    "schema_id": bundle.schema_id,
                    "schema_version": bundle.schema_version,
                }
            },
            validate_result=lambda analysis_type, result: (
                _validate_materialization_recipe_result(stage, analysis_type, result)
            ),
            fig_dump_path=fig_dump_path,
            fig_dump_formats=fig_dump_formats,
        )

    return _execute_stage_common(
        stage,
        input_groups,
        root=root,
        issues=issues,
        manifest_role="analysis_run",
        build_spec=build_spec,
        execute_spec=execute_spec,
    )


def _execute_figure_stage(
    stage: BundleStageSpec,
    input_groups: Sequence[Sequence[ParentRef]],
    *,
    root: Path,
    issues: Sequence[str],
    bundle: AnalysisBundleSpec,
) -> list[StageMaterialization]:
    products: list[StageMaterialization] = []
    if stage.figure is None:
        raise ValueError(f"figure bundle stage {stage.name!r} requires figure")
    for index, inputs in enumerate(input_groups):
        figure_spec = stage.figure.model_copy(
            update={
                "inputs": [*stage.figure.inputs, *inputs],
                "metadata": {
                    **stage.figure.metadata,
                    "bundle": {
                        "name": bundle.name,
                        "stage": stage.name,
                        "index": index,
                        "schema_id": bundle.schema_id,
                        "schema_version": bundle.schema_version,
                    },
                },
            },
            deep=True,
        )
        manifest, path = execute_figure_spec(
            figure_spec,
            root=root,
            provenance=Provenance(
                parents=list(figure_spec.inputs),
                issues=list(issues),
                metadata={"bundle": bundle.name, "stage": stage.name},
            ),
            metadata={
                "bundle": {
                    "name": bundle.name,
                    "stage": stage.name,
                    "schema_id": bundle.schema_id,
                    "schema_version": bundle.schema_version,
                }
            },
        )
        regeneration_payload = _stage_regeneration_payload(
            stage,
            inputs=inputs,
            # A manifest cannot embed an authentication hash of its own final bytes.
            # Its external StageMaterialization ref is created after this rewrite.
            outputs=list(manifest.artifacts),
            issues=issues,
        )
        updated_manifest, updated_path = _with_regeneration_spec(
            manifest,
            regeneration_payload,
            root=root,
        )
        products.append(
            StageMaterialization(
                manifest_ref=_manifest_ref(updated_manifest, updated_path, "figure"),
                artifacts=tuple(updated_manifest.artifacts),
                manifest_path=updated_path,
                regeneration_spec=regeneration_payload,
            )
        )
    return products


def _execute_report_stage(
    stage: BundleStageSpec,
    input_groups: Sequence[Sequence[ParentRef]],
    *,
    root: Path,
    issues: Sequence[str],
    bundle: AnalysisBundleSpec,
) -> list[StageMaterialization]:
    def build_spec(inputs: Sequence[ParentRef], index: int) -> ReportSpec:
        stage_params = _params_for_stage(stage, bundle.params_base)
        bundle_metadata = {
            "name": bundle.name,
            "stage": stage.name,
            "index": index,
            "schema_id": bundle.schema_id,
            "schema_version": bundle.schema_version,
        }
        return ReportSpec(
            report_type=str(stage.report_type or BUNDLE_SUMMARY_REPORT_TYPE),
            inputs=list(inputs),
            params={
                "stage_params": stage_params,
                "bundle": bundle_metadata,
            },
            narrative=stage_params.get("narrative"),
        )

    def execute_spec(
        spec: ReportSpec,
        inputs: Sequence[ParentRef],
        _index: int,
    ) -> tuple[ReportManifest, Path]:
        return execute_report_spec(
            spec,
            root=root,
            provenance=Provenance(
                parents=list(inputs),
                issues=list(issues),
                metadata={"bundle": bundle.name, "stage": stage.name},
            ),
            metadata={
                "bundle": {
                    "name": bundle.name,
                    "stage": stage.name,
                    "schema_id": bundle.schema_id,
                    "schema_version": bundle.schema_version,
                }
            },
        )

    return _execute_stage_common(
        stage,
        input_groups,
        root=root,
        issues=issues,
        manifest_role="report",
        build_spec=build_spec,
        execute_spec=execute_spec,
    )


def _execute_stage_common(
    stage: BundleStageSpec,
    input_groups: Sequence[Sequence[ParentRef]],
    *,
    root: Path,
    issues: Sequence[str],
    manifest_role: Literal["analysis_run", "report"],
    build_spec: Callable[[Sequence[ParentRef], int], _StageSpecT],
    execute_spec: Callable[
        [_StageSpecT, Sequence[ParentRef], int],
        tuple[AnyManifest, Path],
    ],
) -> list[StageMaterialization]:
    """Execute and record one spec-emitting stage for each resolved input group."""

    products: list[StageMaterialization] = []
    for index, input_group in enumerate(input_groups):
        inputs = tuple(input_group)
        spec = build_spec(inputs, index)
        manifest, path = execute_spec(spec, inputs, index)
        regeneration_payload = _stage_regeneration_payload(
            stage,
            inputs=inputs,
            # Self-authentication is carried by the external stage record only.
            outputs=list(manifest.artifacts),
            issues=issues,
        )
        updated_manifest, updated_path = _with_regeneration_spec(
            manifest,
            regeneration_payload,
            root=root,
        )
        products.append(
            StageMaterialization(
                manifest_ref=_manifest_ref(updated_manifest, updated_path, manifest_role),
                artifacts=tuple(updated_manifest.artifacts),
                manifest_path=updated_path,
                regeneration_spec=regeneration_payload,
            )
        )
    return products


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
                    "metadata": dict(bundle.metadata),
                }
            },
            fig_dump_path=fig_dump_path,
            fig_dump_formats=fig_dump_formats,
        )
        outputs.append((expansion, manifest, path))
    return outputs


def _selection_preview_for_bundle(
    bundle: AnalysisBundleSpec,
    matched_manifests: Sequence[AnyManifest],
) -> SelectionPreview:
    query = SelectionSpec(
        mode="query",
        manifest_kind=bundle.predicate.manifest_kind,
        query=bundle.predicate,
    )
    return SelectionPreview(
        selection_spec=query,
        match_count=len(matched_manifests),
        parent_refs=[_parent_ref_for_manifest(manifest) for manifest in matched_manifests],
    )


def _select_dry_run_manifests(
    bundle: AnalysisBundleSpec,
    root: Path,
    *,
    selection_spec: SelectionSpec | None,
    run_ids: Iterable[str] | None,
    preview_limit: int | None,
) -> tuple[list[AnyManifest], SelectionPreview]:
    if selection_spec is None:
        matched = select_bundle_manifests(bundle, root, run_ids=run_ids)
        preview = _selection_preview_for_bundle(bundle, matched)
        if preview_limit is not None and preview_limit >= 0:
            preview = preview.model_copy(
                update={
                    "parent_refs": preview.parent_refs[:preview_limit],
                    "truncated": len(preview.parent_refs) > preview_limit,
                }
            )
        return matched, preview

    candidates = iter_candidate_manifests(root, manifest_kind=selection_spec.manifest_kind)
    rows = [_manifest_index_row_for_manifest(manifest) for manifest in candidates]
    full_preview = preview_selection_spec(selection_spec, rows, limit=None)
    preview = preview_selection_spec(selection_spec, rows, limit=preview_limit)
    selected_ids = {ref.id for ref in full_preview.parent_refs}
    matched = [manifest for manifest in candidates if manifest.id in selected_ids]
    return matched, preview


def _dry_run_stage_outputs(
    stage: BundleStageSpec,
    status: BundleDryRunStageStatus,
    *,
    reason: str | None = None,
) -> list[BundleStageDryRunOutputRecord]:
    return [
        BundleStageDryRunOutputRecord(
            role=output.role,
            required=output.required,
            status=status,
            reason=reason,
        )
        for output in _default_outputs_for_stage(stage)
    ]


def _dry_run_stage_status(
    outputs: Sequence[BundleStageDryRunOutputRecord],
    fallback: BundleDryRunStageStatus,
) -> BundleDryRunStageStatus:
    if any(output.status == "would_run" for output in outputs):
        return "would_run"
    return outputs[0].status if outputs else fallback


def _missing_role_dependencies(
    stage: BundleStageSpec,
    stage_products: dict[str, list[StageMaterialization]],
) -> list[BundleMissingRoleRecord]:
    missing: list[BundleMissingRoleRecord] = []
    for dependency in stage.depends_on_roles:
        products = stage_products.get(dependency.stage)
        if products is None:
            continue
        artifacts = [
            artifact
            for product in products
            for artifact in product.artifacts
            if artifact.role == dependency.role
        ]
        if artifacts or not dependency.required:
            continue
        missing.append(
            BundleMissingRoleRecord(
                stage=dependency.stage,
                role=dependency.role,
                required=dependency.required,
                bind_as=dependency.bind_as,
                reason=(
                    f"required role {dependency.role!r} from stage "
                    f"{dependency.stage!r} is not available"
                ),
            )
        )
    return missing


def _dry_run_manifest_role(stage: BundleStageSpec) -> str:
    return {
        "evaluation": "evaluation_run",
        "analysis": "analysis_run",
        "materialization": "analysis_run",
        "figure": "figure",
        "report": "report",
    }[stage.kind]


def _dry_run_products(stage: BundleStageSpec) -> list[StageMaterialization]:
    outputs = _default_outputs_for_stage(stage)
    artifacts = tuple(
        ArtifactRef(
            role=output.role,
            logical_name=f"dry-run/{stage.name}/{output.role}",
            metadata={"dry_run": True, "stage": stage.name},
        )
        for output in outputs
        if output.role != "manifest"
    )
    manifest_ref = ParentRef(
        kind={
            "evaluation": "EvaluationRunManifest",
            "analysis": "AnalysisRunManifest",
            "materialization": "AnalysisRunManifest",
            "figure": "FigureManifest",
            "report": "ReportManifest",
        }[stage.kind],
        id=f"dry-run:{stage.name}",
        role=_dry_run_manifest_role(stage),
        metadata={"dry_run": True, "stage": stage.name},
    )
    return [StageMaterialization(manifest_ref=manifest_ref, artifacts=artifacts)]


def dry_run_staged_analysis_bundle(
    bundle: AnalysisBundleSpec,
    *,
    root: Path | str | None = None,
    selection_spec: SelectionSpec | None = None,
    run_ids: Iterable[str] | None = None,
    exact_parents: StagedExactParents | None = None,
    preview_limit: int | None = 50,
    execution_descriptor: StagedExecutionDescriptor | Mapping[str, Any] | None = None,
    artifact_provider_bindings: Sequence[StagedArtifactProviderRootBinding] = (),
    checkpoint_custody_bindings: Sequence[StagedCheckpointCustodyRootBinding] = (),
) -> AnalysisBundleDryRunResult:
    """Evaluate staged bundle bindings, conditions, and role dependencies only."""
    _execution_context = resolve_staged_execution_context(
        execution_descriptor,
        artifact_provider_bindings=artifact_provider_bindings,
        checkpoint_custody_bindings=checkpoint_custody_bindings,
    )
    if not bundle.stages:
        raise ValueError(f"Analysis bundle {bundle.name!r} has no staged plan")
    if exact_parents is not None and (run_ids is not None or selection_spec is not None):
        raise ValueError("exact_parents cannot be combined with run_ids or selection_spec")
    if exact_parents is not None and root is None:
        raise ValueError("exact-parent staged dry-run requires an explicit manifest root")
    if exact_parents is not None:
        exact_parents = StagedExactParents.model_validate(exact_parents.model_dump(mode="json"))

    root_path = Path(root) if root is not None else default_manifest_root()
    bundle_parent_refs: tuple[ParentRef, ...] | None = None
    if exact_parents is None:
        matched_manifests, preview = _select_dry_run_manifests(
            bundle,
            root_path,
            selection_spec=selection_spec,
            run_ids=run_ids,
            preview_limit=preview_limit,
        )
    else:
        matched_manifests, bundle_parent_refs, parent_locations = _preflight_staged_exact_parents(
            bundle,
            exact_parents,
            root=root_path,
        )
        _execution_context = with_staged_parent_execution_locations(
            _execution_context,
            parent_locations,
        )
        preview = _selection_preview_for_bundle(bundle, matched_manifests)
        full_parent_refs = list(bundle_parent_refs)
        shown_parent_refs = full_parent_refs
        truncated = False
        if preview_limit is not None and preview_limit >= 0:
            shown_parent_refs = full_parent_refs[:preview_limit]
            truncated = len(full_parent_refs) > preview_limit
        preview = preview.model_copy(
            update={"parent_refs": shown_parent_refs, "truncated": truncated}
        )
    stage_products: dict[str, list[StageMaterialization]] = {}
    records: list[BundleStageDryRunRecord] = []

    for stage in bundle.stages:
        missing_roles = _missing_role_dependencies(stage, stage_products)
        if missing_roles:
            outputs = _dry_run_stage_outputs(
                stage,
                "missing",
                reason=missing_roles[0].reason,
            )
            stage_products[stage.name] = []
            records.append(
                BundleStageDryRunRecord(
                    name=stage.name,
                    kind=stage.kind,
                    status="missing",
                    depends_on=list(stage.depends_on),
                    outputs=outputs,
                    missing_roles=missing_roles,
                    reason=missing_roles[0].reason,
                )
            )
            continue

        resolved_inputs = _resolve_stage_inputs(
            stage,
            matched_manifests,
            stage_products,
            bundle_parent_refs=bundle_parent_refs,
        )
        inputs = list(resolved_inputs.parent_refs)
        if stage.skip_reason is not None:
            outputs = _dry_run_stage_outputs(stage, "would_skip", reason=stage.skip_reason)
            stage_products[stage.name] = []
            records.append(
                BundleStageDryRunRecord(
                    name=stage.name,
                    kind=stage.kind,
                    status="would_skip",
                    depends_on=list(stage.depends_on),
                    inputs=inputs,
                    outputs=outputs,
                    reason=stage.skip_reason,
                )
            )
            continue

        if stage.not_applicable_reason is not None:
            outputs = _dry_run_stage_outputs(
                stage,
                "not_applicable",
                reason=stage.not_applicable_reason,
            )
            stage_products[stage.name] = []
            records.append(
                BundleStageDryRunRecord(
                    name=stage.name,
                    kind=stage.kind,
                    status="not_applicable",
                    depends_on=list(stage.depends_on),
                    inputs=inputs,
                    outputs=outputs,
                    reason=stage.not_applicable_reason,
                )
            )
            continue

        run_condition_skip_reason = _run_condition_skip_reason(
            stage,
            matched_manifests,
            resolved_inputs,
            bundle.params_base,
        )
        if run_condition_skip_reason is not None:
            outputs = _dry_run_stage_outputs(
                stage,
                "would_skip",
                reason=run_condition_skip_reason,
            )
            stage_products[stage.name] = []
            records.append(
                BundleStageDryRunRecord(
                    name=stage.name,
                    kind=stage.kind,
                    status="would_skip",
                    depends_on=list(stage.depends_on),
                    inputs=inputs,
                    outputs=outputs,
                    reason=run_condition_skip_reason,
                )
            )
            continue

        outputs = _dry_run_stage_outputs(stage, "would_run")
        stage_products[stage.name] = _dry_run_products(stage)
        records.append(
            BundleStageDryRunRecord(
                name=stage.name,
                kind=stage.kind,
                status=_dry_run_stage_status(outputs, "would_run"),
                depends_on=list(stage.depends_on),
                inputs=inputs,
                outputs=outputs,
            )
        )

    return AnalysisBundleDryRunResult(
        bundle_name=bundle.name,
        match_preview=preview,
        matched_run_ids=[manifest.id for manifest in matched_manifests],
        stages=records,
        metadata=dict(bundle.metadata),
    )


def execute_staged_analysis_bundle(
    bundle: AnalysisBundleSpec,
    *,
    root: Path | str | None = None,
    run_ids: Iterable[str] | None = None,
    exact_parents: StagedExactParents | None = None,
    execution_descriptor: StagedExecutionDescriptor | Mapping[str, Any] | None = None,
    artifact_provider_bindings: Sequence[StagedArtifactProviderRootBinding] = (),
    checkpoint_custody_bindings: Sequence[StagedCheckpointCustodyRootBinding] = (),
    issues: list[str] | None = None,
    fig_dump_path: Path | str | None = None,
    fig_dump_formats: Sequence[str] = ("html",),
) -> StagedAnalysisBundleExecution:
    """Execute an ordered staged bundle plan and return durable lineage records.

    The staged executor deliberately reuses existing provider-depth manifest
    products: evaluation stages emit ``EvaluationRunManifest`` refs, analysis
    and materialization stages emit ``AnalysisRunManifest`` refs plus artifacts,
    and report stages emit ``ReportManifest`` refs plus report artifacts.
    """
    execution_context = resolve_staged_execution_context(
        execution_descriptor,
        artifact_provider_bindings=artifact_provider_bindings,
        checkpoint_custody_bindings=checkpoint_custody_bindings,
    )

    if not bundle.stages:
        raise ValueError(f"Analysis bundle {bundle.name!r} has no staged plan")

    if exact_parents is not None and run_ids is not None:
        raise ValueError("exact_parents and run_ids are mutually exclusive")
    if exact_parents is not None and root is None:
        raise ValueError("exact-parent staged execution requires an explicit manifest root")

    if exact_parents is not None:
        exact_parents = StagedExactParents.model_validate(exact_parents.model_dump(mode="json"))

    root_path = Path(root) if root is not None else default_manifest_root()
    issue_refs = list(issues or [])
    if exact_parents is None:
        matched_manifests = select_bundle_manifests(bundle, root_path, run_ids=run_ids)
        bundle_parent_refs = None
    else:
        matched_manifests, bundle_parent_refs, parent_locations = _preflight_staged_exact_parents(
            bundle,
            exact_parents,
            root=root_path,
        )
        execution_context = with_staged_parent_execution_locations(
            execution_context,
            parent_locations,
        )
    stage_products: dict[str, list[StageMaterialization]] = {}
    stage_records: list[BundleStageExecutionRecord] = []

    for stage in bundle.stages:
        resolved_inputs = _resolve_stage_inputs(
            stage,
            matched_manifests,
            stage_products,
            bundle_parent_refs=bundle_parent_refs,
        )
        inputs = list(resolved_inputs.parent_refs)
        if stage.skip_reason is not None:
            records = _output_records(stage, products=[], skipped_reason=stage.skip_reason)
            stage_products[stage.name] = []
            stage_records.append(
                BundleStageExecutionRecord(
                    name=stage.name,
                    kind=stage.kind,
                    status=_record_status(records),
                    depends_on=list(stage.depends_on),
                    inputs=inputs,
                    outputs=records,
                    reason=stage.skip_reason,
                )
            )
            continue

        if stage.not_applicable_reason is not None:
            records = _output_records(
                stage,
                products=[],
                not_applicable_reason=stage.not_applicable_reason,
            )
            stage_products[stage.name] = []
            stage_records.append(
                BundleStageExecutionRecord(
                    name=stage.name,
                    kind=stage.kind,
                    status=_record_status(records),
                    depends_on=list(stage.depends_on),
                    inputs=inputs,
                    outputs=records,
                    reason=stage.not_applicable_reason,
                )
            )
            continue

        run_condition_skip_reason = _run_condition_skip_reason(
            stage,
            matched_manifests,
            resolved_inputs,
            bundle.params_base,
        )
        if run_condition_skip_reason is not None:
            records = _output_records(
                stage,
                products=[],
                skipped_reason=run_condition_skip_reason,
            )
            stage_products[stage.name] = []
            stage_records.append(
                BundleStageExecutionRecord(
                    name=stage.name,
                    kind=stage.kind,
                    status=_record_status(records),
                    depends_on=list(stage.depends_on),
                    inputs=inputs,
                    outputs=records,
                    reason=run_condition_skip_reason,
                )
            )
            continue

        input_groups = _stage_input_groups(stage, inputs)
        if stage.kind == "evaluation":
            products = _execute_evaluation_stage(
                stage,
                input_groups,
                root=root_path,
                issues=issue_refs,
                bundle=bundle,
                execution_context=execution_context,
            )
        elif stage.kind == "analysis":
            products = _execute_analysis_stage(
                stage,
                input_groups,
                root=root_path,
                issues=issue_refs,
                bundle=bundle,
                fig_dump_path=fig_dump_path,
                fig_dump_formats=fig_dump_formats,
                execution_context=execution_context,
            )
        elif stage.kind == "materialization":
            products = _execute_materialization_stage(
                stage,
                input_groups,
                root=root_path,
                issues=issue_refs,
                bundle=bundle,
                fig_dump_path=fig_dump_path,
                fig_dump_formats=fig_dump_formats,
                execution_context=execution_context,
            )
        elif stage.kind == "figure":
            products = _execute_figure_stage(
                stage,
                input_groups,
                root=root_path,
                issues=issue_refs,
                bundle=bundle,
            )
        elif stage.kind == "report":
            products = _execute_report_stage(
                stage,
                input_groups,
                root=root_path,
                issues=issue_refs,
                bundle=bundle,
            )
        else:  # pragma: no cover - Literal validation keeps this unreachable.
            raise ValueError(f"Unsupported bundle stage kind {stage.kind!r}")

        records = _output_records(stage, products=products)
        manifest_refs = [product.manifest_ref for product in products if product.manifest_ref]
        regeneration_specs = [
            product.regeneration_spec
            for product in products
            if product.regeneration_spec is not None
        ]
        stage_products[stage.name] = products
        stage_records.append(
            BundleStageExecutionRecord(
                name=stage.name,
                kind=stage.kind,
                status=_record_status(records),
                depends_on=list(stage.depends_on),
                inputs=inputs,
                manifest_refs=list(manifest_refs),
                artifact_groups=_artifact_groups(products),
                outputs=records,
                regeneration_specs=regeneration_specs,
            )
        )

    report_outputs = [
        output for record in stage_records if record.kind == "report" for output in record.outputs
    ]
    return StagedAnalysisBundleExecution(
        bundle_name=bundle.name,
        matched_run_ids=[manifest.id for manifest in matched_manifests],
        stages=stage_records,
        report_outputs=report_outputs,
        metadata=dict(bundle.metadata),
    )
