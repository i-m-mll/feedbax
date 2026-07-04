"""Human-authored analysis bundles over manifest-canonical run records."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.evaluation import execute_evaluation_run_spec
from feedbax.analysis.materialization import ContextMaterializer
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
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ArtifactRef,
    AnyManifest,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    Provenance,
    RegenerationCommand,
    RegenerationSpec,
    ReportManifest,
    ReportSpec,
    SpecPayload,
    StrictModel,
    collect_git_provenance,
    default_manifest_root,
    load_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.persistence.manifest_index import iter_manifest_files
from feedbax.plugins import EXPERIMENT_REGISTRY
from feedbax.plugins.registry import ExperimentRegistry

ANALYSIS_BUNDLE_SCHEMA_ID = "feedbax.spec.analysis_bundle"
ANALYSIS_BUNDLE_SCHEMA_VERSION = "feedbax.spec.analysis_bundle.v2"
ANALYSIS_BUNDLE_EXECUTION_SCHEMA_ID = "feedbax.manifest.analysis_bundle_execution"
ANALYSIS_BUNDLE_EXECUTION_SCHEMA_VERSION = "feedbax.manifest.analysis_bundle_execution.v1"

AnalysisBundleMode = Literal["per-run", "grouped"]
BundleStageKind = Literal["evaluation", "analysis", "materialization", "report"]
BundleOutputStatus = Literal["materialized", "skipped", "missing", "not_applicable"]


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
    report_type: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    states_custody: Literal["cache", "durable"] | None = None
    requested_outputs: list[str] = Field(default_factory=list)
    input_requirements: list[Any] = Field(default_factory=list)
    outputs: list[BundleStageOutputSpec] = Field(default_factory=list)
    skip_reason: str | None = None
    not_applicable_reason: str | None = None
    run_condition: Expr | None = None

    @model_validator(mode="after")
    def _validate_stage_payload(self) -> "BundleStageSpec":
        no_static_status = self.skip_reason is None and self.not_applicable_reason is None
        if self.kind == "evaluation" and not self.evaluation_type and no_static_status:
            raise ValueError(f"evaluation bundle stage {self.name!r} requires evaluation_type")
        if self.kind == "analysis" and not self.analysis_type and no_static_status:
            raise ValueError(f"analysis bundle stage {self.name!r} requires analysis_type")
        if self.kind == "materialization" and not self.analysis_type and no_static_status:
            raise ValueError(
                f"materialization bundle stage {self.name!r} requires analysis_type"
            )
        if self.kind == "report" and not self.report_type and no_static_status:
            raise ValueError(f"report bundle stage {self.name!r} requires report_type")
        if self.skip_reason and any(output.required for output in self.outputs):
            raise ValueError(
                f"bundle stage {self.name!r} cannot skip required outputs"
            )
        if self.run_condition is not None and self.skip_reason is not None:
            raise ValueError(
                f"bundle stage {self.name!r} cannot combine static skip_reason "
                "with run_condition"
            )
        return self


class AnalysisBundleSpec(StrictModel):
    """A named set of analysis spec templates and manifest applicability rules."""

    schema_id: str = ANALYSIS_BUNDLE_SCHEMA_ID
    schema_version: str = ANALYSIS_BUNDLE_SCHEMA_VERSION
    name: str
    description: str | None = None
    predicate: ManifestPredicate = Field(default_factory=ManifestPredicate)
    templates: list[AnalysisSpecTemplate] = Field(default_factory=list)
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
        for stage in self.stages:
            for dependency in stage.depends_on_roles:
                if dependency.stage not in seen:
                    raise ValueError(
                        f"bundle stage {stage.name!r} depends_on_roles entry for "
                        f"stage {dependency.stage!r} must refer to an earlier stage"
                    )
            seen.add(stage.name)
        return self


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
    artifact_refs_by_alias: dict[str, tuple[ArtifactRef, ...]] = field(
        default_factory=dict
    )


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


def _params_for_stage(stage: BundleStageSpec) -> dict[str, Any]:
    params = dict(stage.params)
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
    manifest: EvaluationRunManifest | AnalysisRunManifest | ReportManifest,
    path: Path,
    role: str,
) -> ParentRef:
    return ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role=role,
        uri=str(path),
    )


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
    manifest: AnalysisRunManifest | ReportManifest,
    regeneration_payload: SpecPayload,
    *,
    root: Path,
) -> tuple[AnalysisRunManifest | ReportManifest, Path]:
    updated = manifest.model_copy(
        update={"regeneration_specs": [*manifest.regeneration_specs, regeneration_payload]}
    )
    path = write_manifest(updated, root=root)
    return updated, path


def _artifact_parent_ref(artifact: ArtifactRef, *, role: str) -> ParentRef:
    artifact_id = (
        artifact.artifact_id
        or artifact.sha256
        or artifact.uri
        or artifact.logical_name
    )
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
) -> ResolvedStageInputs:
    inputs: list[ParentRef] = []
    artifacts_by_alias: dict[str, tuple[ArtifactRef, ...]] = {}
    if stage.include_bundle_inputs or (not stage.depends_on and not stage.depends_on_roles):
        inputs.extend(_parent_ref_for_manifest(manifest) for manifest in matched_manifests)
    for dependency in stage.depends_on:
        products = stage_products.get(dependency)
        if products is None:
            raise ValueError(
                f"Bundle stage {stage.name!r} depends on unknown stage {dependency!r}"
            )
        inputs.extend(product.manifest_ref for product in products if product.manifest_ref)
    for dependency in stage.depends_on_roles:
        products = stage_products.get(dependency.stage)
        if products is None:
            raise ValueError(
                f"Bundle stage {stage.name!r} depends on unknown stage "
                f"{dependency.stage!r}"
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
) -> ExpressionContext:
    items: dict[str, ContextItem] = {
        "params": ContextItem(kind="params", payload=_params_for_stage(stage)),
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
            payload=[
                artifact.model_dump(mode="json", exclude_none=True)
                for artifact in artifacts
            ],
        )
    return ExpressionContext(items=items)


def _run_condition_skip_reason(
    stage: BundleStageSpec,
    matched_manifests: Sequence[AnyManifest],
    resolved_inputs: ResolvedStageInputs,
) -> str | None:
    if stage.run_condition is None:
        return None
    ctx = _stage_expression_context(stage, matched_manifests, resolved_inputs)
    result = evaluate_expr(stage.run_condition, ctx)
    if result:
        return None
    return (
        "run_condition evaluated false: "
        f"{canonical_expression_json(stage.run_condition)} -> false"
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


def _output_records(
    stage: BundleStageSpec,
    *,
    products: Sequence[StageMaterialization],
    skipped_reason: str | None = None,
    not_applicable_reason: str | None = None,
) -> list[BundleStageOutputRecord]:
    output_specs = stage.outputs or [BundleStageOutputSpec(role="manifest")]
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


def _execute_evaluation_stage(
    stage: BundleStageSpec,
    input_groups: Sequence[Sequence[ParentRef]],
    *,
    root: Path,
    issues: Sequence[str],
    bundle: AnalysisBundleSpec,
) -> list[StageMaterialization]:
    products: list[StageMaterialization] = []
    for inputs in input_groups:
        spec = EvaluationRunSpec(
            evaluation_type=str(stage.evaluation_type),
            inputs=list(inputs),
            params=_params_for_stage(stage),
        )
        manifest, path = execute_evaluation_run_spec(
            spec,
            root=root,
            issues=list(issues),
            metadata={
                "bundle": {
                    "name": bundle.name,
                    "stage": stage.name,
                    "schema_id": bundle.schema_id,
                    "schema_version": bundle.schema_version,
                }
            },
            force=True,
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
) -> list[StageMaterialization]:
    products: list[StageMaterialization] = []
    for inputs in input_groups:
        spec = AnalysisRunSpec(
            analysis_type=str(stage.analysis_type),
            inputs=list(inputs),
            input_requirements=stage.input_requirements,
            params=_params_for_stage(stage),
        )
        base_provenance = collect_git_provenance()
        base_provenance.parents = list(inputs)
        manifest, path = execute_analysis_run_spec(
            spec,
            root=root,
            issues=list(issues),
            provenance=base_provenance,
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
        manifest_ref = _manifest_ref(manifest, path, "analysis_run")
        regeneration_payload = _stage_regeneration_payload(
            stage,
            inputs=inputs,
            outputs=[manifest_ref, *manifest.artifacts],
            issues=issues,
        )
        updated_manifest, updated_path = _with_regeneration_spec(
            manifest,
            regeneration_payload,
            root=root,
        )
        products.append(
            StageMaterialization(
                manifest_ref=_manifest_ref(updated_manifest, updated_path, "analysis_run"),
                artifacts=tuple(updated_manifest.artifacts),
                manifest_path=updated_path,
                regeneration_spec=regeneration_payload,
            )
        )
    return products


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
) -> list[StageMaterialization]:
    products: list[StageMaterialization] = []
    for inputs in input_groups:
        spec = AnalysisRunSpec(
            analysis_type=str(stage.analysis_type),
            inputs=list(inputs),
            input_requirements=stage.input_requirements,
            params=_params_for_stage(stage),
        )
        base_provenance = collect_git_provenance()
        base_provenance.parents = list(inputs)
        manifest, path = execute_analysis_run_spec(
            spec,
            root=root,
            issues=list(issues),
            provenance=base_provenance,
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
        manifest_ref = _manifest_ref(manifest, path, "analysis_run")
        regeneration_payload = _stage_regeneration_payload(
            stage,
            inputs=inputs,
            outputs=[manifest_ref, *manifest.artifacts],
            issues=issues,
        )
        updated_manifest, updated_path = _with_regeneration_spec(
            manifest,
            regeneration_payload,
            root=root,
        )
        products.append(
            StageMaterialization(
                manifest_ref=_manifest_ref(updated_manifest, updated_path, "analysis_run"),
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
    products: list[StageMaterialization] = []
    for index, inputs in enumerate(input_groups):
        stage_params = _params_for_stage(stage)
        bundle_metadata = {
            "name": bundle.name,
            "stage": stage.name,
            "index": index,
            "schema_id": bundle.schema_id,
            "schema_version": bundle.schema_version,
        }
        spec = ReportSpec(
            report_type=str(stage.report_type or BUNDLE_SUMMARY_REPORT_TYPE),
            inputs=list(inputs),
            params={
                "stage_params": stage_params,
                "bundle": bundle_metadata,
            },
            narrative=stage.params.get("narrative"),
        )
        manifest, path = execute_report_spec(
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
        manifest_ref = _manifest_ref(manifest, path, "report")
        regeneration_payload = _stage_regeneration_payload(
            stage,
            inputs=inputs,
            outputs=[manifest_ref, *manifest.artifacts],
            issues=issues,
        )
        updated_manifest, updated_path = _with_regeneration_spec(
            manifest,
            regeneration_payload,
            root=root,
        )
        products.append(
            StageMaterialization(
                manifest_ref=_manifest_ref(updated_manifest, updated_path, "report"),
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


def execute_staged_analysis_bundle(
    bundle: AnalysisBundleSpec,
    *,
    root: Path | str | None = None,
    run_ids: Iterable[str] | None = None,
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
    if not bundle.stages:
        raise ValueError(f"Analysis bundle {bundle.name!r} has no staged plan")

    root_path = Path(root) if root is not None else default_manifest_root()
    issue_refs = list(issues or [])
    matched_manifests = select_bundle_manifests(bundle, root_path, run_ids=run_ids)
    stage_products: dict[str, list[StageMaterialization]] = {}
    stage_records: list[BundleStageExecutionRecord] = []

    for stage in bundle.stages:
        resolved_inputs = _resolve_stage_inputs(stage, matched_manifests, stage_products)
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
        output
        for record in stage_records
        if record.kind == "report"
        for output in record.outputs
    ]
    return StagedAnalysisBundleExecution(
        bundle_name=bundle.name,
        matched_run_ids=[manifest.id for manifest in matched_manifests],
        stages=stage_records,
        report_outputs=report_outputs,
        metadata=dict(bundle.metadata),
    )
