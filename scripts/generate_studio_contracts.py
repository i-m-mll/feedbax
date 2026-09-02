"""Generate Studio frontend contracts from Python Pydantic models."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel
from pydantic.fields import FieldInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feedbax.contracts.component import (
    ComponentDefinition,
    ComponentIdentity,
    ComponentMigrationInfo,
    DynamicPortPolicy,
    PortType,
    PortTypeSpec,
)
from feedbax.contracts.domain import (
    DomainCompileReport,
    DomainDiagnostic,
    DomainMeta,
    DomainRegistryPayload,
    DomainTheme,
    EditorCapability,
)
from feedbax.contracts.acausal import (
    AcausalConnectionSpec,
    AcausalGraphSpec,
    RootFinderSpec,
    SolverConfigSpec,
)
from feedbax.contracts.representation import (
    RepresentationAnchorSpec,
    RepresentationElementSpec,
    RepresentationFrameProvider,
    RepresentationLiteralBinding,
    RepresentationMusclePathGeometrySpec,
    RepresentationMusclePathPointSpec,
    RepresentationMusclePathSpec,
    RepresentationParamPathBinding,
    RepresentationPlanarChainSpec,
    RepresentationReferencePoseSpec,
    RepresentationReachabilitySpec,
    RepresentationSpec,
    RepresentationStateAnchorSelectorBinding,
    RepresentationStyleSpec,
    RepresentationTrialSpecPathBinding,
    RepresentationValidationIssue,
)
from feedbax.contracts.graph import (
    AdditiveGraphChannelAdapterSpec,
    AdditiveGraphChannelTargetSpec,
    AnalysisInputConsumerSpec,
    AnalysisDataProductRequirement,
    AnalysisInputRequirement,
    AnalysisPageSpec,
    AssemblyViewUIState,
    BarnacleSpec,
    CanvasPositionSpec,
    CanvasSizeSpec,
    CanvasViewportSpec,
    ComponentSpec,
    DerivedDimensionRuleSpec,
    EdgeRouting,
    EdgeRoutingPoint,
    EdgeUIState,
    GraphMetadata,
    GraphProject,
    GraphSpec,
    GraphUIState,
    NodeUIState,
    ParamSchema,
    ParameterConstraintSpec,
    RetainedObservableSpec,
    RetainedObservableTargetSpec,
    RetentionPolicySpec,
    SemanticAnchor,
    StudioBiomechanicsSpec,
    StudioArtifactRef,
    StudioCollectionRef,
    StudioInterventionTransformSpec,
    StudioInterventionValueBounds,
    StudioManifestRef,
    StudioPersistenceDocument,
    StudioScenarioSpec,
    StudioSelectorRef,
    StudioStageSpec,
    StudioTaskBinding,
    StudioTaskBindingSpec,
    StudioTaskDataSpec,
    StudioTaskEpochSpec,
    StudioTaskTimelineSegmentSpec,
    StudioTaskTimelineSignalSpec,
    StudioTaskTimelineSpec,
    StudioValueEnumerableSpec,
    StudioValueVariationSpec,
    StudioValidationIssue,
    StudioValidationState,
    StudioValueSpec,
    StudioWorkspaceSpec,
    WorkspaceDocument,
    TapSpec,
    TapPositionSpec,
    TapTransform,
    TapUIState,
    UserPortSpec,
    ValidationError,
    ValidationResult,
    ValidationWarning,
    WireSpec,
)
from feedbax.contracts.array_values import (
    ConstantArrayValueSpec,
    SparseCooArrayValueSpec,
    SparseCooEntrySpec,
)
from feedbax.contracts.studio_api import (
    AnalysisBundleDryRunPayload,
    AnalysisBundleDryRunRequest,
    AnalysisBundleDryRunResponse,
    AnalysisClassInfo,
    AnalysisJobStatusPayload,
    AnalysisJobStatusResponse,
    AnalysisPackageInfo,
    AnalysisPackagesPayload,
    AnalysisPackagesResponse,
    ComponentDetailResponse,
    DomainListResponse,
    ComponentListPayload,
    ComponentListResponse,
    ComponentRefreshPayload,
    ComponentRefreshResponse,
    PenzaiBuilderInfo,
    PenzaiBuilderListPayload,
    PenzaiBuilderListResponse,
    PenzaiInspectorPayload,
    PenzaiInspectorResponse,
    PenzaiNodeRequest,
    GenerateAnalysisPayload,
    GenerateAnalysisRequest,
    GenerateAnalysisResponse,
    GraphCreatePayload,
    GraphCreateResponse,
    GraphDetailPayload,
    GraphDetailResponse,
    GraphExportPayload,
    GraphExportResponse,
    GraphListItem,
    GraphListPayload,
    GraphListResponse,
    GraphUpdatePayload,
    GraphUpdateResponse,
    GraphValidationResponse,
    SuccessPayload,
    SuccessResponse,
    TrainingCompleteEvent,
    TrainingErrorEvent,
    TrainingLogEvent,
    TrainingProgressEvent,
    TrainingResyncEvent,
    TrainingStartPayload,
    TrainingStartResponse,
    TrainingStatusPayload,
    TrainingStatusResponse,
    TrainingTrajectoryEvent,
    TrainingTrajectoryPayload,
    WorkerConnectEnvelope,
    WorkerConnectResponse,
    WorkerStatusEnvelope,
    WorkerStatusResponse,
    BundleMissingRoleRecord,
    BundleStageDryRunOutputRecord,
    BundleStageDryRunRecord,
    AnalysisBundleDryRunResult,
)
from feedbax.contracts.training import (
    BatchScheduleOriginSpec,
    EarlyStoppingSpec,
    LossTermSpec,
    LrScheduleSpec,
    OptimizerSpec,
    TaskSpec,
    TimeAggregationSpec,
    TrainingConfig,
    TrainingSpec,
)
from feedbax.contracts.manifest import ParentRef
from feedbax.contracts.selection import (
    ManifestPredicate,
    SelectionPreview,
    SelectionRefreshDiff,
    SelectionSpec,
    TopKByMetricPerGroup,
)
from feedbax.contracts.value_schema import ValueSchema
from feedbax.contracts.workspace_replay import (
    ResolvedWorkspaceReplayScene,
    WorkspaceReplayImportedArtifact,
    WorkspaceReplayManifestRefs,
    WorkspaceReplayOverlayChannel,
    WorkspaceReplayProduct,
    WorkspaceReplayRequiredSelector,
    WorkspaceReplaySampleAxis,
    WorkspaceReplayTrack,
    WorkspaceReplayTrial,
    WorkspaceReplayTrialIdentity,
    WorkspaceReplayTrialSpecSnapshot,
    WorkspaceReplayWarning,
)
from feedbax.contracts.studio_refinements import (
    CROSS_FIELD_REFINEMENTS,
    REGISTERED_CONSTRAINT_VALIDATORS,
)
from feedbax.web.api.training import (
    ProbeResponse,
    ValidateLossResponse,
    ValidationErrorResponse,
)
from feedbax.web.api.execution import (
    SampledTaskTrial,
    TaskTimelineCue,
    TaskTrialSampleRequest,
    TaskTrialSampleResponse,
)
from feedbax.web.api.runs import (
    CreateEvalRunRequest,
    EvalRunInfo,
    ManifestImportRequest,
    ManifestImportResponse,
    SelectionPreviewRequest,
    SelectionRefreshRequest,
    TrainingRunCompareRequest,
    TrainingRunCompareResponse,
    TrainingRunCompareRow,
    TrainingRunInfo,
)
from feedbax.web.models.inspection import (
    CycleAnnotationModel,
    InlineTreescopeRequest,
    InspectionStatusResponse,
    TreescopeRequest,
    TreescopeResponse,
)
from feedbax.web.models.statistics import (
    DiagnosticCheck,
    DiagnosticsResponse,
    GroupStatistics,
    HistogramBin,
    HistogramGroup,
    HistogramResponse,
    MetricSummary,
    ScatterPoint,
    ScatterResponse,
    StatisticsResponse,
    TimeseriesPercentiles,
    TimeseriesResponse,
)
from feedbax.web.models.trajectory import (
    DatasetInfo,
    FilterResult,
    TrajectoryData,
    TrajectoryMetadata,
)

OUTPUT = REPO_ROOT / "web" / "src" / "generated" / "studioContracts.ts"
NONE_TYPE = type(None)

MODEL_TYPES: list[type[BaseModel]] = [
    SparseCooEntrySpec,
    SparseCooArrayValueSpec,
    ConstantArrayValueSpec,
    ParamSchema,
    ComponentSpec,
    AcausalConnectionSpec,
    RootFinderSpec,
    SolverConfigSpec,
    AcausalGraphSpec,
    ParameterConstraintSpec,
    WireSpec,
    AdditiveGraphChannelTargetSpec,
    AdditiveGraphChannelAdapterSpec,
    DerivedDimensionRuleSpec,
    UserPortSpec,
    CanvasPositionSpec,
    CanvasSizeSpec,
    CanvasViewportSpec,
    TapPositionSpec,
    TapTransform,
    TapSpec,
    BarnacleSpec,
    RetentionPolicySpec,
    RetainedObservableTargetSpec,
    RetainedObservableSpec,
    AnalysisInputConsumerSpec,
    AnalysisDataProductRequirement,
    AnalysisInputRequirement,
    GraphMetadata,
    GraphSpec,
    EdgeRoutingPoint,
    EdgeRouting,
    EdgeUIState,
    NodeUIState,
    TapUIState,
    AssemblyViewUIState,
    GraphUIState,
    AnalysisPageSpec,
    StudioValidationIssue,
    StudioValidationState,
    StudioManifestRef,
    StudioArtifactRef,
    StudioCollectionRef,
    ValueSchema,
    StudioValueEnumerableSpec,
    StudioValueVariationSpec,
    StudioValueSpec,
    StudioSelectorRef,
    StudioInterventionValueBounds,
    StudioInterventionTransformSpec,
    StudioTaskEpochSpec,
    StudioTaskTimelineSignalSpec,
    StudioTaskTimelineSegmentSpec,
    StudioTaskTimelineSpec,
    StudioTaskDataSpec,
    StudioTaskBinding,
    StudioTaskBindingSpec,
    StudioBiomechanicsSpec,
    StudioScenarioSpec,
    StudioStageSpec,
    StudioWorkspaceSpec,
    SemanticAnchor,
    WorkspaceDocument,
    StudioPersistenceDocument,
    GraphProject,
    ValidationError,
    ValidationWarning,
    ValidationResult,
    PortType,
    PortTypeSpec,
    ComponentIdentity,
    ComponentMigrationInfo,
    DynamicPortPolicy,
    RepresentationParamPathBinding,
    RepresentationStateAnchorSelectorBinding,
    RepresentationTrialSpecPathBinding,
    RepresentationLiteralBinding,
    RepresentationReferencePoseSpec,
    RepresentationPlanarChainSpec,
    RepresentationMusclePathPointSpec,
    RepresentationMusclePathSpec,
    RepresentationMusclePathGeometrySpec,
    RepresentationFrameProvider,
    RepresentationAnchorSpec,
    RepresentationStyleSpec,
    RepresentationElementSpec,
    RepresentationReachabilitySpec,
    RepresentationSpec,
    RepresentationValidationIssue,
    EditorCapability,
    DomainTheme,
    DomainMeta,
    DomainDiagnostic,
    DomainRegistryPayload,
    DomainCompileReport,
    WorkspaceReplayWarning,
    WorkspaceReplayTrialIdentity,
    WorkspaceReplaySampleAxis,
    WorkspaceReplayTrack,
    WorkspaceReplayOverlayChannel,
    WorkspaceReplayTrialSpecSnapshot,
    WorkspaceReplayManifestRefs,
    WorkspaceReplayTrial,
    WorkspaceReplayImportedArtifact,
    WorkspaceReplayProduct,
    WorkspaceReplayRequiredSelector,
    ResolvedWorkspaceReplayScene,
    ComponentDefinition,
    BatchScheduleOriginSpec,
    LrScheduleSpec,
    OptimizerSpec,
    TimeAggregationSpec,
    LossTermSpec,
    EarlyStoppingSpec,
    TrainingSpec,
    TaskSpec,
    TrainingConfig,
    TaskTimelineCue,
    SampledTaskTrial,
    TaskTrialSampleRequest,
    TaskTrialSampleResponse,
    ParentRef,
    TopKByMetricPerGroup,
    ManifestPredicate,
    SelectionSpec,
    SelectionPreview,
    SelectionRefreshDiff,
    SuccessPayload,
    SuccessResponse,
    GraphListItem,
    GraphListPayload,
    GraphListResponse,
    GraphCreatePayload,
    GraphCreateResponse,
    GraphUpdatePayload,
    GraphUpdateResponse,
    GraphDetailPayload,
    GraphDetailResponse,
    GraphValidationResponse,
    GraphExportPayload,
    GraphExportResponse,
    ComponentListPayload,
    ComponentListResponse,
    ComponentDetailResponse,
    ComponentRefreshPayload,
    ComponentRefreshResponse,
    DomainListResponse,
    PenzaiBuilderInfo,
    PenzaiBuilderListPayload,
    PenzaiBuilderListResponse,
    PenzaiNodeRequest,
    PenzaiInspectorPayload,
    PenzaiInspectorResponse,
    TrainingStartPayload,
    TrainingStartResponse,
    TrainingStatusPayload,
    TrainingStatusResponse,
    WorkerConnectResponse,
    WorkerConnectEnvelope,
    WorkerStatusResponse,
    WorkerStatusEnvelope,
    AnalysisClassInfo,
    AnalysisPackageInfo,
    AnalysisPackagesPayload,
    AnalysisPackagesResponse,
    BundleMissingRoleRecord,
    BundleStageDryRunOutputRecord,
    BundleStageDryRunRecord,
    AnalysisBundleDryRunResult,
    AnalysisBundleDryRunRequest,
    AnalysisBundleDryRunPayload,
    AnalysisBundleDryRunResponse,
    GenerateAnalysisRequest,
    GenerateAnalysisPayload,
    GenerateAnalysisResponse,
    AnalysisJobStatusPayload,
    AnalysisJobStatusResponse,
    TrainingProgressEvent,
    TrainingLogEvent,
    TrainingTrajectoryPayload,
    TrainingTrajectoryEvent,
    TrainingCompleteEvent,
    TrainingErrorEvent,
    TrainingResyncEvent,
    ProbeResponse,
    ValidationErrorResponse,
    ValidateLossResponse,
    TrainingRunInfo,
    EvalRunInfo,
    CreateEvalRunRequest,
    TrainingRunCompareRequest,
    TrainingRunCompareRow,
    TrainingRunCompareResponse,
    ManifestImportRequest,
    ManifestImportResponse,
    SelectionPreviewRequest,
    SelectionRefreshRequest,
    DatasetInfo,
    TrajectoryMetadata,
    FilterResult,
    TrajectoryData,
    MetricSummary,
    GroupStatistics,
    StatisticsResponse,
    TimeseriesPercentiles,
    TimeseriesResponse,
    HistogramBin,
    HistogramGroup,
    HistogramResponse,
    ScatterPoint,
    ScatterResponse,
    DiagnosticCheck,
    DiagnosticsResponse,
    CycleAnnotationModel,
    TreescopeRequest,
    InlineTreescopeRequest,
    TreescopeResponse,
    InspectionStatusResponse,
]

EVENT_MODEL_NAMES = [
    "TrainingProgressEvent",
    "TrainingLogEvent",
    "TrainingTrajectoryEvent",
    "TrainingCompleteEvent",
    "TrainingErrorEvent",
    "TrainingResyncEvent",
]

REFINEMENT_HELPERS = """function hasDuplicate(values: readonly unknown[]): boolean {
  return new Set(values).size !== values.length;
}

function validDomainId(value: string): boolean {
  if (!value.startsWith('feedbax.domain.')) return false;
  const suffix = value.slice('feedbax.domain.'.length);
  return suffix.length > 0 && suffix.split('.').every((part) => part.length > 0);
}

function validGeneratedPortTemplate(value: string): boolean {
  return value.includes('{index}') && !/[{}]/.test(value.split('{index}').join(''));
}

function containsArrayValueEnvelope(value: unknown): boolean {
  if (Array.isArray(value)) return value.some(containsArrayValueEnvelope);
  if (value === null || typeof value !== 'object') return false;
  const record = value as Record<string, unknown>;
  if (
    record.schema_id === 'feedbax.spec.component_param.array_value' ||
    (typeof record.schema_version === 'string' &&
      record.schema_version.startsWith('feedbax.spec.component_param.array_value.'))
  ) return true;
  return Object.values(record).some(containsArrayValueEnvelope);
}

function invalidTypedParamEnvelope(value: unknown): boolean {
  if (Array.isArray(value)) return value.some(invalidTypedParamEnvelope);
  if (value === null || typeof value !== 'object') return false;
  const record = value as Record<string, unknown>;
  const claimsArray =
    record.schema_id === 'feedbax.spec.component_param.array_value' ||
    (typeof record.schema_version === 'string' &&
      record.schema_version.startsWith('feedbax.spec.component_param.array_value.'));
  if (claimsArray) {
    return !SparseCooArrayValueSpecSchema.safeParse(record).success &&
      !ConstantArrayValueSpecSchema.safeParse(record).success;
  }
  const claimsValue =
    record.schema_id === 'feedbax.spec.studio.value' ||
    (typeof record.schema_version === 'string' &&
      (record.schema_version.startsWith('feedbax.spec.studio.value.') ||
        record.schema_version.startsWith('feedbax.studio.value.')));
  if (claimsValue) return !StudioValueSpecSchema.safeParse(record).success;
  return Object.values(record).some(invalidTypedParamEnvelope);
}

function arrayScalarInvalid(value: unknown, dtype: string, nonfinite: string): boolean {
  if (typeof value === 'string') return nonfinite !== 'allow' || !dtype.startsWith('float');
  if (typeof value === 'boolean') return dtype !== 'bool';
  if (dtype === 'bool') return true;
  if (typeof value !== 'number' || !Number.isFinite(value)) return true;
  const integerRanges: Record<string, readonly [number, number]> = {
    int8: [-128, 127], int16: [-32768, 32767], int32: [-2147483648, 2147483647],
    int64: [Number.MIN_SAFE_INTEGER, Number.MAX_SAFE_INTEGER], uint8: [0, 255],
    uint16: [0, 65535], uint32: [0, 4294967295],
    uint64: [0, Number.MAX_SAFE_INTEGER],
  };
  const integerRange = integerRanges[dtype];
  if (integerRange) {
    return !Number.isInteger(value) || value < integerRange[0] || value > integerRange[1];
  }
  const floatMax: Record<string, number> = {
    float16: 65504, float32: 3.4028234663852886e38, float64: Number.MAX_VALUE,
  };
  return dtype in floatMax && Math.abs(value) > floatMax[dtype];
}
"""

CONTRACT_MODEL_NAMES = [
    "GraphListResponse",
    "GraphCreateResponse",
    "GraphUpdateResponse",
    "GraphDetailResponse",
    "GraphValidationResponse",
    "GraphExportResponse",
    "ComponentListResponse",
    "ComponentDetailResponse",
    "ComponentRefreshResponse",
    "DomainListResponse",
    "DomainCompileReport",
    "PenzaiBuilderListResponse",
    "PenzaiInspectorResponse",
    "PenzaiNodeRequest",
    "TrainingStartResponse",
    "TrainingStatusResponse",
    "SuccessResponse",
    "WorkerConnectEnvelope",
    "WorkerStatusEnvelope",
    "AnalysisPackagesResponse",
    "AnalysisBundleDryRunResponse",
    "GenerateAnalysisResponse",
    "AnalysisJobStatusResponse",
    "TrainingRunInfo",
    "EvalRunInfo",
    "CreateEvalRunRequest",
    "TrainingRunCompareResponse",
    "ManifestImportResponse",
    "SelectionSpec",
    "SelectionPreview",
    "SelectionRefreshDiff",
    "SelectionPreviewRequest",
    "SelectionRefreshRequest",
    "DatasetInfo",
    "TrajectoryMetadata",
    "FilterResult",
    "TrajectoryData",
    "StatisticsResponse",
    "TimeseriesResponse",
    "HistogramResponse",
    "ScatterResponse",
    "DiagnosticsResponse",
    "TreescopeRequest",
    "InlineTreescopeRequest",
    "TreescopeResponse",
    "InspectionStatusResponse",
    "ProbeResponse",
    "ValidateLossResponse",
    "TaskTrialSampleRequest",
    "TaskTrialSampleResponse",
    "TrainingWebSocketEvent",
]

MIGRATION_OR_NORMALIZATION_VALIDATORS = {
    ("ComponentSpec", "validate_value_spec_params"),
    ("StudioValueSpec", "migrate_legacy_value_spec"),
    ("StudioTaskBinding", "reject_legacy_task_binding_field_names"),
    ("StudioTaskBindingSpec", "reject_legacy_task_binding_contract"),
    ("GraphProject", "drop_unparseable_compile_reports"),
    ("PortType", "migrate_legacy_signal_port"),
    ("ComponentDefinition", "migrate_legacy_definition"),
    ("LrScheduleSpec", "_migrate_v1_origin"),
    ("SelectionSpec", "_normalize_mode_fields"),
}


def _is_union(annotation: Any) -> bool:
    return get_origin(annotation) in (Union, types.UnionType)


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def _literal_value(value: Any) -> str:
    if value is None:
        return "null"
    return json.dumps(value)


def _unwrap_annotated(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _annotation_metadata(annotation: Any) -> tuple[Any, ...]:
    metadata: list[Any] = []
    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation = args[0]
        for item in args[1:]:
            if isinstance(item, FieldInfo):
                metadata.extend(item.metadata)
            else:
                metadata.append(item)
    return tuple(metadata)


def _unique_metadata(metadata: tuple[Any, ...]) -> tuple[Any, ...]:
    result: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for item in metadata:
        key = (type(item).__name__, repr(item))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return tuple(result)


def ts_type(annotation: Any) -> str:
    """Return a TypeScript type expression for a Python annotation."""

    annotation = _unwrap_annotated(annotation)
    if annotation is Any:
        return "unknown"
    if annotation is NONE_TYPE:
        return "null"
    if annotation is str:
        return "string"
    if annotation is int or annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation is dict:
        return "Record<string, unknown>"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.__name__

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list or (origin is tuple and len(args) == 2 and args[1] is Ellipsis):
        item = args[0] if args else Any
        return f"{ts_type(item)}[]"
    if origin is dict:
        value = args[1] if len(args) == 2 else Any
        return f"Record<string, {ts_type(value)}>"
    if origin is tuple:
        if not args:
            return "unknown[]"
        return "[" + ", ".join(ts_type(arg) for arg in args) + "]"
    if origin is Literal:
        return " | ".join(_literal_value(arg) for arg in args)
    if _is_union(annotation):
        return " | ".join(_dedupe([ts_type(arg) for arg in args]))

    return "unknown"


def _apply_zod_constraints(
    schema: str,
    annotation: Any,
    metadata: tuple[Any, ...],
    *,
    context: str,
) -> str:
    base = _unwrap_annotated(annotation)
    for constraint in metadata:
        name = type(constraint).__name__
        if name == "Strict":
            continue
        if name == "Gt":
            schema += f".gt({_literal_value(constraint.gt)})"
        elif name == "Ge":
            schema += f".gte({_literal_value(constraint.ge)})"
        elif name == "Lt":
            schema += f".lt({_literal_value(constraint.lt)})"
        elif name == "Le":
            schema += f".lte({_literal_value(constraint.le)})"
        elif name == "MultipleOf":
            schema += f".multipleOf({_literal_value(constraint.multiple_of)})"
        elif name == "MinLen":
            schema += f".min({constraint.min_length})"
        elif name == "MaxLen":
            schema += f".max({constraint.max_length})"
        elif name == "_PydanticGeneralMetadata":
            values = vars(constraint)
            unsupported = sorted(set(values) - {"pattern"})
            if unsupported:
                raise ValueError(
                    f"Unprojectable material Pydantic constraint at {context}: "
                    f"{name} fields {unsupported}"
                )
            pattern = values.get("pattern")
            if pattern is not None:
                if base is not str:
                    raise ValueError(
                        f"Pattern constraint at {context} is only supported for strings"
                    )
                schema += f".regex(new RegExp({_literal_value(pattern)}))"
        else:
            raise ValueError(
                f"Unprojectable material Pydantic constraint at {context}: {constraint!r}"
            )
    return schema


def zod_schema(
    annotation: Any,
    *,
    field: FieldInfo | None = None,
    context: str = "annotation",
) -> str:
    """Return a zod expression for a Python annotation."""

    annotation_metadata = _annotation_metadata(annotation)
    field_metadata = () if field is None else tuple(field.metadata)
    metadata = _unique_metadata(annotation_metadata + field_metadata)
    annotation = _unwrap_annotated(annotation)
    if annotation is Any:
        return "z.unknown()"
    if annotation is NONE_TYPE:
        return "z.null()"
    if annotation is str:
        return _apply_zod_constraints("z.string()", annotation, metadata, context=context)
    if annotation is int:
        schema = "z.number().int().safe()"
        return _apply_zod_constraints(schema, annotation, metadata, context=context)
    if annotation is float:
        schema = "z.number().finite()"
        return _apply_zod_constraints(schema, annotation, metadata, context=context)
    if annotation is bool:
        return "z.boolean()"
    if annotation is dict:
        return "z.record(z.string(), z.unknown())"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return f"{annotation.__name__}Schema"

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list:
        item = args[0] if args else Any
        schema = f"z.array({zod_schema(item, context=f'{context}[]')})"
        return _apply_zod_constraints(schema, annotation, metadata, context=context)
    if origin is dict:
        value = args[1] if len(args) == 2 else Any
        return f"z.record(z.string(), {zod_schema(value, context=f'{context}.*')})"
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            schema = f"z.array({zod_schema(args[0], context=f'{context}[]')})"
            return _apply_zod_constraints(schema, annotation, metadata, context=context)
        schema = (
            "z.tuple(["
            + ", ".join(
                zod_schema(arg, context=f"{context}[{index}]") for index, arg in enumerate(args)
            )
            + "])"
        )
        return _apply_zod_constraints(schema, annotation, metadata, context=context)
    if origin is Literal:
        schemas = [f"z.literal({_literal_value(arg)})" for arg in args]
        return schemas[0] if len(schemas) == 1 else f"z.union([{', '.join(schemas)}])"
    if _is_union(annotation):
        non_null = [arg for arg in args if arg is not NONE_TYPE]
        has_null = len(non_null) != len(args)
        if has_null and len(non_null) == 1:
            inner = zod_schema(non_null[0], context=context)
            inner = _apply_zod_constraints(
                inner,
                non_null[0],
                field_metadata,
                context=context,
            )
            return f"{inner}.nullable()"
        if field is not None and field.discriminator:
            schemas = ", ".join(zod_schema(arg, context=context) for arg in args)
            return f"z.discriminatedUnion({_literal_value(field.discriminator)}, [{schemas}])"
        return f"z.union([{', '.join(zod_schema(arg, context=context) for arg in args)}])"

    return "z.unknown()"


def emit_interface(model: type[BaseModel]) -> str:
    hints = model_type_hints(model)
    lines = [f"export interface {model.__name__} {{"]
    for name, field in model.model_fields.items():
        annotation = hints[name]
        optional = "?" if not field.is_required() else ""
        lines.append(f"  {name}{optional}: {ts_type(annotation)};")
    lines.append("}")
    return "\n".join(lines)


def emit_schema(model: type[BaseModel]) -> str:
    _validate_model_validator_projection(model)
    hints = model_type_hints(model)
    lines = [
        f"export const {model.__name__}Schema: z.ZodType<{model.__name__}> = z.lazy(() =>",
        "  z",
        "    .object({",
    ]
    for name, field in model.model_fields.items():
        annotation = hints[name]
        schema = zod_schema(
            annotation,
            field=field,
            context=f"{model.__name__}.{name}",
        )
        if not field.is_required():
            schema = f"{schema}.optional()"
        lines.append(f"      {json.dumps(name)}: {schema},")
    lines.extend(
        [
            "    })",
            "    .strict()",
        ]
    )
    refinements = CROSS_FIELD_REFINEMENTS.get(model.__name__, ())
    if refinements:
        lines.append("    .superRefine((value, ctx) => {")
        for refinement in refinements:
            lines.extend(
                [
                    f"      if ({refinement.typescript_invalid}) {{",
                    "        ctx.addIssue({",
                    "          code: z.ZodIssueCode.custom,",
                    f"          message: {json.dumps(refinement.message)},",
                    "        });",
                    "      }",
                ]
            )
        lines.append("    })")
    lines.append(f") as unknown as z.ZodType<{model.__name__}>;")
    return "\n".join(lines)


def _validate_model_validator_projection(model: type[BaseModel]) -> None:
    decorators = model.__pydantic_decorators__
    constraint_validator_names = {
        name
        for name, decorator in decorators.model_validators.items()
        if decorator.info.mode == "after"
    }
    constraint_validator_names.update(decorators.field_validators)
    constraint_validator_names.update(
        name
        for name, decorator in decorators.model_validators.items()
        if decorator.info.mode == "before"
        and (model.__name__, name) not in MIGRATION_OR_NORMALIZATION_VALIDATORS
    )
    registered_names = REGISTERED_CONSTRAINT_VALIDATORS.get(model.__name__, frozenset())
    unregistered_names = constraint_validator_names - registered_names
    stale_names = registered_names - constraint_validator_names
    if unregistered_names:
        names = ", ".join(f"{model.__name__}.{name}" for name in sorted(unregistered_names))
        raise ValueError(
            "Unprojectable material Pydantic model constraint(s): "
            f"{names}. Add equivalent Python and TypeScript predicates to "
            "feedbax.contracts.studio_refinements.CROSS_FIELD_REFINEMENTS and register "
            "the exact Python validator name."
        )
    if stale_names:
        names = ", ".join(f"{model.__name__}.{name}" for name in sorted(stale_names))
        raise ValueError(
            f"Stale Studio constraint-validator registration(s): {names}. "
            "Update the shared refinement registry before generation."
        )
    if constraint_validator_names and model.__name__ not in CROSS_FIELD_REFINEMENTS:
        raise ValueError(
            f"Registered Python validators for {model.__name__} have no TypeScript refinements"
        )


def model_type_hints(model: type[BaseModel]) -> dict[str, Any]:
    """Resolve model type hints, including acausal forward references."""
    module_globals = dict(vars(sys.modules[model.__module__]))
    module_globals.update(
        {
            "ComponentSpec": ComponentSpec,
            "GraphMetadata": GraphMetadata,
            "AcausalGraphSpec": AcausalGraphSpec,
        }
    )
    return get_type_hints(model, globalns=module_globals, include_extras=True)


def model_dependencies(model: type[BaseModel]) -> tuple[type[BaseModel], ...]:
    """Return Pydantic models referenced by a generated model's fields."""
    dependencies: list[type[BaseModel]] = []

    def collect(annotation: Any) -> None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation is not model and annotation not in dependencies:
                dependencies.append(annotation)
            return
        for argument in get_args(annotation):
            collect(argument)

    for annotation in model_type_hints(model).values():
        collect(annotation)
    return tuple(dependencies)


def generate() -> str:
    interfaces = "\n\n".join(emit_interface(model) for model in MODEL_TYPES)
    schemas = "\n\n".join(emit_schema(model) for model in MODEL_TYPES)
    event_type = " | ".join(EVENT_MODEL_NAMES)
    event_schema = (
        f"export type TrainingWebSocketEvent = {event_type};\n\n"
        "export const TrainingWebSocketEventSchema: z.ZodType<TrainingWebSocketEvent> = "
        f"z.union([{', '.join(name + 'Schema' for name in EVENT_MODEL_NAMES)}]) "
        "as unknown as z.ZodType<TrainingWebSocketEvent>;"
    )
    contract_entries = "\n".join(f"  {name}: {name}Schema," for name in CONTRACT_MODEL_NAMES)
    type_entries = "\n".join(f"  {name}: {name};" for name in CONTRACT_MODEL_NAMES)
    return (
        "/* eslint-disable */\n"
        "// Generated by scripts/generate_studio_contracts.py. Do not edit by hand.\n\n"
        "import { z } from 'zod';\n\n"
        "export type JsonPrimitive = string | number | boolean | null;\n"
        "export type ParamValue = JsonPrimitive | unknown[] | Record<string, unknown>;\n\n"
        f"{interfaces}\n\n"
        f"{REFINEMENT_HELPERS}\n"
        f"{schemas}\n\n"
        f"{event_schema}\n\n"
        "export const contractSchemas = {\n"
        f"{contract_entries}\n"
        "} as const;\n\n"
        "export type ContractName = keyof typeof contractSchemas;\n"
        "export interface ContractTypeMap {\n"
        f"{type_entries}\n"
        "}\n\n"
        "function formatPath(path: Array<string | number>): string {\n"
        "  return path.length === 0 ? '<root>' : path.join('.');\n"
        "}\n\n"
        "export function formatZodError(name: string, error: z.ZodError): string {\n"
        "  return error.issues\n"
        "    .slice(0, 5)\n"
        "    .map((issue) => `${name}:${formatPath(issue.path)} ${issue.message}`)\n"
        "    .join('; ');\n"
        "}\n\n"
        "export function parseContract<K extends ContractName>(\n"
        "  name: K,\n"
        "  value: unknown,\n"
        "): ContractTypeMap[K] {\n"
        "  const result = contractSchemas[name].safeParse(value);\n"
        "  if (!result.success) {\n"
        "    throw new Error(formatZodError(name, result.error));\n"
        "  }\n"
        "  return result.data as ContractTypeMap[K];\n"
        "}\n"
    )


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(generate(), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
