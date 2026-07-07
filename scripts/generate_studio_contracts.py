"""Generate Studio frontend contracts from Python Pydantic models."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feedbax.contracts.component import (
    ComponentDefinition,
    ComponentIdentity,
    ComponentMigrationInfo,
    PortType,
    PortTypeSpec,
)
from feedbax.contracts.graph import (
    AdditiveGraphChannelAdapterSpec,
    AdditiveGraphChannelTargetSpec,
    AnalysisInputConsumerSpec,
    AnalysisDataProductRequirement,
    AnalysisInputRequirement,
    AnalysisPageSpec,
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
    StudioArtifactRef,
    StudioCollectionRef,
    StudioInterventionTransformSpec,
    StudioInterventionValueBounds,
    StudioManifestRef,
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
    ComponentListPayload,
    ComponentListResponse,
    ComponentRefreshPayload,
    ComponentRefreshResponse,
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
    EarlyStoppingSpec,
    LossTermSpec,
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
from feedbax.web.api.training import (
    ProbeResponse,
    ValidateLossResponse,
    ValidationErrorResponse,
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
    ParamSchema,
    ComponentSpec,
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
    StudioScenarioSpec,
    StudioStageSpec,
    StudioWorkspaceSpec,
    GraphProject,
    ValidationError,
    ValidationWarning,
    ValidationResult,
    PortType,
    PortTypeSpec,
    ComponentIdentity,
    ComponentMigrationInfo,
    ComponentDefinition,
    OptimizerSpec,
    TimeAggregationSpec,
    LossTermSpec,
    EarlyStoppingSpec,
    TrainingSpec,
    TaskSpec,
    TrainingConfig,
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
    "TrainingWebSocketEvent",
]


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


def ts_type(annotation: Any) -> str:
    """Return a TypeScript type expression for a Python annotation."""

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


def zod_schema(annotation: Any) -> str:
    """Return a zod expression for a Python annotation."""

    if annotation is Any:
        return "z.unknown()"
    if annotation is NONE_TYPE:
        return "z.null()"
    if annotation is str:
        return "z.string()"
    if annotation is int:
        return "z.number().int()"
    if annotation is float:
        return "z.number()"
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
        return f"z.array({zod_schema(item)})"
    if origin is dict:
        value = args[1] if len(args) == 2 else Any
        return f"z.record(z.string(), {zod_schema(value)})"
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return f"z.array({zod_schema(args[0])})"
        return "z.tuple([" + ", ".join(zod_schema(arg) for arg in args) + "])"
    if origin is Literal:
        schemas = [f"z.literal({_literal_value(arg)})" for arg in args]
        return schemas[0] if len(schemas) == 1 else f"z.union([{', '.join(schemas)}])"
    if _is_union(annotation):
        non_null = [arg for arg in args if arg is not NONE_TYPE]
        has_null = len(non_null) != len(args)
        if has_null and len(non_null) == 1:
            return f"{zod_schema(non_null[0])}.nullable()"
        return f"z.union([{', '.join(zod_schema(arg) for arg in args)}])"

    return "z.unknown()"


def emit_interface(model: type[BaseModel]) -> str:
    hints = get_type_hints(model, include_extras=True)
    lines = [f"export interface {model.__name__} {{"]
    for name, field in model.model_fields.items():
        annotation = hints[name]
        optional = "?" if not field.is_required() else ""
        lines.append(f"  {name}{optional}: {ts_type(annotation)};")
    lines.append("}")
    return "\n".join(lines)


def emit_schema(model: type[BaseModel]) -> str:
    hints = get_type_hints(model, include_extras=True)
    lines = [
        f"export const {model.__name__}Schema: z.ZodType<{model.__name__}> = z.lazy(() =>",
        "  z",
        "    .object({",
    ]
    for name, field in model.model_fields.items():
        annotation = hints[name]
        schema = zod_schema(annotation)
        if not field.is_required():
            schema = f"{schema}.optional()"
        lines.append(f"      {json.dumps(name)}: {schema},")
    lines.extend(
        [
            "    })",
            "    .strict()",
            f") as unknown as z.ZodType<{model.__name__}>;",
        ]
    )
    return "\n".join(lines)


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
    contract_entries = "\n".join(
        f"  {name}: {name}Schema," for name in CONTRACT_MODEL_NAMES
    )
    type_entries = "\n".join(f"  {name}: {name};" for name in CONTRACT_MODEL_NAMES)
    return (
        "/* eslint-disable */\n"
        "// Generated by scripts/generate_studio_contracts.py. Do not edit by hand.\n\n"
        "import { z } from 'zod';\n\n"
        "export type JsonPrimitive = string | number | boolean | null;\n"
        "export type ParamValue = JsonPrimitive | unknown[] | Record<string, unknown>;\n\n"
        f"{interfaces}\n\n"
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
