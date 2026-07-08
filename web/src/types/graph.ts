import type {
  AcausalGraphSpec,
  BarnacleSpec,
  ComponentSpec,
  EdgeRouting,
  GraphSpec,
  ParamValue,
  TapSpec,
} from '@/generated/studioContracts';

export type {
  AcausalConnectionSpec,
  AcausalGraphSpec,
  AdditiveGraphChannelAdapterSpec,
  AdditiveGraphChannelTargetSpec,
  AnalysisDataProductRequirement,
  AnalysisInputConsumerSpec,
  AnalysisInputRequirement,
  AnalysisPageSpec,
  AssemblyViewUIState,
  BarnacleSpec,
  ComponentSpec,
  DerivedDimensionRuleSpec,
  DomainDiagnostic,
  EdgeRouting,
  EdgeRoutingPoint,
  EdgeUIState,
  GraphMetadata,
  GraphProject,
  GraphSpec,
  GraphUIState,
  NodeUIState,
  ParamSchema,
  ParamValue,
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
  StudioValidationIssue,
  StudioValidationState,
  StudioValueSpec,
  StudioWorkspaceSpec,
  TapSpec,
  TapTransform,
  TapUIState,
  UserPortSpec,
  ValidationError,
  ValidationResult,
  ValidationWarning,
  WireSpec,
} from '@/generated/studioContracts';

export type GraphSubgraphSpec = NonNullable<GraphSpec['subgraphs']>[string];

export function isCausalGraphSpec(
  value: GraphSubgraphSpec | GraphSpec | null | undefined
): value is GraphSpec {
  return Boolean(value && (!value.schema_id || value.schema_id === 'feedbax.spec.graph'));
}

export function isAcausalGraphSpec(
  value: GraphSubgraphSpec | GraphSpec | AcausalGraphSpec | null | undefined
): value is AcausalGraphSpec {
  return Boolean(value && value.schema_id === 'feedbax.spec.acausal_graph');
}

export type ParamPrimitive = number | string | boolean | null;
export type ParamValueObject = Record<string, ParamValue>;
export type ParamValueArray = ParamValue[];

export type BarnacleKind = BarnacleSpec['kind'];
export type BarnacleTiming = BarnacleSpec['timing'];

/** Internal graph carried by a SubgraphNode for the nested preview canvas. */
export interface SubgraphPreview {
  /** Internal React Flow nodes (type: 'component'). */
  nodes: unknown[];
  /** Internal React Flow edges. */
  edges: unknown[];
  /** Port names exposed as inputs on the parent canvas. */
  inputPorts: string[];
  /** Port names exposed as outputs on the parent canvas. */
  outputPorts: string[];
}

export interface GraphNodeData extends Record<string, unknown> {
  label: string;
  spec: ComponentSpec;
  current_domain?: string | null;
  interior_domain?: string | null;
  status?: 'never_compiled' | 'stale' | 'compiling' | 'ok' | 'ok_with_warnings' | 'error';
  collapsed?: boolean;
  reversed?: boolean;
  size?: { width: number; height: number };
  connected_inputs?: string[];
  connected_outputs?: string[];
  state_in?: boolean;
  state_out?: boolean;
  state_slots?: Array<{
    id: string;
    label: string;
    shape?: unknown[] | null;
    initializer?: Record<string, unknown> | null;
  }>;
  /** Present only on subgraph-typed nodes; carries the nested graph preview. */
  subgraph?: SubgraphPreview;
}

export interface TapNodeData extends Record<string, unknown> {
  tap: TapSpec;
}

export interface GraphEdgeData extends Record<string, unknown> {
  routing?: EdgeRouting;
  primary?: boolean;
  strength?: number;
  schema_status?: 'warning' | 'blocked' | null;
  schema_message?: string | null;
  temporality?: 'instant' | 'recurrent';
  recurrent_initializer?: Record<string, unknown> | null;
}
