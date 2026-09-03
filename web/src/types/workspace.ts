import type {
  RetainedObservableSpec,
  RetentionPolicySpec,
} from '@/types/graph';
import type { AnalysisInputRequirement } from '@/types/analysis';
import type { LossTermSpec, TaskSpec, TimeAggregationSpec, TrainingSpec } from '@/types/training';
import type {
  StudioBiomechanicsSpec,
  StudioEpochValueSpec as GeneratedStudioEpochValueSpec,
  StudioTaskEpochSpec as GeneratedStudioTaskEpochSpec,
  StudioTaskTimelineSegmentSpec as GeneratedStudioTaskTimelineSegmentSpec,
  StudioTaskTimelineSignalSpec as GeneratedStudioTaskTimelineSignalSpec,
  StudioTaskTimelineSpec as GeneratedStudioTaskTimelineSpec,
} from '@/generated/studioContracts';

export type StudioStageKind =
  | 'train'
  | 'eval'
  | 'analysis'
  | 'report'
  | 'import'
  | 'compare'
  | 'export'
  | 'protocol';

export type StudioStageStatus =
  | 'draft'
  | 'invalid'
  | 'ready'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type StudioTopPaneProjection =
  | 'model'
  | 'task'
  | 'workspace'
  | 'observables'
  | 'objectives';

export interface StudioTopPaneState {
  active_projection: StudioTopPaneProjection;
  selected_entity_id: string | null;
  hovered_entity_id: string | null;
  pinned_inspector_entity_id?: string | null;
  metadata: Record<string, unknown>;
}

export const WORKSPACE_VIEW_STATE_SCHEMA_VERSION =
  'feedbax.studio.workspace_view_state.v1' as const;

export interface WorkspaceCameraState {
  pan: { x: number; y: number };
  zoom: number;
}

export interface WorkspacePlaybackState {
  position: number;
  speed: number;
}

export interface WorkspaceComparisonSelection {
  baseline_ref: string | null;
  candidate_ref: string | null;
}

export interface WorkspaceViewState {
  schema_version: typeof WORKSPACE_VIEW_STATE_SCHEMA_VERSION;
  camera: WorkspaceCameraState;
  selected_artifact_ref: string | null;
  selected_trial_ref: string | null;
  overlay_visibility: Record<string, boolean>;
  playback: WorkspacePlaybackState;
  comparison_selection: WorkspaceComparisonSelection;
}

export type WorkspaceViewMode = 'authoring' | 'artifact' | 'playback' | 'comparison';

export interface StudioValidationIssue {
  type: string;
  message: string;
  location?: Record<string, string> | null;
  severity: 'error' | 'warning' | 'info';
}

export interface StudioValidationState {
  valid?: boolean | null;
  checked_at?: string | null;
  errors: StudioValidationIssue[];
  warnings: StudioValidationIssue[];
  metadata: Record<string, unknown>;
}

export interface StudioManifestRef {
  kind: string;
  id: string;
  role?: string | null;
  provider: string;
  uri?: string | null;
  metadata: Record<string, unknown>;
}

export interface StudioArtifactRef {
  kind: string;
  id: string;
  role?: string | null;
  provider: string;
  uri?: string | null;
  media_type?: string | null;
  metadata: Record<string, unknown>;
}

export interface StudioCollectionRef {
  id: string;
  kind: string;
  label?: string | null;
  source_stage_id?: string | null;
  item_refs: StudioManifestRef[];
  filters: Record<string, unknown>;
  facets: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export type StudioSelectorNamespace =
  | 'graph_port'
  | 'graph_edge'
  | 'graph_output'
  | 'recurrent_carry'
  | 'retained_observable'
  | 'probe'
  | 'state_path'
  | 'task_object'
  | 'task_data'
  | 'task_binding'
  | 'mechanics_object'
  | 'biomechanics_object'
  | 'artifact_field'
  | 'analysis_output'
  | 'custom';

export interface StudioSelectorRef {
  namespace: StudioSelectorNamespace;
  compact: string;
  target_id?: string | null;
  path?: string | null;
  expected_shape?: unknown[] | null;
  dtype?: string | null;
  units?: string | null;
  frame?: string | null;
  role?: 'editable' | 'observed' | 'generated' | string | null;
  metadata?: Record<string, unknown>;
}

export interface StudioObjectiveTermSpec {
  id: string;
  type_id: string;
  label: string;
  role: 'loss' | 'metric' | 'constraint' | 'reward' | 'regularizer' | string;
  source_selector?: StudioSelectorRef | null;
  target_selector?: StudioSelectorRef | null;
  target_value?: unknown;
  operator?: string | null;
  penalty?: string | null;
  matrix?: unknown;
  matrix_kind?: 'dense' | 'diagonal';
  temporal_selector?: TimeAggregationSpec | Record<string, unknown> | null;
  retention?: RetentionPolicySpec | null;
  weight: number;
  units?: string | null;
  validation?: StudioValidationState | null;
  metadata: Record<string, unknown>;
}

export interface StudioObjectiveSpec {
  schema_version: 'feedbax.studio.objective.v1' | string;
  terms: StudioObjectiveTermSpec[];
  legacy_loss_spec?: LossTermSpec | null;
  metadata: Record<string, unknown>;
}

export type StudioTaskDataKind =
  | 'signal'
  | 'target'
  | 'initial_state'
  | 'intervention'
  | 'trial_param'
  | 'protocol_value'
  | string;

export type StudioTaskDataRole =
  | 'model_input'
  | 'graph_input'
  | 'target'
  | 'initial_state'
  | 'intervention'
  | 'eval_control'
  | 'trial_control'
  | 'compact_task_trajectory'
  | 'materialized_task_trajectory'
  | 'protocol_value'
  | string;

export interface StudioTaskDataSpec {
  id: string;
  label: string;
  kind: StudioTaskDataKind;
  role?: StudioTaskDataRole | null;
  path: string;
  bindable: boolean;
  expected_shape?: unknown[] | null;
  dtype?: string | null;
  units?: string | null;
  frame?: string | null;
  value_spec?: StudioValueSpec | null;
  metadata: Record<string, unknown>;
}

export interface StudioTaskBinding {
  id: string;
  source_data_id: string;
  /** Node-id path from the root graph to the graph layer that owns this binding. */
  target_graph_path?: string[];
  target_node_id: string;
  target_port: string;
  role: 'model_input' | 'target' | 'initial_state' | 'intervention' | string;
  metadata: Record<string, unknown>;
}

export interface StudioTaskBindingSpec {
  schema_version: 'feedbax.studio.task_bindings.v2' | string;
  exposed_data: StudioTaskDataSpec[];
  bindings: StudioTaskBinding[];
  metadata: Record<string, unknown>;
}

export type StudioValueSpecMode =
  | 'constant'
  | 'reference'
  | 'expression'
  | 'function'
  | 'distribution'
  | 'schedule'
  | string;

export type StudioValueSpecSamplingScope =
  | 'fixed'
  | 'snapshot'
  | 'run'
  | 'replicate'
  | 'trial'
  | 'epoch'
  | 'timestep'
  | 'sweep'
  | string;

export interface StudioValueSpec {
  schema_version?: 'feedbax.spec.studio.value.v2' | 'feedbax.spec.studio.value.v1' | 'feedbax.studio.value.v1' | string;
  value_form: 'literal' | 'reference' | 'expression' | 'function' | 'schedule' | 'distribution';
  variation?: {
    scope: 'fixed' | 'snapshot' | 'run' | 'replicate' | 'trial' | 'epoch' | 'timestep' | 'sweep';
    enumerable?: {
      form: 'list' | 'range' | 'sampler';
      values?: unknown[];
      start?: number;
      stop?: number;
      count?: number;
      scale?: 'linear' | 'log';
      sampler?: Record<string, unknown>;
      n?: number;
    } | null;
    stochastic_policy?: 'shared_per_run' | 'resample_per_replicate' | null;
    metadata?: Record<string, unknown>;
  };
  mode: StudioValueSpecMode;
  value?: unknown;
  reference?: StudioSelectorRef | null;
  expression?: string | null;
  function_id?: string | null;
  parameters?: Record<string, unknown> | null;
  distribution?: Record<string, unknown> | null;
  schedule?: Record<string, unknown> | null;
  sampling_scope?: StudioValueSpecSamplingScope | null;
  dtype?: string | null;
  shape?: unknown[] | null;
  units?: string | null;
  frame?: string | null;
  metadata?: Record<string, unknown>;
}

export type StudioInterventionOperation =
  | 'clamp'
  | 'noise'
  | 'constant'
  | 'offset'
  | 'scale'
  | string;

export interface StudioInterventionValueBounds {
  min?: unknown;
  max?: unknown;
}

export interface StudioInterventionTransformSpec {
  operation: StudioInterventionOperation;
  target_selector?: StudioSelectorRef | null;
  value?: StudioValueSpec | null;
  bounds?: StudioInterventionValueBounds | null;
  parameters?: Record<string, unknown> | null;
  metadata?: Record<string, unknown>;
}

export type StudioSchemaOrigin =
  | 'declared'
  | 'inferred_static'
  | 'runtime_sample'
  | 'curated_fallback'
  | 'unknown';

export interface ValueSchema {
  id: string;
  label: string;
  kind: string;
  dtype?: string | null;
  shape?: unknown[] | null;
  rank?: number | null;
  units?: string | null;
  frame?: string | null;
  origin: StudioSchemaOrigin;
  metadata: Record<string, unknown>;
}

export interface PortSchema {
  id: string;
  label: string;
  node_id?: string | null;
  component_type?: string | null;
  port: string;
  direction: 'input' | 'output';
  value_schema: ValueSchema;
  bound_task_data_id?: string | null;
  origin: StudioSchemaOrigin;
  metadata: Record<string, unknown>;
}

export interface TaskDataSchema {
  id: string;
  label: string;
  kind: string;
  role: string;
  path: string;
  bindable: boolean;
  value_schema: ValueSchema;
  origin: StudioSchemaOrigin;
  metadata: Record<string, unknown>;
}

export interface SelectorTargetSchema {
  id: string;
  label: string;
  kind:
    | 'port'
    | 'edge'
    | 'graph_output'
    | 'recurrent_carry'
    | 'state_path'
    | 'task_data'
    | 'objective'
    | 'probe'
    | 'retained_observable'
    | 'state_hint'
    | 'sample_leaf';
  selector: string;
  value_schema: ValueSchema;
  origin: StudioSchemaOrigin;
  source: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface StudioSchemaRegistry {
  kind: 'studio_schema_registry';
  schema_version: string;
  generated_at: string;
  workspace_id?: string | null;
  scenario_id?: string | null;
  ports: PortSchema[];
  task_data: TaskDataSchema[];
  selector_targets: SelectorTargetSchema[];
  issues: StudioValidationIssue[];
  metadata: Record<string, unknown>;
}

export type StudioTaskEpochSpec = Omit<GeneratedStudioTaskEpochSpec, 'metadata'> & {
  metadata: Record<string, unknown>;
};

export type StudioEpochValueSpec = GeneratedStudioEpochValueSpec;

export type StudioTaskTimelineSignalSpec = Omit<
  GeneratedStudioTaskTimelineSignalSpec,
  'kind' | 'value_schema' | 'task_data_schema' | 'metadata'
> & {
  kind: StudioTaskDataKind | string;
  value_schema?: ValueSchema | null;
  task_data_schema?: TaskDataSchema | null;
  metadata: Record<string, unknown>;
};

export type StudioTaskTimelineSegmentSpec = Omit<
  GeneratedStudioTaskTimelineSegmentSpec,
  'epoch_ids' | 'metadata'
> & {
  epoch_ids: string[];
  metadata: Record<string, unknown>;
};

export type StudioTaskTimelineSpec = Omit<
  GeneratedStudioTaskTimelineSpec,
  'schema_id' | 'schema_version' | 'epochs' | 'signals' | 'epoch_value_specs' | 'segments' | 'metadata'
> & {
  schema_id: 'feedbax.spec.studio.task_timeline';
  schema_version: 'feedbax.spec.studio.task_timeline.v2';
  epochs: StudioTaskEpochSpec[];
  signals: StudioTaskTimelineSignalSpec[];
  epoch_value_specs: StudioEpochValueSpec[];
  segments?: StudioTaskTimelineSegmentSpec[];
  metadata: Record<string, unknown>;
};

export type StudioScenarioEntityKind =
  | 'graph_node'
  | 'graph_port'
  | 'graph_edge'
  | 'task_object'
  | 'task_data'
  | 'task_binding'
  | 'mechanics_object'
  | 'objective_term'
  | 'retained_observable'
  | 'probe'
  | 'metric'
  | 'temporal_event'
  | 'temporal_phase'
  | 'stage_protocol'
  | 'artifact_overlay';

export type StudioScenarioEntityRelationKind =
  | 'contains'
  | 'binds'
  | 'source'
  | 'target'
  | 'references'
  | 'derived_from';

export interface StudioScenarioEntityRelation {
  kind: StudioScenarioEntityRelationKind;
  entity_id: string;
  label?: string | null;
  metadata: Record<string, unknown>;
}

export interface StudioScenarioEntity {
  id: string;
  kind: StudioScenarioEntityKind;
  label: string;
  summary?: string | null;
  scenario_id?: string | null;
  stage_id?: string | null;
  selector?: StudioSelectorRef | null;
  relations: StudioScenarioEntityRelation[];
  metadata: Record<string, unknown>;
}

export interface StudioScenarioEntityRegistry {
  scenario_id: string | null;
  stage_id: string | null;
  entities: Record<string, StudioScenarioEntity>;
  root_entity_ids: string[];
  metadata: Record<string, unknown>;
}

export interface AnalysisPageWire {
  id: string;
  name: string;
  graph_spec: Record<string, unknown>;
  input_requirements?: AnalysisInputRequirement[];
  eval_params: Record<string, unknown>;
  eval_run_id: string | null;
  expanded_field_paths?: string[];
}

export interface StudioScenarioSpec {
  id: string;
  schema_version: 'feedbax.spec.studio.scenario.v3' | string;
  label: string;
  stage_id?: string | null;
  parent_scenario_id?: string | null;
  training_spec?: TrainingSpec | null;
  task_spec?: TaskSpec | null;
  task_binding_spec?: StudioTaskBindingSpec | null;
  objective_spec?: StudioObjectiveSpec | Record<string, unknown> | null;
  probe_specs?: RetainedObservableSpec[] | Array<Record<string, unknown>>;
  temporal_spec?: Record<string, unknown> | null;
  biomechanics_spec?: StudioBiomechanicsSpec | null;
  analysis_spec?: Record<string, unknown> | null;
  report_spec?: Record<string, unknown> | null;
  validation: StudioValidationState;
  ui_state: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface StudioStageSpec {
  id: string;
  schema_id: 'feedbax.spec.studio.stage';
  schema_version: 'feedbax.spec.studio.stage.v2';
  kind: StudioStageKind;
  label: string;
  status: StudioStageStatus;
  scenario_id?: string | null;
  input_collections: StudioCollectionRef[];
  output_collections: StudioCollectionRef[];
  manifest_refs: StudioManifestRef[];
  artifact_refs?: StudioArtifactRef[];
  execution_spec?: Record<string, unknown> | null;
  selection_spec: Record<string, unknown>;
  validation: StudioValidationState;
  ui_state: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface StudioWorkspaceSpec {
  id: string;
  schema_id: 'feedbax.spec.studio.workspace';
  schema_version: 'feedbax.spec.studio.workspace.v2';
  label: string;
  active_stage_id?: string | null;
  stages: StudioStageSpec[];
  scenarios: Record<string, StudioScenarioSpec>;
  collections: StudioCollectionRef[];
  manifest_refs: StudioManifestRef[];
  artifact_refs?: StudioArtifactRef[];
  validation: StudioValidationState;
  ui_state: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface Invocation {
  schema_id: 'feedbax.spec.invocation';
  schema_version: 'feedbax.spec.invocation.v1';
  invocation_id: string;
  workflow_plan_id: string;
  operation_key: string;
  operation: Record<string, unknown>;
  inputs: Array<Record<string, unknown>>;
  requested_outputs: Array<Record<string, unknown>>;
  scientific_seeds: Record<string, number>;
  capabilities: string[];
  execution_policy: Record<string, unknown>;
  publication_policy_ref?: string | null;
}

export interface BackendPlan {
  schema_id: 'feedbax.orchestration.backend_plan';
  schema_version: 'feedbax.orchestration.backend_plan.v1';
  backend_plan_id: string;
  invocation_id: string;
  backend_id: string;
  configuration: Record<string, unknown>;
  [key: string]: unknown;
}

export interface StudioTrainingExecutionPreparation {
  workspace: StudioWorkspaceSpec;
  graph: import('@/types/graph').GraphSpec;
  stage_id: string;
  scenario_id: string;
  invocation: Invocation;
  backend_plan: BackendPlan;
}

export type EvalCheckpointPolicyMode = 'last' | 'best-by-metric' | 'every-k';
export type EvalReprocessMode = 'missing' | 'missing_failed' | 'all' | 'stale';

export interface StudioEvaluationCheckpointPolicy {
  mode: EvalCheckpointPolicyMode;
  metric?: string | null;
  objective?: 'minimize' | 'maximize';
  every_k?: number | null;
  params?: Record<string, unknown>;
}

export interface StudioEvaluationMatrixPreview {
  workspace: StudioWorkspaceSpec;
  stage_id: string;
  selected_training_run_count: number;
  condition_count: number;
  checkpoint_policy_count: number;
  total_eval_count: number;
  materialized_count: number;
  pending_count: number;
  failed_count: number;
  new_manifest_count: number;
  launch_count: number;
  evaluation_run_ids: string[];
  checkpoint_selection_ids: string[];
  summary: string;
}

export interface StudioEvaluationStagingResult {
  workspace: StudioWorkspaceSpec;
  stage_id: string;
  preview: StudioEvaluationMatrixPreview;
  manifest_refs: StudioManifestRef[];
  checkpoint_selection_refs: StudioManifestRef[];
}

export interface StudioEvaluationLocalRunResult {
  workspace: StudioWorkspaceSpec;
  stage_id: string;
  preview: StudioEvaluationMatrixPreview;
  manifest_refs: StudioManifestRef[];
  completed_count: number;
  failed_count: number;
  skipped_count: number;
  skipped_failed_count: number;
  errors: string[];
}

export interface StudioPipelineMaterializationResult {
  workspace: StudioWorkspaceSpec;
  stage_ids: string[];
  manifest_paths: Record<string, string>;
  artifact_refs: StudioArtifactRef[];
}
