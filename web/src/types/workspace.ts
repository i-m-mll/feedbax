import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { LossTermSpec, TaskSpec, TimeAggregationSpec, TrainingSpec } from '@/types/training';

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

export type StudioTopPaneProjection = 'graph' | 'workspace' | 'objectives';

export interface StudioTopPaneState {
  active_projection: StudioTopPaneProjection;
  selected_entity_id: string | null;
  hovered_entity_id: string | null;
  pinned_inspector_entity_id?: string | null;
  metadata: Record<string, unknown>;
}

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
  metadata: Record<string, unknown>;
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
  temporal_selector?: TimeAggregationSpec | Record<string, unknown> | null;
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

export interface StudioTaskDataSpec {
  id: string;
  label: string;
  kind: StudioTaskDataKind;
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

export interface StudioValueSpec {
  schema_version: 'feedbax.studio.value.v1' | string;
  mode: StudioValueSpecMode;
  value?: unknown;
  reference?: StudioSelectorRef | null;
  expression?: string | null;
  function_id?: string | null;
  parameters?: Record<string, unknown> | null;
  distribution?: Record<string, unknown> | null;
  schedule?: Record<string, unknown> | null;
  sampling_scope?:
    | 'snapshot'
    | 'run'
    | 'replicate'
    | 'trial'
    | 'epoch'
    | 'timestep'
    | 'sweep'
    | string
    | null;
  dtype?: string | null;
  shape?: unknown[] | null;
  units?: string | null;
  frame?: string | null;
  metadata: Record<string, unknown>;
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
  metadata: Record<string, unknown>;
}

export type StudioSchemaOrigin = 'declared' | 'inferred_static' | 'curated_fallback' | 'unknown';

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
  path: string;
  bindable: boolean;
  value_schema: ValueSchema;
  origin: StudioSchemaOrigin;
  metadata: Record<string, unknown>;
}

export interface SelectorTargetSchema {
  id: string;
  label: string;
  kind: 'port' | 'task_data' | 'objective' | 'probe' | 'state_hint';
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

export interface StudioTaskEpochSpec {
  id: string;
  label: string;
  index: number;
  length: StudioValueSpec;
  metadata: Record<string, unknown>;
}

export interface StudioTaskTimelineSignalSpec {
  id: string;
  label: string;
  kind: StudioTaskDataKind | string;
  path: string;
  epoch_ids: string[];
  metadata: Record<string, unknown>;
}

export interface StudioTaskTimelineSpec {
  schema_version: 'feedbax.studio.task_timeline.v1' | string;
  epochs: StudioTaskEpochSpec[];
  signals: StudioTaskTimelineSignalSpec[];
  metadata: Record<string, unknown>;
}

export type StudioScenarioEntityKind =
  | 'graph_node'
  | 'graph_port'
  | 'graph_edge'
  | 'task_object'
  | 'task_data'
  | 'task_binding'
  | 'mechanics_object'
  | 'objective_term'
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
  eval_params: Record<string, unknown>;
  viewport: { x: number; y: number; zoom: number };
  eval_run_id: string | null;
  expanded_field_paths?: string[];
}

export interface StudioScenarioSpec {
  id: string;
  schema_version: 'feedbax.studio.scenario.v1' | string;
  label: string;
  stage_id?: string | null;
  parent_scenario_id?: string | null;
  graph?: GraphSpec | null;
  graph_ui_state?: GraphUIState | null;
  training_spec?: TrainingSpec | null;
  task_spec?: TaskSpec | null;
  task_binding_spec?: StudioTaskBindingSpec | null;
  objective_spec?: StudioObjectiveSpec | Record<string, unknown> | null;
  probe_specs: Array<Record<string, unknown>>;
  temporal_spec?: Record<string, unknown> | null;
  biomechanics_spec?: Record<string, unknown> | null;
  analysis_spec?: Record<string, unknown> | null;
  report_spec?: Record<string, unknown> | null;
  validation: StudioValidationState;
  ui_state: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface StudioStageSpec {
  id: string;
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
  schema_version: 'feedbax.studio.workspace.v1' | string;
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

export interface ExecutionPlanStep {
  id: string;
  title: string;
  command?: string | null;
  description: string;
  critical: boolean;
  metadata: Record<string, unknown>;
}

export interface ExecutionArtifactRoute {
  role: string;
  source: string;
  destination?: string | null;
  tracked: boolean;
  description: string;
}

export interface ExecutionPlan {
  kind: 'ExecutionPlan';
  schema_version: string;
  job_id: string;
  backend: string;
  command: string;
  run_directory: string;
  bootstrap: ExecutionPlanStep[];
  health_checks: Array<Record<string, unknown>>;
  launch: ExecutionPlanStep;
  monitor: ExecutionPlanStep[];
  artifact_routes: ExecutionArtifactRoute[];
  cloud_payload: Record<string, unknown>;
  reproducibility: Record<string, unknown>;
  warnings: string[];
}

export interface StudioTrainingExecutionPreparation {
  workspace: StudioWorkspaceSpec;
  stage_id: string;
  scenario_id: string;
  execution_spec: Record<string, unknown>;
  plan: ExecutionPlan;
}

export interface LocalExecutionResult {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  return_code: number;
  stdout_path: string;
  stderr_path: string;
  manifest_path: string;
  manifest_payload: Record<string, unknown>;
  plan: ExecutionPlan;
}

export interface StudioTrainingLocalRunResult {
  workspace: StudioWorkspaceSpec;
  stage_id: string;
  scenario_id: string;
  execution_spec: Record<string, unknown>;
  result: LocalExecutionResult;
  snapshot_dir: string;
}

export interface StudioPipelineMaterializationResult {
  workspace: StudioWorkspaceSpec;
  stage_ids: string[];
  manifest_paths: Record<string, string>;
  artifact_refs: StudioArtifactRef[];
}
