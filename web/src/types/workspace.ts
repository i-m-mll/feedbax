import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { TrainingSpec, TaskSpec } from '@/types/training';

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
  objective_spec?: Record<string, unknown> | null;
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
