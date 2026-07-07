import type { GraphMetadata, GraphSpec, GraphUIState } from '@/types/graph';
import type { ComponentDefinition } from '@/types/components';
import { parseContract, type SelectionSpec } from '@/generated/studioContracts';
import type {
  StudioPipelineMaterializationResult,
  StudioSchemaRegistry,
  StudioEvaluationCheckpointPolicy,
  StudioEvaluationLocalRunResult,
  StudioEvaluationMatrixPreview,
  StudioEvaluationStagingResult,
  StudioTrainingLocalRunResult,
  StudioTrainingExecutionPreparation,
  StudioWorkspaceSpec,
} from '@/types/workspace';
// Note: analysis_pages in the API uses snake_case wire format (graph_spec, eval_params).
// See analysisAPI.ts for the conversion to camelCase AnalysisPageSpec.
import type {
  LossTermSpec,
  ProbeInfo,
  TrainingSpec,
  TaskSpec,
  TrainingConfig,
  LossValidationResult,
  TrainingProgress,
} from '@/types/training';
import type { TrajectorySnapshot } from '@/stores/trainingStore';
import type { TrajectoryDataset, TrajectoryMetadata, TrajectoryData } from '@/types/trajectory';
import type {
  StatisticsResponse,
  TimeseriesResponse,
  HistogramResponse,
  ScatterResponse,
  DiagnosticsResponse,
} from '@/types/statistics';
import { asApiRequestError, requestJson } from '@/api/request';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  return requestJson(path, options) as Promise<T>;
}

function parseContractResponse<K extends import('@/generated/studioContracts').ContractName>(
  name: K,
  path: string,
  value: unknown,
): import('@/generated/studioContracts').ContractTypeMap[K] {
  try {
    return parseContract(name, value);
  } catch (error) {
    throw asApiRequestError(error, path, `${name} response did not match the Studio contract.`);
  }
}

function parseContractArray<K extends import('@/generated/studioContracts').ContractName>(
  name: K,
  path: string,
  value: unknown,
): import('@/generated/studioContracts').ContractTypeMap[K][] {
  if (!Array.isArray(value)) {
    throw asApiRequestError(
      new Error(`${name} response expected an array`),
      path,
      `${name} response did not match the Studio contract.`,
    );
  }
  return value.map((item) => parseContractResponse(name, path, item));
}

export async function fetchComponents(): Promise<ComponentDefinition[]> {
  const response = parseContract('ComponentListResponse', await requestJson('/api/components'));
  return response.data.components as unknown as ComponentDefinition[];
}

export async function fetchGraphs() {
  const response = parseContract('GraphListResponse', await requestJson('/api/graphs'));
  return response.data;
}

export interface DemoTrainingData {
  loss_history: { batch: number; loss: number }[];
  latest_trajectory: {
    effector_pos: [number, number][];
    target_pos: [number, number];
    start_pos: [number, number];
  } | null;
}

export async function fetchGraph(graphId: string) {
  const response = parseContract(
    'GraphDetailResponse',
    await requestJson(`/api/graphs/${graphId}`),
  );
  return response.data as unknown as {
    graph: GraphSpec;
    ui_state: GraphUIState | null;
    demo_training_data: DemoTrainingData | null;
    metadata: GraphMetadata | null;
    analysis_pages: Array<{
      id: string;
      name: string;
      graph_spec: Record<string, unknown>;
      eval_params: Record<string, unknown>;
      viewport: { x: number; y: number; zoom: number };
      eval_run_id: string | null;
    }> | null;
    active_analysis_page_id: string | null;
    workspace: StudioWorkspaceSpec | null;
  };
}

export async function createGraph(
  graph: GraphSpec,
  uiState: GraphUIState | null,
  workspace?: StudioWorkspaceSpec | null,
) {
  const payload: Record<string, unknown> = { graph, ui_state: uiState };
  if (workspace !== undefined) payload.workspace = workspace;
  const response = parseContract(
    'GraphCreateResponse',
    await requestJson(`/api/graphs`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  );
  return response.data;
}

export async function updateGraph(
  graphId: string,
  graph: GraphSpec | null,
  uiState: GraphUIState | null,
  analysisPages?: unknown[] | null,
  activeAnalysisPageId?: string | null,
  workspace?: StudioWorkspaceSpec | null,
  expectedSaveRevision?: number | null,
) {
  const payload: Record<string, unknown> = {};
  if (graph !== null && graph !== undefined) payload.graph = graph;
  if (uiState !== null && uiState !== undefined) payload.ui_state = uiState;
  if (analysisPages !== undefined) payload.analysis_pages = analysisPages;
  if (activeAnalysisPageId !== undefined) payload.active_analysis_page_id = activeAnalysisPageId;
  if (workspace !== undefined) payload.workspace = workspace;
  if (expectedSaveRevision !== undefined && expectedSaveRevision !== null) {
    payload.expected_save_revision = expectedSaveRevision;
  }
  const headers: Record<string, string> = {};
  if (expectedSaveRevision !== undefined && expectedSaveRevision !== null) {
    headers['If-Match'] = String(expectedSaveRevision);
  }
  const response = parseContract(
    'GraphUpdateResponse',
    await requestJson(`/api/graphs/${graphId}`, {
      method: 'PUT',
      headers,
      body: JSON.stringify(payload),
    }),
  );
  return response.data;
}

export async function prepareStudioTrainingExecution(payload: {
  workspace: StudioWorkspaceSpec;
  stage_id?: string | null;
  backend?: 'local' | 'ssh' | 'runpod' | 'modal';
  job_id?: string | null;
  local_cwd?: string | null;
  issues?: string[];
  metadata?: Record<string, unknown>;
}) {
  return request<StudioTrainingExecutionPreparation>('/api/provider/studio/training/plan', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function runStudioTrainingLocalExecution(payload: {
  workspace: StudioWorkspaceSpec;
  stage_id?: string | null;
  job_id?: string | null;
  local_cwd?: string | null;
  root?: string | null;
  timeout?: number | null;
  issues?: string[];
  metadata?: Record<string, unknown>;
}) {
  return request<StudioTrainingLocalRunResult>('/api/provider/studio/training/run-local', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function previewStudioEvaluationMatrix(payload: {
  workspace: StudioWorkspaceSpec;
  stage_id?: string | null;
  selection_spec?: SelectionSpec | null;
  training_run_ids?: string[];
  eval_params?: Record<string, unknown>;
  condition_matrix?: Record<string, unknown>;
  checkpoint_policy?: StudioEvaluationCheckpointPolicy;
  reprocess?: 'missing' | 'missing_failed' | 'all' | 'stale';
  job_id?: string | null;
  root?: string | null;
  issues?: string[];
  metadata?: Record<string, unknown>;
}) {
  return request<StudioEvaluationMatrixPreview>('/api/provider/studio/evaluation/preview', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function stageStudioEvaluationMatrix(payload: {
  workspace: StudioWorkspaceSpec;
  stage_id?: string | null;
  selection_spec?: SelectionSpec | null;
  training_run_ids?: string[];
  eval_params?: Record<string, unknown>;
  condition_matrix?: Record<string, unknown>;
  checkpoint_policy?: StudioEvaluationCheckpointPolicy;
  reprocess?: 'missing' | 'missing_failed' | 'all' | 'stale';
  job_id?: string | null;
  root?: string | null;
  issues?: string[];
  metadata?: Record<string, unknown>;
}) {
  return request<StudioEvaluationStagingResult>('/api/provider/studio/evaluation/stage', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function runStudioEvaluationLocalExecution(payload: {
  workspace: StudioWorkspaceSpec;
  stage_id?: string | null;
  selection_spec?: SelectionSpec | null;
  training_run_ids?: string[];
  eval_params?: Record<string, unknown>;
  condition_matrix?: Record<string, unknown>;
  checkpoint_policy?: StudioEvaluationCheckpointPolicy;
  reprocess?: 'missing' | 'missing_failed' | 'all' | 'stale';
  job_id?: string | null;
  root?: string | null;
  timeout?: number | null;
  issues?: string[];
  metadata?: Record<string, unknown>;
}) {
  return request<StudioEvaluationLocalRunResult>('/api/provider/studio/evaluation/run-local', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function materializeStudioPipeline(payload: {
  workspace: StudioWorkspaceSpec;
  stages?: Array<'eval' | 'analysis' | 'report'>;
  job_id?: string | null;
  root?: string | null;
  issues?: string[];
  metadata?: Record<string, unknown>;
}) {
  return request<StudioPipelineMaterializationResult>('/api/provider/studio/pipeline/materialize', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function fetchStudioSchemaRegistry(payload: {
  workspace: StudioWorkspaceSpec;
  scenario_id?: string | null;
  runtime_introspection?: boolean | { enabled: boolean; max_targets?: number } | null;
}) {
  return request<StudioSchemaRegistry>('/api/provider/studio/schemas', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function exportGraph(graphId: string, format: 'json' | 'python') {
  const response = parseContract('GraphExportResponse', await requestJson(
    `/api/graphs/${graphId}/export`,
    {
      method: 'POST',
      body: JSON.stringify({ format }),
    },
  ));
  return response.data;
}

export async function startTraining(
  graphId: string,
  trainingSpec: TrainingSpec,
  taskSpec: TaskSpec,
  graphSpec?: GraphSpec,
  trainingConfig?: TrainingConfig,
  taskBindingSpec?: StudioWorkspaceSpec['scenarios'][string]['task_binding_spec'],
) {
  const response = parseContract('TrainingStartResponse', await requestJson('/api/training', {
    method: 'POST',
    body: JSON.stringify({
      graph_id: graphId,
      training_spec: trainingSpec,
      task_spec: taskSpec,
      ...(taskBindingSpec !== undefined && taskBindingSpec !== null
        ? { task_binding_spec: taskBindingSpec }
        : {}),
      ...(graphSpec !== undefined ? { graph_spec: graphSpec } : {}),
      ...(trainingConfig !== undefined ? { training_config: trainingConfig } : {}),
    }),
  }));
  return response.data;
}

export async function stopTraining(jobId: string) {
  const response = parseContract(
    'SuccessResponse',
    await requestJson(`/api/training/${jobId}`, { method: 'DELETE' }),
  );
  return response.data;
}

export async function connectWorker(url: string, authToken: string | null) {
  const response = parseContract('WorkerConnectEnvelope', await requestJson(
    '/api/training/worker/connect',
    {
      method: 'POST',
      body: JSON.stringify({ url, auth_token: authToken }),
    },
  ));
  return response.data;
}

export async function fetchWorkerStatus() {
  const response = parseContract(
    'WorkerStatusEnvelope',
    await requestJson('/api/training/worker/status'),
  );
  return response.data;
}

// --- Probe and Loss API ---

export async function fetchProbes(graphId: string): Promise<ProbeInfo[]> {
  const path = `/api/training/probes/${graphId}`;
  return parseContractArray('ProbeResponse', path, await requestJson(path));
}

export async function validateLossSpec(
  graphId: string,
  lossSpec: LossTermSpec
): Promise<LossValidationResult> {
  const path = '/api/training/loss/validate';
  return parseContractResponse('ValidateLossResponse', path, await requestJson(path, {
    method: 'POST',
    body: JSON.stringify({ graph_id: graphId, loss_spec: lossSpec }),
  }));
}

export async function resolveSelector(
  graphId: string,
  selector: string
): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>('/api/training/loss/resolve-selector', {
    method: 'POST',
    body: JSON.stringify({ graph_id: graphId, selector }),
  });
}

export async function fetchCheckpoint(jobId: string) {
  return request<{ batch: number; loss: number; weights_available: boolean }>(
    `/api/training/${jobId}/checkpoint`
  );
}

export async function downloadCheckpoint(jobId: string): Promise<void> {
  const response = await fetch(`/api/training/${jobId}/checkpoint/download`);
  if (!response.ok) throw new Error(`Download failed: ${response.status}`);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `feedbax_checkpoint_${jobId}.eqx`;
  document.body.appendChild(a); // Firefox requires an attached element
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// --- Orchestration API ---

export interface OrchestrationStatusResponse {
  status: string;
  instance_name: string | null;
  worker_url: string | null;
  internal_ip: string | null;
  external_ip: string | null;
  error: string | null;
  orphaned_instance: string | null;
  worker_health_failures: number;
}

export interface FeedbaxInstallSpec {
  schema_version?: 'feedbax.orchestration.install.v1';
  source?: 'git';
  repository?: 'https://github.com/mlll-io/feedbax.git';
  ref?: string;
  extras?: string[];
}

export interface LaunchInstanceRequest {
  project: string;
  zone: string;
  machine_type?: string;
  preemptible?: boolean;
  worker_port?: number;
  auth_token?: string | null;
  ts_auth_key?: string | null;
  install_spec?: FeedbaxInstallSpec;
  confirm_billable_launch: boolean;
  confirmation_token: string;
  max_hourly_cost_usd: number;
}

export async function launchInstance(params: LaunchInstanceRequest) {
  return request<{ status: string; instance_name: string | null; worker_url: string | null }>(
    '/api/orchestration/launch',
    {
      method: 'POST',
      body: JSON.stringify(params),
    }
  );
}

export async function fetchOrchestrationStatus(): Promise<OrchestrationStatusResponse> {
  return request<OrchestrationStatusResponse>('/api/orchestration/status');
}

export async function terminateInstance() {
  return request<{ ok: boolean }>('/api/orchestration/instance', { method: 'DELETE' });
}

// --- Trajectory API ---

export async function fetchTrajectoryDatasets(): Promise<TrajectoryDataset[]> {
  const path = '/api/trajectories/datasets';
  return parseContractArray('DatasetInfo', path, await requestJson(path));
}

export async function fetchTrajectoryMetadata(dataset: string): Promise<TrajectoryMetadata> {
  const path = `/api/trajectories/${encodeURIComponent(dataset)}/metadata`;
  return parseContractResponse('TrajectoryMetadata', path, await requestJson(path));
}

export async function fetchTrajectory(dataset: string, index: number): Promise<TrajectoryData> {
  const path = `/api/trajectories/${encodeURIComponent(dataset)}/${index}`;
  return parseContractResponse('TrajectoryData', path, await requestJson(path));
}

export async function filterTrajectories(
  dataset: string,
  filters: { body_idx?: number; task_type?: number },
): Promise<{ indices: number[]; count: number }> {
  const params = new URLSearchParams();
  if (filters.body_idx !== undefined) params.set('body_idx', String(filters.body_idx));
  if (filters.task_type !== undefined) params.set('task_type', String(filters.task_type));
  const path = `/api/trajectories/${encodeURIComponent(dataset)}/filter?${params}`;
  return parseContractResponse('FilterResult', path, await requestJson(path));
}

// --- Statistics API ---

export async function fetchStatsSummary(
  dataset: string,
  groupBy: string,
): Promise<StatisticsResponse> {
  const params = new URLSearchParams({ group_by: groupBy });
  const path = `/api/trajectories/${encodeURIComponent(dataset)}/stats/summary?${params}`;
  return parseContractResponse('StatisticsResponse', path, await requestJson(path));
}

export async function fetchStatsTimeseries(
  dataset: string,
  metric: string,
  groupBy: string,
): Promise<TimeseriesResponse> {
  const params = new URLSearchParams({ metric, group_by: groupBy });
  const path = `/api/trajectories/${encodeURIComponent(dataset)}/stats/timeseries?${params}`;
  return parseContractResponse('TimeseriesResponse', path, await requestJson(path));
}

export async function fetchStatsHistogram(
  dataset: string,
  metric: string,
  groupBy: string,
  bins?: number,
): Promise<HistogramResponse> {
  const params = new URLSearchParams({ metric, group_by: groupBy });
  if (bins !== undefined) params.set('bins', String(bins));
  const path = `/api/trajectories/${encodeURIComponent(dataset)}/stats/histogram?${params}`;
  return parseContractResponse('HistogramResponse', path, await requestJson(path));
}

export async function fetchStatsScatter(
  dataset: string,
  xMetric: string,
  yMetric: string,
): Promise<ScatterResponse> {
  const params = new URLSearchParams({ x_metric: xMetric, y_metric: yMetric });
  const path = `/api/trajectories/${encodeURIComponent(dataset)}/stats/scatter?${params}`;
  return parseContractResponse('ScatterResponse', path, await requestJson(path));
}

export async function fetchStatsDiagnostics(
  dataset: string,
): Promise<DiagnosticsResponse> {
  const path = `/api/trajectories/${encodeURIComponent(dataset)}/stats/diagnostics`;
  return parseContractResponse('DiagnosticsResponse', path, await requestJson(path));
}
