import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { ComponentDefinition } from '@/types/components';
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

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers ?? {}),
    },
    ...options,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function fetchComponents(): Promise<ComponentDefinition[]> {
  const data = await request<{ components: ComponentDefinition[] }>('/api/components');
  return data.components;
}

export async function fetchGraphs() {
  return request<{ graphs: { id: string; metadata: { name: string; description?: string; created_at: string; updated_at: string; version: string } }[] }>('/api/graphs');
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
  return request<{
    graph: GraphSpec;
    ui_state: GraphUIState | null;
    demo_training_data: DemoTrainingData | null;
    metadata: { name: string; description?: string; created_at?: string; updated_at?: string } | null;
    analysis_pages: Array<{
      id: string;
      name: string;
      graph_spec: Record<string, unknown>;
      eval_params: Record<string, unknown>;
      viewport: { x: number; y: number; zoom: number };
      eval_run_id: string | null;
    }> | null;
    active_analysis_page_id: string | null;
  }>(`/api/graphs/${graphId}`);
}

export async function createGraph(graph: GraphSpec, uiState: GraphUIState | null) {
  return request<{ id: string; metadata: { name: string } }>(`/api/graphs`, {
    method: 'POST',
    body: JSON.stringify({ graph, ui_state: uiState }),
  });
}

export async function updateGraph(
  graphId: string,
  graph: GraphSpec | null,
  uiState: GraphUIState | null,
  analysisPages?: unknown[] | null,
  activeAnalysisPageId?: string | null,
) {
  const payload: Record<string, unknown> = {};
  if (graph !== null && graph !== undefined) payload.graph = graph;
  if (uiState !== null && uiState !== undefined) payload.ui_state = uiState;
  if (analysisPages !== undefined) payload.analysis_pages = analysisPages;
  if (activeAnalysisPageId !== undefined) payload.active_analysis_page_id = activeAnalysisPageId;
  return request<{ success: boolean }>(`/api/graphs/${graphId}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  });
}

export async function exportGraph(graphId: string, format: 'json' | 'python') {
  return request<{ content: string; filename: string }>(`/api/graphs/${graphId}/export`, {
    method: 'POST',
    body: JSON.stringify({ format }),
  });
}

export async function startTraining(
  graphId: string,
  trainingSpec: TrainingSpec,
  taskSpec: TaskSpec,
  graphSpec?: GraphSpec,
  trainingConfig?: TrainingConfig,
) {
  return request<{ job_id: string }>('/api/training', {
    method: 'POST',
    body: JSON.stringify({
      graph_id: graphId,
      training_spec: trainingSpec,
      task_spec: taskSpec,
      ...(graphSpec !== undefined ? { graph_spec: graphSpec } : {}),
      ...(trainingConfig !== undefined ? { training_config: trainingConfig } : {}),
    }),
  });
}

export async function stopTraining(jobId: string) {
  return request<{ success: boolean }>(`/api/training/${jobId}`, { method: 'DELETE' });
}

export async function connectWorker(url: string, authToken: string | null) {
  return request<{ ok: boolean; url: string }>('/api/training/worker/connect', {
    method: 'POST',
    body: JSON.stringify({ url, auth_token: authToken }),
  });
}

export async function fetchWorkerStatus() {
  return request<{ mode: 'local' | 'remote'; url: string | null; connected: boolean }>(
    '/api/training/worker/status'
  );
}

// --- Probe and Loss API ---

export async function fetchProbes(graphId: string): Promise<ProbeInfo[]> {
  return request<ProbeInfo[]>(`/api/training/probes/${graphId}`);
}

export async function validateLossSpec(
  graphId: string,
  lossSpec: LossTermSpec
): Promise<LossValidationResult> {
  return request<LossValidationResult>('/api/training/loss/validate', {
    method: 'POST',
    body: JSON.stringify({ graph_id: graphId, loss_spec: lossSpec }),
  });
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
}

export interface LaunchInstanceRequest {
  project: string;
  zone: string;
  machine_type?: string;
  preemptible?: boolean;
  worker_port?: number;
  auth_token?: string | null;
  ts_auth_key?: string | null;
  feedbax_install_cmd?: string;
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
  return request<TrajectoryDataset[]>('/api/trajectories/datasets');
}

export async function fetchTrajectoryMetadata(dataset: string): Promise<TrajectoryMetadata> {
  return request<TrajectoryMetadata>(`/api/trajectories/${encodeURIComponent(dataset)}/metadata`);
}

export async function fetchTrajectory(dataset: string, index: number): Promise<TrajectoryData> {
  return request<TrajectoryData>(
    `/api/trajectories/${encodeURIComponent(dataset)}/${index}`,
  );
}

export async function filterTrajectories(
  dataset: string,
  filters: { body_idx?: number; task_type?: number },
): Promise<{ indices: number[]; count: number }> {
  const params = new URLSearchParams();
  if (filters.body_idx !== undefined) params.set('body_idx', String(filters.body_idx));
  if (filters.task_type !== undefined) params.set('task_type', String(filters.task_type));
  return request<{ indices: number[]; count: number }>(
    `/api/trajectories/${encodeURIComponent(dataset)}/filter?${params}`,
  );
}

// --- Statistics API ---

export async function fetchStatsSummary(
  dataset: string,
  groupBy: string,
): Promise<StatisticsResponse> {
  const params = new URLSearchParams({ group_by: groupBy });
  return request<StatisticsResponse>(
    `/api/trajectories/${encodeURIComponent(dataset)}/stats/summary?${params}`,
  );
}

export async function fetchStatsTimeseries(
  dataset: string,
  metric: string,
  groupBy: string,
): Promise<TimeseriesResponse> {
  const params = new URLSearchParams({ metric, group_by: groupBy });
  return request<TimeseriesResponse>(
    `/api/trajectories/${encodeURIComponent(dataset)}/stats/timeseries?${params}`,
  );
}

export async function fetchStatsHistogram(
  dataset: string,
  metric: string,
  groupBy: string,
  bins?: number,
): Promise<HistogramResponse> {
  const params = new URLSearchParams({ metric, group_by: groupBy });
  if (bins !== undefined) params.set('bins', String(bins));
  return request<HistogramResponse>(
    `/api/trajectories/${encodeURIComponent(dataset)}/stats/histogram?${params}`,
  );
}

export async function fetchStatsScatter(
  dataset: string,
  xMetric: string,
  yMetric: string,
): Promise<ScatterResponse> {
  const params = new URLSearchParams({ x_metric: xMetric, y_metric: yMetric });
  return request<ScatterResponse>(
    `/api/trajectories/${encodeURIComponent(dataset)}/stats/scatter?${params}`,
  );
}

export async function fetchStatsDiagnostics(
  dataset: string,
): Promise<DiagnosticsResponse> {
  return request<DiagnosticsResponse>(
    `/api/trajectories/${encodeURIComponent(dataset)}/stats/diagnostics`,
  );
}
