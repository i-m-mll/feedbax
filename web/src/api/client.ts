import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { ComponentDefinition } from '@/types/components';
import type { TrainingSpec, TaskSpec } from '@/types/training';

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

export async function fetchGraph(graphId: string) {
  return request<{ graph: GraphSpec; ui_state: GraphUIState | null }>(`/api/graphs/${graphId}`);
}

export async function createGraph(graph: GraphSpec, uiState: GraphUIState | null) {
  return request<{ id: string; metadata: { name: string } }>(`/api/graphs`, {
    method: 'POST',
    body: JSON.stringify({ graph, ui_state: uiState }),
  });
}

export async function updateGraph(graphId: string, graph: GraphSpec, uiState: GraphUIState | null) {
  return request<{ success: boolean }>(`/api/graphs/${graphId}`, {
    method: 'PUT',
    body: JSON.stringify({ graph, ui_state: uiState }),
  });
}

export async function exportGraph(graphId: string, format: 'json' | 'python') {
  return request<{ content: string; filename: string }>(`/api/graphs/${graphId}/export`, {
    method: 'POST',
    body: JSON.stringify({ format }),
  });
}

export async function startTraining(graphId: string, trainingSpec: TrainingSpec, taskSpec: TaskSpec) {
  return request<{ job_id: string }>('/api/training', {
    method: 'POST',
    body: JSON.stringify({ graph_id: graphId, training_spec: trainingSpec, task_spec: taskSpec }),
  });
}

export async function stopTraining(jobId: string) {
  return request<{ success: boolean }>(`/api/training/${jobId}`, { method: 'DELETE' });
}
