import type { GenerateFigureResponse, FigureStatusResponse } from '@/types/analysis';
import type {
  FigureListResponse,
  FigureDetail,
  EvaluationFigureSummary,
  FigureFilters,
} from '@/types/figures';

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

// --- Demand-driven figure generation (from F-C) ---

/** Trigger demand-driven figure generation for an analysis node. */
export async function generateFigure(
  nodeId: string,
  options?: { forceRerun?: boolean }
): Promise<GenerateFigureResponse> {
  return request<GenerateFigureResponse>('/api/analysis/generate', {
    method: 'POST',
    body: JSON.stringify({
      node_id: nodeId,
      force_rerun: options?.forceRerun ?? false,
    }),
  });
}

/** Check the status of a figure generation request. */
export async function getFigureStatus(requestId: string): Promise<FigureStatusResponse> {
  return request<FigureStatusResponse>(`/api/analysis/status/${requestId}`);
}

/** Fetch the Plotly JSON for a generated figure by hash. */
export async function getFigureData(figureHash: string): Promise<unknown> {
  return request<unknown>(`/api/figures/${figureHash}/file?format=json`);
}

// --- Figure gallery browsing (from F-D) ---

/** List evaluations that have at least one figure. */
export async function fetchEvaluationsWithFigures(): Promise<EvaluationFigureSummary[]> {
  return request<EvaluationFigureSummary[]>('/api/figures/evaluations');
}

/** List figures with optional filters and pagination. */
export async function fetchFigures(
  filters: FigureFilters = {},
  limit = 50,
  offset = 0,
): Promise<FigureListResponse> {
  const params = new URLSearchParams();
  if (filters.evaluation_hash) params.set('evaluation_hash', filters.evaluation_hash);
  if (filters.expt_name) params.set('expt_name', filters.expt_name);
  if (filters.figure_type) params.set('figure_type', filters.figure_type);
  if (filters.pert_type) params.set('pert_type', filters.pert_type);
  if (filters.identifier) params.set('identifier', filters.identifier);
  params.set('limit', String(limit));
  params.set('offset', String(offset));
  return request<FigureListResponse>(`/api/figures/?${params}`);
}

/** Get full metadata for a single figure. */
export async function fetchFigureDetail(hash: string): Promise<FigureDetail> {
  return request<FigureDetail>(`/api/figures/${hash}`);
}

/** Fetch Plotly JSON spec or image blob URL for a figure. */
export async function fetchFigureFile(
  hash: string,
  format = 'json',
): Promise<unknown> {
  const response = await fetch(`/api/figures/${hash}/file?format=${format}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Failed to load figure file: ${response.status}`);
  }
  if (format === 'json') {
    return response.json();
  }
  // For image formats, return blob URL
  const blob = await response.blob();
  return URL.createObjectURL(blob);
}
