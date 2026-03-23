import type { GenerateFigureResponse, FigureStatusResponse } from '@/types/analysis';

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

/** Fetch the Plotly JSON for a generated figure. */
export async function getFigureData(figureHash: string): Promise<unknown> {
  return request<unknown>(`/api/figures/${figureHash}/file?format=json`);
}
