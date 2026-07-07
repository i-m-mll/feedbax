import { afterEach, describe, expect, it, vi } from 'vitest';
import { updateGraph } from '@/api/client';
import type { GraphMetadata, GraphSpec, GraphUIState } from '@/types/graph';

afterEach(() => {
  vi.unstubAllGlobals();
});

const metadata: GraphMetadata = {
  name: 'Concurrent graph',
  created_at: '2026-07-07T00:00:00+00:00',
  updated_at: '2026-07-07T00:00:00+00:00',
  version: '1.0.0',
  save_revision: 4,
};

const graph: GraphSpec = {
  nodes: {},
  wires: [],
  input_ports: [],
  output_ports: [],
  input_bindings: {},
  output_bindings: {},
  metadata,
};

const uiState: GraphUIState = {
  viewport: { x: 0, y: 0, zoom: 1 },
  node_states: {},
};

describe('graph API save concurrency', () => {
  it('sends the expected revision in both header and JSON payload', async () => {
    const fetchMock = vi.fn(async () => Response.json({
      data: {
        success: true,
        metadata: { ...metadata, save_revision: 5 },
      },
    }));
    vi.stubGlobal('fetch', fetchMock);

    const response = await updateGraph('graph-1', graph, uiState, undefined, undefined, null, 4);

    expect(response.metadata.save_revision).toBe(5);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, options] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(options.headers).toMatchObject({ 'If-Match': '4' });
    expect(JSON.parse(options.body as string)).toMatchObject({
      expected_save_revision: 4,
      graph,
      ui_state: uiState,
    });
  });
});
