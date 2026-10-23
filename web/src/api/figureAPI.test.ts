import { afterEach, describe, expect, it, vi } from 'vitest';
import { generateFigure } from '@/api/figureAPI';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('generateFigure evaluation-states policy', () => {
  it('forwards the authored policy and emits recompute when absent', async () => {
    const bodies: Array<Record<string, unknown>> = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
        return Response.json({
          data: {
            request_id: `request-${bodies.length}`,
            status: 'pending',
          },
        });
      }),
    );

    await generateFigure('analysis-a', {
      evalRunId: 'evaluation-a',
      evaluationStatesPolicy: 'require_durable',
    });
    await generateFigure('analysis-b', { evalRunId: 'evaluation-b' });

    expect(bodies).toEqual([
      expect.objectContaining({ evaluation_states_policy: 'require_durable' }),
      expect.objectContaining({ evaluation_states_policy: 'recompute' }),
    ]);
  });
});
