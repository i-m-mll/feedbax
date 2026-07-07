import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiRequestError } from '@/api/request';
import { createEvalRun, createTrainingRun, fetchEvalRuns, fetchTrainingRuns } from '@/api/runAPI';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('runAPI failure behavior', () => {
  it('throws a typed error instead of returning fabricated training runs', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('connection refused');
    }));

    await expect(fetchTrainingRuns()).rejects.toMatchObject({
      name: 'ApiRequestError',
      kind: 'network',
      path: '/api/runs/training',
    });
  });

  it('throws a typed error instead of returning fabricated eval runs', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('missing', { status: 404 })));

    await expect(fetchEvalRuns('tr-missing')).rejects.toMatchObject({
      name: 'ApiRequestError',
      kind: 'http',
      status: 404,
    });
  });

  it('throws a typed contract error when training run payloads drift', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify([{ id: 'tr-1', name: 'run', status: 'completed', hyperparams: {} }]),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    )));

    await expect(fetchTrainingRuns()).rejects.toMatchObject({
      name: 'ApiRequestError',
      kind: 'contract',
      path: '/api/runs/training',
    });
  });

  it('does not fabricate successful training-run creation', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('not implemented', { status: 405 })));

    await expect(createTrainingRun('new run')).rejects.toBeInstanceOf(ApiRequestError);
  });

  it('does not fabricate successful eval-run creation', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('backend offline');
    }));

    await expect(createEvalRun('tr-1', 'eval', {})).rejects.toMatchObject({
      name: 'ApiRequestError',
      kind: 'network',
    });
  });
});
