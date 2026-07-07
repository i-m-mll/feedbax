import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiRequestError } from '@/api/request';
import {
  cancelTrainingRun,
  createEvalRun,
  createTrainingRun,
  deleteTrainingRun,
  fetchEvalRuns,
  fetchTrainingRunManifest,
  fetchTrainingRuns,
  supersedeTrainingRun,
} from '@/api/runAPI';

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

  it('maps typed manifest-index training run summaries', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify([
        {
          id: 'feedbax-training-run:pending',
          name: 'Pending train stage',
          created_at: '2026-07-07T12:00:00+00:00',
          status: 'pending',
          hyperparams: { n_batches: 25, ignored: { nested: true } },
          metrics: { final_validation_loss: 0.25 },
          uri: '/tmp/runs/manifests/training_runs/pending.json',
          stage_id: 'stage:train',
          scenario_id: 'scenario:train',
          planned: true,
          checkpoint_available: false,
          source_issue: '9aa8ff2',
          provenance_id: 'feedbax-training-run:pending',
          superseded_by: null,
        },
      ]),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    )));

    await expect(fetchTrainingRuns()).resolves.toEqual([
      expect.objectContaining({
        id: 'feedbax-training-run:pending',
        status: 'pending',
        hyperparams: { n_batches: 25 },
        metrics: { final_validation_loss: 0.25 },
        stageId: 'stage:train',
        scenarioId: 'scenario:train',
        planned: true,
        sourceIssue: '9aa8ff2',
      }),
    ]);
  });

  it('maps typed manifest-index evaluation summaries', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify([
        {
          id: 'feedbax-evaluation-run:abc',
          training_run_id: 'feedbax-training-run:a',
          training_run_ids: ['feedbax-training-run:a', 'feedbax-training-run:b'],
          name: 'Validation',
          created_at: '2026-07-07T12:01:00+00:00',
          status: 'completed',
          description: 'default',
          uri: '/tmp/runs/manifests/evaluation_runs/eval.json',
        },
      ]),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    )));

    await expect(fetchEvalRuns('feedbax-training-run:a')).resolves.toEqual([
      expect.objectContaining({
        id: 'feedbax-evaluation-run:abc',
        trainingRunId: 'feedbax-training-run:a',
        trainingRunIds: ['feedbax-training-run:a', 'feedbax-training-run:b'],
        uri: '/tmp/runs/manifests/evaluation_runs/eval.json',
      }),
    ]);
  });

  it('fetches durable training manifests for snapshot restage', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({
        id: 'feedbax-training-run:pending',
        training_spec: { inline: { n_batches: 25 } },
        task_spec: { inline: { type: 'ReachingTask' } },
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    )));

    await expect(fetchTrainingRunManifest('feedbax-training-run:pending')).resolves.toMatchObject({
      id: 'feedbax-training-run:pending',
      training_spec: { inline: { n_batches: 25 } },
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

  it('uses backend lifecycle endpoints for pending training manifests', async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => new Response(
      JSON.stringify({
        id: 'feedbax-training-run:pending',
        name: 'Pending',
        created_at: '2026-07-07T12:00:00+00:00',
        status: init?.method === 'DELETE' ? 'pending' : 'cancelled',
        hyperparams: {},
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    ));
    vi.stubGlobal('fetch', fetchMock);

    await cancelTrainingRun('feedbax-training-run:pending');
    await deleteTrainingRun('feedbax-training-run:pending');
    await supersedeTrainingRun('feedbax-training-run:completed', {
      superseded_by: 'feedbax-training-run:new',
      reason: 'new sweep',
    });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/runs/training/feedbax-training-run%3Apending/cancel',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/api/runs/training/feedbax-training-run%3Apending',
      expect.objectContaining({ method: 'DELETE' }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      '/api/runs/training/feedbax-training-run%3Acompleted/supersede',
      expect.objectContaining({ method: 'POST' }),
    );
  });
});
