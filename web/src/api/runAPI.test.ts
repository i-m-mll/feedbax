import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiRequestError } from '@/api/request';
import {
  cancelTrainingRun,
  compareTrainingRuns,
  createEvalRun,
  createTrainingRun,
  deleteTrainingRun,
  fetchEvalRunManifest,
  fetchEvalRuns,
  fetchTrainingRunManifest,
  fetchTrainingRuns,
  importManifestPacket,
  importRunsDir,
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

  it('fetches durable evaluation manifests for snapshot provenance', async () => {
    const fetchMock = vi.fn(async () => new Response(
      JSON.stringify({
        id: 'feedbax-evaluation-run:completed',
        evaluation_spec: { inline: { evaluation_type: 'feedbax.validation.default' } },
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    ));
    vi.stubGlobal('fetch', fetchMock);

    await expect(fetchEvalRunManifest('feedbax-evaluation-run:completed')).resolves.toMatchObject({
      id: 'feedbax-evaluation-run:completed',
      evaluation_spec: { inline: { evaluation_type: 'feedbax.validation.default' } },
    });
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/runs/evaluation/feedbax-evaluation-run%3Acompleted/manifest',
      expect.objectContaining({
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
      }),
    );
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

  it('requests only selected compare fields', async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => new Response(
      JSON.stringify({
        rows: [
          {
            id: 'run:a',
            params: { learning_rate: 0.001 },
            metrics: { final_validation_loss: 0.2 },
          },
          {
            id: 'run:b',
            params: { learning_rate: 0.0003 },
            metrics: { final_validation_loss: 0.1 },
          },
        ],
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    ));
    vi.stubGlobal('fetch', fetchMock);

    await expect(compareTrainingRuns({
      runIds: ['run:a', 'run:b'],
      paramFields: ['learning_rate'],
      metricFields: ['final_validation_loss'],
    })).resolves.toMatchObject({
      rows: [
        { id: 'run:a', params: { learning_rate: 0.001 } },
        { id: 'run:b', metrics: { final_validation_loss: 0.1 } },
      ],
    });
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      run_ids: ['run:a', 'run:b'],
      param_fields: ['learning_rate'],
      metric_fields: ['final_validation_loss'],
    });
  });

  it('maps import responses from packet and runs-dir endpoints', async () => {
    const fetchMock = vi.fn(async () => new Response(
      JSON.stringify({
        root: '/tmp/target',
        source_path: '/tmp/source',
        imported_manifest_ids: ['run:a'],
        skipped_manifest_ids: [],
        manifest_count: 1,
        artifact_count: 0,
        included_artifact_count: 0,
        external_artifact_count: 0,
        index_path: '/tmp/target/index/feedbax.sqlite',
        training_runs: [{
          id: 'run:a',
          name: 'Run A',
          created_at: '2026-07-07T12:00:00+00:00',
          status: 'completed',
          hyperparams: {},
          metrics: {},
        }],
        eval_runs: [],
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    ));
    vi.stubGlobal('fetch', fetchMock);

    await expect(importManifestPacket('/tmp/packet')).resolves.toMatchObject({
      importedManifestIds: ['run:a'],
      trainingRuns: [{ id: 'run:a' }],
    });
    await importRunsDir('/tmp/runs');

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/runs/import/packet',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/api/runs/import/runs-dir',
      expect.objectContaining({ method: 'POST' }),
    );
  });
});
