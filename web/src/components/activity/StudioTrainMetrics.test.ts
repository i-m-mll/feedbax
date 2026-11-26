import { expect, it } from 'vitest';
import type { StudioEvaluationObjectiveSeries, StudioJobEvent } from '@/generated/studioContracts';
import { heldOutTracks } from './heldOutObjective';
import { studioRunHistory } from './StudioTrainMetrics';

it('keeps loss intervals and held-out objectives across repeated checkpoint resumes', () => {
  const request = 'train:source';
  const first = { job_id: 'first', continues: null };
  const resumed = { job_id: 'resumed', continues: first.job_id };
  const latest = { job_id: 'latest', continues: resumed.job_id };
  const attempts = [first, resumed, latest];
  const events: StudioJobEvent[] = [];
  function progress(attempt: typeof first | typeof resumed, data: StudioJobEvent['data']) {
    events.push({
      cursor: events.length + 1, kind: 'progress', project_id: 'project',
      request_id: request, job_id: attempt.job_id, data,
    });
  }
  function interval(attempt: typeof first | typeof resumed, start: number, end: number, mean: number, last: number) {
    progress(attempt, {
      completed_updates: end, metrics: { train_loss: last },
      metric_interval: {
        start_exclusive: start, end_inclusive: end,
        metrics: { train_loss: { count: end - start, mean, last, min: last, max: 2 * mean - last } },
      },
    });
  }
  progress(first, { completed_updates: 90, metrics: {} });
  interval(first, 90, 96, 8, 7);
  interval(first, 96, 100, 6, 5);
  progress(resumed, { completed_updates: 100, metrics: { train_loss: 5 } });
  interval(resumed, 100, 112, 4, 3);
  progress(latest, { completed_updates: 112, metrics: { train_loss: 3 } });
  interval(latest, 112, 116, 2, 1);

  const history = studioRunHistory(latest, attempts, events);
  expect(history.metrics.map(({ name }) => name)).toEqual(['train_loss']);
  expect(history.metrics[0].points.map(({ jobId, start, end, elements: [reading] }) => [jobId, start, end, reading?.value])).toEqual([
    ['first', 90, 96, 8], ['first', 96, 100, 6], ['resumed', 100, 112, 4], ['latest', 112, 116, 2],
  ]);
  expect(history.stageStart).toBe(90);
  expect(history.attemptStarts).toEqual(new Map([['first', 90], ['resumed', 100], ['latest', 112]]));
  expect(history.attemptCutoffs).toEqual(new Map([['first', 100], ['resumed', 112], ['latest', Infinity]]));

  const digest = 'a'.repeat(64);
  const receipt = { manifest_id: 'evaluation', manifest_sha256: digest };
  const result: StudioEvaluationObjectiveSeries = {
    job_id: 'evaluation', request_id: 'evaluate:series', receipt, lineage: ['first', 'resumed', 'latest'],
    condition: digest, objective: digest, trials: ['a'], seed: 1, requested: [100, 112, 116], gaps: [],
    points: attempts.map(({ job_id }, index) => {
      const update = [100, 112, 116][index];
      return {
        checkpoint: {
          job_id, request_id: request, completed_updates: update, transaction: receipt,
          transaction_root_sha256: digest, binding_sha256: digest, publication_cursor: update,
        },
        total: 3 - index, terms: [{ path: '/effort', name: 'effort', coverage: 1, value: 3 - index }],
      };
    }),
  };
  const [total] = heldOutTracks('evaluate', result, {
    request, offset: 0, origin: history.stageStart!, through: null,
    starts: history.attemptStarts, cutoffs: history.attemptCutoffs,
  });
  expect(total?.segments.map((segment) => segment.map(({ update, value }) => [update, value]))).toEqual([
    [[10, 3], [22, 2], [26, 1]],
  ]);
});

it('reads every metric a method names, a vector metric as one series per element', () => {
  const attempt = { job_id: 'witness', continues: null };
  const events: StudioJobEvent[] = [];
  const reduction = (value: number) => ({ count: 2, mean: value, min: value - 1, max: value + 1, last: value });
  events.push({
    cursor: 1, kind: 'progress', project_id: 'project', request_id: 'train', job_id: attempt.job_id,
    data: { completed_updates: 0, metrics: {} },
  });
  for (const [start, end] of [[0, 2], [2, 4], [4, 6]]) {
    events.push({
      cursor: events.length + 1, kind: 'progress', project_id: 'project', request_id: 'train', job_id: attempt.job_id,
      data: {
        completed_updates: end, metrics: { price: end, losses: [end, -end], note: 'text' },
        metric_interval: {
          start_exclusive: start, end_inclusive: end,
          metrics: { price: reduction(end), losses: [reduction(end), reduction(-end)], note: null },
        },
      },
    });
  }

  const { metrics } = studioRunHistory(attempt, [attempt], events);
  expect(metrics.map(({ name, vector, points }) => [name, vector, points.map(({ elements }) => elements.map((e) => e?.value))]))
    .toEqual([
      ['price', false, [[2], [4], [6]]],
      ['losses', true, [[2, -2], [4, -4], [6, -6]]],
    ]);
});

it('plots a completed attempt from its retained updates and the attempt it resumes from its intervals', () => {
  const first = { job_id: 'first', continues: null };
  const latest = { job_id: 'latest', continues: first.job_id };
  const progress: [string, number, number | null][] = [['first', 0, null], ['first', 2, 0], ['latest', 2, null], ['latest', 5, 2]];
  const events: StudioJobEvent[] = progress.map(([jobId, end, start], index) => ({
    cursor: index + 1, kind: 'progress', project_id: 'project', request_id: 'train', job_id: jobId,
    data: {
      completed_updates: end, metrics: {},
      ...start !== null && { metric_interval: {
        start_exclusive: start, end_inclusive: end,
        metrics: { loss: { count: end - start, mean: 9, min: 8, max: 10, last: 8 } },
      } },
    },
  }));
  const retained = new Map([['latest', [3, 4, 5].map((update) => ({ update, metrics: { loss: update } }))]]);

  const { metrics: [loss] } = studioRunHistory(latest, [first, latest], events, retained);
  expect(loss.points.map(({ jobId, end, elements: [reading] }) => [jobId, end, reading?.value])).toEqual([
    ['first', 2, 9], ['latest', 3, 3], ['latest', 4, 4], ['latest', 5, 5],
  ]);
});
