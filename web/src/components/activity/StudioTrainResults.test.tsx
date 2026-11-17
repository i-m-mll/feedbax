// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { fetchSavedEvaluationObjective } from '@/api/jobs';
import type {
  StudioEvaluationObjectiveSeries, StudioSavedEvaluationRow, StudioSavedTrainAttempt, StudioSavedTrainRow,
} from '@/generated/studioContracts';
import { StudioTrainResults } from './StudioTrainResults';

vi.mock('@/api/jobs', () => ({ fetchSavedEvaluationObjective: vi.fn() }));
vi.mock('@/hooks/useStudioJobs', () => ({
  useStudioJobActions: () => ({ refresh: vi.fn() }),
  useStudioJobConnection: () => 'connected',
  useStudioJobHistory: () => ({
    data: [{ kind: 'progress', request_id: 'train:source', job_id: 'train', data: { completed_updates: 0, metrics: {} } }],
    isError: false, isPending: false,
  }),
}));
afterEach(cleanup);

it('updates mounted Train Results in place from live points to their retained attempt', async () => {
  const trainKey = { variant: 'default', step: 'train', row: 'default' };
  const digest = 'a'.repeat(64);
  const receipt = { manifest_id: 'evaluation', manifest_sha256: digest };
  const attempt: StudioSavedTrainAttempt = {
    job_id: 'train', continues: null, checkpoint_source: null, status: 'running', reason: null,
    progress: { completed_updates: 2, total_updates: 6 }, successful_receipt: null,
  };
  const row: StudioSavedTrainRow = {
    key: trainKey, label: 'train', request_id: 'train:source', status: 'running', reason: null, attempts: [attempt],
    result: null, previous: null, change: null,
  };
  function point(update: number) {
    return {
      checkpoint: {
        job_id: attempt.job_id, request_id: row.request_id, completed_updates: update,
        transaction: receipt, transaction_root_sha256: digest, binding_sha256: digest, publication_cursor: update,
      },
      total: update * 2, terms: [{ path: '/effort', name: 'effort', value: update, coverage: 1 }],
    };
  }
  const evaluation: StudioSavedEvaluationRow = {
    key: { ...trainKey, step: 'evaluate' }, request_id: 'evaluate:series', status: 'queued', reason: null,
    can_run: true, trials: ['a'], result: null,
    checkpoint_series: { subject: trainKey, requested: [2, 4, 6] },
    checkpoint_attempt: {
      job_id: 'evaluation', placement: { location: 'live', reason: 'within_budget' },
      points: [2, 6].map((update) => ({ ...point(update).checkpoint, status: 'scored' })),
    },
  };
  const live: StudioEvaluationObjectiveSeries = {
    job_id: 'evaluation', request_id: evaluation.request_id, receipt: null, lineage: ['train'],
    condition: digest, objective: digest, seed: 1, trials: ['a'], requested: [2, 4, 6],
    points: [point(2), point(6)], gaps: [],
  };
  const fetch = vi.mocked(fetchSavedEvaluationObjective);
  fetch.mockResolvedValueOnce(live);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function plot(value: StudioSavedEvaluationRow) {
    return <QueryClientProvider client={client}>
      <StudioTrainResults project="project" revision={digest} evaluations={[value]} mapRows={[]}
        rows={[row]} row={row} selection={{ kind: 'attempt', attempt }} />
    </QueryClientProvider>;
  }
  const { rerender } = render(plot(evaluation));
  const group = await screen.findByRole('group', { name: /^Objective evaluate · effort$/ });
  expect(group.querySelectorAll('circle')).toHaveLength(2);
  expect(group.querySelector('polyline')).toBeNull();
  const firstMark = group.querySelector('circle');
  const lastMark = group.querySelector('circle:last-child');

  fetch.mockResolvedValueOnce({ ...live, receipt, points: [point(2), point(4), point(6)] });
  rerender(plot({
    ...evaluation, status: 'succeeded', checkpoint_attempt: undefined,
    result: { job_id: live.job_id, receipt, checkpoint_source: null, size_bytes: 1 },
  }));
  expect(group.querySelectorAll('circle')).toHaveLength(2);
  await waitFor(() => expect(group.querySelectorAll('circle')).toHaveLength(3));
  expect(group.querySelector('circle')).toBe(firstMark);
  expect(group.querySelector('circle:last-child')).toBe(lastMark);
  expect(within(group).getByText(/Updates 4/)).toBeDefined();
  expect(fetch).toHaveBeenCalledTimes(2);
});
