// @vitest-environment jsdom
import { cleanup, render, screen, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import type { StudioSavedEvaluationRow } from '@/generated/studioContracts';
import { StudioEvaluationResults } from './StudioEvaluationResults';

afterEach(cleanup);

it('reports the pending Train update and shows overhead only for a measured over-budget attempt', () => {
  const key = { variant: 'default', step: 'validation', row: 'default' };
  const evaluation: StudioSavedEvaluationRow = {
    key, request_id: 'evaluate:validation', status: 'queued', reason: null, can_run: true, trials: [], result: null,
    checkpoint_rule: { every: 7 },
    checkpoint_series: { subject: { ...key, step: 'train-main' }, requested: [7, 14] },
    checkpoint_attempt: {
      job_id: 'evaluation',
      placement: { location: 'ordinary', reason: 'unmeasured' },
      pending: { producer_job_id: 'training', producer_request_id: 'train:main', state: 'running', next_update: 14 },
    },
  };
  const onReplay = vi.fn();
  const { rerender } = render(<StudioEvaluationResults evaluation={evaluation} onReplay={onReplay} />);
  expect(screen.getByText('From saved checkpoints')).toBeDefined();
  expect(screen.getByText('Waiting for train-main update 14.')).toBeDefined();
  expect(screen.queryByText(/Over budget/)).toBeNull();

  const attempt = evaluation.checkpoint_attempt!;
  rerender(<StudioEvaluationResults evaluation={{ ...evaluation, checkpoint_attempt: {
    ...attempt, placement: { location: 'ordinary', reason: 'over_budget', predicted_update_fraction: 0.14 },
  } }} onReplay={onReplay} />);
  expect(screen.getByText('Over budget · Every 7 updates · 14% of a training update')).toBeDefined();

  rerender(<StudioEvaluationResults evaluation={{ ...evaluation, checkpoint_attempt: {
    ...attempt, placement: { location: 'live', reason: 'within_budget' },
    points: [{
      job_id: 'training', request_id: 'train:main', completed_updates: 7,
      publication_cursor: 4, transaction_root_sha256: 'a'.repeat(64), status: 'scored',
    }],
  } }} onReplay={onReplay} />);
  const results = within(screen.getByRole('table', { name: 'Checkpoint Results' }));
  expect(results.getByText('Scored')).toBeDefined();
  expect(results.queryByRole('button', { name: 'Replay update 7' })).toBeNull();
  expect(screen.queryByText(/Over budget/)).toBeNull();
});
