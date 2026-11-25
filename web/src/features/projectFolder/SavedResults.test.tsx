// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { regionPendingMs } from '@/components/activity/StudioRegion';
import type { StudioSavedEvaluationResult, StudioSavedEvaluationRow } from '@/generated/studioContracts';
import { savedRunId } from './projectWindowPlan';
import { focusedResult, SavedResults } from './SavedResults';

beforeEach(() => { vi.useFakeTimers(); });
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

const key = { variant: 'default', step: 'validation', row: 'default' };
const evaluation: StudioSavedEvaluationRow = {
  key, request_id: 'evaluate:validation', status: 'succeeded', reason: null, can_run: true, trials: [],
  // Only the fields the Results read: the evaluation's job and the update it can replay.
  result: { job_id: 'evaluation', checkpoint_sources: [{ completed_updates: 7 }] } as unknown as StudioSavedEvaluationResult,
  checkpoint_rule: { every: 7 },
  checkpoint_series: { subject: { ...key, step: 'train-main' }, requested: [7] },
};
const shown = { revision: 'before', rows: [], window_map: { evaluations: [evaluation] } } as unknown as Parameters<typeof focusedResult>[0];

it('keeps the Results of the plan on screen while the current plan is read, pending and with Replay available', () => {
  const runIds = new Set([savedRunId(key)]);
  render(<SavedResults shown={shown} current={false} project="project" onReplay={vi.fn()}
    stepLabel={(step) => step} onShowStep={vi.fn()}
    result={focusedResult(shown, 'validation', null, runIds, undefined)} />);
  const replay = screen.getByRole('button', { name: 'Replay update 7' });
  const region = replay.closest('[aria-busy]');
  expect(region?.getAttribute('aria-busy')).toBe('true');
  expect(region?.hasAttribute('data-region-pending')).toBe(false);
  expect(replay.hasAttribute('disabled')).toBe(false);
  act(() => { vi.advanceTimersByTime(regionPendingMs); });
  expect(region?.hasAttribute('data-region-pending')).toBe(true);
});

it('heads an evaluation of a synthesized model with its origin and the step that made it', () => {
  const synthesized: StudioSavedEvaluationRow = {
    ...evaluation, checkpoint_rule: null, checkpoint_series: null,
    result: { job_id: 'evaluation', origin: {
      kind: 'synthesized', key: { variant: 'default', step: 'synthesize', row: 'default' }, job_id: 'synthesis',
      request_id: `synthesize:${'0'.repeat(64)}`,
    } } as unknown as StudioSavedEvaluationResult,
  };
  const plan = { revision: 'current', rows: [], window_map: { evaluations: [synthesized] } } as unknown as Parameters<typeof focusedResult>[0];
  const onShowStep = vi.fn();
  render(<SavedResults shown={plan} current project="project" onReplay={vi.fn()}
    stepLabel={(step) => step === 'synthesize' ? 'Controller' : step} onShowStep={onShowStep}
    result={focusedResult(plan, 'validation', null, new Set([savedRunId(key)]), undefined)} />);
  expect(screen.getByText('Synthesized')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Controller' }));
  expect(onShowStep).toHaveBeenCalledWith('synthesize');
});
