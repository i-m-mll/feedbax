import { expect, it } from 'vitest';
import type { PlanRunView, PlanView } from '@/features/pipeline/planView';
import { planWithRefusal } from './refusals';

it('draws a refused run as Can\'t run with the refusal as its reason and recounts its step', () => {
  const run = (id: string): PlanRunView => ({ id, label: id, values: {}, status: 'succeeded', state: 'succeeded', results: [] });
  const plan = { steps: [{ id: 'train', label: 'train', kind: 'train', sweeps: [], inputs: [],
    runs: [run('a'), run('b')], statusCounts: { succeeded: 2 } }], edges: [] } as unknown as PlanView;
  const [step] = planWithRefusal(plan, { kind: 'runs', reasons: { a: 'Run a is already stopping.' } }).steps;
  expect(step.runs.map(({ state, reason }) => [state, reason])).toEqual([
    ['cannot_run', 'Run a is already stopping.'], ['succeeded', undefined],
  ]);
  expect(step.statusCounts).toEqual({ succeeded: 1, cannot_run: 1 });
});
