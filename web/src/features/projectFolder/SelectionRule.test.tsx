// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, test } from 'vitest';
import type { PlanStepView } from '@/features/pipeline/planView';
import type { SubsetTerms } from '@/generated/studioContracts';
import { SelectionRule } from './SelectionRule';

afterEach(cleanup);

test('a rule naming a removed axis value offers the remaining values at its mark, and a choice replaces only it', async () => {
  const written: SubsetTerms[] = [];
  const terms: SubsetTerms = { clauses: [{ mass: [{ id: 'm1', label: '1 kg' }, { id: 'm3', label: '3 kg' }] }] };
  const source = { id: 'train', runs: [], ruleAxes: {} } as unknown as PlanStepView;
  render(<QueryClientProvider client={new QueryClient()}>
    <SelectionRule scope={{ project: 'p', revision: 'r', variant: 'main' }} source={source} terms={terms} disabled={false}
      problem={{ reason: 'mass has no value 3 kg', value: {
        clause_index: 0, axis: 'mass', missing: { id: 'm3', label: '3 kg' }, remaining: [{ id: 'm1', label: '1 kg' }, { id: 'm2', label: '2 kg' }],
      } }} onCommit={(next) => written.push(next)} />
  </QueryClientProvider>);
  await userEvent.click(screen.getByRole('button', { name: 'Problem' }));
  await userEvent.click(await screen.findByRole('menuitem', { name: '2 kg' }));
  expect(written).toEqual([{ clauses: [{ mass: [{ id: 'm1', label: '1 kg' }, { id: 'm2', label: '2 kg' }] }] }]);
});
