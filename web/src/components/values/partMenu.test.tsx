// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, test } from 'vitest';
import type { ExpressionPartSite, StudioFieldFunction, NodePath } from '@/generated/studioContracts';
import type { FieldEdit, FieldModel } from './fieldModel';
import { ValueField } from './ValueField';

afterEach(cleanup);

const penalty = (id: string): StudioFieldFunction => ({ id, signature: `${id}(x)`, display_class: 'penalty', min_args: 1, max_args: 1 });

/** A Measure `sqnorm(x)` whose root menu the service sends when `release` is called. */
function measure() {
  let release!: (site: ExpressionPartSite) => void;
  const asked: NodePath[] = [];
  const edits: FieldEdit[] = [];
  const model: FieldModel = {
    id: 'measure', label: 'Measure', type: { kind: 'string' }, kind: 'formula', kinds: [], text: 'sqnorm(x)',
    structure: { value: { kind: 'call', function: 'sqnorm', args: [{ kind: 'name', reference: 'x' }] }, arguments: ['x'] },
    names: [], functions: ['sqnorm', 'huber', 'l1'].map(penalty),
    menus: {
      kind: 'served', key: ['view'],
      read: (part) => { asked.push(part); return new Promise((resolve) => { release = resolve; }); },
    },
    commit: async (edit) => { edits.push(edit); return undefined; },
  };
  render(<QueryClientProvider client={new QueryClient()}><ValueField model={model} /></QueryClientProvider>);
  return { asked, edits, release: (site: ExpressionPartSite) => release(site) };
}

// Catches a menu read for every field as the view is drawn, a menu that offers functions the service has not yet said
// fit while it is read, and a pick from the read menu that does not edit the expression.
test("a Formula's function menu is read when it opens and offers only the current function until then", async () => {
  const { asked, edits, release } = measure();
  const menu = screen.getByRole('combobox', { name: 'Function' });
  expect(asked).toEqual([]);
  await userEvent.click(menu);
  await waitFor(() => expect(asked).toEqual([[]]));
  expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual(['sqnorm']);
  release({ path: [], functions: ['sqnorm', 'huber'], names: [] });
  await waitFor(() => expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual(['sqnorm', 'huber']));
  await userEvent.click(screen.getByRole('option', { name: 'huber' }));
  await waitFor(() => expect(edits).toEqual([{ kind: 'text', text: 'huber(x)' }]));
});
