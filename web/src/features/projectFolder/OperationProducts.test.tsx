// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import type { DatasetDescriptor, StudioSavedLightResult } from '@/generated/studioContracts';
import { OperationProducts } from './OperationProducts';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const url = '/api/projects/p/operation-datasets/fixed-points/points';
// Three conditions of five fixed points each, which fit one page, and a gain of more values than two pages.
const descriptor: DatasetDescriptor = {
  dimensions: { condition: 3, point: 5 },
  variables: { mean: { dims: ['condition', 'point'], units: 'm/s', label: 'Speed' } },
  coords: { condition: { dims: ['condition'], inline_values: ['left', 'centre', 'right'] } },
};
const long: DatasetDescriptor = { dimensions: { sample: 300 }, variables: { gain: { dims: ['sample'] } } };

function serve() {
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const at = new URL(input, 'http://studio');
    const body = (() => {
      if (at.pathname === url) return descriptor;
      if (at.pathname === '/long') return long;
      const offset = Number(at.searchParams.get('offset'));
      const limit = Number(at.searchParams.get('limit'));
      const total = at.pathname === url + '/values' ? 15 : 300;
      return {
        field: at.searchParams.get('field'), offset, total,
        values: Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, index) => offset + index),
      };
    })();
    return new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' } });
  }));
}

function result(products: Record<string, string>) {
  return { job_id: 'job', request_id: 'request', products } as unknown as StudioSavedLightResult;
}

function show() {
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <OperationProducts operations={[{ name: 'fixed-points', label: 'Fixed points', history: { result: result({ points: url }) } },
      { name: 'gains', label: 'Gains', history: { result: result({ gain: '/long' }) } }]} />
  </QueryClientProvider>);
}

it('reads each value at its row-major place, with the labels of its dimensions and its units', async () => {
  serve();
  show();
  const values = await screen.findByRole('table', { name: 'Values Speed (m/s)' });
  const rows = within(values).getAllByRole('row').map((row) => [...row.querySelectorAll('th, td')]
    .map((cell) => cell.textContent));
  expect(rows[0]).toEqual(['Condition', 'Point', 'Speed (m/s)']);
  // Flat index 7 is condition 1, point 2 in a 3 × 5 field.
  expect(rows[8]).toEqual(['centre', '2', '7']);
  expect(rows).toHaveLength(16);
  const variables = screen.getAllByRole('table', { name: 'Variables' })[0];
  expect(within(variables).getByRole('row', { name: /Speed/ }).textContent).toContain('Condition 3 × Point 5');
});

it('pages a variable longer than a page without skipping or repeating values', async () => {
  serve();
  show();
  const values = await screen.findByRole('table', { name: 'Values Gain' });
  const section = values.parentElement!.closest('.grid')!;
  const next = within(section as HTMLElement).getByRole('button', { name: 'Next Page' });
  fireEvent.click(next);
  await within(section as HTMLElement).findByText('129–256 of 300');
  fireEvent.click(next);
  await within(section as HTMLElement).findByText('257–300 of 300');
  expect(within(section as HTMLElement).getByRole('button', { name: 'Next Page' }).hasAttribute('disabled')).toBe(true);
  const firstRow = within(values).getAllByRole('row')[1];
  expect(within(firstRow).getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['256', '256']);
});
