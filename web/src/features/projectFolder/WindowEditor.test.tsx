// @vitest-environment jsdom
import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, test } from 'vitest';
import { rawField } from '@/components/values/adapters/raw';
import { WindowSchema, type Window } from '@/generated/studioContracts';
import { WindowEditor, windowText, type WindowHost } from './WindowEditor';

afterEach(cleanup);

const epochs = [{ id: 'go', name: 'go', occurrences: 1 }, { id: 'reach', name: 'reach', occurrences: 2 }];

function offsetAt(window: Window | null | undefined, site: readonly (string | number)[]) {
  let node: unknown = window;
  for (const key of site) node = (node as Record<string | number, unknown> | undefined)?.[key];
  return node as { value: number; unit: string } | undefined;
}

function host(window: Window | null, stored: (Window | null)[]): WindowHost {
  return {
    epochs,
    offset: (site) => {
      const offset = offsetAt(window, site);
      return rawField({ id: site.join('-'), label: site[2], type: { kind: 'string' }, value: offset ? `${offset.value} ${offset.unit}` : '' });
    },
    change: (next) => { stored.push(next); },
  };
}

// Catches: an interval that reads as closed, an epoch end read as its start, or an occurrence shown zero-based.
test('a window reads as its spans: epochs by name, intervals half-open, moments at their anchor', () => {
  const window: Window = { spans: [
    { kind: 'epoch', epoch: 'reach', occurrence: 1 },
    { kind: 'interval', start: { anchor: { kind: 'epoch', epoch: 'go', edge: 'start' }, offset: { value: 0.1, unit: 's' } },
      end: { anchor: { kind: 'epoch', epoch: 'go', edge: 'start' }, offset: { value: 0.3, unit: 's' } } },
    { kind: 'moment', at: { anchor: { kind: 'epoch', epoch: 'go', edge: 'end' } } },
  ] };
  expect(windowText(window, host(window, []))).toBe('reach 2, [go + 0.1 s, go + 0.3 s), @go.end');
});

// Catches: an editor that stores a window the Service refuses, or leaves an empty span list instead of the whole trial.
test('every window the editor stores is one the contract accepts, and removing the last span is the whole trial', async () => {
  const stored: (Window | null)[] = [];
  const user = userEvent.setup();
  const { rerender } = render(<WindowEditor window={null} host={host(null, stored)} />);
  await user.click(screen.getByRole('button', { name: 'Add Span' }));
  await user.click(await screen.findByRole('menuitem', { name: 'Interval' }));
  const interval = stored.at(-1)!;
  expect(WindowSchema.parse(interval)).toEqual(interval);
  expect(interval.spans[0]).toMatchObject({ kind: 'interval', start: { anchor: { epoch: 'go', edge: 'start' } }, end: { anchor: { epoch: 'go', edge: 'end' } } });

  rerender(<WindowEditor window={interval} host={host(interval, stored)} />);
  const end = screen.getByRole('group', { name: 'End' });
  await user.click(within(end).getByRole('combobox', { name: 'Anchor End' }));
  await user.click(await screen.findByRole('option', { name: 'Length' }));
  const lengthened = stored.at(-1)!;
  expect(WindowSchema.parse(lengthened)).toEqual(lengthened);
  expect(lengthened.spans[0]).toMatchObject({ kind: 'interval', end: { offset: { value: 0, unit: 's' } } });
  expect('anchor' in (lengthened.spans[0] as { end: object }).end).toBe(false);

  rerender(<WindowEditor window={lengthened} host={host(lengthened, stored)} />);
  await user.click(screen.getByRole('combobox', { name: 'Kind Span 1' }));
  await user.click(await screen.findByRole('option', { name: 'Epoch' }));
  expect(stored.at(-1)).toEqual({ spans: [{ kind: 'epoch', epoch: 'go' }] });

  rerender(<WindowEditor window={stored.at(-1)!} host={host(stored.at(-1)!, stored)} />);
  await user.click(screen.getByRole('combobox', { name: 'Epoch Span 1' }));
  await user.click(await screen.findByRole('option', { name: 'reach 2' }));
  expect(stored.at(-1)).toEqual({ spans: [{ kind: 'epoch', epoch: 'reach', occurrence: 1 }] });

  await user.click(screen.getByRole('button', { name: 'Remove Span 1' }));
  expect(stored.at(-1)).toBeNull();
});
