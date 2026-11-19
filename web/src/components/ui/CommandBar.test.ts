import { expect, it } from 'vitest';
import { fitBar, type BarWidths } from './CommandBar';

const bar = (available: number): BarWidths => ({
  available, gap: 0, trigger: 24, pinned: [24, 24],
  entries: [
    { width: 70, priority: 'low' },
    { width: 100, priority: 'normal' },
    { width: 60, priority: 'normal' },
    { width: 50, priority: 'low' },
  ],
});

const hidden = (available: number) => [...fitBar(bar(available))].sort();

it('moves low entries to the menu before normal ones, the trailing one first, and keeps the pinned commands', () => {
  expect(hidden(328)).toEqual([]);
  expect(hidden(327)).toEqual([3]);
  expect(hidden(301)).toEqual([0, 3]);
  expect(hidden(231)).toEqual([0, 2, 3]);
  expect(hidden(171)).toEqual([0, 1, 2, 3]);
});
