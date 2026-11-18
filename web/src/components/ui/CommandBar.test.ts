import { expect, it } from 'vitest';
import { fitBar, type BarWidths } from './CommandBar';

const bar = (available: number): BarWidths => ({
  available, gap: 0, trigger: 24, pinnedNamed: [40, 40], pinnedGlyph: [24, 24],
  entries: [
    { width: 70, priority: 'low' },
    { width: 100, priority: 'normal' },
    { width: 60, priority: 'normal' },
    { width: 50, priority: 'low' },
  ],
});

const fit = (available: number) => {
  const { compact, hidden } = fitBar(bar(available));
  return { compact, hidden: [...hidden].sort() };
};

it('moves low entries to the menu before normal ones, the trailing one first, and keeps the pinned commands', () => {
  expect(fit(400)).toEqual({ compact: false, hidden: [] });
  expect(fit(334)).toEqual({ compact: false, hidden: [3] });
  expect(fit(300)).toEqual({ compact: false, hidden: [0, 3] });
  expect(fit(250)).toEqual({ compact: false, hidden: [0, 2, 3] });
  expect(fit(200)).toEqual({ compact: false, hidden: [0, 1, 2, 3] });
  expect(fit(100)).toEqual({ compact: true, hidden: [0, 1, 2, 3] });
});
