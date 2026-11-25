import { describe, expect, it } from 'vitest';
import type { ModelAddress, TrialRecord } from '@/generated/studioContracts';
import { addressKey } from '../model/modelGraph';
import { trialRows, type ShelfTask } from './shelfRows';

const pulse: ModelAddress = { composites: [], element: { kind: 'attachment', parent: { kind: 'wire', id: 'drive' }, id: 'pulse' } };
const curl: ModelAddress = { composites: [], element: { kind: 'attachment', parent: { kind: 'wire', id: 'force' }, id: 'curl' } };

const task: ShelfTask = {
  document: 'task', epochs: [], signals: [], duration: 1,
  interventions: { document: 'model', items: [{ address: pulse, name: 'Add drive' }] },
};

const record = (windows: Record<string, boolean[]>) => ({
  id: 't', timeline: [], intervention_windows: windows,
} as unknown as TrialRecord);

describe('intervention rows', () => {
  // Catches: a window drawn a step late or closed at its end, or one a Set entry adds left off the shelf.
  it("draws each intervention over the steps its record marks, the Model's and those only the record names", () => {
    const rows = trialRows(task, record({
      [addressKey(pulse)]: [false, true, true, false],
      [addressKey(curl)]: [false, false, true, true],
    }), 0.1);
    const bars = Object.fromEntries(rows.flatMap((row) => row.kind === 'bars' ? [[row.key, row.bars]] : []));
    expect(bars[`intervention:${addressKey(pulse)}`]).toEqual([{ start: 0.1, end: expect.closeTo(0.3) }]);
    expect(bars[`intervention:${addressKey(curl)}`]).toEqual([{ start: expect.closeTo(0.2), end: expect.closeTo(0.4) }]);
  });
});
