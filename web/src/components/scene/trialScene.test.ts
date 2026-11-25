import { describe, expect, it } from 'vitest';
import type { TrialRecord } from '@/generated/studioContracts';
import { recordedTracks } from './trialScene';

/** A hand moving one unit in x per state frame, through a fixed epoch, an Until epoch ended at `outcome`, and Rest. */
const record = (outcome: 2 | 3): TrialRecord => ({
  id: 'trial:0', inits: {}, inputs: {}, present: {}, variables: {}, variable_defined: {},
  timeline: [
    { epoch: 'hold', label: 'hold', start: 0, end: 2, outcome: 1 },
    { epoch: 'reach', label: 'reach', start: 2, end: 5, outcome },
    { epoch: 'rest', label: 'rest', start: 5, end: 8, outcome: 1 },
  ],
  states: { hand: { frame: 'world', units: 'm', samples: Array.from({ length: 9 }, (_, frame) => [frame, 0]) } },
});

describe('recordedTracks', () => {
  // Catches a mark one frame off the step where the epoch ended, a limit drawn as a condition, a mark on an epoch that
  // ended at its length, a mark on a trial beside the picked one, and a mark ahead of the playhead under Trails.
  it('marks where each Until epoch of the picked trial ended, and how', () => {
    expect(recordedTracks(record(2), 0, false, true)[0].ended).toEqual([{ point: [5, 0], toward: [6, 0], by: 'condition' }]);
    expect(recordedTracks(record(3), 0, false, true)[0].ended).toEqual([{ point: [5, 0], toward: [6, 0], by: 'limit' }]);
    expect(recordedTracks(record(2), 0, false, false)[0].ended).toEqual([]);
    expect(recordedTracks(record(2), 4, true, true)[0].ended).toEqual([]);
    expect(recordedTracks(record(2), 5, true, true)[0].ended).toEqual([{ point: [5, 0], toward: [6, 0], by: 'condition' }]);
  });
});
