import { describe, expect, it } from 'vitest';
import type { SampledTaskTrial } from '@/api/client';
import {
  compactTimelineCues,
  sampledPreviewTrialLabel,
  timelineCueOffset,
} from '@/features/scenario/samplePreview';

const trial: SampledTaskTrial = {
  id: 'trial:0',
  index: 0,
  start: [0, 0],
  goal: [1, 0],
  n_steps: 11,
  timeline: [
    { label: 'movement', step: 6, kind: 'epoch' },
    { label: 'prep', step: 0, kind: 'epoch' },
    { label: 'go_cue', step: 6, kind: 'event' },
    { label: 'settle', step: 9, kind: 'epoch' },
    { label: 'done', step: 10, kind: 'event' },
  ],
  metadata: {},
};

describe('sample preview helpers', () => {
  it('formats stable trial labels', () => {
    expect(sampledPreviewTrialLabel(trial)).toBe('T1');
  });

  it('converts timeline steps into clamped offsets', () => {
    expect(timelineCueOffset({ step: 5 }, 11)).toBe(0.5);
    expect(timelineCueOffset({ step: 99 }, 11)).toBe(1);
  });

  it('sorts and limits timeline cues for compact rendering', () => {
    expect(compactTimelineCues(trial, 3).map((cue) => cue.label)).toEqual([
      'prep',
      'go_cue',
      'movement',
    ]);
  });
});
