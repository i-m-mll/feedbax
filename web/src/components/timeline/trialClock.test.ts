import { expect, it } from 'vitest';
import { defaultTrialClock, shownPick, type TrialClock, type TrialPick } from './trialClock';

const train = { nominal: true, samples: 8, recorded: 0 };
const evaluate = { nominal: false, samples: 0, recorded: 64 };
const picking = (picked: TrialPick): TrialClock => ({ ...defaultTrialClock, picked });

// Catches a step change that drops All or Mean back to a single trial, and a Mean shown where no set has one.
it('keeps All and Mean across steps whose sets differ, and shows Mean again on returning to a step that has it', () => {
  expect(shownPick(picking({ trials: 'samples', showing: { kind: 'all', index: 3 } }), evaluate))
    .toEqual({ trials: 'recorded', showing: { kind: 'all', index: 0 } });
  expect(shownPick(picking({ trials: 'recorded', showing: { kind: 'all', index: 10 } }), train))
    .toEqual({ trials: 'samples', showing: { kind: 'all', index: 0 } });
  const mean = picking({ trials: 'recorded', showing: { kind: 'mean', index: 0 } });
  expect(shownPick(mean, train)).toEqual({ trials: 'nominal', showing: { kind: 'one', index: 0 } });
  expect(shownPick(mean, evaluate)).toEqual(mean.picked);
});
