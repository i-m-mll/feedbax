import { describe, expect, it } from 'vitest';
import type { StudioProjectWindowMapProblem } from '@/generated/studioContracts';
import { problemEntries } from './StudioProblems';

const missing = (step: string, row: string): StudioProjectWindowMapProblem => ({
  code: 'missing_package', path: `payload.operations.${step}.operation`, detail: 'Package rlrmp3 failed to import.',
  variant: 'default', step, row, place: null, package: 'rlrmp3',
});

describe('problemEntries', () => {
  it('lists an unavailable package once for the project, however many runs need it', () => {
    const project: StudioProjectWindowMapProblem = {
      ...missing('hold', 'default'), code: 'package_unavailable', path: '', variant: null, step: null, row: null,
    };
    const entries = problemEntries([project, missing('hold', 'default'), missing('reach', 'default'), missing('reach', 'b')]);
    expect(entries).toHaveLength(1);
    expect(entries[0].problem.step).toBeNull();
    expect(entries[0].runs).toBe(3);
  });

  it("lists a step's package Problem as the project's when only that step is shown", () => {
    const [entry] = problemEntries([missing('hold', 'default')]);
    expect(entry.problem.step).toBeNull();
  });
});
