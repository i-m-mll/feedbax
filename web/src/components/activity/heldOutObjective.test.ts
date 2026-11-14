import { describe, expect, it } from 'vitest';
import type { StudioEvaluationObjectiveSeries } from '@/generated/studioContracts';
import { heldOutTracks, type ObjectiveStage } from './heldOutObjective';

const stage: ObjectiveStage = {
  offset: 12, origin: 6, through: null, request: 'train:request',
  starts: new Map([['first', 6], ['resumed', 9]]),
  cutoffs: new Map([['resumed', Infinity], ['first', 9]]),
};
const digest = 'a'.repeat(64);
const receipt = { manifest_id: 'receipt', manifest_sha256: digest };

function point(update: number, job: string, coverage = 1) {
  return {
    checkpoint: {
      job_id: job, request_id: stage.request, completed_updates: update, transaction: receipt,
      transaction_root_sha256: digest, binding_sha256: digest, publication_cursor: update,
    },
    total: update,
    terms: [{ path: '/effort', name: 'effort', coverage, value: update / 2 }],
  };
}

const result: StudioEvaluationObjectiveSeries = {
  job_id: 'evaluate', request_id: 'evaluate:request', receipt, lineage: ['first', 'resumed'],
  condition: digest, objective: digest, trials: ['a'], seed: 1,
  requested: [12, 8, 9, 10, 11, 13],
  points: [
    point(12, 'resumed'), point(8, 'first'), point(9, 'first', 0),
    point(11, 'resumed'), point(13, 'unrelated'),
  ],
  gaps: [{ update: 10, reason: 'checkpoint_unavailable' }],
};

describe('held-out objective tracks', () => {
  it('uses the retained attempt clock and splits missing and uncovered observations', () => {
    const [total, effort] = heldOutTracks('evaluate', result, stage);
    expect(total.segments.map((segment) => segment.map(({ update }) => update))).toEqual([[14], [17, 18]]);
    expect(effort.segments.map((segment) => segment.map(({ value }) => value))).toEqual([[4], [5.5, 6]]);
  });

  it('keeps a stopped ancestor and excludes siblings and unknown recovery boundaries', () => {
    const stopped = { ...result, lineage: ['first'], points: [point(8, 'first'), point(11, 'first')] };
    const [total] = heldOutTracks('evaluate', stopped, stage);
    expect(total.segments.map((segment) => segment.map(({ update }) => update))).toEqual([[14]]);
    expect(heldOutTracks('evaluate', { ...result, lineage: ['first', 'sibling'] }, stage)).toEqual([]);
    expect(heldOutTracks('evaluate', stopped, { ...stage, starts: new Map([['first', 6]]) })).toEqual([]);
    expect(heldOutTracks('evaluate', { ...result, points: [] }, stage)).toEqual([]);
  });

  it('keeps a plotted publication identity through finalization and adds the backfilled point once', () => {
    const live = { ...result, receipt: null, points: [point(8, 'first'), point(12, 'resumed')] };
    const [active] = heldOutTracks('evaluate', live, stage);
    const [retained] = heldOutTracks('evaluate', result, stage);
    expect(retained.id).toBe(active.id);
    expect(active.segments.flat()).toEqual(retained.segments.flat().filter(({ update }) => update !== 17));
    const all = retained.segments.flat().map(({ id }) => id);
    expect(new Set(all).size).toBe(all.length);
  });
});
