import { describe, expect, it } from 'vitest';
import type { TrainingRunSummary } from '@/utils/pipelineCollections';
import {
  progressBindingsForRuns,
  sortTrainingRows,
  stageWithTrainingRunLifecyclePatch,
} from '@/utils/trainRunTable';
import type { StudioStageSpec } from '@/types/workspace';

const baseRun: TrainingRunSummary = {
  id: 'run:a',
  label: 'Run A',
  status: 'pending',
  variant: null,
  rampShape: null,
  rampDurationSteps: null,
  nnOutputPreGo: null,
  finalValidationLoss: null,
  velocityRmse: null,
  peakVelocityMean: null,
  peakVelocitySd: null,
  holdDriftMeanMm: null,
  holdDriftSdMm: null,
  metrics: {},
  replicateCount: null,
  batchSize: null,
  warmupBatches: null,
  checkpointAvailable: false,
  sourceIssue: null,
  provenanceId: 'run:a',
  uri: null,
  jobId: null,
  axisCoordinates: {},
  runSetId: null,
  planned: true,
  supersededBy: null,
  supersedes: null,
  statusReason: null,
  stale: false,
  staleReason: null,
  specHashComparisons: [],
};

function run(patch: Partial<TrainingRunSummary>): TrainingRunSummary {
  return { ...baseRun, ...patch };
}

function stage(): StudioStageSpec {
  const ref = {
    kind: 'TrainingRun',
    id: 'run:a',
    role: 'training_run',
    provider: 'manifest',
    uri: '/tmp/run-a.json',
    metadata: { name: 'Run A', status: 'pending', planned: true },
  };
  return {
    id: 'stage:train',
    kind: 'train',
    label: 'Train',
    status: 'draft',
    scenario_id: 'scenario:train',
    input_collections: [],
    output_collections: [{
      id: 'training:runs',
      kind: 'training_runs',
      item_refs: [ref],
      filters: {},
      facets: {},
      metadata: {},
    }],
    manifest_refs: [ref],
    selection_spec: {},
    validation: { valid: null, checked_at: null, errors: [], warnings: [], metadata: {} },
    ui_state: {},
    metadata: {},
  };
}

describe('train run table utilities', () => {
  it('sorts varied axis columns numerically and textually', () => {
    const rows = [
      run({ id: 'run:3', label: 'C', axisCoordinates: { lr: 0.3, seed: 'b' } }),
      run({ id: 'run:1', label: 'A', axisCoordinates: { lr: 0.1, seed: 'c' } }),
      run({ id: 'run:2', label: 'B', axisCoordinates: { lr: 0.2, seed: 'a' } }),
    ];

    expect(sortTrainingRows(rows, { key: 'axis:lr', direction: 'asc' }).map((row) => row.id))
      .toEqual(['run:1', 'run:2', 'run:3']);
    expect(sortTrainingRows(rows, { key: 'axis:seed', direction: 'asc' }).map((row) => row.id))
      .toEqual(['run:2', 'run:3', 'run:1']);
  });

  it('binds shared job progress to the run-set group instead of every row', () => {
    const rows = [
      run({ id: 'run:a', jobId: 'job:matrix', runSetId: 'set:matrix' }),
      run({ id: 'run:b', jobId: 'job:matrix', runSetId: 'set:matrix' }),
    ];

    const binding = progressBindingsForRuns(rows, {
      job_id: 'job:matrix',
      batch: 2,
      total_batches: 10,
    });

    expect(binding.byRunId.size).toBe(0);
    expect(binding.byGroupId.get('set:matrix')).toBe('2/10');
  });

  it('patches manifest-backed collections after lifecycle actions', () => {
    const updated = stageWithTrainingRunLifecyclePatch(stage(), 'update', {
      id: 'run:a',
      name: 'Run A',
      createdAt: '2026-07-07T00:00:00Z',
      status: 'cancelled',
      hyperparams: {},
      planned: true,
      checkpointAvailable: false,
      provenanceId: 'run:a',
    });

    expect(updated.output_collections[0].item_refs[0].metadata.status).toBe('cancelled');
    expect(updated.manifest_refs[0].metadata.status).toBe('cancelled');

    const removed = stageWithTrainingRunLifecyclePatch(updated, 'remove', {
      id: 'run:a',
      name: 'Run A',
      createdAt: '2026-07-07T00:00:00Z',
      status: 'cancelled',
      hyperparams: {},
    });

    expect(removed.output_collections[0].item_refs).toEqual([]);
    expect(removed.manifest_refs).toEqual([]);
  });
});
