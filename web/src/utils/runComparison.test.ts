import { describe, expect, it } from 'vitest';
import type { TrainingRunSummary } from '@/utils/pipelineCollections';
import {
  buildTrainingRunComparison,
  trainingCompareFields,
  visibleCompareFields,
} from '@/utils/runComparison';

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

describe('run comparison utilities', () => {
  it('collapses identical fields while keeping changed params and metrics visible', () => {
    const rows = [
      run({
        id: 'run:a',
        rampShape: 'linear',
        batchSize: 32,
        metrics: { final_validation_loss: 0.2 },
        axisCoordinates: { lr: 0.001 },
      }),
      run({
        id: 'run:b',
        rampShape: 'linear',
        batchSize: 32,
        metrics: { final_validation_loss: 0.1 },
        axisCoordinates: { lr: 0.0003 },
      }),
    ];
    const fields = trainingCompareFields(
      [{ id: 'lr', label: 'Learning rate' }],
      [{
        id: 'final_validation_loss',
        label: 'Loss',
        units: null,
        source: 'manifest',
        summary: null,
        metadata: {},
      }],
    );

    const comparison = buildTrainingRunComparison(rows, fields);
    expect(comparison.paramFields.find((field) => field.id === 'lr')).toMatchObject({
      identical: false,
      values: { 'run:a': 0.001, 'run:b': 0.0003 },
    });
    expect(comparison.paramFields.find((field) => field.id === 'batch_size')).toMatchObject({
      identical: true,
    });
    expect(visibleCompareFields(comparison.paramFields, false).map((field) => field.id))
      .not.toContain('batch_size');
    expect(visibleCompareFields(comparison.metricFields, false).map((field) => field.id))
      .toEqual(['final_validation_loss']);
  });

  it('uses fetched bounded values ahead of local row fallbacks', () => {
    const rows = [
      run({ id: 'run:a', metrics: { final_validation_loss: 0.9 } }),
      run({ id: 'run:b', metrics: { final_validation_loss: 0.8 } }),
    ];
    const fields = trainingCompareFields(
      [],
      [{
        id: 'final_validation_loss',
        label: 'Loss',
        units: null,
        source: 'manifest',
        summary: null,
        metadata: {},
      }],
    );

    const comparison = buildTrainingRunComparison(rows, fields, {
      rows: [
        { id: 'run:a', params: {}, metrics: { final_validation_loss: 0.2 } },
        { id: 'run:b', params: {}, metrics: { final_validation_loss: 0.1 } },
      ],
    });

    expect(comparison.metricFields[0].values).toEqual({
      'run:a': 0.2,
      'run:b': 0.1,
    });
  });
});
