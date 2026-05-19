import { describe, expect, it } from 'vitest';
import type { ScenarioMetricSpec } from '@/features/scenario/integration';
import type { TrainingRunSummary } from '@/utils/pipelineCollections';
import { formatMetricWithUnits, runMetricColumns } from '@/utils/runMetricColumns';

const baseRun: TrainingRunSummary = {
  id: 'run:1',
  label: 'Run 1',
  status: 'completed',
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
  provenanceId: 'run:1',
  uri: null,
};

function metric(id: string, source: ScenarioMetricSpec['source']): ScenarioMetricSpec {
  return {
    id,
    label: id === 'custom_success_rate' ? 'Success rate' : 'Final validation loss',
    role: 'metric',
    source,
    selector: null,
    units: id === 'custom_success_rate' ? '%' : null,
    stageId: 'stage:train',
    scenarioId: 'scenario:train',
    sourceId: `${source}:${id}`,
    summary: null,
    metadata: {},
  };
}

describe('run metric columns', () => {
  it('keeps table columns driven by metric specs with row values', () => {
    const rows = [
      {
        ...baseRun,
        metrics: {
          final_validation_loss: 0.12,
          custom_success_rate: 94.2,
        },
      },
    ];

    expect(
      runMetricColumns(
        [metric('custom_success_rate', 'objective'), metric('final_validation_loss', 'manifest')],
        rows
      )
    ).toEqual([
      expect.objectContaining({ id: 'final_validation_loss', label: 'Loss' }),
      expect.objectContaining({ id: 'custom_success_rate', label: 'Success rate', units: '%' }),
    ]);
  });

  it('does not create columns for metric specs with no compatible row values', () => {
    expect(runMetricColumns([metric('custom_success_rate', 'objective')], [baseRun])).toEqual([]);
    expect(formatMetricWithUnits(1.25, 'm/s')).toBe('1.250 m/s');
  });
});
