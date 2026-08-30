import { describe, expect, it } from 'vitest';
import {
  stageMetadataWithExecutionTarget,
  trainingProtocolSnapshot,
  trainingSpecWithProtocolPatch,
} from '@/utils/stageProtocol';
import type { TrainingSpec } from '@/types/training';
import type { StudioScenarioSpec, StudioStageSpec } from '@/types/workspace';

const trainingSpec: TrainingSpec = {
  optimizer: { type: 'adam', params: { learning_rate: 0.003 } },
  loss: { type: 'Composite', label: 'loss', weight: 1, children: {} },
  n_batches: 200,
  batch_size: 32,
  checkpoint_interval: 25,
};

const stage: StudioStageSpec = {
  id: 'stage:train',
  kind: 'train',
  label: 'Train',
  status: 'draft',
  scenario_id: 'scenario:train',
  input_collections: [],
  output_collections: [],
  manifest_refs: [],
  artifact_refs: [],
  execution_spec: null,
  selection_spec: {},
  validation: { valid: null, checked_at: null, errors: [], warnings: [], metadata: {} },
  ui_state: {},
  metadata: { backend_realization: { execution_target: 'gcp' } },
};

const scenario = {
  id: 'scenario:train',
  training_spec: trainingSpec,
} as StudioScenarioSpec;

describe('stage protocol helpers', () => {
  it('derives protocol state from stage realization metadata and scenario training spec', () => {
    expect(trainingProtocolSnapshot(stage, scenario)).toEqual({
      learningRate: 0.003,
      batchCount: 200,
      batchSize: 32,
      checkpointInterval: 25,
      computeTarget: 'gcp',
    });
  });

  it('distinguishes RunPod from GCP', () => {
    expect(trainingProtocolSnapshot(stage, scenario).computeTarget).toBe('gcp');
    expect(
      trainingProtocolSnapshot(
        { ...stage, metadata: { backend_realization: { execution_target: 'runpod' } } },
        scenario
      ).computeTarget
    ).toBe('runpod');
  });

  it('patches stage protocol and training spec independently', () => {
    expect(stageMetadataWithExecutionTarget(stage, 'manual')).toEqual({
      backend_realization: { execution_target: 'manual' },
    });
    expect(
      trainingSpecWithProtocolPatch(trainingSpec, {
        learningRate: 0.01,
        batchCount: 500,
        checkpointInterval: null,
      })
    ).toMatchObject({
      optimizer: { params: { learning_rate: 0.01 } },
      n_batches: 500,
      batch_size: 32,
      checkpoint_interval: undefined,
    });
  });
});
