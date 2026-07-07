import { describe, expect, it } from 'vitest';
import { defaultTaskSpec, defaultTrainingSpec } from '@/stores/trainingStore';
import type { StudioScenarioSpec, StudioStageSpec } from '@/types/workspace';
import {
  bulkEditGhostRows,
  expandTrainMatrix,
  ghostRowsForMatrix,
  initialMatrixSpec,
  parseAxisValuesInput,
  runCountExpression,
  selectionSpecForMatrix,
  trainAxisColumns,
  validateAxisPath,
  type TrainMatrixSpec,
} from '@/utils/trainMatrix';

function scenario(): StudioScenarioSpec {
  return {
    id: 'scenario:train',
    schema_version: 'feedbax.studio.scenario.v1',
    label: 'Train',
    stage_id: 'stage:train',
    graph: null,
    graph_ui_state: null,
    training_spec: defaultTrainingSpec,
    task_spec: defaultTaskSpec,
    task_binding_spec: null,
    objective_spec: null,
    probe_specs: [],
    temporal_spec: null,
    biomechanics_spec: null,
    analysis_spec: null,
    report_spec: null,
    validation: { valid: null, checked_at: null, errors: [], warnings: [], metadata: {} },
    ui_state: {},
    metadata: {},
  };
}

function stage(selectionSpec: Record<string, unknown>): StudioStageSpec {
  return {
    id: 'stage:train',
    kind: 'train',
    label: 'Train',
    status: 'draft',
    scenario_id: 'scenario:train',
    input_collections: [],
    output_collections: [],
    manifest_refs: [],
    selection_spec: selectionSpec,
    validation: { valid: null, checked_at: null, errors: [], warnings: [], metadata: {} },
    ui_state: {},
    metadata: {},
  };
}

describe('train matrix utilities', () => {
  it('expands cross and zip matrix coordinates deterministically', () => {
    const matrix: TrainMatrixSpec = {
      name: 'Demo',
      mode: 'cross',
      axes: [
        { id: 'lr', label: 'LR', path: 'training_spec.optimizer.params.learning_rate', values: [1, 2], source: 'manual' },
        { id: 'seed', label: 'Seed', path: 'seed', values: [11, 22, 33], source: 'manual' },
      ],
    };

    expect(runCountExpression(matrix.axes, 'cross')).toBe('2 x 3 = 6 runs');
    expect(expandTrainMatrix(matrix.axes, 'cross')).toHaveLength(6);
    expect(expandTrainMatrix(matrix.axes, 'zip')).toEqual([]);
    expect(runCountExpression(matrix.axes, 'zip')).toBe('2 zip 3 = mismatch');
  });

  it('builds backend selection_spec.matrix payloads and ghost rows', () => {
    const matrix: TrainMatrixSpec = {
      name: 'Loss sweep',
      mode: 'cross',
      axes: [
        { id: 'loss_weight', label: 'loss.weight', path: 'training_spec.loss.weight', values: [0, 1e-5], source: 'manual' },
      ],
    };

    expect(selectionSpecForMatrix({ existing: true }, matrix).matrix).toMatchObject({
      name: 'Loss sweep',
      mode: 'cross',
      axes: [{ id: 'loss_weight', path: 'training_spec.loss.weight', values: [0, 1e-5] }],
    });
    expect(ghostRowsForMatrix(matrix).map((row) => row.axisCoordinates)).toEqual([
      { loss_weight: 0 },
      { loss_weight: 1e-5 },
    ]);
  });

  it('reads existing matrix selection and validates strict paths', () => {
    const spec = initialMatrixSpec(
      stage({
        matrix: {
          name: 'Batch sweep',
          mode: 'zip',
          axes: [{ id: 'batch_size', path: 'training_spec.batch_size', values: [64, 128] }],
        },
      }),
      scenario()
    );

    expect(spec).toMatchObject({ name: 'Batch sweep', mode: 'zip' });
    expect(spec.axes[0]).toMatchObject({ id: 'batch_size', values: [64, 128] });
    expect(validateAxisPath('training_spec.batch_size', scenario())).toBeNull();
    expect(validateAxisPath('training_spec.missing_field', scenario())).toContain('missing_field');
    expect(validateAxisPath('seed', scenario())).toBeNull();
  });

  it('derives dynamic axis columns and bulk-edit previews', () => {
    const rows = [
      {
        id: 'run:a',
        label: 'Run A',
        status: 'pending',
        axisCoordinates: { lr: 0.1 },
        runSetId: 'set:a',
      },
      {
        id: 'run:b',
        label: 'Run B',
        status: 'pending',
        axisCoordinates: { lr: 0.2 },
        runSetId: 'set:a',
      },
    ] as any[];

    expect(trainAxisColumns(rows)).toEqual([{ id: 'lr', label: 'lr' }]);
    expect(parseAxisValuesInput('1, 2, true')).toEqual([1, 2, true]);
    expect(
      bulkEditGhostRows({
        rows,
        axis: { id: 'lr', label: 'LR' },
        verb: 'cross',
        values: [0.3, 0.4],
      })
    ).toHaveLength(4);
  });
});
