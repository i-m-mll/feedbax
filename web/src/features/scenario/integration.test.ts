import { describe, expect, it } from 'vitest';
import { createRlrmpModelGraph } from '@/data/rlrmp-model-graph';
import {
  createRlrmpMovementRampAnalysis,
  seedRlrmpMovementRampWorkspace,
} from '@/data/rlrmp-run-example';
import { buildWorkspaceSnapshot } from '@/stores/workspaceStore';
import { defaultTaskSpec, defaultTrainingSpec } from '@/stores/trainingStore';
import {
  artifactOverlaysForWorkspace,
  scenarioMetricSpecs,
  stageProductReferences,
} from '@/features/scenario/integration';

function seededWorkspace() {
  const { graph, uiState } = createRlrmpModelGraph('RLRMP movement-ramp runs');
  const analysisSnapshot = createRlrmpMovementRampAnalysis();
  return seedRlrmpMovementRampWorkspace(
    buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec: defaultTrainingSpec,
      taskSpec: defaultTaskSpec,
      analysisSnapshot,
      projectName: 'RLRMP movement-ramp runs',
    })
  );
}

describe('scenario integration derivation', () => {
  it('derives metric specs from task defaults, analysis pages, and imported manifests', () => {
    const workspace = seededWorkspace();
    const metrics = scenarioMetricSpecs(workspace);

    expect(metrics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: 'target_reach_error',
          source: 'task_default',
          selector: expect.stringContaining('task:'),
          valueSchema: expect.objectContaining({
            id: 'value:task_data:targets',
            kind: 'task_data',
          }),
          metadata: expect.objectContaining({
            task_data_schema: expect.objectContaining({
              id: 'task_data:targets',
            }),
            temporal_aggregation: 'final',
          }),
        }),
        expect.objectContaining({
          id: 'peak_velocity_m_per_s',
          source: 'analysis',
          sourceId: expect.stringContaining('summary_metrics'),
          valueSchema: expect.objectContaining({
            kind: 'metric',
            units: 'm/s',
          }),
        }),
        expect.objectContaining({
          id: 'final_validation_loss',
          source: 'manifest',
          sourceId: 'rlrmp:b399efc:movement_ramp__power6_dur80',
          metadata: expect.objectContaining({
            value_schema: expect.objectContaining({
              id: 'value:metric:manifest:final_validation_loss',
            }),
          }),
        }),
      ])
    );
  });

  it('does not treat ordinary analysis figures as workspace overlays', () => {
    const workspace = seededWorkspace();
    const overlays = artifactOverlaysForWorkspace(workspace);

    expect(overlays).toEqual([]);
  });

  it('exposes stage-owned analysis and report references', () => {
    const workspace = seededWorkspace();
    const analysisStage = workspace.stages.find((stage) => stage.kind === 'analysis');
    const reportStage = workspace.stages.find((stage) => stage.kind === 'report');

    expect(stageProductReferences(workspace, analysisStage?.id)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: 'analysis_page',
          label: 'b399efc summary',
          manifestIds: ['rlrmp:b399efc:eval:centerout-sisu0.5-zero-perturbation'],
        }),
      ])
    );
    expect(stageProductReferences(workspace, reportStage?.id)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: 'report_section',
          collectionId: 'collection:b399efc-analysis-products',
          manifestIds: ['rlrmp:b399efc:analysis:summary-products'],
        }),
      ])
    );
  });
});
