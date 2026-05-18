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
        }),
        expect.objectContaining({
          id: 'peak_velocity_m_per_s',
          source: 'analysis',
          sourceId: expect.stringContaining('summary_metrics'),
        }),
        expect.objectContaining({
          id: 'final_validation_loss',
          source: 'manifest',
          sourceId: 'rlrmp:b399efc:movement_ramp__power6_dur80',
        }),
      ])
    );
  });

  it('maps artifact and analysis outputs into workspace overlay descriptors', () => {
    const workspace = seededWorkspace();
    const overlays = artifactOverlaysForWorkspace(workspace);

    expect(overlays).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: 'Forward Velocity Profiles',
          source: 'artifact',
          uri: '_artifacts/b399efc/figures/forward_velocity_profiles/figure.html',
        }),
        expect.objectContaining({
          label: 'Forward velocity profiles',
          source: 'analysis',
          uri: 'results/b399efc/figures/forward_velocity_profiles',
        }),
        expect.objectContaining({
          source: 'evaluation',
          stageId: 'stage:eval',
        }),
      ])
    );
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
