import { describe, expect, it } from 'vitest';
import { createRlrmpModelGraph } from '@/data/rlrmp-model-graph';
import {
  createRlrmpMovementRampAnalysis,
  seedRlrmpMovementRampWorkspace,
} from '@/data/rlrmp-run-example';
import { buildWorkspaceSnapshot } from '@/stores/workspaceStore';
import { defaultTaskSpec, defaultTrainingSpec } from '@/stores/trainingStore';
import {
  bestTrainingRun,
  evaluationProtocolLabel,
  evaluationRunSummaries,
  selectedIds,
  trainingInputSummaries,
  trainingRunSummaries,
} from '@/utils/pipelineCollections';

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

describe('pipeline collection summaries', () => {
  it('summarizes training runs by user-facing variant and metrics', () => {
    const workspace = seededWorkspace();
    const trainStage = workspace.stages.find((stage) => stage.kind === 'train');
    const rows = trainingRunSummaries(trainStage);

    expect(rows).toHaveLength(7);
    expect(rows[0]).toMatchObject({
      label: 'Power 6 ramp, duration 80',
      finalValidationLoss: 0.1021,
      rampDurationSteps: 80,
      sourceIssue: 'b399efc',
    });
    expect(bestTrainingRun(rows)?.id).toBe(
      'rlrmp:b399efc:movement_ramp__power6_dur80'
    );
  });

  it('summarizes evaluation input selection without exposing collection ids', () => {
    const workspace = seededWorkspace();
    const evalStage = workspace.stages.find((stage) => stage.kind === 'eval');

    expect(trainingInputSummaries(evalStage)).toHaveLength(7);
    expect(selectedIds(evalStage, 'training_run_ids')).toEqual([
      'rlrmp:b399efc:movement_ramp__power6_dur80',
    ]);
  });

  it('summarizes completed evaluation protocol details', () => {
    const workspace = seededWorkspace();
    const evalStage = workspace.stages.find((stage) => stage.kind === 'eval');
    const [summary] = evaluationRunSummaries(evalStage);

    expect(summary.label).toBe('8-direction validation, SISU 0.5, zero perturbation');
    expect(evaluationProtocolLabel(summary)).toBe(
      '8-direction center-out - SISU 0.5 - no perturbation'
    );
  });
});
