import { describe, expect, it } from 'vitest';
import { createRlrmpModelGraph } from '@/data/rlrmp-model-graph';
import {
  createRlrmpMovementRampAnalysis,
  seedRlrmpMovementRampWorkspace,
} from '@/data/rlrmp-run-example';
import { buildWorkspaceSnapshot } from '@/stores/workspaceStore';
import { defaultTaskSpec, defaultTrainingSpec } from '@/stores/trainingStore';

describe('RLRMP movement-ramp run example', () => {
  it('seeds completed training runs through the pipeline stages', () => {
    const { graph, uiState } = createRlrmpModelGraph('RLRMP movement-ramp runs');
    const analysisSnapshot = createRlrmpMovementRampAnalysis();
    const baseWorkspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec: defaultTrainingSpec,
      taskSpec: defaultTaskSpec,
      analysisSnapshot,
      projectName: 'RLRMP movement-ramp runs',
    });

    const workspace = seedRlrmpMovementRampWorkspace(baseWorkspace);
    const trainStage = workspace.stages.find((stage) => stage.kind === 'train');
    const evalStage = workspace.stages.find((stage) => stage.kind === 'eval');
    const analysisStage = workspace.stages.find((stage) => stage.kind === 'analysis');

    expect(workspace.active_stage_id).toBe('stage:eval');
    expect(trainStage?.status).toBe('completed');
    expect(trainStage?.output_collections[0].item_refs).toHaveLength(7);
    expect(evalStage?.input_collections[0].item_refs).toHaveLength(7);
    expect(evalStage?.selection_spec.training_run_ids).toEqual([
      'rlrmp:b399efc:movement_ramp__power6_dur80',
    ]);
    expect(evalStage?.output_collections[0].item_refs[0].id).toBe(
      'rlrmp:b399efc:eval:centerout-sisu0.5-zero-perturbation'
    );
    expect(analysisStage?.input_collections[0].item_refs[0].id).toBe(
      'rlrmp:b399efc:eval:centerout-sisu0.5-zero-perturbation'
    );
  });
});

