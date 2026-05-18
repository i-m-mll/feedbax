import { beforeEach, describe, expect, it } from 'vitest';
import { useAnalysisStore } from '@/stores/analysisStore';
import { buildWorkspaceSnapshot, getStageByKind, useWorkspaceStore } from '@/stores/workspaceStore';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { TaskSpec, TrainingSpec } from '@/types/training';

const graph: GraphSpec = {
  nodes: {},
  wires: [],
  input_ports: [],
  output_ports: [],
  input_bindings: {},
  output_bindings: {},
  metadata: {
    name: 'Analysis store test',
    created_at: '2026-05-18T00:00:00Z',
    updated_at: '2026-05-18T00:00:00Z',
    version: '1.0.0',
  },
};

const uiState: GraphUIState = {
  viewport: { x: 0, y: 0, zoom: 1 },
  node_states: {},
};

const trainingSpec: TrainingSpec = {
  optimizer: { type: 'adam', params: {} },
  loss: { type: 'Composite', label: 'loss', weight: 1, children: {} },
  n_batches: 1,
  batch_size: 1,
};

const taskSpec: TaskSpec = {
  type: 'ReachingTask',
  params: {},
};

beforeEach(() => {
  const workspace = buildWorkspaceSnapshot({
    workspace: null,
    graph,
    uiState,
    trainingSpec,
    taskSpec,
    analysisSnapshot: null,
    projectName: 'Analysis store test',
  });
  useWorkspaceStore.setState({
    workspace,
    lastTrainingExecutionPreparation: null,
    lastTrainingLocalRunResult: null,
    lastPipelineMaterializationResult: null,
  });
  useAnalysisStore.getState().resetAnalysis();
});

describe('useAnalysisStore stage ownership', () => {
  it('mirrors eval selection and page params into the analysis stage spec', () => {
    useAnalysisStore.getState().addPage('Endpoint figures');
    useAnalysisStore.getState().setEvalParams({ perturbation_type: 'curl_field' });
    useAnalysisStore.getState().setEvalRunId('ev-stage-owned');

    const workspace = useWorkspaceStore.getState().workspace;
    const analysisStage = getStageByKind(workspace, 'analysis')!;
    const analysisScenario = workspace?.scenarios[analysisStage.scenario_id!];
    const analysisSpec = analysisScenario?.analysis_spec as Record<string, unknown>;

    expect(analysisStage.input_collections[0].item_refs[0]).toMatchObject({
      id: 'ev-stage-owned',
      role: 'evaluation_run',
    });
    expect(analysisStage.selection_spec.eval_run_ids).toEqual(['ev-stage-owned']);
    expect(analysisSpec.input_collections).toEqual(analysisStage.input_collections);
    expect(analysisSpec.eval_run_id).toBe('ev-stage-owned');
    expect(analysisSpec.eval_params).toEqual({ perturbation_type: 'curl_field' });
    expect(analysisSpec.pages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'Endpoint figures',
          eval_run_id: 'ev-stage-owned',
        }),
      ])
    );
  });
});
