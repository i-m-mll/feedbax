import { beforeEach, describe, expect, it } from 'vitest';
import { useRunStore } from '@/stores/runStore';
import { buildWorkspaceSnapshot, getStageByKind, useWorkspaceStore } from '@/stores/workspaceStore';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { EvalRun, TrainingRun } from '@/types/runs';
import type { TaskSpec, TrainingSpec } from '@/types/training';

const graph: GraphSpec = {
  nodes: {},
  wires: [],
  input_ports: [],
  output_ports: [],
  input_bindings: {},
  output_bindings: {},
  metadata: {
    name: 'Run store test',
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
  optimizer: { type: 'adam', params: { learning_rate: 0.001 } },
  loss: { type: 'Composite', label: 'loss', weight: 1, children: {} },
  n_batches: 10,
  batch_size: 4,
};

const taskSpec: TaskSpec = {
  type: 'ReachingTask',
  params: {},
};

const trainingRun: TrainingRun = {
  id: 'tr-stage-owned',
  name: 'Stage owned training run',
  createdAt: '2026-05-18T12:00:00Z',
  status: 'completed',
  hyperparams: { lr: 0.001 },
};

const evalRun: EvalRun = {
  id: 'ev-stage-owned',
  trainingRunId: trainingRun.id,
  name: 'Stage owned evaluation',
  createdAt: '2026-05-18T12:10:00Z',
  status: 'completed',
  description: 'Stage input selection',
};

beforeEach(() => {
  const workspace = buildWorkspaceSnapshot({
    workspace: null,
    graph,
    uiState,
    trainingSpec,
    taskSpec,
    analysisSnapshot: null,
    projectName: 'Run store test',
  });
  useWorkspaceStore.setState({
    workspace,
    lastTrainingExecutionPreparation: null,
    lastTrainingLocalRunResult: null,
    lastPipelineMaterializationResult: null,
  });
  useRunStore.setState({
    trainingRuns: [],
    evalRuns: [],
    selectedTrainingRunId: null,
    selectedEvalRunId: null,
    loading: false,
  });
});

describe('useRunStore stage collection ownership', () => {
  it('indexes training runs on the train stage and selects eval-stage inputs', async () => {
    useRunStore.getState().addTrainingRun(trainingRun);
    await useRunStore.getState().selectTrainingRun(trainingRun.id);

    const workspace = useWorkspaceStore.getState().workspace;
    const trainStage = getStageByKind(workspace, 'train');
    const evalStage = getStageByKind(workspace, 'eval');

    expect(trainStage?.output_collections[0].item_refs[0]).toMatchObject({
      id: trainingRun.id,
      role: 'training_run',
      provider: 'feedbax',
    });
    expect(evalStage?.input_collections[0].item_refs[0]).toMatchObject({
      id: trainingRun.id,
      role: 'training_run',
    });
    expect(evalStage?.selection_spec.training_run_ids).toEqual([trainingRun.id]);
  });

  it('indexes eval runs on the eval stage and selects analysis-stage inputs', () => {
    useRunStore.setState({ trainingRuns: [trainingRun] });
    useRunStore.getState().addEvalRun(evalRun);

    const workspace = useWorkspaceStore.getState().workspace;
    const evalStage = getStageByKind(workspace, 'eval');
    const analysisStage = getStageByKind(workspace, 'analysis');

    expect(evalStage?.output_collections[0].item_refs[0]).toMatchObject({
      id: evalRun.id,
      role: 'evaluation_run',
      provider: 'feedbax',
    });
    expect(analysisStage?.input_collections[0].item_refs[0]).toMatchObject({
      id: evalRun.id,
      role: 'evaluation_run',
    });
    expect(analysisStage?.selection_spec.eval_run_ids).toEqual([evalRun.id]);
    expect(analysisStage?.selection_spec.input_collection_ids).toEqual([
      'collection:selected-evaluation-runs',
    ]);
  });

  it('hydrates run selector state from existing workspace collections', () => {
    useRunStore.getState().addTrainingRun(trainingRun);
    useRunStore.getState().addEvalRun(evalRun);

    const workspace = useWorkspaceStore.getState().workspace;
    useRunStore.setState({
      trainingRuns: [],
      evalRuns: [],
      selectedTrainingRunId: null,
      selectedEvalRunId: null,
    });
    useRunStore.getState().hydrateFromWorkspace(workspace);

    expect(useRunStore.getState().trainingRuns[0]).toMatchObject({
      id: trainingRun.id,
      name: trainingRun.name,
    });
    expect(useRunStore.getState().evalRuns[0]).toMatchObject({
      id: evalRun.id,
      trainingRunId: trainingRun.id,
    });
    expect(useRunStore.getState().selectedTrainingRunId).toBe(trainingRun.id);
    expect(useRunStore.getState().selectedEvalRunId).toBe(evalRun.id);
  });
});
