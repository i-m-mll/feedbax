import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fetchEvalRuns, fetchTrainingRuns } from '@/api/runAPI';
import { useRunStore } from '@/stores/runStore';
import { useSelectionContextStore } from '@/stores/selectionContextStore';
import { buildWorkspaceSnapshot, getStageByKind, useWorkspaceStore } from '@/stores/workspaceStore';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { EvalRun, TrainingRun } from '@/types/runs';
import type { TaskSpec, TrainingSpec } from '@/types/training';
import { trainingRunSummaries } from '@/utils/pipelineCollections';

vi.mock('@/api/runAPI', () => ({
  fetchTrainingRuns: vi.fn(),
  fetchEvalRuns: vi.fn(),
}));

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

const pendingTrainingRun: TrainingRun = {
  id: 'feedbax-training-run:pending',
  name: 'Pending manifest run',
  createdAt: '2026-05-18T12:20:00Z',
  status: 'pending',
  hyperparams: {
    n_batches: 25,
    batch_size: 8,
    ramp_duration_steps: 80,
  },
  metrics: { final_validation_loss: 0.25 },
  uri: '/tmp/feedbax_runs/manifests/training_runs/pending.json',
  stageId: 'stage:train',
  scenarioId: 'scenario:train',
  planned: true,
  checkpointAvailable: false,
  sourceIssue: '9aa8ff2',
  provenanceId: 'feedbax-training-run:pending',
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
    trainingError: null,
    evalError: null,
  });
  useSelectionContextStore.getState().reset();
  vi.mocked(fetchTrainingRuns).mockResolvedValue([]);
  vi.mocked(fetchEvalRuns).mockResolvedValue([]);
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
    expect(useSelectionContextStore.getState().context).toMatchObject({
      stage: evalStage?.id,
      collection: 'collection:selected-training-runs',
      selectedIds: [trainingRun.id],
      focusedId: trainingRun.id,
    });
  });

  it('clears training selection through the unified selection context', async () => {
    useRunStore.getState().addTrainingRun(trainingRun);
    await useRunStore.getState().selectTrainingRun(trainingRun.id);

    await useRunStore.getState().selectTrainingRun(null);

    const workspace = useWorkspaceStore.getState().workspace;
    const evalStage = getStageByKind(workspace, 'eval');
    expect(useRunStore.getState().selectedTrainingRunId).toBeNull();
    expect(evalStage?.selection_spec.training_run_ids).toEqual([]);
    expect(useSelectionContextStore.getState().context).toMatchObject({
      stage: evalStage?.id,
      collection: 'collection:selected-training-runs',
      selectedIds: [],
      focusedId: null,
    });
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
    expect(useSelectionContextStore.getState().context).toMatchObject({
      stage: analysisStage?.id,
      collection: 'collection:selected-evaluation-runs',
      selectedIds: [evalRun.id],
      focusedId: evalRun.id,
    });
  });

  it('treats runStore selected ids as compatibility mirrors of SelectionContext', () => {
    const workspace = useWorkspaceStore.getState().workspace;
    const evalStage = getStageByKind(workspace, 'eval');
    const analysisStage = getStageByKind(workspace, 'analysis');
    useRunStore.setState({
      selectedTrainingRunId: 'stale-training',
      selectedEvalRunId: 'stale-eval',
    });

    useSelectionContextStore.getState().setContext({
      stage: evalStage?.id ?? null,
      collection: 'collection:selected-training-runs',
      selectedIds: ['tr-a', 'tr-b'],
      focusedId: 'tr-b',
    });

    expect(useRunStore.getState()).toMatchObject({
      selectedTrainingRunId: 'tr-b',
      selectedEvalRunId: null,
    });

    useSelectionContextStore.getState().setContext({
      stage: analysisStage?.id ?? null,
      collection: 'collection:selected-evaluation-runs',
      selectedIds: ['ev-a'],
      focusedId: null,
    });

    expect(useRunStore.getState().selectedEvalRunId).toBe('ev-a');

    useSelectionContextStore.getState().reset();
    expect(useRunStore.getState()).toMatchObject({
      selectedTrainingRunId: null,
      selectedEvalRunId: null,
    });
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

  it('clears stale selection context when hydrating a workspace with no run selection', () => {
    const workspace = useWorkspaceStore.getState().workspace;
    useRunStore.setState({
      selectedTrainingRunId: 'stale-training',
      selectedEvalRunId: 'stale-eval',
    });
    useSelectionContextStore.getState().setContext({
      stage: 'stage:old',
      collection: 'collection:selected-training-runs',
      selectedIds: ['stale-training'],
      focusedId: 'stale-training',
    });

    useRunStore.getState().hydrateFromWorkspace(workspace);

    expect(useRunStore.getState()).toMatchObject({
      selectedTrainingRunId: null,
      selectedEvalRunId: null,
    });
    expect(useSelectionContextStore.getState().context).toMatchObject({
      stage: null,
      collection: null,
      selectedIds: [],
      focusedId: null,
    });
  });

  it('preserves typed pending training-run fields through workspace refs and hydration', () => {
    useRunStore.getState().addTrainingRun(pendingTrainingRun);

    const workspace = useWorkspaceStore.getState().workspace;
    const trainStage = getStageByKind(workspace, 'train');
    const ref = trainStage?.output_collections[0].item_refs[0];
    expect(ref).toMatchObject({
      kind: 'TrainingRunManifest',
      id: pendingTrainingRun.id,
      uri: pendingTrainingRun.uri,
      metadata: expect.objectContaining({
        status: 'pending',
        planned: true,
        stage_id: 'stage:train',
        scenario_id: 'scenario:train',
        source_issue: '9aa8ff2',
        provenance_id: pendingTrainingRun.id,
        final_validation_loss: 0.25,
      }),
    });

    const [summary] = trainingRunSummaries(trainStage);
    expect(summary).toMatchObject({
      id: pendingTrainingRun.id,
      status: 'pending',
      finalValidationLoss: 0.25,
      batchSize: 8,
      rampDurationSteps: 80,
      checkpointAvailable: false,
      sourceIssue: '9aa8ff2',
      provenanceId: pendingTrainingRun.id,
      uri: pendingTrainingRun.uri,
    });

    useRunStore.setState({ trainingRuns: [], selectedTrainingRunId: null });
    useRunStore.getState().hydrateFromWorkspace(workspace);

    expect(useRunStore.getState().trainingRuns[0]).toMatchObject({
      id: pendingTrainingRun.id,
      status: 'pending',
      planned: true,
      stageId: 'stage:train',
      scenarioId: 'scenario:train',
      checkpointAvailable: false,
      sourceIssue: '9aa8ff2',
      provenanceId: pendingTrainingRun.id,
      uri: pendingTrainingRun.uri,
    });
  });

  it('preserves cancelled training-run status through workspace hydration', () => {
    const cancelledRun: TrainingRun = {
      ...pendingTrainingRun,
      id: 'feedbax-training-run:cancelled',
      name: 'Cancelled manifest run',
      status: 'cancelled',
      planned: true,
    };
    useRunStore.getState().addTrainingRun(cancelledRun);

    const workspace = useWorkspaceStore.getState().workspace;
    useRunStore.setState({ trainingRuns: [], selectedTrainingRunId: null });
    useRunStore.getState().hydrateFromWorkspace(workspace);

    expect(useRunStore.getState().trainingRuns[0]).toMatchObject({
      id: cancelledRun.id,
      status: 'cancelled',
      planned: true,
    });
  });

  it('keeps training-run load failures visible without fabricating rows', async () => {
    vi.mocked(fetchTrainingRuns).mockRejectedValue(new Error('backend offline'));

    await useRunStore.getState().loadTrainingRuns();

    expect(useRunStore.getState().trainingRuns).toEqual([]);
    expect(useRunStore.getState().trainingError).toBe('backend offline');
    expect(useRunStore.getState().loading).toBe(false);
  });

  it('keeps eval-run load failures visible without fabricating rows', async () => {
    useRunStore.setState({ trainingRuns: [trainingRun] });
    vi.mocked(fetchEvalRuns).mockRejectedValue(new Error('eval backend offline'));

    await useRunStore.getState().selectTrainingRun(trainingRun.id);

    expect(useRunStore.getState().evalRuns).toEqual([]);
    expect(useRunStore.getState().evalError).toBe('eval backend offline');
    expect(useRunStore.getState().selectedTrainingRunId).toBe(trainingRun.id);
  });
});
