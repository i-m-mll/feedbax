import { beforeEach, describe, expect, it, vi } from 'vitest';
import { deleteAnalysisNodeWithConfirmation } from '@/components/analysis/analysisDeletion';
import { useAnalysisStore } from '@/stores/analysisStore';
import { buildWorkspaceSnapshot, getStageByKind, useWorkspaceStore } from '@/stores/workspaceStore';
import type { AnalysisClassDef } from '@/types/analysis';
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
  it('preserves the authored evaluation-states policy in the persisted analysis draft', () => {
    const workspace = useWorkspaceStore.getState().workspace!;
    const analysisStage = getStageByKind(workspace, 'analysis')!;
    workspace.scenarios[analysisStage.scenario_id!].analysis_spec = {
      evaluation_states_policy: 'require_durable',
    };

    useAnalysisStore.getState().addPage('Durable states');

    const analysisSpec = workspace.scenarios[analysisStage.scenario_id!]
      .analysis_spec as Record<string, unknown>;
    expect(analysisSpec.evaluation_states_policy).toBe('require_durable');
  });

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

  it('serializes data-source wires as analysis input requirements', () => {
    const analysisClass: AnalysisClassDef = {
      name: 'ActivityPlot',
      description: 'Plot activity',
      category: 'Figures',
      inputPorts: ['series'],
      outputPorts: [],
      defaultParams: {},
      icon: 'LineChart',
    };

    useAnalysisStore.getState().addPage('Activity');
    useAnalysisStore.getState().addAnalysisNode(analysisClass, { x: 240, y: 0 });
    const targetNode = useAnalysisStore
      .getState()
      .nodes.find((node) => node.type === 'analysis')!;

    useAnalysisStore.getState().connectNodes({
      source: '__data_source__',
      sourceHandle: 'path:states.net.hidden',
      target: targetNode.id,
      targetHandle: 'series',
    });

    const wire = useAnalysisStore.getState().graphSpec?.wires[0];
    expect(wire?.inputRequirement).toMatchObject({
      id: `analysis-input:${wire?.id}`,
      selector: 'path:states.net.hidden',
      retention: { mode: 'trajectory' },
      consumer: {
        node_id: targetNode.id,
        input_port: 'series',
        analysis_type: 'ActivityPlot',
      },
    });

    const workspace = useWorkspaceStore.getState().workspace;
    const analysisStage = getStageByKind(workspace, 'analysis')!;
    const analysisScenario = workspace?.scenarios[analysisStage.scenario_id!];
    const analysisSpec = analysisScenario?.analysis_spec as Record<string, unknown>;

    expect(analysisSpec.input_requirements).toEqual([
      expect.objectContaining({
        selector: 'path:states.net.hidden',
        retention: { mode: 'trajectory' },
      }),
    ]);
    expect(analysisSpec.pages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          input_requirements: [
            expect.objectContaining({
              selector: 'path:states.net.hidden',
            }),
          ],
        }),
      ])
    );
  });

  it('persists node deletion and removes incident wires across snapshot restore', () => {
    const analysisClass: AnalysisClassDef = {
      name: 'ActivityPlot',
      description: 'Plot activity',
      category: 'Figures',
      inputPorts: ['series'],
      outputPorts: ['figure'],
      defaultParams: {},
      icon: 'LineChart',
    };

    const store = useAnalysisStore.getState();
    store.addPage('Deletion');
    store.addAnalysisNode(analysisClass, { x: 240, y: 0 });
    store.addAnalysisNode(analysisClass, { x: 480, y: 0 });
    const [firstNode, secondNode] = useAnalysisStore
      .getState()
      .nodes.filter((node) => node.type === 'analysis');

    useAnalysisStore.getState().connectNodes({
      source: '__data_source__',
      sourceHandle: 'path:states.net.hidden',
      target: firstNode.id,
      targetHandle: 'series',
    });
    useAnalysisStore.getState().connectNodes({
      source: firstNode.id,
      sourceHandle: 'figure',
      target: secondNode.id,
      targetHandle: 'series',
    });
    useAnalysisStore.getState().connectNodes({
      source: '__data_source__',
      sourceHandle: 'path:outputs.hand_position',
      target: secondNode.id,
      targetHandle: 'series',
    });

    useAnalysisStore.getState().onNodesChange([{ id: firstNode.id, type: 'remove' }]);

    const deletedState = useAnalysisStore.getState();
    expect(deletedState.graphSpec?.nodes[firstNode.id]).toBeUndefined();
    expect(deletedState.graphSpec?.nodes[secondNode.id]).toBeDefined();
    expect(deletedState.graphSpec?.wires).toEqual([
      expect.objectContaining({ sourceId: '__data_source__', targetId: secondNode.id }),
    ]);

    const snapshot = deletedState.captureSnapshot();
    deletedState.resetAnalysis();
    useAnalysisStore.getState().restoreSnapshot(snapshot);

    const restoredState = useAnalysisStore.getState();
    expect(restoredState.graphSpec?.nodes[firstNode.id]).toBeUndefined();
    expect(restoredState.nodes.some((node) => node.id === firstNode.id)).toBe(false);
    expect(restoredState.graphSpec?.wires).toEqual([
      expect.objectContaining({ sourceId: '__data_source__', targetId: secondNode.id }),
    ]);

    const workspace = useWorkspaceStore.getState().workspace;
    const analysisStage = getStageByKind(workspace, 'analysis')!;
    const analysisSpec = workspace?.scenarios[analysisStage.scenario_id!]
      .analysis_spec as Record<string, unknown>;
    const persistedPage = (analysisSpec.pages as Array<Record<string, unknown>>)[0];
    const persistedGraph = persistedPage.graph_spec as {
      nodes: Record<string, unknown>;
      wires: unknown[];
    };
    expect(persistedGraph.nodes[firstNode.id]).toBeUndefined();
    expect(persistedGraph.wires).toEqual([
      expect.objectContaining({ sourceId: '__data_source__', targetId: secondNode.id }),
    ]);
  });

  it('disables render-only deletion for the data source, transforms, and edges', () => {
    const analysisClass: AnalysisClassDef = {
      name: 'ActivityPlot',
      description: 'Plot activity',
      category: 'Figures',
      inputPorts: ['series'],
      outputPorts: ['figure'],
      defaultParams: {},
      icon: 'LineChart',
    };

    useAnalysisStore.getState().addPage('Protected elements');
    useAnalysisStore.getState().addAnalysisNode(analysisClass, { x: 240, y: 0 });
    const node = useAnalysisStore.getState().nodes.find((item) => item.type === 'analysis')!;
    useAnalysisStore.getState().connectNodes({
      source: '__data_source__',
      sourceHandle: 'path:states.net.hidden',
      target: node.id,
      targetHandle: 'series',
    });
    const wireId = useAnalysisStore.getState().graphSpec!.wires[0].id;
    useAnalysisStore.getState().addTransformToEdge(wireId, 'Standardize');
    const transformId = `transform_${wireId}`;
    const edgeId = `${wireId}__to_transform`;

    useAnalysisStore.getState().onNodesChange([
      { id: '__data_source__', type: 'remove' },
      { id: transformId, type: 'remove' },
    ]);
    useAnalysisStore.getState().onEdgesChange([{ id: edgeId, type: 'remove' }]);

    const state = useAnalysisStore.getState();
    expect(state.nodes.some((item) => item.id === '__data_source__')).toBe(true);
    expect(state.nodes.some((item) => item.id === transformId)).toBe(true);
    expect(state.edges.some((edge) => edge.id === edgeId)).toBe(true);
    expect(state.graphSpec?.wires[0].transform?.id).toBe(transformId);
  });

  it('requires confirmation and honors cancellation for durable deletion', () => {
    const analysisClass: AnalysisClassDef = {
      name: 'ActivityPlot',
      description: 'Plot activity',
      category: 'Figures',
      inputPorts: ['series'],
      outputPorts: [],
      defaultParams: {},
      icon: 'LineChart',
    };

    useAnalysisStore.getState().addPage('Confirmed deletion');
    useAnalysisStore.getState().addAnalysisNode(analysisClass, { x: 240, y: 0 });
    const node = useAnalysisStore.getState().nodes.find((item) => item.type === 'analysis')!;
    const cancel = vi.fn(() => false);
    const confirm = vi.fn(() => true);

    expect(deleteAnalysisNodeWithConfirmation(node.id, cancel)).toBe(false);
    expect(useAnalysisStore.getState().graphSpec?.nodes[node.id]).toBeDefined();

    expect(deleteAnalysisNodeWithConfirmation(node.id, confirm)).toBe(true);

    expect(useAnalysisStore.getState().graphSpec?.nodes[node.id]).toBeUndefined();
    expect(useAnalysisStore.getState().captureSnapshot().pages[0].graphSpec.nodes[node.id])
      .toBeUndefined();
    expect(cancel).toHaveBeenCalledWith(expect.stringMatching(/saved immediately/i));
    expect(confirm).toHaveBeenCalledWith(expect.stringMatching(/connected wires/i));
  });

  it('ignores volatile node position changes', () => {
    const analysisClass: AnalysisClassDef = {
      name: 'ActivityPlot',
      description: 'Plot activity',
      category: 'Figures',
      inputPorts: [],
      outputPorts: [],
      defaultParams: {},
      icon: 'LineChart',
    };

    useAnalysisStore.getState().addPage('Fixed layout');
    useAnalysisStore.getState().addAnalysisNode(analysisClass, { x: 240, y: 80 });
    const node = useAnalysisStore.getState().nodes.find((item) => item.type === 'analysis')!;

    useAnalysisStore.getState().onNodesChange([
      { id: node.id, type: 'position', position: { x: 900, y: 700 }, dragging: true },
    ]);

    expect(useAnalysisStore.getState().nodes.find((item) => item.id === node.id)?.position)
      .toEqual({ x: 240, y: 80 });
  });
});
