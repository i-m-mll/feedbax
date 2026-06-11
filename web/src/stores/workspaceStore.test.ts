import { beforeEach, describe, expect, it } from 'vitest';
import {
  buildWorkspaceSnapshot,
  getActiveScenario,
  getActiveStage,
  getTopPaneState,
  getTrainingScenario,
  objectiveSpecFromLossSpec,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { graphNodeEntityId } from '@/features/scenario/entities';
import { addObjectiveTerm, createObjectiveTerm } from '@/features/scenario/objectives';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { TrainingSpec, TaskSpec } from '@/types/training';
import type { AnalysisSnapshot } from '@/types/analysis';
import type {
  StudioObjectiveSpec,
  StudioTopPaneState,
  StudioWorkspaceSpec,
} from '@/types/workspace';

const graph: GraphSpec = {
  nodes: {},
  wires: [],
  input_ports: [],
  output_ports: [],
  input_bindings: {},
  output_bindings: {},
  metadata: {
    name: 'Workspace test',
    created_at: '2026-05-17T00:00:00Z',
    updated_at: '2026-05-17T00:00:00Z',
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
  n_batches: 100,
  batch_size: 32,
};

const taskSpec: TaskSpec = {
  type: 'ReachingTask',
  params: { target_radius: 0.02 },
};

beforeEach(() => {
  useWorkspaceStore.setState({
    workspace: null,
    lastTrainingExecutionPreparation: null,
    lastTrainingLocalRunResult: null,
    lastPipelineMaterializationResult: null,
  });
});

describe('buildWorkspaceSnapshot', () => {
  it('creates train/eval/analysis/report anchors from current Studio state', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });

    expect(workspace.schema_version).toBe('feedbax.studio.workspace.v1');
    expect(workspace.active_stage_id).toBe('stage:train');
    expect(workspace.stages.map((stage) => stage.kind)).toEqual([
      'train',
      'eval',
      'analysis',
      'report',
    ]);
    const trainStage = workspace.stages.find((stage) => stage.kind === 'train')!;
    const scenario = workspace.scenarios[trainStage.scenario_id!];
    expect(scenario.training_spec).toEqual(trainingSpec);
    expect(scenario.task_spec).toEqual(taskSpec);
    expect(scenario.task_binding_spec).toMatchObject({
      schema_version: 'feedbax.studio.task_bindings.v2',
      exposed_data: [
        { id: 'inputs', bindable: true },
        { id: 'targets', bindable: false },
        { id: 'inits', bindable: false },
        { id: 'intervene', bindable: false },
      ],
      bindings: [],
    });
    expect(scenario.objective_spec).toEqual(objectiveSpecFromLossSpec(trainingSpec.loss));
    expect(scenario.graph).toEqual(graph);
  });

  it('persists nested graph edits and analysis pages in one workspace snapshot', () => {
    const nestedGraph: GraphSpec = {
      ...graph,
      nodes: {
        network: {
          type: 'Network',
          params: {},
          input_ports: ['input'],
          output_ports: ['output'],
        },
      },
      subgraphs: {
        network: {
          nodes: {
            gain: {
              type: 'Gain',
              params: { gain: 2 },
              input_ports: ['input'],
              output_ports: ['output'],
            },
          },
          wires: [],
          input_ports: ['input'],
          output_ports: ['output'],
          input_bindings: { input: ['gain', 'input'] },
          output_bindings: { output: ['gain', 'output'] },
        },
      },
    };
    const nestedUiState: GraphUIState = {
      viewport: { x: 0, y: 0, zoom: 1 },
      node_states: {
        network: { position: { x: 100, y: 100 }, collapsed: false, selected: false },
      },
      subgraph_states: {
        network: {
          viewport: { x: 20, y: 40, zoom: 0.8 },
          node_states: {
            gain: { position: { x: 480, y: 120 }, collapsed: false, selected: false },
          },
        },
      },
    };
    const analysisSnapshot: AnalysisSnapshot = {
      pages: [
        {
          id: 'analysis:page:gain',
          name: 'Gain response',
          graphSpec: {
            dataSourceId: '__data_source__',
            nodes: {
              plot: {
                id: 'plot',
                type: 'LinePlot',
                label: 'Line plot',
                category: 'Figures',
                inputPorts: ['series'],
                outputPorts: [],
                params: { color: 'blue' },
                role: 'analysis',
              },
            },
            wires: [],
          },
          inputRequirements: [],
          evalParams: { perturbation: 'gain' },
          viewport: { x: 10, y: 15, zoom: 0.9 },
          evalRunId: 'eval:gain',
          expandedFieldPaths: ['states.network.hidden'],
        },
      ],
      activePageId: 'analysis:page:gain',
    };

    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph: nestedGraph,
      uiState: nestedUiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot,
      projectName: 'Workspace test',
    });

    const trainStage = workspace.stages.find((stage) => stage.kind === 'train')!;
    const trainScenario = workspace.scenarios[trainStage.scenario_id!];
    expect(trainScenario.graph?.subgraphs?.network.nodes.gain).toMatchObject({
      type: 'Gain',
      params: { gain: 2 },
    });
    expect(
      trainScenario.graph_ui_state?.subgraph_states?.network?.node_states.gain
    ).toMatchObject({
      position: { x: 480, y: 120 },
    });

    const analysisStage = workspace.stages.find((stage) => stage.kind === 'analysis')!;
    const analysisSpec = workspace.scenarios[analysisStage.scenario_id!]
      .analysis_spec as Record<string, any>;
    expect(analysisSpec.active_page_id).toBe('analysis:page:gain');
    expect(analysisSpec.pages[0]).toMatchObject({
      id: 'analysis:page:gain',
      name: 'Gain response',
      eval_run_id: 'eval:gain',
      expanded_field_paths: ['states.network.hidden'],
    });
    expect(analysisSpec.pages[0].graph_spec.nodes.plot.params).toEqual({ color: 'blue' });
  });

  it('rejects workspace snapshots when graph UI state references a missing node', () => {
    expect(() =>
      buildWorkspaceSnapshot({
        workspace: null,
        graph,
        uiState: {
          ...uiState,
          node_states: {
            ghost: { position: { x: 0, y: 0 }, collapsed: false, selected: false },
          },
        },
        trainingSpec,
        taskSpec,
        analysisSnapshot: null,
        projectName: 'Workspace test',
      })
    ).toThrow('UI state references missing node "ghost"');
  });

  it('seeds the task input binding when the graph exposes network.input', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph: {
        ...graph,
        nodes: {
          network: {
            type: 'Network',
            params: {},
            input_ports: ['input'],
            output_ports: ['output'],
          },
        },
      },
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });
    const trainStage = workspace.stages.find((stage) => stage.kind === 'train')!;
    const scenario = workspace.scenarios[trainStage.scenario_id!];

    expect(scenario.graph?.wires).toEqual([]);
    expect(scenario.task_binding_spec?.bindings).toEqual([
      {
        id: 'task:inputs->network:input',
        source_data_id: 'inputs',
        target_node_id: 'network',
        target_port: 'input',
        role: 'model_input',
        metadata: {},
      },
    ]);
  });

  it('preserves workspace-owned drafts and future metadata while refreshing graph state', () => {
    const existing = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });
    const trainStage = existing.stages.find((stage) => stage.kind === 'train')!;
    const workspaceOwnedTrainingSpec = { ...trainingSpec, n_batches: 333 };
    const workspaceOwnedTaskSpec = {
      ...taskSpec,
      params: { ...taskSpec.params, target_radius: 0.04 },
    };
    const workspaceOwnedTaskBindingSpec = {
      schema_version: 'feedbax.studio.task_bindings.v2',
      exposed_data: [],
      bindings: [
        {
          id: 'task:inputs->custom:input',
          source_data_id: 'inputs',
          target_node_id: 'custom',
          target_port: 'input',
          role: 'model_input',
          metadata: {},
        },
      ],
      metadata: { authored_in: 'workspace_store' },
    };
    const withFutureStage: StudioWorkspaceSpec = {
      ...existing,
      scenarios: {
        ...existing.scenarios,
        [trainStage.scenario_id!]: {
          ...existing.scenarios[trainStage.scenario_id!],
          training_spec: workspaceOwnedTrainingSpec,
          task_spec: workspaceOwnedTaskSpec,
          task_binding_spec: workspaceOwnedTaskBindingSpec,
          metadata: { authored_in: 'workspace_store' },
        },
      },
      stages: [
        ...existing.stages,
        {
          id: 'stage:future-objective-authoring',
          kind: 'protocol',
          label: 'Future objective authoring',
          status: 'draft',
          scenario_id: null,
          input_collections: [],
          output_collections: [],
          manifest_refs: [],
          execution_spec: null,
          selection_spec: {},
          validation: {
            valid: null,
            checked_at: null,
            errors: [],
            warnings: [],
            metadata: {},
          },
          ui_state: {},
          metadata: { later: { keep: true } },
        },
      ],
    };

    const refreshed = buildWorkspaceSnapshot({
      workspace: withFutureStage,
      graph: { ...graph, output_ports: ['effector'] },
      uiState,
      trainingSpec: { ...trainingSpec, n_batches: 200 },
      taskSpec: { ...taskSpec, params: { target_radius: 0.01 } },
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });

    const futureStage = refreshed.stages.find(
      (stage) => stage.id === 'stage:future-objective-authoring'
    );
    expect(futureStage?.metadata).toEqual({ later: { keep: true } });

    const refreshedTrainStage = refreshed.stages.find((stage) => stage.kind === 'train')!;
    const scenario = refreshed.scenarios[refreshedTrainStage.scenario_id!];
    expect(scenario.training_spec?.n_batches).toBe(333);
    expect(scenario.task_spec?.params.target_radius).toBe(0.04);
    expect(scenario.task_binding_spec).toMatchObject({
      ...workspaceOwnedTaskBindingSpec,
      exposed_data: [
        { id: 'inputs', bindable: true },
        { id: 'targets', bindable: false },
        { id: 'inits', bindable: false },
        { id: 'intervene', bindable: false },
      ],
    });
    expect(scenario.graph?.output_ports).toEqual(['effector']);
  });

  it('switches active stages and exposes active stage/scenario selectors', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });

    useWorkspaceStore.getState().setWorkspace(workspace);
    useWorkspaceStore.getState().setActiveStageByKind('analysis');

    const state = useWorkspaceStore.getState();
    expect(getActiveStage(state.workspace)?.kind).toBe('analysis');
    expect(getActiveScenario(state.workspace)?.id).toBe('scenario:analysis');
    expect(state.workspace?.metadata.dirty).toBe(true);

    useWorkspaceStore.getState().setActiveStageByKind('train');
    expect(getTrainingScenario(useWorkspaceStore.getState().workspace)?.id).toBe(
      'scenario:train'
    );
  });

  it('preserves active stage selection through workspace snapshot refreshes', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });

    useWorkspaceStore.getState().setWorkspace(workspace);
    useWorkspaceStore.getState().setActiveStageByKind('report');
    const selected = useWorkspaceStore.getState().workspace!;

    const refreshed = buildWorkspaceSnapshot({
      workspace: selected,
      graph,
      uiState,
      trainingSpec: { ...trainingSpec, n_batches: 999 },
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });

    expect(getActiveStage(refreshed)?.kind).toBe('report');
    expect(refreshed.ui_state.active_stage_id).toBe(getActiveStage(refreshed)?.id);
  });

  it('stores top-pane projection and scenario entity selection in workspace UI state', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });

    useWorkspaceStore.getState().setWorkspace(workspace);
    useWorkspaceStore.getState().setTopPaneProjection('objectives');
    useWorkspaceStore.getState().selectTopPaneEntity(graphNodeEntityId('task'));
    useWorkspaceStore.getState().hoverTopPaneEntity(graphNodeEntityId('mechanics'));

    const topPane = getTopPaneState(useWorkspaceStore.getState().workspace);
    expect(topPane.active_projection).toBe('objectives');
    expect(topPane.selected_entity_id).toBe(graphNodeEntityId('task'));
    expect(topPane.hovered_entity_id).toBe(graphNodeEntityId('mechanics'));
    expect(useWorkspaceStore.getState().workspace?.ui_state.top_pane).toMatchObject(topPane);
  });

  it('normalizes legacy graph top-pane projection to model', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });

    expect(
      getTopPaneState({
        ...workspace,
        ui_state: {
          ...workspace.ui_state,
          top_pane: { active_projection: 'graph' } as unknown as StudioTopPaneState,
        },
      }).active_projection
    ).toBe('model');
  });

  it('updates train scenario drafts as the primary task/training/objective owner', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });

    useWorkspaceStore.getState().setWorkspace(workspace);
    useWorkspaceStore.getState().updateActiveScenarioTrainingSpec({
      ...trainingSpec,
      loss: {
        ...trainingSpec.loss,
        children: {
          endpoint: {
            type: 'TargetStateLoss',
            label: 'Endpoint',
            weight: 2,
            selector: 'port:effector.position',
            norm: 'l2',
            time_agg: { mode: 'final' },
          },
        },
      },
      n_batches: 500,
    });
    useWorkspaceStore.getState().updateActiveScenarioTaskSpec({
      ...taskSpec,
      params: { ...taskSpec.params, n_targets: 16 },
    });
    useWorkspaceStore.getState().updateActiveScenarioTaskBindingSpec({
      schema_version: 'feedbax.studio.task_bindings.v2',
      exposed_data: [],
      bindings: [],
      metadata: { edited: true },
    });

    const scenario = getTrainingScenario(useWorkspaceStore.getState().workspace)!;
    expect(scenario.training_spec?.n_batches).toBe(500);
    expect(scenario.task_spec?.params.n_targets).toBe(16);
    expect(scenario.task_binding_spec?.metadata.edited).toBe(true);
    expect(scenario.metadata.dirty).toBe(true);
    expect(scenario.objective_spec?.terms).toHaveLength(1);
    expect(scenario.objective_spec?.terms[0].source_selector).toMatchObject({
      namespace: 'graph_port',
      target_id: 'effector',
      path: 'position',
    });
  });

  it('normalizes task binding specs when restoring workspace snapshots', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph: {
        ...graph,
        nodes: {
          network: {
            type: 'Network',
            params: {},
            input_ports: ['input'],
            output_ports: ['output'],
          },
        },
      },
      uiState,
      trainingSpec,
      taskSpec: {
        type: 'DelayedReaches',
        params: {},
      },
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });
    const trainStage = workspace.stages.find((stage) => stage.kind === 'train')!;
    const scenario = workspace.scenarios[trainStage.scenario_id!];
    scenario.task_binding_spec = {
      schema_version: 'feedbax.studio.task_bindings.v2',
      exposed_data: [
        {
          id: 'inputs',
          label: 'Inputs',
          kind: 'signal',
          role: 'model_input',
          path: 'inputs',
          bindable: true,
          metadata: {},
        },
      ],
      bindings: [
        {
          id: 'task:inputs->network:input',
          source_data_id: 'inputs',
          target_node_id: 'network',
          target_port: 'input',
          role: 'model_input',
          metadata: {},
        },
      ],
      metadata: {},
    };

    useWorkspaceStore.getState().setWorkspace(workspace);

    const restored = getTrainingScenario(useWorkspaceStore.getState().workspace)!;
    expect(restored.task_binding_spec?.exposed_data.map((data) => data.id)).toEqual([
      'target_position',
      'hold',
      'target_on',
      'movement_target',
      'inits',
      'intervene',
    ]);
    expect(restored.task_binding_spec?.bindings).toEqual([]);
  });

  it('normalizes runtime graph component names when restoring workspace snapshots', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph: {
        ...graph,
        nodes: {
          network: {
            type: 'SimpleStagedNetwork',
            params: { input_size: 4, hidden_size: 100, output_size: 2 },
            input_ports: ['target'],
            output_ports: ['output'],
          },
        },
        input_ports: ['target'],
        input_bindings: { target: ['network', 'target'] },
      },
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });

    useWorkspaceStore.getState().setWorkspace(workspace);

    const restored = getTrainingScenario(useWorkspaceStore.getState().workspace)!;
    expect(restored.graph?.nodes.network).toMatchObject({
      type: 'Network',
      params: { out_size: 2 },
      input_ports: ['input', 'feedback'],
    });
    expect(restored.graph?.subgraphs?.network.nodes.input_mux).toMatchObject({
      type: 'Mux',
      input_ports: ['in_0', 'in_1'],
    });
    expect(restored.graph?.subgraphs?.network.input_bindings).toMatchObject({
      input: ['input_mux', 'in_0'],
      feedback: ['input_mux', 'in_1'],
    });
    expect(restored.graph?.subgraphs?.network.nodes.cell).toMatchObject({
      type: 'GRU',
      params: { input_size: 4, hidden_size: 100 },
    });
    expect(restored.graph?.input_ports).toEqual(['input']);
    expect(restored.graph?.input_bindings).toEqual({ input: ['network', 'input'] });
  });

  it('retargets active scenario task bindings when a model node is renamed', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph: {
        ...graph,
        nodes: {
          network: {
            type: 'Network',
            params: {},
            input_ports: ['input'],
            output_ports: ['output'],
          },
        },
      },
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });
    useWorkspaceStore.getState().setWorkspace(workspace);

    useWorkspaceStore
      .getState()
      .retargetActiveScenarioTaskBindingsForNodeRename('network', 'controller');

    const scenario = getTrainingScenario(useWorkspaceStore.getState().workspace)!;
    expect(scenario.task_binding_spec?.bindings).toEqual([
      {
        id: 'task:inputs->controller:input',
        source_data_id: 'inputs',
        target_node_id: 'controller',
        target_port: 'input',
        role: 'model_input',
        metadata: {},
      },
    ]);
    expect(scenario.metadata.updated_reason).toBe('task_binding_target_renamed');
  });

  it('maps known legacy probe losses onto graph-port substates', () => {
    const objectiveSpec = objectiveSpecFromLossSpec({
      type: 'Composite',
      label: 'loss',
      weight: 1,
      children: {
        activity: {
          type: 'TargetStateLoss',
          label: 'Network Activity',
          weight: 0.01,
          selector: 'probe:network_hidden',
          norm: 'squared_l2',
          time_agg: { mode: 'all' },
        },
      },
    });

    expect(objectiveSpec.terms[0].source_selector).toMatchObject({
      namespace: 'state_path',
      compact: 'path:states.net.hidden',
      metadata: {
        legacy_selector: 'probe:network_hidden',
        graph_port_node_id: 'network',
        graph_port_name: 'hidden',
        subpath: 'hidden',
      },
    });
  });

  it('lowers active scenario objective edits back into the training loss spec', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });

    useWorkspaceStore.getState().setWorkspace(workspace);
    const current = getTrainingScenario(useWorkspaceStore.getState().workspace)!
      .objective_spec as StudioObjectiveSpec;
    const term = createObjectiveTerm({
      spec: current,
      label: 'Endpoint',
      sourceSelector: {
        namespace: 'graph_port',
        compact: 'port:mechanics.effector',
        target_id: 'mechanics',
        path: 'effector',
        metadata: {},
      },
    });
    useWorkspaceStore.getState().updateActiveScenarioObjectiveSpec(addObjectiveTerm(current, term));

    const scenario = getTrainingScenario(useWorkspaceStore.getState().workspace)!;
    const objectiveSpec = scenario.objective_spec as StudioObjectiveSpec;
    expect(objectiveSpec.terms.some((item) => item.id === term.id)).toBe(true);
    expect(scenario.training_spec?.loss.children?.[term.id]).toMatchObject({
      label: 'Endpoint',
      selector: 'port:mechanics.effector',
    });
  });

  it('updates stage draft collections without dropping custom stage metadata', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });
    const evalStage = workspace.stages.find((stage) => stage.kind === 'eval')!;
    const collection = {
      id: 'collection:selected-training-runs',
      kind: 'training_runs',
      label: 'Selected training runs',
      source_stage_id: 'stage:train',
      item_refs: [],
      filters: { status: 'completed' },
      facets: {},
      metadata: { user_named: true },
    };

    useWorkspaceStore.getState().setWorkspace({
      ...workspace,
      stages: workspace.stages.map((stage) =>
        stage.id === evalStage.id
          ? { ...stage, metadata: { custom: { keep: true } } }
          : stage
      ),
    });
    useWorkspaceStore
      .getState()
      .updateStageCollections(evalStage.id, { input_collections: [collection] });

    const updatedEvalStage = useWorkspaceStore
      .getState()
      .workspace?.stages.find((stage) => stage.kind === 'eval');
    expect(updatedEvalStage?.input_collections).toEqual([collection]);
    expect(updatedEvalStage?.metadata.custom).toEqual({ keep: true });
    expect(updatedEvalStage?.metadata.dirty).toBe(true);
  });

  it('stores prepared execution plans without dropping workspace state', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });
    const prepared = {
      ...workspace,
      stages: workspace.stages.map((stage) =>
        stage.kind === 'train'
          ? {
              ...stage,
              status: 'ready' as const,
              artifact_refs: [
                {
                  kind: 'ExecutionPlan',
                  id: 'execution-plan:studio-plan',
                  role: 'execution_plan',
                  provider: 'feedbax',
                  uri: '/tmp/feedbax_runs/studio-plan/execution-plan.json',
                  media_type: 'application/json',
                  metadata: {},
                },
              ],
            }
          : stage
      ),
    };

    useWorkspaceStore.getState().setWorkspace(workspace);
    useWorkspaceStore.getState().setTrainingExecutionPreparation({
      workspace: prepared,
      stage_id: 'stage:train',
      scenario_id: 'scenario:train',
      execution_spec: { job_id: 'studio-plan' },
      plan: {
        kind: 'ExecutionPlan',
        schema_version: 'feedbax.execution.v1',
        job_id: 'studio-plan',
        backend: 'local',
        command: 'feedbax-provider validate training training-spec.json',
        run_directory: '/tmp/feedbax_runs/studio-plan',
        bootstrap: [],
        health_checks: [],
        launch: {
          id: 'launch',
          title: 'Launch execution',
          command: null,
          description: '',
          critical: true,
          metadata: {},
        },
        monitor: [],
        artifact_routes: [],
        cloud_payload: {},
        reproducibility: {},
        warnings: [],
      },
    });

    const state = useWorkspaceStore.getState();
    expect(state.lastTrainingExecutionPreparation?.plan.job_id).toBe('studio-plan');
    expect(state.workspace?.stages.find((stage) => stage.kind === 'train')?.status).toBe('ready');
  });

  it('stores local execution results and returned workspace refs', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });
    const completed = {
      ...workspace,
      stages: workspace.stages.map((stage) =>
        stage.kind === 'train'
          ? {
              ...stage,
              status: 'completed' as const,
              manifest_refs: [
                {
                  kind: 'TrainingRunManifest',
                  id: 'feedbax-training-run:studio-run',
                  role: 'training_run',
                  provider: 'feedbax',
                  uri: '/tmp/feedbax_runs/manifests/training_runs/studio-run.json',
                  metadata: {},
                },
              ],
            }
          : stage
      ),
    };

    useWorkspaceStore.getState().setWorkspace(workspace);
    useWorkspaceStore.getState().setTrainingLocalRunResult({
      workspace: completed,
      stage_id: 'stage:train',
      scenario_id: 'scenario:train',
      execution_spec: { job_id: 'studio-run' },
      snapshot_dir: '/tmp/feedbax_runs/executions/studio-run/inputs',
      result: {
        job_id: 'studio-run',
        status: 'completed',
        return_code: 0,
        stdout_path: '/tmp/feedbax_runs/executions/studio-run/stdout.log',
        stderr_path: '/tmp/feedbax_runs/executions/studio-run/stderr.log',
        manifest_path: '/tmp/feedbax_runs/manifests/training_runs/studio-run.json',
        manifest_payload: { kind: 'TrainingRunManifest' },
        plan: {
          kind: 'ExecutionPlan',
          schema_version: 'feedbax.execution.v1',
          job_id: 'studio-run',
          backend: 'local',
          command: 'python -m feedbax.bin.provider validate training training-spec.json',
          run_directory: '/tmp/feedbax_runs/studio-run',
          bootstrap: [],
          health_checks: [],
          launch: {
            id: 'launch',
            title: 'Launch execution',
            command: null,
            description: '',
            critical: true,
            metadata: {},
          },
          monitor: [],
          artifact_routes: [],
          cloud_payload: {},
          reproducibility: {},
          warnings: [],
        },
      },
    });

    const state = useWorkspaceStore.getState();
    expect(state.lastTrainingLocalRunResult?.result.status).toBe('completed');
    expect(state.workspace?.stages.find((stage) => stage.kind === 'train')?.status).toBe(
      'completed'
    );
    expect(
      state.workspace?.stages.find((stage) => stage.kind === 'train')?.manifest_refs[0]
        .role
    ).toBe('training_run');
  });

  it('stores pipeline materialization results and downstream stage refs', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });
    const completed = {
      ...workspace,
      stages: workspace.stages.map((stage) =>
        stage.kind === 'report'
          ? {
              ...stage,
              status: 'completed' as const,
              manifest_refs: [
                {
                  kind: 'ReportManifest',
                  id: 'feedbax-report:studio-pipeline-report',
                  role: 'report',
                  provider: 'feedbax',
                  uri: '/tmp/feedbax_runs/manifests/reports/studio-pipeline-report.json',
                  metadata: {},
                },
              ],
            }
          : stage
      ),
    };

    useWorkspaceStore.getState().setWorkspace(workspace);
    useWorkspaceStore.getState().setPipelineMaterializationResult({
      workspace: completed,
      stage_ids: ['stage:eval', 'stage:analysis', 'stage:report'],
      manifest_paths: {
        'stage:eval': '/tmp/eval.json',
        'stage:analysis': '/tmp/analysis.json',
        'stage:report': '/tmp/report.json',
      },
      artifact_refs: [
        {
          kind: 'ReportArtifact',
          id: 'artifact://sha256/report',
          role: 'report',
          provider: 'feedbax',
          uri: '/tmp/report-product.json',
          media_type: 'application/json',
          metadata: {},
        },
      ],
    });

    const state = useWorkspaceStore.getState();
    expect(state.lastPipelineMaterializationResult?.stage_ids).toEqual([
      'stage:eval',
      'stage:analysis',
      'stage:report',
    ]);
    expect(state.workspace?.stages.find((stage) => stage.kind === 'report')?.status).toBe(
      'completed'
    );
    expect(
      state.workspace?.stages.find((stage) => stage.kind === 'report')?.manifest_refs[0]
        .role
    ).toBe('report');
  });
});
