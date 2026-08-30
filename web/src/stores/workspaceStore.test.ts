import { beforeEach, describe, expect, it } from 'vitest';
import {
  buildWorkspaceSnapshot,
  getActiveScenario,
  getActiveStage,
  getProjectedScenario,
  getTopPaneState,
  getTrainingScenario,
  getWorkspaceViewMode,
  getWorkspaceViewState,
  objectiveSpecFromLossSpec,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { graphNodeEntityId } from '@/features/scenario/entities';
import { addObjectiveTerm, createObjectiveTerm } from '@/features/scenario/objectives';
import { WORKSPACE_VIEW_STATE_SCHEMA_VERSION } from '@/types/workspace';
import {
  frozenSnapshotProvenanceMetadata,
  useSelectionContextStore,
  type FrozenSnapshotProjection,
} from '@/stores/selectionContextStore';
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
  useSelectionContextStore.getState().reset();
  useWorkspaceStore.setState({
    workspace: null,
    lastTrainingExecutionPreparation: null,
    lastPipelineMaterializationResult: null,
  });
});

describe('workspace snapshot provenance hydration', () => {
  it('restores the frozen projection when persisted workspace metadata is loaded', () => {
    const projection: FrozenSnapshotProjection = {
      source: 'training_run',
      runId: 'run:42',
      runLabel: 'Frozen run 42',
      runStatus: 'completed',
      manifestId: 'manifest:42',
      manifestHash: 'sha256:42',
      specHashes: { graph_spec: 'sha256:graph' },
      snapshot: { graph_spec: graph as unknown as Record<string, unknown> },
    };
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
    });
    workspace.ui_state.top_pane = {
      active_projection: 'model',
      selected_entity_id: null,
      hovered_entity_id: null,
      pinned_inspector_entity_id: null,
      metadata: {
        run_snapshot_provenance: frozenSnapshotProvenanceMetadata(projection),
      },
    };

    useWorkspaceStore.getState().setWorkspace(JSON.parse(JSON.stringify(workspace)));

    expect(useSelectionContextStore.getState().frozenSnapshot).toEqual(projection);

    useWorkspaceStore.getState().setWorkspace({
      ...workspace,
      ui_state: {
        ...workspace.ui_state,
        top_pane: {
          active_projection: 'model',
          selected_entity_id: null,
          hovered_entity_id: null,
          pinned_inspector_entity_id: null,
          metadata: {},
        },
      },
    });
    expect(useSelectionContextStore.getState().frozenSnapshot).toBeNull();
  });

  it('migrates unversioned provenance metadata and clearly rejects unknown versions', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
    });
    const legacyProvenance = {
      source: 'training_run',
      run_id: 'run:legacy',
      run_label: 'Legacy frozen run',
      run_status: 'completed',
      manifest_id: 'manifest:legacy',
      manifest_hash: 'sha256:legacy',
      spec_hashes: { graph_spec: 'sha256:legacy-graph' },
      mode: 'frozen_snapshot',
      read_only: true,
    };
    const legacyTopPane: StudioTopPaneState = {
      active_projection: 'model',
      selected_entity_id: null,
      hovered_entity_id: null,
      pinned_inspector_entity_id: null,
      metadata: { run_snapshot_provenance: legacyProvenance },
    };
    workspace.ui_state.top_pane = legacyTopPane;

    useWorkspaceStore.getState().setWorkspace(workspace);

    expect(useSelectionContextStore.getState().frozenSnapshot).toMatchObject({
      runId: 'run:legacy',
      runLabel: 'Legacy frozen run',
      snapshot: {},
    });

    const loadedWorkspace = useWorkspaceStore.getState().workspace;
    const loadedProjection = useSelectionContextStore.getState().frozenSnapshot;
    const unknownVersionWorkspace = {
      ...workspace,
      ui_state: {
        ...workspace.ui_state,
        top_pane: {
          ...legacyTopPane,
          metadata: {
            ...legacyTopPane.metadata,
            run_snapshot_provenance: {
              ...legacyProvenance,
              schema_version: 'feedbax.studio.run_snapshot_provenance.v999',
            },
          },
        },
      },
    };

    expect(() => useWorkspaceStore.getState().setWorkspace(unknownVersionWorkspace)).toThrow(
      'Unsupported run snapshot provenance schema version: feedbax.studio.run_snapshot_provenance.v999'
    );
    expect(useWorkspaceStore.getState().workspace).toBe(loadedWorkspace);
    expect(useSelectionContextStore.getState().frozenSnapshot).toBe(loadedProjection);
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

    expect(workspace.schema_version).toBe('feedbax.spec.studio.workspace.v2');
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
    expect(scenario).not.toHaveProperty('graph');
    expect(scenario).not.toHaveProperty('graph_ui_state');
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
    expect(trainScenario).not.toHaveProperty('graph');
    expect(trainScenario).not.toHaveProperty('graph_ui_state');

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

    expect(scenario).not.toHaveProperty('graph');
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
    expect(scenario).not.toHaveProperty('graph');
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

  it('projects stage-scoped scenario data without leaking train losses into eval views', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec: {
        ...trainingSpec,
        loss: {
          type: 'Composite',
          label: 'train loss',
          weight: 1,
          children: {
            endpoint: {
              type: 'TargetStateLoss',
              label: 'Train endpoint loss',
              weight: 1,
              selector: 'port:effector.position',
            },
          },
        },
      },
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });
    const trainStage = workspace.stages.find((stage) => stage.kind === 'train')!;
    const evalStage = workspace.stages.find((stage) => stage.kind === 'eval')!;
    const trainScenario = workspace.scenarios[trainStage.scenario_id!];
    const evalScenario = workspace.scenarios[evalStage.scenario_id!];
    const evalObjective: StudioObjectiveSpec = {
      schema_version: 'feedbax.studio.objective.v1',
      terms: [
        {
          id: 'objective:eval_success',
          type_id: 'SuccessRateMetric',
          label: 'Eval success',
          role: 'metric',
          source_selector: null,
          target_selector: null,
          operator: 'maximize',
          weight: 1,
          metadata: {},
        },
      ],
      legacy_loss_spec: null,
      metadata: { stage: 'eval' },
    };
    const scopedWorkspace: StudioWorkspaceSpec = {
      ...workspace,
      scenarios: {
        ...workspace.scenarios,
        [trainStage.scenario_id!]: {
          ...trainScenario,
          biomechanics_spec: {
            schema_id: 'feedbax.spec.studio.biomechanics',
            schema_version: 'feedbax.spec.studio.biomechanics.v1',
            metadata: { source: 'persisted' },
          },
        },
        [evalStage.scenario_id!]: {
          ...evalScenario,
          task_spec: {
            ...taskSpec,
            params: { ...taskSpec.params, target_radius: 0.08 },
          },
          objective_spec: evalObjective,
        },
      },
    };

    const trainProjection = getProjectedScenario(scopedWorkspace, trainStage)!;
    const evalProjection = getProjectedScenario(scopedWorkspace, evalStage)!;
    const trainObjective = trainProjection.objective_spec as StudioObjectiveSpec;
    const projectedEvalObjective = evalProjection.objective_spec as StudioObjectiveSpec;

    expect(trainObjective.terms[0].label).toBe('Train endpoint loss');
    expect(evalProjection).not.toHaveProperty('graph');
    expect(evalProjection.task_spec?.params.target_radius).toBe(0.08);
    expect(evalProjection.biomechanics_spec).toEqual({
      schema_id: 'feedbax.spec.studio.biomechanics',
      schema_version: 'feedbax.spec.studio.biomechanics.v1',
      metadata: { source: 'persisted' },
    });
    expect(projectedEvalObjective).toEqual(evalObjective);
    expect(projectedEvalObjective.terms.map((term) => term.label)).not.toContain(
      'Train endpoint loss'
    );
  });

  it('persists active workspace view camera and overlay state in scenario UI state', () => {
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
    useWorkspaceStore.getState().updateActiveWorkspaceViewState({
      camera: { zoom: 2.25, pan: { x: 120, y: -40 } },
      overlay_visibility: { objectives: false },
      playback: { position: 12.5, speed: 1.5 },
    });

    const state = useWorkspaceStore.getState();
    const trainStage = getActiveStage(state.workspace)!;
    const trainScenario = state.workspace!.scenarios[trainStage.scenario_id!];
    const viewState = getWorkspaceViewState(state.workspace, trainStage);

    expect(viewState.schema_version).toBe(WORKSPACE_VIEW_STATE_SCHEMA_VERSION);
    expect(viewState.camera).toEqual({ zoom: 2.25, pan: { x: 120, y: -40 } });
    expect(viewState.overlay_visibility.objectives).toBe(false);
    expect(viewState.playback).toEqual({ position: 12.5, speed: 1.5 });
    expect(trainScenario.ui_state.workspace_view_state).toMatchObject({
      schema_version: WORKSPACE_VIEW_STATE_SCHEMA_VERSION,
      camera: { zoom: 2.25, pan: { x: 120, y: -40 } },
    });
  });

  it('clears unsupported or stale workspace view refs to defined fallbacks', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });
    const trainStage = workspace.stages.find((stage) => stage.kind === 'train')!;
    const trainScenario = workspace.scenarios[trainStage.scenario_id!];
    const restoredWorkspace: StudioWorkspaceSpec = {
      ...workspace,
      stages: workspace.stages.map((stage) =>
        stage.id === trainStage.id
          ? {
              ...stage,
              artifact_refs: [
                {
                  id: 'artifact:valid',
                  kind: 'Plot',
                  provider: 'test',
                  role: 'workspace_overlay',
                  uri: 'artifact://valid',
                  media_type: null,
                  metadata: {},
                },
              ],
            }
          : stage
      ),
      scenarios: {
        ...workspace.scenarios,
        [trainScenario.id]: {
          ...trainScenario,
          ui_state: {
            workspace_view_state: {
              schema_version: WORKSPACE_VIEW_STATE_SCHEMA_VERSION,
              camera: { zoom: 99, pan: { x: '4', y: 'bad' } },
              selected_artifact_ref: 'artifact:valid',
              selected_trial_ref: 'trial:missing',
              comparison_selection: {
                baseline_ref: 'artifact:valid',
                candidate_ref: 'artifact:missing',
              },
              overlay_visibility: { objectives: false, artifacts: 'yes' },
              playback: { position: -4, speed: 99 },
            },
          },
        },
      },
    };

    useWorkspaceStore.getState().setWorkspace(restoredWorkspace);
    const restoredStage = getActiveStage(useWorkspaceStore.getState().workspace)!;
    const viewState = getWorkspaceViewState(useWorkspaceStore.getState().workspace, restoredStage);

    expect(viewState.camera).toEqual({ zoom: 8, pan: { x: 4, y: 0 } });
    expect(viewState.selected_artifact_ref).toBe('artifact:valid');
    expect(viewState.selected_trial_ref).toBe('trial:missing');
    expect(viewState.comparison_selection).toEqual({
      baseline_ref: 'artifact:valid',
      candidate_ref: null,
    });
    expect(viewState.overlay_visibility.objectives).toBe(false);
    expect(viewState.overlay_visibility.artifacts).toBe(true);
    expect(viewState.playback).toEqual({ position: 0, speed: 16 });
    expect(
      getWorkspaceViewState({
        ...workspace,
        ui_state: {
          workspace_view_state: { schema_version: 'feedbax.studio.workspace_view_state.v0' },
        },
      }).schema_version
    ).toBe(WORKSPACE_VIEW_STATE_SCHEMA_VERSION);
  });

  it('derives workspace view mode from stage kind and available data', () => {
    const workspace = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });
    const trainStage = workspace.stages.find((stage) => stage.kind === 'train')!;

    expect(getWorkspaceViewMode(workspace, trainStage)).toBe('authoring');
    useWorkspaceStore.getState().setWorkspace(workspace);
    useWorkspaceStore.getState().updateActiveWorkspaceViewState({
      playback: { position: 4 },
    });
    expect(
      getWorkspaceViewMode(useWorkspaceStore.getState().workspace, trainStage)
    ).toBe('playback');
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

  it('preserves semantic task bindings without consulting a shadow graph', () => {
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
    expect(restored.task_binding_spec?.exposed_data.map((data) => data.id)).toEqual(['inputs']);
    expect(restored.task_binding_spec?.bindings).toHaveLength(1);
  });

  it('normalizes runtime graph aliases without synthesizing network subgraphs', () => {
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
    expect(restored).not.toHaveProperty('graph');
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

  it('stores prepared invocations and backend plans without dropping workspace state', () => {
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
                  kind: 'BackendPlan',
                  id: 'backend-plan:studio-plan',
                  role: 'backend_plan',
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
      graph,
      stage_id: 'stage:train',
      scenario_id: 'scenario:train',
      invocation: {
        schema_id: 'feedbax.spec.invocation',
        schema_version: 'feedbax.spec.invocation.v1',
        invocation_id: 'a'.repeat(64),
        workflow_plan_id: 'b'.repeat(64),
        operation_key: 'campaign:studio',
        operation: {},
        inputs: [],
        requested_outputs: [],
        scientific_seeds: {},
        capabilities: ['training'],
        execution_policy: { timeout_seconds: 60, max_attempts: 1 },
      },
      backend_plan: {
        schema_id: 'feedbax.orchestration.backend_plan',
        schema_version: 'feedbax.orchestration.backend_plan.v1',
        backend_plan_id: 'c'.repeat(64),
        invocation_id: 'a'.repeat(64),
        backend_id: 'local',
        configuration: { job_id: 'studio-plan' },
      },
    });

    const state = useWorkspaceStore.getState();
    expect(state.lastTrainingExecutionPreparation?.backend_plan.configuration.job_id).toBe(
      'studio-plan'
    );
    expect(state.workspace?.stages.find((stage) => stage.kind === 'train')?.status).toBe('ready');
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
