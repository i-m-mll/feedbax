import { beforeEach, describe, expect, it } from 'vitest';
import {
  buildWorkspaceSnapshot,
  getActiveScenario,
  getActiveStage,
  getTrainingScenario,
  objectiveSpecFromLossSpec,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { TrainingSpec, TaskSpec } from '@/types/training';
import type { StudioWorkspaceSpec } from '@/types/workspace';

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
    expect(scenario.objective_spec).toEqual(objectiveSpecFromLossSpec(trainingSpec.loss));
    expect(scenario.graph).toEqual(graph);
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
    const withFutureStage: StudioWorkspaceSpec = {
      ...existing,
      scenarios: {
        ...existing.scenarios,
        [trainStage.scenario_id!]: {
          ...existing.scenarios[trainStage.scenario_id!],
          training_spec: workspaceOwnedTrainingSpec,
          task_spec: workspaceOwnedTaskSpec,
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

    const scenario = getTrainingScenario(useWorkspaceStore.getState().workspace)!;
    expect(scenario.training_spec?.n_batches).toBe(500);
    expect(scenario.task_spec?.params.n_targets).toBe(16);
    expect(scenario.metadata.dirty).toBe(true);
    expect(scenario.objective_spec?.terms).toHaveLength(1);
    expect(scenario.objective_spec?.terms[0].source_selector).toMatchObject({
      namespace: 'graph_port',
      target_id: 'effector',
      path: 'position',
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
