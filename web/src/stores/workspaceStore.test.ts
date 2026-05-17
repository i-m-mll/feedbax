import { describe, expect, it } from 'vitest';
import { buildWorkspaceSnapshot } from '@/stores/workspaceStore';
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
    expect(scenario.graph).toEqual(graph);
  });

  it('preserves future product stages and metadata while refreshing active drafts', () => {
    const existing = buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec,
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });
    const withFutureStage: StudioWorkspaceSpec = {
      ...existing,
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
      taskSpec,
      analysisSnapshot: null,
      projectName: 'Workspace test',
    });

    const futureStage = refreshed.stages.find(
      (stage) => stage.id === 'stage:future-objective-authoring'
    );
    expect(futureStage?.metadata).toEqual({ later: { keep: true } });

    const trainStage = refreshed.stages.find((stage) => stage.kind === 'train')!;
    const scenario = refreshed.scenarios[trainStage.scenario_id!];
    expect(scenario.training_spec?.n_batches).toBe(200);
    expect(scenario.graph?.output_ports).toEqual(['effector']);
  });
});
