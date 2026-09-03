import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  previewStudioEvaluationMatrix,
  runStudioEvaluationLocalExecution,
  sampleTaskTrials,
  buildStudioPersistenceDocument,
  persistStudioDocument,
  semanticWorkspaceForSave,
  stageStudioEvaluationMatrix,
} from '@/api/client';
import type { GraphMetadata, GraphSpec, GraphUIState } from '@/types/graph';
import type { WorkspaceDocument } from '@/generated/studioContracts';

afterEach(() => {
  vi.unstubAllGlobals();
});

const metadata: GraphMetadata = {
  name: 'Concurrent graph',
  created_at: '2026-07-07T00:00:00+00:00',
  updated_at: '2026-07-07T00:00:00+00:00',
  version: '1.0.0',
  save_revision: 4,
};

const graph: GraphSpec = {
  nodes: {},
  wires: [],
  input_ports: [],
  output_ports: [],
  input_bindings: {},
  output_bindings: {},
  metadata,
};

const uiState: GraphUIState = {
  viewport: { x: 0, y: 0, zoom: 1 },
  node_states: {},
};
const workspaceDocument: WorkspaceDocument = {
  schema_id: 'feedbax.workspace_document',
  schema_version: '1',
  semantic_root: {
    semantic_document_sha256: '0'.repeat(64),
    authored_path: '/graph',
  },
  graph_ui_state: uiState,
  analysis_pages: [],
  semantic_anchors: {},
};

describe('graph API save concurrency', () => {
  it('sends the expected revision in both header and JSON payload', async () => {
    const fetchMock = vi.fn(async () => Response.json({
      data: {
        success: true,
        metadata: { ...metadata, save_revision: 5 },
      },
    }));
    vi.stubGlobal('fetch', fetchMock);

    const workspace = {
      id: 'workspace:test',
      schema_id: 'feedbax.spec.studio.workspace',
      schema_version: 'feedbax.spec.studio.workspace.v2',
      label: 'Test',
      active_stage_id: 'stage:train',
      ui_state: { top_pane: { kind: 'model' } },
      stages: [{
        id: 'stage:train',
        schema_id: 'feedbax.spec.studio.stage',
        schema_version: 'feedbax.spec.studio.stage.v2',
        ui_state: { collapsed: true },
      }],
      scenarios: {
        'scenario:train': {
          id: 'scenario:train',
          ui_state: { workspace_view_state: { mode: 'model' } },
        },
      },
      collections: [],
      manifest_refs: [],
      validation: { errors: [], warnings: [] },
      metadata: {},
    } as any;
    const document = buildStudioPersistenceDocument({ graph, workspaceDocument, workspace });
    document.expected_save_revision = 4;
    const response = await persistStudioDocument('graph-1', document);

    expect(response.metadata.save_revision).toBe(5);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, options] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(options.headers).toMatchObject({ 'If-Match': '4' });
    const body = JSON.parse(options.body as string);
    expect(body).toMatchObject({
      schema_id: 'feedbax.spec.studio.persistence_document',
      schema_version: 'feedbax.spec.studio.persistence_document.v1',
      expected_save_revision: 4,
      graph,
      workspace_document: workspaceDocument,
    });
    expect(body.workspace).not.toHaveProperty('ui_state');
    expect(body.workspace).toMatchObject({
      schema_id: 'feedbax.spec.studio.workspace',
      schema_version: 'feedbax.spec.studio.workspace.v2',
    });
    expect(body.workspace.stages[0]).toMatchObject({
      schema_id: 'feedbax.spec.studio.stage',
      schema_version: 'feedbax.spec.studio.stage.v2',
    });
    expect(body.workspace.stages[0]).not.toHaveProperty('ui_state');
    expect(body.workspace.scenarios['scenario:train']).not.toHaveProperty('ui_state');
  });

  it('refuses missing current workspace and stage identities before request', () => {
    const workspace = {
      id: 'workspace:test',
      schema_version: 'feedbax.spec.studio.workspace.v2',
      label: 'Test',
      stages: [],
      scenarios: {},
      collections: [],
      manifest_refs: [],
      validation: { errors: [], warnings: [] },
      metadata: {},
    } as any;
    expect(() => semanticWorkspaceForSave(workspace)).toThrow(/workspace schema identity/);

    workspace.schema_id = 'feedbax.spec.studio.workspace';
    workspace.stages = [{
      id: 'stage:train',
      schema_version: 'feedbax.spec.studio.stage.v2',
    }];
    expect(() => semanticWorkspaceForSave(workspace)).toThrow(/stage schema identity/);
  });
});

describe('Studio evaluation provider API', () => {
  it('posts eval matrix preview, stage, and local run payloads to backend state endpoints', async () => {
    const workspace = {
      id: 'workspace:eval',
      schema_version: 'feedbax.spec.studio.workspace.v2',
      label: 'Eval workspace',
      stages: [],
      scenarios: {},
      collections: [],
      artifact_refs: [],
      manifest_refs: [],
      metadata: {},
    };
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/preview')) {
        return Response.json({
          workspace,
          stage_id: 'stage:eval',
          selected_training_run_count: 1,
          condition_count: 1,
          checkpoint_policy_count: 1,
          total_eval_count: 1,
          materialized_count: 0,
          pending_count: 0,
          failed_count: 0,
          new_manifest_count: 1,
          launch_count: 1,
          evaluation_run_ids: ['feedbax-evaluation-run:abc'],
          checkpoint_selection_ids: ['feedbax-checkpoint-selection:def'],
          summary: '1 runs x 1 conditions x 1 checkpoint policy = 1 evals - 0 already materialized',
        });
      }
      if (url.endsWith('/stage')) {
        return Response.json({
          workspace,
          stage_id: 'stage:eval',
          preview: {
            workspace,
            stage_id: 'stage:eval',
            selected_training_run_count: 1,
            condition_count: 1,
            checkpoint_policy_count: 1,
            total_eval_count: 1,
            materialized_count: 0,
            pending_count: 1,
            failed_count: 0,
            new_manifest_count: 0,
            launch_count: 1,
            evaluation_run_ids: ['feedbax-evaluation-run:abc'],
            checkpoint_selection_ids: ['feedbax-checkpoint-selection:def'],
            summary: '1 runs x 1 conditions x 1 checkpoint policy = 1 evals - 0 already materialized',
          },
          manifest_refs: [],
          checkpoint_selection_refs: [],
        });
      }
      return Response.json({
        workspace,
        stage_id: 'stage:eval',
        preview: {
          workspace,
          stage_id: 'stage:eval',
          selected_training_run_count: 1,
          condition_count: 1,
          checkpoint_policy_count: 1,
          total_eval_count: 1,
          materialized_count: 1,
          pending_count: 0,
          failed_count: 0,
          new_manifest_count: 0,
          launch_count: 0,
          evaluation_run_ids: ['feedbax-evaluation-run:abc'],
          checkpoint_selection_ids: ['feedbax-checkpoint-selection:def'],
          summary: '1 runs x 1 conditions x 1 checkpoint policy = 1 evals - 1 already materialized',
        },
        manifest_refs: [],
        completed_count: 1,
        failed_count: 0,
        skipped_count: 0,
        skipped_failed_count: 0,
        errors: [],
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    const selectionSpec = {
      mode: 'query' as const,
      manifest_kind: 'TrainingRunManifest',
      query: {
        statuses: ['completed'],
        has_checkpoint: true,
      },
    };
    const payload = {
      workspace: workspace as never,
      selection_spec: selectionSpec,
      checkpoint_policy: {
        mode: 'best-by-metric' as const,
        metric: 'final_validation_loss',
        objective: 'minimize' as const,
        params: {},
      },
      reprocess: 'missing_failed' as const,
    };

    await previewStudioEvaluationMatrix(payload);
    await stageStudioEvaluationMatrix(payload);
    await runStudioEvaluationLocalExecution(payload);

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/provider/studio/evaluation/preview',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/api/provider/studio/evaluation/stage',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      '/api/provider/studio/evaluation/run-local',
      expect.objectContaining({ method: 'POST' }),
    );
    const [, options] = fetchMock.mock.calls[2] as unknown as [string, RequestInit];
    expect(JSON.parse(options.body as string)).toMatchObject({
      selection_spec: selectionSpec,
      checkpoint_policy: { mode: 'best-by-metric', metric: 'final_validation_loss' },
      reprocess: 'missing_failed',
    });
    expect(JSON.parse(options.body as string)).not.toHaveProperty('training_run_ids');
  });
});

describe('task sampling API', () => {
  it('posts task spec, seed, and count to the sampling endpoint', async () => {
    const responsePayload = {
      schema_version: 'feedbax.execution.sampled_task_trials.v1',
      task_type: 'SimpleReaches',
      seed: 9,
      count: 2,
      trials: [],
    };
    const fetchMock = vi.fn(async () => Response.json(responsePayload));
    vi.stubGlobal('fetch', fetchMock);

    const response = await sampleTaskTrials({
      task_spec: { type: 'SimpleReaches', params: { n_steps: 8 } },
      seed: 9,
      count: 2,
    });

    expect(response).toEqual(responsePayload);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [path, options] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(path).toBe('/api/execution/task-trials/sample');
    expect(options.method).toBe('POST');
    expect(JSON.parse(options.body as string)).toEqual({
      task_spec: { type: 'SimpleReaches', params: { n_steps: 8 } },
      seed: 9,
      count: 2,
    });
  });
});
