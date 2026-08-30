// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { act } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  buildWorkspaceSnapshot,
  getStageByKind,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { TaskSpec, TrainingSpec } from '@/types/training';
import type { StudioWorkspaceSpec } from '@/types/workspace';
import { AnalysisBundlePanel } from './AnalysisBundlePanel';

const graph: GraphSpec = {
  nodes: {},
  wires: [],
  input_ports: [],
  output_ports: [],
  input_bindings: {},
  output_bindings: {},
  metadata: {
    name: 'Analysis bundle panel test',
    created_at: '2026-07-07T00:00:00Z',
    updated_at: '2026-07-07T00:00:00Z',
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

afterEach(() => {
  vi.unstubAllGlobals();
  act(() => {
    useWorkspaceStore.setState({
      workspace: null,
      lastTrainingExecutionPreparation: null,
      lastPipelineMaterializationResult: null,
    });
  });
});

function dryRunResponse(bundleName: string, runIds: string[]) {
  return {
    schema_id: 'feedbax.spec.studio.api_transport',
    schema_version: 'feedbax.spec.studio.api_transport.v3',
    data: {
      schema_id: 'feedbax.spec.studio.api_transport',
      schema_version: 'feedbax.spec.studio.api_transport.v3',
      dry_run: {
        bundle_name: bundleName,
        match_preview: {
          selection_spec: {
            schema_id: 'feedbax.spec.selection',
            schema_version: 'feedbax.spec.selection.v2',
            mode: 'query',
            manifest_kind: 'EvaluationRunManifest',
            query: {
              manifest_kind: 'EvaluationRunManifest',
              run_ids: runIds,
              source_set_ids: [],
              statuses: [],
              has_checkpoint: null,
              tags: [],
              metadata_equals: {},
              params_equals: {},
              path_equals: {},
              expression: null,
              top_k_by_metric_per_group: null,
            },
            frozen_refs: [],
            metadata: {},
          },
          match_count: runIds.length,
          parent_refs: runIds.map((id) => ({
            kind: 'EvaluationRunManifest',
            id,
            role: 'evaluation_run',
            metadata: {},
          })),
          truncated: false,
        },
        matched_run_ids: runIds,
        stages: [],
        metadata: {},
      },
    },
  };
}

function workspaceWithAnalysisBundles(): StudioWorkspaceSpec {
  const workspace = buildWorkspaceSnapshot({
    workspace: null,
    graph,
    uiState,
    trainingSpec,
    taskSpec,
    analysisSnapshot: null,
    projectName: 'Analysis bundle panel test',
  });
  const analysisStage = getStageByKind(workspace, 'analysis')!;
  const analysisScenario = workspace.scenarios[analysisStage.scenario_id!];
  return {
    ...workspace,
    active_stage_id: analysisStage.id,
    ui_state: {
      ...workspace.ui_state,
      active_stage_id: analysisStage.id,
    },
    stages: workspace.stages.map((stage) =>
      stage.id === analysisStage.id
        ? { ...stage, selection_spec: { eval_run_ids: ['active-stage-eval'] } }
        : stage
    ),
    scenarios: {
      ...workspace.scenarios,
      [analysisScenario.id]: {
        ...analysisScenario,
        analysis_spec: {
          bundles: [
            {
              name: 'bundle-a',
              predicate: { manifest_kind: 'EvaluationRunManifest', run_ids: ['eval-a'] },
              stages: [{ name: 'stage-a', kind: 'analysis' }],
            },
            {
              name: 'bundle-b',
              predicate: { manifest_kind: 'EvaluationRunManifest', run_ids: ['eval-b'] },
              stages: [{ name: 'stage-b', kind: 'analysis' }],
            },
          ],
        },
      },
    },
  };
}

describe('AnalysisBundlePanel', () => {
  it('dry-runs each authored bundle with that bundle predicate', async () => {
    const fetchMock = vi.fn(async (_path: string, options?: RequestInit) => {
      const body = JSON.parse(String(options?.body)) as {
        bundle: { name: string; predicate: { run_ids?: string[] } };
        selection_spec: unknown;
      };
      return Response.json(
        dryRunResponse(body.bundle.name, body.bundle.predicate.run_ids ?? [])
      );
    });
    vi.stubGlobal('fetch', fetchMock);
    act(() => {
      useWorkspaceStore.setState({ workspace: workspaceWithAnalysisBundles() });
    });

    render(<AnalysisBundlePanel />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const payloads = fetchMock.mock.calls.map(([, options]) =>
      JSON.parse(String((options as RequestInit).body))
    );

    expect(payloads).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          bundle: expect.objectContaining({
            name: 'bundle-a',
            predicate: expect.objectContaining({ run_ids: ['eval-a'] }),
          }),
          selection_spec: null,
        }),
        expect.objectContaining({
          bundle: expect.objectContaining({
            name: 'bundle-b',
            predicate: expect.objectContaining({ run_ids: ['eval-b'] }),
          }),
          selection_spec: null,
        }),
      ])
    );
  });

  it('retargets an authored bundle array entry in place', async () => {
    const fetchMock = vi.fn(async (_path: string, options?: RequestInit) => {
      const body = JSON.parse(String(options?.body)) as {
        bundle: { name: string; predicate: { run_ids?: string[] } };
      };
      return Response.json(
        dryRunResponse(body.bundle.name, body.bundle.predicate.run_ids ?? [])
      );
    });
    vi.stubGlobal('fetch', fetchMock);
    act(() => {
      useWorkspaceStore.setState({ workspace: workspaceWithAnalysisBundles() });
    });
    render(<AnalysisBundlePanel />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    const predicateEditors = screen.getAllByRole('textbox');
    const retargetButtons = screen.getAllByRole('button', { name: /retarget/i });
    fireEvent.change(predicateEditors[1], {
      target: {
        value: JSON.stringify(
          {
            manifest_kind: 'EvaluationRunManifest',
            run_ids: ['eval-c'],
            source_set_ids: [],
            statuses: [],
            tags: [],
            metadata_equals: {},
            params_equals: {},
            path_equals: {},
          },
          null,
          2
        ),
      },
    });
    fireEvent.click(retargetButtons[1]);

    const workspace = useWorkspaceStore.getState().workspace!;
    const analysisStage = getStageByKind(workspace, 'analysis')!;
    const analysisSpec = workspace.scenarios[analysisStage.scenario_id!]
      .analysis_spec as Record<string, unknown>;

    expect(analysisSpec).not.toHaveProperty('bundle');
    expect(analysisSpec.bundles).toMatchObject([
      { name: 'bundle-a', predicate: { run_ids: ['eval-a'] } },
      {
        name: 'bundle-b',
        predicate: { run_ids: ['eval-c'] },
        metadata: { predicate_updated_from: 'studio_analysis_bundle_panel' },
      },
    ]);
  });
});
