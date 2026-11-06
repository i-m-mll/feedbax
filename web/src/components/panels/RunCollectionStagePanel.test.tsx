// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { EvaluateCollectionPanel, TrainCollectionPanel } from '@/components/panels/RunCollectionStagePanel';
import { useGraphStore } from '@/stores/graphStore';
import { defaultTaskSpec, defaultTrainingSpec } from '@/stores/trainingStore';
import { buildWorkspaceSnapshot, getScenario, getStageByKind, useWorkspaceStore } from '@/stores/workspaceStore';
import { pipelineWorkspaceFixture, TRAINING_RUN_ID } from '@/test/pipelineWorkspaceFixture';
import { currentDraftSpecHashesForScenario } from '@/utils/pipelineCollections';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  useWorkspaceStore.getState().restoreWorkspace(null);
});

it.each([
  { mode: 'dispatch', stagedStatus: 'pending' },
  { mode: 'recover an ambiguous dispatch', stagedStatus: 'pending' },
  { mode: 'resume on mount', stagedStatus: 'claimed' },
])('can $mode and poll one staged row without selecting an evaluation input', async ({ mode, stagedStatus }) => {
  const { graph, uiState } = useGraphStore.getState().capturePersistedGraph();
  const workspace = buildWorkspaceSnapshot({
    workspace: null, graph, uiState,
    trainingSpec: defaultTrainingSpec,
    taskSpec: defaultTaskSpec,
    analysisSnapshot: null,
  });
  const train = getStageByKind(workspace, 'train')!;
  const evaluate = getStageByKind(workspace, 'eval')!;
  const stagingHashes = currentDraftSpecHashesForScenario(getScenario(workspace, train.scenario_id), graph);
  const refs = ['A', 'B'].map((label) => ({
    id: `run:${label}`,
    kind: 'TrainingRunManifest',
    role: 'training_run',
    provider: 'feedbax',
    uri: `/runs/staged-${label}.json`,
    metadata: {
      name: `Row ${label}`,
      status: label === 'B' ? stagedStatus : 'pending',
      planned: true,
      execution_target: 'local',
      staging_generation: `generation-${label}`,
      native_launch_supported: true,
      staging_draft_spec_hashes: stagingHashes,
    },
  }));
  train.manifest_refs = refs;
  train.output_collections[0].item_refs = refs;
  useWorkspaceStore.getState().restoreWorkspace(workspace);

  const fetchRun = vi.fn();
  if (mode === 'dispatch') {
    fetchRun.mockResolvedValueOnce(Response.json({ run_set_id: 'set:1', row_id: 'row:B' }, { status: 202 }));
  } else if (mode === 'recover an ambiguous dispatch') {
    fetchRun.mockResolvedValueOnce(Response.json({ detail: 'dispatch response failed' }, { status: 500 }));
  }
  const summary = {
    id: 'run:B', name: 'Row B', created_at: '2026-07-07T00:00:00Z', hyperparams: {},
    staging_generation: 'generation-B',
  };
  fetchRun.mockResolvedValueOnce(Response.json({ ...summary, status: 'claimed' }));
  fetchRun.mockResolvedValueOnce(Response.json({
    ...summary,
    status: 'completed',
    result_manifest_id: 'result:B',
    metrics: { completed_batches: 5, total_batches: 5 },
    teardown_pending: mode === 'dispatch',
  }));
  if (mode === 'dispatch') {
    fetchRun.mockResolvedValueOnce(Response.json({
      ...summary,
      status: 'completed',
      result_manifest_id: 'result:B',
      metrics: { completed_batches: 5, total_batches: 5 },
      execution_error: 'could not release the content lease',
    }));
  }
  vi.stubGlobal('fetch', fetchRun);

  render(<TrainCollectionPanel />);
  if (stagedStatus === 'pending') {
    const launch = screen.getByRole<HTMLButtonElement>('button', { name: 'Launch Row B' });
    fireEvent.click(launch);
    expect(launch.disabled).toBe(true);
  }
  await screen.findByText('Preflight');
  await waitFor(() => expect(screen.getAllByText('completed').length).toBeGreaterThan(0), { timeout: 2500 });
  if (mode === 'dispatch') {
    await screen.findAllByText('could not release the content lease', {}, { timeout: 2500 });
  }

  expect(fetchRun.mock.calls.map(([path]) => path)).toEqual([
    ...(stagedStatus === 'pending' ? ['/api/provider/studio/training/dispatch'] : []),
    '/api/runs/training/run%3AB', '/api/runs/training/run%3AB',
    ...(mode === 'dispatch' ? ['/api/runs/training/run%3AB'] : []),
  ]);
  if (stagedStatus === 'pending') {
    expect(JSON.parse(fetchRun.mock.calls[0][1].body)).toEqual({ manifest_id: 'run:B', generation: 'generation-B' });
  }
  const current = useWorkspaceStore.getState().workspace;
  expect(getStageByKind(current, 'train')?.output_collections[0].item_refs).toMatchObject([
    { id: 'run:A', metadata: { status: 'pending' } },
    { id: 'run:B', metadata: { status: 'completed', result_manifest_id: 'result:B' } },
  ]);
  expect(getStageByKind(current, 'eval')?.selection_spec).toEqual(evaluate.selection_spec);
});

it('previews the configured evaluation conditions without overriding stage parameters', async () => {
  const workspace = pipelineWorkspaceFixture();
  const evaluate = getStageByKind(workspace, 'eval')!;
  evaluate.selection_spec.condition_matrix = {
    axes: [{ id: 'gain', label: 'Feedback gain', path: 'eval_params.gain', values: [0, 0.5] }],
  };
  useWorkspaceStore.getState().restoreWorkspace(workspace);

  const fetchPreview = vi.fn().mockResolvedValue(Response.json({
    condition_labels: ['Feedback gain=0', 'Feedback gain=0.5'],
    pending_count: 0,
    failed_count: 0,
    launch_count: 2,
    summary: '2 evaluations',
  }));
  vi.stubGlobal('fetch', fetchPreview);

  render(<EvaluateCollectionPanel />);
  fireEvent.click(screen.getByRole('button', { name: 'Preview' }));
  await screen.findByText('Feedback gain=0');
  expect(screen.getByText('Feedback gain=0.5')).toBeTruthy();

  const [path, options] = fetchPreview.mock.calls[0];
  const payload = JSON.parse(options.body);
  expect(path).toBe('/api/provider/studio/evaluation/preview');
  expect(payload.training_run_ids).toEqual([TRAINING_RUN_ID]);
  expect(payload).not.toHaveProperty('eval_params');
  expect(getStageByKind(payload.workspace, 'eval')?.selection_spec.condition_matrix)
    .toEqual(evaluate.selection_spec.condition_matrix);
});
