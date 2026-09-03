import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiRequestError } from '@/api/request';
import { studioPersistenceDocument, type StudioPersistenceResult } from '@/api/client';
import type { WorkspaceDocument } from '@/generated/studioContracts';
import type { GraphMetadata, GraphSpec, GraphUIState } from '@/types/graph';
import type { StudioWorkspaceSpec } from '@/types/workspace';
import type { TaskSpec } from '@/types/training';
import type { AnalysisClassDef } from '@/types/analysis';
import { studioDraftHashes } from '@/utils/studioDraftHash';
import {
  StudioPersistenceCoordinator,
  captureActiveStudioDocument,
  startStudioPersistence,
  studioPersistence,
  type StudioDocumentDraft,
  type StudioPersistenceDependencies,
  type StudioSaveOutcome,
} from '@/services/studioPersistence';
import { useAnalysisStore } from '@/stores/analysisStore';
import { useGraphStore } from '@/stores/graphStore';
import { useTrainingStore } from '@/stores/trainingStore';
import { useProjectsStore } from '@/stores/projectsStore';
import { buildWorkspaceSnapshot, useWorkspaceStore } from '@/stores/workspaceStore';
import {
  delayedReachTimelineFromTask,
  updateTaskTimelineSignalEpochValueSpec,
} from '@/features/scenario/taskTimeline';

const analysisClass: AnalysisClassDef = {
  name: 'ActivityPlot',
  description: 'Plot activity',
  category: 'Figures',
  inputPorts: [],
  outputPorts: [],
  defaultParams: {},
  icon: 'LineChart',
};

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

const uiState: GraphUIState = {
  viewport: { x: 0, y: 0, zoom: 1 },
  node_states: {},
};

function metadata(saveRevision: number): GraphMetadata {
  return {
    name: 'Transaction test',
    created_at: '2026-09-02T00:00:00Z',
    updated_at: '2026-09-02T00:00:00Z',
    version: '1.0.0',
    save_revision: saveRevision,
  };
}

function graph(saveRevision = 4): GraphSpec {
  return {
    nodes: {},
    wires: [],
    input_ports: [],
    output_ports: [],
    input_bindings: {},
    output_bindings: {},
    metadata: metadata(saveRevision),
  };
}

function workspaceDocument(): WorkspaceDocument {
  return {
    schema_id: 'feedbax.workspace_document',
    schema_version: '1',
    semantic_root: {
      semantic_document_sha256: 'a'.repeat(64),
      authored_path: '/graph',
    },
    graph_ui_state: uiState,
    workspace_ui_state: {},
    stage_ui_state: {},
    scenario_ui_state: {},
    analysis_pages: [],
    active_analysis_page_id: null,
    semantic_anchors: {},
  };
}

function workspace(): StudioWorkspaceSpec {
  return {
    id: 'workspace:transaction-test',
    schema_id: 'feedbax.spec.studio.workspace',
    schema_version: 'feedbax.spec.studio.workspace.v2',
    label: 'Transaction test',
    active_stage_id: null,
    ui_state: {},
    stages: [],
    scenarios: {},
    collections: [],
    manifest_refs: [],
    artifact_refs: [],
    validation: { errors: [], warnings: [], metadata: {} },
    metadata: {},
  };
}

function workspaceWithEpochValue(value: number): StudioWorkspaceSpec {
  const taskSpec = {
    type: 'DelayedReaches',
    params: {
      n_steps: 8,
      epoch_len_ranges: [[2, 2], [2, 2]],
      hold_epochs: [0],
      target_on_epochs: [1, 2],
      move_epochs: [2],
    },
  };
  const timeline = delayedReachTimelineFromTask(taskSpec)!;
  const edited = updateTaskTimelineSignalEpochValueSpec(
    timeline,
    'hold',
    'epoch:0',
    {
      schema_version: 'feedbax.spec.studio.value.v2',
      value_form: 'literal',
      variation: { scope: 'fixed', enumerable: null, metadata: {} },
      mode: 'constant',
      value,
      metadata: {},
    }
  );
  const base = workspace();
  return {
    ...base,
    active_stage_id: 'stage:train',
    stages: [{
      id: 'stage:train',
      schema_id: 'feedbax.spec.studio.stage',
      schema_version: 'feedbax.spec.studio.stage.v2',
      kind: 'train',
      label: 'Train',
      status: 'draft',
      scenario_id: 'scenario:train',
      input_collections: [],
      output_collections: [],
      manifest_refs: [],
      artifact_refs: [],
      selection_spec: {},
      validation: { errors: [], warnings: [], metadata: {} },
      ui_state: {},
      metadata: {},
    }],
    scenarios: {
      'scenario:train': {
        id: 'scenario:train',
        schema_version: 'feedbax.spec.studio.scenario.v3',
        label: 'Train',
        stage_id: 'stage:train',
        task_spec: {
          ...taskSpec,
          timeline: edited as unknown as TaskSpec['timeline'],
        },
        validation: { errors: [], warnings: [], metadata: {} },
        ui_state: {},
        metadata: {},
      },
    },
  };
}

function draft({
  documentId = 'document:a',
  graphId = 'graph:a',
  localRevision = 1,
  saveRevision = 4,
  workspaceSpec = workspace(),
}: {
  documentId?: string;
  graphId?: string | null;
  localRevision?: number;
  saveRevision?: number | null;
  workspaceSpec?: StudioWorkspaceSpec;
} = {}): StudioDocumentDraft {
  const graphSpec = graph(saveRevision ?? 0);
  const document = workspaceDocument();
  return {
    documentId,
    label: documentId,
    graphId,
    localRevision,
    saveRevision,
    envelope: studioPersistenceDocument({
      graph: graphSpec,
      workspace_document: document,
      workspace: workspaceSpec,
    }),
    draftHashes: studioDraftHashes({
      graph_spec: graphSpec,
      workspace_document: document,
      workspace: workspaceSpec,
    }),
    workspace: workspaceSpec,
  };
}

function result(saveRevision: number, graphId = 'graph:a', created = false): StudioPersistenceResult {
  return { graphId, metadata: metadata(saveRevision), created };
}

function serverDocument(graphId = 'graph:a') {
  return {
    graph: graph(5),
    workspace_document: workspaceDocument(),
    demo_training_data: null,
    metadata: metadata(5),
    workspace: workspace(),
    compile_reports: null,
    graphId,
  };
}

function harness(
  persist: StudioPersistenceDependencies['persist'],
  fetch: StudioPersistenceDependencies['fetch'] = vi.fn(async () => serverDocument()),
) {
  const started: StudioDocumentDraft[] = [];
  const acknowledged: Array<{
    draft: StudioDocumentDraft;
    result: StudioPersistenceResult;
    workspaceDocument?: WorkspaceDocument;
  }> = [];
  const failed: Array<{ draft: StudioDocumentDraft; outcome: Extract<StudioSaveOutcome, { ok: false }> }> = [];
  const warnings: Array<{ draft: StudioDocumentDraft; message: string; error: unknown }> = [];
  const dependencies: StudioPersistenceDependencies = {
    persist,
    fetch,
    started: (captured) => started.push(captured),
    acknowledged: (captured, persistenceResult, document) => {
      acknowledged.push({ draft: captured, result: persistenceResult, workspaceDocument: document });
    },
    failed: (captured, outcome) => failed.push({ draft: captured, outcome }),
    warning: (captured, message, error) => warnings.push({ draft: captured, message, error }),
  };
  return {
    coordinator: new StudioPersistenceCoordinator(dependencies, 25),
    started,
    acknowledged,
    failed,
    warnings,
  };
}

async function flushPromises(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

afterEach(() => {
  vi.useRealTimers();
});

describe('document-scoped Studio persistence transactions', () => {
  it('keeps the newest exact timeline value through a delayed acknowledgement and retry', async () => {
    const first = deferred<StudioPersistenceResult>();
    const second = deferred<StudioPersistenceResult>();
    const persist = vi
      .fn<StudioPersistenceDependencies['persist']>()
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);
    const { coordinator } = harness(persist);

    coordinator.documentChanged(draft({
      localRevision: 1,
      workspaceSpec: workspaceWithEpochValue(1),
    }));
    const firstSave = coordinator.save(draft({
      localRevision: 1,
      workspaceSpec: workspaceWithEpochValue(1),
    }), 'manual');
    coordinator.documentChanged(draft({
      localRevision: 2,
      workspaceSpec: workspaceWithEpochValue(7),
    }));

    first.resolve(result(5));
    await expect(firstSave).resolves.toMatchObject({ ok: true, localRevision: 1 });
    await flushPromises();
    const latestTimeline = persist.mock.calls[1][1].workspace!.scenarios['scenario:train']
      .task_spec!.timeline!;
    expect(latestTimeline.epoch_value_specs.find(
      (entry) => entry.target_id === 'hold' && entry.epoch_id === 'epoch:0'
    )?.value_spec.value).toBe(7);

    second.resolve(result(6));
    await flushPromises();
  });

  it('keeps a newer autosave edit dirty and submits it after the stale success', async () => {
    vi.useFakeTimers();
    const first = deferred<StudioPersistenceResult>();
    const second = deferred<StudioPersistenceResult>();
    const persist = vi
      .fn<StudioPersistenceDependencies['persist']>()
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);
    const { coordinator, acknowledged } = harness(persist);

    coordinator.documentChanged(draft({ localRevision: 1 }));
    await vi.advanceTimersByTimeAsync(25);
    expect(persist).toHaveBeenCalledTimes(1);

    coordinator.documentChanged(draft({ localRevision: 2 }));
    await vi.advanceTimersByTimeAsync(25);
    expect(persist).toHaveBeenCalledTimes(1);

    first.resolve(result(5));
    await flushPromises();
    expect(acknowledged[0].draft.localRevision).toBe(1);
    expect(persist).toHaveBeenCalledTimes(2);
    expect(persist.mock.calls[1][1].expected_save_revision).toBe(5);

    second.resolve(result(6));
    await flushPromises();
    expect(acknowledged.map((entry) => entry.draft.localRevision)).toEqual([1, 2]);
  });

  it('coalesces overlapping manual and shortcut saves for one document', async () => {
    const pending = deferred<StudioPersistenceResult>();
    const persist = vi.fn<StudioPersistenceDependencies['persist']>(() => pending.promise);
    const { coordinator } = harness(persist);
    const captured = draft({ localRevision: 7 });

    const manual = coordinator.save(captured, 'manual');
    const shortcut = coordinator.save(captured, 'shortcut');
    expect(persist).toHaveBeenCalledTimes(1);

    pending.resolve(result(5));
    await expect(manual).resolves.toMatchObject({ ok: true, localRevision: 7 });
    await expect(shortcut).resolves.toMatchObject({ ok: true, localRevision: 7 });
    expect(persist).toHaveBeenCalledTimes(1);
  });

  it('allows different documents to save independently and acknowledges the captured document', async () => {
    const pendingA = deferred<StudioPersistenceResult>();
    const pendingB = deferred<StudioPersistenceResult>();
    const persist = vi.fn<StudioPersistenceDependencies['persist']>((graphId) =>
      graphId === 'graph:a' ? pendingA.promise : pendingB.promise
    );
    const { coordinator, acknowledged } = harness(persist);

    const saveA = coordinator.save(
      draft({ documentId: 'document:a', graphId: 'graph:a' }),
      'manual',
    );
    const saveB = coordinator.save(
      draft({ documentId: 'document:b', graphId: 'graph:b' }),
      'manual',
    );
    expect(persist).toHaveBeenCalledTimes(2);

    pendingA.resolve(result(5, 'graph:a'));
    await expect(saveA).resolves.toMatchObject({ ok: true, documentId: 'document:a' });
    expect(acknowledged[0].draft.documentId).toBe('document:a');
    expect(coordinator.state('document:b').inFlightRevision).toBe(1);

    pendingB.resolve(result(9, 'graph:b'));
    await expect(saveB).resolves.toMatchObject({ ok: true, documentId: 'document:b' });
  });

  it('creates graph, workspace, presentation, and analysis in one persistence mutation', async () => {
    const persist = vi.fn<StudioPersistenceDependencies['persist']>(async () =>
      result(0, 'graph:created', true)
    );
    const fetch = vi.fn<StudioPersistenceDependencies['fetch']>(async () =>
      serverDocument('graph:created')
    );
    const { coordinator } = harness(persist, fetch);
    const captured = draft({ graphId: null, saveRevision: null });

    const outcome = await coordinator.save(captured, 'template');

    expect(outcome).toMatchObject({
      ok: true,
      result: { graphId: 'graph:created', created: true },
      workspaceDocument: { analysis_pages: [] },
    });
    expect(persist).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(persist.mock.calls[0][0]).toBeNull();
    expect(persist.mock.calls[0][1]).toMatchObject({
      graph: captured.envelope.graph,
      workspace_document: captured.envelope.workspace_document,
      workspace: captured.envelope.workspace,
    });
    expect(captured.draftHashes.schema_version).toBe('feedbax.studio.draft_hashes.v2');
  });

  it('keeps a successful create acknowledged when its follow-up reload fails', async () => {
    const persist = vi.fn<StudioPersistenceDependencies['persist']>(async () =>
      result(0, 'graph:created', true)
    );
    const fetch = vi.fn<StudioPersistenceDependencies['fetch']>(async () => {
      throw new Error('reload unavailable');
    });
    const { coordinator, acknowledged, warnings } = harness(persist, fetch);
    const captured = draft({ graphId: null, saveRevision: null });

    const outcome = await coordinator.save(captured, 'template');

    expect(outcome).toMatchObject({
      ok: true,
      result: { graphId: 'graph:created', created: true },
      workspaceDocument: captured.envelope.workspace_document,
    });
    expect(persist).toHaveBeenCalledTimes(1);
    expect(acknowledged).toHaveLength(1);
    expect(warnings[0].message).toContain('could not reload');
  });

  it('holds a failed transaction for an explicit retry instead of retrying stale data', async () => {
    vi.useFakeTimers();
    const retry = deferred<StudioPersistenceResult>();
    const persist = vi
      .fn<StudioPersistenceDependencies['persist']>()
      .mockRejectedValueOnce(new Error('network unavailable'))
      .mockImplementationOnce(() => retry.promise);
    const { coordinator, failed } = harness(persist);

    const first = await coordinator.save(draft({
      localRevision: 1,
      workspaceSpec: workspaceWithEpochValue(1),
    }), 'manual');
    expect(first).toMatchObject({ ok: false, kind: 'error' });
    expect(failed).toHaveLength(1);
    expect(coordinator.state('document:a').blocked).toBe(true);

    coordinator.documentChanged(draft({
      localRevision: 2,
      workspaceSpec: workspaceWithEpochValue(7),
    }));
    await vi.advanceTimersByTimeAsync(100);
    expect(persist).toHaveBeenCalledTimes(1);

    const explicitRetry = coordinator.save(draft({
      localRevision: 2,
      workspaceSpec: workspaceWithEpochValue(7),
    }), 'manual');
    expect(persist).toHaveBeenCalledTimes(2);
    expect(persist.mock.calls[1][1].workspace!.scenarios['scenario:train']
      .task_spec!.timeline!.epoch_value_specs.find(
        (entry) => entry.target_id === 'hold' && entry.epoch_id === 'epoch:0'
      )?.value_spec.value).toBe(7);
    retry.resolve(result(5));
    await expect(explicitRetry).resolves.toMatchObject({ ok: true, localRevision: 2 });
  });

  it('keeps conflicts blocked and visible with the server comparison', async () => {
    const persist = vi.fn<StudioPersistenceDependencies['persist']>(async () => {
      throw new ApiRequestError('http', '/api/graphs/graph:a', 'stale', { status: 409 });
    });
    const { coordinator, failed } = harness(persist);

    const outcome = await coordinator.save(draft(), 'manual');

    expect(outcome).toMatchObject({ ok: false, kind: 'conflict' });
    expect(failed[0].outcome.message).toContain('Save conflict');
    expect(coordinator.state('document:a').blocked).toBe(true);
  });
});

describe('persisted mutation revisions', () => {
  it('keeps timeline semantic ownership through graph undo and redo', () => {
    useGraphStore.getState().resetGraph();
    useWorkspaceStore.getState().setWorkspace(workspaceWithEpochValue(7));
    const before = useWorkspaceStore.getState().workspace.scenarios['scenario:train']
      .task_spec!.timeline!.epoch_value_specs;

    useGraphStore.getState().addRetainedObservable({
      id: 'observable:timeline-ownership',
      label: 'Timeline ownership',
      source: { node_id: 'missing', port: 'output' },
    } as any);
    useGraphStore.getState().undo();
    useGraphStore.getState().redo();

    expect(useWorkspaceStore.getState().workspace.scenarios['scenario:train']
      .task_spec!.timeline!.epoch_value_specs).toEqual(before);
  });

  it('advances for graph history, workspace, training, analysis, and empty-page clearing', () => {
    useGraphStore.getState().resetGraph();
    useAnalysisStore.getState().resetAnalysis();
    const initialGraph = useGraphStore.getState().capturePersistedGraph();
    useWorkspaceStore.getState().setWorkspace(buildWorkspaceSnapshot({
      workspace: null,
      graph: initialGraph.graph,
      uiState: initialGraph.uiState,
      trainingSpec: useTrainingStore.getState().trainingSpec,
      taskSpec: useTrainingStore.getState().taskSpec,
      analysisSnapshot: useAnalysisStore.getState().captureSnapshot(),
    }));

    const beforeGraph = useGraphStore.getState().localRevision;
    useGraphStore.getState().addRetainedObservable({
      id: 'observable:test',
      label: 'Test observable',
      source: { node_id: 'missing', port: 'output' },
    } as any);
    expect(useGraphStore.getState().localRevision).toBeGreaterThan(beforeGraph);

    const beforeUndo = useGraphStore.getState().localRevision;
    useGraphStore.getState().undo();
    useGraphStore.getState().redo();
    expect(useGraphStore.getState().localRevision).toBeGreaterThanOrEqual(beforeUndo + 2);

    const beforeWorkspace = useGraphStore.getState().localRevision;
    useWorkspaceStore.getState().setActiveStageByKind('analysis');
    expect(useGraphStore.getState().localRevision).toBeGreaterThan(beforeWorkspace);

    const beforeTraining = useGraphStore.getState().localRevision;
    useTrainingStore.getState().setTrainingSpec({ n_batches: 23 });
    expect(useGraphStore.getState().localRevision).toBeGreaterThan(beforeTraining);

    const beforeAnalysis = useGraphStore.getState().localRevision;
    useAnalysisStore.getState().addPage('Temporary page');
    const pageId = useAnalysisStore.getState().activePageId!;
    expect(useGraphStore.getState().localRevision).toBeGreaterThan(beforeAnalysis);

    const beforeNodeAdd = useGraphStore.getState().localRevision;
    useAnalysisStore.getState().addAnalysisNode(analysisClass, { x: 100, y: 200 });
    const nodeId = useAnalysisStore.getState().nodes.find((node) => node.type === 'analysis')!.id;
    expect(useGraphStore.getState().localRevision).toBeGreaterThan(beforeNodeAdd);

    const beforeViewport = useGraphStore.getState().localRevision;
    useAnalysisStore.getState().setViewport({ x: 40, y: -20, zoom: 1.2 });
    expect(useGraphStore.getState().localRevision).toBeGreaterThan(beforeViewport);

    const beforeNodeDelete = useGraphStore.getState().localRevision;
    useAnalysisStore.getState().removeNode(nodeId);
    expect(useGraphStore.getState().localRevision).toBeGreaterThan(beforeNodeDelete);

    const beforeClear = useGraphStore.getState().localRevision;
    useAnalysisStore.getState().removePage(pageId);
    expect(useGraphStore.getState().localRevision).toBeGreaterThan(beforeClear);
    expect(useAnalysisStore.getState().captureSnapshot()).toEqual({
      pages: [],
      activePageId: null,
    });
  });

  it('captures the latest workspace, training, and empty analysis state together', () => {
    useGraphStore.getState().resetGraph();
    useAnalysisStore.getState().resetAnalysis();
    const persistedGraph = useGraphStore.getState().capturePersistedGraph();
    useWorkspaceStore.getState().setWorkspace(buildWorkspaceSnapshot({
      workspace: null,
      graph: persistedGraph.graph,
      uiState: persistedGraph.uiState,
      trainingSpec: useTrainingStore.getState().trainingSpec,
      taskSpec: useTrainingStore.getState().taskSpec,
      analysisSnapshot: useAnalysisStore.getState().captureSnapshot(),
    }));
    useTrainingStore.getState().setTrainingSpec({ n_batches: 31 });
    useAnalysisStore.getState().addPage('Cleared page');
    useAnalysisStore.getState().removePage(useAnalysisStore.getState().activePageId!);

    const captured = captureActiveStudioDocument();
    const document = captured.envelope.workspace_document!;
    const capturedWorkspace = captured.workspace;
    const trainingStage = capturedWorkspace.stages.find((stage) => stage.kind === 'train')!;
    const trainingScenario = capturedWorkspace.scenarios[trainingStage.scenario_id!];

    expect(trainingScenario.training_spec?.n_batches).toBe(31);
    expect(document.analysis_pages).toEqual([]);
    expect(document.active_analysis_page_id).toBeNull();
    expect(captured.draftHashes.schema_version).toBe('feedbax.studio.draft_hashes.v2');
  });

  it('never lets a stale acknowledgement clear a newer active revision', () => {
    useGraphStore.getState().resetGraph();
    const documentId = useProjectsStore.getState().activeTabId;
    useGraphStore.getState().markDirty();
    const capturedRevision = useGraphStore.getState().localRevision;
    useGraphStore.getState().markDirty();

    useProjectsStore.getState().acknowledgeDocumentSave(
      documentId,
      capturedRevision,
      'graph:acknowledged',
      5,
      workspaceDocument(),
    );

    expect(useGraphStore.getState()).toMatchObject({
      graphId: 'graph:acknowledged',
      saveRevision: 5,
      isDirty: true,
      lastSavedAt: null,
    });
  });

  it('keeps semantic hashes stable when only Analysis Canvas layout moves', () => {
    useGraphStore.getState().resetGraph();
    useAnalysisStore.getState().resetAnalysis();
    const persistedGraph = useGraphStore.getState().capturePersistedGraph();
    useWorkspaceStore.getState().setWorkspace(buildWorkspaceSnapshot({
      workspace: null,
      graph: persistedGraph.graph,
      uiState: persistedGraph.uiState,
      trainingSpec: useTrainingStore.getState().trainingSpec,
      taskSpec: useTrainingStore.getState().taskSpec,
      analysisSnapshot: null,
    }));
    useWorkspaceStore.getState().setWorkspaceDocument(workspaceDocument());
    useAnalysisStore.getState().addPage('Layout identity');
    useAnalysisStore.getState().addAnalysisNode(analysisClass, { x: 100, y: 200 });
    const nodeId = useAnalysisStore.getState().nodes.find((node) => node.type === 'analysis')!.id;
    const before = captureActiveStudioDocument();

    useAnalysisStore.getState().onNodesChange([
      { id: nodeId, type: 'position', position: { x: 640, y: -32 }, dragging: false },
    ]);
    useAnalysisStore.getState().setViewport({ x: -120, y: 48, zoom: 1.4 });
    const after = captureActiveStudioDocument();

    expect(after.draftHashes.hashes.graph_spec).toBe(before.draftHashes.hashes.graph_spec);
    expect(after.draftHashes.hashes.workspace).toBe(before.draftHashes.hashes.workspace);
    expect(after.draftHashes.hashes.workspace_document)
      .not.toBe(before.draftHashes.hashes.workspace_document);
    expect(after.workspace.stages.find((stage) => stage.kind === 'analysis')?.execution_spec)
      .toEqual(before.workspace.stages.find((stage) => stage.kind === 'analysis')?.execution_spec);
  });

  it('overlays newer layout after a delayed acknowledgement of an older revision', () => {
    useGraphStore.getState().resetGraph();
    useAnalysisStore.getState().resetAnalysis();
    const persistedGraph = useGraphStore.getState().capturePersistedGraph();
    useWorkspaceStore.getState().setWorkspace(buildWorkspaceSnapshot({
      workspace: null,
      graph: persistedGraph.graph,
      uiState: persistedGraph.uiState,
      trainingSpec: useTrainingStore.getState().trainingSpec,
      taskSpec: useTrainingStore.getState().taskSpec,
      analysisSnapshot: null,
    }));
    useWorkspaceStore.getState().setWorkspaceDocument(workspaceDocument());
    useAnalysisStore.getState().addPage('Delayed layout');
    useAnalysisStore.getState().addAnalysisNode(analysisClass, { x: 100, y: 200 });
    const nodeId = useAnalysisStore.getState().nodes.find((node) => node.type === 'analysis')!.id;
    useAnalysisStore.getState().onNodesChange([
      { id: nodeId, type: 'position', position: { x: 300, y: 400 }, dragging: false },
    ]);
    const captured = captureActiveStudioDocument();
    useAnalysisStore.getState().onNodesChange([
      { id: nodeId, type: 'position', position: { x: 700, y: 800 }, dragging: false },
    ]);

    useProjectsStore.getState().acknowledgeDocumentSave(
      captured.documentId,
      captured.localRevision,
      'graph:acknowledged',
      5,
      captured.envelope.workspace_document,
    );

    const current = captureActiveStudioDocument();
    const pageId = useAnalysisStore.getState().activePageId!;
    expect(current.envelope.workspace_document?.analysis_canvas_layout?.stages?.['stage:analysis']
      ?.pages?.[pageId]?.node_positions?.[nodeId]).toEqual({ x: 700, y: 800 });
    expect(useGraphStore.getState().isDirty).toBe(true);
  });

  it('queues a restored dirty document and preserves its identity across a tab switch', () => {
    useGraphStore.getState().resetGraph();
    useGraphStore.getState().hydrateGraph(graph(), uiState, 'graph:a', [], 4);
    const documentA = useProjectsStore.getState().activeTabId;
    useGraphStore.getState().markDirty();
    const capturedRevision = useGraphStore.getState().localRevision;
    const changed = vi
      .spyOn(studioPersistence, 'documentChanged')
      .mockImplementation(() => undefined);
    const stop = startStudioPersistence();

    try {
      expect(changed).toHaveBeenCalledWith(expect.objectContaining({
        documentId: documentA,
        graphId: 'graph:a',
        localRevision: capturedRevision,
      }));
      changed.mockClear();

      const documentB = useProjectsStore.getState().openNewTab('Document B');

      expect(documentB).not.toBe(documentA);
      expect(changed).toHaveBeenCalledWith(expect.objectContaining({
        documentId: documentA,
        graphId: 'graph:a',
        localRevision: capturedRevision,
      }));
    } finally {
      stop();
      changed.mockRestore();
    }
  });

  it('applies a late acknowledgement to the captured inactive tab only', () => {
    useGraphStore.getState().resetGraph();
    useGraphStore.getState().hydrateGraph(graph(), uiState, 'graph:a', [], 4);
    const documentA = useProjectsStore.getState().activeTabId;
    useGraphStore.getState().markDirty();
    const capturedRevision = useGraphStore.getState().localRevision;
    const documentB = useProjectsStore.getState().openNewTab('Document B');

    useProjectsStore.getState().acknowledgeDocumentSave(
      documentA,
      capturedRevision,
      'graph:a',
      5,
      workspaceDocument(),
    );

    expect(useProjectsStore.getState().activeTabId).toBe(documentB);
    expect(useGraphStore.getState().graphId).toBeNull();
    const capturedTab = useProjectsStore.getState().tabs.find((tab) => tab.tabId === documentA)!;
    expect(capturedTab.graphSnapshot).toMatchObject({
      graphId: 'graph:a',
      saveRevision: 5,
      isDirty: false,
    });
  });
});
