import { toast } from 'sonner';
import {
  buildStudioPersistenceDocument,
  fetchGraph,
  persistStudioDocument,
  type StudioPersistenceEnvelope,
  type StudioPersistenceResult,
} from '@/api/client';
import { isHttpConflict } from '@/api/request';
import { useAnalysisStore } from '@/stores/analysisStore';
import {
  capturePersistedGraphFromSnapshot,
  useGraphStore,
} from '@/stores/graphStore';
import { persistLocalProjectTabs, useProjectsStore } from '@/stores/projectsStore';
import type { OpenTab } from '@/stores/projectsStore';
import { useTrainingStore } from '@/stores/trainingStore';
import {
  buildNewWorkspaceDocumentSnapshot,
  buildWorkspaceDocumentSnapshot,
  buildWorkspaceSnapshot,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import type { WorkspaceDocument } from '@/generated/studioContracts';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { AnalysisSnapshot } from '@/types/analysis';
import type { StudioWorkspaceSpec } from '@/types/workspace';
import {
  studioDraftHashes,
  type CurrentStudioDraftHashes,
} from '@/utils/studioDraftHash';
import { summarizeSaveConflict } from '@/utils/saveConflict';

export const AUTO_SAVE_DELAY_MS = 800;

export type StudioSaveReason = 'autosave' | 'manual' | 'shortcut' | 'template';

export interface StudioDocumentDraft {
  documentId: string;
  label: string;
  graphId: string | null;
  localRevision: number;
  saveRevision: number | null;
  envelope: StudioPersistenceEnvelope;
  draftHashes: CurrentStudioDraftHashes;
  workspace: StudioWorkspaceSpec;
}

export type StudioSaveOutcome =
  | {
      ok: true;
      documentId: string;
      localRevision: number;
      result: StudioPersistenceResult;
      workspaceDocument?: WorkspaceDocument;
      warning?: string;
    }
  | {
      ok: false;
      documentId: string;
      localRevision: number;
      kind: 'error' | 'conflict';
      message: string;
      error: unknown;
    };

interface PendingWaiter {
  localRevision: number;
  resolve: (outcome: StudioSaveOutcome) => void;
}

interface DocumentQueue {
  pending: StudioDocumentDraft | null;
  inFlight: StudioDocumentDraft | null;
  timer: ReturnType<typeof setTimeout> | null;
  blocked: boolean;
  knownSaveRevision: number | null;
  waiters: PendingWaiter[];
}

export interface StudioPersistenceDependencies {
  persist: (
    graphId: string | null,
    envelope: StudioPersistenceEnvelope,
  ) => Promise<StudioPersistenceResult>;
  fetch: typeof fetchGraph;
  started: (draft: StudioDocumentDraft) => void;
  acknowledged: (
    draft: StudioDocumentDraft,
    result: StudioPersistenceResult,
    workspaceDocument?: WorkspaceDocument,
  ) => void;
  failed: (draft: StudioDocumentDraft, outcome: Extract<StudioSaveOutcome, { ok: false }>) => void;
  warning?: (draft: StudioDocumentDraft, message: string, error: unknown) => void;
}

function immutable<T>(value: T): T {
  return structuredClone(value);
}

function transportSnapshot<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function captureWorkspaceDocument(
  graphId: string | null,
  document: WorkspaceDocument | null,
  uiState: GraphUIState,
  analysisSnapshot: AnalysisSnapshot,
  workspace: StudioWorkspaceSpec,
): WorkspaceDocument {
  return graphId
    ? buildWorkspaceDocumentSnapshot(document, uiState, analysisSnapshot, workspace)
    : buildNewWorkspaceDocumentSnapshot(uiState, analysisSnapshot, workspace);
}

function transactionEnvelope(
  draft: StudioDocumentDraft,
  knownSaveRevision: number | null,
): StudioPersistenceEnvelope {
  const envelope = immutable(draft.envelope);
  if (draft.graphId) {
    const expectedSaveRevision = knownSaveRevision ?? draft.saveRevision;
    if (expectedSaveRevision === null) {
      throw new Error(`Project "${draft.label}" has no server save revision.`);
    }
    envelope.expected_save_revision = expectedSaveRevision;
  } else {
    delete envelope.expected_save_revision;
  }
  return envelope;
}

export class StudioPersistenceCoordinator {
  private readonly queues = new Map<string, DocumentQueue>();

  constructor(
    private readonly dependencies: StudioPersistenceDependencies,
    private readonly autoSaveDelayMs = AUTO_SAVE_DELAY_MS,
  ) {}

  documentChanged(draft: StudioDocumentDraft): void {
    const queue = this.queue(draft.documentId);
    if (!queue.pending || draft.localRevision >= queue.pending.localRevision) {
      queue.pending = immutable(draft);
    }
    if (draft.graphId && !queue.blocked) this.schedule(queue, draft.documentId);
  }

  save(draft: StudioDocumentDraft, _reason: Exclude<StudioSaveReason, 'autosave'>): Promise<StudioSaveOutcome> {
    const queue = this.queue(draft.documentId);
    queue.blocked = false;
    if (
      (!queue.inFlight || draft.localRevision > queue.inFlight.localRevision) &&
      (!queue.pending || draft.localRevision >= queue.pending.localRevision)
    ) {
      queue.pending = immutable(draft);
    }
    if (queue.timer) {
      clearTimeout(queue.timer);
      queue.timer = null;
    }
    const outcome = new Promise<StudioSaveOutcome>((resolve) => {
      queue.waiters.push({ localRevision: draft.localRevision, resolve });
    });
    void this.drain(draft.documentId);
    return outcome;
  }

  reset(documentId: string): void {
    const queue = this.queues.get(documentId);
    if (!queue) return;
    if (queue.timer) clearTimeout(queue.timer);
    this.queues.delete(documentId);
  }

  state(documentId: string): Readonly<{
    pendingRevision: number | null;
    inFlightRevision: number | null;
    blocked: boolean;
  }> {
    const queue = this.queues.get(documentId);
    return {
      pendingRevision: queue?.pending?.localRevision ?? null,
      inFlightRevision: queue?.inFlight?.localRevision ?? null,
      blocked: queue?.blocked ?? false,
    };
  }

  private queue(documentId: string): DocumentQueue {
    const existing = this.queues.get(documentId);
    if (existing) return existing;
    const created: DocumentQueue = {
      pending: null,
      inFlight: null,
      timer: null,
      blocked: false,
      knownSaveRevision: null,
      waiters: [],
    };
    this.queues.set(documentId, created);
    return created;
  }

  private schedule(queue: DocumentQueue, documentId: string): void {
    if (queue.timer) clearTimeout(queue.timer);
    queue.timer = setTimeout(() => {
      queue.timer = null;
      void this.drain(documentId);
    }, this.autoSaveDelayMs);
  }

  private async drain(documentId: string): Promise<void> {
    const queue = this.queue(documentId);
    if (queue.inFlight || queue.blocked || !queue.pending) return;

    const draft = queue.pending;
    queue.pending = null;
    queue.inFlight = draft;
    this.dependencies.started(draft);
    let attemptedSaveRevision = draft.saveRevision;

    try {
      const envelope = transactionEnvelope(draft, queue.knownSaveRevision);
      attemptedSaveRevision = envelope.expected_save_revision ?? null;
      const result = await this.dependencies.persist(draft.graphId, envelope);
      let workspaceDocument: WorkspaceDocument | undefined;
      let warning: string | undefined;
      if (result.created) {
        workspaceDocument = draft.envelope.workspace_document;
        try {
          workspaceDocument = (await this.dependencies.fetch(result.graphId)).workspace_document;
        } catch (error) {
          warning = `Created "${draft.label}", but could not reload its admitted workspace document.`;
          this.dependencies.warning?.(draft, warning, error);
        }
      }
      queue.knownSaveRevision = result.metadata.save_revision ?? null;
      this.dependencies.acknowledged(draft, result, workspaceDocument);
      const outcome: StudioSaveOutcome = {
        ok: true,
        documentId,
        localRevision: draft.localRevision,
        result,
        workspaceDocument,
        warning,
      };
      this.resolveWaiters(queue, draft.localRevision, outcome);
      queue.inFlight = null;
      if (queue.pending && queue.pending.localRevision > draft.localRevision) {
        void this.drain(documentId);
      }
    } catch (error) {
      const kind = isHttpConflict(error) ? 'conflict' : 'error';
      let message = `Could not save "${draft.label}". Its changes remain local.`;
      if (kind === 'conflict' && draft.graphId) {
        const server = await this.dependencies.fetch(draft.graphId).catch(() => null);
        const localDocument = draft.envelope.workspace_document;
        message = server && localDocument
          ? summarizeSaveConflict({
              expectedRevision: attemptedSaveRevision,
              serverMetadata: server.metadata,
              local: {
                graph: draft.envelope.graph as GraphSpec,
                uiState: localDocument.graph_ui_state as GraphUIState,
                workspace: draft.workspace,
                analysisPages: localDocument.analysis_pages ?? [],
                activeAnalysisPageId: localDocument.active_analysis_page_id ?? null,
              },
              server: {
                graph: server.graph,
                uiState: server.workspace_document.graph_ui_state,
                workspace: server.workspace,
                analysisPages: server.workspace_document.analysis_pages,
                activeAnalysisPageId:
                  server.workspace_document.active_analysis_page_id ?? null,
              },
            })
          : `Save conflict for "${draft.label}". Reload the server copy or keep the local draft before trying again.`;
      }
      const outcome: Extract<StudioSaveOutcome, { ok: false }> = {
        ok: false,
        documentId,
        localRevision: draft.localRevision,
        kind,
        message,
        error,
      };
      queue.blocked = true;
      queue.inFlight = null;
      this.dependencies.failed(draft, outcome);
      this.resolveWaiters(queue, Number.POSITIVE_INFINITY, outcome);
    }
  }

  private resolveWaiters(
    queue: DocumentQueue,
    throughRevision: number,
    outcome: StudioSaveOutcome,
  ): void {
    const remaining: PendingWaiter[] = [];
    for (const waiter of queue.waiters) {
      if (waiter.localRevision <= throughRevision) waiter.resolve(outcome);
      else remaining.push(waiter);
    }
    queue.waiters = remaining;
  }
}

const defaultDependencies: StudioPersistenceDependencies = {
  persist: persistStudioDocument,
  fetch: fetchGraph,
  started: (draft) => {
    useProjectsStore.getState().markDocumentSaveStarted(draft.documentId);
  },
  acknowledged: (draft, result, workspaceDocument) => {
    useProjectsStore.getState().acknowledgeDocumentSave(
      draft.documentId,
      draft.localRevision,
      result.graphId,
      result.metadata.save_revision!,
      workspaceDocument,
    );
    persistLocalProjectTabs();
  },
  failed: (draft, outcome) => {
    useProjectsStore
      .getState()
      .markDocumentSaveFailed(draft.documentId, outcome.kind, outcome.message);
    persistLocalProjectTabs();
    toast.error(outcome.message, {
      id: `studio-save-${outcome.kind}-${draft.documentId}`,
      duration: outcome.kind === 'conflict' ? 12000 : undefined,
    });
  },
  warning: (draft, message) => {
    toast.warning(message, { id: `studio-save-warning-${draft.documentId}` });
  },
};

export const studioPersistence = new StudioPersistenceCoordinator(defaultDependencies);

export function captureActiveStudioDocument(): StudioDocumentDraft {
  const graphStore = useGraphStore.getState();
  const documentId = useProjectsStore.getState().activeTabId;
  const label =
    useProjectsStore.getState().tabs.find((tab) => tab.tabId === documentId)?.label ??
    graphStore.currentGraphLabel;
  const persistedGraph = graphStore.capturePersistedGraph();
  const analysisSnapshot = useAnalysisStore.getState().captureSnapshot();
  const workspace = buildWorkspaceSnapshot({
    workspace: useWorkspaceStore.getState().workspace,
    graph: persistedGraph.graph,
    uiState: persistedGraph.uiState,
    trainingSpec: useTrainingStore.getState().trainingSpec,
    taskSpec: useTrainingStore.getState().taskSpec,
    analysisSnapshot,
    graphStackPath: persistedGraph.graphStackPath,
  });
  const workspaceDocument = captureWorkspaceDocument(
    graphStore.graphId || null,
    useWorkspaceStore.getState().workspaceDocument,
    persistedGraph.uiState,
    analysisSnapshot,
    workspace,
  );
  useWorkspaceStore.getState().restoreWorkspace(workspace);
  const envelope = transportSnapshot(buildStudioPersistenceDocument({
    graph: persistedGraph.graph,
    workspaceDocument,
    workspace,
  }));
  return immutable({
    documentId,
    label,
    graphId: graphStore.graphId || null,
    localRevision: graphStore.localRevision,
    saveRevision: graphStore.saveRevision,
    envelope,
    draftHashes: studioDraftHashes({
      graph_spec: envelope.graph,
      workspace_document: envelope.workspace_document,
      workspace: envelope.workspace,
    }),
    workspace,
  });
}

export function captureStoredStudioDocument(tab: OpenTab): StudioDocumentDraft {
  const persistedGraph = capturePersistedGraphFromSnapshot(tab.graphSnapshot);
  const workspace = buildWorkspaceSnapshot({
    workspace: tab.workspaceSnapshot,
    graph: persistedGraph.graph,
    uiState: persistedGraph.uiState,
    trainingSpec: tab.trainingSnapshot.trainingSpec,
    taskSpec: tab.trainingSnapshot.taskSpec,
    analysisSnapshot: tab.analysisSnapshot,
    graphStackPath: persistedGraph.graphStackPath,
  });
  const workspaceDocument = captureWorkspaceDocument(
    tab.graphSnapshot.graphId,
    tab.workspaceDocumentSnapshot,
    persistedGraph.uiState,
    tab.analysisSnapshot,
    workspace,
  );
  const envelope = transportSnapshot(buildStudioPersistenceDocument({
    graph: persistedGraph.graph,
    workspaceDocument,
    workspace,
  }));
  return immutable({
    documentId: tab.tabId,
    label: tab.label,
    graphId: tab.graphSnapshot.graphId,
    localRevision: tab.graphSnapshot.localRevision,
    saveRevision: tab.graphSnapshot.saveRevision,
    envelope,
    draftHashes: studioDraftHashes({
      graph_spec: envelope.graph,
      workspace_document: envelope.workspace_document,
      workspace: envelope.workspace,
    }),
    workspace,
  });
}

export function buildDetachedStudioDocument({
  documentId,
  label,
  graph,
  uiState,
  analysisSnapshot,
  workspace,
}: {
  documentId: string;
  label: string;
  graph: GraphSpec;
  uiState: GraphUIState;
  analysisSnapshot: AnalysisSnapshot;
  workspace: StudioWorkspaceSpec;
}): StudioDocumentDraft {
  const workspaceDocument = buildNewWorkspaceDocumentSnapshot(
    uiState,
    analysisSnapshot,
    workspace,
  );
  const envelope = transportSnapshot(
    buildStudioPersistenceDocument({ graph, workspaceDocument, workspace }),
  );
  return immutable({
    documentId,
    label,
    graphId: null,
    localRevision: 1,
    saveRevision: null,
    envelope,
    draftHashes: studioDraftHashes({
      graph_spec: envelope.graph,
      workspace_document: envelope.workspace_document,
      workspace: envelope.workspace,
    }),
    workspace,
  });
}

let stopPersistenceSubscriptions: (() => void) | null = null;

export function startStudioPersistence(): () => void {
  if (stopPersistenceSubscriptions) return stopPersistenceSubscriptions;
  let captureQueued = false;
  const captureDirtyDocument = () => {
    const state = useGraphStore.getState();
    if (!state.isDirty) return;
    if (state.saveStatus === 'error' || state.saveStatus === 'conflict') return;
    try {
      studioPersistence.documentChanged(captureActiveStudioDocument());
    } catch (error) {
      const documentId = useProjectsStore.getState().activeTabId;
      const message = error instanceof Error ? error.message : 'Could not capture Studio document.';
      useProjectsStore.getState().markDocumentSaveFailed(documentId, 'error', message);
      toast.error(message, { id: `studio-save-capture-${documentId}` });
    }
  };
  const queueCapture = () => {
    if (captureQueued) return;
    captureQueued = true;
    queueMicrotask(() => {
      captureQueued = false;
      captureDirtyDocument();
    });
  };
  const unsubscribeGraph = useGraphStore.subscribe((state, previous) => {
    if (state.localRevision !== previous.localRevision) queueCapture();
  });
  const unsubscribeProjects = useProjectsStore.subscribe((state, previous) => {
    if (state.activeTabId === previous.activeTabId) return;
    const outgoing = state.tabs.find((tab) => tab.tabId === previous.activeTabId);
    if (outgoing?.graphSnapshot.isDirty && outgoing.graphSnapshot.graphId) {
      try {
        studioPersistence.documentChanged(captureStoredStudioDocument(outgoing));
      } catch (error) {
        const message = error instanceof Error
          ? error.message
          : 'Could not capture the inactive Studio document.';
        useProjectsStore
          .getState()
          .markDocumentSaveFailed(outgoing.tabId, 'error', message);
        toast.error(message, { id: `studio-save-capture-${outgoing.tabId}` });
      }
    }
    queueCapture();
  });
  captureDirtyDocument();
  stopPersistenceSubscriptions = () => {
    unsubscribeGraph();
    unsubscribeProjects();
    stopPersistenceSubscriptions = null;
  };
  return stopPersistenceSubscriptions;
}

export function saveActiveStudioDocument(
  reason: 'manual' | 'shortcut',
): Promise<StudioSaveOutcome> {
  return studioPersistence.save(captureActiveStudioDocument(), reason);
}
