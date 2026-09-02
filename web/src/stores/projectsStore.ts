import { create } from 'zustand';
import {
  useGraphStore,
  createInitialGraph,
  createBlankGraph,
  createGraphSnapshotFromPersistedGraph,
  type GraphSnapshot,
} from '@/stores/graphStore';
import { useTrainingStore, defaultTrainingSpec, defaultTaskSpec } from '@/stores/trainingStore';
import { useTrajectoryStore } from '@/stores/trajectoryStore';
import { useStatisticsStore } from '@/stores/statisticsStore';
import { useAnalysisStore } from '@/stores/analysisStore';
import { buildWorkspaceSnapshot, useWorkspaceStore } from '@/stores/workspaceStore';
import { normalizeGraphForStudioAuthoring } from '@/features/graph/normalization';
import type { TrainingSpec, TaskSpec, LossValidationError } from '@/types/training';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { AnalysisSnapshot } from '@/types/analysis';
import type { StudioWorkspaceSpec } from '@/types/workspace';
import type { WorkspaceDocument } from '@/generated/studioContracts';

export interface TrainingSnapshot {
  trainingSpec: TrainingSpec;
  taskSpec: TaskSpec;
  selectedLossPath: string[] | null;
  lossValidationErrors: LossValidationError[];
  highlightedProbeSelector: string | null;
}

export interface OpenTab {
  tabId: string;
  label: string;
  graphSnapshot: GraphSnapshot;
  trainingSnapshot: TrainingSnapshot;
  analysisSnapshot: AnalysisSnapshot | null;
  workspaceSnapshot: StudioWorkspaceSpec | null;
  workspaceDocumentSnapshot: WorkspaceDocument | null;
}

type StoredGraphSnapshot = Omit<
  GraphSnapshot,
  'past' | 'future' | 'saveRevision' | 'localRevision' | 'saveStatus' | 'saveError'
> & {
  saveRevision?: number | null;
  localRevision?: number;
  saveStatus?: GraphSnapshot['saveStatus'];
  saveError?: string | null;
  past?: GraphSnapshot['past'];
  future?: GraphSnapshot['future'];
  graphHistory?: unknown;
};

type StoredGraphLayer = GraphSnapshot['graphStack'][number] & {
  past?: unknown;
  future?: unknown;
  graphHistory?: unknown;
};

type StoredOpenTab = Omit<OpenTab, 'graphSnapshot'> & {
  graphSnapshot: StoredGraphSnapshot;
};

const LOCAL_PROJECTS_STORAGE_KEY = 'feedbax:studio-local-tabs';
const LOCAL_PROJECTS_STORAGE_VERSION = 1;
const LOCAL_PERSIST_DELAY_MS = 250;
const LAST_PROJECT_STORAGE_KEY = 'feedbax:lastProjectId';

function captureGraphSnapshot(): GraphSnapshot {
  const s = useGraphStore.getState();
  return {
    graph: s.graph,
    uiState: s.uiState,
    graphId: s.graphId,
    saveRevision: s.saveRevision,
    localRevision: s.localRevision,
    isDirty: s.isDirty,
    lastSavedAt: s.lastSavedAt,
    saveStatus: s.saveStatus,
    saveError: s.saveError,
    graphStack: s.graphStack,
    currentGraphLabel: s.currentGraphLabel,
    currentContext: s.currentContext,
    edgeStyle: s.edgeStyle,
    past: s.past,
    future: s.future,
    selectedTapId: s.selectedTapId,
    selectedEdgeId: s.selectedEdgeId,
    pendingStateMerge: s.pendingStateMerge,
  };
}

function captureTrainingSnapshot(): TrainingSnapshot {
  const s = useTrainingStore.getState();
  const workspace = useWorkspaceStore.getState().workspace;
  const workspaceOwned = workspace ? trainingSnapshotFromWorkspace(workspace) : null;
  return {
    trainingSpec: workspaceOwned?.trainingSpec ?? s.trainingSpec,
    taskSpec: workspaceOwned?.taskSpec ?? s.taskSpec,
    selectedLossPath: s.selectedLossPath,
    lossValidationErrors: s.lossValidationErrors,
    highlightedProbeSelector: s.highlightedProbeSelector,
  };
}

function makeInitialGraphSnapshot(): GraphSnapshot {
  const { graph, uiState } = createInitialGraph();
  return {
    graph,
    uiState,
    graphId: null,
    saveRevision: null,
    localRevision: 0,
    isDirty: false,
    lastSavedAt: null,
    saveStatus: 'idle',
    saveError: null,
    graphStack: [],
    currentGraphLabel: graph.metadata?.name ?? 'Model',
    currentContext: 'top-level',
    edgeStyle: 'bezier',
    past: [],
    future: [],
    selectedTapId: null,
    selectedEdgeId: null,
    pendingStateMerge: null,
  };
}

function makeBlankGraphSnapshot(name: string): GraphSnapshot {
  const graph = createBlankGraph();
  graph.metadata!.name = name;
  const uiState: GraphUIState = {
    viewport: { x: 0, y: 0, zoom: 1 },
    node_states: {},
  };
  return {
    graph,
    uiState,
    graphId: null,
    saveRevision: null,
    localRevision: 0,
    isDirty: false,
    lastSavedAt: null,
    saveStatus: 'idle',
    saveError: null,
    graphStack: [],
    currentGraphLabel: name,
    currentContext: 'top-level',
    edgeStyle: 'bezier',
    past: [],
    future: [],
    selectedTapId: null,
    selectedEdgeId: null,
    pendingStateMerge: null,
  };
}

function makeInitialTrainingSnapshot(): TrainingSnapshot {
  return {
    trainingSpec: defaultTrainingSpec,
    taskSpec: defaultTaskSpec,
    selectedLossPath: null,
    lossValidationErrors: [],
    highlightedProbeSelector: null,
  };
}

function trainingSnapshotFromWorkspace(
  workspace: StudioWorkspaceSpec | null | undefined
): TrainingSnapshot {
  const trainStage = workspace?.stages.find((stage) => stage.kind === 'train');
  const scenario =
    trainStage?.scenario_id
      ? workspace?.scenarios[trainStage.scenario_id]
      : null;
  return {
    ...makeInitialTrainingSnapshot(),
    trainingSpec: scenario?.training_spec ?? defaultTrainingSpec,
    taskSpec: scenario?.task_spec ?? defaultTaskSpec,
  };
}

function generateTabId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `tab-${Date.now()}-${Math.floor(Math.random() * 1_000_000)}`;
}

function resetTrajectoryStoreForTabSwitch() {
  const datasets = useTrajectoryStore.getState().datasets;
  useTrajectoryStore.setState({
    activeDataset: null,
    metadata: null,
    filteredIndices: null,
    activeIndex: null,
    trajectoryData: null,
    loading: false,
    error: null,
    playback: { playing: false, speed: 1, frame: 0, totalFrames: 0 },
    datasets,
  });
}

function resetStatisticsStoreForTabSwitch() {
  useStatisticsStore.setState({
    summaryData: null,
    timeseriesData: null,
    histogramData: null,
    scatterData: null,
    diagnosticsData: null,
  });
}

function captureAnalysisSnapshot(): AnalysisSnapshot {
  return useAnalysisStore.getState().captureSnapshot();
}

function captureWorkspaceSnapshot(
  graphSnapshot: GraphSnapshot,
  trainingSnapshot: TrainingSnapshot,
  analysisSnapshot: AnalysisSnapshot | null,
): StudioWorkspaceSpec {
  const persistedGraph = useGraphStore.getState().capturePersistedGraph();
  return buildWorkspaceSnapshot({
    workspace: useWorkspaceStore.getState().workspace,
    graph: persistedGraph.graph,
    uiState: persistedGraph.uiState,
    trainingSpec: trainingSnapshot.trainingSpec,
    taskSpec: trainingSnapshot.taskSpec,
    analysisSnapshot,
    projectName: graphSnapshot.currentGraphLabel,
    graphStackPath: persistedGraph.graphStackPath,
  });
}

function graphStackPathFromWorkspace(
  workspace: StudioWorkspaceSpec | null | undefined
): string[] {
  const value = workspace?.ui_state.graph_stack_path;
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string' && item.length > 0)
    : [];
}

function restoreAnalysisSnapshot(snapshot: AnalysisSnapshot | null) {
  if (snapshot) {
    useAnalysisStore.getState().restoreSnapshot(snapshot);
  } else {
    useAnalysisStore.getState().resetAnalysis();
  }
}

function makeInitialAnalysisSnapshot(): AnalysisSnapshot {
  return { pages: [], activePageId: null };
}

function captureCurrentTab(tab: OpenTab): OpenTab {
  const graphSnapshot = captureGraphSnapshot();
  const trainingSnapshot = captureTrainingSnapshot();
  const analysisSnapshot = captureAnalysisSnapshot();
  return {
    ...tab,
    label: useGraphStore.getState().currentGraphLabel || 'Model',
    graphSnapshot,
    trainingSnapshot,
    analysisSnapshot,
    workspaceSnapshot: captureWorkspaceSnapshot(
      graphSnapshot,
      trainingSnapshot,
      analysisSnapshot,
    ),
    workspaceDocumentSnapshot: useWorkspaceStore.getState().workspaceDocument,
  };
}

function localStorageOrNull(): Storage | null {
  if (typeof window === 'undefined') return null;
  try {
    const storage = window.localStorage;
    return typeof storage?.getItem === 'function' && typeof storage?.setItem === 'function'
      ? storage
      : null;
  } catch {
    return null;
  }
}

export function getLastProjectId(): string | null {
  return localStorageOrNull()?.getItem(LAST_PROJECT_STORAGE_KEY) ?? null;
}

export function setLastProjectId(id: string): void {
  localStorageOrNull()?.setItem(LAST_PROJECT_STORAGE_KEY, id);
}

function compactGraphLayerForStorage(layer: GraphSnapshot['graphStack'][number]) {
  const {
    past: _past,
    future: _future,
    graphHistory: _graphHistory,
    ...storedLayer
  } = layer as StoredGraphLayer;
  return storedLayer;
}

function graphSnapshotForRuntime(snapshot: StoredGraphSnapshot): GraphSnapshot {
  return {
    ...snapshot,
    saveRevision: snapshot.saveRevision ?? null,
    localRevision: snapshot.localRevision ?? 0,
    saveStatus: snapshot.saveStatus ?? 'idle',
    saveError: snapshot.saveError ?? null,
    graph: normalizeGraphForStudioAuthoring(snapshot.graph),
    graphStack: (snapshot.graphStack ?? []).map(compactGraphLayerForStorage),
    past: [],
    future: [],
    pendingStateMerge: null,
  };
}

function graphSnapshotForStorage(snapshot: GraphSnapshot): StoredGraphSnapshot {
  const {
    past: _past,
    future: _future,
    graphHistory: _graphHistory,
    ...storedSnapshot
  } = graphSnapshotForRuntime(snapshot) as GraphSnapshot & { graphHistory?: unknown };
  return storedSnapshot;
}

function tabForRuntime(tab: StoredOpenTab): OpenTab {
  return {
    ...tab,
    graphSnapshot: graphSnapshotForRuntime(tab.graphSnapshot),
    workspaceSnapshot: tab.workspaceSnapshot,
    workspaceDocumentSnapshot: tab.workspaceDocumentSnapshot ?? null,
  };
}

function tabForStorage(tab: OpenTab): StoredOpenTab {
  return {
    ...tab,
    graphSnapshot: graphSnapshotForStorage(tab.graphSnapshot),
    workspaceSnapshot: tab.workspaceSnapshot,
  };
}

function isOpenTab(value: unknown): value is OpenTab {
  if (!value || typeof value !== 'object') return false;
  const tab = value as Partial<OpenTab>;
  return (
    typeof tab.tabId === 'string' &&
    typeof tab.label === 'string' &&
    Boolean(tab.graphSnapshot?.graph) &&
    Boolean(tab.graphSnapshot?.uiState) &&
    Boolean(tab.trainingSnapshot)
  );
}

function isDisposableStartupPlaceholder(tab: OpenTab): boolean {
  return (
    tab.graphSnapshot.graphId === null &&
    tab.graphSnapshot.isDirty === false &&
    tab.label === 'Reaching Task Model' &&
    tab.graphSnapshot.graph.metadata?.name === 'Reaching Task Model' &&
    tab.trainingSnapshot.taskSpec.type === 'SimpleReaches'
  );
}

function discardRestoredStartupPlaceholders(tabs: OpenTab[]): OpenTab[] {
  if (!tabs.some((tab) => tab.graphSnapshot.graphId !== null)) return tabs;
  const filtered = tabs.filter((tab) => !isDisposableStartupPlaceholder(tab));
  return filtered.length > 0 ? filtered : tabs;
}

function restoreTabStores(tab: OpenTab) {
  const normalizedTab = tabForRuntime(tab);
  useGraphStore.getState().restoreSnapshot(normalizedTab.graphSnapshot);
  useTrainingStore.setState({
    trainingSpec: normalizedTab.trainingSnapshot.trainingSpec,
    taskSpec: normalizedTab.trainingSnapshot.taskSpec,
    selectedLossPath: normalizedTab.trainingSnapshot.selectedLossPath,
    lossValidationErrors: normalizedTab.trainingSnapshot.lossValidationErrors,
    highlightedProbeSelector: normalizedTab.trainingSnapshot.highlightedProbeSelector,
  });
  restoreAnalysisSnapshot(normalizedTab.analysisSnapshot);
  useWorkspaceStore.getState().restoreWorkspace(normalizedTab.workspaceSnapshot);
  useWorkspaceStore
    .getState()
    .setWorkspaceDocument(normalizedTab.workspaceDocumentSnapshot);
  resetTrajectoryStoreForTabSwitch();
  resetStatisticsStoreForTabSwitch();
}

function loadLocalProjectTabs(): { tabs: OpenTab[]; activeTabId: string } | null {
  const storage = localStorageOrNull();
  if (!storage) return null;
  const raw = storage.getItem(LOCAL_PROJECTS_STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as {
      version?: number;
      activeTabId?: string;
      tabs?: unknown[];
    };
    if (parsed.version !== LOCAL_PROJECTS_STORAGE_VERSION || !Array.isArray(parsed.tabs)) {
      return null;
    }
    const tabs = discardRestoredStartupPlaceholders(
      parsed.tabs.filter(isOpenTab).map((tab) => tabForRuntime(tab as StoredOpenTab))
    );
    if (tabs.length === 0) return null;
    const activeTabId =
      typeof parsed.activeTabId === 'string' &&
      tabs.some((tab) => tab.tabId === parsed.activeTabId)
        ? parsed.activeTabId
        : tabs[0].tabId;
    const activeTab = tabs.find((tab) => tab.tabId === activeTabId) ?? tabs[0];
    restoreTabStores(activeTab);
    return { tabs, activeTabId: activeTab.tabId };
  } catch {
    return null;
  }
}

export function persistLocalProjectTabs(): boolean {
  const storage = localStorageOrNull();
  if (!storage) return false;
  const { tabs, activeTabId } = useProjectsStore.getState();
  const persistedTabs = tabs.map((tab) =>
    tabForStorage(tab.tabId === activeTabId ? captureCurrentTab(tab) : tab)
  );
  try {
    storage.setItem(
      LOCAL_PROJECTS_STORAGE_KEY,
      JSON.stringify({
        version: LOCAL_PROJECTS_STORAGE_VERSION,
        activeTabId,
        savedAt: new Date().toISOString(),
        tabs: persistedTabs,
      })
    );
    return true;
  } catch {
    return false;
  }
}

interface ProjectsStoreState {
  tabs: OpenTab[];
  activeTabId: string;
  hasRestoredLocalTabs: boolean;
  openNewTab: (name: string) => string;
  openProjectInTab: (
    graphId: string,
    graph: GraphSpec,
    uiState: GraphUIState,
    projectName?: string,
    analysisSnapshot?: AnalysisSnapshot | null,
    workspaceSnapshot?: StudioWorkspaceSpec | null,
    options?: {
      replaceActiveTab?: boolean;
      saveRevision?: number | null;
      workspaceDocument?: WorkspaceDocument | null;
    },
  ) => string;
  switchTab: (tabId: string) => void;
  closeTab: (tabId: string) => void;
  updateActiveTabLabel: (label: string) => void;
  renameTab: (tabId: string, name: string) => void;
  markDocumentSaveStarted: (documentId: string) => void;
  acknowledgeDocumentSave: (
    documentId: string,
    capturedLocalRevision: number,
    graphId: string,
    saveRevision: number,
    workspaceDocument?: WorkspaceDocument | null,
  ) => void;
  markDocumentSaveFailed: (
    documentId: string,
    status: 'error' | 'conflict',
    message: string,
  ) => void;
}

function buildInitialTab(): OpenTab {
  const graphSnapshot = captureGraphSnapshot();
  const trainingSnapshot = captureTrainingSnapshot();
  const analysisSnapshot = captureAnalysisSnapshot();
  const workspaceSnapshot = captureWorkspaceSnapshot(
    graphSnapshot,
    trainingSnapshot,
    analysisSnapshot,
  );
  useWorkspaceStore.getState().restoreWorkspace(workspaceSnapshot);
  return {
    tabId: generateTabId(),
    label: graphSnapshot.currentGraphLabel || 'Model',
    graphSnapshot,
    trainingSnapshot,
    analysisSnapshot,
    workspaceSnapshot,
    workspaceDocumentSnapshot: null,
  };
}

export const useProjectsStore = create<ProjectsStoreState>((set, get) => {
  const restored = loadLocalProjectTabs();
  const firstTab = restored ? null : buildInitialTab();

  return {
    tabs: restored?.tabs ?? [firstTab as OpenTab],
    activeTabId: restored?.activeTabId ?? (firstTab as OpenTab).tabId,
    hasRestoredLocalTabs: Boolean(restored),

    openNewTab: (name: string) => {
      // Save current tab state
      const { tabs, activeTabId } = get();
      const updatedTabs = tabs.map((tab) =>
        tab.tabId === activeTabId ? captureCurrentTab(tab) : tab
      );

      // Create a blank snapshot for the new tab
      const newGraphSnapshot = makeBlankGraphSnapshot(name);
      const newTrainingSnapshot = makeInitialTrainingSnapshot();
      const newAnalysisSnapshot = makeInitialAnalysisSnapshot();
      const newWorkspaceSnapshot = buildWorkspaceSnapshot({
        workspace: null,
        graph: newGraphSnapshot.graph,
        uiState: newGraphSnapshot.uiState,
        trainingSpec: newTrainingSnapshot.trainingSpec,
        taskSpec: newTrainingSnapshot.taskSpec,
        analysisSnapshot: newAnalysisSnapshot,
        projectName: name,
      });
      const newTab: OpenTab = {
        tabId: generateTabId(),
        label: name,
        graphSnapshot: newGraphSnapshot,
        trainingSnapshot: newTrainingSnapshot,
        analysisSnapshot: newAnalysisSnapshot,
        workspaceSnapshot: newWorkspaceSnapshot,
        workspaceDocumentSnapshot: null,
      };

      // Restore the new tab's state into stores
      useGraphStore.getState().restoreSnapshot(newGraphSnapshot);
      useTrainingStore.setState({
        trainingSpec: newTrainingSnapshot.trainingSpec,
        taskSpec: newTrainingSnapshot.taskSpec,
        selectedLossPath: newTrainingSnapshot.selectedLossPath,
        lossValidationErrors: newTrainingSnapshot.lossValidationErrors,
        highlightedProbeSelector: newTrainingSnapshot.highlightedProbeSelector,
      });
      restoreAnalysisSnapshot(newAnalysisSnapshot);
      useWorkspaceStore.getState().restoreWorkspace(newWorkspaceSnapshot);
      useWorkspaceStore.getState().setWorkspaceDocument(null);
      resetTrajectoryStoreForTabSwitch();
      resetStatisticsStoreForTabSwitch();

      set({ tabs: [...updatedTabs, newTab], activeTabId: newTab.tabId });
      return newTab.tabId;
    },

    openProjectInTab: (
      graphId,
      graph,
      uiState,
      projectName,
      analysisSnapshot,
      workspaceSnapshot,
      options,
    ) => {
      const { tabs, activeTabId } = get();
      const updatedTabs = tabs.map((tab) =>
        tab.tabId === activeTabId ? captureCurrentTab(tab) : tab
      );
      const authoringGraph = normalizeGraphForStudioAuthoring(graph);
      const authoringWorkspace = workspaceSnapshot ?? null;
      const tabLabel = projectName ?? authoringGraph.metadata?.name ?? 'Untitled';

      const graphSnapshot = createGraphSnapshotFromPersistedGraph({
        graph: authoringGraph,
        uiState,
        graphId,
        saveRevision: options?.saveRevision ?? null,
        label: tabLabel,
        graphStackPath: graphStackPathFromWorkspace(authoringWorkspace),
      });
      const trainingSnapshot = trainingSnapshotFromWorkspace(authoringWorkspace);
      const restoredAnalysis = analysisSnapshot ?? makeInitialAnalysisSnapshot();
      const restoredWorkspace = buildWorkspaceSnapshot({
        workspace: authoringWorkspace,
        graph: authoringGraph,
        uiState,
        trainingSpec: trainingSnapshot.trainingSpec,
        taskSpec: trainingSnapshot.taskSpec,
        analysisSnapshot: restoredAnalysis,
        projectName: tabLabel,
        graphStackPath: graphSnapshot.graphStack
          .map((layer) => layer.childNodeId)
          .filter((nodeId): nodeId is string => Boolean(nodeId)),
      });
      const newTab: OpenTab = {
        tabId: generateTabId(),
        label: tabLabel,
        graphSnapshot,
        trainingSnapshot,
        analysisSnapshot: restoredAnalysis,
        workspaceSnapshot: restoredWorkspace,
        workspaceDocumentSnapshot: options?.workspaceDocument ?? null,
      };

      // Restore the new project into stores
      useGraphStore.getState().restoreSnapshot(graphSnapshot);
      useTrainingStore.setState({
        trainingSpec: trainingSnapshot.trainingSpec,
        taskSpec: trainingSnapshot.taskSpec,
        selectedLossPath: trainingSnapshot.selectedLossPath,
        lossValidationErrors: trainingSnapshot.lossValidationErrors,
        highlightedProbeSelector: trainingSnapshot.highlightedProbeSelector,
      });
      restoreAnalysisSnapshot(restoredAnalysis);
      useWorkspaceStore.getState().restoreWorkspace(restoredWorkspace);
      useWorkspaceStore
        .getState()
        .setWorkspaceDocument(options?.workspaceDocument ?? null);
      resetTrajectoryStoreForTabSwitch();
      resetStatisticsStoreForTabSwitch();

      if (options?.replaceActiveTab) {
        set({
          tabs: updatedTabs.map((tab) => (tab.tabId === activeTabId ? newTab : tab)),
          activeTabId: newTab.tabId,
        });
      } else {
        set({ tabs: [...updatedTabs, newTab], activeTabId: newTab.tabId });
      }
      return newTab.tabId;
    },

    switchTab: (tabId) => {
      const { tabs, activeTabId } = get();
      if (tabId === activeTabId) return;
      const target = tabs.find((t) => t.tabId === tabId);
      if (!target) return;
      const targetTrainingSnapshot =
        target.workspaceSnapshot
          ? {
              ...trainingSnapshotFromWorkspace(target.workspaceSnapshot),
              selectedLossPath: target.trainingSnapshot.selectedLossPath,
              lossValidationErrors: target.trainingSnapshot.lossValidationErrors,
              highlightedProbeSelector: target.trainingSnapshot.highlightedProbeSelector,
            }
          : target.trainingSnapshot;

      // Save current tab state
      const updatedTabs = tabs.map((tab) =>
        tab.tabId === activeTabId ? captureCurrentTab(tab) : tab
      );

      // Restore the target tab's store state
      useGraphStore.getState().restoreSnapshot(target.graphSnapshot);
      useTrainingStore.setState({
        trainingSpec: targetTrainingSnapshot.trainingSpec,
        taskSpec: targetTrainingSnapshot.taskSpec,
        selectedLossPath: targetTrainingSnapshot.selectedLossPath,
        lossValidationErrors: targetTrainingSnapshot.lossValidationErrors,
        highlightedProbeSelector: targetTrainingSnapshot.highlightedProbeSelector,
      });
      restoreAnalysisSnapshot(target.analysisSnapshot);
      useWorkspaceStore.getState().restoreWorkspace(target.workspaceSnapshot);
      useWorkspaceStore
        .getState()
        .setWorkspaceDocument(target.workspaceDocumentSnapshot);
      resetTrajectoryStoreForTabSwitch();
      resetStatisticsStoreForTabSwitch();

      set({ tabs: updatedTabs, activeTabId: tabId });
    },

    closeTab: (tabId) => {
      const { tabs, activeTabId } = get();
      if (tabs.length <= 1) return; // Never close last tab

      const idx = tabs.findIndex((t) => t.tabId === tabId);
      if (idx === -1) return;

      const nextTabs = tabs.filter((t) => t.tabId !== tabId);

      if (tabId === activeTabId) {
        // Switch to adjacent tab: prefer left, otherwise right
        const nextIdx = idx > 0 ? idx - 1 : 0;
        const nextTab = nextTabs[nextIdx];
        const nextTrainingSnapshot =
          nextTab.workspaceSnapshot
            ? {
                ...trainingSnapshotFromWorkspace(nextTab.workspaceSnapshot),
                selectedLossPath: nextTab.trainingSnapshot.selectedLossPath,
                lossValidationErrors: nextTab.trainingSnapshot.lossValidationErrors,
                highlightedProbeSelector: nextTab.trainingSnapshot.highlightedProbeSelector,
              }
            : nextTab.trainingSnapshot;

        // Restore next tab's store state
        useGraphStore.getState().restoreSnapshot(nextTab.graphSnapshot);
        useTrainingStore.setState({
          trainingSpec: nextTrainingSnapshot.trainingSpec,
          taskSpec: nextTrainingSnapshot.taskSpec,
          selectedLossPath: nextTrainingSnapshot.selectedLossPath,
          lossValidationErrors: nextTrainingSnapshot.lossValidationErrors,
          highlightedProbeSelector: nextTrainingSnapshot.highlightedProbeSelector,
        });
        restoreAnalysisSnapshot(nextTab.analysisSnapshot);
        useWorkspaceStore.getState().restoreWorkspace(nextTab.workspaceSnapshot);
        useWorkspaceStore
          .getState()
          .setWorkspaceDocument(nextTab.workspaceDocumentSnapshot);
        resetTrajectoryStoreForTabSwitch();
        resetStatisticsStoreForTabSwitch();

        set({ tabs: nextTabs, activeTabId: nextTab.tabId });
      } else {
        set({ tabs: nextTabs });
      }
    },

    updateActiveTabLabel: (label) => {
      const { tabs, activeTabId } = get();
      set({
        tabs: tabs.map((tab) =>
          tab.tabId === activeTabId ? { ...tab, label } : tab
        ),
      });
    },

    renameTab: (tabId, name) => {
      const { tabs, activeTabId } = get();
      set({
        tabs: tabs.map((tab) =>
          tab.tabId === tabId ? { ...tab, label: name } : tab
        ),
      });
      // If renaming the active tab, also update graphStore's currentGraphLabel and graph metadata
      if (tabId === activeTabId) {
        const gs = useGraphStore.getState();
        useGraphStore.setState({
          currentGraphLabel: name,
          graph: {
            ...gs.graph,
            metadata: gs.graph.metadata
              ? { ...gs.graph.metadata, name }
              : { name, created_at: new Date().toISOString(), updated_at: new Date().toISOString(), version: '1.0.0' },
          },
        });
        useGraphStore.getState().markDirty();
      }
    },

    markDocumentSaveStarted: (documentId) => {
      const { activeTabId, tabs } = get();
      if (documentId === activeTabId) useGraphStore.getState().markSaveStarted();
      set({
        tabs: tabs.map((tab) =>
          tab.tabId === documentId
            ? {
                ...tab,
                graphSnapshot: {
                  ...tab.graphSnapshot,
                  saveStatus: 'saving',
                  saveError: null,
                },
              }
            : tab
        ),
      });
    },

    acknowledgeDocumentSave: (
      documentId,
      capturedLocalRevision,
      graphId,
      saveRevision,
      workspaceDocument,
    ) => {
      const { activeTabId, tabs } = get();
      if (documentId === activeTabId) {
        useGraphStore
          .getState()
          .acknowledgeSave(capturedLocalRevision, graphId, saveRevision);
        if (workspaceDocument !== undefined) {
          useWorkspaceStore.getState().setWorkspaceDocument(workspaceDocument);
        }
      }
      const acknowledgedAt = new Date().toISOString();
      set({
        tabs: tabs.map((tab) => {
          if (tab.tabId !== documentId) return tab;
          const liveRevision =
            documentId === activeTabId
              ? useGraphStore.getState().localRevision
              : tab.graphSnapshot.localRevision;
          const isCurrent = liveRevision === capturedLocalRevision;
          return {
            ...tab,
            graphSnapshot: {
              ...tab.graphSnapshot,
              graphId,
              saveRevision,
              localRevision: liveRevision,
              isDirty: isCurrent ? false : tab.graphSnapshot.isDirty,
              lastSavedAt: isCurrent ? acknowledgedAt : tab.graphSnapshot.lastSavedAt,
              saveStatus: 'idle',
              saveError: null,
            },
            workspaceDocumentSnapshot:
              workspaceDocument === undefined
                ? tab.workspaceDocumentSnapshot
                : workspaceDocument,
          };
        }),
      });
    },

    markDocumentSaveFailed: (documentId, status, message) => {
      const { activeTabId, tabs } = get();
      if (documentId === activeTabId) {
        useGraphStore.getState().markSaveFailed(status, message);
      }
      set({
        tabs: tabs.map((tab) =>
          tab.tabId === documentId
            ? {
                ...tab,
                graphSnapshot: {
                  ...tab.graphSnapshot,
                  isDirty: true,
                  saveStatus: status,
                  saveError: message,
                },
              }
            : tab
        ),
      });
    },
  };
});

let localPersistTimer: ReturnType<typeof setTimeout> | null = null;

function scheduleLocalProjectPersistence() {
  if (!localStorageOrNull()) return;
  if (localPersistTimer) clearTimeout(localPersistTimer);
  localPersistTimer = setTimeout(() => {
    localPersistTimer = null;
    persistLocalProjectTabs();
  }, LOCAL_PERSIST_DELAY_MS);
}

type StoreWithSubscribe<S> = {
  getState: () => S;
  subscribe: (listener: (state: S) => void) => () => void;
};

function sameFields(previous: readonly unknown[], next: readonly unknown[]) {
  return (
    previous.length === next.length &&
    previous.every((value, index) => Object.is(value, next[index]))
  );
}

function subscribeToLocalProjectPersistence<S>(
  store: StoreWithSubscribe<S>,
  selectFields: (state: S) => readonly unknown[]
) {
  let previous = selectFields(store.getState());
  store.subscribe((state) => {
    const next = selectFields(state);
    if (sameFields(previous, next)) return;
    previous = next;
    scheduleLocalProjectPersistence();
  });
}

function workspaceUiStatePersistenceSignature(uiState: Record<string, unknown>) {
  const { top_pane: _topPane, ...rest } = uiState;
  return JSON.stringify(rest);
}

if (localStorageOrNull()) {
  subscribeToLocalProjectPersistence(useProjectsStore, (state) => [
    state.tabs,
    state.activeTabId,
  ]);
  subscribeToLocalProjectPersistence(useGraphStore, (state) => [
    state.graph,
    state.graphId,
    state.saveRevision,
    state.lastSavedAt,
    state.graphStack,
    state.currentGraphLabel,
    state.currentContext,
    state.edgeStyle,
  ]);
  subscribeToLocalProjectPersistence(useTrainingStore, (state) => [
    state.trainingSpec,
    state.taskSpec,
  ]);
  subscribeToLocalProjectPersistence(useAnalysisStore, (state) => [
    state.graphSpec,
    state.pages,
    state.activePageId,
    state.evalParams,
    state.evalRunId,
  ]);
  subscribeToLocalProjectPersistence(useWorkspaceStore, (state) => [
    state.workspace?.id,
    state.workspace?.schema_version,
    state.workspace?.label,
    state.workspace?.active_stage_id,
    state.workspace?.stages,
    state.workspace?.scenarios,
    state.workspace?.collections,
    state.workspace?.manifest_refs,
    state.workspace?.artifact_refs,
    state.workspace?.validation,
    state.workspace ? workspaceUiStatePersistenceSignature(state.workspace.ui_state) : null,
  ]);
}

// Subscribe to graphStore graph name changes to keep active tab label in sync.
// Manual deduplication: only call updateActiveTabLabel when the name actually changes.
let _lastGraphName = useGraphStore.getState().graph.metadata?.name ?? '';
useGraphStore.subscribe((state) => {
  const name = state.graph.metadata?.name ?? '';
  if (name && name !== _lastGraphName) {
    _lastGraphName = name;
    useProjectsStore.getState().updateActiveTabLabel(name);
  }
});
