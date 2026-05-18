import { create } from 'zustand';
import { useGraphStore, createInitialGraph, createBlankGraph, type GraphSnapshot, type GraphLayer, type StateMergeRequest } from '@/stores/graphStore';
import { useTrainingStore, defaultTrainingSpec, defaultTaskSpec } from '@/stores/trainingStore';
import { useTrajectoryStore } from '@/stores/trajectoryStore';
import { useStatisticsStore } from '@/stores/statisticsStore';
import { useAnalysisStore } from '@/stores/analysisStore';
import { buildWorkspaceSnapshot, useWorkspaceStore } from '@/stores/workspaceStore';
import type { TrainingSpec, TaskSpec, LossValidationError } from '@/types/training';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { AnalysisSnapshot } from '@/types/analysis';
import type { StudioWorkspaceSpec } from '@/types/workspace';

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
}

function captureGraphSnapshot(): GraphSnapshot {
  const s = useGraphStore.getState();
  return {
    graph: s.graph,
    uiState: s.uiState,
    graphId: s.graphId,
    isDirty: s.isDirty,
    lastSavedAt: s.lastSavedAt,
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
    isDirty: false,
    lastSavedAt: null,
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
    isDirty: false,
    lastSavedAt: null,
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
  return buildWorkspaceSnapshot({
    workspace: useWorkspaceStore.getState().workspace,
    graph: graphSnapshot.graph,
    uiState: graphSnapshot.uiState,
    trainingSpec: trainingSnapshot.trainingSpec,
    taskSpec: trainingSnapshot.taskSpec,
    analysisSnapshot,
    projectName: graphSnapshot.currentGraphLabel,
  });
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
  };
}

interface ProjectsStoreState {
  tabs: OpenTab[];
  activeTabId: string;
  openNewTab: (name: string) => void;
  openProjectInTab: (
    graphId: string,
    graph: GraphSpec,
    uiState: GraphUIState,
    projectName?: string,
    analysisSnapshot?: AnalysisSnapshot | null,
    workspaceSnapshot?: StudioWorkspaceSpec | null,
  ) => void;
  switchTab: (tabId: string) => void;
  closeTab: (tabId: string) => void;
  updateActiveTabLabel: (label: string) => void;
  renameTab: (tabId: string, name: string) => void;
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
  useWorkspaceStore.getState().setWorkspace(workspaceSnapshot);
  return {
    tabId: generateTabId(),
    label: graphSnapshot.currentGraphLabel || 'Model',
    graphSnapshot,
    trainingSnapshot,
    analysisSnapshot,
    workspaceSnapshot,
  };
}

export const useProjectsStore = create<ProjectsStoreState>((set, get) => {
  const firstTab = buildInitialTab();

  return {
    tabs: [firstTab],
    activeTabId: firstTab.tabId,

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
      useWorkspaceStore.getState().setWorkspace(newWorkspaceSnapshot);
      resetTrajectoryStoreForTabSwitch();
      resetStatisticsStoreForTabSwitch();

      set({ tabs: [...updatedTabs, newTab], activeTabId: newTab.tabId });
    },

    openProjectInTab: (
      graphId,
      graph,
      uiState,
      projectName,
      analysisSnapshot,
      workspaceSnapshot,
    ) => {
      const { tabs, activeTabId } = get();
      const updatedTabs = tabs.map((tab) =>
        tab.tabId === activeTabId ? captureCurrentTab(tab) : tab
      );

      const graphSnapshot: GraphSnapshot = {
        graph,
        uiState,
        graphId,
        isDirty: false,
        lastSavedAt: null,
        graphStack: [],
        currentGraphLabel: projectName ?? graph.metadata?.name ?? 'Untitled',
        currentContext: 'top-level',
        edgeStyle: 'bezier',
        past: [],
        future: [],
        selectedTapId: null,
        selectedEdgeId: null,
        pendingStateMerge: null,
      };
      const trainingSnapshot = trainingSnapshotFromWorkspace(workspaceSnapshot);
      const restoredAnalysis = analysisSnapshot ?? makeInitialAnalysisSnapshot();
      const restoredWorkspace = buildWorkspaceSnapshot({
        workspace: workspaceSnapshot ?? null,
        graph,
        uiState,
        trainingSpec: trainingSnapshot.trainingSpec,
        taskSpec: trainingSnapshot.taskSpec,
        analysisSnapshot: restoredAnalysis,
        projectName: graphSnapshot.currentGraphLabel,
      });
      const newTab: OpenTab = {
        tabId: generateTabId(),
        label: graphSnapshot.currentGraphLabel,
        graphSnapshot,
        trainingSnapshot,
        analysisSnapshot: restoredAnalysis,
        workspaceSnapshot: restoredWorkspace,
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
      useWorkspaceStore.getState().setWorkspace(restoredWorkspace);
      resetTrajectoryStoreForTabSwitch();
      resetStatisticsStoreForTabSwitch();

      set({ tabs: [...updatedTabs, newTab], activeTabId: newTab.tabId });
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
      useWorkspaceStore.getState().setWorkspace(target.workspaceSnapshot);
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
        useWorkspaceStore.getState().setWorkspace(nextTab.workspaceSnapshot);
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
          isDirty: true,
        });
      }
    },
  };
});

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
