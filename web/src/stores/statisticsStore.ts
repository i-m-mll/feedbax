import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import type {
  StatisticsResponse,
  TimeseriesResponse,
  HistogramResponse,
  ScatterResponse,
  DiagnosticsResponse,
} from '@/types/statistics';
import {
  fetchStatsSummary,
  fetchStatsTimeseries,
  fetchStatsHistogram,
  fetchStatsScatter,
  fetchStatsDiagnostics,
} from '@/api/client';
import { withStoreActionFeedback } from '@/stores/storeActions';
import { useTrajectoryStore } from '@/stores/trajectoryStore';

export type StatisticsSubTab = 'overview' | 'charts' | 'diagnostics';
export type StatisticsChartSubTab = 'timeseries' | 'histogram' | 'scatter';

interface StatisticsStoreState {
  // Settings
  groupBy: string;
  selectedMetric: string;
  scatterXMetric: string;
  scatterYMetric: string;
  activeSubTab: StatisticsSubTab;
  activeChartSubTab: StatisticsChartSubTab;

  // Data
  summaryData: StatisticsResponse | null;
  timeseriesData: TimeseriesResponse | null;
  histogramData: HistogramResponse | null;
  scatterData: ScatterResponse | null;
  diagnosticsData: DiagnosticsResponse | null;

  // Loading / error
  loading: boolean;
  error: string | null;

  // Actions
  setGroupBy: (groupBy: string) => void;
  setSelectedMetric: (metric: string) => void;
  setScatterMetrics: (x: string, y: string) => void;
  setActiveSubTab: (tab: StatisticsSubTab) => void;
  setActiveChartSubTab: (tab: StatisticsChartSubTab) => void;
  loadSummary: () => Promise<void>;
  loadTimeseries: () => Promise<void>;
  loadHistogram: () => Promise<void>;
  loadScatter: () => Promise<void>;
  loadDiagnostics: () => Promise<void>;
}

type PersistedStatisticsState = Pick<
  StatisticsStoreState,
  | 'groupBy'
  | 'selectedMetric'
  | 'scatterXMetric'
  | 'scatterYMetric'
  | 'activeSubTab'
  | 'activeChartSubTab'
>;

const DEFAULT_PERSISTED_STATISTICS: PersistedStatisticsState = {
  groupBy: 'none',
  selectedMetric: 'distance_to_target',
  scatterXMetric: 'final_distance',
  scatterYMetric: 'effort',
  activeSubTab: 'overview',
  activeChartSubTab: 'timeseries',
};

export const useStatisticsStore = create<StatisticsStoreState>()(
  persist(
    (set, get) => ({
  // Settings
  groupBy: 'none',
  selectedMetric: 'distance_to_target',
  scatterXMetric: 'final_distance',
  scatterYMetric: 'effort',
  activeSubTab: 'overview',
  activeChartSubTab: 'timeseries',

  // Data
  summaryData: null,
  timeseriesData: null,
  histogramData: null,
  scatterData: null,
  diagnosticsData: null,

  // Loading / error
  loading: false,
  error: null,

  // Actions
  setGroupBy: (groupBy: string) => {
    // Bug: 4cb86c8 — clear cached chart data so stale groupBy results don't linger
    set({
      groupBy,
      timeseriesData: null,
      histogramData: null,
      scatterData: null,
    });
  },

  setSelectedMetric: (metric: string) => {
    set({ selectedMetric: metric });
  },

  setScatterMetrics: (x: string, y: string) => {
    set({ scatterXMetric: x, scatterYMetric: y });
  },

  setActiveSubTab: (tab) => {
    set({ activeSubTab: tab });
  },

  setActiveChartSubTab: (tab) => {
    set({ activeChartSubTab: tab });
  },

  loadSummary: async () => {
    const dataset = useTrajectoryStore.getState().activeDataset;
    if (!dataset) return;

    const { groupBy } = get();
    set({ loading: true, error: null });
    const summaryData = await withStoreActionFeedback(
      () => fetchStatsSummary(dataset, groupBy),
      {
        errorToast: 'Failed to load statistics summary.',
        toastId: 'stats-summary-load-error',
        onError: (err) => set({ error: String(err), loading: false }),
      },
    );
    if (!summaryData) return;
    // Bug: 4cb86c8 - discard stale response if params changed during fetch
    const current = get();
    if (current.groupBy !== groupBy || useTrajectoryStore.getState().activeDataset !== dataset) {
      set({ loading: false });
      return;
    }
    set({ summaryData, loading: false });
  },

  loadTimeseries: async () => {
    const dataset = useTrajectoryStore.getState().activeDataset;
    if (!dataset) return;

    const { selectedMetric, groupBy } = get();
    set({ loading: true, error: null });
    const timeseriesData = await withStoreActionFeedback(
      () => fetchStatsTimeseries(dataset, selectedMetric, groupBy),
      {
        errorToast: 'Failed to load timeseries statistics.',
        toastId: 'stats-timeseries-load-error',
        onError: (err) => set({ error: String(err), loading: false }),
      },
    );
    if (!timeseriesData) return;
    const current = get();
    if (current.groupBy !== groupBy || current.selectedMetric !== selectedMetric || useTrajectoryStore.getState().activeDataset !== dataset) {
      set({ loading: false });
      return;
    }
    set({ timeseriesData, loading: false });
  },

  loadHistogram: async () => {
    const dataset = useTrajectoryStore.getState().activeDataset;
    if (!dataset) return;

    const { selectedMetric, groupBy } = get();
    set({ loading: true, error: null });
    const histogramData = await withStoreActionFeedback(
      () => fetchStatsHistogram(dataset, selectedMetric, groupBy),
      {
        errorToast: 'Failed to load histogram statistics.',
        toastId: 'stats-histogram-load-error',
        onError: (err) => set({ error: String(err), loading: false }),
      },
    );
    if (!histogramData) return;
    const current = get();
    if (current.groupBy !== groupBy || current.selectedMetric !== selectedMetric || useTrajectoryStore.getState().activeDataset !== dataset) {
      set({ loading: false });
      return;
    }
    set({ histogramData, loading: false });
  },

  loadScatter: async () => {
    const dataset = useTrajectoryStore.getState().activeDataset;
    if (!dataset) return;

    const { scatterXMetric, scatterYMetric } = get();
    set({ loading: true, error: null });
    const scatterData = await withStoreActionFeedback(
      () => fetchStatsScatter(dataset, scatterXMetric, scatterYMetric),
      {
        errorToast: 'Failed to load scatter statistics.',
        toastId: 'stats-scatter-load-error',
        onError: (err) => set({ error: String(err), loading: false }),
      },
    );
    if (!scatterData) return;
    const current = get();
    if (current.scatterXMetric !== scatterXMetric || current.scatterYMetric !== scatterYMetric || useTrajectoryStore.getState().activeDataset !== dataset) {
      set({ loading: false });
      return;
    }
    set({ scatterData, loading: false });
  },

  loadDiagnostics: async () => {
    const dataset = useTrajectoryStore.getState().activeDataset;
    if (!dataset) return;

    set({ loading: true, error: null });
    const diagnosticsData = await withStoreActionFeedback(
      () => fetchStatsDiagnostics(dataset),
      {
        errorToast: 'Failed to load diagnostics.',
        toastId: 'stats-diagnostics-load-error',
        onError: (err) => set({ error: String(err), loading: false }),
      },
    );
    if (!diagnosticsData) return;
    if (useTrajectoryStore.getState().activeDataset !== dataset) {
      set({ loading: false });
      return;
    }
    set({ diagnosticsData, loading: false });
  },
    }),
    {
      name: 'feedbax-studio-statistics',
      storage: createJSONStorage(() => window.localStorage),
      version: 1,
      migrate: (persistedState): PersistedStatisticsState => {
        const persisted =
          persistedState && typeof persistedState === 'object'
            ? (persistedState as Partial<PersistedStatisticsState>)
            : {};
        return {
          ...DEFAULT_PERSISTED_STATISTICS,
          ...persisted,
          activeSubTab:
            persisted.activeSubTab === 'charts' || persisted.activeSubTab === 'diagnostics'
              ? persisted.activeSubTab
              : DEFAULT_PERSISTED_STATISTICS.activeSubTab,
          activeChartSubTab:
            persisted.activeChartSubTab === 'histogram' ||
            persisted.activeChartSubTab === 'scatter'
              ? persisted.activeChartSubTab
              : DEFAULT_PERSISTED_STATISTICS.activeChartSubTab,
        };
      },
      partialize: (state) => ({
        groupBy: state.groupBy,
        selectedMetric: state.selectedMetric,
        scatterXMetric: state.scatterXMetric,
        scatterYMetric: state.scatterYMetric,
        activeSubTab: state.activeSubTab,
        activeChartSubTab: state.activeChartSubTab,
      }),
    },
  )
);

// Subscribe to activeDataset changes from trajectoryStore —
// auto-load summary + diagnostics when dataset changes.
let _prevDataset: string | null = null;
const _unsubDataset = useTrajectoryStore.subscribe((state) => {
  const activeDataset = state.activeDataset;
  if (activeDataset && activeDataset !== _prevDataset) {
    _prevDataset = activeDataset;
    // Reset data when dataset changes
    useStatisticsStore.setState({
      summaryData: null,
      timeseriesData: null,
      histogramData: null,
      scatterData: null,
      diagnosticsData: null,
    });
    useStatisticsStore.getState().loadSummary();
    useStatisticsStore.getState().loadDiagnostics();
  } else if (!activeDataset) {
    _prevDataset = null;
  }
});

// Bug: 4cb86c8 — prevent subscription stacking on HMR reload
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    _unsubDataset();
    _prevDataset = null;
  });
}
