import { create } from 'zustand';
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
import { useTrajectoryStore } from '@/stores/trajectoryStore';

interface StatisticsStoreState {
  // Settings
  groupBy: string;
  selectedMetric: string;
  scatterXMetric: string;
  scatterYMetric: string;

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
  loadSummary: () => Promise<void>;
  loadTimeseries: () => Promise<void>;
  loadHistogram: () => Promise<void>;
  loadScatter: () => Promise<void>;
  loadDiagnostics: () => Promise<void>;
}

export const useStatisticsStore = create<StatisticsStoreState>((set, get) => ({
  // Settings
  groupBy: 'none',
  selectedMetric: 'final_distance',
  scatterXMetric: 'final_distance',
  scatterYMetric: 'effort',

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
    set({ groupBy });
  },

  setSelectedMetric: (metric: string) => {
    set({ selectedMetric: metric });
  },

  setScatterMetrics: (x: string, y: string) => {
    set({ scatterXMetric: x, scatterYMetric: y });
  },

  loadSummary: async () => {
    const dataset = useTrajectoryStore.getState().activeDataset;
    if (!dataset) return;

    const { groupBy } = get();
    set({ loading: true, error: null });
    try {
      const summaryData = await fetchStatsSummary(dataset, groupBy);
      set({ summaryData, loading: false });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  loadTimeseries: async () => {
    const dataset = useTrajectoryStore.getState().activeDataset;
    if (!dataset) return;

    const { selectedMetric, groupBy } = get();
    set({ loading: true, error: null });
    try {
      const timeseriesData = await fetchStatsTimeseries(dataset, selectedMetric, groupBy);
      set({ timeseriesData, loading: false });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  loadHistogram: async () => {
    const dataset = useTrajectoryStore.getState().activeDataset;
    if (!dataset) return;

    const { selectedMetric, groupBy } = get();
    set({ loading: true, error: null });
    try {
      const histogramData = await fetchStatsHistogram(dataset, selectedMetric, groupBy);
      set({ histogramData, loading: false });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  loadScatter: async () => {
    const dataset = useTrajectoryStore.getState().activeDataset;
    if (!dataset) return;

    const { scatterXMetric, scatterYMetric } = get();
    set({ loading: true, error: null });
    try {
      const scatterData = await fetchStatsScatter(dataset, scatterXMetric, scatterYMetric);
      set({ scatterData, loading: false });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  loadDiagnostics: async () => {
    const dataset = useTrajectoryStore.getState().activeDataset;
    if (!dataset) return;

    set({ loading: true, error: null });
    try {
      const diagnosticsData = await fetchStatsDiagnostics(dataset);
      set({ diagnosticsData, loading: false });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },
}));

// Subscribe to activeDataset changes from trajectoryStore —
// auto-load summary + diagnostics when dataset changes.
let _prevDataset: string | null = null;
useTrajectoryStore.subscribe((state) => {
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
