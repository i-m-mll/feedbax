// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createJSONStorage } from 'zustand/middleware';
import { useLayoutStore } from '@/stores/layoutStore';
import { useTrajectoryStore } from '@/stores/trajectoryStore';
import { useStatisticsStore } from '@/stores/statisticsStore';

function storedState<T>(key: string): T {
  const raw = localStorage.getItem(key);
  expect(raw).not.toBeNull();
  return JSON.parse(raw as string).state as T;
}

function makeStorage(): Storage {
  const values = new Map<string, string>();
  return {
    get length() {
      return values.size;
    },
    clear: () => values.clear(),
    getItem: (key) => values.get(key) ?? null,
    key: (index) => [...values.keys()][index] ?? null,
    removeItem: (key) => {
      values.delete(key);
    },
    setItem: (key, value) => {
      values.set(key, value);
    },
  };
}

describe('Studio preference persistence', () => {
  beforeEach(() => {
    const storage = makeStorage();
    vi.stubGlobal('localStorage', storage);
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      value: storage,
    });
    useLayoutStore.persist.setOptions({ storage: createJSONStorage(() => storage) });
    useTrajectoryStore.persist.setOptions({ storage: createJSONStorage(() => storage) });
    useStatisticsStore.persist.setOptions({ storage: createJSONStorage(() => storage) });
  });

  it('persists layout-level Studio UI preferences', () => {
    const layout = useLayoutStore.getState();

    layout.setBottomShelfMode('console');
    layout.setComponentLibraryExpandedCategories(['Math', 'Structure']);
    layout.setAnalysisLibraryExpandedCategories(['Computation']);
    layout.setSubgraphPreviewExpanded('controller', true);

    expect(storedState<Record<string, unknown>>('feedbax-studio-layout')).toMatchObject({
      bottomShelfMode: 'console',
      componentLibraryExpandedCategories: ['Math', 'Structure'],
      analysisLibraryExpandedCategories: ['Computation'],
      subgraphPreviewExpanded: { controller: true },
    });
  });

  it('persists trajectory preferences without fetched payloads', () => {
    useTrajectoryStore.setState({
      datasets: [{ name: 'scratch', file_size: 128, modified: 1 }],
      activeDataset: 'scratch',
      activeIndex: 3,
      filterBodyIdx: 1,
      filterTaskType: 2,
    });
    useTrajectoryStore.getState().setSpeed(1.5);
    useTrajectoryStore.getState().toggleTargetTrace();

    const persisted = storedState<Record<string, unknown>>('feedbax-studio-trajectory');
    expect(persisted).toMatchObject({
      activeDataset: 'scratch',
      activeIndex: 3,
      filterBodyIdx: 1,
      filterTaskType: 2,
      playbackSpeed: 1.5,
      showTargetTrace: false,
    });
    expect(persisted.datasets).toBeUndefined();
    expect(persisted.trajectoryData).toBeUndefined();
  });

  it('persists statistics controls and panel tabs without chart data', () => {
    const statistics = useStatisticsStore.getState();

    statistics.setGroupBy('body_idx');
    statistics.setSelectedMetric('effort');
    statistics.setScatterMetrics('final_distance', 'path_length');
    statistics.setActiveSubTab('charts');
    statistics.setActiveChartSubTab('scatter');
    useStatisticsStore.setState({
      summaryData: { dataset: 'scratch', group_by: 'none', groups: [] },
    });

    const persisted = storedState<Record<string, unknown>>('feedbax-studio-statistics');
    expect(persisted).toMatchObject({
      groupBy: 'body_idx',
      selectedMetric: 'effort',
      scatterXMetric: 'final_distance',
      scatterYMetric: 'path_length',
      activeSubTab: 'charts',
      activeChartSubTab: 'scatter',
    });
    expect(persisted.summaryData).toBeUndefined();
    expect(persisted.timeseriesData).toBeUndefined();
  });
});
