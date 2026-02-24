import { create } from 'zustand';
import type {
  TrajectoryDataset,
  TrajectoryMetadata,
  TrajectoryData,
  PlaybackState,
} from '@/types/trajectory';
import {
  fetchTrajectoryDatasets,
  fetchTrajectoryMetadata,
  fetchTrajectory,
  filterTrajectories,
} from '@/api/client';

interface TrajectoryStoreState {
  // Data
  datasets: TrajectoryDataset[];
  activeDataset: string | null;
  metadata: TrajectoryMetadata | null;
  filteredIndices: number[] | null;
  activeIndex: number | null;
  trajectoryData: TrajectoryData | null;
  loading: boolean;
  error: string | null;

  // Playback
  playback: PlaybackState;

  // Filters
  filterBodyIdx: number | null;
  filterTaskType: number | null;

  // Actions
  loadDatasets: () => Promise<void>;
  selectDataset: (name: string) => Promise<void>;
  applyFilter: (bodyIdx: number | null, taskType: number | null) => Promise<void>;
  selectTrajectory: (index: number) => Promise<void>;

  // Playback actions
  setFrame: (frame: number) => void;
  togglePlay: () => void;
  setSpeed: (speed: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
}

const defaultPlayback: PlaybackState = {
  playing: false,
  speed: 1,
  frame: 0,
  totalFrames: 0,
};

export const useTrajectoryStore = create<TrajectoryStoreState>((set, get) => ({
  // Data
  datasets: [],
  activeDataset: null,
  metadata: null,
  filteredIndices: null,
  activeIndex: null,
  trajectoryData: null,
  loading: false,
  error: null,

  // Playback
  playback: { ...defaultPlayback },

  // Filters
  filterBodyIdx: null,
  filterTaskType: null,

  // Actions
  loadDatasets: async () => {
    set({ loading: true, error: null });
    try {
      const datasets = await fetchTrajectoryDatasets();
      set({ datasets, loading: false });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  selectDataset: async (name: string) => {
    set({
      activeDataset: name,
      metadata: null,
      filteredIndices: null,
      activeIndex: null,
      trajectoryData: null,
      loading: true,
      error: null,
      playback: { ...defaultPlayback },
    });
    try {
      const metadata = await fetchTrajectoryMetadata(name);
      set({ metadata, loading: false });
      // Apply current filters after loading metadata
      const { filterBodyIdx, filterTaskType } = get();
      await get().applyFilter(filterBodyIdx, filterTaskType);
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  applyFilter: async (bodyIdx: number | null, taskType: number | null) => {
    const { activeDataset } = get();
    if (!activeDataset) return;

    set({
      filterBodyIdx: bodyIdx,
      filterTaskType: taskType,
      loading: true,
      error: null,
    });

    try {
      const filters: { body_idx?: number; task_type?: number } = {};
      if (bodyIdx !== null) filters.body_idx = bodyIdx;
      if (taskType !== null) filters.task_type = taskType;

      const result = await filterTrajectories(activeDataset, filters);
      set({ filteredIndices: result.indices, loading: false });

      // Auto-select first trajectory from filtered results
      if (result.indices.length > 0) {
        await get().selectTrajectory(result.indices[0]);
      } else {
        set({ activeIndex: null, trajectoryData: null });
      }
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  selectTrajectory: async (index: number) => {
    const { activeDataset } = get();
    if (!activeDataset) return;

    set({ activeIndex: index, loading: true, error: null });
    try {
      const data = await fetchTrajectory(activeDataset, index);
      set({
        trajectoryData: data,
        loading: false,
        playback: {
          ...defaultPlayback,
          totalFrames: data.timestamps.length,
        },
      });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  // Playback actions
  setFrame: (frame: number) => {
    set((state) => ({
      playback: { ...state.playback, frame },
    }));
  },

  togglePlay: () => {
    set((state) => ({
      playback: { ...state.playback, playing: !state.playback.playing },
    }));
  },

  setSpeed: (speed: number) => {
    set((state) => ({
      playback: { ...state.playback, speed },
    }));
  },

  stepForward: () => {
    set((state) => {
      const next = Math.min(state.playback.frame + 1, state.playback.totalFrames - 1);
      return { playback: { ...state.playback, frame: next, playing: false } };
    });
  },

  stepBackward: () => {
    set((state) => {
      const prev = Math.max(state.playback.frame - 1, 0);
      return { playback: { ...state.playback, frame: prev, playing: false } };
    });
  },
}));
