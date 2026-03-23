import { create } from 'zustand';
import type { TrainingRun, EvalRun } from '@/types/runs';
import { fetchTrainingRuns, fetchEvalRuns } from '@/api/runAPI';

interface RunStoreState {
  /** All known training runs. */
  trainingRuns: TrainingRun[];
  /** Eval runs for the currently selected training run. */
  evalRuns: EvalRun[];
  /** Currently selected training run ID (null = none selected). */
  selectedTrainingRunId: string | null;
  /** Currently selected evaluation run ID (null = none selected). */
  selectedEvalRunId: string | null;
  /** Whether runs are being loaded. */
  loading: boolean;

  // Actions
  loadTrainingRuns: () => Promise<void>;
  selectTrainingRun: (id: string | null) => Promise<void>;
  selectEvalRun: (id: string | null) => void;
  addTrainingRun: (run: TrainingRun) => void;
}

export const useRunStore = create<RunStoreState>((set, get) => ({
  trainingRuns: [],
  evalRuns: [],
  selectedTrainingRunId: null,
  selectedEvalRunId: null,
  loading: false,

  loadTrainingRuns: async () => {
    set({ loading: true });
    try {
      const runs = await fetchTrainingRuns();
      set({ trainingRuns: runs, loading: false });
      // Auto-select the first run if none is selected
      if (runs.length > 0 && get().selectedTrainingRunId === null) {
        await get().selectTrainingRun(runs[0].id);
      }
    } catch {
      set({ loading: false });
    }
  },

  selectTrainingRun: async (id) => {
    set({ selectedTrainingRunId: id, selectedEvalRunId: null, evalRuns: [] });
    if (id === null) return;
    try {
      const evals = await fetchEvalRuns(id);
      set({ evalRuns: evals });
      // Auto-select first eval run
      if (evals.length > 0) {
        set({ selectedEvalRunId: evals[0].id });
      }
    } catch {
      // eval fetch failed silently
    }
  },

  selectEvalRun: (id) => {
    set({ selectedEvalRunId: id });
  },

  addTrainingRun: (run) => {
    set((state) => ({
      trainingRuns: [run, ...state.trainingRuns],
    }));
  },
}));
