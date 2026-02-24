import { create } from 'zustand';
import type {
  LossTermSpec,
  LossValidationError,
  ProbeInfo,
  TimeAggregationSpec,
  TrainingSpec,
  TaskSpec,
  TrainingProgress,
} from '@/types/training';

export type TrainingStatus = 'idle' | 'running' | 'paused' | 'completed' | 'error';

const defaultTrainingSpec: TrainingSpec = {
  optimizer: {
    type: 'adam',
    params: {
      learning_rate: 0.001,
      b1: 0.9,
      b2: 0.999,
    },
  },
  loss: {
    type: 'Composite',
    label: 'reach_loss',
    weight: 1.0,
    children: {
      position: {
        type: 'TargetStateLoss',
        label: 'Effector Position',
        weight: 1.0,
        selector: 'probe:effector_pos',
        norm: 'squared_l2',
        time_agg: {
          mode: 'all',
          discount: 'power',
          discount_exp: 6,
        },
      },
      final_velocity: {
        type: 'TargetStateLoss',
        label: 'Final Velocity',
        weight: 0.5,
        selector: 'probe:effector_vel',
        norm: 'squared_l2',
        time_agg: {
          mode: 'final',
        },
      },
      regularization: {
        type: 'TargetStateLoss',
        label: 'Network Activity',
        weight: 0.01,
        selector: 'probe:network_hidden',
        norm: 'squared_l2',
        time_agg: {
          mode: 'all',
        },
      },
    },
  },
  n_batches: 1000,
  batch_size: 64,
};

const defaultTaskSpec: TaskSpec = {
  type: 'ReachingTask',
  params: {
    n_targets: 8,
    target_radius: 0.02,
  },
  timeline: {
    epochs: {
      pre_movement: [0, 50],
      movement: [50, 150],
      hold: [150, 200],
    },
  },
};

interface TrainingStoreState {
  trainingSpec: TrainingSpec;
  taskSpec: TaskSpec;
  status: TrainingStatus;
  jobId: string | null;
  progress: TrainingProgress | null;
  // Loss UI state
  availableProbes: ProbeInfo[];
  selectedLossPath: string[] | null;
  lossValidationErrors: LossValidationError[];
  highlightedProbeSelector: string | null;
  // Actions
  setTrainingSpec: (spec: Partial<TrainingSpec>) => void;
  setTaskSpec: (spec: Partial<TaskSpec>) => void;
  setStatus: (status: TrainingStatus) => void;
  setJobId: (jobId: string | null) => void;
  setProgress: (progress: TrainingProgress | null) => void;
  // Loss actions
  setAvailableProbes: (probes: ProbeInfo[]) => void;
  setSelectedLossPath: (path: string[] | null) => void;
  setLossValidationErrors: (errors: LossValidationError[]) => void;
  setHighlightedProbeSelector: (selector: string | null) => void;
  updateLossTerm: (path: string[], updates: Partial<LossTermSpec>) => void;
  addLossTerm: (parentPath: string[], key: string, term: LossTermSpec) => void;
  removeLossTerm: (path: string[]) => void;
}

export const useTrainingStore = create<TrainingStoreState>((set) => ({
  trainingSpec: defaultTrainingSpec,
  taskSpec: defaultTaskSpec,
  status: 'idle',
  jobId: null,
  progress: null,
  // Loss UI state
  availableProbes: [],
  selectedLossPath: null,
  lossValidationErrors: [],
  highlightedProbeSelector: null,
  // Actions
  setTrainingSpec: (spec) =>
    set((state) => ({
      trainingSpec: {
        ...state.trainingSpec,
        ...spec,
        optimizer: {
          ...state.trainingSpec.optimizer,
          ...(spec.optimizer ?? {}),
        },
      },
    })),
  setTaskSpec: (spec) =>
    set((state) => ({
      taskSpec: {
        ...state.taskSpec,
        ...spec,
      },
    })),
  setStatus: (status) => set({ status }),
  setJobId: (jobId) => set({ jobId }),
  setProgress: (progress) => set({ progress }),
  // Loss actions
  setAvailableProbes: (probes) => set({ availableProbes: probes }),
  setSelectedLossPath: (path) => set({ selectedLossPath: path }),
  setLossValidationErrors: (errors) => set({ lossValidationErrors: errors }),
  setHighlightedProbeSelector: (selector) => set({ highlightedProbeSelector: selector }),
  updateLossTerm: (path, updates) =>
    set((state) => ({
      trainingSpec: {
        ...state.trainingSpec,
        loss: updateLossTermAtPath(state.trainingSpec.loss, path, updates),
      },
    })),
  addLossTerm: (parentPath, key, term) =>
    set((state) => ({
      trainingSpec: {
        ...state.trainingSpec,
        loss: addLossTermAtPath(state.trainingSpec.loss, parentPath, key, term),
      },
    })),
  removeLossTerm: (path) =>
    set((state) => ({
      trainingSpec: {
        ...state.trainingSpec,
        loss: removeLossTermAtPath(state.trainingSpec.loss, path),
      },
    })),
}));

// Helper functions for loss term manipulation

function updateLossTermAtPath(
  term: LossTermSpec,
  path: string[],
  updates: Partial<LossTermSpec>
): LossTermSpec {
  if (path.length === 0) {
    return { ...term, ...updates };
  }
  if (!term.children) return term;
  const [head, ...rest] = path;
  const child = term.children[head];
  if (!child) return term;
  return {
    ...term,
    children: {
      ...term.children,
      [head]: updateLossTermAtPath(child, rest, updates),
    },
  };
}

function addLossTermAtPath(
  term: LossTermSpec,
  parentPath: string[],
  key: string,
  newTerm: LossTermSpec
): LossTermSpec {
  if (parentPath.length === 0) {
    return {
      ...term,
      children: {
        ...(term.children ?? {}),
        [key]: newTerm,
      },
    };
  }
  if (!term.children) return term;
  const [head, ...rest] = parentPath;
  const child = term.children[head];
  if (!child) return term;
  return {
    ...term,
    children: {
      ...term.children,
      [head]: addLossTermAtPath(child, rest, key, newTerm),
    },
  };
}

function removeLossTermAtPath(term: LossTermSpec, path: string[]): LossTermSpec {
  if (path.length === 0) {
    // Cannot remove root
    return term;
  }
  if (path.length === 1) {
    if (!term.children) return term;
    const { [path[0]]: _removed, ...remaining } = term.children;
    return {
      ...term,
      children: Object.keys(remaining).length > 0 ? remaining : undefined,
    };
  }
  if (!term.children) return term;
  const [head, ...rest] = path;
  const child = term.children[head];
  if (!child) return term;
  return {
    ...term,
    children: {
      ...term.children,
      [head]: removeLossTermAtPath(child, rest),
    },
  };
}
