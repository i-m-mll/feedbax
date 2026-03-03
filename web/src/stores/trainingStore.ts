import { create } from 'zustand';
import type {
  LossTermSpec,
  LossValidationError,
  ProbeInfo,
  TimeAggregationSpec,
  TrainingSpec,
  TaskSpec,
  TrainingProgress,
  TrainingLogLine,
} from '@/types/training';

export interface TrajectorySnapshot {
  batch: number;
  effector: [number, number][];
  target: [number, number];
  t: number[];
}

export type TrainingStatus = 'idle' | 'running' | 'paused' | 'completed' | 'error';
export type WorkerMode = 'local' | 'remote';

export const defaultTrainingSpec: TrainingSpec = {
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

export const defaultTaskSpec: TaskSpec = {
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

const MAX_LOSS_HISTORY = 2000;
const MAX_CONSOLE_LOGS = 5000;

interface TrainingStoreState {
  trainingSpec: TrainingSpec;
  taskSpec: TaskSpec;
  status: TrainingStatus;
  jobId: string | null;
  progress: TrainingProgress | null;
  lossHistory: TrainingProgress[];
  consoleLogs: TrainingLogLine[];
  // Trajectory snapshot streamed during training
  latestTrajectory: TrajectorySnapshot | null;
  // Loss UI state
  availableProbes: ProbeInfo[];
  selectedLossPath: string[] | null;
  lossValidationErrors: LossValidationError[];
  highlightedProbeSelector: string | null;
  // Remote worker state
  workerMode: WorkerMode;
  workerUrl: string | null;
  workerConnected: boolean;
  // Cloud orchestration state
  orchestrationStatus: string;
  orchestrationInstanceName: string | null;
  orchestrationWorkerUrl: string | null;
  // Actions
  setTrainingSpec: (spec: Partial<TrainingSpec>) => void;
  setTaskSpec: (spec: Partial<TaskSpec>) => void;
  setStatus: (status: TrainingStatus) => void;
  setJobId: (jobId: string | null) => void;
  setProgress: (progress: TrainingProgress | null) => void;
  appendProgress: (p: TrainingProgress) => void;
  appendLog: (l: TrainingLogLine) => void;
  clearHistory: () => void;
  setLatestTrajectory: (snapshot: TrajectorySnapshot | null) => void;
  // Loss actions
  setAvailableProbes: (probes: ProbeInfo[]) => void;
  setSelectedLossPath: (path: string[] | null) => void;
  setLossValidationErrors: (errors: LossValidationError[]) => void;
  setHighlightedProbeSelector: (selector: string | null) => void;
  updateLossTerm: (path: string[], updates: Partial<LossTermSpec>) => void;
  addLossTerm: (parentPath: string[], key: string, term: LossTermSpec) => void;
  removeLossTerm: (path: string[]) => void;
  // Worker actions
  setWorkerConfig: (mode: WorkerMode, url: string | null, connected: boolean) => void;
  // Orchestration actions
  setOrchestrationState: (
    status: string,
    instanceName: string | null,
    workerUrl: string | null
  ) => void;
}

export const useTrainingStore = create<TrainingStoreState>((set) => ({
  trainingSpec: defaultTrainingSpec,
  taskSpec: defaultTaskSpec,
  status: 'idle',
  jobId: null,
  progress: null,
  lossHistory: [],
  consoleLogs: [],
  latestTrajectory: null,
  // Loss UI state
  availableProbes: [],
  selectedLossPath: null,
  lossValidationErrors: [],
  highlightedProbeSelector: null,
  // Remote worker state
  workerMode: 'local',
  workerUrl: null,
  workerConnected: false,
  // Cloud orchestration state
  orchestrationStatus: 'idle',
  orchestrationInstanceName: null,
  orchestrationWorkerUrl: null,
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
  setProgress: (progress) => {
    set((state) => {
      if (progress === null) return { progress };
      const next = state.lossHistory.length >= MAX_LOSS_HISTORY
        ? [...state.lossHistory.slice(1), progress]
        : [...state.lossHistory, progress];
      return { progress, lossHistory: next };
    });
  },
  appendProgress: (p) =>
    set((state) => {
      const next = state.lossHistory.length >= MAX_LOSS_HISTORY
        ? [...state.lossHistory.slice(1), p]
        : [...state.lossHistory, p];
      return { lossHistory: next };
    }),
  appendLog: (l) =>
    set((state) => {
      const next = state.consoleLogs.length >= MAX_CONSOLE_LOGS
        ? [...state.consoleLogs.slice(1), l]
        : [...state.consoleLogs, l];
      return { consoleLogs: next };
    }),
  clearHistory: () => set({ lossHistory: [], consoleLogs: [], latestTrajectory: null }),
  setLatestTrajectory: (snapshot) => set({ latestTrajectory: snapshot }),
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
  // Worker actions
  setWorkerConfig: (mode, url, connected) =>
    set({ workerMode: mode, workerUrl: url, workerConnected: connected }),
  // Orchestration actions
  setOrchestrationState: (status, instanceName, workerUrl) =>
    set({
      orchestrationStatus: status,
      orchestrationInstanceName: instanceName,
      orchestrationWorkerUrl: workerUrl,
    }),
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
