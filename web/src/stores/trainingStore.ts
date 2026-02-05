import { create } from 'zustand';
import type { TrainingSpec, TaskSpec, TrainingProgress } from '@/types/training';

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
  setTrainingSpec: (spec: Partial<TrainingSpec>) => void;
  setTaskSpec: (spec: Partial<TaskSpec>) => void;
  setStatus: (status: TrainingStatus) => void;
  setJobId: (jobId: string | null) => void;
  setProgress: (progress: TrainingProgress | null) => void;
}

export const useTrainingStore = create<TrainingStoreState>((set) => ({
  trainingSpec: defaultTrainingSpec,
  taskSpec: defaultTaskSpec,
  status: 'idle',
  jobId: null,
  progress: null,
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
}));
