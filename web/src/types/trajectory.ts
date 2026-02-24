export interface TrajectoryDataset {
  name: string;
  file_size: number;
  modified: number;
}

export interface TrajectoryMetadata {
  n_trajectories: number;
  n_timesteps: number;
  n_joints: number;
  n_muscles: number;
  n_bodies: number;
  rollouts_per_body: number;
  task_types: number[];
  body_indices: number[];
  angle_convention: string; // "relative_radians"
}

export interface TrajectoryData {
  timestamps: number[];
  joint_angles: number[][];   // [T][n_joints]
  muscle_activations: number[][];  // [T][n_muscles]
  effector_pos: number[][];   // [T][2]
  task_target: number[][];    // [T][2]
  body_preset_flat: number[];
  task_type: number;
  body_idx: number;
}

export interface PlaybackState {
  playing: boolean;
  speed: number;  // 0.25, 0.5, 1, 2
  frame: number;
  totalFrames: number;
}

export const TASK_TYPE_LABELS: Record<number, string> = {
  0: 'Reach',
  1: 'Hold',
  2: 'Track',
  3: 'Swing',
};
