export type {
  DatasetInfo as TrajectoryDataset,
  FilterResult,
  TrajectoryData,
  TrajectoryMetadata,
} from '@/generated/studioContracts';

export interface PlaybackState {
  playing: boolean;
  speed: number;
  frame: number;
  totalFrames: number;
}

export const TASK_TYPE_LABELS: Record<number, string> = {
  0: 'Reach',
  1: 'Hold',
  2: 'Track',
  3: 'Swing',
};
