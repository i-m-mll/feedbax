import type {
  TrainingTrajectoryPayload,
  StudioTrajectoryTrack,
} from '@/generated/studioContracts';

export const LIVE_TRAINING_TRAJECTORY_SCHEMA_ID = 'feedbax.event.studio.training_trajectory';
export const LIVE_TRAINING_TRAJECTORY_SCHEMA_VERSION =
  'feedbax.event.studio.training_trajectory.v1';

export interface LiveTrainingFrame {
  tracks: StudioTrajectoryTrack[];
}

function primaryEffectorTrack(tracks: StudioTrajectoryTrack[]): StudioTrajectoryTrack | null {
  return (
    tracks.find((track) => track.selector.role !== 'target' && track.anchor_id === 'effector') ??
    tracks.find((track) => track.selector.role !== 'target' && track.selector.compact.includes('effector')) ??
    tracks.find((track) => track.selector.role !== 'target' && track.samples.length > 0) ??
    null
  );
}

export function liveTrainingTargetTrack(
  frame: Pick<LiveTrainingFrame, 'tracks'> | null | undefined
): StudioTrajectoryTrack | null {
  return (
    frame?.tracks.find((track) => track.selector.role === 'target') ??
    frame?.tracks.find((track) => track.selector.compact.includes('target')) ??
    null
  );
}

export function liveTrainingEffectorTrack(
  frame: Pick<LiveTrainingFrame, 'tracks'> | null | undefined
): StudioTrajectoryTrack | null {
  return primaryEffectorTrack(frame?.tracks ?? []);
}

export function normalizeTrainingTrajectoryPayload(
  payload: TrainingTrajectoryPayload,
): LiveTrainingFrame {
  if (payload.schema_id && payload.schema_id !== LIVE_TRAINING_TRAJECTORY_SCHEMA_ID) {
    throw new Error(`Unsupported training trajectory schema_id '${payload.schema_id}'.`);
  }
  if (payload.schema_version && payload.schema_version !== LIVE_TRAINING_TRAJECTORY_SCHEMA_VERSION) {
    throw new Error(`Unsupported training trajectory schema_version '${payload.schema_version}'.`);
  }

  return { tracks: Object.values(payload.tracks ?? {}) };
}
