import type {
  TrainingTrajectoryPayload,
  WorkspaceReplaySampleAxis,
  WorkspaceReplayTrack,
} from '@/generated/studioContracts';

export const LIVE_TRAINING_TRAJECTORY_SCHEMA_ID = 'feedbax.event.studio.training_trajectory';
export const LIVE_TRAINING_TRAJECTORY_SCHEMA_VERSION =
  'feedbax.event.studio.training_trajectory.v1';

export interface LiveTrainingFrame {
  batch: number;
  schema_id: typeof LIVE_TRAINING_TRAJECTORY_SCHEMA_ID;
  schema_version: typeof LIVE_TRAINING_TRAJECTORY_SCHEMA_VERSION;
  fidelity: 'lower_fidelity_live_snapshot';
  time: WorkspaceReplaySampleAxis;
  tracks: WorkspaceReplayTrack[];
  trackBySelector: Record<string, WorkspaceReplayTrack>;
  effector: [number, number][];
  target?: [number, number][] | [number, number] | null;
  t: number[];
  observables: Record<string, unknown>;
  outputs: Record<string, unknown>;
}

function finitePoint(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) return null;
  const x = Number(value[0]);
  const y = Number(value[1]);
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}

function pointsFromUnknown(value: unknown): Array<[number, number]> {
  const point = finitePoint(value);
  if (point) return [point];
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => finitePoint(item))
    .filter((item): item is [number, number] => item !== null);
}

function numericAxis(values: unknown, fallbackLength: number): number[] {
  if (Array.isArray(values)) {
    const axis = values.map(Number).filter(Number.isFinite);
    if (axis.length > 0) return axis;
  }
  return Array.from({ length: Math.max(1, fallbackLength) }, (_, index) => index);
}

function legacyTrack(
  selector: string,
  role: 'observed' | 'target',
  samples: Array<[number, number]>,
  label: string
): WorkspaceReplayTrack | null {
  if (samples.length === 0) return null;
  const [, targetId = selector] = selector.split(':');
  return {
    anchor_id: targetId,
    selector: {
      namespace: selector.startsWith('task_data:') ? 'task_data' : 'graph_output',
      compact: selector,
      target_id: targetId,
      role,
    },
    samples,
    dim: 2,
    dtype: 'float32',
    units: null,
    frame: 'world',
    label,
    metadata: { migrated_from: 'legacy_training_trajectory' },
  };
}

function tracksFromPayload(payload: TrainingTrajectoryPayload): Record<string, WorkspaceReplayTrack> {
  const tracks = { ...(payload.tracks ?? {}) };
  if (Object.keys(tracks).length > 0) return tracks;

  const effector = legacyTrack(
    'graph_output:effector',
    'observed',
    pointsFromUnknown(payload.effector),
    'Effector'
  );
  if (effector) tracks[effector.selector.compact] = effector;

  const targetSamples = pointsFromUnknown(payload.target);
  const target = legacyTrack('task_data:target', 'target', targetSamples, 'Target');
  if (target) tracks[target.selector.compact] = target;
  return tracks;
}

function primaryEffectorTrack(tracks: WorkspaceReplayTrack[]): WorkspaceReplayTrack | null {
  return (
    tracks.find((track) => track.selector.role !== 'target' && track.anchor_id === 'effector') ??
    tracks.find((track) => track.selector.role !== 'target' && track.selector.compact.includes('effector')) ??
    tracks.find((track) => track.selector.role !== 'target' && track.samples.length > 0) ??
    null
  );
}

export function liveTrainingTargetTrack(
  frame: Pick<LiveTrainingFrame, 'tracks'> | null | undefined
): WorkspaceReplayTrack | null {
  return (
    frame?.tracks.find((track) => track.selector.role === 'target') ??
    frame?.tracks.find((track) => track.selector.compact.includes('target')) ??
    null
  );
}

export function liveTrainingEffectorTrack(
  frame: Pick<LiveTrainingFrame, 'tracks'> | null | undefined
): WorkspaceReplayTrack | null {
  return primaryEffectorTrack(frame?.tracks ?? []);
}

export function normalizeTrainingTrajectoryPayload(
  payload: TrainingTrajectoryPayload,
  batch: number
): LiveTrainingFrame {
  if (payload.schema_id && payload.schema_id !== LIVE_TRAINING_TRAJECTORY_SCHEMA_ID) {
    throw new Error(`Unsupported training trajectory schema_id '${payload.schema_id}'.`);
  }
  if (payload.schema_version && payload.schema_version !== LIVE_TRAINING_TRAJECTORY_SCHEMA_VERSION) {
    throw new Error(`Unsupported training trajectory schema_version '${payload.schema_version}'.`);
  }

  const trackBySelector = tracksFromPayload(payload);
  const tracks = Object.values(trackBySelector);
  const effectorTrack = primaryEffectorTrack(tracks);
  const targetTrack = liveTrainingTargetTrack({ tracks });
  const effector = effectorTrack?.samples.map((point) => [point[0], point[1]] as [number, number]) ?? [];
  const targetSamples =
    targetTrack?.samples.map((point) => [point[0], point[1]] as [number, number]) ??
    pointsFromUnknown(payload.target);
  const length =
    payload.time?.length ??
    Math.max(effector.length, targetSamples.length, payload.n_steps ?? 0, 1);
  const t = payload.time?.values ?? numericAxis(payload.t, length);

  return {
    batch,
    schema_id: LIVE_TRAINING_TRAJECTORY_SCHEMA_ID,
    schema_version: LIVE_TRAINING_TRAJECTORY_SCHEMA_VERSION,
    fidelity: 'lower_fidelity_live_snapshot',
    time: payload.time ?? {
      length,
      units: 'step',
      values: t,
      metadata: { migrated_from: payload.schema_id ? 'selector_tracks' : 'legacy_training_trajectory' },
    },
    tracks,
    trackBySelector,
    effector,
    target: targetSamples.length > 1 ? targetSamples : targetSamples[0] ?? null,
    t,
    observables: payload.observables ?? {},
    outputs: payload.outputs ?? {},
  };
}
