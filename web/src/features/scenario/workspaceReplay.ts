import type { WorkspaceReplayTrack, WorkspaceReplayTrial } from '@/generated/studioContracts';

export interface WorkspaceReplayTimelineBand {
  id: string;
  label: string;
  start: number;
  end: number;
  kind: 'epoch' | 'loss_window';
}

export interface WorkspaceReplayEventTick {
  id: string;
  label: string;
  time: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

export function workspaceReplayTrialRef(trial: WorkspaceReplayTrial): string {
  if (trial.identity.stable_id) return `trial:${trial.identity.stable_id}`;
  return `trial:index:${trial.identity.index}`;
}

export function primaryWorkspaceReplayTrack(
  trial: WorkspaceReplayTrial | null | undefined
): WorkspaceReplayTrack | null {
  return (
    trial?.tracks?.find((track) => track.dim >= 2 && track.samples.length > 0) ??
    trial?.tracks?.[0] ??
    null
  );
}

export function workspaceReplayFrameTimes(trial: WorkspaceReplayTrial | null | undefined): number[] {
  if (!trial) return [0];
  const explicit = trial.time.values?.filter(Number.isFinite);
  if (explicit && explicit.length === trial.time.length) return explicit;
  const dt = typeof trial.time.dt === 'number' && Number.isFinite(trial.time.dt)
    ? trial.time.dt
    : 1;
  return Array.from({ length: Math.max(1, trial.time.length) }, (_, index) => index * dt);
}

export function workspaceReplayDuration(trial: WorkspaceReplayTrial | null | undefined): number {
  const times = workspaceReplayFrameTimes(trial);
  return times[times.length - 1] ?? 0;
}

export function workspaceReplayFrameIndex(trial: WorkspaceReplayTrial, position: number): number {
  const times = workspaceReplayFrameTimes(trial);
  const bounded = Math.max(0, Math.min(workspaceReplayDuration(trial), position));
  let best = 0;
  for (let index = 1; index < times.length; index += 1) {
    if (Math.abs(times[index] - bounded) < Math.abs(times[best] - bounded)) best = index;
  }
  return best;
}

export function workspaceReplaySampleAt(
  track: WorkspaceReplayTrack | null | undefined,
  frameIndex: number
): [number, number] | null {
  const sample = track?.samples[Math.max(0, Math.min(track.samples.length - 1, frameIndex))];
  if (!sample || sample.length < 2) return null;
  const x = Number(sample[0]);
  const y = Number(sample[1]);
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}

export function workspaceReplayPolyline(track: WorkspaceReplayTrack | null | undefined): Array<[number, number]> {
  if (!track) return [];
  return track.samples
    .map((sample) => {
      const x = Number(sample[0]);
      const y = Number(sample[1]);
      return Number.isFinite(x) && Number.isFinite(y) ? ([x, y] as [number, number]) : null;
    })
    .filter((point): point is [number, number] => point !== null);
}

function numberFromRecord(record: Record<string, unknown>, keys: string[]): number | null {
  for (const key of keys) {
    const value = Number(record[key]);
    if (Number.isFinite(value)) return value;
  }
  return null;
}

export function workspaceReplayTimelineBands(trial: WorkspaceReplayTrial | null | undefined): WorkspaceReplayTimelineBand[] {
  const timeline = trial?.trial_spec?.timeline;
  if (!isRecord(timeline) || !Array.isArray(timeline.epochs)) return [];
  let cursor = 0;
  return timeline.epochs.flatMap((epoch, index) => {
    if (!isRecord(epoch)) return [];
    const start = numberFromRecord(epoch, ['start', 'start_time', 'start_s']) ?? cursor;
    const explicitEnd = numberFromRecord(epoch, ['end', 'end_time', 'end_s']);
    const length = numberFromRecord(epoch, ['length', 'duration', 'duration_s']);
    const end = explicitEnd ?? start + (length ?? 0);
    cursor = end;
    if (end <= start) return [];
    return [{
      id: String(epoch.id ?? `epoch:${index}`),
      label: String(epoch.label ?? epoch.id ?? `Epoch ${index + 1}`),
      start,
      end,
      kind: 'epoch' as const,
    }];
  });
}

export function workspaceReplayEventTicks(trial: WorkspaceReplayTrial | null | undefined): WorkspaceReplayEventTick[] {
  const timeline = trial?.trial_spec?.timeline;
  if (!isRecord(timeline) || !Array.isArray(timeline.events)) return [];
  return timeline.events.flatMap((event, index) => {
    if (!isRecord(event)) return [];
    const time = numberFromRecord(event, ['time', 'at', 'start', 'step']);
    if (time === null) return [];
    return [{
      id: String(event.id ?? `event:${index}`),
      label: String(event.label ?? event.id ?? `Event ${index + 1}`),
      time,
    }];
  });
}
