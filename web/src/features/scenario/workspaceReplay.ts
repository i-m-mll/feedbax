import type { WorkspaceReplayTrack, WorkspaceReplayTrial } from '@/generated/studioContracts';

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
