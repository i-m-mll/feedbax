import type { StudioTrajectoryTrack, TrialRecord } from '@/generated/studioContracts';

export function primaryWorkspaceReplayTrack(
  trial: TrialRecord, position: string | null,
): StudioTrajectoryTrack | null {
  if (!position) return null;
  const track = trial.states?.[position];
  if (!track) return null;
  return {
    anchor_id: 'effector',
    selector: { namespace: 'state_path', compact: position },
    samples: track.samples,
    dim: 2,
    frame: track.frame,
    units: track.units,
  };
}

export function workspaceReplayFrameIndex(times: number[], position: number): number {
  const bounded = Math.max(0, Math.min(times[times.length - 1] ?? 0, position));
  let best = 0;
  for (let index = 1; index < times.length; index += 1) {
    if (Math.abs(times[index] - bounded) < Math.abs(times[best] - bounded)) best = index;
  }
  return best;
}

export function workspaceReplaySampleAt(
  track: StudioTrajectoryTrack | null | undefined,
  frameIndex: number
): [number, number] | null {
  const sample = track?.samples[Math.max(0, Math.min(track.samples.length - 1, frameIndex))];
  if (!sample || sample.length < 2) return null;
  const x = Number(sample[0]);
  const y = Number(sample[1]);
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}

export function workspaceReplayPolyline(track: StudioTrajectoryTrack | null | undefined): Array<[number, number]> {
  if (!track) return [];
  return track.samples
    .map((sample) => {
      const x = Number(sample[0]);
      const y = Number(sample[1]);
      return Number.isFinite(x) && Number.isFinite(y) ? ([x, y] as [number, number]) : null;
    })
    .filter((point): point is [number, number] => point !== null);
}
