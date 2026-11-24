import type { TrialRecord, TrialStateTrack } from '@/generated/studioContracts';

export function primaryWorkspaceReplayTrack(
  trial: TrialRecord, position: string | null,
): TrialStateTrack | null {
  return position ? trial.states?.[position] ?? null : null;
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
  track: Pick<TrialStateTrack, 'samples'> | null | undefined,
  frameIndex: number
): [number, number] | null {
  const sample = track?.samples[Math.max(0, Math.min(track.samples.length - 1, frameIndex))];
  if (!sample || sample.length < 2) return null;
  const x = Number(sample[0]);
  const y = Number(sample[1]);
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}

export function workspaceReplayPolyline(track: Pick<TrialStateTrack, 'samples'> | null | undefined): Array<[number, number]> {
  if (!track) return [];
  return track.samples
    .map((sample) => {
      const x = Number(sample[0]);
      const y = Number(sample[1]);
      return Number.isFinite(x) && Number.isFinite(y) ? ([x, y] as [number, number]) : null;
    })
    .filter((point): point is [number, number] => point !== null);
}
