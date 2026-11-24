import type { TrialStateTrack } from '@/generated/studioContracts';

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
