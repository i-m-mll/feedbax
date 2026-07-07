import type { SampledTaskTrial } from '@/api/client';

export function sampledPreviewTrialLabel(trial: Pick<SampledTaskTrial, 'index'>): string {
  return `T${trial.index + 1}`;
}

export function timelineCueOffset(
  cue: { step: number },
  nSteps: number
): number {
  if (!Number.isFinite(cue.step) || !Number.isFinite(nSteps) || nSteps <= 1) return 0;
  return Math.max(0, Math.min(1, cue.step / (nSteps - 1)));
}

export function compactTimelineCues(trial: SampledTaskTrial, limit = 4) {
  return trial.timeline
    .filter((cue) => Number.isFinite(cue.step))
    .sort((left, right) => left.step - right.step || left.label.localeCompare(right.label))
    .slice(0, limit);
}
