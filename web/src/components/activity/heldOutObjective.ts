import type { StudioEvaluationObjectiveSeries } from '@/generated/studioContracts';
import { studioName } from './studioVocabulary';

export type ObjectiveStage = {
  offset: number;
  origin: number;
  through: number | null;
  request: string;
  starts: ReadonlyMap<string, number | undefined>;
  cutoffs: ReadonlyMap<string, number>;
};

export type ObjectiveSample = {
  id: string;
  update: number;
  value: number;
  coverage?: number;
};

export type ObjectiveTrack = { id: string; label: string; segments: ObjectiveSample[][] };

export function heldOutTracks(
  label: string, result: StudioEvaluationObjectiveSeries, stage: ObjectiveStage,
): ObjectiveTrack[] {
  // A result settled on a stopped ancestor stays visible after Resume.
  // A result settled on a different or later branch does not describe this stage.
  const attempts = [...stage.cutoffs.keys()].reverse();
  if (attempts.some((job) => stage.starts.get(job) === undefined)) return [];
  const tip = attempts.indexOf(result.lineage.at(-1) ?? '');
  if (tip < 0 || result.lineage.some((job, index) => (
    job !== attempts[tip - result.lineage.length + 1 + index]
  ))) return [];

  const retained = new Map(result.points.filter(({ checkpoint }) => {
    const { job_id: job, request_id: request, completed_updates: update } = checkpoint;
    const start = stage.starts.get(job);
    return request === stage.request && start !== undefined && update > start &&
      update <= Math.min(stage.through ?? Infinity, stage.cutoffs.get(job) ?? -Infinity);
  }).map((point) => [point.checkpoint.completed_updates, point]));
  if (!retained.size) return [];

  const terms = new Map(result.points.flatMap((point) => point.terms.map(({ path, name }) => [path, name])));
  const updates = [...result.requested].sort((first, second) => first - second);
  const metrics: [string | null, string][] = [[null, studioName('total')], ...terms];
  return metrics.map(([path, name]) => {
    const id = JSON.stringify([result.job_id, result.condition, path]);
    const segments: ObjectiveSample[][] = [];
    let segment: ObjectiveSample[] = [];
    for (const update of updates) {
      const point = retained.get(update);
      const term = point?.terms.find((candidate) => candidate.path === path);
      const value = path === null
        ? point?.terms.some(({ coverage }) => coverage > 0) ? point.total : undefined
        : term?.coverage ? term.value : undefined;
      if (!point || value === undefined) {
        segment = [];
        continue;
      }
      if (!segment.length) segments.push(segment);
      segment.push({
        id: JSON.stringify([id, point.checkpoint.transaction_root_sha256, point.checkpoint.publication_cursor]),
        update: stage.offset + update - stage.origin, value, coverage: term?.coverage,
      });
    }
    return { id, label: `${label} · ${name}`, segments };
  }).filter(({ segments }) => segments.length > 0);
}
