import { z } from 'zod';
import type { StudioActivityJob, StudioJobEvent } from '@/generated/studioContracts';
import { useElementSize } from '@/hooks/useElementSize';
import type { ObjectiveTrack } from './heldOutObjective';
import { PlotTrack, formatTrainingLoss } from './studioPlot';
import { studioAccessibleName, studioFormat, studioName } from './studioVocabulary';

type Attempt = Pick<StudioActivityJob, 'job_id' | 'continues'>;

export type LossObservation = {
  end: number;
  value: number;
  last: number;
  jobId: string;
  interval?: { start: number; count: number; min: number; max: number };
};

export type LossStage = {
  label: string;
  offset: number;
  completed: number;
  points: LossObservation[];
};
const lossIntervalSchema = z.object({
  start_exclusive: z.number().int().nonnegative(),
  end_inclusive: z.number().int().positive(),
  metrics: z.object({
    train_loss: z.object({
      count: z.number().int().positive(),
      mean: z.number().finite(),
      min: z.number().finite(),
      max: z.number().finite(),
      last: z.number().finite(),
    }),
  }),
});
export function studioRunHistory(
  job: Attempt,
  jobs: readonly Attempt[],
  events: readonly StudioJobEvent[],
) {
  const byId = new Map(jobs.map((attempt) => [attempt.job_id, attempt]));
  const startByAttempt = new Map<string, number>();
  for (const event of events) {
    if (event.kind !== 'progress' || typeof event.data.completed_updates !== 'number') continue;
    const metrics = event.data.metrics;
    if (metrics && typeof metrics === 'object' && Object.keys(metrics).length === 0) {
      startByAttempt.set(event.job_id, event.data.completed_updates);
    }
  }
  const attemptCutoffs = new Map([[job.job_id, Infinity]]);
  const attemptStarts = new Map([[job.job_id, startByAttempt.get(job.job_id)]]);
  let cutoff = startByAttempt.get(job.job_id) ?? Infinity;
  let root = job;
  let previous = root.continues;
  while (previous && !attemptCutoffs.has(previous)) {
    attemptCutoffs.set(previous, cutoff);
    attemptStarts.set(previous, startByAttempt.get(previous));
    cutoff = Math.min(cutoff, startByAttempt.get(previous) ?? Infinity);
    const predecessor = byId.get(previous);
    if (!predecessor) break;
    root = predecessor;
    previous = root.continues;
  }

  const loss = new Map<number, LossObservation>();
  const log: string[] = [];
  let trajectory: StudioJobEvent | undefined;
  for (const event of events) {
    const cutoff = attemptCutoffs.get(event.job_id);
    if (cutoff === undefined) continue;
    if (event.kind === 'progress') {
      const completed = event.data.completed_updates;
      if (typeof completed !== 'number' || completed > cutoff) continue;
      const interval = lossIntervalSchema.safeParse(event.data.metric_interval);
      if (interval.success && interval.data.end_inclusive === completed) {
        const { start_exclusive: start, metrics: { train_loss: reduction } } = interval.data;
        loss.set(completed, {
          end: completed,
          jobId: event.job_id,
          value: reduction.mean,
          last: reduction.last,
          interval: { start, count: reduction.count, min: reduction.min, max: reduction.max },
        });
        continue;
      }
      const metrics = event.data.metrics;
      const value = metrics && typeof metrics === 'object' && 'train_loss' in metrics
        ? metrics.train_loss
        : undefined;
      if (typeof value === 'number' && Number.isFinite(value)) {
        loss.set(completed, { end: completed, jobId: event.job_id, value, last: value });
      }
    } else if (event.kind === 'log' && typeof event.data.line === 'string') {
      log.push(event.data.line);
    } else if (event.kind === 'frame' && (!trajectory || event.cursor > trajectory.cursor)) {
      trajectory = event;
    }
  }
  return {
    loss: [...loss.values()].sort((first, second) => first.end - second.end),
    stageStart: root.continues ? undefined : startByAttempt.get(root.job_id),
    attemptStarts,
    attemptCutoffs,
    log: log.join('\n'),
    trajectory,
  };
}

export function LossPlot({ stages, objectives = [] }: { stages: LossStage[]; objectives?: ObjectiveTrack[] }) {
  const { ref, width } = useElementSize<HTMLElement>();
  const chained = stages.length > 1;
  const samples = stages.flatMap((stage) => stage.points.map((point) => ({
    ...point, stage, update: stage.offset + point.end,
  })));
  const ranges = samples.flatMap((sample) => sample.interval ? [{ ...sample, ...sample.interval }] : []);
  const heldOut = objectives.flatMap(({ segments }) => segments.flat());
  const hasPoints = samples.length > 0 || heldOut.length > 0;
  const latest = samples.at(-1);
  const plotWidth = width || 320;
  const plotHeight = 148;
  const left = 58;
  const right = Math.max(left + 1, plotWidth - 16);
  const top = 12;
  const bottom = plotHeight - 26;
  const stageEnd = stages.reduce((end, { offset, completed }) => Math.max(end, offset + completed), latest?.update ?? 0);
  const endUpdate = heldOut.reduce((end, { update }) => Math.max(end, update), stageEnd);
  const x = (update: number) => left + (right - left) * update / Math.max(1, endUpdate);
  const bounds = heldOut.reduce(([min, max], { value }) =>
    [Math.min(min, value), Math.max(max, value)], [Infinity, -Infinity]);
  const [minLoss, maxLoss] = samples.reduce(([min, max], { value, interval }) =>
    [Math.min(min, value, interval?.min ?? value), Math.max(max, value, interval?.max ?? value)],
  bounds);
  const log = minLoss > 0;
  const transform = log ? Math.log10 : (value: number) => value;
  const low = transform(minLoss);
  const span = transform(maxLoss) - low;
  const y = (value: number) => span > 0
    ? bottom - (bottom - top) * (transform(value) - low) / span
    : (top + bottom) / 2;
  const yTicks = !hasPoints ? [] : span > 0
    ? [minLoss, log ? 10 ** (low + span / 2) : minLoss + span / 2, maxLoss] : [minLoss];
  const xTicks = [...new Set([0, Math.round(endUpdate / 3), Math.round(endUpdate * 2 / 3), endUpdate])];
  return (
    <section ref={ref} aria-label={studioName('loss')} className="min-w-0 space-y-1">
      <div className="flex justify-between gap-2 text-xs font-medium">
        <h4>{studioName('loss')}</h4>
        {latest && (
          <span className="tabular-nums" aria-description={studioFormat('lastLossAtUpdate', { end: latest.update })}>
            {formatTrainingLoss(latest.last)}
          </span>
        )}
      </div>
      <svg
        viewBox={`0 0 ${plotWidth} ${plotHeight}`}
        role="img"
        aria-label={studioName('loss')}
        className="h-[148px] w-full rounded border border-shell-divider bg-shell-surface"
      >
        {ranges.length > 0 && <desc>{studioFormat('lossIntervalDescription')}</desc>}
        <g aria-hidden="true" className="fill-shell-muted text-[11px] tabular-nums">
          {yTicks.map((value) => (
            <g key={value}>
              <line x1={left} x2={right} y1={y(value)} y2={y(value)} className="stroke-shell-divider" />
              <text x={left - 6} y={y(value)} dominantBaseline="middle" textAnchor="end">{formatTrainingLoss(value)}</text>
            </g>
          ))}
          {hasPoints && <line x1={left} x2={right} y1={bottom} y2={bottom} className="stroke-shell-divider" />}
          {hasPoints && xTicks.map((update) => (
            <text key={update} x={x(update)} y={plotHeight - 8}
              textAnchor={update === endUpdate ? 'end' : update === 0 ? 'start' : 'middle'}>{update.toLocaleString()}</text>
          ))}
        </g>
        {hasPoints && stages.slice(1).map(({ offset }) => (
          <line key={offset} x1={x(offset)} x2={x(offset)} y1={top} y2={bottom}
            className="text-shell-divider" stroke="currentColor" strokeDasharray="3 3"
            vectorEffect="non-scaling-stroke" />
        ))}
        {ranges.map(({ update, end, start, count, min, max, stage, jobId }) => (
          <line
            key={update}
            data-attempt={jobId}
            x1={x(update)}
            x2={x(update)}
            y1={y(min)}
            y2={y(max)}
            className="text-shell-foreground opacity-50"
            stroke="currentColor"
            vectorEffect="non-scaling-stroke"
          >
            <title>{[
              studioFormat('lossInterval', {
                start: stage.offset + start + 1, end: update, count,
                min: formatTrainingLoss(min), max: formatTrainingLoss(max),
              }),
              chained && studioFormat('lossStepSample', { step: stage.label, start: start + 1, end }),
            ].filter(Boolean).join(' · ')}</title>
          </line>
        ))}
        <PlotTrack points={samples.map(({ update, value }) => [x(update), y(value)])} />
        {objectives.map(({ id, label, segments }, index) => (
          <g key={id} role="group" aria-label={studioAccessibleName('objective', label)} style={{ color: objectiveColor(index) }}>
            {segments.map((segment) => (
              <g key={segment[0].id}>
                {segment.length > 1 && <polyline
                  points={segment.map(({ update, value }) => `${x(update)},${y(value)}`).join(' ')}
                  fill="none" stroke="currentColor" strokeWidth={1.5} strokeDasharray="4 3"
                  vectorEffect="non-scaling-stroke"
                />}
                {segment.map(({ id, update, value, coverage }) => (
                  <circle key={id} cx={x(update)} cy={y(value)} r={2.5} fill="currentColor">
                    <title>{[
                      label, `${studioName('updates')} ${update}`, String(value),
                      coverage === undefined ? undefined : `${studioName('coverage')} ${coverage}`,
                    ].filter(Boolean).join(' · ')}</title>
                  </circle>
                ))}
              </g>
            ))}
          </g>
        ))}
      </svg>
      <div className="flex flex-wrap items-baseline justify-end gap-x-3 text-xs text-shell-muted">
        {objectives.length > 0 && <ul aria-label={studioName('objective')} className="mr-auto flex flex-wrap gap-x-3">
          {objectives.map(({ id, label }, index) => <li key={id} style={{ color: objectiveColor(index) }}>{label}</li>)}
        </ul>}
        {chained && (
          <ol aria-label={studioName('trainSteps')} className="mr-auto flex flex-wrap gap-x-3">
            {stages.map(({ label, offset, completed }) => (
              <li key={offset}>
                {label}{completed > 0 && ` ${offset + 1}–${offset + completed}`}
              </li>
            ))}
          </ol>
        )}
        <span>{studioName(chained ? 'cumulativeUpdates' : 'updates')}</span>
      </div>
    </section>
  );
}

function objectiveColor(index: number) {
  return `oklch(0.59 0.16 ${(305 + 137.508 * index) % 360})`;
}
