import { z } from 'zod';
import type { StudioActivityJob, StudioJobEvent } from '@/generated/studioContracts';
import { PlotTrack, plotHeight, plotWidth, projectSeries } from './studioPlot';
import { studioFormat, studioName } from './studioVocabulary';

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
const formatLoss = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 5 });

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
  let cutoff = startByAttempt.get(job.job_id) ?? Infinity;
  let root = job;
  let previous = root.continues;
  while (previous && !attemptCutoffs.has(previous)) {
    attemptCutoffs.set(previous, cutoff);
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
    log: log.join('\n'),
    trajectory,
  };
}

export function LossPlot({ stages }: { stages: LossStage[] }) {
  const chained = stages.length > 1;
  const samples = stages.flatMap((stage) => stage.points.map((point) => ({
    ...point, stage, update: stage.offset + point.end,
  })));
  const ranges = samples.flatMap((sample) => sample.interval ? [{ ...sample, ...sample.interval }] : []);
  const latest = samples.at(-1);
  const [plot, minima, maxima, boundaries] = projectSeries([
    samples.map(({ update, value }) => [update, value]),
    ranges.map(({ update, min }) => [update, min]),
    ranges.map(({ update, max }) => [update, max]),
    latest ? stages.slice(1).map(({ offset }) => [offset, latest.value]) : [],
  ]);
  return (
    <section aria-label={studioName('loss')} className="min-w-0 space-y-1">
      <div className="flex justify-between gap-2 text-xs font-medium">
        <h4>{studioName('loss')}</h4>
        {latest && (
          <span className="tabular-nums" title={studioFormat('lastLossAtUpdate', { end: latest.update })}>
            {formatLoss.format(latest.last)}
          </span>
        )}
      </div>
      <svg
        viewBox={`0 0 ${plotWidth} ${plotHeight}`}
        role="img"
        aria-label={studioName('loss')}
        className="h-28 w-full rounded border border-slate-200 bg-white"
      >
        {ranges.length > 0 && <desc>{studioFormat('lossIntervalDescription')}</desc>}
        {boundaries.map(([x], index) => (
          <line key={stages[index + 1].offset} x1={x} x2={x} y1={0} y2={plotHeight}
            className="text-slate-300" stroke="currentColor" strokeDasharray="3 3"
            vectorEffect="non-scaling-stroke" />
        ))}
        {ranges.map(({ update, end, start, count, min, max, stage, jobId }, index) => (
          <line
            key={update}
            x1={minima[index][0]}
            x2={maxima[index][0]}
            y1={minima[index][1]}
            y2={maxima[index][1]}
            className="text-blue-400"
            stroke="currentColor"
            vectorEffect="non-scaling-stroke"
          >
            <title>{[
              studioFormat('lossInterval', {
                start: stage.offset + start + 1, end: update, count,
                min: formatLoss.format(min), max: formatLoss.format(max),
              }),
              chained && studioFormat('lossStageSample', { stage: stage.label, start: start + 1, end, attempt: jobId }),
            ].filter(Boolean).join(' · ')}</title>
          </line>
        ))}
        <PlotTrack points={plot} stroke="text-blue-700" />
      </svg>
      <div className="flex flex-wrap items-baseline justify-end gap-x-3 text-xs text-slate-700">
        {chained && (
          <ol aria-label={studioName('trainStages')} className="mr-auto flex flex-wrap gap-x-3">
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
