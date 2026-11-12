import { useMemo } from 'react';
import { z } from 'zod';
import {
  TrainingTrajectoryPayloadSchema,
  type StudioActivityJob,
  type StudioJobEvent,
} from '@/generated/studioContracts';
import {
  liveTrainingEffectorTrack,
  liveTrainingTargetTrack,
  normalizeTrainingTrajectoryPayload,
} from '@/features/scenario/liveTraining';
import { workspaceReplayPolyline } from '@/features/scenario/workspaceReplay';
import { useStudioJobActions, useStudioJobConnection, useStudioJobHistory, useStudioJobs } from '@/hooks/useStudioJobs';
import { StudioRegion, StudioRegionBoundary, studioRegionIssue, type StudioRegionState } from './StudioRegion';
import { StudioRunProgress, StudioRunReason, StudioRunStatus } from './StudioRunStatus';
import { studioName, studioReason } from './studioVocabulary';

type Point = [number, number];
type LossObservation = {
  end: number;
  value: number;
  last: number;
  interval?: { start: number; count: number; min: number; max: number };
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
const emptyJobs: readonly StudioActivityJob[] = [];
const emptyEvents: readonly StudioJobEvent[] = [];

function runHistory(
  job: StudioActivityJob,
  jobs: readonly StudioActivityJob[],
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
  let previous = job.continues;
  while (previous && !attemptCutoffs.has(previous)) {
    attemptCutoffs.set(previous, cutoff);
    cutoff = Math.min(cutoff, startByAttempt.get(previous) ?? Infinity);
    previous = byId.get(previous)?.continues;
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
        loss.set(completed, { end: completed, value, last: value });
      }
    } else if (event.kind === 'log' && typeof event.data.line === 'string') {
      log.push(event.data.line);
    } else if (event.kind === 'frame' && (!trajectory || event.cursor > trajectory.cursor)) {
      trajectory = event;
    }
  }
  return {
    loss: [...loss.values()].sort((first, second) => first.end - second.end),
    log: log.join('\n'),
    trajectory,
  };
}

const plotWidth = 240;
const plotHeight = 126;
const plotInset = 8;

function projectSeries(series: readonly Point[][], preserveAspect = false) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const track of series) {
    for (const [x, y] of track) {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }
  if (!Number.isFinite(minX)) return series;
  const width = plotWidth - 2 * plotInset;
  const height = plotHeight - 2 * plotInset;
  const spanX = maxX - minX;
  const spanY = maxY - minY;
  let scaleX = spanX > 0 ? width / spanX : Infinity;
  let scaleY = spanY > 0 ? height / spanY : Infinity;
  if (preserveAspect) scaleX = scaleY = Math.min(scaleX, scaleY);
  if (!Number.isFinite(scaleX)) scaleX = 1;
  if (!Number.isFinite(scaleY)) scaleY = 1;
  return series.map((track) => track.map(([x, y]): Point => [
    plotWidth / 2 + (x - minX - spanX / 2) * scaleX,
    plotHeight / 2 - (y - minY - spanY / 2) * scaleY,
  ]));
}

function PlotTrack({ points, stroke }: { points: Point[]; stroke: string }) {
  const last = points[points.length - 1];
  if (!last) return null;
  return (
    <g className={stroke}>
      <polyline
        points={points.map(([x, y]) => `${x},${y}`).join(' ')}
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
      <circle cx={last[0]} cy={last[1]} r={3} fill="currentColor" />
    </g>
  );
}

function LossPlot({ points }: { points: LossObservation[] }) {
  const ranges = points.flatMap(({ end, interval }) =>
    interval ? [{ end, ...interval }] : [],
  );
  const [plot, minima, maxima] = projectSeries([
    points.map(({ end, value }) => [end, value]),
    ranges.map(({ end, min }) => [end, min]),
    ranges.map(({ end, max }) => [end, max]),
  ]);
  const latest = points[points.length - 1];
  return (
    <section aria-label={studioName('loss')} className="min-w-0 space-y-1">
      <div className="flex justify-between gap-2 text-xs font-medium">
        <h4>{studioName('loss')}</h4>
        {latest && (
          <span className="tabular-nums" title={`Last loss at update ${latest.end}`}>
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
        {ranges.length > 0 && <desc>Interval mean loss with minimum and maximum bars</desc>}
        {ranges.map(({ end, start, count, min, max }, index) => (
          <line
            key={end}
            x1={minima[index][0]}
            x2={maxima[index][0]}
            y1={minima[index][1]}
            y2={maxima[index][1]}
            className="text-blue-400"
            stroke="currentColor"
            vectorEffect="non-scaling-stroke"
          >
            <title>{`Updates ${start + 1}–${end} (${count} samples), ${formatLoss.format(min)}–${formatLoss.format(max)}`}</title>
          </line>
        ))}
        <PlotTrack points={plot} stroke="text-blue-700" />
      </svg>
      <div className="text-right text-xs text-slate-700">{studioName('updates')}</div>
    </section>
  );
}

function RunScene({ event, retry }: { event?: StudioJobEvent; retry: () => void }) {
  const parsed = useMemo(() => event ? TrainingTrajectoryPayloadSchema.safeParse(event.data.trajectory) : null, [event]);
  const tracks = useMemo(() => {
    if (!parsed?.success) return [];
    const frame = normalizeTrainingTrajectoryPayload(parsed.data);
    return projectSeries([
      workspaceReplayPolyline(liveTrainingEffectorTrack(frame)),
      workspaceReplayPolyline(liveTrainingTargetTrack(frame)),
    ], true);
  }, [parsed]);
  const [effector = [], goal = []] = tracks;
  return (
    <section aria-label={studioName('scene')} className="min-w-0 space-y-1">
      <h4 className="text-xs font-medium">{studioName('scene')}</h4>
      {parsed?.success === false ? (
        <StudioRegion
          state={{
            kind: 'failed',
            issue: { reason: studioReason('unavailable', studioName('scene')), detail: parsed.error.message },
            retry,
          }}
          skeleton={null}
          empty={null}
        >{null}</StudioRegion>
      ) : (
        <>
          <svg
            viewBox={`0 0 ${plotWidth} ${plotHeight}`}
            role="img"
            aria-label={studioName('scene')}
            className="h-28 w-full rounded border border-slate-200 bg-white"
          >
            <PlotTrack points={goal} stroke="text-violet-700" />
            <PlotTrack points={effector} stroke="text-blue-700" />
          </svg>
          <div className="flex flex-wrap gap-2 text-xs">
            {effector.length > 0 && <span className="text-blue-700">{studioName('effector')}</span>}
            {goal.length > 0 && <span className="text-violet-700">{studioName('goal')}</span>}
          </div>
        </>
      )}
    </section>
  );
}

function RunInspector({ job }: { job: StudioActivityJob }) {
  const jobs = useStudioJobs().data?.jobs ?? emptyJobs;
  const history = useStudioJobHistory(job.project_id);
  const events = history.data ?? emptyEvents;
  const connection = useStudioJobConnection();
  const actions = useStudioJobActions();
  const data = useMemo(() => runHistory(job, jobs, events), [job, jobs, events]);
  const retry = () => { void actions.refresh(); };
  const state: StudioRegionState = connection === 'disconnected' || history.isError
    ? { kind: 'failed', issue: studioRegionIssue('inspector', history.error, connection === 'disconnected'), retry }
    : history.isPending
      ? { kind: 'loading' }
      : events.length > 0
        ? { kind: 'result', missing: [] }
        : { kind: 'empty' };
  const fields = (
    <>
      <div className="grid gap-3 sm:grid-cols-2">
        <LossPlot points={data.loss} />
        <RunScene event={data.trajectory} retry={retry} />
      </div>
      <section aria-label={studioName('log')} className="min-w-0 space-y-1">
        <h4 className="text-xs font-medium">{studioName('log')}</h4>
        <pre className="min-h-9 max-h-32 select-text overflow-auto whitespace-pre-wrap break-words rounded border border-slate-200 bg-white p-2 text-xs text-slate-700">
          {data.log}
        </pre>
      </section>
    </>
  );
  return (
    <section aria-label={studioName('inspector')} className="space-y-3 rounded-md border border-slate-200 bg-slate-50 p-3">
      <header className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="break-words font-medium">{job.label}</h3>
          <StudioRunReason job={job} />
        </div>
        <StudioRunStatus status={job.status} />
      </header>
      {job.progress && <StudioRunProgress progress={job.progress} etaSeconds={job.eta_seconds} />}
      <StudioRegion
        state={state}
        skeleton={<div className="h-36 animate-pulse rounded bg-slate-200" />}
        empty={fields}
      >
        {fields}
      </StudioRegion>
    </section>
  );
}

export function StudioRunInspector({ job }: { job: StudioActivityJob }) {
  return (
    <StudioRegionBoundary name="inspector">
      <RunInspector job={job} />
    </StudioRegionBoundary>
  );
}
