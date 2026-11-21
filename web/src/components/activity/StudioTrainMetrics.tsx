import type { CSSProperties } from 'react';
import { z } from 'zod';
import type { StudioActivityJob, StudioJobEvent } from '@/generated/studioContracts';
import { useElementSize } from '@/hooks/useElementSize';
import type { ObjectiveTrack } from './heldOutObjective';
import { PlotTrack, formatMetric } from './studioPlot';
import { studioAccessibleName, studioFormat, studioName, studioReason } from './studioVocabulary';

type Attempt = Pick<StudioActivityJob, 'job_id' | 'continues'>;

/** One number of a metric at one progress event: the interval's mean and last value, and its range when reduced. */
export type MetricReading = {
  value: number;
  last: number;
  interval?: { count: number; min: number; max: number };
};

/** A metric at one progress event, with one reading per element; a scalar metric has one element. */
export type MetricObservation = {
  end: number;
  jobId: string;
  start?: number;
  elements: (MetricReading | null)[];
};

/** Everything one metric reported, in the order the run reported it. */
export type MetricHistory = { name: string; vector: boolean; points: MetricObservation[] };

export type MetricStage = {
  label: string;
  offset: number;
  completed: number;
  metrics: MetricHistory[];
};

export type MetricResultName = 'currentResult' | 'previousResult';

export type MetricSeries = {
  result?: { name: MetricResultName; jobId: string };
  stages: MetricStage[];
};

const reductionSchema = z.object({
  count: z.number().int().positive(),
  mean: z.number().finite(),
  min: z.number().finite(),
  max: z.number().finite(),
  last: z.number().finite(),
});
const intervalSchema = z.object({
  start_exclusive: z.number().int().nonnegative(),
  end_inclusive: z.number().int().positive(),
  metrics: z.record(z.unknown()),
});

type Reading = { name: string; vector: boolean; elements: (MetricReading | null)[] };

function reducedReading(value: unknown): MetricReading | null {
  const reduction = reductionSchema.safeParse(value);
  if (!reduction.success) return null;
  const { mean, last, count, min, max } = reduction.data;
  return { value: mean, last, interval: { count, min, max } };
}

function rawReading(value: unknown): MetricReading | null {
  return typeof value === 'number' && Number.isFinite(value) ? { value, last: value } : null;
}

/** The metrics of one event, a mapping's entries named by their path, a list read as a vector of its numbers. */
function metricReadings(metrics: unknown, leaf: (value: unknown) => MetricReading | null, prefix = ''): Reading[] {
  if (!metrics || typeof metrics !== 'object' || Array.isArray(metrics)) return [];
  return Object.entries(metrics).flatMap(([key, value]): Reading[] => {
    const name = `${prefix}${key}`;
    const scalar = leaf(value);
    if (scalar) return [{ name, vector: false, elements: [scalar] }];
    if (Array.isArray(value)) {
      const elements = (value as unknown[]).flat(Infinity).map(leaf);
      return elements.some(Boolean) ? [{ name, vector: true, elements }] : [];
    }
    return metricReadings(value, leaf, `${name}.`);
  });
}

export function studioRunHistory(
  job: Attempt,
  jobs: readonly Attempt[],
  events: readonly StudioJobEvent[],
) {
  const byId = new Map(jobs.map((attempt) => [attempt.job_id, attempt]));
  const startByAttempt = new Map<string, number>();
  for (const event of events) {
    if (event.kind !== 'progress' || typeof event.data.completed_updates !== 'number') continue;
    if (!startByAttempt.has(event.job_id)) {
      startByAttempt.set(event.job_id, event.data.completed_updates);
    }
  }
  const attemptCutoffs = new Map([[job.job_id, Infinity]]);
  const attemptStarts = new Map([[job.job_id, startByAttempt.get(job.job_id)]]);
  const lineageDepth = new Map([[job.job_id, 0]]);
  let cutoff = startByAttempt.get(job.job_id) ?? Infinity;
  let root = job;
  let previous = root.continues;
  while (previous && !attemptCutoffs.has(previous)) {
    attemptCutoffs.set(previous, cutoff);
    lineageDepth.set(previous, lineageDepth.size);
    attemptStarts.set(previous, startByAttempt.get(previous));
    cutoff = Math.min(cutoff, startByAttempt.get(previous) ?? Infinity);
    const predecessor = byId.get(previous);
    if (!predecessor) break;
    root = predecessor;
    previous = root.continues;
  }

  const histories = new Map<string, { vector: boolean; points: Map<number, MetricObservation> }>();
  const log: string[] = [];
  let trajectory: StudioJobEvent | undefined;
  const depth = (id: string) => lineageDepth.get(id) ?? Infinity;
  const record = (end: number, jobId: string, start: number | undefined, readings: Reading[]) => {
    for (const { name, vector, elements } of readings) {
      const history = histories.get(name) ?? { vector, points: new Map() };
      histories.set(name, history);
      const existing = history.points.get(end);
      if (!existing || depth(jobId) <= depth(existing.jobId)) history.points.set(end, { end, jobId, start, elements });
    }
  };
  for (const event of events) {
    const cutoff = attemptCutoffs.get(event.job_id);
    if (cutoff === undefined) continue;
    if (event.kind === 'progress') {
      const completed = event.data.completed_updates;
      if (typeof completed !== 'number' || completed === startByAttempt.get(event.job_id) || completed > cutoff) {
        continue;
      }
      const interval = intervalSchema.safeParse(event.data.metric_interval);
      if (interval.success && interval.data.end_inclusive === completed) {
        record(completed, event.job_id, interval.data.start_exclusive, metricReadings(interval.data.metrics, reducedReading));
      } else {
        record(completed, event.job_id, undefined, metricReadings(event.data.metrics, rawReading));
      }
    } else if (event.kind === 'log' && typeof event.data.line === 'string') {
      log.push(event.data.line);
    } else if (event.kind === 'frame' && (!trajectory || depth(event.job_id) <= depth(trajectory.job_id))) {
      trajectory = event;
    }
  }
  return {
    metrics: [...histories].map(([name, { vector, points }]): MetricHistory => ({
      name, vector, points: [...points.values()].sort((first, second) => first.end - second.end),
    })),
    stageStart: root.continues ? undefined : startByAttempt.get(root.job_id),
    attemptStarts,
    attemptCutoffs,
    log: log.join('\n'),
    trajectory,
  };
}

const elementLimit = 8;

type PlotStage = Omit<MetricStage, 'metrics'>;
type PlotSample = {
  update: number;
  end: number;
  start?: number;
  jobId: string;
  stage: PlotStage;
  reading: MetricReading;
  band?: { min: number; max: number; count: number };
};
type PlotLine = { element?: number; samples: PlotSample[] };

/** One plot per metric the series report, in the order they report them, the held-out objectives on the first. */
export function MetricPlots({ series, objectives = [] }: { series: MetricSeries[]; objectives?: ObjectiveTrack[] }) {
  const histories = new Map<string, boolean>();
  for (const { stages } of series) {
    for (const { metrics } of stages) for (const { name, vector } of metrics) histories.set(name, histories.get(name) || vector);
  }
  const plots = histories.size > 0 ? [...histories] : objectives.length > 0 ? [[studioName('objective'), false] as const] : [];
  return (
    <div className="min-w-0 space-y-3">
      {plots.map(([name, vector], index) => (
        <MetricPlot key={name} name={name} vector={vector} objectives={index === 0 ? objectives : []}
          series={series.map(({ result, stages }) => ({
            result,
            stages: stages.map(({ metrics, ...stage }) => ({
              stage, points: metrics.find((metric) => metric.name === name)?.points ?? [],
            })),
          }))} />
      ))}
    </div>
  );
}

function plotLines(samples: (Omit<PlotSample, 'reading' | 'band'> & { elements: (MetricReading | null)[] })[], vector: boolean): PlotLine[] {
  const count = samples.reduce((most, { elements }) => Math.max(most, elements.length), 0);
  if (vector && count > elementLimit) {
    return [{
      samples: samples.flatMap(({ elements, ...sample }) => {
        const present = elements.filter((element): element is MetricReading => element !== null);
        if (present.length === 0) return [];
        const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
        const values = present.map(({ value }) => value);
        return [{
          ...sample,
          reading: { value: mean(values), last: mean(present.map(({ last }) => last)) },
          band: { min: Math.min(...values), max: Math.max(...values), count: elements.length },
        }];
      }),
    }];
  }
  return Array.from({ length: count }, (_, element) => ({
    element: vector ? element : undefined,
    samples: samples.flatMap(({ elements, ...sample }) => {
      const reading = elements[element];
      return reading ? [{ ...sample, reading }] : [];
    }),
  }));
}

function MetricPlot({ name, vector, series, objectives }: {
  name: string;
  vector: boolean;
  series: { result?: MetricSeries['result']; stages: { stage: PlotStage; points: MetricObservation[] }[] }[];
  objectives: ObjectiveTrack[];
}) {
  const { ref, width } = useElementSize<HTMLElement>();
  const primary = series[0];
  const chained = series.some(({ stages }) => stages.length > 1);
  const tracks = series.map(({ result, stages }) => {
    const lines = plotLines(stages.flatMap(({ stage, points }) => points.map(({ end, start, jobId, elements }) => ({
      end, start, jobId, elements, stage, update: stage.offset + end,
    }))), vector);
    return {
      result, lines,
      dashed: result?.name === 'previousResult' && series.length > 1,
      empty: lines.every(({ samples }) => samples.length === 0),
    };
  });
  const samples = tracks.flatMap(({ lines }) => lines.flatMap((line) => line.samples));
  const elements = vector ? [...new Set(tracks.flatMap(({ lines }) => lines.flatMap(({ element }) =>
    element === undefined ? [] : [element])))] : [];
  const banded = samples.find(({ band }) => band)?.band;
  const intervals = samples.some(({ reading }) => reading.interval);
  const heldOut = objectives.flatMap(({ segments }) => segments.flat());
  const hasPoints = samples.length > 0 || heldOut.length > 0;
  const latest = vector ? undefined : tracks[0]?.lines[0]?.samples.at(-1);
  const plotWidth = width || 320;
  const plotHeight = 148;
  const left = 58;
  const right = Math.max(left + 1, plotWidth - 16);
  const top = 12;
  const bottom = plotHeight - 26;
  const stageEnd = series.flatMap(({ stages }) => stages)
    .reduce((end, { stage: { offset, completed } }) => Math.max(end, offset + completed),
      samples.reduce((end, sample) => Math.max(end, sample.update), 0));
  const endUpdate = heldOut.reduce((end, { update }) => Math.max(end, update), stageEnd);
  const x = (update: number) => left + (right - left) * update / Math.max(1, endUpdate);
  const bounds = heldOut.reduce(([min, max], { value }) =>
    [Math.min(min, value), Math.max(max, value)], [Infinity, -Infinity]);
  const [minValue, maxValue] = samples.reduce(([min, max], { reading: { value, interval }, band }) => [
    Math.min(min, value, interval?.min ?? value, band?.min ?? value),
    Math.max(max, value, interval?.max ?? value, band?.max ?? value),
  ], bounds);
  const log = minValue > 0;
  const transform = log ? Math.log10 : (value: number) => value;
  const low = transform(minValue);
  const span = transform(maxValue) - low;
  const y = (value: number) => span > 0
    ? bottom - (bottom - top) * (transform(value) - low) / span
    : (top + bottom) / 2;
  const yTicks = !hasPoints ? [] : span > 0
    ? [minValue, log ? 10 ** (low + span / 2) : minValue + span / 2, maxValue] : [minValue];
  const xTicks = [...new Set([0, Math.round(endUpdate / 3), Math.round(endUpdate * 2 / 3), endUpdate])];
  const color = (element: number | undefined) => element === undefined
    ? { className: 'text-shell-foreground' } : { className: 'studio-hue', style: elementHue(element) };
  return (
    <section ref={ref} aria-label={studioAccessibleName('metric', name)} className="min-w-0 space-y-1">
      <div className="flex justify-between gap-2 text-xs font-medium">
        <h4>{name}</h4>
        {latest && !primary?.result && (
          <span className="tabular-nums" aria-description={studioFormat('lastValueAtUpdate', { end: latest.update })}>
            {formatMetric(latest.reading.last)}
          </span>
        )}
      </div>
      <svg
        viewBox={`0 0 ${plotWidth} ${plotHeight}`}
        role="img"
        aria-label={studioAccessibleName('metric', name)}
        className="h-[148px] w-full rounded border border-shell-divider bg-shell-surface"
      >
        {banded ? <desc>{studioFormat('metricElementBand', { count: banded.count })}</desc>
          : intervals && <desc>{studioFormat('metricIntervalDescription')}</desc>}
        <g aria-hidden="true" className="fill-shell-muted text-xs tabular-nums">
          {yTicks.map((value) => (
            <g key={value}>
              <line x1={left} x2={right} y1={y(value)} y2={y(value)} className="stroke-shell-divider" />
              <text x={left - 6} y={y(value)} dominantBaseline="middle" textAnchor="end">{formatMetric(value)}</text>
            </g>
          ))}
          {hasPoints && <line x1={left} x2={right} y1={bottom} y2={bottom} className="stroke-shell-divider" />}
          {hasPoints && xTicks.map((update) => (
            <text key={update} x={x(update)} y={plotHeight - 8}
              textAnchor={update === endUpdate ? 'end' : update === 0 ? 'start' : 'middle'}>{update.toLocaleString()}</text>
          ))}
        </g>
        {hasPoints && [...new Set(series.flatMap(({ stages }) => stages.slice(1).map(({ stage: { offset } }) => offset)))].map((offset) => (
          <line key={offset} x1={x(offset)} x2={x(offset)} y1={top} y2={bottom}
            className="text-shell-divider" stroke="currentColor" strokeDasharray="3 3"
            vectorEffect="non-scaling-stroke" />
        ))}
        {tracks.map(({ result, dashed, lines }) => (
          <g key={result?.name ?? 'metric'} data-result-attempt={result?.jobId}
            role={result && 'group'} aria-label={result && studioName(result.name)}>
            {lines.map(({ element, samples }) => (
              <g key={element ?? 'line'} data-element={element} {...color(element)}>
                {samples.some(({ band }) => band) && (
                  <polygon aria-hidden="true" fill="currentColor" className="opacity-20"
                    points={[...samples.map(({ update, band, reading }) => `${x(update)},${y(band?.max ?? reading.value)}`),
                      ...[...samples].reverse().map(({ update, band, reading }) => `${x(update)},${y(band?.min ?? reading.value)}`)].join(' ')} />
                )}
                {samples.map(({ update, end, start, reading: { interval, value }, stage, jobId }) => {
                  if (!interval && !result) return null;
                  const { count, min, max } = interval ?? { count: 1, min: value, max: value };
                  const first = start ?? end - 1;
                  return (
                    <line key={update} data-attempt={jobId} data-update={result && update} data-value={result && value}
                      x1={x(update)} x2={x(update)} y1={y(min)} y2={y(max)}
                      className="opacity-50" stroke="currentColor"
                      strokeDasharray={dashed ? '4 3' : undefined} vectorEffect="non-scaling-stroke">
                      <title>{[
                        element !== undefined && `${name}[${element}]`,
                        studioFormat('metricInterval', {
                          start: stage.offset + first + 1, end: update, count,
                          min: formatMetric(min), max: formatMetric(max),
                        }),
                        chained && studioFormat('metricStepSample', { step: stage.label, start: first + 1, end }),
                      ].filter(Boolean).join(' · ')}</title>
                    </line>
                  );
                })}
                <PlotTrack points={samples.map(({ update, reading }) => [x(update), y(reading.value)])} dashed={dashed}
                  {...color(element)} />
              </g>
            ))}
          </g>
        ))}
        {objectives.map(({ id, label, segments }, index) => (
          <g key={id} role="group" aria-label={studioAccessibleName('objective', label)} className="studio-hue" style={objectiveHue(index)}>
            {segments.filter((segment) => segment.length > 1).map((segment) => (
              <polyline key={segment[0].id}
                points={segment.map(({ update, value }) => `${x(update)},${y(value)}`).join(' ')}
                fill="none" stroke="currentColor" strokeWidth={1.5} strokeDasharray="4 3"
                vectorEffect="non-scaling-stroke"
              />
            ))}
            {segments.flat().map(({ id, update, value, coverage }) => (
              <circle key={id} cx={x(update)} cy={y(value)} r={2.5} fill="currentColor">
                <title>{[
                  label, `${studioName('updates')} ${update}`, String(value),
                  coverage === undefined ? undefined : `${studioName('coverage')} ${coverage}`,
                ].filter(Boolean).join(' · ')}</title>
              </circle>
            ))}
          </g>
        ))}
      </svg>
      <div className="flex flex-wrap items-baseline justify-end gap-x-3 text-xs text-shell-muted">
        {elements.length > 0 && <ul aria-label={studioName('elements')} className="mr-auto flex flex-wrap gap-x-3">
          {elements.map((element) => <li key={element} className="studio-hue" style={elementHue(element)}>{String(element)}</li>)}
        </ul>}
        {objectives.length > 0 && <ul aria-label={studioName('objective')} className="mr-auto flex flex-wrap gap-x-3">
          {objectives.map(({ id, label }, index) => <li key={id} className="studio-hue" style={objectiveHue(index)}>{label}</li>)}
        </ul>}
        {chained && !primary?.result && (
          <ol aria-label={studioName('trainSteps')} className="mr-auto flex flex-wrap gap-x-3">
            {primary.stages.map(({ stage: { label, offset, completed } }) => (
              <li key={offset}>
                {label}{completed > 0 && ` ${offset + 1}–${offset + completed}`}
              </li>
            ))}
          </ol>
        )}
        {primary?.result && <ul aria-label={studioAccessibleName('metric', name)} className="mr-auto flex flex-wrap gap-x-3">
          {tracks.map(({ result, dashed, empty }) => result && (
            <li key={result.name} className="flex items-center gap-1">
              <span aria-hidden="true" className={`w-4 border-t-2 ${dashed ? 'border-dashed' : ''}`} />
              {studioName(result.name)}{empty &&
                <span role="status">{studioReason('unavailable', name)}</span>}
            </li>
          ))}
        </ul>}
        <span>{studioName(chained ? 'cumulativeUpdates' : 'updates')}</span>
      </div>
    </section>
  );
}

function objectiveHue(index: number) {
  return { '--studio-hue': (305 + 137.508 * index) % 360 } as CSSProperties;
}

function elementHue(index: number) {
  return { '--studio-hue': (200 + 137.508 * index) % 360 } as CSSProperties;
}
