import { createContext } from 'react';
import type {
  FigureLayoutMeta, FigureProduct, FigureRun, StudioSavedEvaluationRow, StudioSavedTrainRuns,
} from '@/generated/studioContracts';

/** The recorded run whose trials the clock shows: the job of the Evaluate result they were read from. */
export interface ClockRun {
  readonly job_id: string;
}

/**
 * What a figure draws from the window's clock: the time and its setter; the epoch the time is measured from (null: the
 * trial's start); the recorded run the clock shows trials of; the trial picked among them and whether it is shown
 * alone or among all; and how a trial is picked.
 */
export interface FigureClock {
  readonly time: number;
  readonly setTime: (time: number) => void;
  readonly epoch: string | null;
  readonly run: ClockRun | undefined;
  readonly picked: { readonly trial: string; readonly kind: 'one' | 'all' } | undefined;
  readonly pickTrial: (trial: string) => void;
}

/** The window's clock as figures read it, and where a shown figure declares the runs its data was read from. */
export const FigureClockContext = createContext<{
  readonly clock?: FigureClock;
  readonly declare?: (runs: FigureLayoutMeta['runs'] | undefined) => void;
}>({});

const jobs = (runs: FigureLayoutMeta['runs']) => new Set(Object.values(runs).flat().map(({ job_id }: FigureRun) => job_id));

/**
 * The saved evaluation whose current or previous result a figure's declared runs read, when they read exactly one; aliases
 * that project the same job are one evaluation.
 */
export function declaredEvaluation(projection: StudioSavedTrainRuns | undefined, runs: FigureLayoutMeta['runs'] | undefined) {
  if (!runs) return undefined;
  const declared = jobs(runs);
  const named = projection?.window_map?.evaluations?.filter(({ result, previous }) =>
    !!result && declared.has(result.job_id) || !!previous && declared.has(previous.job_id)) ?? [];
  const byJob = new Map(named.map((row) => [(row.result ?? row.previous)!.job_id, row]));
  return byJob.size === 1 ? named[0] : undefined;
}

/** The clock's run for a saved evaluation: the job of the result its trials are drawn from. */
export const clockRun = (evaluation: StudioSavedEvaluationRow | undefined): ClockRun | undefined => {
  const result = evaluation?.result ?? evaluation?.previous;
  return result ? { job_id: result.job_id } : undefined;
};

/** Whether the clock shows trials of a run `figure` declares, whose trials it may then mark. */
export const marksTrials = (figure: FigureProduct, clock: FigureClock | undefined) =>
  !!clock?.run && jobs(figure.layout.meta.runs).has(clock.run.job_id);

/** A panel the playhead is drawn in: one whose x is the clock's time, measured from the clock's epoch. */
export interface TimePanel {
  readonly x: string;
  readonly y: string;
}

/** The panels of `figure` whose x is the recorded trial's time from `epoch` (null: the trial's start). */
export function timePanels(figure: FigureProduct, epoch: string | null): TimePanel[] {
  return Object.values(figure.layout.meta.panels).flatMap(({ x_axis, y_axis, time }) =>
    time && (time.zero_epoch ?? null) === epoch ? [{ x: x_axis, y: y_axis }] : []);
}

/** Plotly's layout attribute for an axis a trace names: `x` is `xaxis`, `y2` is `yaxis2`. */
export const axisAttribute = (axis: string) => `${axis[0]}axis${axis.slice(1)}`;

/** The trial a trace draws, when it is one trial's trace. */
export function traceTrial(figure: FigureProduct, index: number): string | undefined {
  const trial = figure.data[index]?.meta.group.trial;
  return trial === undefined || trial === null ? undefined : String(trial);
}

/** Plotly's restyle of every per-trial trace: whether it is drawn, its line width and its opacity. */
export interface TrialRestyle {
  readonly traces: number[];
  readonly update: { visible: unknown[]; 'line.width': number[]; opacity: number[] };
}

const baseWidth = 2;

/**
 * How the per-trial traces are drawn for the clock: under One the picked trial alone, under All every trial thin and
 * the picked one heavier; as authored when the clock shows no trial of the run the figure declares.
 */
export function trialRestyle(figure: FigureProduct, clock: FigureClock | undefined): TrialRestyle | undefined {
  const traces = figure.data.flatMap((_, index) => traceTrial(figure, index) === undefined ? [] : [index]);
  if (!traces.length) return undefined;
  const marked = marksTrials(figure, clock) ? clock!.picked : undefined;
  const update: TrialRestyle['update'] = { visible: [], 'line.width': [], opacity: [] };
  for (const index of traces) {
    const trace = figure.data[index];
    const width = (trace.line as { width?: number } | undefined)?.width ?? baseWidth;
    const opacity = typeof trace.opacity === 'number' ? trace.opacity : 1;
    const picked = traceTrial(figure, index) === marked?.trial;
    update.visible.push(marked?.kind === 'one' ? picked : trace.visible ?? true);
    update['line.width'].push(marked?.kind === 'all' ? picked ? width * 1.75 : width * 0.5 : width);
    update.opacity.push(marked?.kind === 'all' && !picked ? opacity * 0.35 : opacity);
  }
  return { traces, update };
}
