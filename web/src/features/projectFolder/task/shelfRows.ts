import { createElement } from 'react';
import { studioName } from '@/components/activity/studioVocabulary';
import type { ShelfBar, ShelfRow, ShelfSegment } from '@/components/timeline/Shelf';
import type { NamedElementAddress } from '@/features/projectWindow/elementSelection';
import { viewGlyphs } from '@/features/projectWindow/ProjectWindow';
import type { EpochBounds } from './taskModel';
import type { Problem, TrialAggregate, TrialMean, TrialRecord } from '@/generated/studioContracts';
import { endedBy, signalSteps } from './trialSample';

/**
 * What the shelf draws of a task document: its epochs' bounds in the order of one trial, a repeated epoch once per
 * repetition, and the epochs over which each signal has a cell.
 */
export interface ShelfTask {
  /** The task document, whose elements the rows select. */
  readonly document: string;
  readonly epochs: readonly EpochBounds[];
  readonly signals: readonly {
    readonly id: string;
    readonly name: string;
    /** The ids of the epochs its cells span. */
    readonly covers: readonly string[];
  }[];
  /** The trial's length, in seconds. */
  readonly duration: number;
  /** The Objective's terms that are on, in order, and the document whose elements their rows select. */
  readonly objective?: {
    readonly document: string;
    readonly terms: readonly { readonly id: string; readonly name: string }[];
  };
}

const named = (document: string, id: string): NamedElementAddress => ({ document, element: { kind: 'named', id } });

/** Adjacent or overlapping spans joined, in order. */
function joined(spans: readonly ShelfBar[]): ShelfBar[] {
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const out: { start: number; end: number }[] = [];
  for (const span of sorted) {
    const last = out.at(-1);
    if (last && span.start <= last.end + 1e-9) last.end = Math.max(last.end, span.end);
    else out.push({ ...span });
  }
  return out;
}

/**
 * The nominal trial's rows: each epoch at its nominal length, a drawn length with a whisker from where its end falls
 * least to greatest, a condition's dashed out to its limit, Rest what remains; each signal present over its cells.
 */
export function nominalRows(task: ShelfTask): ShelfRow[] {
  let cursor = 0;
  const segments = task.epochs.map((epoch, index): ShelfSegment => {
    const start = cursor;
    const last = index === task.epochs.length - 1;
    const length = epoch.ends === 'condition' ? epoch.greatest : epoch.nominal;
    const end = epoch.ends === 'rest' || last ? Math.max(start, task.duration) : Math.min(task.duration, start + length);
    cursor = end;
    return {
      key: `${epoch.id}:${index}`, name: epoch.name, start, end, selects: named(task.document, `epoch:${epoch.id}`),
      ...epoch.ends === 'drawn' && epoch.greatest > epoch.least && {
        spread: { least: start + epoch.least, greatest: Math.min(task.duration, start + epoch.greatest) },
      },
      ...epoch.ends === 'condition' && { dashedFrom: start + epoch.least },
    };
  });
  return [
    { kind: 'segments', key: 'epochs', name: studioName('epochs'), segments },
    ...task.signals.map((signal): ShelfRow => ({
      kind: 'bars', key: `signal:${signal.id}`, name: signal.name, selects: named(task.document, `signal:${signal.id}`),
      bars: joined(segments.flatMap(({ start, end }, index) =>
        signal.covers.includes(task.epochs[index].id) && end > start ? [{ start, end }] : [])),
    })),
  ];
}

/** Where a signal is present, or a term applies, in one trial: its runs of steps that hold. */
function presence(steps: readonly unknown[], stepSeconds: number): ShelfBar[] {
  const bars: ShelfBar[] = [];
  let start: number | undefined;
  steps.forEach((value, step) => {
    if (value && start === undefined) start = step;
    if (!value && start !== undefined) { bars.push({ start: start * stepSeconds, end: step * stepSeconds }); start = undefined; }
  });
  if (start !== undefined) bars.push({ start: start * stepSeconds, end: steps.length * stepSeconds });
  return bars;
}

/**
 * One row per term that is on, with the bars `drawn` gives of the term's id (none where the term applies nowhere) and
 * its first Problem, marked at its name.
 */
function termRows(task: ShelfTask, drawn: (term: string) => { bars: ShelfBar[]; problems?: readonly Problem[] }): ShelfRow[] {
  const objective = task.objective;
  if (!objective) return [];
  const glyph = createElement(viewGlyphs.objective, { 'aria-hidden': true, className: 'size-3 shrink-0' });
  return objective.terms.map(({ id, name }): ShelfRow => {
    const { bars, problems } = drawn(id);
    const problem = problems?.[0]?.reason;
    return { kind: 'bars', key: `term:${id}`, name, glyph, selects: named(objective.document, id), bars, ...problem && { problem } };
  });
}

/** The terms' rows for one trial: each term where `record` has it apply. */
export const trialTermRows = (task: ShelfTask, record: TrialRecord, stepSeconds: number) => termRows(task, (term) => ({
  bars: presence(record.terms?.[term]?.steps ?? [], stepSeconds), problems: record.terms?.[term]?.problems,
}));

/**
 * One trial's rows: each epoch instance where it began and ended, one that ends on a condition marked with how; each
 * signal present where the trial has it; each term where it applies.
 */
export function trialRows(task: ShelfTask, record: TrialRecord, stepSeconds: number): ShelfRow[] {
  return [
    {
      kind: 'segments', key: 'epochs', name: studioName('epochs'),
      segments: record.timeline.flatMap(({ epoch, label, start, end, outcome }, index) => end > start ? [{
        key: `${epoch}:${index}`, name: label, start: start * stepSeconds, end: end * stepSeconds,
        selects: named(task.document, `epoch:${epoch}`), ended: endedBy(outcome),
      }] : []),
    },
    ...task.signals.map((signal): ShelfRow => ({
      kind: 'bars', key: `signal:${signal.id}`, name: signal.name, selects: named(task.document, `signal:${signal.id}`),
      bars: presence(signalSteps(record, signal.id), stepSeconds),
    })),
    ...trialTermRows(task, record, stepSeconds),
  ];
}

/**
 * Where a share of trials is above 0, in runs: heavy where every trial is counted, light where only some are. The
 * share's first entry is the step `first` steps from zero.
 */
function shareBars(share: readonly number[], stepSeconds: number, first = 0): ShelfBar[] {
  const bars: ShelfBar[] = [];
  share.forEach((value, index) => {
    if (value <= 0) return;
    const step = first + index;
    const some = value < 1;
    const last = bars.at(-1);
    if (last && last.end === step * stepSeconds && !!last.some === some) bars[bars.length - 1] = { ...last, end: (step + 1) * stepSeconds };
    else bars.push({ start: step * stepSeconds, end: (step + 1) * stepSeconds, ...some && { some } });
  });
  return bars;
}

/**
 * A batch of trials' rows: one per epoch, since epochs of different trials overlap in time, one per signal and one per
 * term; each heavy where every trial is in the epoch, has the signal or has the term apply, light where some do.
 */
export function batchRows(task: ShelfTask, aggregate: TrialAggregate, stepSeconds: number): ShelfRow[] {
  return [
    ...aggregate.epochs.map(({ epoch, label, share }): ShelfRow => ({
      kind: 'bars', key: `epoch:${epoch}`, name: label, selects: named(task.document, `epoch:${epoch}`), bars: shareBars(share, stepSeconds),
    })),
    ...task.signals.map((signal): ShelfRow => ({
      kind: 'bars', key: `signal:${signal.id}`, name: signal.name, selects: named(task.document, `signal:${signal.id}`),
      bars: shareBars(aggregate.signals[signal.id]?.presence ?? [], stepSeconds),
    })),
    ...termRows(task, (term) => ({
      bars: shareBars(aggregate.terms?.[term]?.steps ?? [], stepSeconds), problems: aggregate.terms?.[term]?.problems,
    })),
  ];
}

/**
 * The bars of the steps `held` gives of each trial of every condition, on time from Align's epoch: heavy where every
 * trial holds, light where some do.
 */
function alignedBars(mean: TrialMean, records: ReadonlyMap<string, TrialRecord>, stepSeconds: number,
  held: (record: TrialRecord) => readonly unknown[]): ShelfBar[] {
  const counts = new Map<number, number>();
  let trials = 0;
  for (const condition of mean.conditions) {
    for (const [id, { origin }] of Object.entries(condition.trials)) {
      const record = records.get(id);
      if (!record) continue;
      trials += 1;
      const shift = Math.round(origin / stepSeconds);
      held(record).forEach((value, step) => {
        if (value) counts.set(step - shift, (counts.get(step - shift) ?? 0) + 1);
      });
    }
  }
  if (!counts.size) return [];
  const first = Math.min(...counts.keys());
  const share = Array.from({ length: Math.max(...counts.keys()) - first + 1 }, (_, index) => (counts.get(first + index) ?? 0) / trials);
  return shareBars(share, stepSeconds, first);
}

/**
 * Under Mean, on time from Align's epoch: the epochs row, each epoch instance from its mean start to its mean end over
 * every trial of every condition with a whisker from where its end falls least to greatest; and one row per signal
 * and per term, over every trial `records` holds.
 */
export function meanRows(task: ShelfTask, mean: TrialMean, records: readonly TrialRecord[], stepSeconds: number): ShelfRow[] {
  const pooled = new Map<string, { epoch: string; label: string; trials: number; start: number; end: number; least: number; greatest: number }>();
  for (const condition of mean.conditions) {
    const trials = Object.keys(condition.trials).length;
    for (const { epoch, label, occurrence, start, end } of condition.epochs) {
      const key = `${epoch}:${occurrence}`;
      const prior = pooled.get(key);
      pooled.set(key, prior ? {
        ...prior, trials: prior.trials + trials, start: prior.start + start.mean * trials, end: prior.end + end.mean * trials,
        least: Math.min(prior.least, end.lower), greatest: Math.max(prior.greatest, end.upper),
      } : { epoch, label, trials, start: start.mean * trials, end: end.mean * trials, least: end.lower, greatest: end.upper });
    }
  }
  const segments = [...pooled].flatMap(([key, { epoch, label, trials, start, end, least, greatest }]): ShelfSegment[] =>
    trials > 0 && end > start ? [{
      key, name: label, start: start / trials, end: end / trials, selects: named(task.document, `epoch:${epoch}`),
      ...greatest > least && { spread: { least, greatest } },
    }] : []).sort((a, b) => a.start - b.start);
  const byId = new Map(records.map((record) => [record.id, record]));
  return [
    { kind: 'segments', key: 'epochs', name: studioName('epochs'), segments },
    ...task.signals.map((signal): ShelfRow => ({
      kind: 'bars', key: `signal:${signal.id}`, name: signal.name, selects: named(task.document, `signal:${signal.id}`),
      bars: alignedBars(mean, byId, stepSeconds, (record) => signalSteps(record, signal.id)),
    })),
    ...termRows(task, (term) => ({
      bars: alignedBars(mean, byId, stepSeconds, (record) => record.terms?.[term]?.steps ?? []),
      problems: records.flatMap((record) => record.terms?.[term]?.problems ?? []),
    })),
  ];
}
