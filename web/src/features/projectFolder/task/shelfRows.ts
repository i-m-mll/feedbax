import { createElement } from 'react';
import { studioName } from '@/components/activity/studioVocabulary';
import type { ShelfBar, ShelfRow, ShelfSegment } from '@/components/timeline/Shelf';
import type { NamedElementAddress } from '@/features/projectWindow/elementSelection';
import { viewGlyphs } from '@/features/projectWindow/ProjectWindow';
import type { EpochBounds } from './taskModel';
import { ModelAddressSchema, type Problem, type TrialAggregate, type TrialMean, type TrialRecord } from '@/generated/studioContracts';
import { endedBy, signalSteps } from './trialSample';
import { addressKey, type ModelIntervention } from '../model/modelGraph';

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
  /** The Model's interventions, and the Model document whose elements their rows select. */
  readonly interventions?: {
    readonly document: string;
    readonly items: readonly ModelIntervention[];
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

/**
 * One row per intervention, with the bars `drawn` gives of its steps' mask, keyed by its address: each of the Model's,
 * then each one only the records name, which a run's own Model (a Set entry's) adds.
 */
function interventionRows(task: ShelfTask, recorded: readonly string[], drawn: (key: string) => ShelfBar[]): ShelfRow[] {
  const interventions = task.interventions;
  if (!interventions) return [];
  const known = new Map(interventions.items.map((item) => [addressKey(item.address), item]));
  const added = [...new Set(recorded)].filter((key) => !known.has(key)).flatMap((key) => {
    const parsed = ModelAddressSchema.safeParse(JSON.parse(key));
    return parsed.success && parsed.data.element.kind === 'attachment' ? [[key, { address: parsed.data, name: studioName('intervention') }] as const] : [];
  });
  const glyph = createElement('span', { 'aria-hidden': true, className: 'shrink-0 text-xs' }, '◇');
  return [...known, ...added].map(([key, { address, name }]): ShelfRow => ({
    kind: 'bars', key: `intervention:${key}`, name, glyph, selects: { document: interventions.document, address }, bars: drawn(key),
  }));
}

/** The rows one trial's record gives beyond its epochs and signals: each term where it applies, each intervention where it acts. */
export const trialTermRows = (task: ShelfTask, record: TrialRecord, stepSeconds: number) => [
  ...termRows(task, (term) => ({
    bars: presence(record.terms?.[term]?.steps ?? [], stepSeconds), problems: record.terms?.[term]?.problems,
  })),
  ...interventionRows(task, Object.keys(record.intervention_windows ?? {}),
    (key) => presence(record.intervention_windows?.[key] ?? [], stepSeconds)),
];

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
 * Where a share of trials is above 0, in runs: heavy where every trial is counted, light where only some are. Share
 * `index` is of the interval from `times[index]` to `times[index + 1]`.
 */
function shareBars(share: readonly number[], times: readonly number[]): ShelfBar[] {
  const bars: ShelfBar[] = [];
  share.forEach((value, index) => {
    if (value <= 0) return;
    const some = value < 1;
    const last = bars.at(-1);
    if (last && last.end === times[index] && !!last.some === some) bars[bars.length - 1] = { ...last, end: times[index + 1] };
    else bars.push({ start: times[index], end: times[index + 1], ...some && { some } });
  });
  return bars;
}

/** A share per step's edges, in seconds. */
const stepEdges = (share: readonly number[], stepSeconds: number) =>
  Array.from({ length: share.length + 1 }, (_, step) => step * stepSeconds);
const stepShareBars = (share: readonly number[], stepSeconds: number) => shareBars(share, stepEdges(share, stepSeconds));

/**
 * A batch of trials' rows: one per epoch, since epochs of different trials overlap in time, one per signal, term and
 * intervention; each heavy where every trial is in the epoch, has the signal, has the term apply or has the
 * intervention act, light where some do.
 */
export function batchRows(task: ShelfTask, aggregate: TrialAggregate, stepSeconds: number): ShelfRow[] {
  return [
    ...aggregate.epochs.map(({ epoch, label, share }): ShelfRow => ({
      kind: 'bars', key: `epoch:${epoch}`, name: label, selects: named(task.document, `epoch:${epoch}`), bars: stepShareBars(share, stepSeconds),
    })),
    ...task.signals.map((signal): ShelfRow => ({
      kind: 'bars', key: `signal:${signal.id}`, name: signal.name, selects: named(task.document, `signal:${signal.id}`),
      bars: stepShareBars(aggregate.signals[signal.id]?.presence ?? [], stepSeconds),
    })),
    ...termRows(task, (term) => ({
      bars: stepShareBars(aggregate.terms?.[term]?.steps ?? [], stepSeconds), problems: aggregate.terms?.[term]?.problems,
    })),
    ...interventionRows(task, Object.keys(aggregate.interventions ?? {}),
      (key) => stepShareBars(aggregate.interventions?.[key] ?? [], stepSeconds)),
  ];
}

/**
 * Under Mean, on time from Align's epoch, the Service's shelf: the epochs row, each epoch instance from its mean start
 * to its mean end over every trial of every condition with a whisker from where its end falls least to greatest; and
 * one row per signal, term and intervention, each heavy where every trial holds and light where some do.
 */
export function meanRows(task: ShelfTask, { shelf }: TrialMean): ShelfRow[] {
  const segments = shelf.epochs.flatMap(({ epoch, label, occurrence, start, end }): ShelfSegment[] => end.mean > start.mean ? [{
    key: `${epoch}:${occurrence}`, name: label, start: start.mean, end: end.mean, selects: named(task.document, `epoch:${epoch}`),
    ...end.upper > end.lower && { spread: { least: end.lower, greatest: end.upper } },
  }] : []).sort((a, b) => a.start - b.start);
  return [
    { kind: 'segments', key: 'epochs', name: studioName('epochs'), segments },
    ...task.signals.map((signal): ShelfRow => ({
      kind: 'bars', key: `signal:${signal.id}`, name: signal.name, selects: named(task.document, `signal:${signal.id}`),
      bars: shareBars(shelf.signals[signal.id] ?? [], shelf.times),
    })),
    ...termRows(task, (term) => ({
      bars: shareBars(shelf.terms[term]?.steps ?? [], shelf.times), problems: shelf.terms[term]?.problems,
    })),
    ...interventionRows(task, Object.keys(shelf.interventions ?? {}), (key) => shareBars(shelf.interventions?.[key] ?? [], shelf.times)),
  ];
}
