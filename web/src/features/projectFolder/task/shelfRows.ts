import { studioName } from '@/components/activity/studioVocabulary';
import type { ShelfBar, ShelfRow, ShelfSegment } from '@/components/timeline/Shelf';
import type { NamedElementAddress } from '@/features/projectWindow/elementSelection';
import type { EpochBounds } from './taskModel';
import type { SampledTrial, SignalSteps } from './trialSample';

/** What the shelf draws of a task document: its epochs' bounds in order, and the epochs each signal's cells cover. */
export interface ShelfTask {
  /** The task document, whose elements the rows select. */
  readonly document: string;
  readonly epochs: readonly EpochBounds[];
  readonly signals: readonly {
    readonly id: string;
    readonly name: string;
    readonly cells: readonly { readonly first: string; readonly last: string }[];
  }[];
  /** The trial's length, in seconds. */
  readonly duration: number;
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
      key: epoch.id, name: epoch.name, start, end, selects: named(task.document, `epoch:${epoch.id}`),
      ...epoch.ends === 'drawn' && epoch.greatest > epoch.least && {
        spread: { least: start + epoch.least, greatest: Math.min(task.duration, start + epoch.greatest) },
      },
      ...epoch.ends === 'condition' && { dashedFrom: start + epoch.least },
    };
  });
  const at = new Map(segments.map((segment) => [segment.key, segment]));
  return [
    { kind: 'segments', key: 'epochs', name: studioName('epochs'), segments },
    ...task.signals.map((signal): ShelfRow => ({
      kind: 'bars', key: `signal:${signal.id}`, name: signal.name, selects: named(task.document, `signal:${signal.id}`),
      bars: joined(signal.cells.flatMap(({ first, last }) => {
        const from = at.get(first);
        const to = at.get(last);
        return from && to && to.end > from.start ? [{ start: from.start, end: to.end }] : [];
      })),
    })),
  ];
}

/** Where a signal is present in one trial: its runs of steps with a value. */
function presence(steps: SignalSteps, stepSeconds: number): ShelfBar[] {
  const bars: ShelfBar[] = [];
  let start: number | undefined;
  steps.forEach((value, step) => {
    if (value && start === undefined) start = step;
    if (!value && start !== undefined) { bars.push({ start: start * stepSeconds, end: step * stepSeconds }); start = undefined; }
  });
  if (start !== undefined) bars.push({ start: start * stepSeconds, end: steps.length * stepSeconds });
  return bars;
}

/** One trial's rows: each epoch instance where it began and ended, each signal present where the trial has it. */
export function trialRows(task: ShelfTask, trial: SampledTrial, stepSeconds: number): ShelfRow[] {
  return [
    {
      kind: 'segments', key: 'epochs', name: studioName('epochs'),
      segments: trial.intervals.flatMap((interval, index) => interval.end > interval.start ? [{
        key: `${interval.id}:${index}`, name: interval.name, start: interval.start, end: interval.end,
        selects: named(task.document, `epoch:${interval.id}`),
      }] : []),
    },
    ...task.signals.map((signal): ShelfRow => ({
      kind: 'bars', key: `signal:${signal.id}`, name: signal.name, selects: named(task.document, `signal:${signal.id}`),
      bars: presence(trial.signals[signal.id] ?? [], stepSeconds),
    })),
  ];
}
