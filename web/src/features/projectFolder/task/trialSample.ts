import type { Problem, TrialAggregate, TrialRecord } from '@/generated/studioContracts';

/** A signal's value at each step of one trial, one number per component; null where the signal is absent. */
export type SignalSteps = readonly (readonly number[] | null)[];

/** The 5th, 25th, 50th, 75th and 95th percentiles of one component at one step, over the trials where it is present. */
export type Percentiles = readonly [number, number, number, number, number];

/** A signal over several trials: per step, the share of trials in which it is present and, if numeric, each component's percentiles. */
export interface SignalAggregate {
  readonly presence: readonly number[];
  /** Per step, per component; null at a step where no trial has the signal. */
  readonly percentiles?: readonly (readonly Percentiles[] | null)[];
}

/**
 * What every view that shows a trial is given by the clock: the task's definition (the nominal trial), one trial's
 * record of the `count` the clock holds, or the aggregate of all of them. A view handles each case and does not know
 * whether a record was sampled or recorded.
 */
export type TrialShown =
  | { readonly kind: 'definition' }
  | { readonly kind: 'one'; readonly index: number; readonly count: number; readonly record: TrialRecord }
  | { readonly kind: 'all'; readonly count: number; readonly aggregate: TrialAggregate };

/** Why no trial can be shown: the Problems the sampling route answered with. */
export interface TrialsUnavailable {
  readonly kind: 'problems';
  readonly problems: readonly Problem[];
}

const numbers = (value: unknown): number[] => Array.isArray(value) ? value.flatMap(numbers)
  : typeof value === 'boolean' ? [Number(value)] : typeof value === 'number' ? [value] : [];

/** The signal `name`'s value at each step of `record`, null where it is absent. */
export function signalSteps(record: TrialRecord, name: string): SignalSteps {
  const values = record.inputs[name];
  const present = record.present[name];
  return (Array.isArray(values) ? values : []).map((value, step) =>
    Array.isArray(present) && present[step] ? numbers(value) : null);
}

/** The signal `name` over the trials `aggregate` holds. */
export function signalAggregate(aggregate: TrialAggregate, name: string): SignalAggregate | undefined {
  const signal = aggregate.signals[name];
  return signal && {
    presence: signal.presence,
    ...signal.percentiles && { percentiles: signal.percentiles as (Percentiles[] | null)[] },
  };
}

/** The steps at which each epoch instance after the first of `record` begins. */
export const epochBoundaries = (record: TrialRecord) => record.timeline.slice(1).map(({ start }) => start);

function valueText(value: unknown): string {
  if (typeof value === 'boolean') return String(value);
  if (typeof value === 'number') return String(Number(value.toPrecision(3)));
  if (Array.isArray(value)) return `[${value.map(valueText).join(', ')}]`;
  return '';
}

/** The range of a variable's drawn values, component by component, as text: one value where they all agree. */
function rangeText(least: unknown, greatest: unknown): string {
  if (Array.isArray(least) && Array.isArray(greatest)) return `[${least.map((low, index) => rangeText(low, greatest[index])).join(', ')}]`;
  if (least === greatest) return valueText(least);
  return typeof least === 'boolean' ? `${valueText(least)}, ${valueText(greatest)}` : `${valueText(least)}…${valueText(greatest)}`;
}

/**
 * Each of `variables`' values in what is shown, as text, by variable id: its drawn value in one trial, or the range of
 * its values over all of them; nothing for the definition.
 */
export function drawnVariables(shown: TrialShown | undefined, variables: readonly { readonly id: string; readonly name: string }[]) {
  if (!shown) return undefined;
  switch (shown.kind) {
    case 'definition': return undefined;
    case 'one': return Object.fromEntries(variables.flatMap(({ id, name }) =>
      name in shown.record.variables ? [[id, valueText(shown.record.variables[name])]] : []));
    case 'all': return Object.fromEntries(variables.flatMap(({ id, name }) => {
      const range = shown.aggregate.variables[name];
      return range ? [[id, rangeText(range.least, range.greatest)]] : [];
    }));
  }
}
