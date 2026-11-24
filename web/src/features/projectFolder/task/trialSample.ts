import type { TimelineInterval, TimelineShare } from '@/components/timeline/EpochTimeline';

/** A signal's value at each step of one trial, one number per component; null where the signal is absent. */
export type SignalSteps = readonly (readonly number[] | null)[];

/** One trial sampled from the task without a model: its epochs' intervals, each variable's drawn value, each signal's values. */
export interface SampledTrial {
  readonly intervals: readonly TimelineInterval[];
  /** Each variable's drawn value, as text, by variable id. */
  readonly variables: Readonly<Record<string, string>>;
  readonly signals: Readonly<Record<string, SignalSteps>>;
}

/** The 5th, 25th, 50th, 75th and 95th percentiles of one component at one step, over the trials where it is present. */
export type Percentiles = readonly [number, number, number, number, number];

/** A signal over a batch: per step, the share of trials in which it is present and, if numeric, each component's percentiles. */
export interface SignalAggregate {
  readonly presence: readonly number[];
  /** Per step, per component; null at a step where no trial has the signal. */
  readonly percentiles?: readonly (readonly Percentiles[] | null)[];
}

/** A batch of sampled trials as its aggregate. */
export interface TrialBatch {
  readonly count: number;
  readonly epochs: readonly TimelineShare[];
  /** The range of each variable's drawn values, as text, by variable id. */
  readonly variables: Readonly<Record<string, string>>;
  readonly signals: Readonly<Record<string, SignalAggregate>>;
}

/** What the view shows of the task: its definition, one sampled trial of a set, or the set's aggregate. */
export type TrialShown =
  | { readonly kind: 'definition' }
  | { readonly kind: 'trial'; readonly index: number; readonly count: number; readonly trial: SampledTrial }
  | { readonly kind: 'batch'; readonly batch: TrialBatch };
