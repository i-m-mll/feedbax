import { useState } from 'react';
import { useStore } from 'zustand';
import { createStore } from 'zustand/vanilla';
import { studioFormat } from '@/components/activity/studioVocabulary';

/**
 * Which trials the clock shows: the task at its nominal values, the samples drawn from the task with the clock's seed
 * and count, or the trials the selected run recorded.
 */
export type TrialSet = 'nominal' | 'samples' | 'recorded';

/**
 * How the clock shows its trials: one of them, or all of them at once. `index`, from 0, is the trial picked: the one
 * shown, or the one drawn heavier among all.
 */
export interface TrialShowing {
  readonly kind: 'one' | 'all';
  readonly index: number;
}

/** What the clock shows: which trials, and how. Every view that shows a trial draws this and keeps no pick of its own. */
export interface TrialPick {
  readonly trials: TrialSet;
  readonly showing: TrialShowing;
}

/**
 * The clock's window state: what was picked, the seed and count samples are drawn with, by view whether the shelf is
 * open there, and whether drawn paths end at the playhead.
 */
export interface TrialClock {
  readonly picked?: TrialPick;
  readonly seed: number;
  readonly count: number;
  readonly shelf: Readonly<Record<string, boolean>>;
  readonly trails: boolean;
}

export const defaultTrialClock: TrialClock = { seed: 0, count: 8, shelf: {}, trails: false };

/** Whether the shelf is open in `view`: as the user left it there, else open in the Task view and closed elsewhere. */
export const shelfOpen = (clock: TrialClock, view: string) => clock.shelf[view] ?? view === 'task';

/**
 * The trials that exist for the selection: whether it has a nominal trial, the most trials it can be sampled in (the
 * step's batch size; 0 when it cannot be sampled), and how many trials its run recorded.
 */
export interface TrialOffer {
  readonly nominal: boolean;
  readonly samples: number;
  readonly recorded: number;
}

/** How many samples are drawn: the clock's count, up to the step's batch size. */
export const sampleCount = (clock: TrialClock, offer: TrialOffer) => Math.min(clock.count, offer.samples);

/** How many trials `trials` holds for `offer`. */
export const setSize = (trials: TrialSet, clock: TrialClock, offer: TrialOffer) =>
  trials === 'nominal' ? Number(offer.nominal) : trials === 'samples' ? sampleCount(clock, offer) : offer.recorded;

/** Whether `trials` can be shown all at once: a set of more than one trial that has an aggregate. */
export const aggregates = (trials: TrialSet) => trials !== 'nominal';

const whole = (value: unknown, least: number) => typeof value === 'number' && Number.isInteger(value) && value >= least;

function pickOf(value: unknown): TrialPick | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const { trials, showing } = value as { trials?: unknown; showing?: { kind?: unknown; index?: unknown } };
  if (trials !== 'nominal' && trials !== 'samples' && trials !== 'recorded') return undefined;
  const kind = showing?.kind;
  if ((kind !== 'one' && kind !== 'all') || !whole(showing?.index, 0)) return undefined;
  return { trials, showing: { kind, index: showing!.index as number } };
}

/** The clock's state as stored with the window's arrangement, or nothing when the stored value is not one. */
export function trialClockOf(value: unknown): TrialClock | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const stored = value as Partial<Record<keyof TrialClock, unknown>>;
  const picked = pickOf(stored.picked);
  const shelf = stored.shelf && typeof stored.shelf === 'object'
    ? Object.fromEntries(Object.entries(stored.shelf).filter((entry): entry is [string, boolean] => typeof entry[1] === 'boolean'))
    : {};
  return {
    ...picked && { picked },
    seed: whole(stored.seed, Number.MIN_SAFE_INTEGER) ? stored.seed as number : defaultTrialClock.seed,
    count: whole(stored.count, 1) ? stored.count as number : defaultTrialClock.count,
    shelf,
    trails: stored.trails === true,
  };
}

/** The single trials the picker lists for `offer`, in order. */
export function offeredPicks(offer: TrialOffer, count: number): TrialPick[] {
  const ones = (trials: TrialSet, length: number) =>
    Array.from({ length }, (_, index): TrialPick => ({ trials, showing: { kind: 'one', index } }));
  return [...ones('nominal', Number(offer.nominal)), ...ones('samples', count), ...ones('recorded', offer.recorded)];
}

function offered({ trials, showing }: TrialPick, clock: TrialClock, offer: TrialOffer) {
  return showing.index < setSize(trials, clock, offer) && (showing.kind === 'one' || aggregates(trials));
}

/**
 * What is shown: the pick while it exists for the selection, else the most specific single trial that does (the first
 * recorded trial, then the nominal trial, then the first sample); nothing when no trial exists.
 */
export function shownPick(clock: TrialClock, offer: TrialOffer): TrialPick | undefined {
  if (clock.picked && offered(clock.picked, clock, offer)) return clock.picked;
  const first = (trials: TrialSet): TrialPick => ({ trials, showing: { kind: 'one', index: 0 } });
  if (offer.recorded) return first('recorded');
  if (offer.nominal) return first('nominal');
  if (offer.samples) return first('samples');
  return undefined;
}

/** What the picker says is shown: `nominal`, `sample k of n`, `trial k of n`, or `n trials`. */
export function pickText({ trials, showing }: TrialPick, clock: TrialClock, offer: TrialOffer) {
  const count = setSize(trials, clock, offer);
  switch (showing.kind) {
    case 'all': return studioFormat('trialCount', { count });
    case 'one': return trials === 'nominal' ? studioFormat('nominalTrial')
      : studioFormat(trials === 'samples' ? 'sampleOf' : 'trialOf', { index: showing.index + 1, count });
  }
}

export const samePick = (a: TrialPick | undefined, b: TrialPick | undefined) =>
  !!a && !!b && a.trials === b.trials && a.showing.kind === b.showing.kind && a.showing.index === b.showing.index;

/** The frame of `frameTimes` nearest `time`, which views read a trial's values at. */
export function frameAt(frameTimes: readonly number[], time: number): number {
  let best = 0;
  for (let index = 1; index < frameTimes.length; index += 1) {
    if (Math.abs(frameTimes[index] - time) < Math.abs(frameTimes[best] - time)) best = index;
  }
  return best;
}

/** The playhead: the time, in seconds, in the trial `scope` names. It is saved nowhere. */
interface Playhead {
  readonly scope: string;
  readonly time: number;
}

export type PlayheadStore = ReturnType<typeof usePlayheadStore>;

/** The window's one playhead, which the Scene, the shelf, and every plot over time read and move. */
export function usePlayheadStore() {
  const [store] = useState(() => createStore<Playhead>(() => ({ scope: '', time: 0 })));
  return store;
}

/**
 * The playhead's time in the trial `scope` names, and its setter. A trial the playhead has not been moved in starts at
 * 0, so the time returns to the start whenever the shown trial changes.
 */
export function usePlayhead(store: PlayheadStore, scope: string) {
  const playhead = useStore(store);
  return [
    playhead.scope === scope ? playhead.time : 0,
    (time: number) => store.setState({ scope, time }),
  ] as const;
}
