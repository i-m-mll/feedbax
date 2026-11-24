import { useState } from 'react';
import { useStore } from 'zustand';
import { createStore } from 'zustand/vanilla';
import { studioFormat } from '@/components/activity/studioVocabulary';

/**
 * A trial the clock can show: the task at its nominal values, the `index`-th trial sampled from the task without a
 * model, or the `index`-th trial the selected run recorded. Indices count from 0.
 */
export type TrialEntry =
  | { readonly kind: 'nominal' }
  | { readonly kind: 'sample'; readonly index: number }
  | { readonly kind: 'recorded'; readonly index: number };

/** The clock's window state: the picked entry, the seed and count samples are drawn with, and whether the shelf is open. */
export interface TrialClock {
  readonly picked?: TrialEntry;
  readonly seed: number;
  readonly count: number;
  readonly shelf: boolean;
}

export const defaultTrialClock: TrialClock = { seed: 0, count: 8, shelf: false };

/** The most trials one sample draws, as the sampling route allows. */
export const maxSampleCount = 64;

/** The entries that exist for the selection: whether it has a nominal trial, whether it can be sampled, and how many trials its run recorded. */
export interface TrialOffer {
  readonly nominal: boolean;
  readonly samples: boolean;
  readonly recorded: number;
}

const index = (value: unknown) => typeof value === 'number' && Number.isInteger(value) && value >= 0;

function entryOf(value: unknown): TrialEntry | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const entry = value as { kind?: unknown; index?: unknown };
  if (entry.kind === 'nominal') return { kind: 'nominal' };
  if ((entry.kind === 'sample' || entry.kind === 'recorded') && index(entry.index)) {
    return { kind: entry.kind, index: entry.index as number };
  }
  return undefined;
}

/** The clock's state as stored with the window's arrangement, or nothing when the stored value is not one. */
export function trialClockOf(value: unknown): TrialClock | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const stored = value as Partial<Record<keyof TrialClock, unknown>>;
  const count = typeof stored.count === 'number' && Number.isInteger(stored.count) && stored.count >= 1 && stored.count <= maxSampleCount
    ? stored.count : defaultTrialClock.count;
  const picked = entryOf(stored.picked);
  return {
    ...picked && { picked },
    seed: typeof stored.seed === 'number' && Number.isInteger(stored.seed) ? stored.seed : defaultTrialClock.seed,
    count,
    shelf: stored.shelf === true,
  };
}

/** The entries the picker lists for `offer`, in order. */
export function offeredEntries(offer: TrialOffer, count: number): TrialEntry[] {
  return [
    ...offer.nominal ? [{ kind: 'nominal' } as const] : [],
    ...offer.samples ? Array.from({ length: count }, (_, index) => ({ kind: 'sample', index }) as const) : [],
    ...Array.from({ length: offer.recorded }, (_, index) => ({ kind: 'recorded', index }) as const),
  ];
}

const offered = (entry: TrialEntry, offer: TrialOffer, count: number) =>
  entry.kind === 'nominal' ? offer.nominal : entry.kind === 'sample' ? offer.samples && entry.index < count : entry.index < offer.recorded;

/**
 * The entry shown: the picked one while it exists for the selection, else the most specific that does (the first
 * recorded trial, then the nominal trial, then the first sample); nothing when no entry exists.
 */
export function shownEntry(clock: TrialClock, offer: TrialOffer): TrialEntry | undefined {
  if (clock.picked && offered(clock.picked, offer, clock.count)) return clock.picked;
  if (offer.recorded) return { kind: 'recorded', index: 0 };
  if (offer.nominal) return { kind: 'nominal' };
  if (offer.samples) return { kind: 'sample', index: 0 };
  return undefined;
}

/** What the picker says an entry is: `nominal`, `sample k of n`, `trial k of n`. */
export function entryText(entry: TrialEntry, count: number, recorded: number) {
  if (entry.kind === 'nominal') return studioFormat('nominalTrial');
  return studioFormat(entry.kind === 'sample' ? 'sampleOf' : 'trialOf', {
    index: entry.index + 1, count: entry.kind === 'sample' ? count : recorded,
  });
}

export const sameEntry = (a: TrialEntry | undefined, b: TrialEntry | undefined) =>
  !!a && !!b && a.kind === b.kind && (a.kind === 'nominal' || a.index === (b as { index: number }).index);

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

/** The playhead's time in the trial `scope` names, which is its start until it is moved there, and its setter. */
export function usePlayhead(store: PlayheadStore, scope: string) {
  const playhead = useStore(store);
  return [
    playhead.scope === scope ? playhead.time : 0,
    (time: number) => store.setState({ scope, time }),
  ] as const;
}
