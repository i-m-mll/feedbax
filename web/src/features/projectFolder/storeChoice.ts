import { isHttpConflict, isServiceFailure, isUnanswered } from '@/api/request';
import { shownError } from '@/api/shownErrors';
import { studioFormat, studioName, studioReason } from '@/components/activity/studioVocabulary';
import type { AskChoice, Choice } from '@/components/ui/ChoiceMenu';
import type { StoreKind, StudioSavedTrainKey, StudioStoreChoice } from '@/generated/studioContracts';
import { createContext, useContext } from 'react';
import { ProjectEditConflictError } from './projectEditController';

/** Asks at `anchor` which store a pending edit is written to; `undefined` discards the edit. */
export type AskStore = (anchor: Element, choices: readonly StudioStoreChoice[]) => Promise<string | undefined>;

/** The step and run labels a store's tip names. */
export interface RunNames {
  step(id: string): string;
  run(key: StudioSavedTrainKey): string;
}

const byStep = new Set<StoreKind>(['all_readers', 'selected_readers', 'whole_step']);

/** A store's entry: the service's label, and a tip naming the steps it changes, or its runs where it changes some of a step's. */
export function storeEntry(choice: StudioStoreChoice, names: RunNames): Choice {
  const named = byStep.has(choice.kind) ? choice.steps.map(names.step)
    : choice.runs.map((key) => choice.steps.length > 1
      ? studioFormat('stepRun', { step: names.step(key.step), run: names.run(key) })
      : names.run(key));
  return { id: choice.id, label: choice.label, tip: named.join(', ') };
}

export function askStoreWith(ask: AskChoice, names: RunNames): AskStore {
  return (anchor, choices) => ask(anchor, studioName('applyTo'), choices.map((choice) => storeEntry(choice, names)));
}

export const AskStoreContext = createContext<AskStore | null>(null);

export function useAskStore() {
  const ask = useContext(AskStoreContext);
  if (!ask) throw new Error('useAskStore needs an AskStoreContext');
  return ask;
}

/** A store choice the service no longer takes, or a request that failed after one was chosen: the command's own failure. */
export class StoreChoiceFailed extends Error {}

/**
 * Sends an edit's request. When the service answers with the stores the edit can be written to, it asks at the edited
 * element and sends the same request with the chosen one; `undefined` when the ask is dismissed, and nothing is written.
 * A choice that is out of date, or a request that fails after it, fails the command once the project is read again.
 */
export async function sendWithStore<T>(
  ask: AskStore, anchor: () => Element, refresh: () => Promise<unknown>,
  send: (choice: string | undefined) => Promise<T>, offered: (result: T) => readonly StudioStoreChoice[] | null | undefined,
): Promise<T | undefined> {
  const first = await send(undefined);
  const choices = offered(first);
  if (!choices?.length) return first;
  const choice = await ask(anchor(), choices);
  if (!choice) return undefined;
  const fail = async (reason: string): Promise<never> => {
    await refresh().catch(() => undefined);
    throw new StoreChoiceFailed(reason);
  };
  let chosen: T;
  try {
    chosen = await send(choice);
  } catch (error) {
    if (error instanceof ProjectEditConflictError || isHttpConflict(error)) {
      return fail(studioReason('nameChangedDuringEdit', studioName('project')));
    }
    if (isServiceFailure(error) && !isUnanswered(error)) return fail(shownError(error));
    throw error;
  }
  if (offered(chosen)?.length) {
    console.error('The service asked again after a store was chosen.', { choice, answer: chosen });
    return fail(studioReason('nameChangedDuringEdit', studioName('project')));
  }
  return chosen;
}
