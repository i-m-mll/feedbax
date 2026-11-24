import { studioFormat, studioName } from '@/components/activity/studioVocabulary';
import type { AskChoice, Choice } from '@/components/ui/ChoiceMenu';
import type { StoreKind, StudioSavedTrainKey, StudioStoreChoice } from '@/generated/studioContracts';

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
