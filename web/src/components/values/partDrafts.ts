import { useSyncExternalStore } from 'react';
import type { CallDraft, DrawDraft } from './fieldModel';

/**
 * A value's parts that are not yet a value: a draw or a call with an argument unfilled, or a Link whose name is not
 * chosen. Nothing is written for one until its parts are filled.
 */
export type PartDraft =
  | { readonly kind: 'draw'; readonly draw: DrawDraft }
  | { readonly kind: 'formula' | 'shape'; readonly call: CallDraft }
  | { readonly kind: 'link' };

/**
 * Drafts by field id. Kept outside the field, since a table cell's kind menu starts the draft whose parts the
 * Inspector shows.
 */
const drafts = new Map<string, PartDraft>();
const listeners = new Set<() => void>();

export function setPartDraft(id: string, draft: PartDraft | undefined) {
  if (draft) drafts.set(id, draft);
  else if (!drafts.delete(id)) return;
  for (const listener of listeners) listener();
}

export const partDraft = (id: string) => drafts.get(id);

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export function usePartDraft(id: string) {
  return useSyncExternalStore(subscribe, () => drafts.get(id));
}
