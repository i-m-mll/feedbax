import { useState } from 'react';

/** An unsaved edit of a saved value; it is discarded whenever the saved value changes, so the view stays mounted. */
export function useDraft<T>(saved: T): [T, (draft: T) => void] {
  const key = JSON.stringify(saved);
  const [state, setState] = useState({ key, draft: saved });
  if (state.key !== key) {
    setState({ key, draft: saved });
    return [saved, (draft) => setState({ key, draft })];
  }
  return [state.draft, (draft) => setState({ key, draft })];
}
