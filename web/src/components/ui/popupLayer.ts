import { createContext, useContext } from 'react';

/** The element popups portal into; a modal dialog sets it so its popups stay in the top layer. */
export const PopupLayer = createContext<HTMLElement | null>(null);

export function usePopupLayer() {
  return useContext(PopupLayer) ?? undefined;
}
