import { create } from 'zustand';

interface LayoutStoreState {
  topCollapsed: boolean;
  bottomCollapsed: boolean;
  bottomHeight: number;
  toggleTop: () => void;
  toggleBottom: () => void;
  setBottomHeight: (height: number) => void;
}

const DEFAULT_BOTTOM_HEIGHT = 280;
const MIN_BOTTOM_HEIGHT = 200;
const MAX_BOTTOM_HEIGHT = 520;

export const useLayoutStore = create<LayoutStoreState>((set, get) => ({
  topCollapsed: false,
  bottomCollapsed: true,
  bottomHeight: DEFAULT_BOTTOM_HEIGHT,
  toggleTop: () => {
    set((state) => {
      const nextTop = !state.topCollapsed;
      const nextBottom = state.bottomCollapsed && nextTop ? false : state.bottomCollapsed;
      return {
        topCollapsed: nextTop,
        bottomCollapsed: nextBottom,
      };
    });
  },
  toggleBottom: () => {
    set((state) => {
      const nextBottom = !state.bottomCollapsed;
      const nextTop = state.topCollapsed && nextBottom ? false : state.topCollapsed;
      return {
        topCollapsed: nextTop,
        bottomCollapsed: nextBottom,
      };
    });
  },
  setBottomHeight: (height) => {
    const clamped = Math.max(MIN_BOTTOM_HEIGHT, Math.min(MAX_BOTTOM_HEIGHT, height));
    set({ bottomHeight: clamped });
  },
}));
