import { create } from 'zustand';

interface LayoutStoreState {
  topCollapsed: boolean;
  bottomCollapsed: boolean;
  bottomHeight: number;
  initialized: boolean;
  toggleTop: () => void;
  toggleBottom: () => void;
  setBottomHeight: (height: number) => void;
  initializeBottomHeight: (height: number) => void;
}

const DEFAULT_BOTTOM_HEIGHT = 320;
export const MIN_BOTTOM_HEIGHT = 200;
export const MIN_TOP_HEIGHT = 180;
export const MAX_BOTTOM_HEIGHT = 560;
export const BOTTOM_COLLAPSED_HEIGHT = 44;

export const useLayoutStore = create<LayoutStoreState>((set, get) => ({
  topCollapsed: false,
  bottomCollapsed: false,
  bottomHeight: DEFAULT_BOTTOM_HEIGHT,
  initialized: false,
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
    set({ bottomHeight: clamped, initialized: true });
  },
  initializeBottomHeight: (height) => {
    const clamped = Math.max(MIN_BOTTOM_HEIGHT, Math.min(MAX_BOTTOM_HEIGHT, height));
    set((state) => (state.initialized ? state : { bottomHeight: clamped, initialized: true }));
  },
}));
