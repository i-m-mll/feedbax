import { create } from 'zustand';

interface LayoutStoreState {
  topCollapsed: boolean;
  bottomCollapsed: boolean;
  bottomHeight: number;
  initialized: boolean;
  resizeMode: boolean;
  leftSidebarWidth: number;
  rightSidebarWidth: number;
  leftSidebarVisible: boolean;
  rightSidebarVisible: boolean;
  toggleTop: (availableHeight: number) => void;
  toggleBottom: (availableHeight: number) => void;
  setBottomHeight: (height: number, availableHeight: number) => void;
  initializeBottomHeight: (availableHeight: number) => void;
  toggleResizeMode: () => void;
  setLeftSidebarWidth: (width: number) => void;
  setRightSidebarWidth: (width: number) => void;
  toggleLeftSidebar: () => void;
  toggleRightSidebar: () => void;
}

const DEFAULT_BOTTOM_HEIGHT = 320;
export const SHELF_HEADER_HEIGHT = 44;
export const MIN_BOTTOM_HEIGHT = SHELF_HEADER_HEIGHT;
export const MIN_TOP_HEIGHT = SHELF_HEADER_HEIGHT;
export const MAX_BOTTOM_HEIGHT = Number.MAX_SAFE_INTEGER;
export const BOTTOM_COLLAPSED_HEIGHT = SHELF_HEADER_HEIGHT;
export const TOP_COLLAPSED_HEIGHT = SHELF_HEADER_HEIGHT;
const DEFAULT_SPLIT_RATIO = 0.5;

export const MIN_LEFT_WIDTH = 200;
export const MAX_LEFT_WIDTH = 400;
export const MIN_RIGHT_WIDTH = 240;
export const MAX_RIGHT_WIDTH = 500;
export const DEFAULT_LEFT_WIDTH = 256;
export const DEFAULT_RIGHT_WIDTH = 320;

const clampBottomHeight = (height: number, availableHeight: number) => {
  const maxBottom = Math.max(availableHeight - MIN_TOP_HEIGHT, BOTTOM_COLLAPSED_HEIGHT);
  const minBottom = BOTTOM_COLLAPSED_HEIGHT;
  return Math.max(minBottom, Math.min(maxBottom, height));
};

const clampLeftWidth = (width: number) =>
  Math.max(MIN_LEFT_WIDTH, Math.min(MAX_LEFT_WIDTH, width));

const clampRightWidth = (width: number) =>
  Math.max(MIN_RIGHT_WIDTH, Math.min(MAX_RIGHT_WIDTH, width));

export const useLayoutStore = create<LayoutStoreState>((set) => ({
  topCollapsed: false,
  bottomCollapsed: false,
  bottomHeight: DEFAULT_BOTTOM_HEIGHT,
  initialized: false,
  resizeMode: false,
  leftSidebarWidth: DEFAULT_LEFT_WIDTH,
  rightSidebarWidth: DEFAULT_RIGHT_WIDTH,
  leftSidebarVisible: true,
  rightSidebarVisible: true,
  toggleTop: (availableHeight) => {
    if (availableHeight <= 0) return;
    set((state) => {
      if (state.topCollapsed) {
        const target = clampBottomHeight(
          Math.round(availableHeight * DEFAULT_SPLIT_RATIO),
          availableHeight
        );
        return {
          topCollapsed: false,
          bottomCollapsed: false,
          bottomHeight: target,
        };
      }
      const expandedBottom = clampBottomHeight(
        availableHeight - TOP_COLLAPSED_HEIGHT,
        availableHeight
      );
      return {
        topCollapsed: true,
        bottomCollapsed: false,
        bottomHeight: expandedBottom,
      };
    });
  },
  toggleBottom: (availableHeight) => {
    if (availableHeight <= 0) return;
    set((state) => {
      if (state.bottomCollapsed) {
        const target = clampBottomHeight(
          Math.round(availableHeight * DEFAULT_SPLIT_RATIO),
          availableHeight
        );
        return {
          topCollapsed: false,
          bottomCollapsed: false,
          bottomHeight: target,
        };
      }
      return {
        topCollapsed: false,
        bottomCollapsed: true,
        bottomHeight: BOTTOM_COLLAPSED_HEIGHT,
      };
    });
  },
  setBottomHeight: (height, availableHeight) => {
    const clamped = clampBottomHeight(height, availableHeight);
    set({ bottomHeight: clamped, initialized: true });
  },
  initializeBottomHeight: (availableHeight) => {
    const target = clampBottomHeight(
      Math.round(availableHeight * DEFAULT_SPLIT_RATIO),
      availableHeight
    );
    set((state) => (state.initialized ? state : { bottomHeight: target, initialized: true }));
  },
  toggleResizeMode: () => {
    set((state) => ({ resizeMode: !state.resizeMode }));
  },
  setLeftSidebarWidth: (width) => {
    set({ leftSidebarWidth: clampLeftWidth(width) });
  },
  setRightSidebarWidth: (width) => {
    set({ rightSidebarWidth: clampRightWidth(width) });
  },
  toggleLeftSidebar: () => {
    set((state) => ({ leftSidebarVisible: !state.leftSidebarVisible }));
  },
  toggleRightSidebar: () => {
    set((state) => ({ rightSidebarVisible: !state.rightSidebarVisible }));
  },
}));
