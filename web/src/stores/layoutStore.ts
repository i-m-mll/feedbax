import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

interface LayoutStoreState {
  topCollapsed: boolean;
  bottomCollapsed: boolean;
  bottomHeight: number;
  initialized: boolean;
  resizeMode: boolean;
  leftSidebarWidth: number;
  taskSidebarWidth: number;
  rightSidebarWidth: number;
  leftSidebarVisible: boolean;
  rightSidebarVisible: boolean;
  bottomSidebarWidth: number;
  bottomSidebarCollapsed: boolean;
  bottomRightSidebarCollapsed: boolean;
  bottomShelfMode: 'stage' | 'console';
  componentLibraryExpandedCategories: string[];
  analysisLibraryExpandedCategories: string[];
  subgraphPreviewExpanded: Record<string, boolean>;
  toggleTop: (availableHeight: number) => void;
  toggleBottom: (availableHeight: number) => void;
  setBottomHeight: (height: number, availableHeight: number) => void;
  initializeBottomHeight: (availableHeight: number) => void;
  toggleResizeMode: () => void;
  setLeftSidebarWidth: (width: number) => void;
  setTaskSidebarWidth: (width: number) => void;
  setRightSidebarWidth: (width: number) => void;
  toggleLeftSidebar: () => void;
  toggleRightSidebar: () => void;
  setBottomSidebarWidth: (width: number) => void;
  toggleBottomSidebar: () => void;
  toggleBottomRightSidebar: () => void;
  setBottomShelfMode: (mode: 'stage' | 'console') => void;
  setComponentLibraryExpandedCategories: (categories: string[]) => void;
  setAnalysisLibraryExpandedCategories: (categories: string[]) => void;
  setSubgraphPreviewExpanded: (nodeId: string, expanded: boolean) => void;
}

type PersistedLayoutState = Pick<
  LayoutStoreState,
  | 'topCollapsed'
  | 'bottomCollapsed'
  | 'bottomHeight'
  | 'leftSidebarWidth'
  | 'taskSidebarWidth'
  | 'rightSidebarWidth'
  | 'leftSidebarVisible'
  | 'rightSidebarVisible'
  | 'bottomSidebarWidth'
  | 'bottomSidebarCollapsed'
  | 'bottomRightSidebarCollapsed'
  | 'bottomShelfMode'
  | 'componentLibraryExpandedCategories'
  | 'analysisLibraryExpandedCategories'
  | 'subgraphPreviewExpanded'
>;

const DEFAULT_BOTTOM_HEIGHT = 320;
export const SHELF_HEADER_HEIGHT = 44;
export const DIVIDER_HEIGHT = 1;
export const MIN_BOTTOM_HEIGHT = SHELF_HEADER_HEIGHT;
export const MIN_TOP_HEIGHT = 80;
export const MAX_BOTTOM_HEIGHT = Number.MAX_SAFE_INTEGER;
export const BOTTOM_COLLAPSED_HEIGHT = SHELF_HEADER_HEIGHT;
export const TOP_COLLAPSED_HEIGHT = SHELF_HEADER_HEIGHT;
const DEFAULT_SPLIT_RATIO = 0.5;

export const MIN_LEFT_WIDTH = 200;
export const MAX_LEFT_WIDTH = 400;
export const MIN_TASK_SIDEBAR_WIDTH = 320;
export const MAX_TASK_SIDEBAR_WIDTH = 640;
export const MIN_RIGHT_WIDTH = 240;
export const MAX_RIGHT_WIDTH = 500;
export const DEFAULT_LEFT_WIDTH = 256;
export const DEFAULT_TASK_SIDEBAR_WIDTH = 440;
export const DEFAULT_RIGHT_WIDTH = 320;
export const MIN_BOTTOM_SIDEBAR_WIDTH = 200;
export const MAX_BOTTOM_SIDEBAR_WIDTH = 400;
export const DEFAULT_BOTTOM_SIDEBAR_WIDTH = 256;

const DEFAULT_PERSISTED_LAYOUT: PersistedLayoutState = {
  topCollapsed: false,
  bottomCollapsed: false,
  bottomHeight: DEFAULT_BOTTOM_HEIGHT,
  leftSidebarWidth: DEFAULT_LEFT_WIDTH,
  taskSidebarWidth: DEFAULT_TASK_SIDEBAR_WIDTH,
  rightSidebarWidth: DEFAULT_RIGHT_WIDTH,
  leftSidebarVisible: true,
  rightSidebarVisible: false,
  bottomSidebarWidth: DEFAULT_BOTTOM_SIDEBAR_WIDTH,
  bottomSidebarCollapsed: false,
  bottomRightSidebarCollapsed: false,
  bottomShelfMode: 'stage',
  componentLibraryExpandedCategories: ['Neural Networks', 'CDE Controllers', 'Sensorimotor'],
  analysisLibraryExpandedCategories: ['Visualization'],
  subgraphPreviewExpanded: {},
};

const clampBottomHeight = (height: number, availableHeight: number) => {
  const maxBottom = Math.max(availableHeight - MIN_TOP_HEIGHT, BOTTOM_COLLAPSED_HEIGHT);
  const minBottom = BOTTOM_COLLAPSED_HEIGHT;
  return Math.max(minBottom, Math.min(maxBottom, height));
};

const clampLeftWidth = (width: number) =>
  Math.max(MIN_LEFT_WIDTH, Math.min(MAX_LEFT_WIDTH, width));

const clampTaskSidebarWidth = (width: number) =>
  Math.max(MIN_TASK_SIDEBAR_WIDTH, Math.min(MAX_TASK_SIDEBAR_WIDTH, width));

const clampRightWidth = (width: number) =>
  Math.max(MIN_RIGHT_WIDTH, Math.min(MAX_RIGHT_WIDTH, width));

const clampBottomSidebarWidth = (width: number) =>
  Math.max(MIN_BOTTOM_SIDEBAR_WIDTH, Math.min(MAX_BOTTOM_SIDEBAR_WIDTH, width));

export const useLayoutStore = create<LayoutStoreState>()(
  persist(
    (set) => ({
      topCollapsed: false,
      bottomCollapsed: false,
      bottomHeight: DEFAULT_BOTTOM_HEIGHT,
      initialized: false,
      resizeMode: false,
      leftSidebarWidth: DEFAULT_LEFT_WIDTH,
      taskSidebarWidth: DEFAULT_TASK_SIDEBAR_WIDTH,
      rightSidebarWidth: DEFAULT_RIGHT_WIDTH,
      leftSidebarVisible: true,
      rightSidebarVisible: false,
      bottomSidebarWidth: DEFAULT_BOTTOM_SIDEBAR_WIDTH,
      bottomSidebarCollapsed: false,
      bottomRightSidebarCollapsed: false,
      bottomShelfMode: 'stage',
      componentLibraryExpandedCategories: [
        'Neural Networks',
        'CDE Controllers',
        'Sensorimotor',
      ],
      analysisLibraryExpandedCategories: ['Visualization'],
      subgraphPreviewExpanded: {},
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
      setTaskSidebarWidth: (width) => {
        set({ taskSidebarWidth: clampTaskSidebarWidth(width) });
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
      setBottomSidebarWidth: (width) => {
        set({ bottomSidebarWidth: clampBottomSidebarWidth(width) });
      },
      toggleBottomSidebar: () => {
        set((state) => ({ bottomSidebarCollapsed: !state.bottomSidebarCollapsed }));
      },
      toggleBottomRightSidebar: () => {
        set((state) => ({ bottomRightSidebarCollapsed: !state.bottomRightSidebarCollapsed }));
      },
      setBottomShelfMode: (mode) => {
        set({ bottomShelfMode: mode });
      },
      setComponentLibraryExpandedCategories: (categories) => {
        set({ componentLibraryExpandedCategories: categories });
      },
      setAnalysisLibraryExpandedCategories: (categories) => {
        set({ analysisLibraryExpandedCategories: categories });
      },
      setSubgraphPreviewExpanded: (nodeId, expanded) => {
        set((state) => {
          const next = { ...state.subgraphPreviewExpanded };
          if (expanded) {
            next[nodeId] = true;
          } else {
            delete next[nodeId];
          }
          return { subgraphPreviewExpanded: next };
        });
      },
    }),
    {
      name: 'feedbax-studio-layout',
      storage: createJSONStorage(() => window.localStorage),
      version: 3,
      migrate: (persistedState, version): PersistedLayoutState => {
        const persisted =
          persistedState && typeof persistedState === 'object'
            ? (persistedState as Partial<PersistedLayoutState>)
            : {};
        return {
          ...DEFAULT_PERSISTED_LAYOUT,
          ...persisted,
          rightSidebarVisible:
            version < 2
              ? false
              : persisted.rightSidebarVisible ?? DEFAULT_PERSISTED_LAYOUT.rightSidebarVisible,
          bottomShelfMode:
            persisted.bottomShelfMode === 'console' ? 'console' : DEFAULT_PERSISTED_LAYOUT.bottomShelfMode,
          componentLibraryExpandedCategories: Array.isArray(
            persisted.componentLibraryExpandedCategories
          )
            ? persisted.componentLibraryExpandedCategories
            : DEFAULT_PERSISTED_LAYOUT.componentLibraryExpandedCategories,
          analysisLibraryExpandedCategories: Array.isArray(
            persisted.analysisLibraryExpandedCategories
          )
            ? persisted.analysisLibraryExpandedCategories
            : DEFAULT_PERSISTED_LAYOUT.analysisLibraryExpandedCategories,
          subgraphPreviewExpanded:
            persisted.subgraphPreviewExpanded &&
            typeof persisted.subgraphPreviewExpanded === 'object'
              ? persisted.subgraphPreviewExpanded
              : DEFAULT_PERSISTED_LAYOUT.subgraphPreviewExpanded,
        };
      },
      partialize: (state) => ({
        topCollapsed: state.topCollapsed,
        bottomCollapsed: state.bottomCollapsed,
        bottomHeight: state.bottomHeight,
        leftSidebarWidth: state.leftSidebarWidth,
        taskSidebarWidth: state.taskSidebarWidth,
        rightSidebarWidth: state.rightSidebarWidth,
        leftSidebarVisible: state.leftSidebarVisible,
        rightSidebarVisible: state.rightSidebarVisible,
        bottomSidebarWidth: state.bottomSidebarWidth,
        bottomSidebarCollapsed: state.bottomSidebarCollapsed,
        bottomRightSidebarCollapsed: state.bottomRightSidebarCollapsed,
        bottomShelfMode: state.bottomShelfMode,
        componentLibraryExpandedCategories: state.componentLibraryExpandedCategories,
        analysisLibraryExpandedCategories: state.analysisLibraryExpandedCategories,
        subgraphPreviewExpanded: state.subgraphPreviewExpanded,
      }),
    },
  )
);
