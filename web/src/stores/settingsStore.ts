import { create } from 'zustand';

type EdgeStyle = 'default' | 'straight' | 'step';

interface SettingsState {
  showMinimap: boolean;
  snapToGrid: boolean;
  snapGridSize: number;
  showGridBackground: boolean;
  showDomainLegend: boolean;
  reduceAnimations: boolean;
  defaultEdgeStyle: EdgeStyle;
  toggleMinimap: () => void;
  setSnapToGrid: (value: boolean) => void;
  setSnapGridSize: (value: number) => void;
  setShowGridBackground: (value: boolean) => void;
  setShowDomainLegend: (value: boolean) => void;
  setReduceAnimations: (value: boolean) => void;
  setDefaultEdgeStyle: (value: EdgeStyle) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  showMinimap: false,
  snapToGrid: false,
  snapGridSize: 20,
  showGridBackground: false,
  showDomainLegend: false,
  reduceAnimations: false,
  defaultEdgeStyle: 'default',
  toggleMinimap: () => set((state) => ({ showMinimap: !state.showMinimap })),
  setSnapToGrid: (value) => set({ snapToGrid: value }),
  setSnapGridSize: (value) => set({ snapGridSize: value }),
  setShowGridBackground: (value) => set({ showGridBackground: value }),
  setShowDomainLegend: (value) => set({ showDomainLegend: value }),
  setReduceAnimations: (value) => set({ reduceAnimations: value }),
  setDefaultEdgeStyle: (value) => set({ defaultEdgeStyle: value }),
}));
