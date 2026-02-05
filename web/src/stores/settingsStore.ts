import { create } from 'zustand';

interface SettingsState {
  showMinimap: boolean;
  toggleMinimap: () => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  showMinimap: true,
  toggleMinimap: () => set((state) => ({ showMinimap: !state.showMinimap })),
}));
