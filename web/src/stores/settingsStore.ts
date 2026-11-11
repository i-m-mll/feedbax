import { create } from 'zustand';

interface SettingsState {
  showMinimap: boolean;
  showDomainLegend: boolean;
  toggleMinimap: () => void;
  setShowDomainLegend: (value: boolean) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  showMinimap: false,
  showDomainLegend: false,
  toggleMinimap: () => set((state) => ({ showMinimap: !state.showMinimap })),
  setShowDomainLegend: (value) => set({ showDomainLegend: value }),
}));

const preferenceKeys = ['showMinimap', 'showDomainLegend'] as const;

export function synchronizeDesktopPreferences(onError: (message: string) => void) {
  const endpoint = '/api/desktop/preferences';
  const edited = new Set<string>();
  let stopped = false;
  let hydrating = false;
  let writes = Promise.resolve();
  const stop = useSettingsStore.subscribe((state, previous) => {
    if (hydrating) return;
    const changed = preferenceKeys.filter((key) => state[key] !== previous[key]);
    if (!changed.length) return;
    changed.forEach((key) => edited.add(key));
    const body = JSON.stringify(Object.fromEntries(changed.map((key) => [key, state[key]])));
    writes = writes.then(async () => {
      const response = await fetch(endpoint, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body,
      });
      if (!response.ok) throw new Error('Studio could not save your preferences.');
    }).catch(() => onError('Studio could not save your preferences.'));
  });
  void fetch(endpoint).then(async (response) => {
    if (!response.ok) throw new Error('Studio could not load your preferences.');
    const preferences = await response.json() as Record<string, unknown>;
    if (stopped) return;
    const state = useSettingsStore.getState();
    const accepted = preferenceKeys.filter((key) =>
      !edited.has(key) && typeof preferences[key] === typeof state[key],
    );
    hydrating = true;
    useSettingsStore.setState(Object.fromEntries(accepted.map((key) => [key, preferences[key]])));
    hydrating = false;
  }).catch(() => {
    if (!stopped) onError('Studio could not load your preferences.');
  });
  return () => {
    stopped = true;
    stop();
  };
}
