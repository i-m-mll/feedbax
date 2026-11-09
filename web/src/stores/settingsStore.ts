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

const preferenceKeys = [
  'showMinimap', 'snapToGrid', 'snapGridSize', 'showGridBackground',
  'showDomainLegend', 'reduceAnimations', 'defaultEdgeStyle',
] as const;

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
      !edited.has(key) && typeof preferences[key] === typeof state[key]
      && (key !== 'defaultEdgeStyle' || ['default', 'straight', 'step'].includes(String(preferences[key])))
      && (key !== 'snapGridSize' || Number(preferences[key]) > 0),
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
