import { useCallback, useEffect, useState, type Dispatch, type SetStateAction } from 'react';

export const projectWindowViews = [
  'runs', 'model', 'task', 'objective', 'training', 'scene', 'results', 'analysis', 'figures', 'document',
] as const;

export type ProjectWindowView = typeof projectWindowViews[number];
export type ProjectWindowSlot = 'primary' | 'secondary';
export type ProjectWindowMode = 'map' | 'both' | 'details';

export interface ProjectWindowState {
  mode: ProjectWindowMode;
  primaryView: ProjectWindowView;
  secondaryView: ProjectWindowView | null;
  focusedSlot: ProjectWindowSlot;
  mapDrawerOpen: boolean;
  detailsDrawerOpen: boolean;
  inspectorOpen: boolean;
  mapHeight: number;
  mapDrawerWidth: number;
  detailsDrawerWidth: number;
  inspectorWidth: number;
  secondaryWidth: number;
  modelPaths?: Partial<Record<ProjectWindowSlot, { document: string; path: string[] }>>;
}

export const defaultProjectWindowState: Readonly<ProjectWindowState> = {
  mode: 'both',
  primaryView: 'runs',
  secondaryView: null,
  focusedSlot: 'primary',
  mapDrawerOpen: true,
  detailsDrawerOpen: true,
  inspectorOpen: true,
  mapHeight: 320,
  mapDrawerWidth: 264,
  detailsDrawerWidth: 220,
  inspectorWidth: 244,
  secondaryWidth: 440,
};

export function showProjectWindowView(
  state: ProjectWindowState,
  view: ProjectWindowView,
  slot = state.focusedSlot,
): ProjectWindowState {
  return {
    ...state,
    mode: state.mode === 'map' ? 'both' : state.mode,
    focusedSlot: slot,
    ...(slot === 'primary' ? { primaryView: view } : { secondaryView: view }),
  };
}

export function projectWindowSlotView(
  state: ProjectWindowState,
  slot: ProjectWindowSlot,
  availableViews: readonly ProjectWindowView[],
): ProjectWindowView | null {
  const preferred = slot === 'primary' ? state.primaryView : state.secondaryView;
  if (!preferred) return null;
  return availableViews.includes(preferred) ? preferred : availableViews[0] ?? null;
}

function isView(value: unknown): value is ProjectWindowView {
  return typeof value === 'string' && projectWindowViews.some((view) => view === value);
}

function windowArrangement(value: unknown): ProjectWindowState {
  if (!value || typeof value !== 'object') return { ...defaultProjectWindowState };
  const prefs = value as Partial<ProjectWindowState>;
  const secondaryView = isView(prefs.secondaryView) ? prefs.secondaryView : null;
  function preferredSize(key: 'mapDrawerWidth' | 'detailsDrawerWidth' | 'inspectorWidth' | 'secondaryWidth' | 'mapHeight') {
    const size = prefs[key];
    return typeof size === 'number' && Number.isFinite(size) && size > 0
      ? size
      : defaultProjectWindowState[key];
  }
  function drawer(key: 'mapDrawerOpen' | 'detailsDrawerOpen' | 'inspectorOpen') {
    return typeof prefs[key] === 'boolean' ? prefs[key] : defaultProjectWindowState[key];
  }
  const modelPaths: NonNullable<ProjectWindowState['modelPaths']> = {};
  for (const slot of ['primary', 'secondary'] as const) {
    const entry = prefs.modelPaths?.[slot];
    if (entry && typeof entry.document === 'string' && Array.isArray(entry.path) &&
        entry.path.every((id) => typeof id === 'string')) modelPaths[slot] = entry;
  }
  return {
    mode: prefs.mode === 'map' || prefs.mode === 'details' ? prefs.mode : 'both',
    primaryView: isView(prefs.primaryView) ? prefs.primaryView : defaultProjectWindowState.primaryView,
    secondaryView,
    focusedSlot: secondaryView && prefs.focusedSlot === 'secondary' ? 'secondary' : 'primary',
    mapDrawerOpen: drawer('mapDrawerOpen'),
    detailsDrawerOpen: drawer('detailsDrawerOpen'),
    inspectorOpen: drawer('inspectorOpen'),
    mapHeight: preferredSize('mapHeight'),
    mapDrawerWidth: preferredSize('mapDrawerWidth'),
    detailsDrawerWidth: preferredSize('detailsDrawerWidth'),
    inspectorWidth: preferredSize('inspectorWidth'),
    secondaryWidth: preferredSize('secondaryWidth'),
    modelPaths,
  };
}

function readArrangement(key: string): ProjectWindowState {
  try {
    return windowArrangement(JSON.parse(window.localStorage.getItem(key) ?? 'null'));
  } catch {
    return { ...defaultProjectWindowState };
  }
}

export function useProjectWindowState(windowKey: string): [ProjectWindowState, Dispatch<SetStateAction<ProjectWindowState>>] {
  const key = `feedbax:studio-window:v1:${windowKey}`;
  const [entry, setEntry] = useState(() => ({ key, state: readArrangement(key) }));
  const state = entry.key === key ? entry.state : readArrangement(key);

  useEffect(() => {
    if (entry.key !== key) {
      setEntry({ key, state });
      return;
    }
    try {
      const arranged = JSON.stringify(windowArrangement(entry.state));
      if (window.localStorage.getItem(key) !== arranged) window.localStorage.setItem(key, arranged);
    } catch {
      // A window can still be rearranged when local storage is unavailable.
    }
  }, [entry, key, state]);

  const setState = useCallback<Dispatch<SetStateAction<ProjectWindowState>>>((update) => {
    setEntry((previous) => {
      const current = previous.key === key ? previous.state : readArrangement(key);
      const next = typeof update === 'function' ? update(current) : update;
      return next === current && previous.key === key ? previous : { key, state: next };
    });
  }, [key]);

  return [state, setState];
}
