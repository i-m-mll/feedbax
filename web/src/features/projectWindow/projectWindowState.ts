import { useCallback, useEffect, useLayoutEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react';

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

declare global {
  interface Window {
    __FEEDBAX_FLUSH_WINDOW_PREFERENCES__?: (closing?: boolean) => Promise<void>;
  }
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
    focusedSlot: 'primary',
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

function arrangementPayload(state: ProjectWindowState) {
  const { focusedSlot: _focusedSlot, ...payload } = windowArrangement(state);
  return JSON.stringify(payload);
}

export function useProjectWindowState(windowKey: string): [ProjectWindowState, Dispatch<SetStateAction<ProjectWindowState>>] {
  const native = !!window.__FEEDBAX_STUDIO_MENU__;
  const endpoint = `/api/desktop/preferences/windows/${encodeURIComponent(windowKey)}`;
  const [entry, setEntry] = useState(() => ({ key: windowKey, state: defaultProjectWindowState, edited: false }));
  const lastSaved = useRef<string>();
  const desired = useRef<string>();
  const closing = useRef(false);
  const saving = useRef(Promise.resolve());
  const state = entry.key === windowKey ? entry.state : defaultProjectWindowState;

  useEffect(() => {
    lastSaved.current = undefined;
    desired.current = undefined;
    setEntry((current) => current.key === windowKey
      ? current
      : { key: windowKey, state: defaultProjectWindowState, edited: false });
    if (!native) return;
    const request = new AbortController();
    void fetch(endpoint, { signal: request.signal }).then(async (response) => {
      if (!response.ok) throw new Error(`Studio preferences returned ${response.status}`);
      const restored = windowArrangement(await response.json());
      if (request.signal.aborted) return;
      if (desired.current === undefined) lastSaved.current = arrangementPayload(restored);
      setEntry((current) => current.key !== windowKey || current.edited
        ? current
        : { key: windowKey, state: restored, edited: false });
    }).catch((error: unknown) => {
      if (!request.signal.aborted) console.error('Studio window preferences could not be read', error);
    });
    return () => request.abort();
  }, [endpoint, native, windowKey]);

  const flush = useCallback(() => {
    const operation = saving.current.then(async () => {
      for (let body = desired.current; body !== undefined && body !== lastSaved.current; body = desired.current) {
        const response = await fetch(endpoint, {
          method: 'PUT', headers: { 'Content-Type': 'application/json' }, body, keepalive: true,
        });
        if (!response.ok) throw new Error(`Studio preferences returned ${response.status}`);
        lastSaved.current = body;
      }
    });
    saving.current = operation.catch(() => {});
    return operation;
  }, [endpoint]);

  useEffect(() => {
    if (!native) return;
    window.__FEEDBAX_FLUSH_WINDOW_PREFERENCES__ = (close = true) => {
      closing.current = close;
      return close ? flush() : Promise.resolve();
    };
    const reconnected = () => {
      void flush().catch((error: unknown) => console.error('Studio window preferences could not be saved', error));
    };
    window.addEventListener('studio-native-reconnected', reconnected);
    return () => {
      delete window.__FEEDBAX_FLUSH_WINDOW_PREFERENCES__;
      window.removeEventListener('studio-native-reconnected', reconnected);
    };
  }, [flush, native]);

  useLayoutEffect(() => {
    if (!native || entry.key !== windowKey || !entry.edited) return;
    const body = arrangementPayload(entry.state);
    if (body === desired.current || (desired.current === undefined && body === lastSaved.current)) return;
    desired.current = body;
    void flush().catch((error: unknown) => console.error('Studio window preferences could not be saved', error));
  }, [entry, flush, native, windowKey]);

  const setState = useCallback<Dispatch<SetStateAction<ProjectWindowState>>>((update) => {
    if (closing.current) return;
    setEntry((previous) => {
      const current = previous.key === windowKey ? previous.state : defaultProjectWindowState;
      const next = typeof update === 'function' ? update(current) : update;
      return next === current && previous.key === windowKey
        ? previous
        : { key: windowKey, state: next, edited: true };
    });
  }, [windowKey]);

  return [state, setState];
}
