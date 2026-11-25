import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react';
import { shellHost } from '@window-shell/browser/titlebar';
import type { RunAddress } from '@/features/pipeline/planView';
import { trialClockOf, type TrialClock } from '@/components/timeline/trialClock';
import { leaveJobStream, rejoinJobStream } from '@/jobs/studioJobStream';

export const projectWindowViews = [
  'runs', 'model', 'task', 'objective', 'training', 'scene', 'results', 'analysis', 'figures', 'document',
] as const;

export type ProjectWindowView = typeof projectWindowViews[number];
export type ProjectWindowSlot = 'primary' | 'secondary';
export type ProjectWindowMode = 'map' | 'both' | 'details';

/** The views of a selection: those with something to show, the first being the default, and why each other has nothing. */
export interface ViewOffer {
  readonly available: readonly ProjectWindowView[];
  readonly absent: Readonly<Partial<Record<ProjectWindowView, string>>>;
}

export const noViews: ViewOffer = { available: [], absent: {} };

/** A pinned slot's step, which it keeps while another is selected, and the run of it the slot shows. */
export interface SlotPin {
  readonly stepId: string;
  readonly runId?: string;
}

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
  sections?: Partial<Record<ProjectWindowSection, boolean>>;
  /** The pinned slots of a split. */
  pins?: Partial<Record<ProjectWindowSlot, SlotPin>>;
  /** The selected step, which the window opens on when no run is in focus. */
  stepId?: string;
  /** The run in focus, which the Results show. */
  focus?: RunAddress;
  /** The clock's picked trial, the seed and count it samples with, and whether its shelf is open. */
  clock?: TrialClock;
}

/** A drawer section, named by its view. */
export type ProjectWindowSection = `${ProjectWindowView}.${string}`;

export function sectionOpen(state: ProjectWindowState, section: ProjectWindowSection, initially = true) {
  return state.sections?.[section] ?? initially;
}

export function withSectionOpen(state: ProjectWindowState, section: ProjectWindowSection, open: boolean): ProjectWindowState {
  return { ...state, sections: { ...state.sections, [section]: open } };
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

/** Close one of two open slots; the other stays open, as the primary slot, and follows the selection. */
export function closeProjectWindowSlot(state: ProjectWindowState, slot: ProjectWindowSlot): ProjectWindowState {
  if (!state.secondaryView) return state;
  if (slot === 'secondary') return { ...state, secondaryView: null, focusedSlot: 'primary', pins: {} };
  const { secondary } = state.modelPaths ?? {};
  return {
    ...state, primaryView: state.secondaryView, secondaryView: null, focusedSlot: 'primary',
    modelPaths: secondary ? { primary: secondary } : {}, pins: {},
  };
}

/** Pin `slot` to `pin`, or unpin it. */
export function pinProjectWindowSlot(state: ProjectWindowState, slot: ProjectWindowSlot, pin: SlotPin | undefined): ProjectWindowState {
  const { [slot]: _held, ...others } = state.pins ?? {};
  return { ...state, pins: pin ? { ...others, [slot]: pin } : others };
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
  const sections: NonNullable<ProjectWindowState['sections']> = {};
  for (const [key, open] of Object.entries(prefs.sections ?? {})) {
    if (typeof open === 'boolean' && isView(key.split('.')[0])) sections[key as ProjectWindowSection] = open;
  }
  const pins: NonNullable<ProjectWindowState['pins']> = {};
  for (const slot of secondaryView ? ['primary', 'secondary'] as const : []) {
    const pin = prefs.pins?.[slot];
    if (pin && typeof pin.stepId === 'string') {
      pins[slot] = { stepId: pin.stepId, ...typeof pin.runId === 'string' && { runId: pin.runId } };
    }
  }
  const focus = prefs.focus && typeof prefs.focus.stepId === 'string' && typeof prefs.focus.runId === 'string'
    ? { stepId: prefs.focus.stepId, runId: prefs.focus.runId } : undefined;
  const clock = trialClockOf(prefs.clock);
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
    sections,
    pins,
    ...(typeof prefs.stepId === 'string' && { stepId: prefs.stepId }),
    ...(focus && { focus }),
    ...(clock && { clock }),
  };
}

function arrangementPayload(state: ProjectWindowState) {
  const { focusedSlot: _focusedSlot, ...payload } = windowArrangement(state);
  return JSON.stringify(payload);
}

/** Where a window's arrangement is kept: the native shell's preferences, or this browser's storage. */
interface ArrangementStore {
  read(signal: AbortSignal): Promise<unknown>;
  write(body: string): Promise<void>;
}

function arrangementStore(native: boolean, windowKey: string): ArrangementStore {
  if (!native) {
    const key = `feedbax.studio.window.${windowKey}`;
    return {
      read: async () => JSON.parse(localStorage.getItem(key) ?? 'null'),
      write: async (body) => localStorage.setItem(key, body),
    };
  }
  const endpoint = `/api/desktop/preferences/windows/${encodeURIComponent(windowKey)}`;
  return {
    read: async (signal) => {
      const response = await fetch(endpoint, { signal });
      if (!response.ok) throw new Error(`Studio preferences returned ${response.status}`);
      return response.json();
    },
    write: async (body) => {
      const response = await fetch(endpoint, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body, keepalive: true,
      });
      if (!response.ok) throw new Error(`Studio preferences returned ${response.status}`);
    },
  };
}

export function useProjectWindowState(windowKey: string): [ProjectWindowState, Dispatch<SetStateAction<ProjectWindowState>>] {
  const native = shellHost() === 'native';
  const store = useMemo(() => arrangementStore(native, windowKey), [native, windowKey]);
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
    const request = new AbortController();
    void store.read(request.signal).then((saved) => {
      const restored = windowArrangement(saved);
      if (request.signal.aborted) return;
      if (desired.current === undefined) lastSaved.current = arrangementPayload(restored);
      setEntry((current) => current.key !== windowKey || current.edited
        ? current
        : { key: windowKey, state: restored, edited: false });
    }).catch((error: unknown) => {
      if (!request.signal.aborted) console.error('Studio window preferences could not be read', error);
    });
    return () => request.abort();
  }, [store, windowKey]);

  const flush = useCallback(() => {
    const operation = saving.current.then(async () => {
      for (let body = desired.current; body !== undefined && body !== lastSaved.current; body = desired.current) {
        await store.write(body);
        lastSaved.current = body;
      }
    });
    saving.current = operation.catch(() => {});
    return operation;
  }, [store]);

  useEffect(() => {
    if (!native) return;
    window.__FEEDBAX_FLUSH_WINDOW_PREFERENCES__ = (close = true) => {
      closing.current = close;
      if (!close) {
        rejoinJobStream();
        return Promise.resolve();
      }
      return flush().then(leaveJobStream);
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
    if (entry.key !== windowKey || !entry.edited) return;
    const body = arrangementPayload(entry.state);
    if (body === desired.current || (desired.current === undefined && body === lastSaved.current)) return;
    desired.current = body;
    void flush().catch((error: unknown) => console.error('Studio window preferences could not be saved', error));
  }, [entry, flush, windowKey]);

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
