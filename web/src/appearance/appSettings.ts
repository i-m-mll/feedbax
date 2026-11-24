import { shellHost } from '@window-shell/browser/titlebar';

/** The app's settings as stored: each reader validates its own entry. */
export type StoredSettings = Readonly<Record<string, unknown>>;

declare global {
  interface Window {
    __FEEDBAX_SETTINGS__?: StoredSettings;
  }
}

const storageKey = 'feedbax.settings';
const endpoint = '/api/desktop/preferences/app';
const changeEvent = 'feedbax-settings';

/**
 * Where the app's settings are kept: the packaged app's settings file, whose origin changes with each service port, or
 * this browser's storage.
 */
interface SettingsStore {
  readNow(): StoredSettings;
  read(): Promise<StoredSettings>;
  write(settings: StoredSettings): Promise<void>;
}

function parsed(text: string | null): StoredSettings {
  const value: unknown = text === null ? null : JSON.parse(text);
  return value && typeof value === 'object' && !Array.isArray(value) ? value as StoredSettings : {};
}

function settingsStore(view: Window): SettingsStore {
  if (shellHost(view.document) !== 'native') {
    return {
      readNow: () => parsed(view.localStorage.getItem(storageKey)),
      read: async () => parsed(view.localStorage.getItem(storageKey)),
      write: async (settings) => view.localStorage.setItem(storageKey, JSON.stringify(settings)),
    };
  }
  return {
    // The boot script reads before the first paint, so it waits on the settings file.
    readNow: () => {
      const request = new XMLHttpRequest();
      request.open('GET', endpoint, false);
      request.send();
      if (request.status !== 200) throw new Error(`Studio settings returned ${request.status}`);
      return parsed(request.responseText);
    },
    read: async () => {
      const response = await view.fetch(endpoint);
      if (!response.ok) throw new Error(`Studio settings returned ${response.status}`);
      return parsed(await response.text());
    },
    write: async (settings) => {
      const response = await view.fetch(endpoint, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings), keepalive: true,
      });
      if (!response.ok) throw new Error(`Studio settings returned ${response.status}`);
    },
  };
}

const writes = new WeakMap<Window, Promise<void>>();

function replace(view: Window, settings: StoredSettings) {
  view.__FEEDBAX_SETTINGS__ = settings;
  view.dispatchEvent(new Event(changeEvent));
}

/** The settings this window holds, read from the store the first time; the boot script and the app share them. */
export function readSettings(view: Window): StoredSettings {
  if (!view.__FEEDBAX_SETTINGS__) {
    try {
      view.__FEEDBAX_SETTINGS__ = settingsStore(view).readNow();
    } catch (error) {
      console.error('Studio settings could not be read; this window shows the defaults', error);
      view.__FEEDBAX_SETTINGS__ = {};
    }
  }
  return view.__FEEDBAX_SETTINGS__;
}

export function writeSetting(view: Window, name: string, value: unknown) {
  const settings = { ...readSettings(view), [name]: value };
  replace(view, settings);
  const store = settingsStore(view);
  const written = (writes.get(view) ?? Promise.resolve()).then(() => store.write(settings))
    .catch((error: unknown) => console.error('Studio settings could not be saved', error));
  writes.set(view, written);
  return written;
}

/** Calls `listener` when this window's settings change, or another window's change reaches it. */
export function subscribeSettings(view: Window, listener: () => void) {
  view.addEventListener(changeEvent, listener);
  return () => view.removeEventListener(changeEvent, listener);
}

/**
 * Follows changes made in other windows: a browser's storage event, or, in the packaged app, a fresh read whenever the
 * window comes to the front.
 */
export function followOtherWindows(view: Window) {
  const store = settingsStore(view);
  const reread = () => {
    void store.read().then((settings) => {
      if (JSON.stringify(settings) !== JSON.stringify(view.__FEEDBAX_SETTINGS__)) replace(view, settings);
    }).catch((error: unknown) => console.error('Studio settings could not be read', error));
  };
  if (shellHost(view.document) === 'native') view.addEventListener('focus', reread);
  else view.addEventListener('storage', (event) => { if (event.key === storageKey) reread(); });
}
