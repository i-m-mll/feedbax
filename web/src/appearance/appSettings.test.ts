// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { readSettings, writeSetting } from './appSettings';

afterEach(() => {
  localStorage.clear();
  delete window.__FEEDBAX_SETTINGS__;
  delete document.documentElement.dataset.shellHost;
  vi.unstubAllGlobals();
});

/** The packaged app's settings file, served at one path whatever port the service listens on. */
function settingsFile() {
  let body = 'null';
  vi.stubGlobal('fetch', vi.fn(async (_path: string, init?: RequestInit) => {
    if (init?.method === 'PUT') body = String(init.body);
    return new Response(init?.method === 'PUT' ? null : body, { status: init?.method === 'PUT' ? 204 : 200 });
  }));
  vi.stubGlobal('XMLHttpRequest', class {
    status = 0;
    responseText = '';
    open() {}
    send() { this.status = 200; this.responseText = body; }
  });
}

/**
 * Catches the packaged app keeping settings in browser storage, which is keyed by origin and so empty again after the
 * service restarts on another port.
 */
it('reads a setting back in the packaged app after a reload on another port', async () => {
  document.documentElement.dataset.shellHost = 'native';
  settingsFile();
  await writeSetting(window, 'density', 'compact');
  localStorage.clear();
  delete window.__FEEDBAX_SETTINGS__;
  expect(readSettings(window).density).toBe('compact');
});
