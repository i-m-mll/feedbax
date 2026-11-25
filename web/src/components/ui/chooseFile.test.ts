// @vitest-environment jsdom

import { afterEach, expect, it, vi } from 'vitest';
import { saveFile } from './chooseFile';

afterEach(() => {
  delete document.documentElement.dataset.shellHost;
  vi.unstubAllGlobals();
});

// Catches: Export in the desktop app handing the file to an anchor download, which the app's web view drops.
it('saves through the desktop window\'s save panel in the desktop app', async () => {
  document.documentElement.dataset.shellHost = 'native';
  const save = vi.fn(async (_name: string, _bytes: Uint8Array) => true);
  vi.stubGlobal('__FEEDBAX_STUDIO_MENU__', { publish: vi.fn(), saveFile: save });
  expect(await saveFile(new Blob(['model']), 'reach.feedbax-model')).toBe(true);
  const [name, bytes] = save.mock.calls[0];
  expect([name, new TextDecoder().decode(bytes)]).toEqual(['reach.feedbax-model', 'model']);
});

// Catches: a cancelled save dialog reported as a failed Export.
it('answers false when the browser\'s save dialog is cancelled', async () => {
  vi.stubGlobal('showSaveFilePicker', vi.fn(async () => { throw new DOMException('Cancelled', 'AbortError'); }));
  expect(await saveFile(new Blob(['model']), 'reach.feedbax-model')).toBe(false);
});
