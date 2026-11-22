// @vitest-environment jsdom

import { afterEach, expect, it, vi } from 'vitest';
import { patchProjectFolder } from '@/api/projectFolders';
import { parseContract } from '@/generated/studioContracts';
import { shownError, watchShownErrors } from './shownErrors';

afterEach(() => vi.unstubAllGlobals());

// Catches an error shown with no way back to its cause: no ref on its element, or no record of it in the log.
it('shows a refused write as Studio\'s defect, under a ref its element carries and the service log records', async () => {
  const fetch = vi.fn(async (_path: string, _init?: RequestInit) => new Response(null, { status: 204 }));
  vi.stubGlobal('fetch', fetch);
  vi.spyOn(console, 'error').mockImplementation(() => {});
  const stop = watchShownErrors(document.body);
  const revision = 'a'.repeat(64);
  const edit = { revision, address: { kind: 'step' as const, id: 'evaluate' }, patches: [{ op: 'add' as const, path: '/inputs/subject', value: { step: 'train', where: undefined } }] };

  const error = await patchProjectFolder('reaches', edit as never).catch((failure: unknown) => failure);
  const text = shownError(error);
  const alert = document.createElement('p');
  alert.setAttribute('role', 'alert');
  document.body.append(alert);
  alert.textContent = text;
  await Promise.resolve();
  stop();

  const ref = alert.getAttribute('data-note-ref');
  expect(ref).toMatch(/^[0-9a-f]{8}$/);
  expect(text).toContain(ref);
  const logged = fetch.mock.calls.filter(([path]) => path === '/api/studio/errors');
  expect(logged.map(([, init]) => parseContract('StudioErrorReport', JSON.parse(String(init?.body))))).toEqual([
    expect.objectContaining({
      ref, kind: 'request', path: '/api/projects/reaches', method: 'PATCH', contract: 'StudioProjectPatch',
      issues: expect.arrayContaining([expect.stringContaining('patches.0.value.where')]), sent: expect.objectContaining({ revision }),
    }),
  ]);
  expect(fetch.mock.calls.some(([path]) => path === '/api/projects/reaches')).toBe(false);
  expect(shownError(error)).toBe(text);
  expect(fetch.mock.calls.filter(([path]) => path === '/api/studio/errors')).toHaveLength(1);
});
