// @vitest-environment jsdom

import { expect, it, vi } from 'vitest';
import type { StudioJobCommand, StudioJobMessage } from './studioJobConnection';
import { watchProjectFolder } from './watchProjectFolder';

const makePort = vi.hoisted(() => vi.fn(() => ({
  onmessage: null as ((event: { data: StudioJobMessage }) => void) | null,
  postMessage: vi.fn<(message: StudioJobCommand) => void>(),
  close: vi.fn(),
})));

vi.mock('@/jobs/studioJobStream', () => ({ openJobPort: makePort }));

it('releases the folder port on pagehide and reopens it on pageshow', () => {
  const connected = vi.fn();
  const revision = vi.fn();
  const release = watchProjectFolder('reach', { connected, revision });
  const value = {
    project_id: 'reach',
    revision: 'a'.repeat(64),
    presentation_revision: 'b'.repeat(64),
  };

  try {
    expect(makePort).toHaveBeenCalledOnce();
    const first = makePort.mock.results[0].value;
    const fromFirst = first.onmessage!;
    expect(first.postMessage).toHaveBeenCalledWith({ type: 'subscribe-folder', folderId: 'reach' });
    fromFirst({ data: { type: 'folder-connected' } });
    expect(connected).toHaveBeenCalledOnce();
    window.dispatchEvent(new Event('pagehide'));
    expect(first.postMessage).toHaveBeenLastCalledWith({ type: 'release' });
    expect(first.close).toHaveBeenCalledOnce();
    fromFirst({ data: { type: 'folder-revision', revision: value } });
    expect(revision).not.toHaveBeenCalled();

    window.dispatchEvent(new Event('pageshow'));
    const restored = makePort.mock.results[1].value;
    expect(restored.postMessage).toHaveBeenCalledWith({ type: 'subscribe-folder', folderId: 'reach' });
    restored.onmessage!({ data: { type: 'folder-connected' } });
    restored.onmessage!({ data: { type: 'folder-revision', revision: value } });
    expect(connected).toHaveBeenCalledTimes(2);
    expect(revision).toHaveBeenCalledWith(value);
  } finally {
    release();
  }
  expect(makePort.mock.results[1].value.close).toHaveBeenCalledOnce();
  window.dispatchEvent(new Event('pageshow'));
  expect(makePort).toHaveBeenCalledTimes(2);
});
