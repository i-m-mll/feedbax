// @vitest-environment jsdom

import { expect, it, vi } from 'vitest';
import {
  STUDIO_JOBS_WORKER_NAME,
  type StudioJobWorkerCommand,
  type StudioJobWorkerMessage,
} from './studioJobConnection';
import { watchProjectFolder } from './watchProjectFolder';

const makePort = vi.hoisted(() => vi.fn((_options: { name: string }) => ({
  onmessage: null as ((event: { data: StudioJobWorkerMessage }) => void) | null,
  postMessage: vi.fn<(message: StudioJobWorkerCommand) => void>(),
  close: vi.fn(),
})));

vi.mock('@/jobs/studioJobWorker?sharedworker', () => ({
  default: class {
    port: ReturnType<typeof makePort>;

    constructor(options: { name: string }) {
      this.port = makePort(options);
    }
  },
}));

it('releases the folder port on pagehide and reopens it for the same worker on pageshow', () => {
  const connected = vi.fn();
  const revision = vi.fn();
  const release = watchProjectFolder('reach', { connected, revision });
  const value = {
    project_id: 'reach',
    revision: 'a'.repeat(64),
    presentation_revision: 'b'.repeat(64),
  };

  try {
    expect(makePort).toHaveBeenCalledWith({ name: STUDIO_JOBS_WORKER_NAME });
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
