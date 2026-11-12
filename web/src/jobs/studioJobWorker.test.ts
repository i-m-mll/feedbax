import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { StudioJobWorkerCommand, StudioJobWorkerMessage } from './studioJobConnection';

class EventStream {
  closed = false;
  onopen?: () => void;
  onerror?: () => void;
  onmessage?: (event: MessageEvent) => void;
  listeners = new Map<string, (event: MessageEvent) => void>();

  constructor(readonly url: string) {
    streams.push(this);
  }

  close() {
    this.closed = true;
  }

  addEventListener(name: string, listener: (event: MessageEvent) => void) {
    this.listeners.set(name, listener);
  }

  emit(name: string, data: unknown, lastEventId = '') {
    const event = new MessageEvent(name, { data: JSON.stringify(data), lastEventId });
    if (name === 'message') this.onmessage?.(event);
    else this.listeners.get(name)?.(event);
  }

  query() {
    return new URL(this.url, 'http://localhost').searchParams;
  }
}

const streams: EventStream[] = [];
const worker: { onconnect?: (event: { ports: [unknown] }) => void } = {};

function attach(command: StudioJobWorkerCommand) {
  const port = {
    onmessage: undefined as ((event: { data: StudioJobWorkerCommand }) => void) | undefined,
    postMessage: vi.fn<(message: StudioJobWorkerMessage) => void>(),
    close: vi.fn(),
  };
  worker.onconnect!({ ports: [port] });
  const send = (data: StudioJobWorkerCommand) => port.onmessage!({ data });
  send(command);
  return {
    send,
    messages: () => port.postMessage.mock.calls.map(([message]) => message),
    closed: port.close,
  };
}

const jobs = (cursor: number) => ({
  cursor,
  job_id: 'train-1',
  project_id: 'training',
  request_id: 'fit',
  kind: 'progress',
  data: {},
});

const revision = (id: string, content = 'a') => ({
  project_id: id,
  revision: content.repeat(64),
  presentation_revision: 'b'.repeat(64),
});

beforeEach(async () => {
  vi.resetModules();
  vi.stubGlobal('self', worker);
  vi.stubGlobal('EventSource', EventStream);
  await import('./studioJobWorker');
});

afterEach(() => {
  vi.unstubAllGlobals();
  streams.length = 0;
  worker.onconnect = undefined;
});

it('shares the union of folder subscriptions without advancing Jobs replay for a revision', () => {
  const training = attach({ type: 'subscribe', projectId: 'training' });
  const [original] = streams;
  expect([...original.query()]).toEqual([
    ['machine', 'true'], ['after', '0'], ['project', 'training'],
  ]);
  original.emit('message', jobs(7));

  const reach = attach({ type: 'subscribe-folder', folderId: 'reach' });
  const reachStream = streams[1];
  expect(original.closed).toBe(true);
  expect(reachStream.query().get('after')).toBe('7');
  expect(reachStream.query().getAll('folder')).toEqual(['reach']);
  reachStream.emit('replay_complete', { cursor: 7 });
  reachStream.emit('project_revision', revision('reach'), '200');
  expect(reach.messages()).toEqual([
    { type: 'folder-connected' },
    { type: 'folder-revision', revision: revision('reach') },
  ]);

  const perturbation = attach({ type: 'subscribe-folder', folderId: 'perturbation' });
  const combined = streams[2];
  expect(streams.filter((stream) => !stream.closed)).toEqual([combined]);
  expect(combined.query().getAll('folder')).toEqual(['reach', 'perturbation']);
  expect(combined.query().getAll('project')).toEqual(['training']);
  expect(combined.query().get('after')).toBe('7');
  reachStream.emit('project_revision', revision('reach', 'c'));
  combined.emit('replay_complete', { cursor: 7 });
  combined.emit('project_revision', revision('perturbation'));
  combined.emit('message', jobs(8));
  expect(perturbation.messages()).toEqual([
    { type: 'folder-connected' },
    { type: 'folder-revision', revision: revision('perturbation') },
  ]);
  expect(reach.messages().filter(({ type }) => type === 'folder-revision')).toEqual([
    { type: 'folder-revision', revision: revision('reach') },
  ]);
  expect(training.messages().filter(({ type }) => type === 'event')).toEqual([
    { type: 'event', event: jobs(7) },
    { type: 'event', event: jobs(8) },
  ]);
  expect(training.messages().some(({ type }) => type.startsWith('folder-'))).toBe(false);

  perturbation.send({ type: 'release' });
  const remaining = streams[3];
  expect(combined.closed).toBe(true);
  expect(perturbation.closed).toHaveBeenCalledOnce();
  expect(remaining.query().getAll('folder')).toEqual(['reach']);
  expect(remaining.query().get('after')).toBe('8');
});

it('gives late and reconnected subscribers a fresh-read notice at the replay boundary', () => {
  const first = attach({ type: 'subscribe-folder', folderId: 'reach' });
  const [stream] = streams;
  stream.onopen!();
  stream.emit('heartbeat', { cursor: 7 });
  expect(first.messages()).toEqual([]);
  stream.emit('replay_complete', { cursor: 7 });
  stream.emit('project_revision', revision('reach'));

  const late = attach({ type: 'subscribe-folder', folderId: 'reach' });
  expect(streams).toEqual([stream]);
  expect(late.messages()).toEqual([{ type: 'folder-connected' }]);

  stream.onerror!();
  const offline = attach({ type: 'subscribe-folder', folderId: 'reach' });
  expect(offline.messages()).toEqual([]);
  stream.onopen!();
  expect(offline.messages()).toEqual([]);
  stream.emit('replay_complete', { cursor: 7 });
  stream.emit('project_revision', revision('reach', 'c'));
  expect(late.messages()).toEqual([
    { type: 'folder-connected' },
    { type: 'folder-connected' },
    { type: 'folder-revision', revision: revision('reach', 'c') },
  ]);
  expect(offline.messages()).toEqual([
    { type: 'folder-connected' },
    { type: 'folder-revision', revision: revision('reach', 'c') },
  ]);

  late.send({ type: 'release' });
  first.send({ type: 'release' });
  expect(streams).toEqual([stream]);
  expect(stream.closed).toBe(false);
  offline.send({ type: 'release' });
  expect(stream.closed).toBe(true);
});
