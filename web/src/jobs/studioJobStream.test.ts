import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { StudioJobCommand, StudioJobMessage } from './studioJobConnection';

class EventStream {
  static readonly CLOSED = 2;
  readyState = 0;
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

  emit(name: string, data: unknown) {
    const event = new MessageEvent(name, { data: JSON.stringify(data) });
    if (name === 'message') this.onmessage?.(event);
    else this.listeners.get(name)?.(event);
  }

  query() {
    return new URL(this.url, 'http://localhost').searchParams;
  }
}

/** Channels of every simulated window, delivering asynchronously to the others as a browser does. */
const channels = new Set<Channel>();
class Channel {
  onmessage: ((event: { data: unknown }) => void) | null = null;
  constructor() {
    channels.add(this);
  }
  postMessage(data: unknown) {
    for (const other of channels) {
      if (other !== this) queueMicrotask(() => other.onmessage?.({ data: structuredClone(data) }));
    }
  }
}

/** One origin's lock manager: a request is granted when the holder's callback settles. */
const waiting: (() => Promise<void>)[] = [];
let held = false;
function grantNext() {
  const next = waiting.shift();
  if (!next) return;
  held = true;
  void next().then(() => {
    held = false;
    grantNext();
  });
}
const locks = {
  request(_name: string, callback: () => Promise<void>) {
    waiting.push(callback);
    if (!held) queueMicrotask(grantNext);
    return Promise.resolve();
  },
};

const streams: EventStream[] = [];
const open = () => streams.filter(({ closed }) => !closed);
const settle = () => vi.advanceTimersByTimeAsync(100);

/** A window of the origin with its own job hub. */
async function windowOf() {
  const page = new EventTarget();
  vi.stubGlobal('window', page);
  vi.resetModules();
  const { openJobPort } = await import('./studioJobStream');
  return {
    hide: () => page.dispatchEvent(new Event('pagehide')),
    attach(command: StudioJobCommand) {
      const received: StudioJobMessage[] = [];
      const port = openJobPort();
      port.onmessage = ({ data }) => received.push(data);
      port.postMessage(command);
      return { received, send: (next: StudioJobCommand) => port.postMessage(next) };
    },
  };
}

const jobs = (cursor: number) => ({
  cursor, job_id: 'train-1', project_id: 'training', request_id: 'fit', kind: 'progress', data: {},
});
const revision = (id: string) => ({ project_id: id, revision: 'a'.repeat(64), presentation_revision: 'b'.repeat(64) });

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal('EventSource', EventStream);
  vi.stubGlobal('BroadcastChannel', Channel);
  vi.stubGlobal('navigator', { locks });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  streams.length = 0;
  channels.clear();
  waiting.length = 0;
  held = false;
});

it('opens one stream for every window and hands it to another when its holder goes', async () => {
  const first = await windowOf();
  const training = first.attach({ type: 'subscribe', projectId: 'training' });
  const second = await windowOf();
  const reach = second.attach({ type: 'subscribe', projectId: 'reach' });
  await settle();

  expect(open()).toHaveLength(1);
  const [stream] = open();
  expect(stream.query().getAll('project').sort()).toEqual(['reach', 'training']);
  stream.onopen!();
  stream.emit('message', jobs(5));
  await settle();
  for (const window of [training, reach]) {
    expect(window.received).toContainEqual({ type: 'state', state: 'connected' });
    expect(window.received).toContainEqual({ type: 'event', event: jobs(5) });
  }

  first.hide();
  await settle();
  expect(stream.closed).toBe(true);
  expect(open()).toHaveLength(1);
  const [taken] = open();
  expect(taken.query().getAll('project')).toEqual(['reach']);
  expect(taken.query().get('after')).toBe('5');
  const before = reach.received.length;
  taken.onopen!();
  taken.emit('message', jobs(6));
  await settle();
  expect(reach.received.slice(before)).toEqual([
    { type: 'state', state: 'connected' },
    { type: 'event', event: jobs(6) },
  ]);
  expect(reach.received).toContainEqual({ type: 'state', state: 'connecting' });
});

it('gives a folder subscriber in another window a fresh-read notice and its revisions', async () => {
  const first = await windowOf();
  first.attach({ type: 'subscribe', projectId: 'training' });
  await settle();
  const [stream] = open();
  stream.onopen!();
  stream.emit('replay_complete', { cursor: 3 });

  const second = await windowOf();
  const folder = second.attach({ type: 'subscribe-folder', folderId: 'reach' });
  await settle();
  const [reopened] = open();
  expect(reopened.query().getAll('folder')).toEqual(['reach']);
  expect(reopened.query().get('after')).toBe('3');
  reopened.onopen!();
  reopened.emit('replay_complete', { cursor: 3 });
  reopened.emit('project_revision', revision('other'));
  reopened.emit('project_revision', revision('reach'));
  await settle();
  expect(folder.received.filter(({ type }) => type !== 'state')).toEqual([
    { type: 'folder-connected' },
    { type: 'folder-revision', revision: revision('reach') },
  ]);
});

it('reopens a stream the browser closed, after the last event it delivered', async () => {
  const only = await windowOf();
  const training = only.attach({ type: 'subscribe', projectId: 'training' });
  await settle();
  const [first] = open();
  first.onopen!();
  first.emit('message', jobs(4));
  first.readyState = EventStream.CLOSED;
  first.onerror!();
  expect(training.received.at(-1)).toEqual({ type: 'state', state: 'disconnected' });
  expect(open()).toHaveLength(0);
  await vi.advanceTimersByTimeAsync(250);
  const [second] = open();
  expect(second.query().get('after')).toBe('4');
  second.onopen!();
  expect(training.received.at(-1)).toEqual({ type: 'state', state: 'connected' });
});

it('marks a job seen in every window', async () => {
  const first = await windowOf();
  const one = first.attach({ type: 'subscribe', projectId: 'training' });
  const second = await windowOf();
  const two = second.attach({ type: 'subscribe', projectId: 'training' });
  await settle();
  two.send({ type: 'seen', jobId: 'train-1' });
  await settle();
  for (const window of [one, two]) expect(window.received).toContainEqual({ type: 'seen', jobId: 'train-1' });
});
