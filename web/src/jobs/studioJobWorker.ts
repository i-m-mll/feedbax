/// <reference lib="webworker" />
import {
  StudioEventResetSchema,
  StudioEventWatermarkSchema,
  StudioJobEventSchema,
  StudioProjectRevisionSchema,
} from '@/generated/studioContracts';
import type {
  StudioJobConnection,
  StudioJobWorkerCommand,
  StudioJobWorkerMessage,
} from './studioJobConnection';
import { reconnectDelay } from '@/api/backoff';

const worker = self as unknown as SharedWorkerGlobalScope;
type Subscription =
  | { kind: 'jobs'; projectId: string | null }
  | { kind: 'folder'; folderId: string };

const subscriptions = new Map<MessagePort, Subscription>();
let source: EventSource | null = null;
let subscription = '';
let cursor = 0;
let connection: StudioJobConnection = 'connecting';
let foldersReady = false;
let reopenAttempt = 0;
let reopenTimer: ReturnType<typeof setTimeout> | undefined;

function broadcastJobs(message: StudioJobWorkerMessage) {
  for (const [port, topic] of subscriptions) {
    if (topic.kind === 'jobs') port.postMessage(message);
  }
}

function folderConnected(port: MessagePort) {
  port.postMessage({ type: 'folder-connected' } satisfies StudioJobWorkerMessage);
}

function setConnection(next: StudioJobConnection) {
  if (connection === next) return;
  connection = next;
  broadcastJobs({ type: 'state', state: next });
}

/** Closes a failed stream and opens it again after the next backoff step. */
function lose(stream: EventSource) {
  stream.close();
  source = null;
  foldersReady = false;
  setConnection('disconnected');
  clearTimeout(reopenTimer);
  reopenTimer = setTimeout(() => reconcile(true), reconnectDelay(reopenAttempt++));
}

function reconcile(force = false) {
  const projects = new Set<string>();
  const folders = new Set<string>();
  for (const topic of subscriptions.values()) {
    if (topic.kind === 'folder') folders.add(topic.folderId);
    else if (topic.projectId !== null) projects.add(topic.projectId);
  }
  const selected = JSON.stringify([[...projects].sort(), [...folders].sort()]);
  if (subscriptions.size && source && selected === subscription && !force) return;
  subscription = selected;
  clearTimeout(reopenTimer);
  reopenTimer = undefined;
  source?.close();
  source = null;
  foldersReady = false;
  if (!subscriptions.size) {
    setConnection('connecting');
    return;
  }

  const query = new URLSearchParams({ machine: 'true', after: String(cursor) });
  for (const project of projects) query.append('project', project);
  for (const folder of folders) query.append('folder', folder);
  const stream = new EventSource(`/api/events?${query}`);
  source = stream;
  const read = (receive: (value: unknown) => void) => (event: Event) => {
    if (source !== stream || !(event instanceof MessageEvent)) return;
    try {
      receive(JSON.parse(event.data));
    } catch {
      lose(stream);
    }
  };
  stream.onopen = () => {
    if (source !== stream) return;
    foldersReady = false;
    reopenAttempt = 0;
    setConnection('connected');
  };
  stream.onerror = () => {
    if (source !== stream) return;
    // The browser retries a stream that lost its connection; one it closed is reopened here.
    if (stream.readyState === EventSource.CLOSED) lose(stream);
    else {
      foldersReady = false;
      setConnection('disconnected');
    }
  };
  stream.onmessage = read((data) => {
    const event = StudioJobEventSchema.parse(data);
    if (event.cursor <= cursor) return;
    cursor = event.cursor;
    broadcastJobs({ type: 'event', event });
  });
  stream.addEventListener('venue_event', read((data) => {
    const event = StudioJobEventSchema.parse(data);
    if (event.venue) broadcastJobs({ type: 'event', event });
  }));
  function advanceWatermark(data: unknown) {
    const marker = StudioEventWatermarkSchema.parse(data);
    cursor = Math.max(cursor, marker.cursor);
    setConnection('connected');
  }
  stream.addEventListener('heartbeat', read(advanceWatermark));
  stream.addEventListener('replay_complete', read((data) => {
    advanceWatermark(data);
    if (foldersReady) return;
    foldersReady = true;
    for (const [port, topic] of subscriptions) {
      if (topic.kind === 'folder') folderConnected(port);
    }
  }));
  stream.addEventListener('project_revision', read((data) => {
    const revision = StudioProjectRevisionSchema.parse(data);
    for (const [port, topic] of subscriptions) {
      if (topic.kind === 'folder' && topic.folderId === revision.project_id) {
        port.postMessage({ type: 'folder-revision', revision } satisfies StudioJobWorkerMessage);
      }
    }
  }));
  stream.addEventListener('machines_changed', () => {
    if (source === stream) broadcastJobs({ type: 'machines-changed' });
  });
  stream.addEventListener('reset', read((data) => {
    cursor = StudioEventResetSchema.parse(data).cursor;
    broadcastJobs({ type: 'reset' });
    reconcile(true);
  }));
}

worker.onconnect = ({ ports: [port] }) => {
  port.onmessage = ({ data }: MessageEvent<StudioJobWorkerCommand>) => {
    switch (data.type) {
      case 'subscribe':
        subscriptions.set(port, { kind: 'jobs', projectId: data.projectId });
        reconcile();
        port.postMessage({ type: 'attached' } satisfies StudioJobWorkerMessage);
        port.postMessage({ type: 'state', state: connection } satisfies StudioJobWorkerMessage);
        break;
      case 'subscribe-folder':
        subscriptions.set(port, { kind: 'folder', folderId: data.folderId });
        reconcile();
        if (foldersReady) folderConnected(port);
        break;
      case 'release':
        subscriptions.delete(port);
        port.close();
        reconcile();
        break;
      case 'retry':
        reconcile(true);
        break;
      case 'seen':
        if (subscriptions.get(port)?.kind === 'jobs') broadcastJobs(data);
        break;
    }
  };
};
