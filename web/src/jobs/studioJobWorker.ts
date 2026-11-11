/// <reference lib="webworker" />
import {
  StudioEventResetSchema,
  StudioEventWatermarkSchema,
  StudioJobEventSchema,
} from '@/generated/studioContracts';
import type {
  StudioJobConnection,
  StudioJobWorkerCommand,
  StudioJobWorkerMessage,
} from './studioJobConnection';

const worker = self as unknown as SharedWorkerGlobalScope;
const projectsByPort = new Map<MessagePort, string | null>();
let source: EventSource | null = null;
let subscription = '';
let cursor = 0;
let connection: StudioJobConnection = 'connecting';

function broadcast(message: StudioJobWorkerMessage) {
  for (const port of projectsByPort.keys()) port.postMessage(message);
}

function setConnection(next: StudioJobConnection) {
  if (connection === next) return;
  connection = next;
  broadcast({ type: 'state', state: next });
}

function reconcile(force = false) {
  const projects = new Set([...projectsByPort.values()].filter((project) => project !== null));
  const selected = JSON.stringify([...projects].sort());
  if (projectsByPort.size && source && selected === subscription && !force) return;
  subscription = selected;
  source?.close();
  source = null;
  if (!projectsByPort.size) {
    setConnection('connecting');
    return;
  }

  const query = new URLSearchParams({ machine: 'true', after: String(cursor) });
  for (const project of projects) query.append('project', project);
  const stream = new EventSource(`/api/events?${query}`);
  source = stream;
  const read = (receive: (value: unknown) => void) => (event: Event) => {
    if (source !== stream || !(event instanceof MessageEvent)) return;
    try {
      receive(JSON.parse(event.data));
    } catch {
      stream.close();
      source = null;
      setConnection('disconnected');
    }
  };
  stream.onopen = () => {
    if (source === stream) setConnection('connected');
  };
  stream.onerror = () => {
    if (source === stream) setConnection('disconnected');
  };
  stream.onmessage = read((data) => {
    const event = StudioJobEventSchema.parse(data);
    if (event.cursor <= cursor) return;
    cursor = event.cursor;
    broadcast({ type: 'event', event });
  });
  const watermark = read((data) => {
    const marker = StudioEventWatermarkSchema.parse(data);
    cursor = Math.max(cursor, marker.cursor);
    setConnection('connected');
  });
  stream.addEventListener('heartbeat', watermark);
  stream.addEventListener('replay_complete', watermark);
  stream.addEventListener('reset', read((data) => {
    cursor = StudioEventResetSchema.parse(data).cursor;
    broadcast({ type: 'reset' });
    reconcile(true);
  }));
}

worker.onconnect = ({ ports: [port] }) => {
  port.onmessage = ({ data }: MessageEvent<StudioJobWorkerCommand>) => {
    switch (data.type) {
      case 'subscribe':
        projectsByPort.set(port, data.projectId);
        reconcile();
        port.postMessage({ type: 'attached' } satisfies StudioJobWorkerMessage);
        port.postMessage({ type: 'state', state: connection } satisfies StudioJobWorkerMessage);
        break;
      case 'release':
        projectsByPort.delete(port);
        port.close();
        reconcile();
        break;
      case 'retry':
        reconcile(true);
        break;
      case 'seen':
        broadcast(data);
        break;
    }
  };
};
