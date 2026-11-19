import {
  StudioEventResetSchema,
  StudioEventWatermarkSchema,
  StudioJobEventSchema,
  StudioProjectRevisionSchema,
  type StudioJobEvent,
  type StudioProjectRevision,
} from '@/generated/studioContracts';
import { reconnectDelay } from '@/api/backoff';
import type { StudioJobConnection, StudioJobMessage, StudioJobPort } from './studioJobConnection';

/**
 * The job event stream of every Studio window of one origin. The window holding the lock opens
 * the one EventSource for the union of all windows' subscriptions and publishes what it receives
 * on the channel; the others listen. When the holder goes, the lock passes to another window,
 * which reopens the stream after the last cursor it saw.
 */
const lockName = 'feedbax-studio-jobs';
const channelName = 'feedbax-studio-jobs';
/** How long a new holder collects the other windows' subscriptions before it opens the stream. */
const gatherMs = 50;

type Topic = { kind: 'jobs'; projectId: string | null } | { kind: 'folder'; folderId: string };
type Topics = { projects: string[]; folders: string[] };

type Broadcast =
  | { type: 'topics'; window: string; topics: Topics }
  | { type: 'leave'; window: string }
  | { type: 'hello' }
  | { type: 'retry' }
  | { type: 'state'; state: StudioJobConnection; to?: string }
  | { type: 'folders-ready'; to?: string }
  | { type: 'cursor'; cursor: number }
  | { type: 'reset'; cursor: number }
  | { type: 'event'; event: StudioJobEvent; cursor: number }
  | { type: 'machines-changed' }
  | { type: 'folder-revision'; revision: StudioProjectRevision }
  | { type: 'seen'; jobId: string };

class JobStreamHub {
  private readonly id = crypto.randomUUID();
  private readonly channel = new BroadcastChannel(channelName);
  private readonly ports = new Map<StudioJobPort, Topic>();
  private connection: StudioJobConnection = 'connecting';
  private foldersReady = false;
  private cursor = 0;
  private shown = true;
  private releaseLock: (() => void) | null = null;

  // Held only while this window holds the lock.
  private holding = false;
  private readonly remote = new Map<string, Topics>();
  private source: EventSource | null = null;
  private opened = '';
  private reopenAttempt = 0;
  private reopenTimer: ReturnType<typeof setTimeout> | undefined;
  private gatherTimer: ReturnType<typeof setTimeout> | undefined;
  private gathering = false;

  constructor() {
    this.channel.onmessage = ({ data }: MessageEvent<Broadcast>) => this.receive(data);
    window.addEventListener('pagehide', () => this.hide());
    window.addEventListener('pageshow', () => this.show());
    this.requestLock();
  }

  open(): StudioJobPort {
    const hub = this;
    const port: StudioJobPort = {
      onmessage: null,
      postMessage(command) {
        switch (command.type) {
          case 'subscribe':
            hub.subscribe(port, { kind: 'jobs', projectId: command.projectId });
            break;
          case 'subscribe-folder':
            hub.subscribe(port, { kind: 'folder', folderId: command.folderId });
            break;
          case 'release':
            hub.ports.delete(port);
            hub.topicsChanged();
            break;
          case 'retry':
            if (hub.holding) hub.reconcile(true);
            else hub.channel.postMessage({ type: 'retry' } satisfies Broadcast);
            break;
          case 'seen':
            hub.publish({ type: 'seen', jobId: command.jobId });
            break;
        }
      },
      close() {
        port.onmessage = null;
      },
    };
    return port;
  }

  private subscribe(port: StudioJobPort, topic: Topic) {
    this.ports.set(port, topic);
    this.topicsChanged();
    if (topic.kind === 'jobs') {
      this.deliver(port, { type: 'attached' });
      this.deliver(port, { type: 'state', state: this.connection });
    } else if (this.foldersReady) {
      this.deliver(port, { type: 'folder-connected' });
    }
  }

  private topics(): Topics {
    const projects = new Set<string>();
    const folders = new Set<string>();
    for (const topic of this.ports.values()) {
      if (topic.kind === 'folder') folders.add(topic.folderId);
      else if (topic.projectId !== null) projects.add(topic.projectId);
    }
    return { projects: [...projects], folders: [...folders] };
  }

  private topicsChanged() {
    if (this.holding) this.reconcile();
    else this.channel.postMessage({ type: 'topics', window: this.id, topics: this.topics() } satisfies Broadcast);
  }

  private deliver(port: StudioJobPort, data: StudioJobMessage) {
    port.onmessage?.({ data });
  }

  private toJobs(data: StudioJobMessage) {
    for (const [port, topic] of this.ports) if (topic.kind === 'jobs') this.deliver(port, data);
  }

  private toFolders(data: StudioJobMessage, folderId?: string) {
    for (const [port, topic] of this.ports) {
      if (topic.kind === 'folder' && (folderId === undefined || topic.folderId === folderId)) this.deliver(port, data);
    }
  }

  /** Sends a message to the other windows and applies it in this one. */
  private publish(message: Broadcast) {
    this.channel.postMessage(message);
    this.apply(message);
  }

  private receive(message: Broadcast) {
    switch (message.type) {
      case 'topics':
        if (!this.holding) return;
        this.remote.set(message.window, message.topics);
        this.channel.postMessage({ type: 'state', state: this.connection, to: message.window } satisfies Broadcast);
        // A stream reopened for the new topics announces its own replay boundary to every window.
        if (!this.reconcile() && this.foldersReady) {
          this.channel.postMessage({ type: 'folders-ready', to: message.window } satisfies Broadcast);
        }
        return;
      case 'leave':
        if (this.holding && this.remote.delete(message.window)) this.reconcile();
        return;
      case 'hello':
        if (this.shown) this.channel.postMessage({ type: 'topics', window: this.id, topics: this.topics() } satisfies Broadcast);
        return;
      case 'retry':
        if (this.holding) this.reconcile(true);
        return;
      default:
        if ('to' in message && message.to !== undefined && message.to !== this.id) return;
        this.apply(message);
    }
  }

  private apply(message: Broadcast) {
    switch (message.type) {
      case 'state':
        this.foldersReady = false;
        this.connection = message.state;
        this.toJobs({ type: 'state', state: message.state });
        break;
      case 'folders-ready':
        if (this.foldersReady) break;
        this.foldersReady = true;
        this.toFolders({ type: 'folder-connected' });
        break;
      case 'cursor':
        this.cursor = Math.max(this.cursor, message.cursor);
        break;
      case 'reset':
        this.cursor = message.cursor;
        this.toJobs({ type: 'reset' });
        break;
      case 'event':
        this.cursor = Math.max(this.cursor, message.cursor);
        this.toJobs({ type: 'event', event: message.event });
        break;
      case 'machines-changed':
        this.toJobs({ type: 'machines-changed' });
        break;
      case 'folder-revision':
        this.toFolders({ type: 'folder-revision', revision: message.revision }, message.revision.project_id);
        break;
      case 'seen':
        this.toJobs({ type: 'seen', jobId: message.jobId });
        break;
    }
  }

  private requestLock() {
    void navigator.locks.request(lockName, () => new Promise<void>((release) => {
      if (!this.shown) {
        release();
        return;
      }
      this.releaseLock = release;
      this.hold();
    }));
  }

  private hold() {
    this.holding = true;
    this.remote.clear();
    this.channel.postMessage({ type: 'hello' } satisfies Broadcast);
    this.setConnection('connecting');
    clearTimeout(this.gatherTimer);
    this.gathering = true;
    this.gatherTimer = setTimeout(() => {
      this.gathering = false;
      this.reconcile(true);
    }, gatherMs);
  }

  private hide() {
    this.shown = false;
    if (this.holding) {
      this.holding = false;
      this.gathering = false;
      clearTimeout(this.gatherTimer);
      clearTimeout(this.reopenTimer);
      this.source?.close();
      this.source = null;
      this.opened = '';
      this.releaseLock?.();
      this.releaseLock = null;
    } else {
      this.channel.postMessage({ type: 'leave', window: this.id } satisfies Broadcast);
    }
  }

  private show() {
    if (this.shown) return;
    this.shown = true;
    this.channel.postMessage({ type: 'topics', window: this.id, topics: this.topics() } satisfies Broadcast);
    this.requestLock();
  }

  private setConnection(next: StudioJobConnection) {
    if (this.connection === next) return;
    this.publish({ type: 'state', state: next });
  }

  /** Opens the stream for the union of every window's topics; says whether it opened one. */
  private reconcile(force = false) {
    if (!this.holding || this.gathering && !force) return false;
    this.gathering = false;
    const projects = new Set<string>();
    const folders = new Set<string>();
    for (const topics of [this.topics(), ...this.remote.values()]) {
      for (const project of topics.projects) projects.add(project);
      for (const folder of topics.folders) folders.add(folder);
    }
    const selected = JSON.stringify([[...projects].sort(), [...folders].sort()]);
    if (this.source && selected === this.opened && !force) return false;
    this.opened = selected;
    clearTimeout(this.gatherTimer);
    clearTimeout(this.reopenTimer);
    this.reopenTimer = undefined;
    this.source?.close();
    this.source = null;
    this.foldersReady = false;

    const query = new URLSearchParams({ machine: 'true', after: String(this.cursor) });
    for (const project of projects) query.append('project', project);
    for (const folder of folders) query.append('folder', folder);
    const stream = new EventSource(`/api/events?${query}`);
    this.source = stream;
    const read = (receive: (value: unknown) => void) => (event: Event) => {
      if (this.source !== stream || !(event instanceof MessageEvent)) return;
      try {
        receive(JSON.parse(event.data));
      } catch {
        this.lose(stream);
      }
    };
    stream.onopen = () => {
      if (this.source !== stream) return;
      this.reopenAttempt = 0;
      this.publish({ type: 'state', state: 'connected' });
    };
    stream.onerror = () => {
      if (this.source !== stream) return;
      // The browser retries a stream that lost its connection; one it closed is reopened here.
      if (stream.readyState === EventSource.CLOSED) this.lose(stream);
      else this.publish({ type: 'state', state: 'disconnected' });
    };
    stream.onmessage = read((data) => {
      const event = StudioJobEventSchema.parse(data);
      if (event.cursor <= this.cursor) return;
      this.publish({ type: 'event', event, cursor: event.cursor });
    });
    stream.addEventListener('venue_event', read((data) => {
      const event = StudioJobEventSchema.parse(data);
      if (event.venue) this.publish({ type: 'event', event, cursor: this.cursor });
    }));
    const watermark = (data: unknown) => {
      this.publish({ type: 'cursor', cursor: StudioEventWatermarkSchema.parse(data).cursor });
      this.setConnection('connected');
    };
    stream.addEventListener('heartbeat', read(watermark));
    stream.addEventListener('replay_complete', read((data) => {
      watermark(data);
      if (!this.foldersReady) this.publish({ type: 'folders-ready' });
    }));
    stream.addEventListener('project_revision', read((data) => {
      this.publish({ type: 'folder-revision', revision: StudioProjectRevisionSchema.parse(data) });
    }));
    stream.addEventListener('machines_changed', () => {
      if (this.source === stream) this.publish({ type: 'machines-changed' });
    });
    stream.addEventListener('reset', read((data) => {
      this.publish({ type: 'reset', cursor: StudioEventResetSchema.parse(data).cursor });
      this.reconcile(true);
    }));
    return true;
  }

  /** Closes a failed stream and opens it again after the next backoff step. */
  private lose(stream: EventSource) {
    stream.close();
    this.source = null;
    this.publish({ type: 'state', state: 'disconnected' });
    clearTimeout(this.reopenTimer);
    this.reopenTimer = setTimeout(() => this.reconcile(true), reconnectDelay(this.reopenAttempt++));
  }
}

let hub: JobStreamHub | undefined;

/** Opens a subscriber's port to the window's job stream. */
export function openJobPort(): StudioJobPort {
  hub ??= new JobStreamHub();
  return hub.open();
}
