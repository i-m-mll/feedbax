import { useSyncExternalStore } from 'react';
import { reconnectDelay } from './backoff';

/**
 * The window's connection to the Studio service. It is connected only while its requests are
 * answered and its event stream is open; it is unreachable while a request goes unanswered and
 * the service's health does not answer either. `since` is when the window last stopped being
 * connected: page load, or the loss.
 */
export type StudioConnection =
  | { kind: 'connected' }
  | { kind: 'connecting'; since: number }
  | { kind: 'unreachable'; since: number };

export const connectionGraceMs = 1000;
const probeTimeoutMs = 5000;

let state: StudioConnection = { kind: 'connecting', since: Date.now() };
let probing = false;
let probeFailed = false;
let streamOpen = false;
let attempt = 0;
let timer: ReturnType<typeof setTimeout> | undefined;
const listeners = new Set<() => void>();

function publish(next: StudioConnection) {
  state = next;
  for (const listener of listeners) listener();
}

function settle() {
  const kind = probing ? probeFailed ? 'unreachable' : 'connecting' : streamOpen ? 'connected' : 'connecting';
  if (kind === state.kind) return;
  publish(kind === 'connected' ? { kind } : { kind, since: state.kind === 'connected' ? Date.now() : state.since });
}

function schedule(delay: number) {
  clearTimeout(timer);
  timer = setTimeout(() => void probe(), delay);
}

async function probe() {
  timer = undefined;
  try {
    const response = await fetch('/api/health', { signal: AbortSignal.timeout(probeTimeoutMs) });
    void response.body?.cancel();
    serviceResponded();
  } catch {
    if (!probing) return;
    probeFailed = true;
    settle();
    schedule(reconnectDelay(++attempt));
  }
}

export function serviceResponded() {
  probing = false;
  probeFailed = false;
  attempt = 0;
  clearTimeout(timer);
  timer = undefined;
  settle();
}

export function serviceDidNotRespond() {
  if (probing) return;
  probing = true;
  attempt = 0;
  settle();
  schedule(reconnectDelay(0));
}

/** Records whether the window's event stream is open; the stream reopens itself. */
export function streamChanged(open: boolean) {
  streamOpen = open;
  settle();
}

/** Tries the service now instead of at the next backoff step. */
export function retryConnection() {
  if (!probing) return;
  schedule(0);
}

export function getConnection() {
  return state;
}

export function subscribeConnection(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export function useConnection() {
  return useSyncExternalStore(subscribeConnection, getConnection);
}
