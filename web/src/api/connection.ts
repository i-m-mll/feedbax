import { useSyncExternalStore } from 'react';

/**
 * The window's connection to the Studio service, derived from requests that got no response.
 * `since` is when the window last stopped being connected: page load, or the loss.
 */
export type StudioConnection =
  | { kind: 'connected' }
  | { kind: 'connecting'; since: number }
  | { kind: 'unreachable'; since: number };

export const connectionGraceMs = 1000;
const probeDelays = [250, 500, 1000, 2000, 4000] as const;
const probeTimeoutMs = 5000;

let state: StudioConnection = { kind: 'connecting', since: Date.now() };
let probing = false;
let attempt = 0;
let timer: ReturnType<typeof setTimeout> | undefined;
const listeners = new Set<() => void>();

function publish(next: StudioConnection) {
  state = next;
  for (const listener of listeners) listener();
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
    if (state.kind !== 'unreachable') publish({ kind: 'unreachable', since: state.kind === 'connected' ? Date.now() : state.since });
    schedule(probeDelays[Math.min(++attempt, probeDelays.length - 1)]);
  }
}

export function serviceResponded() {
  probing = false;
  attempt = 0;
  clearTimeout(timer);
  timer = undefined;
  if (state.kind !== 'connected') publish({ kind: 'connected' });
}

export function serviceDidNotRespond() {
  if (probing) return;
  probing = true;
  attempt = 0;
  if (state.kind === 'connected') publish({ kind: 'connecting', since: Date.now() });
  schedule(probeDelays[0]);
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

export function isServiceConnected() {
  return state.kind === 'connected';
}
