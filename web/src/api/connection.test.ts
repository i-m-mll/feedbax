import { beforeEach, expect, it, vi } from 'vitest';

let connection: typeof import('./connection');

beforeEach(async () => {
  vi.resetModules();
  connection = await import('./connection');
});

it('is connected only while requests are answered and the event stream is open', () => {
  connection.serviceResponded();
  expect(connection.getConnection().kind).toBe('connecting');
  connection.streamChanged(true);
  expect(connection.getConnection().kind).toBe('connected');
  connection.streamChanged(false);
  connection.serviceResponded();
  expect(connection.getConnection().kind).toBe('connecting');
  connection.streamChanged(true);
  expect(connection.getConnection().kind).toBe('connected');
});

it('is unreachable when a request and the health probe both go unanswered', async () => {
  vi.useFakeTimers();
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
  try {
    connection.streamChanged(true);
    connection.serviceResponded();
    connection.serviceDidNotRespond();
    expect(connection.getConnection().kind).toBe('connecting');
    await vi.advanceTimersByTimeAsync(250);
    expect(connection.getConnection().kind).toBe('unreachable');
    connection.serviceResponded();
    expect(connection.getConnection().kind).toBe('connected');
  } finally {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  }
});
