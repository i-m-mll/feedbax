// @vitest-environment jsdom

import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { expect, it } from 'vitest';
import { useFollowingQuery } from './useFollowingQuery';

/** A server whose answers the test releases by hand; each answer reports the version it was asked at. */
function slowServer(version: () => string) {
  const calls: { signal: AbortSignal; answer: () => void }[] = [];
  const fetch = ({ signal }: { signal: AbortSignal }) => {
    const asked = version();
    return new Promise<string>((resolve) => calls.push({ signal, answer: () => resolve(asked) }));
  };
  return { calls, fetch };
}

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

it('finishes a fetch while the version keeps moving, then catches up once', async () => {
  let version = '1';
  const server = slowServer(() => version);
  const { result, rerender } = renderHook(
    ({ at }) => useFollowingQuery({ queryKey: ['plan'], queryFn: server.fetch, gcTime: 0 }, at),
    { wrapper, initialProps: { at: version } },
  );
  await waitFor(() => expect(server.calls).toHaveLength(1));
  for (const next of ['2', '3', '4']) {
    version = next;
    rerender({ at: next });
  }
  expect(server.calls).toHaveLength(1);
  expect(server.calls[0].signal.aborted).toBe(false);

  await act(async () => server.calls[0].answer());
  await waitFor(() => expect(server.calls).toHaveLength(2));
  expect(result.current.data).toBe('1');

  await act(async () => server.calls[1].answer());
  await waitFor(() => expect(result.current.data).toBe('4'));
  expect(server.calls).toHaveLength(2);
});
