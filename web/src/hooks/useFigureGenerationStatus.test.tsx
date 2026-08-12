// @vitest-environment jsdom

import { act, cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getFigureStatus } from '@/api/figureAPI';
import { useFigureGenerationStatus } from '@/hooks/useFigureGenerationStatus';
import { useDemandStore } from '@/stores/demandStore';
import type { FigureStatusResponse } from '@/types/analysis';

vi.mock('@/api/figureAPI', () => ({
  getFigureStatus: vi.fn(),
}));

vi.mock('sonner', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

const NODE_ID = 'analysis-node';
const POLL_MS = 2000;

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function FigureStatusHarness() {
  const status = useDemandStore((state) => state.requests[NODE_ID]?.status ?? 'idle');
  useFigureGenerationStatus(NODE_ID, status);
  return null;
}

function setRequestId(requestId: string) {
  useDemandStore.getState().setStatus(NODE_ID, 'running', { figureHash: requestId });
}

describe('useFigureGenerationStatus', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(getFigureStatus).mockReset();
    useDemandStore.setState({ requests: {} });
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it('arms polling when the request ID arrives after running status', async () => {
    vi.mocked(getFigureStatus).mockResolvedValue({
      request_id: 'request-a',
      status: 'running',
    });
    useDemandStore.getState().requestGeneration(NODE_ID);
    render(<FigureStatusHarness />);

    await act(async () => {
      vi.advanceTimersByTime(POLL_MS);
    });
    expect(getFigureStatus).not.toHaveBeenCalled();

    act(() => setRequestId('request-a'));
    await act(async () => {
      vi.advanceTimersByTime(POLL_MS);
    });
    expect(getFigureStatus).toHaveBeenCalledWith('request-a');
  });

  it('replaces request A with B, ignores A success, and cleans up on unmount', async () => {
    const requestA = deferred<FigureStatusResponse>();
    vi.mocked(getFigureStatus).mockImplementation((requestId) => {
      if (requestId === 'request-a') return requestA.promise;
      return Promise.resolve({ request_id: requestId, status: 'running' });
    });
    useDemandStore.getState().requestGeneration(NODE_ID);
    setRequestId('request-a');
    const view = render(<FigureStatusHarness />);

    await act(async () => {
      vi.advanceTimersByTime(POLL_MS);
    });
    act(() => setRequestId('request-b'));
    await act(async () => {
      vi.advanceTimersByTime(POLL_MS);
    });
    expect(getFigureStatus).toHaveBeenCalledWith('request-b');

    await act(async () => {
      requestA.resolve({
        request_id: 'request-a',
        status: 'complete',
        figure_hashes: ['figure-a'],
      });
      await requestA.promise;
    });
    expect(useDemandStore.getState().requests[NODE_ID]).toMatchObject({
      status: 'running',
      figureHash: 'request-b',
    });

    view.unmount();
    await act(async () => {
      vi.advanceTimersByTime(POLL_MS * 2);
    });
    expect(vi.mocked(getFigureStatus).mock.calls.filter(([id]) => id === 'request-b')).toHaveLength(1);
  });

  it('ignores stale failure and keeps polling after a transient error', async () => {
    const requestA = deferred<FigureStatusResponse>();
    vi.mocked(getFigureStatus).mockImplementationOnce(() => requestA.promise);
    useDemandStore.getState().requestGeneration(NODE_ID);
    setRequestId('request-a');
    render(<FigureStatusHarness />);

    await act(async () => {
      vi.advanceTimersByTime(POLL_MS);
    });
    act(() => setRequestId('request-b'));
    vi.mocked(getFigureStatus)
      .mockRejectedValueOnce(new Error('temporary'))
      .mockResolvedValueOnce({ request_id: 'request-b', status: 'running' });

    await act(async () => {
      vi.advanceTimersByTime(POLL_MS);
    });
    await act(async () => {
      requestA.resolve({ request_id: 'request-a', status: 'error', error: 'stale failure' });
      await requestA.promise;
    });
    expect(useDemandStore.getState().requests[NODE_ID]).toMatchObject({
      status: 'running',
      figureHash: 'request-b',
      error: undefined,
    });

    await act(async () => {
      vi.advanceTimersByTime(POLL_MS);
    });
    expect(vi.mocked(getFigureStatus).mock.calls.filter(([id]) => id === 'request-b')).toHaveLength(2);
  });

  it('stops polling after active request completion', async () => {
    vi.mocked(getFigureStatus).mockResolvedValue({
      request_id: 'request-a',
      status: 'complete',
      figure_hashes: ['figure-a'],
    });
    useDemandStore.getState().requestGeneration(NODE_ID);
    setRequestId('request-a');
    render(<FigureStatusHarness />);

    await act(async () => {
      vi.advanceTimersByTime(POLL_MS);
    });
    expect(useDemandStore.getState().requests[NODE_ID]).toMatchObject({
      status: 'ready',
      figureHash: 'figure-a',
    });

    await act(async () => {
      vi.advanceTimersByTime(POLL_MS * 2);
    });
    expect(getFigureStatus).toHaveBeenCalledTimes(1);
  });

  it('stops polling after active request failure', async () => {
    vi.mocked(getFigureStatus).mockResolvedValue({
      request_id: 'request-a',
      status: 'error',
      error: 'generation failed',
    });
    useDemandStore.getState().requestGeneration(NODE_ID);
    setRequestId('request-a');
    render(<FigureStatusHarness />);

    await act(async () => {
      vi.advanceTimersByTime(POLL_MS);
    });
    expect(useDemandStore.getState().requests[NODE_ID]).toMatchObject({
      status: 'error',
      figureHash: 'request-a',
      error: 'generation failed',
    });

    await act(async () => {
      vi.advanceTimersByTime(POLL_MS * 2);
    });
    expect(getFigureStatus).toHaveBeenCalledTimes(1);
  });
});
