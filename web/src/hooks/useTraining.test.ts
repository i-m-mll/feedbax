// @vitest-environment jsdom

import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as api from '@/api/client';
import {
  TRAINING_PROGRESS_BATCH_INTERVAL_MS,
  TRAINING_WS_MAX_RECONNECT_ATTEMPTS,
  buildTrainingConfig,
  createTrainingProgressBatcher,
  formatTrainingDiagnostic,
  shouldReconnectTrainingWebSocket,
  trainingWebSocketReconnectDelayMs,
  useTraining,
} from '@/hooks/useTraining';
import { useGraphStore } from '@/stores/graphStore';
import { useTrainingStore } from '@/stores/trainingStore';

describe('training config authoring', () => {
  it('refuses to invent a rollout length when the task omits one', () => {
    expect(() =>
      buildTrainingConfig({ type: 'Generic', params: {} }, 10, 1, 0.001)
    ).toThrow('Training requires a positive task rollout length in task.params.n_steps.');
  });
});

function renderTrainingStream(jobId: string) {
  const initialGraphId = useGraphStore.getState().graphId;
  const initialTraining = useTrainingStore.getState();
  const sockets: Socket[] = [];
  class Socket {
    static CONNECTING = 0;
    static OPEN = 1;
    readyState = Socket.OPEN;
    onmessage?: (event: MessageEvent) => void;
    onclose?: () => void;
    constructor(readonly url: string) {
      sockets.push(this);
    }
    close = vi.fn(() => this.onclose?.());
    emit(payload: object) {
      this.onmessage?.(new MessageEvent('message', {
        data: JSON.stringify({ job_id: jobId, emitted_at_ms: 1, ...payload }),
      }));
    }
  }
  vi.useFakeTimers();
  vi.stubGlobal('WebSocket', Socket);
  vi.spyOn(api, 'startTraining').mockResolvedValue({ job_id: jobId });
  useGraphStore.setState({ graphId: 'test-graph' });
  const { result, unmount } = renderHook(() => useTraining());
  return {
    result,
    sockets,
    cleanup() {
      unmount();
      useGraphStore.setState({ graphId: initialGraphId });
      useTrainingStore.setState(initialTraining);
      vi.restoreAllMocks();
      vi.unstubAllGlobals();
      vi.useRealTimers();
    },
  };
}

describe('training WebSocket reconnect policy', () => {
  it('resumes after the last worker event despite synthetic Studio sequence numbers', async () => {
    const { result, sockets, cleanup } = renderTrainingStream('job-resume');
    try {
      await act(() => result.current.start());
      expect(new URL(sockets[0].url).search).toBe('');
      act(() => {
        sockets[0].emit({
          type: 'training_progress', seq: 3, worker_seq: 3, program_step: 7,
          total_batches: 5, loss: 0.5,
        });
        sockets[0].emit({
          type: 'training_log', seq: 4, worker_seq: 4, program_step: 7, message: 'First',
        });
        sockets[0].emit({ type: 'training_resync', seq: 100, reason: 'resumed', message: 'Resumed' });
        sockets[0].close();
        vi.advanceTimersByTime(500);
      });
      expect(new URL(sockets[1].url).searchParams.get('from_seq')).toBe('5');
      expect(useTrainingStore.getState().progress?.program_step).toBe(7);
      expect(useTrainingStore.getState().progress?.batch).toBeUndefined();
      act(() => {
        sockets[1].emit({ type: 'training_log', seq: 4, worker_seq: 4, batch: 1, message: 'First' });
        sockets[1].emit({
          type: 'training_progress', seq: 5, worker_seq: 5, batch: 5, program_step: 12,
          total_batches: 5, loss: 0.1,
        });
        sockets[1].emit({
          type: 'training_trajectory', seq: 6, worker_seq: 6, batch: 5, program_step: 12,
          trajectory: {},
        });
        sockets[1].emit({ type: 'training_complete', seq: 7, worker_seq: 7, batch: 5 });
      });
      const state = useTrainingStore.getState();
      expect(state.consoleLogs.filter(({ message }) => message === 'First')).toHaveLength(1);
      expect(state.progress).toMatchObject({ batch: 5, program_step: 12, loss: 0.1 });
      expect(state.latestTrajectory).toMatchObject({ batch: 5, program_step: 12 });
      expect(state.status).toBe('completed');
      expect(state.trainingStreamError).toBeNull();
    } finally {
      cleanup();
    }
  });

  it.each(['acknowledged', 'failed', 'completed first'] as const)(
    'follows the terminal stream when the stop request is %s',
    async (outcome) => {
      const { result, sockets, cleanup } = renderTrainingStream('job-stop');
      let settleStop!: () => void;
      const response = new Promise<Awaited<ReturnType<typeof api.stopTraining>>>((resolve, reject) => {
        settleStop = outcome === 'failed'
          ? () => reject(new Error('Network error'))
          : () => resolve({ success: true });
      });
      const stop = vi.spyOn(api, 'stopTraining').mockReturnValue(response);
      try {
        await act(() => result.current.start());
        const request = result.current.stop();
        expect(stop).toHaveBeenCalledWith('job-stop');
        if (outcome !== 'completed first') {
          await act(async () => {
            settleStop();
            await request;
          });
        }
        expect(useTrainingStore.getState()).toMatchObject({ status: 'running', jobId: 'job-stop' });
        expect(sockets[0].close).not.toHaveBeenCalled();

        act(() => {
          sockets[0].emit({
            type: 'training_progress', seq: 2, worker_seq: 2, batch: 1, program_step: 5,
            total_batches: 5, loss: 0.4,
          });
          sockets[0].close();
          vi.advanceTimersByTime(500);
        });
        expect(new URL(sockets[1].url).searchParams.get('from_seq')).toBe('3');

        const completed = outcome === 'completed first';
        const batch = completed ? 5 : 2;
        act(() => {
          if (completed) {
            sockets[1].emit({
              type: 'training_progress', seq: 3, worker_seq: 3, batch, program_step: 9,
              total_batches: 5, loss: 0.3,
            });
          }
          sockets[1].emit({
            type: completed ? 'training_complete' : 'training_stopped',
            seq: 4, worker_seq: 4, batch, program_step: 9, loss: 0.3,
          });
        });
        if (completed) {
          await act(async () => {
            settleStop();
            await request;
          });
        }
        expect(useTrainingStore.getState()).toMatchObject({
          status: completed ? 'completed' : 'stopped',
          jobId: 'job-stop',
          progress: completed
            ? { batch, program_step: 9, loss: 0.3 }
            : { batch: 1, program_step: 5, loss: 0.4 },
        });
        expect(useTrainingStore.getState().consoleLogs.at(-1)).toMatchObject({
          batch,
          program_step: 9,
        });
        expect(sockets[1].close).toHaveBeenCalledOnce();
        act(() => vi.advanceTimersByTime(4_000));
        expect(sockets).toHaveLength(2);
      } finally {
        cleanup();
      }
    },
  );

  it('backs off retries and caps the delay', () => {
    expect(trainingWebSocketReconnectDelayMs(0)).toBe(500);
    expect(trainingWebSocketReconnectDelayMs(1)).toBe(1000);
    expect(trainingWebSocketReconnectDelayMs(2)).toBe(2000);
    expect(trainingWebSocketReconnectDelayMs(3)).toBe(4000);
    expect(trainingWebSocketReconnectDelayMs(4)).toBe(4000);
  });

  it('only retries running training streams within the bounded attempt count', () => {
    expect(
      shouldReconnectTrainingWebSocket({
        attempt: TRAINING_WS_MAX_RECONNECT_ATTEMPTS - 1,
        intentionalClose: false,
        status: 'running',
      })
    ).toBe(true);
    expect(
      shouldReconnectTrainingWebSocket({
        attempt: TRAINING_WS_MAX_RECONNECT_ATTEMPTS,
        intentionalClose: false,
        status: 'running',
      })
    ).toBe(false);
    expect(
      shouldReconnectTrainingWebSocket({
        attempt: 0,
        intentionalClose: true,
        status: 'running',
      })
    ).toBe(false);
    expect(
      shouldReconnectTrainingWebSocket({
        attempt: 0,
        intentionalClose: false,
        status: 'completed',
      })
    ).toBe(false);
  });
});

describe('training stream error state', () => {
  beforeEach(() => {
    useTrainingStore.getState().clearHistory();
  });

  it('persists visible stream errors until training history is cleared', () => {
    useTrainingStore.getState().setTrainingStreamError('Invalid training WebSocket payload');

    expect(useTrainingStore.getState().trainingStreamError).toBe(
      'Invalid training WebSocket payload'
    );

    useTrainingStore.getState().clearHistory();

    expect(useTrainingStore.getState().trainingStreamError).toBeNull();
  });
});

describe('training diagnostics formatting', () => {
  it('includes severity, code, and node ids', () => {
    expect(
      formatTrainingDiagnostic({
        schema_id: 'feedbax.diagnostic.domain',
        schema_version: 'feedbax.diagnostic.domain.v1',
        severity: 'error',
        code: 'graph.missing_subgraph',
        message: 'Missing subgraph',
        node_ids: ['network'],
        details: {},
      })
    ).toBe('ERROR graph.missing_subgraph [network]: Missing subgraph');
  });
});

describe('training progress batching', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('coalesces tight-loop progress events into one store update per interval', () => {
    vi.useFakeTimers();
    const applied: Array<{ batch: number; loss: number }> = [];
    let runningStatusWrites = 0;
    const batcher = createTrainingProgressBatcher(
      (progress) => applied.push({ batch: progress.batch, loss: progress.loss }),
      () => {
        runningStatusWrites += 1;
      }
    );

    batcher.enqueue({
      seq: 1,
      emitted_at_ms: 1000,
      batch: 1,
      total_batches: 10,
      loss: 0.9,
    });
    batcher.enqueue({
      seq: 2,
      emitted_at_ms: 1001,
      batch: 2,
      total_batches: 10,
      loss: 0.8,
    });
    batcher.enqueue({
      seq: 3,
      emitted_at_ms: 1002,
      batch: 3,
      total_batches: 10,
      loss: 0.7,
    });

    expect(applied).toEqual([]);

    vi.advanceTimersByTime(TRAINING_PROGRESS_BATCH_INTERVAL_MS);

    expect(applied).toEqual([{ batch: 3, loss: 0.7 }]);
    expect(runningStatusWrites).toBe(1);
  });
});
