import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  TRAINING_PROGRESS_BATCH_INTERVAL_MS,
  TRAINING_WS_MAX_RECONNECT_ATTEMPTS,
  createTrainingProgressBatcher,
  formatTrainingDiagnostic,
  shouldReconnectTrainingWebSocket,
  trainingWebSocketReconnectDelayMs,
} from '@/hooks/useTraining';
import { useTrainingStore } from '@/stores/trainingStore';

describe('training WebSocket reconnect policy', () => {
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
