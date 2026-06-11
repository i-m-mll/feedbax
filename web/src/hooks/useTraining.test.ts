import { beforeEach, describe, expect, it } from 'vitest';
import {
  TRAINING_WS_MAX_RECONNECT_ATTEMPTS,
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
