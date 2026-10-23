import { beforeEach, describe, expect, it } from 'vitest';
import { useDemandStore } from '@/stores/demandStore';

const NODE_ID = 'analysis-node';

function setActiveRequest(requestId: string) {
  useDemandStore.getState().requestGeneration(NODE_ID);
  useDemandStore.getState().setStatus(NODE_ID, 'running', { figureHash: requestId });
}

describe('demandStore figure request identity', () => {
  beforeEach(() => {
    useDemandStore.setState({ requests: {} });
  });

  it('accepts completion only from the active request', () => {
    setActiveRequest('request-a');
    setActiveRequest('request-b');

    expect(
      useDemandStore.getState().setResultForRequest(NODE_ID, 'request-a', 'figure-a')
    ).toBe(false);
    expect(useDemandStore.getState().requests[NODE_ID]).toMatchObject({
      status: 'running',
      figureHash: 'request-b',
    });

    expect(
      useDemandStore.getState().setResultForRequest(NODE_ID, 'request-b', 'figure-b')
    ).toBe(true);
    expect(useDemandStore.getState().requests[NODE_ID]).toMatchObject({
      status: 'ready',
      figureHash: 'figure-b',
    });
  });

  it('accepts failure only from the active request', () => {
    setActiveRequest('request-a');
    setActiveRequest('request-b');

    expect(
      useDemandStore.getState().setErrorForRequest(NODE_ID, 'request-a', 'stale failure')
    ).toBe(false);
    expect(useDemandStore.getState().requests[NODE_ID]).toMatchObject({
      status: 'running',
      figureHash: 'request-b',
      error: undefined,
    });

    expect(
      useDemandStore.getState().setErrorForRequest(NODE_ID, 'request-b', 'current failure')
    ).toBe(true);
    expect(useDemandStore.getState().requests[NODE_ID]).toMatchObject({
      status: 'error',
      figureHash: 'request-b',
      error: 'current failure',
    });
  });
});
