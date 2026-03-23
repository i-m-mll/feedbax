import { create } from 'zustand';
import type { FigureRequest, FigureRequestStatus } from '@/types/analysis';

interface DemandState {
  /** Map from node ID to its figure generation request state. */
  requests: Record<string, FigureRequest>;

  /** Start a generation request for an analysis node. */
  requestGeneration: (nodeId: string) => void;

  /** Update the status of a generation request. */
  setStatus: (nodeId: string, status: FigureRequestStatus, extra?: Partial<FigureRequest>) => void;

  /** Mark a request as complete with figure hash(es). */
  setResult: (nodeId: string, figureHash: string) => void;

  /** Mark a request as failed. */
  setError: (nodeId: string, error: string) => void;

  /** Clear a request (reset to idle). */
  clearRequest: (nodeId: string) => void;

  /** Get the status for a node, defaulting to idle. */
  getStatus: (nodeId: string) => FigureRequestStatus;
}

export const useDemandStore = create<DemandState>((set, get) => ({
  requests: {},

  requestGeneration: (nodeId) =>
    set((state) => ({
      requests: {
        ...state.requests,
        [nodeId]: {
          nodeId,
          status: 'running',
          requestedAt: Date.now(),
          figureHash: undefined,
          error: undefined,
          completedAt: undefined,
        },
      },
    })),

  setStatus: (nodeId, status, extra) =>
    set((state) => ({
      requests: {
        ...state.requests,
        [nodeId]: {
          ...state.requests[nodeId],
          nodeId,
          status,
          ...extra,
        },
      },
    })),

  setResult: (nodeId, figureHash) =>
    set((state) => ({
      requests: {
        ...state.requests,
        [nodeId]: {
          ...state.requests[nodeId],
          nodeId,
          status: 'ready',
          figureHash,
          completedAt: Date.now(),
          error: undefined,
        },
      },
    })),

  setError: (nodeId, error) =>
    set((state) => ({
      requests: {
        ...state.requests,
        [nodeId]: {
          ...state.requests[nodeId],
          nodeId,
          status: 'error',
          error,
          completedAt: Date.now(),
        },
      },
    })),

  clearRequest: (nodeId) =>
    set((state) => ({
      requests: {
        ...state.requests,
        [nodeId]: {
          nodeId,
          status: 'idle',
          figureHash: undefined,
          error: undefined,
          requestedAt: undefined,
          completedAt: undefined,
        },
      },
    })),

  getStatus: (nodeId) => {
    const req = get().requests[nodeId];
    return req?.status ?? 'idle';
  },
}));
