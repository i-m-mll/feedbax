import { create } from 'zustand';

export type SelectionSyncMode = 'linked' | 'decoupled';

export interface SelectionContext {
  stage: string | null;
  collection: string | null;
  selectedIds: string[];
  focusedId: string | null;
}

export interface FrozenSnapshotProjection {
  source: 'training_run' | 'evaluation_run';
  runId: string;
  runLabel: string;
  runStatus: string;
  manifestId: string | null;
  manifestHash: string | null;
  specHashes: Record<string, string | null>;
  snapshot: {
    graph_spec?: Record<string, unknown> | null;
    training_spec?: Record<string, unknown> | null;
    task_spec?: Record<string, unknown> | null;
    task_binding_spec?: Record<string, unknown> | null;
  };
}

interface SelectionContextState {
  context: SelectionContext;
  previewId: string | null;
  syncMode: SelectionSyncMode;
  frozenSnapshot: FrozenSnapshotProjection | null;
  setContext: (context: Partial<SelectionContext>) => void;
  syncCollection: (stage: string | null, collection: string | null, availableIds: string[]) => void;
  setSelectedIds: (ids: Iterable<string>) => void;
  toggleSelectedId: (id: string) => void;
  focusId: (id: string | null) => void;
  previewFocus: (id: string | null) => void;
  setSyncMode: (mode: SelectionSyncMode) => void;
  setFrozenSnapshot: (snapshot: FrozenSnapshotProjection | null) => void;
  reset: () => void;
}

const EMPTY_CONTEXT: SelectionContext = {
  stage: null,
  collection: null,
  selectedIds: [],
  focusedId: null,
};

function uniqueIds(ids: Iterable<string>): string[] {
  return Array.from(new Set(Array.from(ids).filter((id) => id.length > 0)));
}

export function selectedIdSet(context: SelectionContext): Set<string> {
  return new Set(context.selectedIds);
}

export const useSelectionContextStore = create<SelectionContextState>((set) => ({
  context: EMPTY_CONTEXT,
  previewId: null,
  syncMode: 'linked',
  frozenSnapshot: null,

  setContext: (patch) =>
    set((state) => ({
      context: {
        ...state.context,
        ...patch,
        selectedIds: patch.selectedIds ? uniqueIds(patch.selectedIds) : state.context.selectedIds,
      },
    })),

  syncCollection: (stage, collection, availableIds) =>
    set((state) => {
      const available = new Set(availableIds);
      const sameCollection =
        state.context.stage === stage && state.context.collection === collection;
      const selectedIds = sameCollection
        ? state.context.selectedIds.filter((id) => available.has(id))
        : [];
      const focusedId =
        sameCollection && state.context.focusedId && available.has(state.context.focusedId)
          ? state.context.focusedId
          : null;
      const previewId =
        sameCollection && state.previewId && available.has(state.previewId)
          ? state.previewId
          : null;
      return {
        context: { stage, collection, selectedIds, focusedId },
        previewId,
      };
    }),

  setSelectedIds: (ids) =>
    set((state) => ({
      context: { ...state.context, selectedIds: uniqueIds(ids) },
    })),

  toggleSelectedId: (id) =>
    set((state) => {
      const selected = new Set(state.context.selectedIds);
      if (selected.has(id)) selected.delete(id);
      else selected.add(id);
      return {
        context: { ...state.context, selectedIds: Array.from(selected) },
      };
    }),

  focusId: (id) =>
    set((state) => ({
      context: { ...state.context, focusedId: id },
      previewId: null,
    })),

  previewFocus: (id) =>
    set((state) => ({
      previewId: state.syncMode === 'linked' ? id : state.previewId,
    })),

  setSyncMode: (mode) =>
    set((state) => ({
      syncMode: mode,
      previewId: mode === 'linked' ? state.previewId : null,
    })),

  setFrozenSnapshot: (snapshot) => set({ frozenSnapshot: snapshot }),

  reset: () =>
    set({
      context: EMPTY_CONTEXT,
      previewId: null,
      syncMode: 'linked',
      frozenSnapshot: null,
    }),
}));
