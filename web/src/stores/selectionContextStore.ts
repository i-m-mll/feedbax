import { create } from 'zustand';
import type { StudioWorkspaceSpec } from '@/types/workspace';
import {
  admitStudioDraftHashes,
  type StudioDraftHashes,
} from '@/utils/studioDraftHash';

export type SelectionSyncMode = 'linked' | 'decoupled';
export type SnapshotSpecKey =
  | 'graph_spec'
  | 'training_spec'
  | 'task_spec'
  | 'task_binding_spec'
  | 'evaluation_spec';
export type SnapshotSpecPayload = Partial<Record<SnapshotSpecKey, Record<string, unknown> | null>>;

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
  specHashes: StudioDraftHashes;
  snapshot: SnapshotSpecPayload;
}

const RUN_SNAPSHOT_PROVENANCE_SCHEMA_VERSION =
  'feedbax.studio.run_snapshot_provenance.v1';

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function nullableString(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function snapshotSpecPayload(value: unknown): SnapshotSpecPayload {
  if (!isRecord(value)) return {};
  const snapshot: SnapshotSpecPayload = {};
  const keys: SnapshotSpecKey[] = [
    'graph_spec',
    'training_spec',
    'task_spec',
    'task_binding_spec',
    'evaluation_spec',
  ];
  for (const key of keys) {
    const payload = value[key];
    if (payload === null) snapshot[key] = null;
    else if (isRecord(payload)) snapshot[key] = payload;
  }
  return snapshot;
}

export function frozenSnapshotProvenanceMetadata(
  projection: FrozenSnapshotProjection
): Record<string, unknown> {
  return {
    schema_version: RUN_SNAPSHOT_PROVENANCE_SCHEMA_VERSION,
    source: projection.source,
    run_id: projection.runId,
    run_label: projection.runLabel,
    run_status: projection.runStatus,
    manifest_id: projection.manifestId,
    manifest_hash: projection.manifestHash,
    spec_hashes: projection.specHashes,
    snapshot: projection.snapshot,
    mode: 'frozen_snapshot',
    read_only: true,
  };
}

export function frozenSnapshotProjectionFromWorkspace(
  workspace: StudioWorkspaceSpec | null
): FrozenSnapshotProjection | null {
  const topPane = workspace?.ui_state.top_pane;
  const metadata = isRecord(topPane) && isRecord(topPane.metadata) ? topPane.metadata : {};
  const provenance = metadata.run_snapshot_provenance;
  if (!isRecord(provenance) || provenance.mode !== 'frozen_snapshot') return null;
  if (
    provenance.schema_version !== undefined &&
    provenance.schema_version !== RUN_SNAPSHOT_PROVENANCE_SCHEMA_VERSION
  ) {
    throw new Error(
      `Unsupported run snapshot provenance schema version: ${String(provenance.schema_version)}`
    );
  }
  if (provenance.source !== 'training_run' && provenance.source !== 'evaluation_run') {
    return null;
  }
  if (
    typeof provenance.run_id !== 'string' ||
    typeof provenance.run_label !== 'string' ||
    typeof provenance.run_status !== 'string'
  ) {
    return null;
  }
  return {
    source: provenance.source,
    runId: provenance.run_id,
    runLabel: provenance.run_label,
    runStatus: provenance.run_status,
    manifestId: nullableString(provenance.manifest_id),
    manifestHash: nullableString(provenance.manifest_hash),
    specHashes: admitStudioDraftHashes(provenance.spec_hashes ?? {}),
    snapshot: snapshotSpecPayload(provenance.snapshot),
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
