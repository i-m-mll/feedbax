import { create } from 'zustand';
import { compileGraphNode } from '@/api/client';
import type { DomainCompileReport } from '@/generated/studioContracts';
import { compileStatusForReport, graphPathKey, tryAcausalGraphHash } from '@/features/domains/acausal';
import { isAcausalGraphSpec, type GraphSubgraphSpec } from '@/types/graph';
import { useGraphStore } from './graphStore';
import { useProjectsStore } from './projectsStore';

type AcceptedReport = { report: DomainCompileReport; sequence: number };

interface CompileLayer {
  path: string[];
  accepted: readonly AcceptedReport[];
  pending: ReadonlyMap<string, number>;
  failed?: { hash: string; message: string };
  latestSequence: number;
}

type CompileLayers = ReadonlyMap<string, CompileLayer>;

interface CompileStatusState {
  documents: ReadonlyMap<string, CompileLayers>;
  setReports: (
    documentId: string,
    graph: GraphSubgraphSpec,
    reports: Record<string, DomainCompileReport> | null | undefined,
  ) => void;
}

const emptyLayers: CompileLayers = new Map();
function makeLayer(path: string[]): CompileLayer {
  return { path, accepted: [], pending: new Map(), latestSequence: 0 };
}

function hydrateLayers(
  graph: GraphSubgraphSpec,
  reports: Record<string, DomainCompileReport> | null | undefined,
): CompileLayers {
  if (!reports || Object.keys(reports).length === 0) return emptyLayers;
  const pathsByPersistedKey = new Map<string, string[] | null>();
  function collectPath(path: string[]) {
    const persistedKey = path.join('/');
    pathsByPersistedKey.set(persistedKey, pathsByPersistedKey.has(persistedKey) ? null : path);
  }
  function collectChildPaths(layer: GraphSubgraphSpec, path: string[]) {
    const childIds = new Set([...Object.keys(layer.nodes ?? {}), ...Object.keys(layer.subgraphs ?? {})]);
    for (const nodeId of childIds) {
      const childPath = [...path, nodeId];
      collectPath(childPath);
      const child = layer.subgraphs?.[nodeId];
      if (child) collectChildPaths(child, childPath);
    }
  }
  if (isAcausalGraphSpec(graph)) collectPath([]);
  collectChildPaths(graph, []);

  const layers = new Map<string, CompileLayer>();
  for (const [persistedKey, report] of Object.entries(reports)) {
    const path = pathsByPersistedKey.get(persistedKey);
    if (path) layers.set(graphPathKey(path), { ...makeLayer(path), accepted: [{ report, sequence: 0 }] });
  }
  return layers;
}

export const useCompileStatusStore = create<CompileStatusState>((set) => ({
  documents: new Map(),
  setReports: (documentId, graph, reports) => set((state) => ({
    documents: new Map(state.documents).set(documentId, hydrateLayers(graph, reports)),
  })),
}));

useProjectsStore.subscribe((state, previous) => {
  if (state.tabs === previous.tabs) return;
  const { documents } = useCompileStatusStore.getState();
  const activeTabs = new Set(state.tabs.map((tab) => tab.tabId));
  const openDocuments = new Map([...documents].filter(([documentId]) => activeTabs.has(documentId)));
  if (openDocuments.size !== documents.size) useCompileStatusStore.setState({ documents: openDocuments });
});

export function useDocumentCompileLayers(documentId: string): CompileLayers {
  return useCompileStatusStore((state) => state.documents.get(documentId) ?? emptyLayers);
}

export function reportForLayer(layer: CompileLayer | undefined, hash?: string): DomainCompileReport | undefined {
  const accepted = layer?.accepted;
  return accepted?.find(({ report }) => report.interior_content_hash === hash)?.report
    ?? accepted?.[0]?.report;
}

export function compileLayerStatus(layer: CompileLayer | undefined, hash: string) {
  if (layer?.failed?.hash === hash) return 'error';
  return compileStatusForReport(reportForLayer(layer, hash), hash, layer?.pending.has(hash));
}

type GraphLayerState = Pick<ReturnType<typeof useGraphStore.getState>, 'graphId' | 'graph' | 'graphStack'>;

export function compileTarget(documentId: string, { graphId, graph, graphStack }: GraphLayerState) {
  if (!graphId || !isAcausalGraphSpec(graph)) return null;
  const path = graphStack.map((layer) => layer.childNodeId).filter((id): id is string => Boolean(id));
  const pathKey = graphPathKey(path);
  const identity = tryAcausalGraphHash(graph);
  if (identity.ok === false) return { kind: 'invalid' as const, pathKey, message: identity.message };
  const { hash } = identity;
  return {
    kind: 'ready' as const,
    documentId,
    graphId,
    path,
    pathKey,
    hash,
    interior: graph,
    key: JSON.stringify([documentId, graphId, pathKey, hash]),
  };
}

export function shouldAutoCompile(layer: CompileLayer | undefined, hash: string): boolean {
  const status = compileLayerStatus(layer, hash);
  return status === 'never_compiled' || status === 'stale';
}

function viewedDraftHash(documentId: string, pathKey: string): string | undefined {
  const { activeTabId, tabs } = useProjectsStore.getState();
  const state = activeTabId === documentId
    ? useGraphStore.getState()
    : tabs.find((tab) => tab.tabId === documentId)?.graphSnapshot;
  const target = state && compileTarget(documentId, state);
  return target?.kind === 'ready' && target.pathKey === pathKey ? target.hash : undefined;
}

let nextSequence = 0;

export async function compileActiveLayer(key: string, mode: 'automatic' | 'retry'): Promise<void> {
  const target = compileTarget(useProjectsStore.getState().activeTabId, useGraphStore.getState());
  if (target?.kind !== 'ready' || target.key !== key) return;
  const { documentId, graphId, path, pathKey, hash, interior } = target;
  const { getState, setState } = useCompileStatusStore;
  const layer = getState().documents.get(documentId)?.get(pathKey);
  if (layer?.pending.has(hash)) return;
  if (mode === 'automatic' && !shouldAutoCompile(layer, hash)) return;

  function updateLayer(update: (current: CompileLayer) => CompileLayer) {
    setState((state) => {
      const layers = state.documents.get(documentId) ?? emptyLayers;
      const current = layers.get(pathKey) ?? makeLayer(path);
      const next = update(current);
      if (next === current) return state;
      return { documents: new Map(state.documents).set(
        documentId,
        new Map(layers).set(pathKey, next),
      ) };
    });
  }

  const sequence = ++nextSequence;
  updateLayer((current) => ({
    ...current,
    pending: new Map(current.pending).set(hash, sequence),
    failed: undefined,
    latestSequence: sequence,
  }));

  let report: DomainCompileReport;
  try {
    report = await compileGraphNode(graphId, path, interior);
    if (report.interior_content_hash !== hash) {
      throw new Error('The compile report does not match the submitted draft.');
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const viewedHash = viewedDraftHash(documentId, pathKey);
    updateLayer((current) => {
      if (current.pending.get(hash) !== sequence) return current;
      const pending = new Map(current.pending);
      pending.delete(hash);
      const retainFailure = viewedHash === hash || (
        current.latestSequence === sequence && (!current.failed || current.failed.hash !== viewedHash)
      );
      const failed = retainFailure ? { hash, message } : current.failed;
      return { ...current, pending, failed };
    });
    return;
  }

  const viewedHash = viewedDraftHash(documentId, pathKey);
  updateLayer((current) => {
    if (current.pending.get(hash) !== sequence) return current;
    const pending = new Map(current.pending);
    pending.delete(hash);
    const candidates = [
      ...current.accepted.filter(({ report }) => report.interior_content_hash !== hash),
      { report, sequence },
    ].sort((a, b) => b.sequence - a.sequence);
    // Keep a stale fallback and the currently displayed draft if it differs.
    const alternate = candidates.slice(1).find(({ report }) => report.interior_content_hash === viewedHash)
      ?? candidates[1];
    const accepted = alternate ? [candidates[0], alternate] : [candidates[0]];
    return { ...current, pending, accepted };
  });
}
