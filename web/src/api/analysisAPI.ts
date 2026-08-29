/**
 * API client for the analysis system.
 *
 * Analysis package and page data must come from the backend. Backend and
 * contract failures are surfaced to callers instead of being replaced with
 * fabricated analysis definitions.
 */

import type {
  AnalysisBundleDryRunResult,
  SelectionSpec,
} from '@/generated/studioContracts';
import type {
  AnalysisGraphSpec,
  AnalysisPageSpec,
  AnalysisPackage,
  AnalysisClassDef,
  AnalysisSnapshot,
} from '@/types/analysis';
import { fetchGraph, updateGraph } from '@/api/client';
import { asApiRequestError, requestJson } from '@/api/request';
import { parseContract } from '@/generated/studioContracts';

// ---------------------------------------------------------------------------
// Wire format conversion - backend uses snake_case, frontend uses camelCase
// ---------------------------------------------------------------------------

/** Backend wire format for an analysis page. */
interface AnalysisPageWire {
  id: string;
  name: string;
  graph_spec: Record<string, unknown>;
  eval_params: Record<string, unknown>;
  viewport: { x: number; y: number; zoom: number };
  eval_run_id: string | null;
  expanded_field_paths?: string[];
}

/** Convert a backend wire-format page to the frontend camelCase type. */
function pageFromWire(wire: AnalysisPageWire): AnalysisPageSpec {
  return {
    id: wire.id,
    name: wire.name,
    graphSpec: wire.graph_spec as unknown as AnalysisGraphSpec,
    evalParams: wire.eval_params,
    viewport: wire.viewport,
    evalRunId: wire.eval_run_id ?? null,
    expandedFieldPaths: wire.expanded_field_paths ?? [],
  };
}

/** Convert a frontend camelCase page to the backend wire format. */
function pageToWire(page: AnalysisPageSpec): AnalysisPageWire {
  return {
    id: page.id,
    name: page.name,
    graph_spec: page.graphSpec as unknown as Record<string, unknown>,
    eval_params: page.evalParams,
    viewport: page.viewport,
    eval_run_id: page.evalRunId,
    expanded_field_paths: page.expandedFieldPaths ?? [],
  };
}

/**
 * Fetch available analysis packages.
 */
export async function fetchAnalysisPackages(): Promise<AnalysisPackage[]> {
  const path = '/api/analyses/packages';
  const result = await requestJson(path);
  try {
    return parseContract('AnalysisPackagesResponse', result).data.packages as AnalysisPackage[];
  } catch (error) {
    throw asApiRequestError(error, path, 'Analysis package response did not match the Studio contract.');
  }
}

/**
 * Fetch all available analysis classes (flattened from packages).
 */
export async function fetchAnalysisClasses(): Promise<AnalysisClassDef[]> {
  const packages = await fetchAnalysisPackages();
  return packages.flatMap((pkg) => pkg.analyses);
}

export async function dryRunAnalysisBundle(payload: {
  bundle: Record<string, unknown>;
  selectionSpec?: SelectionSpec | null;
  previewLimit?: number;
}): Promise<AnalysisBundleDryRunResult> {
  const path = '/api/analyses/bundles/dry-run';
  const result = await requestJson(path, {
    method: 'POST',
    body: JSON.stringify({
      bundle: payload.bundle,
      selection_spec: payload.selectionSpec ?? null,
      preview_limit: payload.previewLimit ?? 50,
    }),
  });
  try {
    return parseContract('AnalysisBundleDryRunResponse', result).data.dry_run;
  } catch (error) {
    throw asApiRequestError(error, path, 'Analysis bundle dry-run response did not match the Studio contract.');
  }
}

/**
 * Fetch analysis pages for a project from the graph endpoint.
 * Returns null if no analysis pages exist yet.
 */
export async function fetchAnalysisPages(
  graphId: string
): Promise<AnalysisSnapshot | null> {
  const data = await fetchGraph(graphId);
  const wirePages = data.workspace_document.analysis_pages as AnalysisPageWire[];
  if (!wirePages || wirePages.length === 0) return null;
  const pages = wirePages.map(pageFromWire);
  return {
    pages,
    activePageId: pages[0].id,
  };
}

/**
 * Save analysis pages for a project via the graph update endpoint.
 * Replaces the analysis view inside the versioned WorkspaceDocument.
 */
export async function saveAnalysisPages(
  graphId: string,
  snapshot: AnalysisSnapshot,
  expectedSaveRevision?: number | null,
): Promise<void> {
  const wirePages = snapshot.pages.map(pageToWire);
  const current = await fetchGraph(graphId);
  await updateGraph(
    graphId,
    null,
    {
      ...current.workspace_document,
      analysis_pages: wirePages,
      active_analysis_page_id: snapshot.activePageId,
    },
    undefined,
    expectedSaveRevision,
  );
}
