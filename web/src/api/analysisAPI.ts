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
  AnalysisPackage,
  AnalysisClassDef,
  AnalysisSnapshot,
} from '@/types/analysis';
import { fetchGraph } from '@/api/client';
import { asApiRequestError, requestJson } from '@/api/request';
import { parseContract } from '@/generated/studioContracts';
import { analysisSnapshotFromWorkspaceDocument } from '@/utils/analysisCanvasLayout';

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
  return analysisSnapshotFromWorkspaceDocument(data.workspace_document, data.workspace);
}
