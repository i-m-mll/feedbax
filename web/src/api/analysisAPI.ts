/**
 * API client for the analysis system.
 *
 * The backend endpoints may not exist yet — these functions define the
 * contract the frontend expects. A stub/mock layer is provided so the
 * UI is functional without a live backend.
 */

import type {
  AnalysisGraphSpec,
  AnalysisPageSpec,
  AnalysisPackage,
  AnalysisClassDef,
  AnalysisSnapshot,
} from '@/types/analysis';
import { updateGraph } from '@/api/client';

// ---------------------------------------------------------------------------
// Wire format conversion — backend uses snake_case, frontend uses camelCase
// ---------------------------------------------------------------------------

/** Backend wire format for an analysis page. */
interface AnalysisPageWire {
  id: string;
  name: string;
  graph_spec: Record<string, unknown>;
  eval_params: Record<string, unknown>;
  viewport: { x: number; y: number; zoom: number };
  eval_run_id: string | null;
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
  };
}

// ---------------------------------------------------------------------------
// Stub data — used when the backend is not available
// ---------------------------------------------------------------------------

const STUB_PACKAGES: AnalysisPackage[] = [
  {
    name: 'Visualization',
    description: 'Standard visualization analyses',
    analyses: [
      {
        name: 'Violins',
        description: 'Violin plots of state distributions across conditions',
        category: 'Visualization',
        inputPorts: ['data'],
        outputPorts: ['figure'],
        defaultParams: { metric: 'position_error', split_by: 'condition' },
        icon: 'BarChart3',
      },
      {
        name: 'Profiles',
        description: 'Time-series profiles averaged across trials',
        category: 'Visualization',
        inputPorts: ['data'],
        outputPorts: ['figure'],
        defaultParams: { metric: 'velocity', aggregate: 'mean' },
        icon: 'TrendingUp',
      },
      {
        name: 'Heatmap',
        description: 'Heatmap of activations or correlations',
        category: 'Visualization',
        inputPorts: ['data'],
        outputPorts: ['figure'],
        defaultParams: { colormap: 'viridis' },
        icon: 'Grid3x3',
      },
    ],
  },
  {
    name: 'Dimensionality Reduction',
    description: 'Decomposition and embedding analyses',
    analyses: [
      {
        name: 'PCA',
        description: 'Principal component analysis of neural states',
        category: 'Dimensionality Reduction',
        inputPorts: ['data'],
        outputPorts: ['components', 'variance'],
        defaultParams: { n_components: 3 },
        icon: 'Axis3d',
      },
      {
        name: 'UMAP',
        description: 'UMAP embedding of high-dimensional data',
        category: 'Dimensionality Reduction',
        inputPorts: ['data'],
        outputPorts: ['embedding'],
        defaultParams: { n_neighbors: 15, min_dist: 0.1 },
        icon: 'Scatter',
      },
    ],
  },
  {
    name: 'Statistics',
    description: 'Statistical analysis tools',
    analyses: [
      {
        name: 'Summary',
        description: 'Summary statistics (mean, std, quantiles)',
        category: 'Statistics',
        inputPorts: ['data'],
        outputPorts: ['stats'],
        defaultParams: { metrics: 'all' },
        icon: 'Calculator',
      },
      {
        name: 'Correlation',
        description: 'Pairwise correlation analysis',
        category: 'Statistics',
        inputPorts: ['data_a', 'data_b'],
        outputPorts: ['matrix'],
        defaultParams: { method: 'pearson' },
        icon: 'GitCompare',
      },
    ],
  },
  {
    name: 'Preprocessing',
    description: 'Data preparation and transforms',
    analyses: [
      {
        name: 'Filter',
        description: 'Filter trials by condition or metric threshold',
        category: 'Preprocessing',
        inputPorts: ['data'],
        outputPorts: ['filtered'],
        defaultParams: {},
        icon: 'Filter',
      },
      {
        name: 'Normalize',
        description: 'Z-score or min-max normalization',
        category: 'Preprocessing',
        inputPorts: ['data'],
        outputPorts: ['normalized'],
        defaultParams: { method: 'zscore' },
        icon: 'SlidersHorizontal',
      },
    ],
  },
];

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

async function tryFetch<T>(path: string): Promise<T | null> {
  try {
    const response = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) return null;
    return (await response.json()) as T;
  } catch {
    return null;
  }
}

/**
 * Fetch available analysis packages. Falls back to stub data if the
 * backend endpoint is not available.
 */
export async function fetchAnalysisPackages(): Promise<AnalysisPackage[]> {
  const result = await tryFetch<{ packages: AnalysisPackage[] }>(
    '/api/analyses/packages'
  );
  return result?.packages ?? STUB_PACKAGES;
}

/**
 * Fetch all available analysis classes (flattened from packages).
 */
export async function fetchAnalysisClasses(): Promise<AnalysisClassDef[]> {
  const packages = await fetchAnalysisPackages();
  return packages.flatMap((pkg) => pkg.analyses);
}

/**
 * Fetch the current analysis graph for a project/graph.
 * Returns null if no analysis graph exists yet.
 *
 * @deprecated Use fetchAnalysisPages instead for multi-page support.
 */
export async function fetchAnalysisGraph(
  graphId: string
): Promise<AnalysisGraphSpec | null> {
  return tryFetch<AnalysisGraphSpec>(`/api/graphs/${graphId}/analysis`);
}

/**
 * Save the analysis graph for a project/graph.
 *
 * @deprecated Use saveAnalysisPages instead for multi-page support.
 */
export async function saveAnalysisGraph(
  graphId: string,
  spec: AnalysisGraphSpec
): Promise<boolean> {
  try {
    const response = await fetch(`/api/graphs/${graphId}/analysis`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(spec),
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Fetch analysis pages for a project from the graph endpoint.
 * Returns null if no analysis pages exist yet.
 */
export async function fetchAnalysisPages(
  graphId: string
): Promise<AnalysisSnapshot | null> {
  try {
    const response = await fetch(`/api/graphs/${graphId}`, {
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) return null;
    const data = await response.json();
    const wirePages = data.analysis_pages as AnalysisPageWire[] | null;
    if (!wirePages || wirePages.length === 0) return null;
    const pages = wirePages.map(pageFromWire);
    return {
      pages,
      activePageId: pages[0].id,
    };
  } catch {
    return null;
  }
}

/**
 * Save analysis pages for a project via the graph update endpoint.
 * Sends only the analysis_pages field (graph/ui_state are omitted).
 */
export async function saveAnalysisPages(
  graphId: string,
  snapshot: AnalysisSnapshot,
): Promise<boolean> {
  try {
    const wirePages = snapshot.pages.map(pageToWire);
    await updateGraph(graphId, null, null, wirePages);
    return true;
  } catch {
    return false;
  }
}
