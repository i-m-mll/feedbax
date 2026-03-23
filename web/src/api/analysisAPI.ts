/**
 * API client for the analysis system.
 *
 * The backend endpoints may not exist yet — these functions define the
 * contract the frontend expects. A stub/mock layer is provided so the
 * UI is functional without a live backend.
 */

import type {
  AnalysisGraphSpec,
  AnalysisPackage,
  AnalysisClassDef,
} from '@/types/analysis';

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
 */
export async function fetchAnalysisGraph(
  graphId: string
): Promise<AnalysisGraphSpec | null> {
  return tryFetch<AnalysisGraphSpec>(`/api/graphs/${graphId}/analysis`);
}

/**
 * Save the analysis graph for a project/graph.
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
