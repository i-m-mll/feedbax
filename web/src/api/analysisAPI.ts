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

// ---------------------------------------------------------------------------
// Stub data — used when the backend is not available
// ---------------------------------------------------------------------------

const STUB_PACKAGES: AnalysisPackage[] = [
  // -----------------------------------------------------------------------
  // Preprocessing — data transforms that reshape / subset / reindex
  // -----------------------------------------------------------------------
  {
    name: 'Preprocessing',
    description: 'Data preparation and transforms',
    analyses: [
      {
        name: 'GetBestReplicate',
        description: 'Select the best-performing model replicate from an ensemble',
        category: 'Preprocessing',
        inputPorts: ['data'],
        outputPorts: ['data'],
        defaultParams: { axis: 0 },
        icon: 'Trophy',
      },
      {
        name: 'GetitemAtLevel',
        description: 'Subset data by selecting a single key at a named LDict level',
        category: 'Preprocessing',
        inputPorts: ['data'],
        outputPorts: ['data'],
        defaultParams: { level: '', key: '' },
        icon: 'ListFilter',
      },
      {
        name: 'LevelToBottom',
        description: 'Move a named LDict level to the innermost position',
        category: 'Preprocessing',
        inputPorts: ['data'],
        outputPorts: ['data'],
        defaultParams: { level: '' },
        icon: 'ArrowDownToLine',
      },
      {
        name: 'LevelToTop',
        description: 'Move a named LDict level to the outermost position',
        category: 'Preprocessing',
        inputPorts: ['data'],
        outputPorts: ['data'],
        defaultParams: { level: '' },
        icon: 'ArrowUpToLine',
      },
      {
        name: 'Indexing',
        description: 'Select specific indices along an array axis',
        category: 'Preprocessing',
        inputPorts: ['data'],
        outputPorts: ['data'],
        defaultParams: { axis: -2, indices: [], axis_label: 'timestep' },
        icon: 'ListOrdered',
      },
      {
        name: 'Unstacking',
        description: 'Expand a packed array axis into a named LDict dimension',
        category: 'Preprocessing',
        inputPorts: ['data'],
        outputPorts: ['data'],
        defaultParams: { axis: 0, level_name: '' },
        icon: 'Ungroup',
      },
      {
        name: 'SubdictAtLevel',
        description: 'Keep only a subset of keys at a specific LDict level',
        category: 'Preprocessing',
        inputPorts: ['data'],
        outputPorts: ['data'],
        defaultParams: { level: '', keys: [] },
        icon: 'FolderOpen',
      },
      {
        name: 'RearrangeLevels',
        description: 'Reorder multiple LDict levels in the data tree',
        category: 'Preprocessing',
        inputPorts: ['data'],
        outputPorts: ['data'],
        defaultParams: { level_order: [] },
        icon: 'ArrowUpDown',
      },
      {
        name: 'Stacking',
        description: 'Collapse a named LDict dimension into an array axis',
        category: 'Preprocessing',
        inputPorts: ['data'],
        outputPorts: ['data'],
        defaultParams: { level: '' },
        icon: 'Group',
      },
      {
        name: 'SegmentEpochs',
        description: 'Split trajectories into time epochs (e.g. accel/decel phases)',
        category: 'Preprocessing',
        inputPorts: ['data'],
        outputPorts: ['data'],
        defaultParams: { epoch_boundaries: [] },
        icon: 'SplitSquareHorizontal',
      },
      {
        name: 'ComplexToPolar',
        description: 'Convert complex-valued arrays to magnitude and angle components',
        category: 'Preprocessing',
        inputPorts: ['data'],
        outputPorts: ['data'],
        defaultParams: {},
        icon: 'Compass',
      },
    ],
  },

  // -----------------------------------------------------------------------
  // Computation — measure extraction and function application
  // -----------------------------------------------------------------------
  {
    name: 'Computation',
    description: 'Measure extraction and function application',
    analyses: [
      {
        name: 'AlignedVars',
        description: 'Align state trajectories to reach directions and extract response variables',
        category: 'Computation',
        inputPorts: ['data'],
        outputPorts: ['aligned_vars'],
        defaultParams: { varset: ['pos', 'vel', 'command', 'force'], directions_fn: 'reach_directions', align_epoch: null },
        icon: 'AlignHorizontalDistributeCenter',
      },
      {
        name: 'ApplyFns',
        description: 'Apply a set of measure functions to a data PyTree',
        category: 'Computation',
        inputPorts: ['input'],
        outputPorts: ['measures'],
        defaultParams: { measure_names: [] },
        icon: 'FunctionSquare',
      },
      {
        name: 'IdentityNode',
        description: 'Pass-through node for routing data without transformation',
        category: 'Computation',
        inputPorts: ['data'],
        outputPorts: ['data'],
        defaultParams: {},
        icon: 'Equal',
      },
      {
        name: 'CallWithDeps',
        description: 'Apply a function to data using results from a dependency analysis',
        category: 'Computation',
        inputPorts: ['data', 'dependency'],
        outputPorts: ['data'],
        defaultParams: { fn_name: '' },
        icon: 'Workflow',
      },
    ],
  },

  // -----------------------------------------------------------------------
  // Decomposition — dimensionality reduction and matrix decompositions
  // -----------------------------------------------------------------------
  {
    name: 'Decomposition',
    description: 'Dimensionality reduction and matrix decompositions',
    analyses: [
      {
        name: 'StatesPCA',
        description: 'Principal component analysis on neural hidden states across conditions',
        category: 'Decomposition',
        inputPorts: ['data'],
        outputPorts: ['pca_result'],
        defaultParams: { n_components: 3, aggregate_over: [] },
        icon: 'Axis3d',
      },
      {
        name: 'Jacobians',
        description: 'Compute Jacobian matrices of network functions w.r.t. their inputs',
        category: 'Decomposition',
        inputPorts: ['funcs', 'func_args'],
        outputPorts: ['jacobians'],
        defaultParams: { argnums: null },
        icon: 'Grid3x3',
      },
      {
        name: 'SVD',
        description: 'Singular value decomposition of rectangular matrices',
        category: 'Decomposition',
        inputPorts: ['matrices'],
        outputPorts: ['svd_result'],
        defaultParams: {},
        icon: 'Columns3',
      },
      {
        name: 'Eig',
        description: 'Eigenvalue decomposition of square matrices',
        category: 'Decomposition',
        inputPorts: ['matrices'],
        outputPorts: ['eig_result'],
        defaultParams: {},
        icon: 'Orbit',
      },
    ],
  },

  // -----------------------------------------------------------------------
  // Dynamics — dynamical systems analysis
  // -----------------------------------------------------------------------
  {
    name: 'Dynamics',
    description: 'Dynamical systems analysis of recurrent networks',
    analyses: [
      {
        name: 'FixedPoints',
        description: 'Find fixed points in network state space via optimization',
        category: 'Dynamics',
        inputPorts: ['funcs', 'candidates', 'func_args'],
        outputPorts: ['fp_results'],
        defaultParams: { fp_tol: 1e-7, unique_tol: 0.025, outlier_tol: 1.0, stride_candidates: 1 },
        icon: 'Crosshair',
      },
      {
        name: 'Tangling',
        description: 'Compute trajectory tangling metric measuring flow-field consistency',
        category: 'Dynamics',
        inputPorts: ['state'],
        outputPorts: ['tangling_values'],
        defaultParams: { eps: 1e-6, method: 'direct', k_neighbours: 50 },
        icon: 'Waves',
      },
      {
        name: 'FrequencyResponse',
        description: 'FFT-based frequency analysis of input-output transfer functions',
        category: 'Dynamics',
        inputPorts: ['data'],
        outputPorts: ['freq_response'],
        defaultParams: {},
        icon: 'AudioWaveform',
      },
      {
        name: 'UnitPreferences',
        description: 'Compute preferred direction tuning of individual network units',
        category: 'Dynamics',
        inputPorts: ['data'],
        outputPorts: ['unit_prefs'],
        defaultParams: { feature_fn: 'goal_positions' },
        icon: 'Radar',
      },
    ],
  },

  // -----------------------------------------------------------------------
  // Visualization — figure-producing analyses
  // -----------------------------------------------------------------------
  {
    name: 'Visualization',
    description: 'Figure generation analyses',
    analyses: [
      {
        name: 'Violins',
        description: 'Violin plots of measure distributions grouped by condition',
        category: 'Visualization',
        inputPorts: ['input', 'input_split'],
        outputPorts: ['figure'],
        defaultParams: { violinmode: 'overlay', zero_hline: false },
        icon: 'BarChart3',
      },
      {
        name: 'Profiles',
        description: 'Time-series trajectory profiles aligned to reach direction',
        category: 'Visualization',
        inputPorts: ['vars'],
        outputPorts: ['figure'],
        defaultParams: { varset: ['pos', 'vel', 'command', 'force'], mode: 'std', n_std_plot: 1 },
        icon: 'TrendingUp',
      },
      {
        name: 'EffectorTrajectories',
        description: '2D effector position plots with optional reach endpoints',
        category: 'Visualization',
        inputPorts: ['data'],
        outputPorts: ['figure'],
        defaultParams: { pos_endpoints: true, straight_guides: true, colorscale_key: 'reach_condition' },
        icon: 'Route',
      },
      {
        name: 'ScatterPlots',
        description: 'General 2D scatter/line plots over a PyTree of arrays',
        category: 'Visualization',
        inputPorts: ['input'],
        outputPorts: ['figure'],
        defaultParams: { colorscale_axis: 0, subplot_level: null },
        icon: 'ScatterChart',
      },
      {
        name: 'EigvalsPlot',
        description: 'Eigenvalue scatter plot in the complex plane',
        category: 'Visualization',
        inputPorts: ['eig_result'],
        outputPorts: ['figure'],
        defaultParams: {},
        icon: 'CircleDot',
      },
      {
        name: 'NetworkActivitySampleUnits',
        description: 'Activity traces for a random sample of individual network units',
        category: 'Visualization',
        inputPorts: ['data'],
        outputPorts: ['figure'],
        defaultParams: { n_units: 4 },
        icon: 'BrainCircuit',
      },
      {
        name: 'PlotInPCSpace',
        description: 'Visualize trajectories or fixed points projected into PCA space',
        category: 'Visualization',
        inputPorts: ['data', 'pca_result'],
        outputPorts: ['figure'],
        defaultParams: { n_components: 3 },
        icon: 'Box',
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
