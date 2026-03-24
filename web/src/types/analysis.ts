/** Types for the analysis DAG system and figure generation. */

/** A parameter value on an analysis or transform node. */
export type AnalysisParamValue = number | string | boolean | null;

/** Configuration for a transform (prep op) applied on an edge. */
export interface TransformSpec {
  /** Unique ID within the analysis graph. */
  id: string;
  /** Transform class name (e.g. "Normalize", "PCA", "Downsample"). */
  type: string;
  /** User-facing label. */
  label: string;
  /** Config parameters for this transform. */
  params: Record<string, AnalysisParamValue>;
}

/** An analysis node in the DAG. */
export interface AnalysisNodeSpec {
  /** Unique ID within the analysis graph. */
  id: string;
  /** Analysis class name (e.g. "Violins", "Profiles", "PCA"). */
  type: string;
  /** User-facing label. */
  label: string;
  /** Category for grouping in the palette. */
  category: string;
  /** Input port names. */
  inputPorts: string[];
  /** Output port names. */
  outputPorts: string[];
  /** Config parameters. */
  params: Record<string, AnalysisParamValue>;
  /** Whether this is a dependency (smaller/muted) or a full analysis. */
  role: 'analysis' | 'dependency';
}

/** A wire in the analysis DAG. */
export interface AnalysisWire {
  id: string;
  sourceId: string;
  sourcePort: string;
  targetId: string;
  targetPort: string;
  /** Whether this is an implicit data dependency (dashed/muted). */
  implicit: boolean;
  /** Transform applied on this edge, if any. */
  transform?: TransformSpec;
}

/** The complete analysis graph specification. */
export interface AnalysisGraphSpec {
  nodes: Record<string, AnalysisNodeSpec>;
  wires: AnalysisWire[];
  /** The data source node ID (always present as the leftmost node). */
  dataSourceId: string;
}

/** An analysis class available in the palette. */
export interface AnalysisClassDef {
  /** Class name (e.g. "Violins"). */
  name: string;
  /** Human-readable description. */
  description: string;
  /** Category for grouping. */
  category: string;
  /** Default input port names. */
  inputPorts: string[];
  /** Default output port names. */
  outputPorts: string[];
  /** Default parameter values. */
  defaultParams: Record<string, AnalysisParamValue>;
  /** Icon name from lucide-react. */
  icon: string;
}

/** An analysis package (group of related analysis classes). */
export interface AnalysisPackage {
  name: string;
  description: string;
  analyses: AnalysisClassDef[];
}

/** A fig op applied to an analysis node's figure generation pipeline. */
export interface FigOpSpec {
  name: string;
  params: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

/** A final op applied to results or figures after generation. */
export interface FinalOpSpec {
  name: string;
  fn_name: string;
  params?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

/** Analysis-specific metadata attached to a node spec. */
export interface AnalysisNodeMeta {
  /** Whether this analysis implements make_figs(). */
  has_make_figs: boolean;
  /** Ordered list of fig ops in the pipeline. */
  fig_ops: FigOpSpec[];
  /** Final ops grouped by type (results, figs). */
  final_ops_by_type: Record<string, FinalOpSpec[]>;
  /** Dependency port names this analysis expects. */
  dependency_ports: string[];
  /** Analysis class name. */
  analysis_class: string;
}

// ---------------------------------------------------------------------------
// Multi-page analysis persistence types
// ---------------------------------------------------------------------------

/** Viewport state for an analysis canvas page. */
export interface AnalysisViewport {
  x: number;
  y: number;
  zoom: number;
}

/** Extensible evaluation parametrization for an analysis page. */
export interface EvalParametrization {
  [key: string]: unknown;
}

/** Persisted specification for a single analysis page (tab within analysis). */
export interface AnalysisPageSpec {
  /** Unique page ID. */
  id: string;
  /** User-facing page name. */
  name: string;
  /** The analysis DAG graph for this page. */
  graphSpec: AnalysisGraphSpec;
  /** Evaluation parameters for this page. */
  evalParams: EvalParametrization;
  /** Viewport position/zoom for this page. */
  viewport: AnalysisViewport;
}

/** Snapshot of all analysis state — used by projectsStore for tab switching. */
export interface AnalysisSnapshot {
  /** All analysis pages. */
  pages: AnalysisPageSpec[];
  /** Which page is currently active (loaded into React Flow). */
  activePageId: string | null;
}

// ---------------------------------------------------------------------------
// Figure request types
// ---------------------------------------------------------------------------

/** Status of a demand-driven figure generation request. */
export type FigureRequestStatus = 'idle' | 'running' | 'ready' | 'error';

/** Tracks a single figure generation request. */
export interface FigureRequest {
  nodeId: string;
  status: FigureRequestStatus;
  figureHash?: string;
  error?: string;
  requestedAt?: number;
  completedAt?: number;
}

/** Response from the figure generation endpoint. */
export interface GenerateFigureResponse {
  request_id: string;
  status: string;
}

/** Response from the figure status check endpoint. */
export interface FigureStatusResponse {
  request_id: string;
  status: 'pending' | 'running' | 'complete' | 'error';
  figure_hashes?: string[];
  error?: string;
}
