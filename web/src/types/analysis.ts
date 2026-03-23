/** Types for analysis node configuration and figure generation. */

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
