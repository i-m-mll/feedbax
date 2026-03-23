/** Types for figure gallery, matching backend Pydantic models in feedbax/web/api/figures.py */

export interface FigureInfo {
  hash: string;
  evaluation_hash: string;
  identifier: string;
  figure_type: string;
  saved_formats: string[];
  created_at: string;
  modified_at: string;
  expt_name: string | null;
  pert__type: string | null;
  pert__std: number | null;
  model_hashes: string[] | null;
}

export interface FigureListResponse {
  items: FigureInfo[];
  total: number;
  limit: number;
  offset: number;
}

export interface FigureDetail extends FigureInfo {
  available_files: string[];
}

export interface EvaluationFigureSummary {
  evaluation_hash: string;
  expt_name: string | null;
  figure_count: number;
  latest_figure_date: string | null;
}

export interface FigureFilters {
  evaluation_hash?: string;
  expt_name?: string;
  figure_type?: string;
  pert_type?: string;
  identifier?: string;
}
