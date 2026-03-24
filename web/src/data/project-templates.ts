/**
 * Registry of example projects (formerly "templates").
 *
 * Each entry provides pre-configured analysis pages and an optional
 * model graph that can be loaded into a new project tab. When loaded,
 * these are saved to the backend as real projects.
 */

import type { AnalysisSnapshot } from '@/types/analysis';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import { createRlrmpPart1Analysis, RLRMP_PART1_TEMPLATE } from '@/data/rlrmp-part1';
import { createRlrmpPart2Project, RLRMP_PART2_META } from '@/data/rlrmp-part2';
import { createRlrmpModelGraph } from '@/data/rlrmp-model-graph';

/** Metadata for a project template. */
export interface ProjectTemplate {
  /** Unique template ID. */
  id: string;
  /** Display name. */
  name: string;
  /** Short description. */
  description: string;
  /** Names of analysis pages in the template. */
  pageNames: readonly string[];
  /** Factory function that builds the analysis snapshot. */
  createAnalysis: () => AnalysisSnapshot;
  /** Optional factory for a pre-populated model graph + UI state. */
  createModelGraph?: () => { graph: GraphSpec; uiState: GraphUIState };
}

/** All available project templates. */
export const PROJECT_TEMPLATES: ProjectTemplate[] = [
  {
    ...RLRMP_PART1_TEMPLATE,
    createAnalysis: createRlrmpPart1Analysis,
    createModelGraph: () => createRlrmpModelGraph('RLRMP: Part 1'),
  },
  {
    ...RLRMP_PART2_META,
    createAnalysis: createRlrmpPart2Project,
    createModelGraph: () => createRlrmpModelGraph('rlrmp: Part 2'),
  },
];
