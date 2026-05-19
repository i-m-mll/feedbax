/**
 * Registry of example projects (formerly "templates").
 *
 * Each entry provides pre-configured analysis pages and an optional
 * model graph that can be loaded into a new project tab. When loaded,
 * these are saved to the backend as real projects.
 */

import type { AnalysisSnapshot } from '@/types/analysis';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { StudioWorkspaceSpec } from '@/types/workspace';
import { createRlrmpPart1Analysis, RLRMP_PART1_TEMPLATE } from '@/data/rlrmp-part1';
import { createRlrmpPart2Project, RLRMP_PART2_META } from '@/data/rlrmp-part2';
import { createRlrmpModelGraph } from '@/data/rlrmp-model-graph';
import {
  createRlrmpMovementRampAnalysis,
  RLRMP_MOVEMENT_RAMP_TEMPLATE,
  seedRlrmpMovementRampWorkspace,
} from '@/data/rlrmp-run-example';

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
  /** Optional hook for enriching the default Studio workspace snapshot. */
  createWorkspace?: (args: {
    baseWorkspace: StudioWorkspaceSpec;
    graph: GraphSpec;
    uiState: GraphUIState;
    analysisSnapshot: AnalysisSnapshot;
  }) => StudioWorkspaceSpec;
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
  {
    ...RLRMP_MOVEMENT_RAMP_TEMPLATE,
    createAnalysis: createRlrmpMovementRampAnalysis,
    createModelGraph: () => createRlrmpModelGraph('RLRMP movement-ramp training runs'),
    createWorkspace: ({ baseWorkspace }) => seedRlrmpMovementRampWorkspace(baseWorkspace),
  },
];
