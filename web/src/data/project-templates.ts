/**
 * Registry of project templates.
 *
 * Templates provide pre-configured analysis pages that can be loaded
 * into a new project tab. Each template creates a complete
 * AnalysisSnapshot that is restored into the analysis store.
 */

import type { AnalysisSnapshot } from '@/types/analysis';
import { createRlrmpPart1Analysis, RLRMP_PART1_TEMPLATE } from '@/data/rlrmp-part1';
import { createRlrmpPart2Project, RLRMP_PART2_META } from '@/data/rlrmp-part2';

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
}

/** All available project templates. */
export const PROJECT_TEMPLATES: ProjectTemplate[] = [
  {
    ...RLRMP_PART1_TEMPLATE,
    createAnalysis: createRlrmpPart1Analysis,
  },
  {
    ...RLRMP_PART2_META,
    createAnalysis: createRlrmpPart2Project,
  },
];
