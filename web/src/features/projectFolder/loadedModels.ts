import { studioFormat } from '@/components/activity/studioVocabulary';
import type { ModelFitProblem, StudioLoadedModelImportResult } from '@/generated/studioContracts';

function fitProblem(problem: ModelFitProblem) {
  switch (problem.kind) {
    case 'missing_slot': return studioFormat('fitMissing', problem);
    case 'unexpected_slot': return studioFormat('fitUnexpected', problem);
    case 'mismatched_slot': return studioFormat('fitMismatched', problem);
    default: return studioFormat('fitModel', problem);
  }
}

/** Each reason an import was not taken, in the service's order. */
export function importProblems(result: Exclude<StudioLoadedModelImportResult, { status: 'imported' }>): readonly string[] {
  return result.status === 'does_not_fit' ? result.report.problems.map(fitProblem)
    : [studioFormat('unreadableArchive', result)];
}
