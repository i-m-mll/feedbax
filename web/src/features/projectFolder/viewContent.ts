import { studioName, studioReason, studioRunReason } from '@/components/activity/studioVocabulary';
import type { PlanStepView } from '@/features/pipeline/planView';
import type { ProjectWindowView, ViewOffer } from '@/features/projectWindow/projectWindowState';
import type { ModelDocument, StudioProjectWindowMapProblem } from '@/generated/studioContracts';

export const viewsByKind = {
  train: ['runs', 'model', 'task', 'objective', 'training', 'scene', 'results'],
  evaluate: ['runs', 'model', 'task', 'objective', 'scene', 'results'],
  analyze: ['runs', 'analysis', 'figures', 'results'],
  report: ['runs', 'document'],
} as const satisfies Record<PlanStepView['kind'], readonly ProjectWindowView[]>;

/** What the selection gives a step's views to show. */
export interface ViewSubject {
  readonly step: Pick<PlanStepView, 'kind' | 'label'>;
  /** Whether the Training form applies: the step reads a training document and the selection edits it. */
  readonly trainingForm: boolean;
  /** Whether the plan has been read; until it has, the views drawn from it are loading. */
  readonly planRead: boolean;
  /** The step's run its Task and Objective are read through: none, one with a definition, or one whose own definition
   * is refused, for which the definition views are dimmed by its problems. */
  readonly run: 'none' | 'resolved' | 'unresolved';
  /** Whether the step is scored by an Objective: its own, or that of the Train whose checkpoint it evaluates. */
  readonly objective: boolean;
  /** The stored Model the step's Model and Scene show. */
  readonly model: ModelDocument | undefined;
  /** Whether the Model has a mechanism the Scene draws; unknown while its geometry is read. */
  readonly scenePlaces: boolean | undefined;
  /** Why the focused run cannot run. */
  readonly runProblems: readonly StudioProjectWindowMapProblem[];
  /** Whether the Results have something to show: a focused run's results or its operations' products, or the plan's
   * read failed. */
  readonly results: boolean;
  /** Whether the step holds a report document. */
  readonly report: boolean;
}

/** Why a view of the step has nothing to show for the selection; undefined when it has something. */
export function viewAbsence(view: ProjectWindowView, subject: ViewSubject): string | undefined {
  const { step } = subject;
  if (!subject.planRead && (view === 'task' || view === 'objective' || view === 'results')) return undefined;
  if (subject.run === 'unresolved' && (view === 'task' || view === 'objective' || view === 'training')) {
    return studioRunReason({ problems: subject.runProblems });
  }
  switch (view) {
    case 'model':
      return subject.model ? undefined : studioReason('definitionUnavailable', studioName('model'));
    case 'objective':
      if (!subject.objective) return studioReason('definitionUnavailable', studioName('objective'));
      return subject.run === 'none' ? studioReason('noRuns', step.label) : undefined;
    case 'task':
      return subject.run === 'none' ? studioReason('noRuns', step.label) : undefined;
    case 'training':
      return subject.trainingForm ? undefined : studioReason('definitionUnavailable', studioName('training'));
    case 'scene':
      if (!subject.model) return studioReason('definitionUnavailable', studioName('model'));
      return subject.scenePlaces === false ? studioReason('planarSceneUnavailable', studioName('model')) : undefined;
    case 'results':
      return subject.results ? undefined : studioReason('emptySelection', studioName('run'));
    case 'document':
      return subject.report ? undefined : studioReason('definitionUnavailable', studioName('document'));
    default:
      return undefined;
  }
}

export function viewOffer(subject: ViewSubject): ViewOffer {
  const available: ProjectWindowView[] = [];
  const absent: Partial<Record<ProjectWindowView, string>> = {};
  for (const view of viewsByKind[subject.step.kind]) {
    const reason = viewAbsence(view, subject);
    if (reason === undefined) available.push(view);
    else absent[view] = reason;
  }
  return { available, absent };
}
