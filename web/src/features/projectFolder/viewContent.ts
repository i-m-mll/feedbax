import { studioName, studioReason, studioRunReason } from '@/components/activity/studioVocabulary';
import type { PlanStepView } from '@/features/pipeline/planView';
import type { ProjectWindowView, ViewOffer } from '@/features/projectWindow/projectWindowState';
import type {
  GraphInterior, ModelDocument, StudioProjectWindowMapProblem, StudioSavedEvaluationRow,
} from '@/generated/studioContracts';

export const viewsByKind = {
  train: ['runs', 'model', 'task', 'objective', 'training', 'scene', 'results'],
  evaluate: ['runs', 'model', 'task', 'scene', 'results'],
  analyze: ['runs', 'analysis', 'figures'],
  report: ['runs', 'document'],
} as const satisfies Record<PlanStepView['kind'], readonly ProjectWindowView[]>;

/** What the selection gives a step's views to show. */
export interface ViewSubject {
  readonly step: Pick<PlanStepView, 'kind' | 'label'>;
  /** Whether the Training form applies: the step reads a training document and the selection edits it. */
  readonly trainingForm: boolean;
  /** Whether the plan has been read; until it has, the views drawn from it are loading. */
  readonly planRead: boolean;
  /** Whether the step has a run, which its Task and Objective are read through. */
  readonly run: boolean;
  /** The stored Model the step's Model and Scene show. */
  readonly model: ModelDocument | undefined;
  /** The Evaluate run whose trials the Scene replays. */
  readonly evaluation: StudioSavedEvaluationRow | undefined;
  /** Why the focused run cannot run. */
  readonly runProblems: readonly StudioProjectWindowMapProblem[];
  /** Whether the Results have something to show: a focused run's results, or the plan's read failed. */
  readonly results: boolean;
  /** Whether the step holds a report document. */
  readonly report: boolean;
}

/** The interiors of a Model drawn in the Scene: its planar multibody composites, outermost first. */
export function planarInteriors(graph: GraphInterior, prefix: string[] = []): string[][] {
  return Object.entries(graph.components ?? {}).flatMap(([id, component]) => {
    if (component.kind !== 'composite') return [];
    const path = [...prefix, id];
    const child = component.interior;
    if (child.physical_domain === 'planar_multibody') return [path];
    return child.physical_domain == null ? planarInteriors(child as GraphInterior, path) : [];
  });
}

/** Why a view of the step has nothing to show for the selection; undefined when it has something. */
export function viewAbsence(view: ProjectWindowView, subject: ViewSubject): string | undefined {
  const { step } = subject;
  if (!subject.planRead && (view === 'task' || view === 'objective' || view === 'results' ||
    view === 'scene' && step.kind === 'evaluate')) return undefined;
  switch (view) {
    case 'model':
      return subject.model ? undefined : studioReason('definitionUnavailable', studioName('model'));
    case 'task':
    case 'objective':
      return subject.run ? undefined : studioReason('noRuns', step.label);
    case 'training':
      return subject.trainingForm ? undefined : studioReason('definitionUnavailable', studioName('training'));
    case 'scene':
      if (step.kind === 'evaluate') {
        const { evaluation, runProblems } = subject;
        if (evaluation?.result?.job_id && evaluation.trials.length) return undefined;
        if (runProblems.length) return studioRunReason({ problems: runProblems });
        return evaluation?.reason ?? studioReason('scenePlaybackUnavailable', studioName('playback'));
      }
      if (!subject.model) return studioReason('definitionUnavailable', studioName('model'));
      return planarInteriors(subject.model.interior).length
        ? undefined : studioReason('planarSceneUnavailable', studioName('model'));
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
