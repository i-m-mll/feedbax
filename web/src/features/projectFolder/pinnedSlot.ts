import type { PlanStepView, PlanView } from '@/features/pipeline/planView';
import { noViews, type ProjectWindowView, type SlotPin, type ViewOffer } from '@/features/projectWindow/projectWindowState';
import type { StudioSavedTrainRuns } from '@/generated/studioContracts';
import { useSceneGeometry } from '@/components/scene/useSceneGeometry';
import type { StateValues } from '@/components/scene/trialScene';
import { modelDocumentId } from './authoredEdits';
import { evaluateSubject } from './evaluateAuthoring';
import { useStoredModel } from './managedProjectDetails';
import type { ProjectEditController } from './projectEditController';
import { savedRunId } from './projectWindowPlan';
import { focusedResult } from './SavedResults';
import { useTrialAuthoring } from './StudioTrialAuthoring';
import { viewOffer } from './viewContent';
import { useSyncExternalStore } from 'react';

/** The views a slot can keep while another step is selected: those that show a step without editing it. */
export const pinnableViews: ReadonlySet<ProjectWindowView> = new Set(['model', 'scene', 'analysis', 'figures']);

const restPose: readonly StateValues[] = [{}];

/** The Model a step's Model and Scene show: its own, or for an Evaluate the one its subject's Train trains. */
export function stepModelSource(plan: PlanView, step: PlanStepView | undefined) {
  if (step?.kind !== 'evaluate') return undefined;
  const subject = step.inputs.find(({ id }) => id === 'subject')?.sourceStepId;
  return modelDocumentId(plan.steps.find(({ id }) => id === subject)?.documentIds);
}

/**
 * What a pinned slot shows of its step while another step is selected: the step, the run of it the pin names (else its
 * first), that run's results, its Model, the views it offers among those a pin keeps, and, when `clocked`, the trials
 * the clock reads from it. Nothing while the pin's step is the selected one, which the selection shows.
 */
export function usePinnedSubject({ editor, plan, shown, variant, pin, selectedStep, clocked }: {
  editor: ProjectEditController;
  plan: PlanView;
  shown: StudioSavedTrainRuns | undefined;
  variant: string | null;
  pin: SlotPin | undefined;
  selectedStep: string | undefined;
  clocked: boolean;
}) {
  const { saved } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const step = pin && pin.stepId !== selectedStep ? plan.steps.find(({ id }) => id === pin.stepId) : undefined;
  const rows = step ? shown?.window_map?.rows.filter(({ key }) => key.step === step.id && key.variant === variant) ?? [] : [];
  const namedRow = rows.find(({ key }) => savedRunId(key) === pin?.runId) ?? rows[0];
  const namedRun = namedRow?.key;
  const result = focusedResult(shown, step?.id, namedRun && savedRunId(namedRun), new Set(step?.runs.map(({ id }) => id)), undefined);
  const stored = useStoredModel(editor, step, stepModelSource(plan, step));
  const rest = useSceneGeometry(saved.project_id, step && stored ? { document_id: stored.id, revision: saved.revision } : null, restPose)?.(0);
  const authored = step && saved.project.steps?.[step.id];
  const scoredBy = authored?.kind === 'evaluate' ? evaluateSubject(authored) : undefined;
  const trialStep = step?.kind === 'train' || step?.kind === 'synthesize' || step?.kind === 'evaluate';
  const trial = useTrialAuthoring(editor, saved.project_id, saved.revision, namedRun, clocked && trialStep && namedRow?.resolved !== false,
    false, () => undefined, async () => false, async () => undefined);
  const offered = step ? viewOffer({
    step, trainingForm: false, planRead: !!shown, run: !namedRow ? 'none' : namedRow.resolved === false ? 'unresolved' : 'resolved',
    objective: authored?.kind !== 'evaluate' || scoredBy?.kind === 'steps' || scoredBy?.kind === 'document' && !!scoredBy.objective,
    model: stored?.value,
    scenePlaces: rest?.kind === 'ready' ? rest.geometry.mechanisms.length > 0 : undefined,
    runProblems: [], results: false, report: false,
  }) : noViews;
  const views: ViewOffer = {
    available: offered.available.filter((view) => pinnableViews.has(view)),
    absent: offered.absent,
  };
  return { step, namedRun, evaluation: result.evaluation, stored, views, trial: clocked ? trial.view : undefined };
}
