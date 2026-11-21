import type { StudioResultState } from '@/components/activity/studioVocabulary';
import type { PlanStepView, PlanView } from '@/features/pipeline/planView';
import type { EvaluateField } from './StudioEvaluateAuthoring';

/** A gesture the service or the editor refused, with its reason, on the object the gesture acted on. */
export type Refusal =
  | { kind: 'field'; step: string; field: EvaluateField; reason: string }
  /** Reasons by run id. */
  | { kind: 'runs'; reasons: Readonly<Record<string, string>> }
  | { kind: 'step'; step: string; reason: string }
  | { kind: 'key'; step: string; reason: string }
  | { kind: 'axis'; step: string; axis: string; reason: string }
  | { kind: 'exception'; step: string; exception: string; reason: string }
  | { kind: 'quote'; reason: string }
  | { kind: 'projectName'; reason: string }
  /** Keyed by `problemKey`. */
  | { kind: 'problem'; problem: string; reason: string };

/** A view's channel for its gestures' outcomes: `clear` as one starts, `fail` with what it threw. */
export interface GestureFailures {
  clear(): void;
  /** A service failure goes to the bottom bar; a refusal goes to `land` when given, otherwise to the step's mark. */
  fail(cause: unknown, land?: (reason: string) => void): void;
}

/** Identifies a projected problem across refreshes of the projection. */
export const problemKey = (problem: object) => JSON.stringify(problem);

function count(counts: PlanStepView['statusCounts'], state: StudioResultState, change: number) {
  const next = (counts[state] ?? 0) + change;
  const { [state]: _omitted, ...rest } = counts;
  return next > 0 ? { ...rest, [state]: next } : rest;
}

/** The plan as drawn: a refused run reads Can't run with the refusal as its reason; a refused step carries it. */
export function planWithRefusal(plan: PlanView, refusal: Refusal | undefined): PlanView {
  if (refusal?.kind === 'step') {
    return { ...plan, steps: plan.steps.map((step) => step.id === refusal.step ? { ...step, refusal: refusal.reason } : step) };
  }
  if (refusal?.kind !== 'runs') return plan;
  const { reasons } = refusal;
  return {
    ...plan,
    steps: plan.steps.map((step) => {
      if (!step.runs.some(({ id }) => reasons[id] !== undefined)) return step;
      let statusCounts = step.statusCounts;
      const runs = step.runs.map((run) => {
        const reason = reasons[run.id];
        if (reason === undefined) return run;
        statusCounts = count(count(statusCounts, run.state, -1), 'cannot_run', 1);
        return { ...run, state: 'cannot_run' as const, reason, problems: [], change: undefined, openingReason: undefined };
      });
      return { ...step, runs, statusCounts };
    }),
  };
}
