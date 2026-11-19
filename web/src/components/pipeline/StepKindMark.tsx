import { appTip } from '@window-shell/browser/tooltip';
import { studioName, studioStep } from '@/components/activity/studioVocabulary';
import type { PlanStepView } from '@/features/pipeline/planView';

/** A step's kind as its glyph, named by its tip, before the step's name. */
export function StepKindMark({ kind }: { kind: PlanStepView['kind'] }) {
  const { Icon } = studioStep(kind);
  return <span className="inline-flex shrink-0" role="img" aria-label={studioName(kind)} {...appTip(studioName(kind))}>
    <Icon aria-hidden="true" className="size-4" />
  </span>;
}
