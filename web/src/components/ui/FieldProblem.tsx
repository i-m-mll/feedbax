import { appTip } from '@window-shell/browser/tooltip';
import clsx from 'clsx';
import { TriangleAlert } from 'lucide-react';
import { studioName } from '@/components/activity/studioVocabulary';

export interface FieldProblemProps {
  /** Why the field's draft is refused, or why its stored value asks for something that does not exist. */
  reason: string;
  /** The id of the reason's text, for the field's control to be described by it. A mark inside a control whose text
   * is its value leaves it out and draws `FieldProblemReason` beside the control. */
  id?: string;
  className?: string;
}

/** A Problem's mark, as at the trailing end inside a field: its tip and its description the reason. */
export function FieldProblem({ reason, id, className }: FieldProblemProps) {
  return <span role="img" aria-label={studioName('problem')} aria-description={reason} data-field-problem="" {...appTip(reason)}
    className={clsx('inline-flex size-[1.5em] shrink-0 items-center justify-center text-shell-status-failure [&_svg]:size-[1.15em]', className)}>
    <TriangleAlert aria-hidden="true" />
    {id && <FieldProblemReason reason={reason} id={id} />}
  </span>;
}

/** The reason's text, which describes the field's control. */
export function FieldProblemReason({ reason, id }: { reason: string; id: string }) {
  return <span id={id} hidden>{reason}</span>;
}
