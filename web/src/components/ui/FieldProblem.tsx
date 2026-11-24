import { appTip } from '@window-shell/browser/tooltip';
import clsx from 'clsx';
import { TriangleAlert } from 'lucide-react';
import { studioName } from '@/components/activity/studioVocabulary';

export interface FieldProblemProps {
  /** Why the field's draft is refused, or why its stored value asks for something that does not exist. */
  reason: string;
  /** The id of the reason's text, for the field's control to be described by it. */
  id: string;
  className?: string;
}

/** A field's problem: the exception mark at the trailing end inside the field, its tip the reason. */
export function FieldProblem({ reason, id, className }: FieldProblemProps) {
  return <span role="img" aria-label={studioName('problem')} data-field-problem="" {...appTip(reason)}
    className={clsx('inline-flex size-[1.5em] shrink-0 items-center justify-center text-shell-status-failure [&_svg]:size-[1.15em]', className)}>
    <TriangleAlert aria-hidden="true" />
    <span id={id} hidden>{reason}</span>
  </span>;
}
