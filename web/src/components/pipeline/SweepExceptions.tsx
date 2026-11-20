import { X } from 'lucide-react';
import { appTip } from '@window-shell/browser/tooltip';
import { studioName, studioRunReason, studioRunStatus } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { ListRow } from '@/components/ui/ListRow';
import { StatusMark } from '@/components/ui/StatusMark';

/** An exception to a step's sweep: what it changes (`Batch Size 20`) and for which runs (`seed 2 · 2 runs`). */
export interface SweepExceptionView {
  readonly id: string;
  readonly text: string;
}

export function SweepExceptions({ exceptions, refusal, onRemove }: {
  exceptions: readonly SweepExceptionView[];
  /** The exception whose last gesture was refused, and why. */
  refusal?: { id: string; reason: string };
  onRemove?: (id: string) => void;
}) {
  return <div className="grid gap-px p-1" role="list" aria-label={studioName('exceptions')}>
      {exceptions.map(({ id, text }) => <ListRow key={id} role="listitem" name={text}
        trailing={<>
          {refusal?.id === id && <span className="inline-flex" aria-description={refusal.reason} {...appTip(studioRunReason({ reason: refusal.reason }))}>
            <StatusMark status="cannot_run" label={studioRunStatus('cannot_run')} />
          </span>}
          {onRemove && <IconButton size="mark" label={studioName('removeException')} icon={<X aria-hidden="true" />}
            onClick={() => onRemove(id)} />}
        </>} />)}
  </div>;
}
