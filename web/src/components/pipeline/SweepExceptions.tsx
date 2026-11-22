import { X } from 'lucide-react';
import { useEffect, useRef, type ReactNode } from 'react';
import { appTip } from '@window-shell/browser/tooltip';
import { studioName, studioRunReason, studioRunStatus } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { ListRow } from '@/components/ui/ListRow';
import { StatusMark } from '@/components/ui/StatusMark';

/** An exception to a step's sweep: what it changes (`Batch Size 20`) and for which runs (`seed 2 · 2 runs`). */
export interface SweepExceptionView {
  readonly id: string;
  readonly text: string;
  /** The field of the rule choosing its runs, under its row. */
  readonly rule?: ReactNode;
}

/** A step's exceptions; `revealed` is the one a mention of it showed, drawn selected and scrolled to. */
export function SweepExceptions({ exceptions, revealed, refusal, onRemove }: {
  exceptions: readonly SweepExceptionView[];
  revealed?: string;
  /** The exception whose last gesture was refused, and why. */
  refusal?: { id: string; reason: string };
  onRemove?: (id: string) => void;
}) {
  const shown = useRef<HTMLDivElement>(null);
  useEffect(() => { shown.current?.scrollIntoView({ block: 'nearest' }); }, [revealed]);
  return <div className="grid gap-px p-1" role="list" aria-label={studioName('exceptions')}>
      {exceptions.map(({ id, text, rule }) => <div key={id} role="listitem" className="grid gap-px"
        aria-current={id === revealed || undefined}><ListRow name={text}
        ref={id === revealed ? shown : undefined}
        data-selected={id === revealed || undefined}
        trailing={<>
          {refusal?.id === id && <span className="inline-flex" aria-description={refusal.reason} {...appTip(studioRunReason({ reason: refusal.reason }))}>
            <StatusMark status="cannot_run" label={studioRunStatus('cannot_run')} />
          </span>}
          {onRemove && <IconButton size="mark" label={studioName('removeException')} icon={<X aria-hidden="true" />}
            onClick={() => onRemove(id)} />}
        </>} />{rule && <div className="px-1">{rule}</div>}</div>)}
  </div>;
}
