import { X } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { studioName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { ListRow } from '@/components/ui/ListRow';

/** An exception to a step's sweep: what it changes (`Batch Size 20`) and for which runs (`seed 2 · 2 runs`). */
export interface SweepExceptionView {
  readonly id: string;
  readonly text: string;
}

/** A step's exceptions; `revealed` is the one a mention of it showed, drawn selected and scrolled to. */
export function SweepExceptions({ exceptions, revealed, onRemove }: {
  exceptions: readonly SweepExceptionView[];
  revealed?: string;
  onRemove?: (id: string) => void;
}) {
  const shown = useRef<HTMLDivElement>(null);
  useEffect(() => { shown.current?.scrollIntoView({ block: 'nearest' }); }, [revealed]);
  return <div className="grid gap-px p-1" role="list" aria-label={studioName('exceptions')}>
      {exceptions.map(({ id, text }) => <ListRow key={id} role="listitem" name={text}
        ref={id === revealed ? shown : undefined} aria-current={id === revealed || undefined}
        data-selected={id === revealed || undefined}
        trailing={onRemove && <IconButton size="mark" label={studioName('removeException')} icon={<X aria-hidden="true" />}
          onClick={() => onRemove(id)} />} />)}
  </div>;
}
