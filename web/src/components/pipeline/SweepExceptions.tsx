import { X } from 'lucide-react';
import { studioName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { ListRow } from '@/components/ui/ListRow';
import { Section } from '@/components/ui/Section';

/** An exception to a step's sweep: what it changes (`Batch Size 20`) and for which runs (`seed 2 · 2 runs`). */
export interface SweepExceptionView {
  readonly id: string;
  readonly text: string;
}

export function SweepExceptions({ exceptions, onRemove }: {
  exceptions: readonly SweepExceptionView[];
  onRemove?: (id: string) => void;
}) {
  return <Section name={studioName('exceptions')} count={exceptions.length}>
    <div className="grid gap-px p-1" role="list">
      {exceptions.map(({ id, text }) => <ListRow key={id} role="listitem" name={text}
        trailing={onRemove && <IconButton size="mark" label={studioName('removeException')} icon={<X aria-hidden="true" />}
          onClick={() => onRemove(id)} />} />)}
    </div>
  </Section>;
}
