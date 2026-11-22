import { appTip } from '@window-shell/browser/tooltip';
import clsx from 'clsx';
import { studioName } from '@/components/activity/studioVocabulary';
import { Menu, MenuContent, MenuItem, MenuTrigger } from './Menu';
import { focusRing } from './styles';

/** M on a value that differs, A on an element that was added, D on one that was removed. */
export type ChangeLetter = 'M' | 'A' | 'D';

const names = { M: 'changed', A: 'added', D: 'removed' } as const satisfies Record<ChangeLetter, string>;

export interface ChangeMarkProps {
  letter?: ChangeLetter;
  /** What the thing differs from; the mark's tip, and the description of the element `id` names it for. */
  tip: string;
  /** The id of the tip's text, for the control the mark sits on to be described by it. */
  id?: string;
  /** Gives the thing what it differs from; without it the mark is only the mark. */
  reset?: () => void;
  className?: string;
}

/**
 * The mark of a change, at the trailing end inside a field or cell or in a card's top trailing corner. It never takes a
 * line of its own; pressing it offers Reset.
 */
export function ChangeMark({ letter = 'M', tip, id, reset, className }: ChangeMarkProps) {
  const name = studioName(names[letter]);
  const look = clsx(
    'inline-flex size-[1.5em] shrink-0 select-none items-center justify-center rounded-control text-xs font-semibold leading-none text-shell-change',
    className,
  );
  const description = id ? <span id={id} hidden>{tip}</span> : null;
  if (!reset) {
    return <span role="img" aria-label={name} aria-description={tip} data-change-mark={letter} className={look} {...appTip(tip)}>
      {letter}{description}
    </span>;
  }
  return <Menu>
    <MenuTrigger aria-label={name} aria-description={tip} data-change-mark={letter} {...appTip(tip)}
      className={clsx(look, focusRing, 'hover:bg-shell-hover data-[popup-open]:bg-shell-hover')}>
      {letter}{description}
    </MenuTrigger>
    <MenuContent aria-label={name}>
      <MenuItem onClick={reset}>{studioName('reset')}</MenuItem>
    </MenuContent>
  </Menu>;
}
