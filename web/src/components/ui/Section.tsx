import clsx from 'clsx';
import { ChevronRight } from 'lucide-react';
import { useId, useState, type ReactNode } from 'react';
import { ScrollFade } from './ScrollFade';
import { focusRing, glyph } from './styles';

export interface SectionProps {
  name: ReactNode;
  /** Id for the name, for a body that is labelled by it. */
  nameId?: string;
  count?: number;
  /** Trailing commands, drawn in the header. */
  actions?: ReactNode;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: ReactNode;
  /** A section inside another's body: no rule above it, its header indented under its parent's name. */
  nested?: boolean;
  className?: string;
  bodyClassName?: string;
}

/**
 * One fold of a drawer. Sections are stacked in a flex column: each takes its content's height,
 * shrinks with the others when the column is short, and scrolls its own body.
 */
export function Section({
  name, nameId, count, actions, open: controlled, defaultOpen = true, onOpenChange, nested, children, className, bodyClassName,
}: SectionProps) {
  const [uncontrolled, setUncontrolled] = useState(defaultOpen);
  const open = controlled ?? uncontrolled;
  const body = useId();
  const generated = useId();
  const title = nameId ?? generated;
  const toggle = () => {
    if (controlled === undefined) setUncontrolled(!open);
    onOpenChange?.(!open);
  };
  return (
    <section
      aria-labelledby={title}
      data-open={open || undefined}
      className={clsx(
        'flex min-h-[var(--row-height)] min-w-0 flex-col',
        !nested && 'border-t border-shell-divider first:border-t-0',
        open ? 'shrink' : 'shrink-0',
        className,
      )}
    >
      <div className={clsx('flex h-row shrink-0 items-center gap-0.5 px-1 text-sm text-shell-muted', nested && 'pl-4', glyph)}>
        <button
          type="button"
          aria-expanded={open}
          aria-controls={body}
          onClick={toggle}
          className={clsx(
            'flex h-full min-w-0 flex-1 cursor-default select-none items-center gap-1 rounded-row px-1 text-left hover:bg-shell-row-hover',
            focusRing,
          )}
        >
          <ChevronRight aria-hidden className={clsx('transition-transform', open && 'rotate-90')} />
          <span id={title} className="min-w-0 truncate font-medium text-shell-foreground">{name}</span>
          {count === undefined ? null : <span className="tabular-nums">{count}</span>}
        </button>
        {actions ? <div className="flex shrink-0 items-center gap-0.5">{actions}</div> : null}
      </div>
      <ScrollFade id={body} hidden={!open} className={clsx('flex-1', bodyClassName)}>{children}</ScrollFade>
    </section>
  );
}
