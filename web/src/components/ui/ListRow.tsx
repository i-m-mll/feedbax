import clsx from 'clsx';
import { ChevronRight } from 'lucide-react';
import { forwardRef, type HTMLAttributes, type ReactNode } from 'react';
import { focusRing, glyph } from './styles';

export interface ListRowProps extends Omit<HTMLAttributes<HTMLDivElement>, 'children'> {
  name: ReactNode;
  depth?: number;
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
  glyph?: ReactNode;
  trailing?: ReactNode;
  selected?: boolean;
  disabled?: boolean;
}

export const ListRow = forwardRef<HTMLDivElement, ListRowProps>(function ListRow(
  { name, depth = 0, expanded, onExpandedChange, glyph: icon, trailing, selected, disabled, className, style, ...props }, ref,
) {
  return (
    <div
      ref={ref}
      aria-selected={selected}
      aria-disabled={disabled || undefined}
      aria-expanded={expanded}
      data-selected={selected || undefined}
      data-disabled={disabled || undefined}
      className={clsx(
        'flex h-row min-w-0 cursor-default select-none items-center gap-1.5 rounded-row pr-row-inline text-shell-foreground',
        'hover:[&:not([data-disabled])]:bg-shell-hover data-[selected]:bg-shell-selected data-[selected]:font-medium',
        'data-[disabled]:text-shell-muted',
        focusRing, glyph, className,
      )}
      style={{ paddingInlineStart: `calc(var(--row-padding-inline) + ${depth} * 1.25em)`, ...style }}
      {...props}
    >
      <span
        className={clsx('flex w-[1.15em] justify-center text-shell-muted transition-transform', expanded && 'rotate-90')}
        onClick={expanded === undefined ? undefined : (event) => { event.stopPropagation(); onExpandedChange?.(!expanded); }}
        aria-hidden
      >
        {expanded === undefined ? null : <ChevronRight />}
      </span>
      {icon ? <span className="flex text-shell-muted">{icon}</span> : null}
      <span className="min-w-0 flex-1 truncate">{name}</span>
      {trailing ? <span className="flex shrink-0 items-center gap-1 text-sm text-shell-muted">{trailing}</span> : null}
    </div>
  );
});
