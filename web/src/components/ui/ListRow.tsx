import clsx from 'clsx';
import { ChevronRight } from 'lucide-react';
import { forwardRef, type CSSProperties, type HTMLAttributes, type ReactNode } from 'react';
import { focusRing, glyph } from './styles';

export interface ListRowProps extends Omit<HTMLAttributes<HTMLDivElement>, 'children'> {
  name: ReactNode;
  depth?: number;
  /** A branch's state; `null` keeps the disclosure's place empty for a leaf among branches; undefined draws no place. */
  expanded?: boolean | null;
  onExpandedChange?: (expanded: boolean, deep: boolean) => void;
  glyph?: ReactNode;
  trailing?: ReactNode;
  selected?: boolean;
  holdsSelection?: boolean;
  disabled?: boolean;
}

export const ListRow = forwardRef<HTMLDivElement, ListRowProps>(function ListRow(
  { name, depth = 0, expanded, onExpandedChange, glyph: icon, trailing, selected, holdsSelection, disabled, className, style, ...props }, ref,
) {
  const indent = `calc(var(--space-1) + ${depth} * 1.25em)`;
  return (
    <div
      ref={ref}
      data-list-row=""
      aria-selected={selected}
      aria-disabled={disabled || undefined}
      aria-expanded={expanded ?? undefined}
      data-selected={selected || undefined}
      data-holds-selection={holdsSelection || undefined}
      data-disabled={disabled || undefined}
      className={clsx(
        'relative flex h-row min-w-0 cursor-default select-none items-center gap-1.5 rounded-list-row pr-row-inline text-shell-foreground',
        'hover:[&:not([data-disabled])]:bg-shell-row-hover data-[selected]:bg-shell-row-selected data-[selected]:font-medium',
        'data-[disabled]:text-shell-muted',
        'after:pointer-events-none after:absolute after:bottom-[2px] after:left-[calc(var(--row-indent)+1.15em+0.375rem)]',
        'after:h-[2px] after:w-[1.25em] after:rounded-full after:bg-shell-foreground after:opacity-0',
        'data-[holds-selection]:after:opacity-100',
        focusRing, glyph, className,
      )}
      style={{ paddingInlineStart: indent, '--row-indent': indent, ...style } as CSSProperties}
      {...props}
    >
      {expanded === undefined ? null : <span
        className={clsx('flex w-[1.15em] shrink-0 justify-center text-shell-muted transition-transform', expanded && 'rotate-90')}
        onClick={expanded === null ? undefined : (event) => { event.stopPropagation(); onExpandedChange?.(!expanded, event.altKey); }}
        onDoubleClick={expanded === null ? undefined : (event) => event.stopPropagation()}
        aria-hidden
      >
        {expanded === null ? null : <ChevronRight />}
      </span>}
      {icon ? <span className="flex text-shell-muted">{icon}</span> : null}
      <span className="min-w-0 flex-1 truncate">{name}</span>
      {trailing ? <span className="flex shrink-0 items-center gap-1 text-sm text-shell-muted">{trailing}</span> : null}
    </div>
  );
});
