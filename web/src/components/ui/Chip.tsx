import clsx from 'clsx';
import { forwardRef, type HTMLAttributes, type ReactNode } from 'react';
import { focusRing, glyph } from './styles';

export interface ChipProps extends HTMLAttributes<HTMLSpanElement> {
  glyph?: ReactNode;
  trailing?: ReactNode;
  selected?: boolean;
}

export const Chip = forwardRef<HTMLSpanElement, ChipProps>(function Chip(
  { glyph: icon, trailing, selected, children, className, ...props }, ref,
) {
  return (
    <span
      ref={ref}
      data-selected={selected || undefined}
      className={clsx(
        'inline-flex h-[calc(var(--row-height)-6px)] min-w-0 max-w-full cursor-default select-none items-center gap-1',
        'rounded-row bg-shell-subtle px-1.5 text-sm text-shell-foreground',
        'data-[selected]:bg-shell-selected data-[selected]:font-medium',
        focusRing, glyph, className,
      )}
      {...props}
    >
      {icon ? <span className="flex text-shell-muted">{icon}</span> : null}
      <span className="min-w-0 truncate">{children}</span>
      {trailing ? <span className="flex shrink-0 items-center text-shell-muted">{trailing}</span> : null}
    </span>
  );
});
