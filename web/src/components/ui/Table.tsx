import clsx from 'clsx';
import { forwardRef, type HTMLAttributes, type TdHTMLAttributes, type ThHTMLAttributes } from 'react';

export function Table({ className, ...props }: HTMLAttributes<HTMLTableElement>) {
  return <table className={clsx('w-full border-separate border-spacing-0 text-shell-foreground', className)} {...props} />;
}

export function TableHeaderCell({ className, ...props }: ThHTMLAttributes<HTMLTableCellElement>) {
  return (
    <th
      className={clsx(
        'sticky top-0 h-row whitespace-nowrap border-b border-shell-divider bg-shell-chrome px-row-inline text-left text-sm font-medium text-shell-muted',
        className,
      )}
      {...props}
    />
  );
}

export interface TableRowProps extends HTMLAttributes<HTMLTableRowElement> {
  selected?: boolean;
  related?: boolean;
}

export const TableRow = forwardRef<HTMLTableRowElement, TableRowProps>(function TableRow(
  { selected, related, className, ...props }, ref,
) {
  return (
    <tr
      ref={ref}
      aria-selected={selected}
      data-selected={selected || undefined}
      data-related={related || undefined}
      className={clsx(
        'h-row hover:[&:not([data-selected])]:bg-shell-row-hover data-[selected]:bg-shell-row-selected',
        'data-[selected]:[&>:first-child]:font-medium data-[related]:[&>:first-child]:shadow-[inset_2px_0_0_var(--row-related-mark)]',
        'outline-none focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-shell-accent',
        className,
      )}
      {...props}
    />
  );
});

const cell = 'h-row truncate border-b border-shell-divider px-row-inline';

export function TableCell({ className, ...props }: TdHTMLAttributes<HTMLTableCellElement>) {
  return <td className={clsx(cell, className)} {...props} />;
}

export function TableRowHeader({ className, ...props }: ThHTMLAttributes<HTMLTableCellElement>) {
  return <th scope="row" className={clsx(cell, 'text-left font-normal', className)} {...props} />;
}
