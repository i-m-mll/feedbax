import clsx from 'clsx';
import type { HTMLAttributes, TdHTMLAttributes, ThHTMLAttributes } from 'react';

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

export function TableRow({ selected, related, className, ...props }: TableRowProps) {
  return (
    <tr
      aria-selected={selected}
      data-selected={selected || undefined}
      data-related={related || undefined}
      className={clsx(
        'h-row hover:bg-shell-hover data-[related]:bg-shell-hover data-[selected]:bg-shell-selected data-[selected]:font-medium',
        'outline-none focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-shell-accent',
        className,
      )}
      {...props}
    />
  );
}

export function TableCell({ className, ...props }: TdHTMLAttributes<HTMLTableCellElement>) {
  return <td className={clsx('h-row truncate border-b border-shell-divider px-row-inline', className)} {...props} />;
}
