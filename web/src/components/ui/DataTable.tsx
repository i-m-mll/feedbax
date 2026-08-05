import clsx from 'clsx';
import type {
  ComponentPropsWithoutRef,
  CSSProperties,
  ReactNode,
} from 'react';

export interface DataTableColumn {
  id: string;
  label?: ReactNode;
  ariaLabel?: string;
  width?: CSSProperties['width'];
  headerClassName?: string;
}

interface DataTableProps extends Omit<ComponentPropsWithoutRef<'table'>, 'children'> {
  columns: DataTableColumn[];
  children: ReactNode;
  containerClassName?: string;
  minWidth?: CSSProperties['minWidth'];
}

/**
 * Shared Studio table shell for aligned headers, rows, and narrow-view overflow.
 */
export function DataTable({
  columns,
  children,
  className,
  containerClassName,
  minWidth = '100%',
  style,
  ...props
}: DataTableProps) {
  return (
    <div
      className={clsx(
        'w-full overflow-x-auto rounded-md border border-slate-200 bg-white',
        containerClassName
      )}
      data-testid="data-table-scroll-container"
    >
      <table
        className={clsx('w-full table-fixed border-collapse text-left', className)}
        style={{ ...style, minWidth }}
        {...props}
      >
        <colgroup>
          {columns.map((column) => (
            <col key={column.id} style={{ width: column.width }} />
          ))}
        </colgroup>
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50">
            {columns.map((column) => (
              <th
                key={column.id}
                scope="col"
                className={clsx(
                  'px-1 py-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400 first:pl-4 last:pr-4',
                  column.headerClassName
                )}
              >
                {column.label ?? (
                  <span className="sr-only">{column.ariaLabel ?? column.id}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function DataTableRow({ className, ...props }: ComponentPropsWithoutRef<'tr'>) {
  return (
    <tr
      className={clsx(
        'border-b border-slate-100 text-xs last:border-b-0',
        className
      )}
      {...props}
    />
  );
}

export function DataTableCell({ className, ...props }: ComponentPropsWithoutRef<'td'>) {
  return (
    <td
      className={clsx('px-1 py-3 align-middle first:pl-4 last:pr-4', className)}
      {...props}
    />
  );
}

export function DataTableEmpty({
  children,
  className,
  ...props
}: ComponentPropsWithoutRef<'td'>) {
  return (
    <td
      className={clsx('px-4 py-8 text-center text-sm text-slate-400', className)}
      {...props}
    >
      {children}
    </td>
  );
}
