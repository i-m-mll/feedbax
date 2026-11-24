import clsx from 'clsx';
import { Plus } from 'lucide-react';
import {
  createContext, forwardRef, useContext, type CSSProperties, type HTMLAttributes, type Ref, type TableHTMLAttributes,
  type TdHTMLAttributes, type ThHTMLAttributes,
} from 'react';
import { IconButton } from './Button';
import { ScrollFade, type ScrollFadeProps } from './ScrollFade';

/**
 * A column's width, as a CSS length. A `fixed` column is always `least` wide; the others are at least `least` wide and
 * share the table's spare width equally, so the table's least width counts each of them at the largest of their leasts.
 */
export interface TableColumn {
  least: string;
  fixed?: boolean;
}

/** A control that adds a row or a column, with the name it is read by. */
export interface TableAdd {
  label: string;
  onAdd(): void;
  disabled?: boolean;
}

export interface TableProps extends TableHTMLAttributes<HTMLTableElement> {
  columns: readonly TableColumn[];
  /** How many leading columns, all fixed, stay in place while the rest scroll: those up to and including a row's name. */
  held?: number;
  /** A row added at the foot of the last held column (the name column), or of the first column when none is held. */
  addRow?: TableAdd;
  /** A column added at the trailing end of the header, as wide as the last column's least; `TableAddColumn` draws it. */
  addColumn?: TableAdd;
  /** The table's scroll box. It scrolls horizontally once it is narrower than the table's least width. */
  box?: Omit<ScrollFadeProps, 'children'> & { ref?: Ref<HTMLDivElement> };
}

interface TableShape {
  count: number;
  held: number;
  addColumn?: TableAdd;
}

const Shape = createContext<TableShape>({ count: 1, held: 0 });

const sum = (lengths: readonly string[]) => lengths.length ? `calc(${lengths.join(' + ')})` : '0px';

/**
 * A table that keeps its columns' least widths: in a box narrower than their sum it scrolls on its own, inside its box,
 * with its held columns in place.
 */
export function Table({ columns, held = 0, addRow, addColumn, box, className, style, children, ...props }: TableProps) {
  if (columns.slice(0, held).some(({ fixed }) => !fixed)) throw new Error('A held column has a fixed width.');
  const fixed = columns.filter(({ fixed }) => fixed).map(({ least }) => least);
  const growing = columns.filter(({ fixed }) => !fixed).map(({ least }) => least);
  const least = sum([...fixed, ...growing.length ? [`${growing.length} * max(${growing.join(', ')})`] : []]);
  const all = [...columns, ...addColumn ? [{ least: columns.at(-1)?.least ?? 'var(--row-height)', fixed: true }] : []];
  const offsets: Record<string, string> = {};
  for (let index = 0; index < held; index += 1) offsets[`--held-left-${index + 1}`] = sum(columns.slice(0, index).map(({ least }) => least));
  const { ref, axis = 'x', className: boxClassName, style: boxStyle, ...boxProps } = box ?? {};
  const shape = { count: columns.length, held, addColumn };
  return (
    <ScrollFade ref={ref} axis={axis} className={clsx('studio-table-box', boxClassName)} {...boxProps}
      style={{ '--fade-inset-left': sum(columns.slice(0, held).map(({ least }) => least)), ...boxStyle } as CSSProperties}>
      <Shape.Provider value={shape}>
        <table data-held={held || undefined}
          className={clsx('studio-table w-full table-fixed border-separate border-spacing-0 text-shell-foreground', className)}
          style={{ minWidth: addColumn ? `calc(${least} + ${all.at(-1)!.least})` : least, ...offsets, ...style } as CSSProperties}
          {...props}>
          <colgroup>{all.map((column, index) => <col key={index} style={column.fixed ? { width: column.least } : undefined} />)}</colgroup>
          {children}
          {addRow && <AddRow add={addRow} />}
        </table>
      </Shape.Provider>
    </ScrollFade>
  );
}

/**
 * The header cell of the potential column the table's `addColumn` declares, last in the header row. Rows below it
 * have no cell there, so the rest of the column draws no lines.
 */
export function TableAddColumn() {
  const { addColumn } = useContext(Shape);
  if (!addColumn) throw new Error('TableAddColumn is in a table without addColumn.');
  return <th className="h-row p-0 text-center align-middle">
    <IconButton variant="ghost" size="mark" label={addColumn.label} icon={<Plus aria-hidden="true" />}
      disabled={addColumn.disabled} onClick={addColumn.onAdd} />
  </th>;
}

/** The potential row: the control in its name cell, the other cells drawing no lines. */
function AddRow({ add }: { add: TableAdd }) {
  const { count, held } = useContext(Shape);
  const at = Math.max(0, held - 1);
  return <tbody>
    <tr className="studio-table-add-row h-row">
      {at > 0 && <td colSpan={at} />}
      <td className="text-center">
        <IconButton variant="ghost" size="mark" label={add.label} icon={<Plus aria-hidden="true" />} disabled={add.disabled}
          onClick={add.onAdd} />
      </td>
      {count - at - 1 > 0 && <td colSpan={count - at - 1} />}
    </tr>
  </tbody>;
}

/** A cell centres its content between the row's lines: inline content sits in no line box, so no strut lowers it. */
const centred = 'align-middle leading-[0] [&>*]:leading-normal';

export function TableHeaderCell({ className, ...props }: ThHTMLAttributes<HTMLTableCellElement>) {
  return (
    <th
      className={clsx(
        'sticky top-0 h-row truncate border-b border-shell-divider bg-shell-header px-row-inline text-left text-sm font-medium text-shell-muted',
        centred,
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
        'h-row bg-[--row-fill] hover:[&:not([data-selected])]:[--row-fill:var(--row-hover-fill)] data-[selected]:[--row-fill:var(--row-selected-fill)]',
        'data-[selected]:[&>:first-child]:font-medium data-[related]:[&>:first-child]:shadow-[inset_2px_0_0_var(--row-related-mark)]',
        'outline-none focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-shell-accent',
        className,
      )}
      {...props}
    />
  );
});

export interface TableCellProps extends TdHTMLAttributes<HTMLTableCellElement> {
  /** The cell is the selection: its fill is the selection's. */
  selected?: boolean;
}

const cell = (selected?: boolean) => clsx('h-row truncate border-b border-shell-divider px-row-inline', centred,
  selected && 'bg-shell-row-selected [--cell-fill:var(--row-selected-fill)]');

export function TableCell({ selected, className, ...props }: TableCellProps) {
  return <td data-selected={selected || undefined} className={clsx(cell(selected), className)} {...props} />;
}

export function TableRowHeader({ selected, className, ...props }: ThHTMLAttributes<HTMLTableCellElement> & { selected?: boolean }) {
  return <th scope="row" data-selected={selected || undefined} className={clsx(cell(selected), 'text-left font-normal', className)}
    {...props} />;
}

/**
 * A cell holding a field or a name's input, which is the cell (P35): the field fills it, and the cell's lines are its
 * border. A `header` cell is its row's name.
 */
export function TableFieldCell({ header, className, ...props }: TableCellProps & { header?: boolean }) {
  const classes = clsx('studio-table-field-cell overflow-visible', className);
  return header ? <TableRowHeader className={classes} {...props} /> : <TableCell className={classes} {...props} />;
}
