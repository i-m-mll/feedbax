import clsx from 'clsx';
import { GripHorizontal, GripVertical, MoreHorizontal, Plus } from 'lucide-react';
import {
  createContext, forwardRef, useCallback, useContext, useEffect, useLayoutEffect, useRef, type CSSProperties, type HTMLAttributes,
  type PointerEvent as ReactPointerEvent, type ReactNode, type Ref, type TableHTMLAttributes, type TdHTMLAttributes,
  type ThHTMLAttributes,
} from 'react';
import { IconButton } from './Button';
import { Menu, MenuContent, MenuTrigger } from './Menu';
import { ScrollFade, type ScrollFadeProps } from './ScrollFade';
import {
  clearDrag, dragTarget, drawDrag, fitColumns, memberOrder, startDrag, type Drag, type DragAxis,
} from './tableLayout';

/** A row or column that can be dragged to another place: its id, and its handle's accessible name. */
export interface TableMember {
  id: string;
  label: string;
}

/**
 * A column. It is as wide as its widest content plus the cell's padding, between `least` and `greatest` (CSS
 * lengths). A `sortable` column is dragged by a handle above its header and moved by the table's `moveColumn`.
 */
export interface TableColumn {
  least: string;
  greatest?: string;
  sortable?: TableMember;
}

/** A control that adds a row or a column, with the name it is read by. */
export interface TableAdd {
  label: string;
  onAdd(): void;
  disabled?: boolean;
}

/** A row's or a column's menu: the button's name and the menu's items. `shown` draws the button without attention. */
export interface TableMenu {
  label: string;
  items: ReactNode;
  shown?: boolean;
}

/** Writes a new order: the member `id` moved to `index` among its axis's sortable members. */
export type TableMove = (id: string, index: number) => unknown;

export interface TableProps extends TableHTMLAttributes<HTMLTableElement> {
  columns: readonly TableColumn[];
  /** How many leading columns stay in place while the rest scroll: those up to and including a row's name. */
  held?: number;
  /** A row added at the foot of the last held column (the name column), or of the first column when none is held. */
  addRow?: TableAdd;
  /** A column added at the trailing end of the header, as wide as its control; `TableAddColumn` draws it. */
  addColumn?: TableAdd;
  /** Moves a `sortable` row. Given, the table has a leading margin that holds each row's handle. */
  moveRow?: TableMove;
  /** Moves a `sortable` column. */
  moveColumn?: TableMove;
  /** The table's scroll box. It scrolls horizontally once it is narrower than the table. */
  box?: Omit<ScrollFadeProps, 'children'> & { ref?: Ref<HTMLDivElement> };
}

interface TableShape {
  /** The caller's columns, after the margin when there is one. */
  columns: readonly TableColumn[];
  margin: boolean;
  held: number;
  addColumn?: TableAdd;
  grab(event: ReactPointerEvent, axis: DragAxis, id: string): void;
}

const Shape = createContext<TableShape>({ columns: [], margin: false, held: 0, grab: () => undefined });
const RowMenu = createContext<TableMenu | undefined>(undefined);

const defaultGreatest = '24rem';
const marginWidth = '1.25rem';
/** The add column's bounds: its control and the cell's padding. */
const addBounds = { least: '0px', greatest: '4rem' };
/** How far the pointer moves before a press on a handle is a drag. */
const slop = 3;

/**
 * A table whose columns fit their content between their bounds: in a box narrower than their sum it scrolls on its
 * own, inside its box, with its held columns in place. Rows and columns given `sortable` are dragged by handles drawn
 * outside their cells; while one is dragged the others are drawn where they will be, and the order is written once,
 * on release. Each row's menu is drawn over the trailing end of its name cell.
 */
export function Table({ columns, held = 0, addRow, addColumn, moveRow, moveColumn, box, className, style, children, ...props }: TableProps) {
  const table = useRef<HTMLTableElement>(null);
  const probe = useRef<HTMLSpanElement>(null);
  const widths = useRef<number[]>([]);
  const margin = !!moveRow;
  const bounds = [
    ...margin ? [{ least: marginWidth, greatest: marginWidth }] : [],
    ...columns.map(({ least, greatest = defaultGreatest }) => ({ least, greatest })),
    ...addColumn ? [addBounds] : [],
  ];
  const shown = held + (margin ? 1 : 0);
  const moves = useRef({ moveRow, moveColumn });
  moves.current = { moveRow, moveColumn };
  const drag = useRef<Drag>();
  const holding = useRef<{ axis: DragAxis; order: string; token: number }>();

  /** The columns under the pointer, focused or selected, whose handles are drawn. */
  const hovered = useRef<number>();
  const markColumns = useCallback(() => {
    const element = table.current;
    if (!element) return;
    const chosen = new Set<number>();
    if (hovered.current !== undefined) chosen.add(hovered.current);
    const header = element.tHead?.rows[element.tHead.rows.length - 1];
    let at = 0;
    for (const cell of header?.cells ?? []) {
      if (cell.matches('[data-selected], :focus-within')) chosen.add(at);
      at += cell.colSpan;
    }
    for (const cell of element.querySelectorAll<HTMLElement>('[data-handle-column]')) {
      cell.toggleAttribute('data-attention', chosen.has(Number(cell.dataset.columnIndex)));
    }
  }, []);

  const fit = useCallback(() => {
    if (!table.current || !probe.current) return;
    widths.current = fitColumns(table.current, bounds, probe.current);
    const offsets = widths.current.slice(0, shown).map((_, index) =>
      widths.current.slice(0, index).reduce((sum, width) => sum + width, 0));
    offsets.forEach((offset, index) => table.current!.style.setProperty(`--held-left-${index + 1}`, `${offset}px`));
    table.current.parentElement?.style.setProperty('--fade-inset-left',
      `${widths.current.slice(0, shown).reduce((sum, width) => sum + width, 0)}px`);
  }, [JSON.stringify(bounds), shown]);

  useLayoutEffect(() => {
    fit();
    markColumns();
    const element = table.current;
    const hold = holding.current;
    if (element && hold && memberOrder(element, hold.axis).join('\n') !== hold.order) {
      clearDrag(element, false);
      holding.current = undefined;
    }
  });
  useEffect(() => {
    const element = table.current;
    if (!element) return;
    const observer = new MutationObserver(fit);
    observer.observe(element, { childList: true, characterData: true, subtree: true });
    const resize = new ResizeObserver(fit);
    resize.observe(element.parentElement!);
    element.addEventListener('input', fit);
    void document.fonts?.ready.then(fit);
    return () => {
      observer.disconnect();
      resize.disconnect();
      element.removeEventListener('input', fit);
    };
  }, [fit]);

  const grab = useCallback((event: ReactPointerEvent, axis: DragAxis, id: string) => {
    const element = table.current;
    const move = axis === 'rows' ? moves.current.moveRow : moves.current.moveColumn;
    if (event.button !== 0 || !element || !move || drag.current) return;
    event.preventDefault();
    const coordinate = (pointer: { clientX: number; clientY: number }) => axis === 'rows' ? pointer.clientY : pointer.clientX;
    const origin = coordinate(event);
    const pointer = event.pointerId;
    let started = false;
    const onMove = (moved: PointerEvent) => {
      if (moved.pointerId !== pointer) return;
      const at = coordinate(moved);
      if (!started) {
        if (Math.abs(at - origin) < slop) return;
        if (holding.current) {
          clearDrag(element, false);
          holding.current = undefined;
        }
        drag.current = startDrag(element, axis, id, pointer, origin);
        started = !!drag.current;
        if (!started) return finish();
      }
      const current = drag.current!;
      current.target = dragTarget(current, at);
      drawDrag(element, current, at);
    };
    const onUp = (released: PointerEvent) => {
      if (released.pointerId !== pointer) return;
      const current = drag.current;
      finish();
      if (!current) return;
      const at = coordinate(released);
      current.target = dragTarget(current, at);
      if (current.target === current.from) {
        clearDrag(element, true);
        return;
      }
      drawDrag(element, current, at, true);
      const token = Date.now();
      holding.current = { axis, order: memberOrder(element, axis).join('\n'), token };
      void Promise.resolve(move(current.members[current.from].id, current.target)).finally(() => {
        if (holding.current?.token !== token) return;
        holding.current = undefined;
        clearDrag(element, true);
      });
    };
    const onCancel = (cancelled: PointerEvent) => {
      if (cancelled.pointerId !== pointer) return;
      finish();
      clearDrag(element, true);
    };
    function finish() {
      drag.current = undefined;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onCancel);
    }
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onCancel);
  }, []);

  const sortableColumns = columns.some(({ sortable }) => sortable);
  const columnAt = (clientX: number) => {
    const left = table.current?.getBoundingClientRect().left ?? 0;
    let edge = left;
    return widths.current.findIndex((width) => (edge += width) > clientX);
  };
  const { ref, axis = 'x', className: boxClassName, style: boxStyle, ...boxProps } = box ?? {};
  const shape = { columns, margin, held: shown, addColumn, grab };
  return (
    <ScrollFade ref={ref} axis={axis} className={clsx('studio-table-box relative', boxClassName)} {...boxProps} style={boxStyle}>
      <span ref={probe} aria-hidden="true" className="pointer-events-none invisible absolute left-0 top-0 h-0" />
      <Shape.Provider value={shape}>
        <table ref={table} data-held={shown || undefined}
          className={clsx('studio-table table-fixed border-separate border-spacing-0 text-shell-foreground', className)}
          style={style as CSSProperties}
          onPointerMove={sortableColumns ? (event) => {
            const at = columnAt(event.clientX);
            if (at !== hovered.current) {
              hovered.current = at < 0 ? undefined : at;
              markColumns();
            }
          } : undefined}
          onPointerLeave={sortableColumns ? () => { hovered.current = undefined; markColumns(); } : undefined}
          onFocus={sortableColumns ? markColumns : undefined} onBlur={sortableColumns ? () => queueMicrotask(markColumns) : undefined}
          {...props}>
          <colgroup>{bounds.map((_, index) => <col key={index} />)}</colgroup>
          {children}
          {addRow && <AddRow add={addRow} />}
        </table>
      </Shape.Provider>
    </ScrollFade>
  );
}

/** A drag handle: pressing it and moving drags its row or column. */
function Handle({ axis, member }: { axis: DragAxis; member: TableMember }) {
  const { grab } = useContext(Shape);
  const Glyph = axis === 'rows' ? GripVertical : GripHorizontal;
  return <span role="img" aria-label={member.label} className="studio-table-handle"
    onPointerDown={(event) => grab(event, axis, member.id)}>
    <Glyph aria-hidden="true" />
  </span>;
}

/** The cell of the table's leading margin in a row: empty, or holding the row's handle. */
export function TableMarginCell({ member, header }: { member?: TableMember; header?: boolean }) {
  const { margin } = useContext(Shape);
  if (!margin) return null;
  const Cell = header ? 'th' : 'td';
  return <Cell className="studio-table-margin" aria-hidden={member ? undefined : true}>
    {member && <Handle axis="rows" member={member} />}
  </Cell>;
}

/**
 * The header row, after the row of column handles when a column is sortable. Its cells are the caller's; the
 * margin's cell is drawn before them.
 */
export function TableHeaderRow({ children, ...props }: HTMLAttributes<HTMLTableRowElement>) {
  const { columns, margin, addColumn } = useContext(Shape);
  const offset = margin ? 1 : 0;
  return <>
    {columns.some(({ sortable }) => sortable) && <tr className="studio-table-handles" aria-hidden="true">
      {margin && <td className="studio-table-margin" />}
      {columns.map(({ sortable }, index) => sortable
        ? <td key={sortable.id} data-handle-column={sortable.id} data-column-index={index + offset}>
          <Handle axis="columns" member={sortable} />
        </td>
        : <td key={index} />)}
      {addColumn && <td />}
    </tr>}
    <tr {...props}>
      <TableMarginCell header />
      {children}
    </tr>
  </>;
}

/**
 * The header cell of the potential column the table's `addColumn` declares, last in the header row. Rows below it
 * have no cell there, so the rest of the column draws no lines.
 */
export function TableAddColumn() {
  const { addColumn } = useContext(Shape);
  if (!addColumn) throw new Error('TableAddColumn is in a table without addColumn.');
  return <th className="studio-table-add-column h-row px-0.5 text-center align-middle">
    <IconButton variant="ghost" size="mark" label={addColumn.label} icon={<Plus aria-hidden="true" strokeWidth={3} />}
      disabled={addColumn.disabled} onClick={addColumn.onAdd} />
  </th>;
}

/** The potential row: the control in its name cell, the other cells drawing no lines. */
function AddRow({ add }: { add: TableAdd }) {
  const { columns, held, margin } = useContext(Shape);
  const offset = margin ? 1 : 0;
  const at = Math.max(0, held - offset - 1);
  return <tbody>
    <tr className="studio-table-add-row h-row">
      {margin && <td />}
      {at > 0 && <td colSpan={at} />}
      <td className="text-center">
        <IconButton variant="ghost" size="mark" label={add.label} icon={<Plus aria-hidden="true" strokeWidth={3} />}
          disabled={add.disabled} onClick={add.onAdd} />
      </td>
      {columns.length - at - 1 > 0 && <td colSpan={columns.length - at - 1} />}
    </tr>
  </tbody>;
}

/** A cell centres its content between the row's lines: inline content sits in no line box, so no strut lowers it. */
const centred = 'align-middle leading-[0] [&>*]:leading-normal';

/**
 * The trailing end of a name cell: the row's or column's menu, drawn there while its owner has attention, over a
 * fade of the name; without attention, the fade alone where the name is longer than its column.
 */
function NameEnd({ menu }: { menu?: TableMenu }) {
  return <span className="studio-table-name-end" data-menu={menu ? '' : undefined} data-shown={menu?.shown || undefined}>
    {menu && <Menu>
      <MenuTrigger render={<IconButton variant="ghost" size="mark" label={menu.label}
        icon={<MoreHorizontal aria-hidden="true" />} />} />
      <MenuContent aria-label={menu.label}>{menu.items}</MenuContent>
    </Menu>}
  </span>;
}

export interface TableHeaderCellProps extends ThHTMLAttributes<HTMLTableCellElement> {
  /** The column's menu, drawn over the cell's trailing end; the cell is then its column's name. */
  menu?: TableMenu;
  selected?: boolean;
}

export const TableHeaderCell = forwardRef<HTMLTableCellElement, TableHeaderCellProps>(function TableHeaderCell(
  { menu, selected, className, children, ...props }, ref,
) {
  return (
    <th
      ref={ref}
      data-selected={selected || undefined}
      className={clsx(
        'sticky top-0 h-row truncate border-b border-shell-divider bg-shell-header px-row-inline text-left text-sm font-medium text-shell-muted',
        centred,
        menu && 'studio-table-name',
        selected && 'bg-shell-row-selected text-shell-foreground [--cell-fill:var(--row-selected-fill)]',
        className,
      )}
      {...props}
    >
      {children}
      {menu && <NameEnd menu={menu} />}
    </th>
  );
});

export interface TableRowProps extends HTMLAttributes<HTMLTableRowElement> {
  selected?: boolean;
  related?: boolean;
  /** The row is dragged by a handle in the table's leading margin. */
  sortable?: TableMember;
  /** The row's menu, drawn over the trailing end of its name cell. */
  menu?: TableMenu;
}

export const TableRow = forwardRef<HTMLTableRowElement, TableRowProps>(function TableRow(
  { selected, related, sortable, menu, className, children, ...props }, ref,
) {
  return (
    <tr
      ref={ref}
      aria-selected={selected}
      data-selected={selected || undefined}
      data-related={related || undefined}
      data-member={sortable?.id}
      className={clsx(
        'studio-table-row h-row bg-[--row-fill] hover:[&:not([data-selected])]:[--row-fill:var(--row-hover-fill)] data-[selected]:[--row-fill:var(--row-selected-fill)]',
        'outline-none focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-shell-accent',
        className,
      )}
      {...props}
    >
      <TableMarginCell member={sortable} />
      <RowMenu.Provider value={menu}>{children}</RowMenu.Provider>
    </tr>
  );
});

export interface TableCellProps extends TdHTMLAttributes<HTMLTableCellElement> {
  /** The cell is the selection: its fill is the selection's. */
  selected?: boolean;
}

const cell = (selected?: boolean) => clsx('h-row truncate border-b border-shell-divider px-row-inline', centred,
  selected && 'bg-shell-row-selected [--cell-fill:var(--row-selected-fill)]');

export const TableCell = forwardRef<HTMLTableCellElement, TableCellProps>(function TableCell({ selected, className, ...props }, ref) {
  return <td ref={ref} data-selected={selected || undefined} className={clsx(cell(selected), className)} {...props} />;
});

/** A row's name. Its row's menu is drawn over its trailing end, and a name longer than its column fades there. */
export const TableRowHeader = forwardRef<HTMLTableCellElement, ThHTMLAttributes<HTMLTableCellElement> & { selected?: boolean }>(
  function TableRowHeader({ selected, className, children, ...props }, ref) {
    const menu = useContext(RowMenu);
    return <th ref={ref} scope="row" data-selected={selected || undefined}
      className={clsx(cell(selected), 'studio-table-name relative text-left font-normal', className)} {...props}>
      {children}
      <NameEnd menu={menu} />
    </th>;
  },
);

/**
 * A cell holding a field or a name's input, which is the cell (P35): the field fills it, and the cell's lines are its
 * border. A `header` cell is its row's name.
 */
export const TableFieldCell = forwardRef<HTMLTableCellElement, TableCellProps & { header?: boolean }>(function TableFieldCell(
  { header, className, ...props }, ref,
) {
  const classes = clsx('studio-table-field-cell', !header && 'overflow-visible', className);
  return header ? <TableRowHeader ref={ref} className={classes} {...props} /> : <TableCell ref={ref} className={classes} {...props} />;
});
