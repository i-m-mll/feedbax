/**
 * The DOM side of `Table`: fitting each column to its content, and moving rows or columns while they are dragged.
 * Both act on the rendered table directly, so that a width or a drag's offsets are drawn in the same frame React
 * commits the content they follow.
 */

/** A column's bounds as CSS lengths. */
export interface ColumnBounds {
  least: string;
  greatest: string;
}

/** Each cell of `table` with the first and last column it covers. */
function* cellsOf(table: HTMLTableElement) {
  for (const row of table.rows) {
    let at = 0;
    for (const cell of row.cells) {
      yield { row, cell, first: at, last: at + cell.colSpan - 1 };
      at += cell.colSpan;
    }
  }
}

let context: CanvasRenderingContext2D | null | undefined;

/** The width an input's text takes, with the input's own padding and border. */
function inputWidth(input: HTMLInputElement) {
  context ??= document.createElement('canvas').getContext('2d');
  const style = getComputedStyle(input);
  const sides = ['paddingLeft', 'paddingRight', 'borderLeftWidth', 'borderRightWidth'] as const;
  const inset = sides.reduce((sum, side) => sum + (parseFloat(style[side]) || 0), 0);
  if (!context) return inset;
  context.font = style.font;
  return Math.ceil(context.measureText(input.value || input.placeholder).width) + inset + 1;
}

const textual = new Set(['text', 'search', 'number', 'url', 'email', 'tel', '']);

/**
 * Sets each column of `table` as wide as its widest content plus the cell's padding, between its bounds, and returns
 * the widths. The table is laid out once at its content's width, with each text input as wide as its text, and read;
 * `probe` resolves the bounds' lengths. A cell of a column held at its greatest width is marked `data-overflow`.
 */
export function fitColumns(table: HTMLTableElement, bounds: readonly ColumnBounds[], probe: HTMLElement) {
  const box = table.parentElement;
  const scroll = box && { left: box.scrollLeft, top: box.scrollTop };
  for (const input of table.querySelectorAll('input')) {
    if (!textual.has(input.getAttribute('type') ?? '')) continue;
    input.style.setProperty('--studio-measured', `${inputWidth(input)}px`);
    input.dataset.measured = '';
  }
  table.dataset.measuring = '';
  const natural = bounds.map(() => 0);
  for (const { cell, first, last } of cellsOf(table)) {
    if (first === last && first < natural.length) natural[first] = Math.max(natural[first], cell.getBoundingClientRect().width);
  }
  delete table.dataset.measuring;
  const px = (length: string) => {
    probe.style.width = length;
    return probe.getBoundingClientRect().width;
  };
  const widths = bounds.map(({ least, greatest }, index) => {
    const low = px(least);
    return Math.max(low, Math.min(px(greatest), Math.ceil(natural[index])));
  });
  const cols = table.querySelectorAll(':scope > colgroup > col');
  widths.forEach((width, index) => { (cols[index] as HTMLElement | undefined)?.style.setProperty('width', `${width}px`); });
  table.style.width = `${widths.reduce((sum, width) => sum + width, 0)}px`;
  for (const { cell, first, last } of cellsOf(table)) {
    cell.toggleAttribute('data-overflow', first === last && natural[first] > widths[first] + 0.5);
  }
  if (box && scroll) {
    box.scrollLeft = scroll.left;
    box.scrollTop = scroll.top;
  }
  return widths;
}

export type DragAxis = 'rows' | 'columns';

/** A member of a reorderable axis: where it starts along the axis and its size, as it was when the drag began. */
interface Member {
  id: string;
  start: number;
  size: number;
}

/** A drag in progress: the dragged member's index, the pointer's start, and the index it will land at. */
export interface Drag {
  axis: DragAxis;
  pointer: number;
  members: readonly Member[];
  /** For columns, each member's column index in the table. */
  columns: readonly number[];
  from: number;
  origin: number;
  target: number;
}

/** The members' order, as the table draws them now. */
export function memberOrder(table: HTMLTableElement, axis: DragAxis): string[] {
  const selector = axis === 'rows' ? ':scope > tbody > tr[data-member]' : ':scope > thead > tr > [data-handle-column]';
  return [...table.querySelectorAll<HTMLElement>(selector)].map((element) => element.dataset.member ?? element.dataset.handleColumn!);
}

/** Starts a drag of `id` along `axis`, from the pointer's position `origin`. */
export function startDrag(table: HTMLTableElement, axis: DragAxis, id: string, pointer: number, origin: number): Drag | undefined {
  const selector = axis === 'rows' ? ':scope > tbody > tr[data-member]' : ':scope > thead > tr > [data-handle-column]';
  const elements = [...table.querySelectorAll<HTMLElement>(selector)];
  const members = elements.map((element) => {
    const box = element.getBoundingClientRect();
    return { id: element.dataset.member ?? element.dataset.handleColumn!, start: axis === 'rows' ? box.top : box.left,
      size: axis === 'rows' ? box.height : box.width };
  });
  const from = members.findIndex((member) => member.id === id);
  if (from < 0) return undefined;
  const columns = axis === 'columns' ? elements.map((element) => Number(element.dataset.columnIndex)) : [];
  return { axis, pointer, members, columns, from, origin, target: from };
}

/** Where the dragged member lands with the pointer at `at`: past every other member whose middle it has passed. */
export function dragTarget(drag: Drag, at: number) {
  const active = drag.members[drag.from];
  const middle = active.start + active.size / 2 + at - drag.origin;
  return drag.members.filter((member, index) => index !== drag.from && member.start + member.size / 2 < middle).length;
}

/** Each member's offset along the axis: the dragged one's `own`, and those it passes moved by its size. */
function offsets(drag: Drag, own: number) {
  const { from, target, members } = drag;
  const size = members[from].size;
  return members.map((_, index) => {
    if (index === from) return own;
    if (from < target && index > from && index <= target) return -size;
    if (target < from && index >= target && index < from) return size;
    return 0;
  });
}

/** The dragged member's offset at its landing place. */
function landing(drag: Drag) {
  const { from, target, members } = drag;
  const passed = from < target ? members.slice(from + 1, target + 1) : members.slice(target, from);
  return Math.sign(target - from) * passed.reduce((sum, member) => sum + member.size, 0);
}

/** The dragged member's offset following the pointer at `at`, kept within the members' extent. */
function following(drag: Drag, at: number) {
  const { members, from } = drag;
  const active = members[from];
  const last = members.at(-1)!;
  return Math.max(members[0].start - active.start, Math.min(last.start + last.size - active.start - active.size, at - drag.origin));
}

const moving = 'transform 150ms ease';

/** Draws every member where it will be: the dragged one at the pointer (or, `settled`, at its landing place). */
export function drawDrag(table: HTMLTableElement, drag: Drag, at: number, settled = false) {
  const shifts = offsets(drag, settled ? landing(drag) : following(drag, at));
  if (drag.axis === 'rows') {
    const rows = table.querySelectorAll<HTMLElement>(':scope > tbody > tr[data-member]');
    rows.forEach((row, index) => {
      row.style.transition = index === drag.from && !settled ? 'none' : moving;
      row.style.transform = shifts[index] ? `translateY(${shifts[index]}px)` : '';
      row.toggleAttribute('data-dragging', index === drag.from);
    });
    return;
  }
  const byColumn = new Map(drag.columns.map((column, index) => [column, index]));
  for (const { cell, first, last } of cellsOf(table)) {
    const members = Array.from({ length: last - first + 1 }, (_, offset) => byColumn.get(first + offset));
    const shift = members.every((member) => member !== undefined && shifts[member] === shifts[members[0]!])
      ? shifts[members[0]!] : 0;
    const dragged = first === last && members[0] === drag.from;
    cell.style.transition = dragged && !settled ? 'none' : moving;
    cell.style.transform = shift ? `translateX(${shift}px)` : '';
    cell.toggleAttribute('data-dragging', dragged);
  }
}

/** Puts every member back at its own place; `animated` slides them there, else they are there at once. */
export function clearDrag(table: HTMLTableElement, animated: boolean) {
  for (const element of table.querySelectorAll<HTMLElement>('[data-dragging], tr[data-member], [style*="transform"]')) {
    element.style.transition = animated ? moving : '';
    element.style.transform = '';
    element.removeAttribute('data-dragging');
  }
}
